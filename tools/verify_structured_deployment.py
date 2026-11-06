"""Isolated native-checkpoint versus local/remote AutoModel verification.

Run with python -I and pass an exact v0.3.0 package source directory. The
serving model loads its own exported custom source, independently of that
training package. No optimizer or accuracy training is performed.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time


TRAIN_SOURCE_SHA256 = "094bf61ef954c61a2b200e5e8760348160b76a5fbfd186602dd992e4b62ccfa6"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Independent completed training run")
    p.add_argument("--training-source", required=True, help="Exact v0.3.0 package root")
    p.add_argument("--target", required=True, help="Serving directory or Hugging Face model ID")
    p.add_argument("--revision", help="Required immutable commit for a remote target")
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--memory-limit-gb", type=float, default=20.)
    p.add_argument("--long-shapes", action="store_true")
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()
    if Path(args.cache).drive.upper() != "E:" and sys.platform == "win32":
        p.error("Keep model caches on E:")
    package = Path(args.training_source) / "captionbert_rotary"
    if hashlib.sha256((package / "modeling_captionbert.py").read_bytes()).hexdigest() != TRAIN_SOURCE_SHA256:
        raise ValueError("The independent training reference is not the exact frozen source")
    sys.path.insert(0, str(Path(args.training_source)))
    import torch
    from safetensors import safe_open
    from transformers import AutoModel, AutoTokenizer
    from captionbert_rotary.configuration_captionbert import CaptionBertConfig
    from captionbert_rotary.modeling_captionbert import CaptionBertModel
    from captionbert_rotary.memory import MemoryGuard
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    guard = MemoryGuard(args.memory_limit_gb, args.device)
    source = Path(args.source)
    result = json.loads((source / "result.json").read_text(encoding="utf8"))
    state = torch.load(source / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    if state["step"] != result["steps"] or any(state["identity"].get(key) != result.get(key)
                 for key in ("arm", "seed", "config_id", "dataset_id")):
        raise ValueError("Independent training state identity differs")
    native = CaptionBertModel(CaptionBertConfig.from_pretrained(source))
    native.load_state_dict(state["model"], strict=True)
    del state
    native = native.to(args.device).eval()
    target = Path(args.target)
    remote = not target.is_dir()
    if remote and (not args.revision or len(args.revision) != 40):
        p.error("Remote parity requires a full immutable Hub commit")
    kwargs = dict(trust_remote_code=True, cache_dir=args.cache)
    if remote:
        kwargs["revision"] = args.revision
    auto, info = AutoModel.from_pretrained(args.target, use_safetensors=True,
                                          output_loading_info=True, **kwargs)
    if not type(auto).__module__.startswith("transformers_modules."):
        raise ValueError("AutoModel did not independently load the exported custom source")
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"AutoModel loading mismatch: {info}")
    if remote:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(args.target, "model.safetensors", revision=args.revision, cache_dir=args.cache)
    else:
        weights = target / "model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as header:
        metadata = header.metadata()
        if metadata.get("format") != "pt":
            raise ValueError("Serving file has incorrect safetensors format metadata")
    auto = auto.to(args.device).eval()
    if set(native.state_dict()) != set(auto.state_dict()):
        raise ValueError("Native/AutoModel tensor census differs")
    for key, value in native.state_dict().items():
        if not torch.equal(value, auto.state_dict()[key]):
            raise ValueError(f"Native/AutoModel tensor differs: {key}")
    tokenizer = AutoTokenizer.from_pretrained(args.target, **kwargs)
    natural = tokenizer(["A red dog runs past another red dog.",
                         "Several people stand beside a blue building."],
                        padding=True, return_tensors="pt")
    examples = [("natural", natural["input_ids"], natural["attention_mask"]),
                ("all_padding", torch.zeros(2, 17, dtype=torch.long), torch.zeros(2, 17, dtype=torch.long))]
    for batch, length in ((1, 1), (2, 17), (3, 128), (1, 512)):
        ids = torch.full((batch, length), 2100, dtype=torch.long)
        examples.append((f"synthetic_{batch}x{length}", ids, torch.ones_like(ids)))
    if args.long_shapes:
        for length in (1025, 8192):
            ids = torch.full((1, length), 2100, dtype=torch.long)
            ids[:, 1::2] = 1012
            examples.append((f"boundary_shape_{length}", ids, torch.ones_like(ids)))
    cases = []
    dtypes = [("fp32", None)]
    if args.amp:
        if not str(args.device).startswith("cuda"):
            p.error("The AMP interface check requires CUDA")
        dtypes += [("fp16_autocast", torch.float16), ("bf16_autocast", torch.bfloat16)]
    with torch.inference_mode():
        for precision, dtype in dtypes:
            selected = examples if dtype is None else examples[:2]
            for name, ids, mask in selected:
                ids, mask = ids.to(args.device), mask.to(args.device)
                if guard.enabled:
                    torch.cuda.synchronize()
                before = time.monotonic()
                with torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext():
                    expected = native(ids, mask)
                    actual = auto(ids, mask)
                if guard.enabled:
                    torch.cuda.synchronize()
                torch.testing.assert_close(actual.pooler_output, expected.pooler_output, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=2e-6, rtol=1e-5)
                assert actual.pooler_output.shape == (len(ids), 768)
                assert actual.last_hidden_state.shape == (*ids.shape, 512)
                assert torch.isfinite(actual.pooler_output).all() and torch.isfinite(actual.last_hidden_state).all()
                if not mask.any():
                    assert torch.count_nonzero(actual.pooler_output) == 0
                else:
                    torch.testing.assert_close(actual.pooler_output.norm(dim=-1),
                                               torch.ones(len(ids), device=ids.device), atol=2e-6, rtol=1e-5)
                guard.check()
                cases.append(dict(case=name, precision=precision, input_shape=list(ids.shape),
                    embedding_shape=list(actual.pooler_output.shape), hidden_shape=list(actual.last_hidden_state.shape),
                    max_embedding_delta=float((actual.pooler_output-expected.pooler_output).abs().max()),
                    max_hidden_delta=float((actual.last_hidden_state-expected.last_hidden_state).abs().max()),
                    seconds=time.monotonic()-before))
                print(f"Parity {name}/{precision}: {cases[-1]}", flush=True)
        ids = natural["input_ids"][:1].to(args.device)
        mask = natural["attention_mask"][:1].to(args.device)
        base = auto(ids, mask).pooler_output
        for side in ("left", "right"):
            padding = (7, 0) if side == "left" else (0, 7)
            padded_ids = torch.nn.functional.pad(ids, padding)
            padded_mask = torch.nn.functional.pad(mask, padding)
            got = auto(padded_ids, padded_mask).pooler_output
            torch.testing.assert_close(base, got, atol=2e-6, rtol=1e-5)
            cases.append(dict(case=side+"_padding_invariance", max_embedding_delta=float((base-got).abs().max())))
        interface = auto(ids, mask, output_hidden_states=True, output_attentions=True,
                         pointer_targets=torch.full_like(ids, -1))
        assert len(interface.hidden_states) == 13 and len(interface.attentions) == 12
        assert all(x.shape == (len(ids), native.config.num_attention_heads, ids.shape[1], ids.shape[1])
                   for x in interface.attentions)
        assert all(torch.isfinite(x).all() for x in interface.attentions)
        assert float(interface.pointer_loss) == 0.0
        if args.amp:
            for dtype in (torch.float16, torch.bfloat16):
                with torch.autocast("cuda", dtype=dtype):
                    diagnostic = auto(ids, mask, output_attentions=True, pointer_targets=torch.full_like(ids, -1))
                assert all(torch.isfinite(x).all() for x in diagnostic.attentions)
                assert torch.isfinite(diagnostic.pointer_loss)
        guard.check()
    report = dict(target=args.target, revision=args.revision, remote=remote,
        native_reference="independent training state.pt; exact v0.3.0 source",
        source_arm=result["arm"], source_seed=result["seed"], source_config_id=result["config_id"],
        state_tensors_equal=True, safetensors_metadata=metadata,
        loading_info={key: sorted(value) if isinstance(value, set) else value for key, value in info.items()},
        native_training_source_sha256=TRAIN_SOURCE_SHA256, tf32=False, cases=cases,
        hidden_states_count=13, attention_layers=12, attention_head_count=native.config.num_attention_heads,
        memory=guard.report(), training_performed=False,
        long_shapes_are_engineering_only=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Independent deployment verification passed:", args.output, flush=True)


if __name__ == "__main__":
    main()
