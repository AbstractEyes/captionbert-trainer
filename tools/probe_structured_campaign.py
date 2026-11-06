"""Frozen natural-caption lookup and independent checkpoint/AutoModel checks.

Standalone for a completed Colab lane; imports the installed training release.
Run after that lane's optimizer has finished, never beside its GPU writer.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

LOCAL_ROOT = Path(__file__).resolve().parents[1]
if (LOCAL_ROOT / "captionbert_rotary").is_dir():
    sys.path.insert(0, str(LOCAL_ROOT))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from safetensors import safe_open

from captionbert_rotary.configuration_captionbert import CaptionBertConfig
from captionbert_rotary.modeling_captionbert import CaptionBertModel
from captionbert_rotary.data import Store, sha256_file
from captionbert_rotary.structured_data import exact_word_targets, tokenizer_metadata
from captionbert_rotary.memory import MemoryGuard


@contextmanager
def intervention(model, name):
    cfg = model.config
    layer = model.layers[cfg.lookup_layer]
    original = cfg.rotary_mode, cfg.key_rho, layer.rho, cfg.phase_seed
    frequencies = model.frequencies.detach().clone()
    values = model.value_address.detach().clone() if model.value_address is not None else None
    try:
        if name == "binding_off" and cfg.rotary_mode in ("key_end", "key_all"):
            cfg.rotary_mode = "hierarchical"
        elif name == "norm_off":
            cfg.key_rho = layer.rho = 0.0
        elif name == "phase_redraw":
            cfg.phase_seed = 981243653
        elif name == "anchor_angles_off" and model.structured:
            with torch.no_grad():
                model.frequencies[model.kinds >= 2] = 0
        elif name == "value_addresses_constant" and values is not None:
            # Keep each level's learned digit-zero offset, remove digit variation.
            with torch.no_grad():
                model.value_address.copy_(values[:, :1].expand_as(values))
        elif name != "intact":
            raise ValueError(f"Unknown or inapplicable intervention {name}")
        yield
    finally:
        cfg.rotary_mode, cfg.key_rho, layer.rho, cfg.phase_seed = original
        with torch.no_grad():
            model.frequencies.copy_(frequencies)
            if values is not None:
                model.value_address.copy_(values)


def summarize(arrays):
    if not arrays or not len(arrays["row"]):
        return dict(queries=0, captions=0)
    n = len(arrays["row"])
    result = {}
    groups = dict(all=np.ones(n, dtype=bool), ending=arrays["ending"],
                  non_ending=~arrays["ending"], continuation=arrays["continuation"],
                  above128=arrays["length"] > 128)
    for name, selected in groups.items():
        if not selected.any():
            result[name] = dict(queries=0, captions=0)
            continue
        rows = arrays["row"][selected]
        gold = arrays["gold_mass"][selected]
        row_means = [float(gold[rows == row].mean()) for row in np.unique(rows)]
        result[name] = dict(queries=int(selected.sum()), captions=int(len(np.unique(rows))),
            record_mass=float(gold.mean()), caption_mean_record_mass=float(np.mean(row_means)),
            median_record_mass=float(np.median(gold)), exact_record_top1=float(arrays["hit"][selected].mean()),
            wrong_adjacent_mass=float(arrays["wrong_mass"][selected].mean()),
            uniform_mass=float(arrays["uniform"][selected].mean()))
    return result


@torch.no_grad()
def lookup_probe(model, store, indices, vocabulary, special_ids, batch, device):
    fields = {k: [] for k in ("row", "position", "target", "gold_mass", "wrong_mass", "uniform",
                              "hit", "ending", "continuation", "length")}
    continuation = model.wordpiece_continuation
    for start in range(0, len(indices), batch):
        ix = indices[start:start + batch]
        ids_cpu, mask_cpu, _, _ = store.batch(ix)
        target_cpu = torch.stack([torch.from_numpy(exact_word_targets(row.tolist(), vocabulary, special_ids)) for row in ids_cpu]).long()
        ids, mask, target = (x.to(device) for x in (ids_cpu, mask_cpu, target_cpu))
        selected = target >= 0
        if not selected.any():
            continue
        # FP32 explicitly: measures the trained attention, independently of the
        # optional FP16 diagnostic mask defect in the original training release.
        out = model(ids, mask, output_attentions=True)
        attention = out.attentions[model.config.lookup_layer][:, model.config.lookup_head].float()
        rows, positions = selected.nonzero(as_tuple=True)
        gold = target[rows, positions]
        scores = attention[rows, positions]
        lengths = mask.sum(1)
        wrong = torch.minimum(gold + 1, lengths[rows] - 1)
        next_ids = torch.nn.functional.pad(ids[:, 1:], (0, 1), value=model.config.pad_token_id)
        ending = ~continuation[next_ids[rows, positions]]
        columns = dict(row=torch.as_tensor(ix, device=device)[rows], position=positions, target=gold,
            gold_mass=scores.gather(1, gold[:, None]).squeeze(1),
            wrong_mass=scores.gather(1, wrong[:, None]).squeeze(1), uniform=1.0/lengths[rows],
            hit=scores.argmax(-1) == gold, ending=ending, continuation=continuation[ids[rows, positions]],
            length=lengths[rows])
        for key, value in columns.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite lookup measurement: {key}")
            fields[key].append(value.cpu().numpy())
        del out, attention, scores
        if (start // batch + 1) % 50 == 0:
            print(f"Lookup: {min(start + batch, len(indices)):,}/{len(indices):,} development captions", flush=True)
    arrays = {k: np.concatenate(v) if v else np.array([], dtype=np.int64) for k, v in fields.items()}
    return summarize(arrays), arrays


@torch.no_grad()
def independent_parity(directory, native, store, indices, device, long_shapes=False):
    # Native parameters came from the independent torch.save optimizer/RNG
    # checkpoint, not from the safetensors file being tested.
    with safe_open(directory / "model.safetensors", framework="pt") as handle:
        if handle.metadata().get("format") != "pt":
            raise ValueError("AutoModel requires the PyTorch safetensors metadata")
    auto, info = AutoModel.from_pretrained(directory, trust_remote_code=True, output_loading_info=True,
                                          use_safetensors=True)
    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise ValueError(f"AutoModel loading mismatch: {info}")
    auto = auto.to(device).eval()
    if set(native.state_dict()) != set(auto.state_dict()):
        raise ValueError("Independent checkpoint and safetensors keys differ")
    for key, value in native.state_dict().items():
        if not torch.equal(value, auto.state_dict()[key]):
            raise ValueError(f"Independent checkpoint/safetensors differ: {key}")
    cases = []
    ids, mask, _, _ = store.batch(indices[:2])
    examples = [("real_captions", ids, mask), ("all_padding", torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.long))]
    if long_shapes:
        for length in (1, 17, 128, 512, 8192):
            x = torch.full((1, length), 2100, dtype=torch.long)
            examples.append((f"synthetic_length_{length}", x, torch.ones_like(x)))
    for name, ids, mask in examples:
        ids, mask = ids.to(device), mask.to(device)
        before = time.monotonic()
        ref, got = native(ids, mask), auto(ids, mask)
        torch.testing.assert_close(ref.pooler_output, got.pooler_output, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(ref.last_hidden_state, got.last_hidden_state, atol=2e-6, rtol=1e-5)
        assert got.pooler_output.shape == (len(ids), native.config.output_dim)
        assert got.last_hidden_state.shape == (*ids.shape, native.config.hidden_size)
        assert torch.isfinite(got.pooler_output).all() and torch.isfinite(got.last_hidden_state).all()
        cases.append(dict(case=name, input_shape=list(ids.shape), embedding_shape=list(got.pooler_output.shape),
            hidden_shape=list(got.last_hidden_state.shape), max_embedding_delta=float((ref.pooler_output-got.pooler_output).abs().max()),
            max_hidden_delta=float((ref.last_hidden_state-got.last_hidden_state).abs().max()), seconds=time.monotonic()-before))
    del auto
    return dict(reference="independent final state.pt model tensors", format="model.safetensors", metadata_format="pt", dtype="fp32",
                state_tensors_equal=True, loading_info={k: sorted(v) if isinstance(v, set) else v for k,v in info.items()}, cases=cases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/content/captionbert-structured/screen")
    p.add_argument("--data", default="/content/captionbert-data")
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--arms", nargs="+")
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--sentence-ablations", action="store_true",
                   help="Also score each frozen intervention on development teacher/STSB measurements")
    p.add_argument("--structural-ablations", action="store_true",
                   help="Exploratory anchor-angle and value-address interventions on structured models")
    p.add_argument("--long-shapes", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-limit-gb", type=float, default=92.)
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    guard = MemoryGuard(args.memory_limit_gb, args.device)
    root, data = Path(args.root), Path(args.data)
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    vocabulary, _ = tokenizer_metadata(tokenizer)
    special = set(tokenizer.all_special_ids)
    store = Store([data / "chunk_061_L256"])
    ix = np.random.default_rng(20261106).choice(len(store), min(args.rows, len(store)), replace=False)
    for directory in sorted(root.iterdir()):
        result_path = directory / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        if args.arms and result["arm"] not in args.arms:
            continue
        if result.get("status") != "complete":
            continue
        if args.push and result.get("engineering_only"):
            raise ValueError("Engineering fixtures must never ship as scientific runs")
        if sha256_file(directory / "model.safetensors") != result["checkpoint_sha256"]:
            raise ValueError("Completed safetensors failed its recorded hash")
        output = directory / "frozen-development"
        output.mkdir(exist_ok=True)
        state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True, mmap=True)
        if state["step"] != result["steps"] or any(state["identity"].get(k) != result.get(k) for k in ("arm", "seed", "config_id", "dataset_id")):
            raise ValueError("Independent final checkpoint identity/step mismatch")
        model = CaptionBertModel(CaptionBertConfig.from_pretrained(directory)).to(args.device)
        model.load_state_dict(state["model"], strict=True)
        del state
        model.eval()
        print(f"Frozen development: {directory.name}, {len(ix):,} rows; FP32, no optimizer", flush=True)
        parity = independent_parity(directory, model, store, ix, args.device, args.long_shapes)
        measurements = {}
        sentence_measurements = {}
        names = ["intact"]
        if args.ablations and model.config.rotary_mode in ("key_end", "key_all"):
            names += ["binding_off", "norm_off", "phase_redraw"]
        if args.structural_ablations and model.structured:
            names.append("anchor_angles_off")
            if model.value_address is not None:
                names.append("value_addresses_constant")
        for name in names:
            with intervention(model, name):
                summary, arrays = lookup_probe(model, store, ix, vocabulary, special, args.batch, args.device)
                if args.sentence_ablations:
                    from captionbert_rotary.metrics import evaluate
                    from captionbert_rotary.structured_campaign import dev_sts
                    cfg = SimpleNamespace(device=args.device, eval_batch=64, eval_rows=5000, data_dir=str(data))
                    condition = output / name
                    condition.mkdir(exist_ok=True)
                    sentence_measurements[name] = dict(
                        evaluation=evaluate(model, store, cfg),
                        development=dev_sts(model, tokenizer, cfg, condition),
                        precision="legacy FP16 autocast evaluation; FP32 parameters; TF32 off",
                    )
            measurements[name] = summary
            np.savez_compressed(output / f"{name}-queries.npz", **arrays)
        row_hash = hashlib.sha256(ix.astype("<i8").tobytes()).hexdigest()
        report = dict(arm=result["arm"], seed=result["seed"], config_id=result["config_id"], dataset_id=result["dataset_id"],
            split="chunk061 development", rows=len(ix), row_seed=20261106, row_ids_sha256=row_hash,
            precision="fp32 TF32 off", measurements=measurements,
            sentence_measurements=sentence_measurements, independent_checkpoint_auto_parity=parity,
            probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            memory=guard.report(), selection_use=False, training_performed=False)
        np.save(output / "row-ids.npy", ix)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(arm=result["arm"], seed=result["seed"], measurements=measurements)), flush=True)
        del model
        if guard.enabled:
            torch.cuda.empty_cache()
        if args.push:
            from huggingface_hub import HfApi
            HfApi().upload_folder(repo_id="AbstractPhil/captionbert-8192-rotary", folder_path=output,
                path_in_repo=f"experiments/{result['config_id']}/{directory.name}/frozen-development",
                commit_message=f"Frozen development lookup/parity: {directory.name}")


if __name__ == "__main__":
    main()
