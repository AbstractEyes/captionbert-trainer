"""Export the locked trained model as a complete safe AutoModel directory."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Completed selected full4m arm/seed folder")
    p.add_argument("--selection", required=True, help="Locked development-only deployment JSON")
    p.add_argument("--output", required=True, help="New local AutoModel release directory")
    args = p.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        p.error("Preserve existing exports; choose a new output directory")
    chosen = json.loads(Path(args.selection).read_text(encoding="utf8"))
    result = json.loads((source / "result.json").read_text(encoding="utf8"))
    expected = dict(arm=chosen["arm"], seed=chosen["seed"], status="complete", steps=7808,
                    training_rows=4000000, config_id=chosen["source_config_id"],
                    dataset_id=chosen["source_dataset_id"])
    if chosen["stage"] != "full4m" or chosen["test_scores_used"] or result.get("engineering_only"):
        raise ValueError("Only the locked development-selected four-million-row model may be exported")
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Source artifact does not match the locked deployment choice")
    if sha256(source / "model.safetensors") != result["checkpoint_sha256"]:
        raise ValueError("Trained source export failed its recorded hash")
    import torch
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from captionbert_rotary.configuration_captionbert import CaptionBertConfig
    from captionbert_rotary.modeling_captionbert import CaptionBertModel
    torch.set_num_threads(2)
    state = torch.load(source / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    if state["step"] != 7808 or any(state["identity"].get(key) != result[key]
                                  for key in ("arm", "seed", "config_id", "dataset_id")):
        raise ValueError("Independent training state identity/step differs")
    config = CaptionBertConfig.from_pretrained(source)
    config._name_or_path = "AbstractPhil/captionbert-8192-rotary"
    model = CaptionBertModel(config).eval()
    model.load_state_dict(state["model"], strict=True)
    # Compare to the training-time safe file before producing the serving copy.
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as original:
        if original.metadata().get("format") != "pt":
            raise ValueError("Training export has incorrect safetensors metadata")
        if set(original.keys()) != set(model.state_dict()):
            raise ValueError("Training state and source safetensors keys differ")
        for key, value in model.state_dict().items():
            if not torch.equal(value, original.get_tensor(key)):
                raise ValueError(f"Independent training state differs: {key}")
    del state
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.model_max_length = config.max_position_embeddings
    tokenizer.init_kwargs["model_max_length"] = config.max_position_embeddings
    CaptionBertConfig.register_for_auto_class("AutoConfig")
    CaptionBertModel.register_for_auto_class("AutoModel")
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as exported:
        if exported.metadata().get("format") != "pt":
            raise ValueError("Serving export must declare safetensors format pt")
        if set(exported.keys()) != set(model.state_dict()):
            raise ValueError("Serving tensor census differs")
        for key, value in model.state_dict().items():
            if not torch.equal(value, exported.get_tensor(key)):
                raise ValueError(f"Serving tensor differs: {key}")
    exported_config = json.loads((output / "config.json").read_text(encoding="utf8"))
    expected_map = {"AutoConfig": "configuration_captionbert.CaptionBertConfig",
                    "AutoModel": "modeling_captionbert.CaptionBertModel"}
    if exported_config.get("auto_map") != expected_map:
        raise ValueError("Serving AutoClass source mapping is incomplete")
    manifest = dict(selected_arm=chosen["arm"], selected_seed=chosen["seed"],
        source_config_id=chosen["source_config_id"], source_dataset_id=chosen["source_dataset_id"],
        source_hub_revision=chosen["results_source_revision"], training_steps=7808,
        training_rows=4000000, training_release="v0.3.0",
        inference_source_sha256=sha256(inspect.getsourcefile(CaptionBertModel)),
        source_model_safetensors_sha256=result["checkpoint_sha256"],
        independent_state_pt_sha256=sha256(source / "state.pt"),
        model_safetensors_sha256=sha256(output / "model.safetensors"),
        safetensors_metadata={"format": "pt"}, safe_serialization=True,
        every_tensor_equal_to_independent_training_state=True, auto_map=expected_map,
        output_dim=config.output_dim, hidden_size=config.hidden_size,
        training_max_tokens=256, execution_max_tokens=config.max_position_embeddings,
        long_context_semantic_quality_measured=False,
        parameter_count=sum(value.numel() for value in model.parameters()))
    (output / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
