from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import torch
from transformers import BertTokenizerFast
from captionbert_rotary import build_model
from captionbert_rotary.analyse import analyse_available
from captionbert_rotary.data import atomic_json


def test_posthoc_uses_final_weights_and_reuses_identical_published_measurement(tmp_path, tiny_config, monkeypatch):
    import huggingface_hub
    import captionbert_rotary.analyse as analysis
    remote = tmp_path / "hub"
    experiment = "test-experiment"
    prefix = f"experiments/{experiment}/A2-seed1"
    source = remote / prefix
    source.mkdir(parents=True)
    model = build_model("A2", tiny_config)
    torch.save(model.state_dict(), source / "model.pt")
    before = hashlib.sha256((source / "model.pt").read_bytes()).hexdigest()
    atomic_json(source / "result.json", {"arm": "A2", "seed": 1, "steps": 7, "config_id": experiment,
                "dataset_id": "test-data", "model_config": asdict(tiny_config)})
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "!", "?", "the"]) + "\n")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(source / "tokenizer")

    class API:
        def model_info(self, *args, **kwargs):
            return SimpleNamespace(sha="fixed-revision", siblings=[SimpleNamespace(
                rfilename=p.relative_to(remote).as_posix(), size=p.stat().st_size,
                lfs={"sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
                for p in remote.rglob("*") if p.is_file()])
        def create_repo(self, *args, **kwargs):
            pass
        def upload_folder(self, **kwargs):
            shutil.copytree(kwargs["folder_path"], remote / kwargs["path_in_repo"], dirs_exist_ok=True)
    api = API()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    def download(repo, filename, **kwargs):
        assert kwargs["revision"] == "fixed-revision"
        return str(remote / filename)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    original = analysis.run_rotary_diagnostics
    calls = []
    def count(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(analysis, "run_rotary_diagnostics", count)
    first = analyse_available("test/repo", experiment, arms=("A2",), seeds=(1, 2), rows=2,
                              output_dir=str(tmp_path / "analysis"), push=True)
    assert first["training_steps_started"] == 0 and first["awaiting_completed_arms"] == ["A2-seed2"]
    report = json.loads((tmp_path / "analysis/A2-seed1/rotary_diagnostics.json").read_text())
    assert report["identity"]["step"] == 7 and report["identity"]["checkpoint_sha256"] == before
    second = analyse_available("test/repo", experiment, arms=("A2",), seeds=(1,), rows=2,
                               output_dir=str(tmp_path / "analysis-second"))
    assert second["completed"][0]["action"] == "reused published measurement"
    assert len(calls) == 1
    assert hashlib.sha256((source / "model.pt").read_bytes()).hexdigest() == before
