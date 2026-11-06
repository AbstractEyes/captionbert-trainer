from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from captionbert_rotary.config import ScreenConfig
from captionbert_rotary.data import atomic_json
from captionbert_rotary.hub import Uploader
from captionbert_rotary.memory import MemoryGuard
from captionbert_rotary.train import train_arm, vram_probe, capture_rng, seed_all
from captionbert_rotary.model import build_model
from conftest import FakeStore


@pytest.mark.parametrize("arm", ["A0", "A2"])
def test_interrupted_resume_matches_uninterrupted(tmp_path, tiny_config, arm):
    cfg = ScreenConfig(model=tiny_config, device="cpu", batch_size=4, epochs=2,
                       max_steps=5, checkpoint_every=1, warmup_steps=2, eval_every=3,
                       eval_rows=8, benchmark=False, output_dir=str(tmp_path / "full"))
    store = FakeStore()
    guard = MemoryGuard(device="cpu")
    train_arm(cfg, arm, 1, store, store, "same-data", guard, probe=False)
    interrupted = replace(cfg, output_dir=str(tmp_path / "resumed"))

    def stop(step):
        if step == 2:
            raise RuntimeError("simulated interruption")
    with pytest.raises(RuntimeError, match="simulated"):
        train_arm(interrupted, arm, 1, store, store, "same-data", guard, probe=False, on_step=stop)
    train_arm(interrupted, arm, 1, store, store, "same-data", guard, probe=False)
    name = f"{arm}-seed1"
    full = torch.load(Path(cfg.output_dir) / name / "model.pt", weights_only=True)
    resumed = torch.load(Path(interrupted.output_dir) / name / "model.pt", weights_only=True)
    for key in full:
        torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
    def rows(root):
        return [json.loads(line) for line in (Path(root) / name / "train.jsonl").read_text().splitlines()]
    for a, b in zip(rows(cfg.output_dir), rows(interrupted.output_dir)):
        assert a["loss"] == b["loss"] and a["batch_sha256"] == b["batch_sha256"]


def test_probe_preserves_model_and_rng(tiny_config):
    cfg = ScreenConfig(model=tiny_config, device="cpu", batch_size=2, max_tokens=8)
    seed_all(7)
    model = build_model("A2", tiny_config)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = capture_rng()
    vram_probe(model, cfg, MemoryGuard(device="cpu"))
    for key in before:
        torch.testing.assert_close(before[key], model.state_dict()[key], atol=0, rtol=0)
    assert torch.equal(rng["torch"], capture_rng()["torch"])
    assert all(p.grad is None for p in model.parameters())


def test_upload_is_snapshot_and_retries(tmp_path):
    cfg = ScreenConfig(output_dir=str(tmp_path), push=True)
    atomic_json(tmp_path / "experiment.json", {})
    atomic_json(tmp_path / "predictions.json", {})
    run = tmp_path / "A2-seed1"
    run.mkdir()
    (run / "model.txt").write_text("original")

    class API:
        calls = 0
        payload = None
        def create_repo(self, *args, **kwargs):
            pass
        def upload_folder(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("transient")
            self.payload = (Path(kwargs["folder_path"]) / "model.txt").read_text()
    api = API()
    uploader = Uploader(cfg, api=api, sleep=lambda _: None)
    uploader.submit("A2-seed1")
    (run / "model.txt").write_text("changed")
    uploader.close()
    assert api.calls == 3 and api.payload == "original"
    assert not list((tmp_path / ".upload_queue").iterdir())


def test_4090_limit_cannot_be_raised(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (21 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr("captionbert_rotary.memory.device_used_bytes", lambda _: 3_000_000_000)
    fractions = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d: fractions.append(f))
    guard = MemoryGuard(80)
    assert guard.limit == 20_000_000_000
    assert fractions[0] * 24 * 1024**3 + 3_000_000_000 < 20_000_000_000
