from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from threading import Event
import pytest
import torch
from captionbert_rotary import ScreenConfig
from captionbert_rotary.data import atomic_json
from captionbert_rotary.hub import Uploader, pull_completed
from captionbert_rotary.memory import MemoryGuard
from captionbert_rotary.train import train_arm
from conftest import FakeStore


class FakeHub:
    def __init__(self, root):
        self.root = root
    def create_repo(self, *args, **kwargs):
        pass
    def upload_folder(self, **kwargs):
        shutil.copytree(kwargs["folder_path"], self.root / kwargs["path_in_repo"], dirs_exist_ok=True)
    def model_info(self, *args, **kwargs):
        return SimpleNamespace(sha="pinned-hub-revision")
    def list_repo_files(self, *args, **kwargs):
        assert kwargs["revision"] == "pinned-hub-revision"
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]
    def download(self, repo, filename, **kwargs):
        assert kwargs["revision"] == "pinned-hub-revision"
        return str(self.root / filename)
    def snapshot(self, repo, **kwargs):
        assert kwargs["revision"] == "pinned-hub-revision"
        return str(self.root)


def test_partial_hub_checkpoint_survives_lost_runtime(tmp_path, tiny_config, monkeypatch):
    import huggingface_hub
    cfg = ScreenConfig(model=tiny_config, device="cpu", batch_size=4, epochs=2,
                       max_steps=5, checkpoint_every=1, eval_every=3, eval_rows=8, benchmark=False,
                       rotary_diagnostics=False, warmup_steps=2, arms=("A2",), seeds=(1,),
                       output_dir=str(tmp_path / "reference"), resume_upload_minutes=1e-12)
    store, guard = FakeStore(), MemoryGuard(device="cpu")
    train_arm(cfg, "A2", 1, store, store, "same-data", guard, probe=False)
    remote = FakeHub(tmp_path / "remote")
    interrupted = replace(cfg, output_dir=str(tmp_path / "lost-runtime"), push=True)
    uploader = Uploader(interrupted, api=remote, sleep=lambda _: None)
    def stop(step):
        if step == 2:
            raise RuntimeError("runtime lost")
    def durable_checkpoint(*args, **kwargs):
        uploader.submit_checkpoint(*args, **kwargs)
        uploader.flush()  # Deterministically administer the completed remote snapshot.
    with pytest.raises(RuntimeError, match="runtime lost"):
        train_arm(interrupted, "A2", 1, store, store, "same-data", guard, probe=False,
                  on_step=stop, on_checkpoint=durable_checkpoint)
    uploader.close()
    marker = next(remote.root.rglob("resume.json"))
    assert json.loads(marker.read_text())["step"] == 2
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: remote)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", remote.download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", remote.snapshot)
    cold = replace(cfg, output_dir=str(tmp_path / "fresh-machine"), push=True)
    pull_completed(cold, "same-data")
    state = torch.load(Path(cold.output_dir) / "A2-seed1/state.pt", weights_only=True)
    assert state["step"] == 2
    train_arm(cold, "A2", 1, store, store, "same-data", guard, probe=False)
    def weights(root):
        return torch.load(Path(root) / "A2-seed1/model.pt", weights_only=True)
    for key, value in weights(cfg.output_dir).items():
        torch.testing.assert_close(value, weights(cold.output_dir)[key], rtol=0, atol=0)
    rows = [json.loads(line) for line in (Path(cold.output_dir) / "A2-seed1/train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
    newer = replace(cold, output_dir=str(tmp_path / "newer-local"))
    newer_dir = Path(newer.output_dir) / "A2-seed1"
    newer_dir.mkdir(parents=True)
    shutil.copy2(Path(cfg.output_dir) / "A2-seed1/state.pt", newer_dir / "state.pt")
    before_newer = (newer_dir / "state.pt").read_bytes()
    pull_completed(newer, "same-data")
    assert (newer_dir / "state.pt").read_bytes() == before_newer
    # Rejected source identities never overwrite local progress.
    metadata = json.loads(marker.read_text())
    metadata["dataset_id"] = "foreign-data"
    atomic_json(marker, metadata)
    with pytest.raises(ValueError, match="Incompatible remote"):
        pull_completed(replace(cold, output_dir=str(tmp_path / "reject")), "same-data")


def test_upload_failure_is_durable_and_never_invalidates_completed_run(tmp_path):
    class FailingHub:
        calls = 0
        def create_repo(self, *args, **kwargs):
            pass
        def upload_folder(self, **kwargs):
            self.calls += 1
            raise ConnectionError("temporary outage")
    cfg = ScreenConfig(output_dir=str(tmp_path), push=True)
    run = tmp_path / "A2-seed1"
    atomic_json(run / "result.json", {"status": "complete"})
    (run / "model.pt").write_bytes(b"test artifact")
    before = (run / "result.json").read_bytes()
    api = FailingHub()
    uploader = Uploader(cfg, api=api, sleep=lambda _: None)
    uploader.submit("A2-seed1")
    uploader.close()
    assert api.calls == 16  # Initial retry batch and session-close retry batch.
    assert (run / "result.json").read_bytes() == before
    assert list((tmp_path / ".upload_queue").glob("*/job.json"))
    assert json.loads((tmp_path / "upload_status.json").read_text())["pending_runs"] == ["A2-seed1"]


def test_transport_and_diagnostic_options_preserve_v011_identity():
    original = ScreenConfig()
    changed = replace(original, resume_upload_minutes=2, rotary_diagnostics=False, diagnostic_rows=4)
    assert original.fingerprint() == changed.fingerprint()
    assert original.fingerprint() == "817daa9e66a8119dc6b18a9b63fad32490d1c891e5f26750e91125d9e5e14e6c"


def test_active_upload_does_not_block_training_or_next_arm(tmp_path):
    started, release = Event(), Event()
    class BusyHub:
        completed = 0
        def create_repo(self, *args, **kwargs):
            pass
        def upload_folder(self, **kwargs):
            started.set()
            assert release.wait(10), "Test failed to release upload worker"
            self.completed += 1
    cfg = ScreenConfig(output_dir=str(tmp_path), push=True)
    api = BusyHub()
    uploader = Uploader(cfg, api=api, sleep=lambda _: None)
    try:
        for name in ("A0-seed1", "A1-seed1"):
            atomic_json(tmp_path / name / "result.json", {"status": "complete"})
            uploader.submit(name)
        assert started.wait(5)
        uploader.submit_checkpoint("A2-seed1", 100, {})
        assert len(uploader.jobs) == 2
        assert api.completed == 0  # Both submissions returned while the worker remains blocked.
    finally:
        release.set()
        uploader.close()
    assert api.completed == 2
