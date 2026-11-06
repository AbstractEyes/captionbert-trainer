import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import pytest
from captionbert_rotary.notebook import run_logged
from captionbert_rotary.config import ScreenConfig
from captionbert_rotary.data import atomic_json
from captionbert_rotary.memory import MemoryGuard
from captionbert_rotary.train import train_arm
from conftest import FakeStore

spec = importlib.util.spec_from_file_location("watch_run", Path("tools/watch_run.py"))
watch_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch_run)


def test_streams_before_child_finishes_and_logs_stderr(tmp_path):
    events = []
    def record(*args, **kwargs):
        events.append((" ".join(map(str, args)), time.monotonic()))
    with patch("builtins.print", record):
        run_logged([sys.executable, "-c",
                    "import time,sys; print('first',flush=True); time.sleep(0.4); "
                    "print('last',file=sys.stderr,flush=True)"], tmp_path / "console.log")
    first = next(t for s, t in events if s.strip() == "first")
    last = next(t for s, t in events if s.strip() == "last")
    assert last - first >= .2
    assert (tmp_path / "console.log").read_text() == "first\nlast\n"


def test_streamed_failure_surfaces_output_and_exit_code(tmp_path, capsys):
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_logged([sys.executable, "-c", "print('failure detail'); raise SystemExit(7)"],
                   tmp_path / "console.log")
    assert error.value.returncode == 7
    assert "failure detail" in capsys.readouterr().out
    assert "failure detail" in (tmp_path / "console.log").read_text()


def test_monitor_reads_running_v010_logs_without_modifying_them(tmp_path, capsys):
    root = tmp_path / "screen"
    atomic_json(root / "experiment.json", {"config": {"arms": ["A0"], "seeds": [1],
                "epochs": 2, "batch_size": 4}, "data": [[{"rows": 20}], []]})
    log = root / "A0-seed1/train.jsonl"
    log.parent.mkdir()
    log.write_text(json.dumps({"step": 2, "epoch": 0, "loss": 2.3, "nce": 2.2,
                   "mse": .1, "lr": .0001, "grad_norm": 1.2, "seconds": .5}) + '\n{"step":')
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    state = watch_run.watch(root, once=True)
    output = capsys.readouterr().out
    assert "step 2/10" in output and "NCE 2.20000" in output and "MSE 0.100000" in output
    assert "excludes evaluation/uploads" in output
    assert state["runs"][0]["last"]["step"] == 2
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_training_displays_step_and_metrics_and_saves_enriched_log(tmp_path, tiny_config, capsys):
    cfg = ScreenConfig(model=tiny_config, device="cpu", batch_size=4, epochs=1, max_steps=2,
                       checkpoint_every=2, log_every=25, eval_rows=8, benchmark=False,
                       output_dir=str(tmp_path))
    store = FakeStore()
    train_arm(cfg, "A2", 1, store, store, "same-data", MemoryGuard(device="cpu"), probe=False)
    output = capsys.readouterr().out
    for text in ("step 1/2", "step 2/2", "NCE", "MSE", "rows/s", "lr", "grad",
                 "saving checkpoint", "checkpoint saved", "Validation heldout", "completed 2/2"):
        assert text in output
    rows = [json.loads(line) for line in (tmp_path / "A2-seed1/train.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["total_steps"] == 2 and row["rows_per_second"] > 0
        assert row["tokens_per_second"] > 0 and row["lr_used"] > 0
    assert rows[-1]["eta_train_seconds"] == 0
