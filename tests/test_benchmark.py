from types import SimpleNamespace
import numpy as np
import pytest
import torch
from captionbert_rotary.benchmark import TASKS, run_benchmark


def test_all_eight_tasks_are_required_and_revisions_pinned(tmp_path, monkeypatch):
    import datasets
    import huggingface_hub
    revisions, calls = [], []

    class API:
        def dataset_info(self, repo):
            calls.append(repo)
            return SimpleNamespace(sha="pinned-sha")

    class Dataset:
        column_names = ["sentence1", "sentence2", "score"]
        _fingerprint = "example-fingerprint"
        def __getitem__(self, key):
            return [0.1, 0.5, 0.9] if key == "score" else [key] * 3

    def load(repo, **kwargs):
        revisions.append(kwargs)
        return Dataset()

    def encode(model, tokenizer, texts, device, batch_size):
        if texts[0] == "sentence1":
            return torch.tensor([[1., 0.]] * 3)
        cos = torch.tensor([0.1, 0.5, 0.9])
        return torch.stack((cos, (1 - cos.square()).sqrt()), -1)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(datasets, "load_dataset", load)
    monkeypatch.setattr("captionbert_rotary.benchmark.encode_texts", encode)
    cfg = SimpleNamespace(output_dir=str(tmp_path), device="cpu", eval_batch=2)
    result = run_benchmark(None, None, cfg)
    assert len(result["tasks"]) == 8 and result["mean"] == pytest.approx(1.0)
    assert all(r == {"split": "test", "revision": "pinned-sha"} for r in revisions)
    run_benchmark(None, None, cfg)
    assert len(calls) == 8  # The second arm reuses the same revisions.

    def fail_one(repo, **kwargs):
        if repo == TASKS[-1][1]:
            raise ConnectionError("task unavailable")
        return Dataset()
    monkeypatch.setattr(datasets, "load_dataset", fail_one)
    with pytest.raises(ConnectionError):
        run_benchmark(None, None, cfg)
