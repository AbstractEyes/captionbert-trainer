from pathlib import Path
import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from captionbert_rotary.configuration_captionbert import CaptionBertConfig
from captionbert_rotary.structured_campaign import CampaignConfig
from captionbert_rotary import structured_campaign as campaign
from captionbert_rotary.hub import Uploader
from captionbert_rotary.memory import MemoryGuard


class SyntheticStore:
    lengths = np.full(16, 8, dtype=np.int32)

    def __len__(self):
        return 16

    def batch(self, indices):
        ids = torch.tensor([[1, 10, 11, 12, 10, 11, 2, 0]]).repeat(len(indices), 1)
        teacher = torch.randn(16, 768, generator=torch.Generator().manual_seed(100))[indices]
        targets = torch.full_like(ids, -1)
        targets[:, 4:6] = 2
        return ids, ids.ne(0), torch.zeros(len(ids), 8, 2, dtype=torch.long), teacher, targets


class SyntheticTokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer_config.json").write_text('{}', encoding="utf-8")


def test_interrupted_resume_matches_uninterrupted_and_logs_tensorboard(tmp_path, monkeypatch):
    """Three synthetic optimizer steps test state replay, never training accuracy."""
    torch.set_num_threads(2)
    monkeypatch.setattr(campaign, "model_config", lambda arm, tokenizer: CaptionBertConfig(
        vocab_size=128, hidden_size=128, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=256, lookup_layer=0, continuation_ids=[11], special_ids=[0, 1, 2],
        punctuation_ids=[5], hidden_dropout_prob=0.1))
    scores = {"heldout": {"cos_to_target": 0.0, "erank": 8.0}}
    monkeypatch.setattr(campaign, "evaluate", lambda *args: scores)
    monkeypatch.setattr(campaign, "dev_sts", lambda *args: {"spearman": 0.0, "n": 16})
    store, tokenizer = SyntheticStore(), SyntheticTokenizer()

    def run(root):
        cfg = CampaignConfig(stage="engineering", arms=("KALLP",), seeds=(1,),
            epochs=1, batch_size=4, warmup_steps=1, max_steps=3, device="cpu",
            checkpoint_every=1, eval_every=3, output_dir=str(root), push=False)
        uploader = Uploader(cfg)
        try:
            return campaign.run_one(cfg, "KALLP", 1, store, None, tokenizer,
                                    "synthetic-state-replay", MemoryGuard(device="cpu"), uploader)
        finally:
            uploader.close()

    complete = tmp_path / "complete"
    resumed = tmp_path / "resumed"
    run(complete)
    original_step = campaign.step
    calls = 0

    def interrupt(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated loss of runtime")
        return original_step(*args)

    monkeypatch.setattr(campaign, "step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(resumed)
    monkeypatch.setattr(campaign, "step", original_step)
    run(resumed)
    expected = load_file(complete / "KALLP-seed1/model.safetensors")
    actual = load_file(resumed / "KALLP-seed1/model.safetensors")
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    events = EventAccumulator(str(resumed / "KALLP-seed1/tensorboard")).Reload()
    assert "train/loss" in events.Tags()["scalars"]
    assert "final/stsb_development/spearman" in events.Tags()["scalars"]
    assert [event.step for event in events.Scalars("train/loss")] == [1, 2, 3]
