from dataclasses import replace
import json
import pytest
import torch
from transformers import BertTokenizerFast
from captionbert_rotary import build_model, ScreenConfig
from captionbert_rotary.diagnostics import analytic_report, controlled_lookup, run_rotary_diagnostics


@pytest.fixture
def probe_tokenizer(tmp_path):
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "!", "?", "the", "red", "box",
             "is", "here", "blue", "ball", "nearby", "a", "small", "bird", "watches", "street",
             "person", "walks", "past", "window"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(words) + "\n")
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)


def test_analytic_capacity_and_oracle_controls():
    report = analytic_report("A2")
    anchor = report["bands"]["anchor"]
    assert anchor["pairs"] == 12 and anchor["code_count"] == 25
    assert anchor["raw_coherence"] == pytest.approx(1 / 24, abs=1e-12)
    assert anchor["explicit_constant_lift_max_identity_error"] < 1e-12
    assert not anchor["encoder_implements_constant_lift"]
    assert anchor["first_periodic_alias"] == [0, 25]
    assert anchor["first_clamp_collision"] == [24, 25]
    assert controlled_lookup("A2")["tie_aware_success"] == pytest.approx(1.0)
    assert controlled_lookup("A3")["tie_aware_success"] == pytest.approx(.125)
    assert controlled_lookup("A3")["mean_maximum_tie_count"] == 8
    assert controlled_lookup("A1")["tie_aware_success"] == pytest.approx(1.0)
    assert "global target" in controlled_lookup("A1")["address_source"]


@pytest.mark.parametrize("arm", ["A0", "A1", "A2", "A3", "A2g"])
def test_frozen_interventions_and_band_controls(tmp_path, tiny_config, probe_tokenizer, arm):
    cfg = ScreenConfig(model=replace(tiny_config, dropout=0), device="cpu", eval_batch=2)
    torch.manual_seed(3)
    model = build_model(arm, cfg.model).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "diagnostics.json"
    result = run_rotary_diagnostics(model, probe_tokenizer, arm, cfg, path, rows=4)
    assert result["baseline_finite"] and model.training
    assert result["repeat_floor"]["max_absolute_change"] == 0
    assert json.loads(path.read_text())["protocol"] == "captionbert-rotary-mechanisms-v1"
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    if arm == "A0":
        assert not result["layers"] and not result["analytic"]["applicable"]
        return
    for layer in model.encoder.layers:
        assert not layer.rope._forward_hooks and not layer.rope._forward_pre_hooks
    assert len(result["layers"]) == cfg.model.n_layers
    assert result["interventions"]["common_slot_shift"]["max_absolute_change"] < 1e-5
    assert result["interventions"]["common_sentence_shift"]["max_absolute_change"] < 1e-5
    if arm in ("A1", "A3"):
        assert result["interventions"]["sentence_ids_collapsed"]["max_absolute_change"] == 0
        assert result["interventions"]["remove_anchor"]["max_absolute_change"] == 0
    else:
        assert result["interventions"]["sentence_ids_collapsed"]["mean_l2_change"] > 1e-5
    for layer in result["layers"]:
        for head in layer["heads"]:
            assert sum(head[b + "_q_energy_fraction"] for b in ("content", "slot", "anchor")) == pytest.approx(1, abs=1e-5)


def test_diagnostic_exception_cleans_hooks(tiny_config, probe_tokenizer, monkeypatch):
    model = build_model("A2", tiny_config)
    original = model.forward
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected probe fault")
        return original(*args, **kwargs)
    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected probe fault"):
        run_rotary_diagnostics(model, probe_tokenizer, "A2", ScreenConfig(model=tiny_config, device="cpu"), rows=2)
    for layer in model.encoder.layers:
        assert not layer.rope._forward_hooks and not layer.rope._forward_pre_hooks
