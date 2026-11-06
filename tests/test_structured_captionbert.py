import json
import torch
import pytest
from transformers import AutoConfig, AutoModel
from safetensors import safe_open
from captionbert_rotary.configuration_captionbert import CaptionBertConfig
from captionbert_rotary.modeling_captionbert import CaptionBertModel, hierarchical_frequencies, hold_pairs, rotate_half, word_features
from captionbert_rotary.structured_data import exact_word_targets


def config(mode="key_all", **kwargs):
    return CaptionBertConfig(vocab_size=128, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=256, lookup_layer=1,
        hidden_dropout_prob=0.0, continuation_ids=[11], special_ids=[0, 1, 2],
        punctuation_ids=[5], rotary_mode=mode, **kwargs)


def inputs():
    return torch.tensor([[1, 10, 11, 12, 10, 11, 2, 0], [1, 13, 14, 2, 0, 0, 0, 0]])


def test_complete_key_and_propagation_controls():
    model = CaptionBertModel(config()).eval()
    ids = torch.tensor([[1, 10, 11, 12, 10, 11, 13, 11, 2]])
    phases, active, ends, records = word_features(ids, ids != 0, model.wordpiece_continuation, model.special_token)
    assert torch.equal(phases[0, 1], phases[0, 2])
    assert torch.equal(phases[0, 1], phases[0, 4])
    assert torch.equal(phases[0, 4], phases[0, 5])
    # Same suffix, different first piece must not share the complete key.
    assert not torch.allclose(phases[0, 2], phases[0, 7])
    assert ends[0, 2] and ends[0, 5] and not ends[0, 1]
    assert records[0, 2] and not records[0, 5]
    assert not active[0, 0] and not active[0, -1]


def test_kernel_simplex_and_norm_hold():
    w, kinds = hierarchical_frequencies()
    for level in range(3):
        band = w[kinds == level + 2]
        assert torch.allclose(torch.cos(torch.arange(1, 9)[:, None] * band).sum(-1), torch.full((8,), -0.5, dtype=torch.float64), atol=1e-13)
    torch.manual_seed(0)
    q, k = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
    angle = torch.randn(2, 5, 16)
    assert torch.allclose((q * k).sum(-1), (rotate_half(q, angle) * rotate_half(k, angle)).sum(-1), atol=4e-6)
    on = torch.tensor([[True, False, True, False, False]]).expand(2, -1)
    held = hold_pairs(k, on, 2.18)
    norms = (held[..., :16].square() + held[..., 16:].square()).sqrt()
    assert torch.allclose(norms[on], torch.full_like(norms[on], 2.18), atol=3e-7)
    assert torch.equal(held[~on], k[~on])
    zero = torch.zeros(2, 5, 32, requires_grad=True)
    hold_pairs(zero, torch.zeros(2, 5, dtype=torch.bool), 2.18).sum().backward()
    assert torch.equal(zero.grad, torch.ones_like(zero))


@pytest.mark.parametrize("mode", ["rope", "ntk", "hierarchical", "key_end", "key_all"])
def test_outputs_padding_and_gradient(mode):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = CaptionBertModel(config(mode)).eval()
    ids = inputs()
    out = model(ids, output_hidden_states=True)
    assert out.pooler_output.shape == (2, 768)
    assert out.last_hidden_state.shape == (2, 8, 128)
    assert len(out.hidden_states) == 3
    assert torch.allclose(out.pooler_output.norm(dim=-1), torch.ones(2), atol=2e-6)
    extended = torch.nn.functional.pad(ids, (0, 5))
    assert torch.allclose(out.pooler_output, model(extended).pooler_output, atol=2e-6)
    assert torch.equal(model(torch.zeros_like(ids)).pooler_output, torch.zeros(2, 768))
    model.train()
    target = torch.full_like(ids, -1)
    target[0, 4:6] = 2
    out = model(ids, pointer_targets=target if mode.startswith("key") else None)
    loss = (out.pooler_output[:, :10].square().mean() + out.pointer_loss * 0.1)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.layers[1].self_attn.in_proj_weight.grad.norm() > 0


def test_exact_targets_are_not_self_or_suffix_targets():
    vocabulary = [f"unused{i}" for i in range(128)]
    vocabulary[10:14] = ["walk", "##ing", "talk", "tree"]
    ids = [1, 10, 11, 12, 11, 10, 11, 2]
    targets = exact_word_targets(ids, vocabulary, {0, 1, 2})
    assert targets.tolist() == [-1, -1, -1, -1, -1, 2, 2, -1]


def test_safetensors_auto_model_round_trip(tmp_path):
    torch.manual_seed(11)
    model = CaptionBertModel(config()).eval()
    CaptionBertConfig.register_for_auto_class("AutoConfig")
    CaptionBertModel.register_for_auto_class("AutoModel")
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / "pytorch_model.bin").exists()
    with safe_open(tmp_path / "model.safetensors", framework="pt") as file:
        assert file.metadata()["format"] == "pt"
    raw = json.loads((tmp_path / "config.json").read_text())
    assert set(raw["auto_map"]) == {"AutoConfig", "AutoModel"}
    auto_config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)
    assert auto_config.model_type == CaptionBertConfig.model_type
    loaded, info = AutoModel.from_pretrained(tmp_path, trust_remote_code=True, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    loaded.eval()
    expected, actual = model(inputs()).pooler_output, loaded(inputs()).pooler_output
    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=1e-5)
