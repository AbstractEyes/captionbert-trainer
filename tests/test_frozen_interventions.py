import torch
import pytest
from captionbert_rotary.configuration_captionbert import CaptionBertConfig
from captionbert_rotary.modeling_captionbert import CaptionBertModel
from tools.probe_structured_campaign import intervention
import captionbert_rotary.modeling_captionbert as modeling


@pytest.mark.parametrize("name", ["anchor_angles_off", "value_addresses_constant"])
def test_frozen_tensor_intervention_restores_even_after_exception(name):
    cfg = CaptionBertConfig(hidden_size=128, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=256, lookup_layer=0)
    model = CaptionBertModel(cfg)
    with torch.no_grad():
        model.value_address.copy_(torch.arange(model.value_address.numel()).reshape_as(model.value_address))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="interrupted"):
        with intervention(model, name):
            if name == "anchor_angles_off":
                assert torch.count_nonzero(model.frequencies[model.kinds >= 2]) == 0
                assert torch.equal(model.frequencies[model.kinds == 0], original["frequencies"][model.kinds == 0])
            else:
                assert torch.equal(model.value_address, original["value_address"][:, :1].expand_as(model.value_address))
            raise RuntimeError("interrupted probe")
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())


def test_constant_phase_preserves_record_identity_norm_and_restores_on_exception():
    cfg = CaptionBertConfig(hidden_size=128, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=128, lookup_layer=0, continuation_ids=[2101])
    model = CaptionBertModel(cfg)
    ids = torch.tensor([[101, 2100, 2101, 2102, 2100, 2101, 102]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    original_function = modeling.word_features
    arguments = (ids, valid, model.wordpiece_continuation, model.special_token)
    expected = original_function(*arguments)
    assert torch.count_nonzero(expected[0]) > 0 and expected[3].any()
    with pytest.raises(RuntimeError, match="interrupted"):
        with intervention(model, "phase_constant"):
            got = modeling.word_features(*arguments)
            assert torch.count_nonzero(got[0]) == 0
            assert all(torch.equal(a, b) for a, b in zip(got[1:], expected[1:]))
            assert model.config.rotary_mode == "key_all"
            assert model.layers[0].rho == model.config.key_rho == 2.18
            raise RuntimeError("interrupted phase control")
    assert modeling.word_features is original_function
    assert all(torch.equal(a, b) for a, b in zip(modeling.word_features(*arguments), expected))
