import torch
import pytest
from captionbert_rotary.configuration_captionbert import CaptionBertConfig
from captionbert_rotary.modeling_captionbert import CaptionBertModel
from tools.probe_structured_campaign import intervention


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
