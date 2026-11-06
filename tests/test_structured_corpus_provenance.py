import numpy as np
import torch

from captionbert_rotary.configuration_captionbert import CaptionBertConfig
from captionbert_rotary.modeling_captionbert import CaptionBertModel
from tools.ship_structured_provenance import histogram_summary, maximum_segment_address


def test_corpus_address_count_agrees_with_actual_model_at_boundaries():
    rows = [[101, 2100, 1012, 102], [101, 1012, 999, 2100, 102],
            [101, 2100, 1029], [101] + [1012]*64 + [102]]
    offsets = np.asarray([0] + list(np.cumsum([len(row) for row in rows])))
    got = maximum_segment_address(np.asarray(sum(rows, [])), offsets, [1012, 999, 1029])
    cfg = CaptionBertConfig(hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=128, lookup_layer=0)
    model = CaptionBertModel(cfg)
    expected = []
    for row in rows:
        ids = torch.tensor([row])
        digits = model.addresses(ids, torch.ones_like(ids, dtype=torch.bool))[0]
        expected.append(int((digits[:, 0]*64 + digits[:, 1]*8 + digits[:, 2]).max()))
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got, [1, 2, 0, 64])


def test_histogram_quantiles_match_direct_numpy_observations():
    values = np.asarray([0, 1, 1, 3, 7, 7, 7, 9])
    summary = histogram_summary(np.bincount(values, minlength=10))
    for q in (.5, .9, .99, 1.):
        assert summary["quantiles"][str(q)] == np.quantile(values, q)
    assert summary["rows"] == len(values) and summary["mean"] == values.mean()
