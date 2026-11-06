import ast
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from captionbert_rotary import build_model
from captionbert_rotary.model import RotaryLayer, embed
from captionbert_rotary.metrics import objective
from captionbert_rotary.rotary import RopeMulti, anchor_simplex, bands_for_arm, mpar_bands


def test_replication_class_is_verbatim():
    def source(path):
        text = Path(path).read_text(encoding="utf-8")
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "CaptionEncoder")
        return ast.dump(node, include_attributes=False)
    assert source("captionbert_rotary/_v2.py") == source("reference/pretrain_trainer.py")


def test_zero_rotary_attention_and_gradients_match_original(tiny_config):
    cfg = replace(tiny_config, dropout=0.0)
    reference = torch.nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, 0.0,
                                                activation="gelu", batch_first=True, norm_first=True)
    rotary = RotaryLayer(cfg, "A2")
    rotary.load_state_dict(reference.state_dict())
    rotary.rope.w.zero_()
    x1 = torch.randn(2, 7, cfg.d_model, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    valid = torch.tensor([[1] * 7, [1] * 4 + [0] * 3], dtype=torch.bool)
    pos = torch.randint(0, 5, (2, 7, 2))
    a = reference(x1, src_key_padding_mask=~valid)
    b = rotary(x2, valid, pos)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=3e-5, atol=2e-5)
    for p, q in zip(reference.parameters(), rotary.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=5e-5)


@pytest.mark.parametrize("arm", ["A0", "A1", "A2", "A3", "A2g"])
def test_padding_invariance_and_gradients(arm, tiny_config):
    cfg = replace(tiny_config, dropout=0.0)
    model = build_model(arm, cfg)
    ids = torch.tensor([[1, 3, 5, 2]])
    pos = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]])
    model.eval()
    a = embed(model, ids, torch.ones_like(ids), pos)
    padded = torch.cat((ids, torch.tensor([[6, 7, 8]])), 1)
    padded_pos = torch.cat((pos, torch.ones(1, 3, 2, dtype=torch.long)), 1)
    b = embed(model, padded, torch.tensor([[1, 1, 1, 1, 0, 0, 0]]), padded_pos)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a.norm(dim=-1), torch.ones(1))
    model.train()
    loss, _, _ = objective(embed(model, ids, torch.ones_like(ids), pos), torch.randn(1, cfg.output_dim))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_anchor_simplex_capacity():
    w = anchor_simplex(12)
    positions = torch.arange(25, dtype=torch.float64)
    delta = positions[:, None] - positions[None, :]
    kernel = torch.cos(delta[..., None] * w).mean(-1)
    expected = torch.full((25, 25), -1 / 24, dtype=torch.float64)
    expected.fill_diagonal_(1)
    torch.testing.assert_close(kernel, expected, atol=1e-13, rtol=1e-12)
    # One constant real dimension lifts the simplex to orthogonal anchors.
    torch.testing.assert_close((24 * kernel + 1) / 25, torch.eye(25, dtype=torch.float64), atol=1e-13, rtol=1e-12)


def test_bands_and_address_kernel():
    w, stream = bands_for_arm("A2", 64)
    assert w.shape == (32,) and (w[:8] == 0).all()
    assert stream.tolist() == [0] * 20 + [1] * 12
    q = torch.ones(1, 1, 1, 64)
    pq, pk = torch.tensor([[[3, 2]]]), torch.tensor([[[7, 4]]])
    qr, kr = RopeMulti(w, stream)(q, q, pq, pk)
    expected = torch.cos(w * (pq.float()[..., stream] - pk.float()[..., stream])).mean()
    torch.testing.assert_close((qr * kr).sum() / 64, expected.float())
    a, s = bands_for_arm("A3", 64)
    assert not s.any() and (a[:8] == 0).all() and (a[8:] > 0).all()
    with pytest.raises(ValueError):
        mpar_bands(32, 8, 12, 11)


def test_parameter_counts():
    a = build_model("A0")
    assert sum(p.numel() for p in a.parameters()) == 58_308_864
    del a
    b = build_model("A2")
    assert sum(p.numel() for p in b.parameters()) == 54_114_560
    assert not any("pos_emb" in k for k in b.state_dict())
