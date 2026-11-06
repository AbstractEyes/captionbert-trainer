import ast
import json
from pathlib import Path
from types import SimpleNamespace
import textwrap
import numpy as np
import pytest
import torch
from captionbert_rotary.config import ScreenConfig
from captionbert_rotary.data import Store, atomic_json, read_captions, plan_batches, sentence_positions, sha256_file
from captionbert_rotary.metrics import geometry, objective


def test_sentence_boundaries_and_clamp():
    pos, reaches, overflows = sentence_positions([101, 10, 11, 12, 10, 13, 102], {11, 13})
    assert pos.tolist() == [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2]]
    assert not reaches and not overflows
    pos, reaches, overflows = sentence_positions([11] * 27, {11})
    assert pos[-1].tolist() == [0, 24] and reaches and overflows


def test_batch_plan_matches_verbatim_v2():
    source = Path("reference/pretrain_trainer.py").read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "RamStore")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "plan_batches")
    namespace = {"np": np}
    exec(textwrap.dedent(ast.get_source_segment(source, method)), namespace)
    lengths = np.arange(103) % 17 + 2
    for seed in (1, 2):
        expected = namespace["plan_batches"](SimpleNamespace(n=len(lengths), lens=lengths), 8, seed, 4)
        actual = plan_batches(lengths, 8, seed, 4)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        assert len(np.unique(np.concatenate(actual))) == 96


def test_strict_row_pairing_and_corruption(tmp_path):
    d = tmp_path / "chunk"
    d.mkdir()
    # This is already flattened; one missing caption never changes the subsequent row IDs.
    tokens = np.array([101, 10, 102, 101, 20, 102], dtype=np.uint16)
    np.save(d / "tokens.npy", tokens)
    np.save(d / "positions.npy", np.zeros((6, 2), dtype=np.int16))
    np.save(d / "offsets.npy", np.array([0, 3, 6], dtype=np.int64))
    target = d / "consensus.pt"
    torch.save(torch.tensor([[1., 2.], [7., 8.]]), target)
    atomic_json(d / "manifest.json", {"rows": 2, "target_path": str(target),
                "target_sha256": sha256_file(target), "prepared_sha256": {
                    n: sha256_file(d / n) for n in ("tokens.npy", "positions.npy", "offsets.npy")}})
    store = Store([d])
    ids, _, _, targets = store.batch([1, 0])
    assert ids[:, 1].tolist() == [20, 10]
    assert targets[:, 0].tolist() == [7., 1.]
    with target.open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="Corrupt consensus"):
        Store([d])


def test_caption_schema_not_image_join(tmp_path):
    path = tmp_path / "captions.json"
    path.write_text(json.dumps(["long", "short", "alt", "short without long", "alt next"]))
    assert read_captions(path)[3:] == ["short without long", "alt next"]
    path.write_text('[{"caption": "bad schema"}]')
    with pytest.raises(ValueError):
        read_captions(path)


def test_geometry_is_original_participation_ratio():
    x = torch.randn(24, 16)
    centered = (x - x.mean(0)).double()
    variance = torch.linalg.svdvals(centered).square()
    expected = (variance.sum().square() / variance.square().sum()).item()
    actual = geometry(x)
    assert actual["erank"] == pytest.approx(expected, rel=1e-6)
    u = torch.nn.functional.normalize(x.double(), dim=-1)
    matrix = u @ u.T
    matrix.fill_diagonal_(0)
    assert actual["self_cos"] == pytest.approx(float(matrix.sum() / (24 * 23)), abs=1e-12)
    assert geometry(torch.ones(4, 16))["erank"] == 0


def test_objective_is_symmetric():
    a = torch.nn.functional.normalize(torch.randn(7, 16), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(7, 16), dim=-1)
    loss, nce, mse = objective(a, b)
    logits = a @ b.T / .07
    labels = torch.arange(7)
    expected = (torch.nn.functional.cross_entropy(logits, labels) +
                torch.nn.functional.cross_entropy(logits.T, labels)) / 2
    torch.testing.assert_close(nce, expected)
    torch.testing.assert_close(loss, expected + torch.nn.functional.mse_loss(a, b))


@pytest.mark.parametrize("options", [{"train_chunks": (0, 61)}, {"train_chunks": (5,)},
                                           {"batch_size": 0}, {"seeds": (1, 1)}, {"max_tokens": 257}])
def test_invalid_screen_config(options):
    with pytest.raises(ValueError):
        ScreenConfig(**options)
