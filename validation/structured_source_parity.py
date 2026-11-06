"""Read-only parity against the exact reviewed v0.1.103 source functions.

Usage: python validation/structured_source_parity.py --source-dir PATH --output PATH
The source directory is an audit input, never an installed dependency.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import types
import torch
from captionbert_rotary.configuration_captionbert import CaptionBertConfig
from captionbert_rotary.modeling_captionbert import CaptionBertModel, hierarchical_frequencies, hold_pairs, rotate_half


def selected_source(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            assigned = {item.id for target in node.targets for item in ast.walk(target) if isinstance(item, ast.Name)}
            if assigned and assigned <= names:
                keep.append(node)
    code = ast.Module(body=keep, type_ignores=[])
    exec(compile(code, str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    anchor_path = args.source_dir / "anchored_rope.py"
    key_path = args.source_dir / "key_rotary.py"
    namespace = {"torch": torch, "math": math}
    names = {"REL_PAIRS", "CONTENT_PAIRS", "ANCHOR_FREQS", "LEVELS", "SHARED", "PAIRS", "NTK_SCALE", "BASE", "NULL", "CODES", "CODE", "code_shape", "shared", "_check", "frequencies", "AnchorStreams"}
    source = selected_source(anchor_path, names, namespace)
    expected_w, expected_kind = source["frequencies"](2, "8x3")
    actual_w, actual_kind = hierarchical_frequencies()
    stream = source["AnchorStreams"](2, "8x3")
    torch.manual_seed(72)
    dk = torch.randint(0, 9, (2, 17, 3))
    dq = torch.randint(0, 9, (2, 17, 3))
    stream.set(dk, dq)
    expected_aq, expected_ak = stream.angles(2, 17, 0, "cpu")
    model = CaptionBertModel(CaptionBertConfig(vocab_size=128, hidden_size=128,
        num_attention_heads=4, num_hidden_layers=1, intermediate_size=256, lookup_layer=0))
    ids = torch.full((2, 17), 11)
    aq, ak, _ = model.angles(ids, ids != 0, dq, dk)
    key_source = selected_source(key_path, {"norm_key"}, {"torch": torch, "J": 8, "ar": types.SimpleNamespace(PAIRS=16)})
    key = torch.randn(2, 8, 17, 32)
    on = torch.rand(2, 17) > 0.4
    expected_key = key_source["norm_key"](key, 3, on, 2.18, 16)
    actual_key = torch.cat((key[:, :3], hold_pairs(key[:, 3], on, 2.18)[:, None], key[:, 4:]), 1)
    q = torch.randn(2, 1, 17, 32)
    wrong = aq.clone()
    wrong[..., 4] += 0.1
    wrong_delta = float((rotate_half(q, aq) - rotate_half(q, wrong)).abs().max())
    checks = {
        "frequency_fp64_max_error": float((expected_w - actual_w).abs().max()),
        "kinds_equal": torch.equal(expected_kind, actual_kind),
        "query_angles_fp32_max_error": float((expected_aq - aq).abs().max()),
        "key_angles_fp32_max_error": float((expected_ak - ak).abs().max()),
        "record_only_norm_max_error": float((expected_key - actual_key).abs().max()),
        "wrong_rotation_control_delta": wrong_delta,
    }
    passed = all(checks[name] == 0 for name in checks if name.endswith("error")) and checks["kinds_equal"] and wrong_delta > 1e-4
    record = dict(passed=passed, source_revision="3f174e3f391f983428171cefc6901b34b22274d8",
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (anchor_path, key_path)},
        checks=checks, scope="Exact hierarchical angles and record-only pair norm; failing control. WordPiece phase construction is an explicit adaptation, not a byte-table parity claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
