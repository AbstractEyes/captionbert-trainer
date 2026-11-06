import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from captionbert_rotary.data import atomic_json


def test_diagnostic_notebook_is_separate_cpu_and_pinned():
    notebook = json.loads(Path("notebooks/captionbert_rotary_diagnostics.ipynb").read_text())
    assert "colab-badge.svg" in notebook["cells"][0]["source"]
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
        assert "argparse" not in source and "__file__" not in source
    assert 'release = "v0.2.0"' in code[0]
    assert "requirements-cpu.txt" in code[0]
    assert "/content/captionbert-diagnostic-env" in code[0]
    assert "torch.cuda.is_available" not in code[0]
    assert "analyse_available(" in code[1] and "device='cpu'" in code[1]
    assert "run_screen(" not in "\n".join(code)


def test_diagnostic_summary_cell_handles_absolute_and_rotary_controls(tmp_path, monkeypatch):
    output = []
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(Markdown=lambda text: text,
                                                                      display=output.append))
    entries = []
    for arm, change in (("A0", None), ("A1", 0.0), ("A2", .2)):
        path = tmp_path / f"{arm}-seed1/rotary_diagnostics.json"
        atomic_json(path, {"identity": {"step": 1952}, "repeat_floor": {"max_absolute_change": 0},
                          "interventions": {"sentence_ids_collapsed": {"mean_l2_change": change},
                                            "remove_anchor": {"mean_l2_change": change}}})
        entries.append({"name": f"{arm}-seed1", "path": str(path)})
    atomic_json(tmp_path / "summary.json", {"completed": entries, "awaiting_completed_arms": []})
    notebook = json.loads(Path("notebooks/captionbert_rotary_diagnostics.ipynb").read_text())
    table = notebook["cells"][-1]["source"]
    assert "/content/captionbert-diagnostics" in table
    table = table.replace("/content/captionbert-diagnostics", tmp_path.as_posix())
    exec(compile(table, "diagnostic-summary-cell", "exec"), {})
    assert "A0-seed1" in output[0] and "n/a" in output[0] and "A2-seed1" in output[0]
