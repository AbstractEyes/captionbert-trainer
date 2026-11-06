import ast
import json
from pathlib import Path
import tomllib


def test_colab_cells_are_complete_python_and_tagged():
    nb = json.loads(Path("notebooks/captionbert_rotary_screen.ipynb").read_text())
    assert "colab-badge.svg" in nb["cells"][0]["source"]
    code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
        assert "argparse" not in source and "__file__" not in source
    assert 'release = "v0.2.0"' in code[0]
    assert '"--target"' in code[0] and '"virtualenv==20.38.0"' in code[0]
    assert '"--no-download"' in code[0] and '"--system-site-packages"' not in code[0]
    assert "userdata.get(\"HF_TOKEN\")" in "\n".join(code)
    assert "print(child_env" not in "\n".join(code)
    for source in code[:4]:
        assert "def run_logged(" in source and "stderr=subprocess.STDOUT" in source
        assert 'child_env["PYTHONUNBUFFERED"] = "1"' in source


def test_requirements_match_package_metadata():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    reqs = Path("requirements.txt").read_text().splitlines()
    assert set(project["dependencies"]) == set(reqs)
    assert "-r requirements-cuda.txt" in Path("requirements-colab.txt").read_text()
