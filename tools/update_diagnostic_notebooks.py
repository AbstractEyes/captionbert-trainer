"""Build the tagged diagnostic notebook from the tested existing notebook launcher."""
import copy
import json
from pathlib import Path

path = Path("notebooks/captionbert_rotary_screen.ipynb")
main = json.loads(path.read_text(encoding="utf-8"))
for cell in main["cells"]:
    cell["source"] = cell["source"].replace("v0.1.1", "v0.2.0")
continuation_text = (
    "\nThe current release also ships partial resume checkpoints after the first saved\n"
    "training checkpoint and then every ten minutes. A lost runtime fetches its\n"
    "latest compatible durable state; newer local progress wins. Upload outages\n"
    "retain their queue and are listed in the final summary. Future arms collect\n"
    "frozen rotary mechanism diagnostics and retain evaluation-step weights.\n")
if continuation_text not in main["cells"][6]["source"]:
    main["cells"][6]["source"] += continuation_text
path.write_text(json.dumps(main, indent=2) + "\n", encoding="utf-8")

def markdown(text, ident):
    return {"cell_type": "markdown", "id": ident, "metadata": {}, "source": text}

def code(text, ident):
    return {"cell_type": "code", "id": ident, "metadata": {}, "source": text,
            "execution_count": None, "outputs": []}

setup = main["cells"][1]["source"]
for old, new in (("/content/captionbert-env", "/content/captionbert-diagnostic-env"),
                 ("/content/captionbert-install", "/content/captionbert-diagnostic-install"),
                 ("/content/captionbert-bootstrap", "/content/captionbert-diagnostic-bootstrap"),
                 ('("requirements.txt", "requirements-cuda.txt", "requirements-colab.txt")',
                  '("requirements.txt", "requirements-cpu.txt")'),
                 ('req_dir / "requirements-colab.txt"', 'req_dir / "requirements-cpu.txt"'),
                 ('"assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"',
                  '"print(\'CPU diagnostic environment ready\')"')):
    assert old in setup, old
    setup = setup.replace(old, new)

training_source = main["cells"][7]["source"]
helper = training_source[:training_source.index("import os\nimport subprocess\nimport textwrap\nfrom google.colab")]
analysis = helper + '''import os
import textwrap
from google.colab import userdata

python = "/content/captionbert-diagnostic-env/bin/python"
child_env = os.environ.copy()
child_env["HF_TOKEN"] = userdata.get("HF_TOKEN")
run_logged([python, "-u", "-c", textwrap.dedent("""
    from captionbert_rotary.analyse import analyse_available
    analyse_available(
        repo='AbstractPhil/captionbert-8192-rotary',
        experiment='817daa9e66a8119dc6b18a9b63fad32490d1c891e5f26750e91125d9e5e14e6c',
        arms=('A0', 'A1', 'A2', 'A3'), seeds=(1, 2),
        device='cpu', threads=2, rows=16, batch_size=4,
        data_dir='/content/captionbert-diagnostic-data',
        output_dir='/content/captionbert-diagnostics', push=True,
    )
    from captionbert_rotary.model_card import finalize_model_card
    finalize_model_card(
        repo='AbstractPhil/captionbert-8192-rotary',
        experiment='817daa9e66a8119dc6b18a9b63fad32490d1c891e5f26750e91125d9e5e14e6c',
        output_dir='/content/captionbert-diagnostics/model-card', push=True,
    )
""")], env=child_env, log_path="/content/captionbert-logs/diagnostics.log")
'''
table = '''import json
from pathlib import Path
from IPython.display import Markdown, display

root = Path("/content/captionbert-diagnostics")
summary = json.loads((root / "summary.json").read_text())
lines = ["| Arm/seed | Step | Repeat max error | Sentence-ID change (mean L2) | Anchor removal (mean L2) |",
         "|---|---:|---:|---:|---:|"]
def number(value):
    return f"{value:.6g}" if value is not None else "n/a"
for entry in summary["completed"]:
    report = json.loads(Path(entry["path"]).read_text())
    cases = report["interventions"]
    lines.append(f"| {entry['name']} | {report['identity']['step']} | "
                 f"{number(report['repeat_floor']['max_absolute_change'])} | "
                 f"{number(cases['sentence_ids_collapsed'].get('mean_l2_change'))} | "
                 f"{number(cases['remove_anchor'].get('mean_l2_change'))} |")
display(Markdown("\\n".join(lines)))
print("Awaiting completed arms:", summary["awaiting_completed_arms"])
print("Unshipped analyses:", summary.get("shipping", {}).get("pending_runs", []))
print("Full per-layer/head tables and probe inputs:", root)
'''
metadata = copy.deepcopy(main["metadata"])
metadata.pop("accelerator", None)
metadata["colab"]["name"] = "CaptionBERT rotary mechanism diagnostics"
notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": metadata, "cells": [
    markdown("""# CaptionBERT rotary mechanism diagnostics
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.2.0/notebooks/captionbert_rotary_diagnostics.ipynb)

Use a **separate CPU runtime** for this notebook. The existing GPU training notebook
continues unchanged. Install tagged v0.2.0 in its own environment, then measure only
completed frozen checkpoints. No training or consensus-bank download occurs.
Each checkpoint is at most 300 MB; the fixed probe set has 16 texts of at most 256
tokens. Two CPU threads are used. Enable HF_TOKEN to upload the additive JSON reports.
""", "diagnostics-00"),
    code(setup, "diagnostics-01"),
    markdown("""The analysis first pins the Hub revision. It skips unfinished arms and reuses
published measurements with matching checkpoint hashes and probe inputs. Run the
analysis cell again later for newly completed arms. It never overwrites original
weights or task results; uploads go under each run's `analyses/` directory.

Measurements: ideal address kernels/coherence/periodic aliases/clamping; identical-content
requested-address ambiguity; frozen address changes and per-band q/k removals;
and per-layer/head score, energy, and attention-mass measurements. The code-only
lookup has an oracle request. It is not a trained semantic-retrieval result.

The cell also prepares the model README in the original v2 card's format. It
publishes only when every arm/seed, all eight task scores and matching frozen
measurements are present. Until then it saves a clearly marked local draft and
prints the missing checks. Original run results remain unchanged.
""", "diagnostics-02"),
    code(analysis, "diagnostics-03"),
    markdown("""The compact table is a navigation aid. The JSON reports retain exact probe
texts/tokens/addresses, checkpoint hashes, dtype/device, numerical repeat, all
interventions, and every layer/head. A0 has no rotary bands, so its band-specific
entries are n/a. A1/A3 have no anchor band and should respond by zero to anchor
removal. A2's sensitivity alone does not establish an improvement; compare both
seeds and the original held-out/task metrics separately.
""", "diagnostics-04"),
    code(table, "diagnostics-05"),
]}
Path("notebooks/captionbert_rotary_diagnostics.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")

readme = Path("README.md")
text = readme.read_text(encoding="utf-8").replace("v0.1.1", "v0.2.0")
# Keep the legacy monitor link pinned to its original published implementation.
text = text.replace("/v0.2.0/tools/watch_run.py", "/v0.1.1/tools/watch_run.py")
readme.write_text(text, encoding="utf-8")
test = Path("tests/test_notebook_package.py")
test.write_text(test.read_text().replace('release = "v0.1.1"', 'release = "v0.2.0"'))
print("Updated tagged screen and separate CPU diagnostics notebooks")
