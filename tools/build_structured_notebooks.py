"""Build self-contained Colab lanes from reviewed Python cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "v0.3.0"
RELAY = (ROOT / "captionbert_rotary/notebook.py").read_text(encoding="utf-8")

SETUP = '''import os
import pathlib
import subprocess
import sys
import urllib.request

CONFIG = {"release": "RELEASE", "lane": LANE, "env": "/content/captionbert-structured-env"}
required = {"release", "lane", "env"}
if required - CONFIG.keys():
    raise ValueError("Incomplete setup configuration; restore this tagged notebook and rerun setup")
if not (3, 11) <= sys.version_info[:2] <= (3, 13):
    raise RuntimeError("Choose Colab Python3.11-3.13, then rerun this setup cell")
os.environ["HF_HOME"] = "/content/captionbert-hf"
os.environ["HF_HUB_CACHE"] = "/content/captionbert-hf/hub"
env_dir = pathlib.Path(CONFIG["env"])
python = str(env_dir / "bin/python")
if not pathlib.Path(python).exists():
    bootstrap = pathlib.Path("/content/captionbert-bootstrap")
    subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(bootstrap), "virtualenv>=20.38.0"], check=True)
    bootstrap_env = os.environ.copy()
    bootstrap_env["PYTHONPATH"] = str(bootstrap)
    subprocess.run([sys.executable, "-m", "virtualenv", "--no-download", str(env_dir)], env=bootstrap_env, check=True)
req_dir = pathlib.Path("/content/captionbert-structured-install")
req_dir.mkdir(exist_ok=True)
for name in ("requirements.txt", "requirements-cuda.txt", "requirements-colab.txt"):
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/{CONFIG['release']}/{name}", req_dir / name)
subprocess.run([python, "-m", "pip", "install", "-r", str(req_dir / "requirements-colab.txt")], check=True)
subprocess.run([python, "-m", "pip", "install", "--no-deps", f"git+https://github.com/AbstractEyes/captionbert-trainer.git@{CONFIG['release']}"], check=True)
subprocess.run([python, "-m", "pip", "check"], check=True)
subprocess.run([python, "-c", "import torch,numpy,scipy,transformers,datasets,safetensors,captionbert_rotary; from torch.utils.tensorboard import SummaryWriter; from captionbert_rotary.modeling_captionbert import CaptionBertModel; print('READY',captionbert_rotary.__version__,torch.__version__,torch.version.cuda); assert torch.cuda.is_available(), 'Select G4 via the arrow beside Connect -> Change runtime type'; print(torch.cuda.get_device_name(0)); assert '6000' in torch.cuda.get_device_name(0), 'This campaign requires the selected G4 / RTX PRO6000'; print('GPU GB',torch.cuda.get_device_properties(0).total_memory/1e9)"], check=True)
from google.colab import userdata
try:
    token = userdata.get("HF_TOKEN")
except Exception as error:
    raise RuntimeError("Enable HF_TOKEN notebook access in the Secrets sidebar, then rerun setup") from error
os.environ["HF_TOKEN"] = token
subprocess.run([python, "-c", "from huggingface_hub import HfApi; print('Hub access:',HfApi().whoami()['name'])"], check=True)
print("Setup ready; next run the full-batch engineering gate, TensorBoard, then this lane's training cell.", flush=True)
'''

PREFLIGHT = '''import os
import pathlib
python = "/content/captionbert-structured-env/bin/python"
root = pathlib.Path("/content/captionbert-structured/screen")
root.mkdir(parents=True, exist_ok=True)
if not pathlib.Path(python).exists():
    raise RuntimeError("Run setup cell first; the isolated Python environment is missing")
RELAY
run_logged([python, "-m", "captionbert_rotary.structured_preflight", "--full", "--output", str(root / "preflight.json")], root / "preflight-console.log")
print("Full-width batch2048 x256 gate passed. Next start TensorBoard, then training.", flush=True)
'''

TENSORBOARD = '''import os
import pathlib
import subprocess
from google.colab import output

root = pathlib.Path("/content/captionbert-structured")
root.mkdir(parents=True, exist_ok=True)
pid_path = root / "tensorboard.pid"
alive = False
if pid_path.exists():
    try:
        os.kill(int(pid_path.read_text()), 0)
        alive = True
    except (OSError, ValueError):
        pass
if not alive:
    log = (root / "tensorboard-console.log").open("a")
    process = subprocess.Popen(["/content/captionbert-structured-env/bin/python", "-m", "tensorboard.main", "--logdir", str(root), "--host", "127.0.0.1", "--port", "6006"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(process.pid))
print("TensorBoard: each arm/seed has its own run. Scalar data appear as training starts.")
output.serve_kernel_port_as_iframe(6006, height=420)
'''

TRAIN = '''import os
import pathlib
from google.colab import userdata

RUN = {"stage": "screen", "lane": LANE, "data": "/content/captionbert-data", "output": "/content/captionbert-structured/screen"}
required = {"stage", "lane", "data", "output"}
if required - RUN.keys():
    raise ValueError("Incomplete run configuration; restore this tagged cell")
python = "/content/captionbert-structured-env/bin/python"
if not pathlib.Path(python).exists():
    raise RuntimeError("Run setup, then the engineering gate, before training")
gate = pathlib.Path(RUN["output"]) / "preflight.json"
if not gate.exists():
    raise RuntimeError("Run the full-batch engineering gate cell before training")
import json
if not json.loads(gate.read_text()).get("passed"):
    raise RuntimeError("Engineering gate failed; inspect its output before starting training")
env = os.environ.copy()
env["HF_TOKEN"] = userdata.get("HF_TOKEN")
env["HF_HOME"] = "/content/captionbert-hf"
env["HF_HUB_CACHE"] = "/content/captionbert-hf/hub"
env["OMP_NUM_THREADS"] = "2"
env["MKL_NUM_THREADS"] = "2"
RELAY
print("Starting lane", RUN["lane"] + 1, "with two seeds per registered option. Check the step display and TensorBoard.")
run_logged([python, "-m", "captionbert_rotary.structured_campaign", "--stage", RUN["stage"], "--lane", str(RUN["lane"]), "--data", RUN["data"], "--output", RUN["output"]], pathlib.Path(RUN["output"]) / "console.log", env=env)
'''

MONITOR = '''import pathlib
import subprocess
python = "/content/captionbert-structured-env/bin/python"
code = """import json,pathlib
root=pathlib.Path('/content/captionbert-structured/screen')
for directory in sorted(p for p in root.iterdir() if p.is_dir() and '-seed' in p.name):
    result=directory/'result.json'
    if result.exists():
        value=json.loads(result.read_text()); print(directory.name, 'COMPLETE', value['steps'], 'devSTS', value['development']['spearman'], 'cos', value['evaluation']['heldout']['cos_to_target'])
    elif (directory/'train.jsonl').exists():
        last=json.loads((directory/'train.jsonl').read_text().splitlines()[-1]); print(directory.name, 'step',last['step'],'loss',last['loss'],'lr',last['lr'])
print('TensorBoard directories:',len(list(root.glob('*/tensorboard'))))
"""
subprocess.run([python, "-c", code], check=True)
'''


def cell(kind, text, cell_id):
    value = {"cell_type": kind, "metadata": {"id": cell_id}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build():
    for lane in range(3):
        markdown = f"""# CaptionBERT structured rotary — lane {lane + 1}

Use **G4 / RTX PRO6000** via the arrow right of Connect → Change runtime type.
Enable **HF_TOKEN** notebook access. Run setup, the full-batch gate, TensorBoard,
then training. This tagged **{RELEASE}** notebook preserves Colab's host packages.

The [registered protocol](https://github.com/AbstractEyes/captionbert-trainer/blob/{RELEASE}/docs/structured_campaign_protocol.md)
defines the eight options, paired seeds, repeat, development-only selection and
4M-caption promotion. Each lane runs six models. Logs, safetensors checkpoints,
optimizer/RNG state and TensorBoard events ship to
[the model repo](https://huggingface.co/AbstractPhil/captionbert-8192-rotary).

After an interrupted runtime, repeat setup/gate/TensorBoard/training with the
same tag and lane. Compatible remote checkpoints resume; completed runs are verified
and skipped. The final model format is `model.safetensors` with AutoModel support.
No 8K semantic-quality claim follows from the position limit.
"""
        rendered = []
        for number, (kind, source) in enumerate((("markdown", markdown), ("code", SETUP), ("code", PREFLIGHT), ("code", TENSORBOARD), ("code", TRAIN), ("code", MONITOR))):
            source = source.replace("RELEASE", RELEASE).replace("LANE", str(lane)).replace("RELAY", RELAY)
            if kind == "code":
                compile(source, f"lane{lane + 1}-cell{number}", "exec")
            rendered.append(cell(kind, source, f"structured-lane{lane + 1}-{number}"))
        notebook = dict(nbformat=4, nbformat_minor=5, metadata={"accelerator": "GPU", "colab": {"name": f"CaptionBERT structured rotary lane {lane + 1}"}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, cells=rendered)
        path = ROOT / f"notebooks/captionbert_structured_lane{lane + 1}.ipynb"
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    build()
