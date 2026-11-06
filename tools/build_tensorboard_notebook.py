"""Build a CPU-only TensorBoard viewer for the completed campaign."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''import os
import pathlib
import subprocess
import sys

CONFIG = {"env": "/content/captionbert-tensorboard-env"}
env_dir = pathlib.Path(CONFIG["env"])
python = str(env_dir / "bin/python")
if not pathlib.Path(python).exists():
    bootstrap = pathlib.Path("/content/captionbert-tensorboard-bootstrap")
    subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(bootstrap), "virtualenv>=20.38.0"], check=True)
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(bootstrap)
    subprocess.run([sys.executable, "-m", "virtualenv", "--no-download", str(env_dir)], env=child_env, check=True)
subprocess.run([python, "-m", "pip", "install", *VIEWER_REQUIREMENTS], check=True)
subprocess.run([python, "-m", "pip", "check"], check=True)
subprocess.run([python, "-c", "import tensorboard,huggingface_hub; print('READY',tensorboard.__version__,huggingface_hub.__version__)"], check=True)
print("Ready. This viewer needs CPU only; it downloads events, not model weights or training data.")
'''
FETCH = r'''import os
import pathlib
import subprocess

python = "/content/captionbert-tensorboard-env/bin/python"
root = pathlib.Path("/content/captionbert-tensorboard")
root.mkdir(exist_ok=True)
code = r"""import hashlib,json,shutil,sys
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
root=Path(sys.argv[1]);repo="AbstractPhil/captionbert-8192-rotary"
api=HfApi();head=api.model_info(repo).sha
path=hf_hub_download(repo,"campaigns/structured-20261106/artifact-census.json",revision=head,cache_dir=str(root/"hub"))
census=json.loads(Path(path).read_text())
if census["status"]!="complete" or census["trained_models"]!=24:
 raise RuntimeError("The complete campaign census is not yet published; wait for release, then rerun this cell")
revision=census["source_revision"];count=0;total=0
for key,run in census["runs"].items():
 print("Restoring TensorBoard:",key,flush=True)
 for name,meta in run["files"].items():
  if "events.out.tfevents." not in name:continue
  remote=run["prefix"]+name
  cached=hf_hub_download(repo,remote,revision=revision,cache_dir=str(root/"hub"))
  source=Path(cached)
  raw=source.read_bytes()
  digest=hashlib.sha256(raw).hexdigest() if meta["lfs_sha256"] else hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
  if len(raw)!=meta["bytes"] or digest!=(meta["lfs_sha256"] or meta["git_blob_sha1"]):
   raise RuntimeError("An event file differs from the pinned census")
  destination=root/"logs"/key/name
  destination.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(source,destination);count+=1;total+=source.stat().st_size
print("Event files:",count,"; bytes:",total,"; all24 models and available frozen endpoints")
print("Pinned artifact revision:",revision)
(root/"source.json").write_text(json.dumps(dict(repo=repo,revision=revision,event_files=count,bytes=total),indent=2)+"\n")
"""
env = os.environ.copy()
env.update(HF_HOME=str(root / "hf"), HF_HUB_CACHE=str(root / "hf/hub"), OMP_NUM_THREADS="2")
subprocess.run([python, "-u", "-c", code, str(root)], env=env, check=True)
print("Next run the display cell. Repeating this cell reuses cached files.")
'''
DISPLAY = '''import os
import pathlib
import subprocess
from google.colab import output

root = pathlib.Path("/content/captionbert-tensorboard")
python = "/content/captionbert-tensorboard-env/bin/python"
if not (root / "source.json").is_file():
    raise RuntimeError("Run setup and the event-download cell before starting TensorBoard")
pid_path = root / "tensorboard.pid"
alive = False
if pid_path.is_file():
    try:
        os.kill(int(pid_path.read_text()), 0)
        alive = True
    except (OSError, ValueError):
        pass
if not alive:
    log = (root / "server.log").open("a")
    process = subprocess.Popen([python, "-m", "tensorboard.main", "--logdir", str(root / "logs"), "--host", "127.0.0.1", "--port", "6006"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(process.pid))
print("Filter by screen/, full4m/, exposure1m/, or frozen-followup.")
print("train/loss, train/lr, train/step, final/stsb_development/spearman and frozen/ metrics are available.")
output.serve_kernel_port_as_iframe(6006, height=650)
'''


def main():
    intro = """# CaptionBERT campaign TensorBoard — CPU viewer

Select a **CPU runtime**. Run setup, download the saved events, then display.
This notebook installs a small isolated viewer and downloads only event files
for the completed24-model structured campaign. It requires no GPU, training
dataset, model checkpoint, or Hugging Face secret for this public repository.

The completed artifact census pins the event revision. Training and frozen
endpoint directories retain readable stage/arm/seed names. Use the run filter
and scalar search to compare losses, learning rates, progress, development
scores and mechanism measurements. The setup leaves Colab host packages intact.

[Model and report](https://huggingface.co/AbstractPhil/captionbert-8192-rotary)
"""
    cells = []
    requirements = [line for line in (ROOT / "requirements-tensorboard.txt").read_text().splitlines() if line and not line.startswith("#")]
    setup = SETUP.replace("VIEWER_REQUIREMENTS", repr(requirements))
    for index, (kind, text) in enumerate([("markdown", intro), ("code", setup), ("code", FETCH), ("code", DISPLAY)]):
        if kind == "code":
            compile(text, f"tensorboard-viewer-{index}", "exec")
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "code" for t in node.targets):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        compile(node.value.value, f"tensorboard-viewer-worker-{index}", "exec")
        cell = dict(cell_type=kind, metadata={"id": f"captionbert-tensorboard-{index}"},
                    source=text.splitlines(keepends=True))
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
        metadata={"colab":{"name":"CaptionBERT TensorBoard — CPU viewer"},
                  "kernelspec":{"display_name":"Python3","language":"python","name":"python3"}})
    path = ROOT / "notebooks/captionbert_tensorboard_viewer.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf8", newline="\n")
    print(path)


if __name__ == "__main__":
    main()
