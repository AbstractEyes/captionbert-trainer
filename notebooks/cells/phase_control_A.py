import os
import json
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone
from google.colab import userdata

CONFIG = {"lane": "A", "source": "24fc4b6368f107d987b9b8894908cfec7dbb3252", "root": "/content/captionbert-structured", "data": "/content/captionbert-data"}
root = pathlib.Path(CONFIG["root"])
seed = {"A": 1, "B": 2}[CONFIG["lane"]]
directory = root / "screen" / f"KALLP-seed{seed}"
result = json.loads((directory / "result.json").read_text())
if result["status"] != "complete" or result["steps"] != 1952 or result["dataset_id"] != "659b21c1c6eb1074eeec7acdbfa9bd010965595a4669338690cf3ae38a4448d5":
    raise RuntimeError("Restore this lane's completed binding checkpoint before the frozen control")
python = "/content/captionbert-structured-env/bin/python"
probe = pathlib.Path("/content/captionbert-isolated-phase-probe.py")
urllib.request.urlretrieve("https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/" + CONFIG["source"] + "/tools/probe_structured_campaign.py", probe)
env = os.environ.copy()
env.update(HF_TOKEN=userdata.get("HF_TOKEN"), HF_HOME="/content/captionbert-hf", HF_HUB_CACHE="/content/captionbert-hf/hub", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
check = "import hashlib,pathlib,captionbert_rotary as c; assert c.__version__=='0.3.0'; assert hashlib.sha256((pathlib.Path(c.__file__).parent/'modeling_captionbert.py').read_bytes()).hexdigest()=='094bf61ef954c61a2b200e5e8760348160b76a5fbfd186602dd992e4b62ccfa6'; print('Original training source verified; no upgrade')"
subprocess.run([python, "-c", check], env=env, check=True)
folder = directory / "frozen-phase-constant-20261106"
if (folder / "report.json").is_file():
    saved = json.loads((folder / "report.json").read_text())
    if saved["probe_source_sha256"] != "1cf1ba909939b1b9f95f100fc87cfe9c21e7698c689658fb125ac522ca2985b9" or saved.get("checkpoint_sha256") != result["checkpoint_sha256"] or saved["rows"] != 4096 or set(saved["measurements"]) != {"intact", "phase_constant"}:
        raise RuntimeError("Preserve this different existing analysis; inspect its source and use a new versioned folder")
    retry = "import sys; from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='AbstractPhil/captionbert-8192-rotary',folder_path=sys.argv[1],path_in_repo=sys.argv[2],commit_message='Preserve compatible completed isolated phase control')"
    command = [python, "-c", retry, str(folder), f"experiments/{result['config_id']}/{directory.name}/{folder.name}"]
else:
    command = [python, str(probe), "--root", str(root / "screen"), "--data", CONFIG["data"], "--arms", "KALLP", "--seeds", str(seed), "--rows", "4096", "--batch", "8", "--sentence-ablations", "--phase-constant", "--output-name", folder.name, "--device", "cuda", "--memory-limit-gb", "92", "--push"]
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
console = root / ("phase-control-" + CONFIG["lane"] + "-" + stamp + ".log")
print("Frozen phase variation only; record norm retained. No optimizer or training.", flush=True)
with console.open("a", encoding="utf-8", buffering=1) as log:
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    except KeyboardInterrupt:
        import signal
        process.send_signal(signal.SIGINT)
        raise
    finally:
        process.stdout.close()
ship = "import sys; from huggingface_hub import HfApi; HfApi().upload_file(repo_id='AbstractPhil/captionbert-8192-rotary',path_or_fileobj=sys.argv[1],path_in_repo=sys.argv[2],commit_message='Preserve complete isolated phase-control console')"
subprocess.run([python, "-c", ship, str(console), "campaigns/structured-20261106/operations/phase-control-" + CONFIG["lane"] + "/" + stamp + "/console.log"], env=env, check=True)
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
print("Isolated phase-control report, queries, TensorBoard scalars and console uploaded.", flush=True)
