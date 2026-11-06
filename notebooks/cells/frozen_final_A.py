import os
import json
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone
from google.colab import userdata

CONFIG = {"lane": "A", "root": "/content/captionbert-structured", "data": "/content/captionbert-data", "source": "59ec95344958120a328f65a0b056632db66ee088"}
root = pathlib.Path(CONFIG["root"])
seed = {"A": 1, "B": 2}[CONFIG["lane"]]
expected = [("full4m", "H16", 4000000), ("full4m", "N16", 4000000), ("exposure1m", "H16", 1000000)]
for stage, arm, rows in expected:
    path = root / stage / f"{arm}-seed{seed}" / "result.json"
    if not path.is_file():
        raise RuntimeError("Finish or resume this lane's three larger models before running the frozen follow-up")
    result = json.loads(path.read_text())
    if result.get("status") != "complete" or result.get("steps") != 7808 or result.get("training_rows") != rows:
        raise RuntimeError("A larger model is incomplete or has a different exposure; inspect and resume its registered training cell")
python = "/content/captionbert-structured-env/bin/python"
source = pathlib.Path("/content/captionbert-final-frozen-probe.py")
urllib.request.urlretrieve("https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/" + CONFIG["source"] + "/tools/probe_structured_campaign.py", source)
env = os.environ.copy()
env.update(HF_TOKEN=userdata.get("HF_TOKEN"), HF_HOME="/content/captionbert-hf", HF_HUB_CACHE="/content/captionbert-hf/hub", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
base = [python, str(source), "--data", CONFIG["data"], "--rows", "4096", "--batch", "8", "--push", "--output-name", "frozen-followup-" + CONFIG["source"][:12]]
jobs = [
    ["--root", str(root / "full4m"), "--arms", "H16", "N16", "--long-shapes", "--structural-ablations", "--sentence-ablations"],
    ["--root", str(root / "exposure1m"), "--arms", "H16", "--long-shapes", "--structural-ablations", "--sentence-ablations"],
    ["--root", str(root / "screen"), "--arms", "KALLP", "--ablations", "--sentence-ablations", "--initialization-control"],
]
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
console = root / ("frozen-final-" + CONFIG["lane"] + "-" + stamp + ".log")
print("Frozen follow-up after all optimizer work; preserving original intact reports.", flush=True)
print("Console:", console, flush=True)
returncode = 0
with console.open("a", encoding="utf-8", buffering=1) as log:
    for options in jobs:
        command = base + options
        print("Frozen command:", command, flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
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
        if returncode:
            break
ship = "import sys; from huggingface_hub import HfApi; HfApi().upload_file(repo_id='AbstractPhil/captionbert-8192-rotary', path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2], commit_message='Preserve complete final frozen follow-up console')"
remote = "campaigns/structured-20261106/operations/frozen-final-" + CONFIG["lane"] + "/" + stamp + "/console.log"
subprocess.run([python, "-c", ship, str(console), remote], env=env, check=True)
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
print("Final frozen reports, query arrays, TensorBoard scalars, long shapes, and console uploaded.", flush=True)
