import os
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone
from google.colab import userdata

CONFIG = {"lane": "B", "root": "/content/captionbert-structured/screen", "data": "/content/captionbert-data"}
python = "/content/captionbert-structured-env/bin/python"
root = pathlib.Path(CONFIG["root"])
original = {"A": ("R8", "N16", "N16-repeat"), "B": ("H16", "KEND", "KALLP_N0")}[CONFIG["lane"]]
expected = [f"{arm}-seed{seed}" for arm in original for seed in (1, 2)]
expected += ["KALL-seed1", "KALL-seed2", "KALLP-seed1"] if CONFIG["lane"] == "A" else ["KALLP_L8-seed1", "KALLP_L8-seed2", "KALLP-seed2"]
if any(not (root / name / "result.json").is_file() for name in expected):
    raise RuntimeError("Wait for all nine assigned screen models to finish before running these frozen measurements")
source = pathlib.Path("/content/captionbert-frozen-screen.py")
urllib.request.urlretrieve("https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/768bce0a1af71f785ddbcd4bb5a7ab36da39046b/tools/probe_structured_campaign.py", source)
env = os.environ.copy()
env.update(HF_TOKEN=userdata.get("HF_TOKEN"), HF_HOME="/content/captionbert-hf", HF_HUB_CACHE="/content/captionbert-hf/hub", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
console = root / ("frozen-" + CONFIG["lane"] + "-" + stamp + ".log")
command = [python, str(source), "--root", str(root), "--data", CONFIG["data"], "--rows", "4096", "--batch", "8", "--push"]
print("Frozen screen measurements on", CONFIG["lane"], "after optimizer completion; no training.", flush=True)
print("Console:", console, flush=True)
with console.open("a", encoding="utf-8", buffering=1) as log:
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
ship = "import sys; from huggingface_hub import HfApi; HfApi().upload_file(repo_id='AbstractPhil/captionbert-8192-rotary', path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2], commit_message='Preserve frozen screen measurement console')"
remote = "campaigns/structured-20261106/operations/frozen-screen-" + CONFIG["lane"] + "/" + stamp + "/console.log"
subprocess.run([python, "-c", ship, str(console), remote], env=env, check=True)
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
print("Frozen screen reports, per-query arrays, independent AutoModel parity, and console uploaded.", flush=True)
