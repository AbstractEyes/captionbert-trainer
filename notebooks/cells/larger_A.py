import os
import hashlib
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone
from google.colab import userdata

CONFIG = {"lane": "A", "root": "/content/captionbert-structured", "data": "/content/captionbert-data", "selection_url": "https://huggingface.co/AbstractPhil/captionbert-8192-rotary/resolve/becdda8590d138a92e947976005357fd6ccb9de2/campaigns/structured-20261106/locked-screen-selection.json"}
root = pathlib.Path(CONFIG["root"])
python = "/content/captionbert-structured-env/bin/python"
selection = pathlib.Path("/content/captionbert-locked-selection.json")
source = pathlib.Path("/content/captionbert-final-stage.py")
urllib.request.urlretrieve(CONFIG["selection_url"], selection)
if hashlib.sha256(selection.read_bytes()).hexdigest() != "bdbf782e86b29545ec4a0731a8669b69ed5794f82914bebd1236fab007820338":
    raise RuntimeError("Locked selection hash differs; restore the complete published cell")
urllib.request.urlretrieve("https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/768bce0a1af71f785ddbcd4bb5a7ab36da39046b/tools/run_structured_final_stage.py", source)
env = os.environ.copy()
env.update(HF_TOKEN=userdata.get("HF_TOKEN"), HF_HOME="/content/captionbert-hf", HF_HUB_CACHE="/content/captionbert-hf/hub", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
console = root / ("larger-" + CONFIG["lane"] + "-" + stamp + ".log")
command = [python, str(source), "--lane", CONFIG["lane"], "--selection", str(selection), "--root", str(root), "--data", CONFIG["data"]]
print("Starting the locked larger experiment on lane", CONFIG["lane"], "; original training release preserved.", flush=True)
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
print("Larger lane exit:", returncode, "; preserving its complete console.", flush=True)
subprocess.run(command + ["--ship-only", "--console", str(console)], env=env, check=True)
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
print("Larger lane complete: all three seed-owned models and logs uploaded.", flush=True)
