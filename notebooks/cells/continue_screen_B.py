import os
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone
from google.colab import userdata

CONFIG = {"lane": "B", "release": "v0.3.0", "root": "/content/captionbert-structured/screen", "data": "/content/captionbert-data"}
required = {"lane", "release", "root", "data"}
if required - CONFIG.keys():
    raise ValueError("Incomplete continuation configuration; restore this complete cell")
python = "/content/captionbert-structured-env/bin/python"
if not pathlib.Path(python).exists():
    raise RuntimeError("Run the original v0.3.0 setup cell first")
root = pathlib.Path(CONFIG["root"])
source = pathlib.Path("/content/captionbert-screen-continuation.py")
url = "https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/d5efc0cc467b3704d3da863d88da436777862447/tools/continue_structured_screen.py"
urllib.request.urlretrieve(url, source)
env = os.environ.copy()
env["HF_TOKEN"] = userdata.get("HF_TOKEN")
env["HF_HOME"] = "/content/captionbert-hf"
env["HF_HUB_CACHE"] = "/content/captionbert-hf/hub"
env["OMP_NUM_THREADS"] = "2"
env["MKL_NUM_THREADS"] = "2"
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONIOENCODING"] = "utf-8"
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
console = root / ("continuation-" + CONFIG["lane"] + "-" + stamp + ".log")
command = [python, str(source), "--lane", CONFIG["lane"], "--root", str(root), "--data", CONFIG["data"]]
print("Continuing registered screen on", CONFIG["lane"], "; preserving the training release and paired data identity.", flush=True)
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
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
subprocess.run(command + ["--ship-only", "--console", str(console)], env=env, check=True)
print("Continuation completed; exact original/continuation consoles and helper source uploaded.", flush=True)
