import os
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone
from google.colab import userdata

CONFIG = {"lane": "A", "root": "/content/captionbert-structured", "data": "/content/captionbert-data", "source": "d7b64092b2156349c25e71678f54848e82bd1eb5"}
root = pathlib.Path(CONFIG["root"])
python = "/content/captionbert-structured-env/bin/python"
script = pathlib.Path("/content/captionbert-prepared-provenance.py")
urllib.request.urlretrieve("https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/" + CONFIG["source"] + "/tools/ship_structured_provenance.py", script)
env = os.environ.copy()
env.update(HF_TOKEN=userdata.get("HF_TOKEN"), HF_HOME="/content/captionbert-hf", HF_HUB_CACHE="/content/captionbert-hf/hub", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
console = root / ("prepared-provenance-console-" + CONFIG["lane"] + "-" + stamp + ".log")
command = [python, str(script), "--root", str(root), "--data", CONFIG["data"], "--lane", CONFIG["lane"], "--push"]
print("CPU-only prepared-data provenance after the registered queues; no optimizer or GPU.", flush=True)
with console.open("a", encoding="utf-8", buffering=1) as log:
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        raise
    finally:
        process.stdout.close()
ship = "import sys; from huggingface_hub import HfApi; HfApi().upload_file(repo_id='AbstractPhil/captionbert-8192-rotary',path_or_fileobj=sys.argv[1],path_in_repo=sys.argv[2],commit_message='Preserve complete prepared-data provenance console')"
subprocess.run([python, "-c", ship, str(console), "campaigns/structured-20261106/operations/data-provenance-" + CONFIG["lane"] + "/" + stamp + "/console.log"], env=env, check=True)
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
print("Prepared-data manifests, exact identities, full-corpus address census and console uploaded.", flush=True)
