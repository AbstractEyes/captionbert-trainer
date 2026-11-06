"""Run only after lane A completes; verifies the locked remote default, no training."""
import os
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone
from google.colab import userdata

CONFIG = {
    "root": "/content/captionbert-deployment-validation",
    "source": "/content/captionbert-structured/full4m/N16-seed1",
    "repo": "AbstractPhil/captionbert-8192-rotary",
    "revision": "ad14b77f4cc9017d513173f3b340e735af02ef14",
    "verifier_source": "192e29ec49cf70d2ef8e9eb3430b70eb79c6cc20",
}
root = pathlib.Path(CONFIG["root"])
root.mkdir(exist_ok=True)
python = "/content/captionbert-structured-env/bin/python"
env = os.environ.copy()
env.update(HF_TOKEN=userdata.get("HF_TOKEN"), HF_HOME="/content/captionbert-hf",
           HF_HUB_CACHE="/content/captionbert-hf/hub", OMP_NUM_THREADS="2",
           MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
worker = root / ("verify-default-" + stamp + ".py")
code = r'''import hashlib,json,subprocess,sys,urllib.request
from pathlib import Path
import captionbert_rotary
root=Path(sys.argv[1]); source=Path(sys.argv[2]);repo=sys.argv[3];revision=sys.argv[4];verifier_revision=sys.argv[5];stamp=sys.argv[6]
result=json.loads((source/"result.json").read_text())
if any(result.get(k)!=v for k,v in dict(arm="N16",seed=1,status="complete",steps=7808,training_rows=4000000,config_id="aafe9c59b9d696107cb4393f3a26815e84714daddb7d6eb028b5d6137b88418e",dataset_id="7adb10f8833fab44e3ebb44315b53cb65bd8010b038115d15350b18536b57e86").items()):
 raise RuntimeError("Restore the locked completed full4m/N16-seed1 checkpoint")
verifier=root/("verify-"+verifier_revision[:12]+".py")
urllib.request.urlretrieve("https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/"+verifier_revision+"/tools/verify_structured_deployment.py",verifier)
if hashlib.sha256(verifier.read_bytes()).hexdigest()!=sys.argv[7]:
 raise RuntimeError("Verifier source hash differs; preserve the reviewed cell")
reference=Path(captionbert_rotary.__file__).resolve().parent.parent
report=root/("default-remote-parity-"+stamp+".json")
subprocess.run([sys.executable,"-I",str(verifier),"--source",str(source),"--training-source",str(reference),"--target",repo,"--revision",revision,"--cache",str(root/"auto-cache/hub"),"--output",str(report),"--device","cuda","--memory-limit-gb","92","--amp","--long-shapes"],check=True)
from huggingface_hub import HfApi
import huggingface_hub
saved=json.loads(report.read_text())
saved.update(verifier_source_commit=verifier_revision,source_experiment="experiments/"+result["config_id"]+"/N16-seed1")
saved["environment"]["huggingface_hub"]=huggingface_hub.__version__
report.write_text(json.dumps(saved,indent=2)+"\n")
prefix="campaigns/structured-20261106/validation/deployment-colab/"+stamp
HfApi().upload_file(repo_id=repo,path_or_fileobj=str(report),path_in_repo=prefix+"/report.json",commit_message="Verify locked remote safe AutoModel against independent trained state on latest Colab")
print("SELECTED REMOTE AUTOMODEL COLAB CHECK COMPLETE; no optimizer or training",flush=True)
'''
worker.write_text(code, encoding="utf8")
verifier_hash = "c26fc8a70dd97b589693f0f5c1e38e5b8a0164d6fa07c9d6a1c6da9c0d384521"
command = [python, str(worker), str(root), CONFIG["source"], CONFIG["repo"],
           CONFIG["revision"], CONFIG["verifier_source"], stamp, verifier_hash]
console = root / ("console-" + stamp + ".log")
print("Selected remote AutoModel engineering; immutable Hub revision", CONFIG["revision"], flush=True)
with console.open("a", encoding="utf8", buffering=1) as log:
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf8", errors="replace", bufsize=1)
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
ship = "import sys; from huggingface_hub import HfApi; HfApi().upload_file(repo_id='AbstractPhil/captionbert-8192-rotary',path_or_fileobj=sys.argv[1],path_in_repo=sys.argv[2],commit_message='Preserve complete selected remote AutoModel Colab console')"
subprocess.run([python, "-c", ship, str(console),
               "campaigns/structured-20261106/validation/deployment-colab/" + stamp + "/console.log"],
              env=env, check=True)
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
print("Selected remote AutoModel report and console uploaded.", flush=True)
