import os
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone
from google.colab import userdata

CONFIG = {"root": "/content/captionbert-serving-validation", "data": "/content/captionbert-data", "verifier_source": "d0df6204ce21f012f3fc8e89c515177277020bdf", "inference_source": "caeb0e24e0f68bd6e3a7090b2380af5717dab572"}
root = pathlib.Path(CONFIG["root"])
root.mkdir(exist_ok=True)
python = "/content/captionbert-structured-env/bin/python"
env = os.environ.copy()
env.update(HF_TOKEN=userdata.get("HF_TOKEN"), HF_HOME="/content/captionbert-hf", HF_HUB_CACHE="/content/captionbert-hf/hub", HF_MODULES_CACHE=str(root / "modules"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
worker = root / ("verify-binding-" + stamp + ".py")
code = r'''import json,shutil,subprocess,sys,urllib.request
from pathlib import Path
from captionbert_rotary.structured_campaign import CampaignConfig
from captionbert_rotary.hub import pull_completed
import captionbert_rotary
root=Path(sys.argv[1]);data=sys.argv[2];verifier_revision=sys.argv[3];inference_revision=sys.argv[4]
cfg=CampaignConfig(stage="screen",arms=("KALLP",),seeds=(1,),output_dir=str(root/"screen"),data_dir=data)
if cfg.fingerprint()!="dc37215529c188be7b93804c4a21f125afa353e037b1a2a8fc11e5b2fc817d3a":
 raise RuntimeError("Preserve the original v0.3.0 setup for this independent reference")
pull_completed(cfg,"659b21c1c6eb1074eeec7acdbfa9bd010965595a4669338690cf3ae38a4448d5")
source=root/"screen/KALLP-seed1"
if not (source/"result.json").is_file():raise RuntimeError("Restore the completed binding checkpoint before verification")
serving=root/"serving-v031";serving.mkdir(exist_ok=True)
for name in ("model.safetensors","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt"):
 if (source/name).is_file():shutil.copy2(source/name,serving/name)
for name in ("modeling_captionbert.py","configuration_captionbert.py"):
 urllib.request.urlretrieve("https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/"+inference_revision+"/captionbert_rotary/"+name,serving/name)
verifier=root/("verify-"+verifier_revision[:12]+".py")
urllib.request.urlretrieve("https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/"+verifier_revision+"/tools/verify_structured_deployment.py",verifier)
reference=Path(captionbert_rotary.__file__).resolve().parent.parent
report=root/"trained-binding-colab-parity.json"
subprocess.run([sys.executable,"-I",str(verifier),"--source",str(source),"--training-source",str(reference),"--target",str(serving),"--cache",str(root/"auto-cache/hub"),"--output",str(report),"--device","cuda","--memory-limit-gb","92","--amp","--long-shapes"],check=True)
from huggingface_hub import HfApi
api=HfApi()
prefix="campaigns/structured-20261106/validation/binding-serving-colab/"+sys.argv[5]
for name in ("config.json","modeling_captionbert.py","configuration_captionbert.py"):
 api.upload_file(repo_id=cfg.hub_repo,path_or_fileobj=str(serving/name),path_in_repo=prefix+"/"+name,commit_message="Preserve trained-binding serving source for latest Colab parity")
result=json.loads(report.read_text())
import transformers,safetensors,huggingface_hub
result["environment"].update(transformers=transformers.__version__,safetensors=safetensors.__version__,huggingface_hub=huggingface_hub.__version__)
result.update(inference_source_commit=inference_revision,verifier_source_commit=verifier_revision,source_experiment="experiments/"+cfg.fingerprint()+"/KALLP-seed1",serving_weight_bytes_unchanged_from_source=True)
report.write_text(json.dumps(result,indent=2)+"\n")
api.upload_file(repo_id=cfg.hub_repo,path_or_fileobj=str(report),path_in_repo=prefix+"/report.json",commit_message="Verify complete trained binding AutoModel on latest Colab with safe weights")
print("TRAINED BINDING COLAB SERVING CHECK COMPLETE; no optimizer or accuracy training",flush=True)
'''
worker.write_text(code, encoding="utf-8")
console = root / ("console-" + stamp + ".log")
command = [python, str(worker), str(root), CONFIG["data"], CONFIG["verifier_source"], CONFIG["inference_source"], stamp]
print("Serving engineering only: one completed binding checkpoint, latest Colab libraries, no optimizer.", flush=True)
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
ship = "import sys; from huggingface_hub import HfApi; HfApi().upload_file(repo_id='AbstractPhil/captionbert-8192-rotary',path_or_fileobj=sys.argv[1],path_in_repo=sys.argv[2],commit_message='Preserve complete Colab serving verification console')"
subprocess.run([python, "-c", ship, str(console), "campaigns/structured-20261106/validation/binding-serving-colab/" + stamp + "/console.log"], env=env, check=True)
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
print("Serving check and console uploaded.", flush=True)
