# CaptionBERT rotary trainer

The completed structured-rotary campaign compares eight full CaptionBERT
options under two seeds, a repeated relative reference and six larger models.
The selected four-million-row sentence release is **Relative 16 heads (N16),
seed 1**, following the development-only rule. The two larger
architecture pair means are a near tie. Every trained option remains available.

[Plain-English analysis](docs/STRUCTURED_REPORT.md) ·
[Technical companion](docs/TECHNICAL_STRUCTURED.md) ·
[Model and card](https://huggingface.co/AbstractPhil/captionbert-8192-rotary) ·
[Artifact census](docs/results/structured_artifact_census.json)

## Colab and recovery

Use G4 / RTX PRO6000, enable notebook access to HF_TOKEN, then run setup,
the full-batch gate and TensorBoard.

- [Larger seed 1 / recovery](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/campaign-structured-20261106/notebooks/captionbert_structured_followup_seed1.ipynb)
- [Larger seed 2 / recovery](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/campaign-structured-20261106/notebooks/captionbert_structured_followup_seed2.ipynb)
- [Saved TensorBoard logs / CPU viewer](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/campaign-structured-20261106/notebooks/captionbert_tensorboard_viewer.ipynb)

Each larger notebook owns one seed:4M hierarchical,4M relative, then the
1M matched-exposure control. Partial compatible state resumes; verified
completed runs are skipped. The frozen cell also restores the earlier binding
checkpoint after a fresh runtime. Do not start duplicate notebooks for a seed.

These campaign notebooks keep the **v0.3.0 training source**. Helper/probe
revisions are explicit in their cells. The package's **v0.3.1 serving source**
includes the optional FP16 diagnostic-mask correction and is checked against
independent v0.3.0 final weights. Do not upgrade an active training environment.

Colab dependencies use floors in an isolated environment. The actual campaign
ran on Python3.13.16, torch2.15.0+cu132, Transformers5.22.1 and Safetensors0.8.0.
Setup verifies imports and hardware and leaves host packages intact.
The CPU viewer installs only TensorBoard/Hub dependencies and downloads events,
without model weights, consensus targets, training or GPU use.

## Load the released model

~~~python
import torch
from transformers import AutoModel, AutoTokenizer

repo = "AbstractPhil/captionbert-8192-rotary"
revision = "ad14b77f4cc9017d513173f3b340e735af02ef14"
tok = AutoTokenizer.from_pretrained(repo, revision=revision)
model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True, use_safetensors=True).eval()
inputs = tok(["A cat watches the street.", "A feline looks outside."],
             padding=True, truncation=True, max_length=256, return_tensors="pt")
with torch.inference_mode():
    output = model(**inputs)
embeddings = output.pooler_output       # [B,768]
token_states = output.last_hidden_state # [B,L,512]
~~~

The root uses model.safetensors, safe_serialization=True and format:pt metadata.
The complete config, tokenizer and AutoModel source are included. Nonempty
sentence vectors have unit norm; all-padding sentence vectors are zero.
Right padding matches training.8192 is a checked execution limit; training
stops at 256 tokens and sentence tests at 64.

## Local installation and checks

Python3.11–3.13. The local reference environment is Python3.12.10,
torch2.11.0+cu128 and Transformers4.57.6; it also passes the trained serving checks.

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -m "not cuda"
~~~

PyTorch's working CUDA wheel supplies the runtime; a separate CUDA toolkit was
unnecessary for this campaign. Local engineering uses one 4090 writer, a process
fraction at most0.73 and a total-device ceiling below 20 decimal GB including other
applications. Model/data caches stay on E:. Accuracy training ran on G4.

## Progress, TensorBoard and continuation

The structured console shows arm/seed, step/total, epoch, loss components,
pointer loss, LR, seconds/step and ETA, with preparation, evaluation, save and
upload stage messages. JSONL retains every training step, including gradient
norm and rows/second. TensorBoard adds all loss components, progress, validation,
memory samples and final task scores; frozen follow-ups add mechanism endpoints.

Final, best and midpoint safetensors, independent model/Adam/scheduler/RNG/cursor
state, exact environments, predictions, logs and event files ship for all 24
models. Checkpoint transport is throttled to ten minutes; a reconnect can repeat
work after the last durable boundary. state.pt is authoritative; resume.json
may describe an earlier transport snapshot.

[Event verification](docs/results/structured_tensorboard_verification.json)
checks actual saved event hashes, every training step, final development values
and frozen endpoints. The viewer reads the pinned completed artifact census,
so plots remain available after GPU runtimes end.

## Reproduce analysis

The [registered protocol](docs/structured_campaign_protocol.md),
[screen rules](docs/structured_analysis_rules.md) and
[final decision rule](docs/structured_final_analysis_rules.md) retain the
hypotheses, exposure controls, development selection and scope of conclusions.

tools/collect_structured_campaign.py and tools/collect_structured_final.py
collect small reports. tools/analyze_structured_final.py locks the default using
development values before test tables are built. tools/build_structured_report.py
and the plotting tools generate measured tables/figures.
tools/verify_structured_artifacts.py and tools/verify_structured_tensorboard.py
check the durable evidence. tools/export_structured_deployment.py and
tools/verify_structured_deployment.py perform independent safe export/parity.

The original three structured screen notebooks remain pinned to v0.3.0:
[1](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.3.0/notebooks/captionbert_structured_lane1.ipynb),
[2](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.3.0/notebooks/captionbert_structured_lane2.ipynb),
[3](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.3.0/notebooks/captionbert_structured_lane3.ipynb).
They reproduce the screen; the two larger recovery notebooks resume the locked
follow-up directly.

## Earlier positional screen

The v0.2.0 A0–A3 notebook and its rotary diagnostics remain a separate historical
experiment. Its2000-step warmup covered nearly its entire short training run.
Those results are preserved, and are not a test of the later complete whole-key
mechanism. Its legacy model-card finalizer describes those old experiments;
the current root card is the structured campaign release above.

See[THIRD_PARTY.md](THIRD_PARTY.md) for upstream attribution. MIT.
