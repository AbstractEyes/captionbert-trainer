# CaptionBERT rotary trainer

The **v0.3.0 structured campaign** adapts the final whole-key/every-position
rotary mechanism to the full CaptionBERT encoder. Its [registered protocol](docs/structured_campaign_protocol.md)
defines eight options, two seeds each, a hardware repeat, development-only
selection, and matched-exposure experiments at four million caption rows.
Scientific results are pending; the earlier screen below remains unchanged.

Colab lanes: [1](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.3.0/notebooks/captionbert_structured_lane1.ipynb),
[2](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.3.0/notebooks/captionbert_structured_lane2.ipynb),
[3](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.3.0/notebooks/captionbert_structured_lane3.ipynb).
Each includes TensorBoard, a full-batch GPU gate, streaming progress, durable
checkpoints and remote continuation. Releases use `model.safetensors` and
self-contained AutoConfig/AutoModel code; paired training-state replay and
AutoModel round-trip tests are part of the suite.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.2.0/notebooks/captionbert_rotary_screen.ipynb)

Controlled stage-3 screen from `captionbert_8192_plan_doc.md`: learned absolute
positions (A0), plain token RoPE (A1), sentence-address rotary (A2), its anchor
ablation (A3), and optional global-slot A2g. Mean pooling, symmetric InfoNCE at
0.07 plus MSE, pure Adam without weight decay, two seeds, and identical
deterministic row plans within each seed.

## Install and verify locally

Python 3.11–3.13. The local development environment uses Python 3.12 and the
CUDA 12.8 PyTorch wheel. A separate system CUDA toolkit is unnecessary.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m captionbert_rotary smoke --batch-size 8 --steps 2
```

`smoke` runs the **full 12-layer model** for every arm and both seeds on synthetic
256-token inputs. It writes `runs/smoke.json`; these are engineering checks, not
benchmark findings. `--tiny --device cpu` provides a quick CPU diagnostic.
The tiny model is not an experimental arm.

## Colab

Open the notebook above on a G4 / RTX PRO 6000 with 96 GB. Its first code cell
creates an isolated Python environment and installs the **tagged GitHub release**.
Each subsequent cell is self-contained and invokes that environment directly.
This supports Colab's current Python 3.13 without downgrading the host kernel's
PyTorch, NumPy, Transformers, Hugging Face Hub, or other preinstalled libraries.
No runtime restart, argparse, or `__file__` is used by notebook cells.

Child stdout and stderr are streamed through the notebook kernel immediately,
with unbuffered Python output and a saved console copy for each cell in
`/content/captionbert-logs/`. Training shows the first step and every 25th step:
arm/seed, epoch/batch, step/total and percentage, total loss, InfoNCE, MSE, actual
step learning rate, gradient norm, seconds/step, rows/second, elapsed time,
remaining training compute time, and CUDA allocated/reserved memory. Validation,
checkpoint writes, eight-task benchmarks, and upload attempts/retries print their
own stage updates. The JSONL record still includes **every** training step.

For a v0.1.0 job already training, keep it running and open a **separate Colab
terminal**. This standard-library monitor reads that job's existing logs without
upgrading its environment, allocating GPU memory, or changing its files:

```bash
curl -fLsS https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/v0.1.1/tools/watch_run.py -o /content/captionbert-watch.py
python -u /content/captionbert-watch.py /content/captionbert-screen
```

It displays losses, LR, gradient norm, step counts, speed, estimated remaining
training compute, saved validation metrics, checkpoints, and upload queue counts.
Its ETA excludes validation, checkpoint, and upload overhead. A heartbeat indicates
when no new step has been logged; it does not diagnose the training process.
Ctrl+C stops only the monitor. Another notebook cell queues behind a busy training
cell, so use the separate terminal for this existing session.

Compatibility checked against the [Colab runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html)
and [published GPU package inventory](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt)
on 2026-11-05. The inventory can lead production by several days. Training uses
our isolated, pinned stack (`requirements-colab.txt` -> `requirements-cuda.txt`
-> `requirements.txt`), not the changing host versions. The Linux CPython 3.13
wheel set is checked separately from the local Windows GPU tests.

Add a Colab secret named `HF_TOKEN` with write permission to
`AbstractPhil/captionbert-8192-rotary`. The training cell reads it into the child
process environment; it is never printed or saved to result files. Every completed
arm/seed ships from an immutable local snapshot with retries. Failed uploads stay
in a durable queue, retry at startup/session close, and are listed in
`upload_status.json` and the final summary. A Hub outage does not invalidate a
completed local run or raise from its background upload worker.

Local checkpoints preserve optimizer, scheduler, scaler, all RNG states, epoch and
batch cursor. The first saved training checkpoint also ships in the background;
subsequent remote resume snapshots are throttled to ten minutes. A fresh runtime
fetches completed **and partial** runs at a pinned Hub revision, verifies their
configuration/data identity, and keeps a newer surviving local checkpoint. It
continues from the saved boundary; work after the last durable checkpoint can be
repeated. The transport/evaluation additions preserve v0.1.x training fingerprints.
Evaluation-step weights are retained alongside their metrics. Interrupted uploads
stay local for retry; persistent network failure limits remote durability.

## Run the screen

```bash
python -m captionbert_rotary prepare --chunks 0 1 2 3 61
python -m captionbert_rotary screen --config configs/screen.json
python -m captionbert_rotary grade runs/screen
```

Preparation downloads about 4 GB of consensus tensors plus five caption JSON
files and token caches. Captions and targets are matched **only by flat row index**;
a count mismatch aborts rather than silently truncating or joining by image ID.
Ragged uint16 tokens, precomputed sentence positions, and fp16 target tensors
remain memory-mapped on disk. Each file is checked against its preparation hash.

The reference batch size is **2048 negatives**. Ordinary gradient accumulation
does not preserve that objective and is not used. A worst-case full-batch,
256-token forward/backward/Adam probe runs before training and restores the model
and RNG afterward. If it cannot fit, the job stops explicitly. Use the G4 notebook
for the full screen. Reducing `batch_size` produces a separate diagnostic recipe.

On an RTX 4090 the guard always caps **total device use at 20 decimal GB**, even
if a larger Colab allowance is accidentally passed. It reserves 768 MiB plus
currently observed use by other applications outside the PyTorch allocator
allowance. Total use is sampled before each step and reported in results. Other
applications can allocate asynchronously, so the cap cannot control their memory.
The Colab notebook uses an 80 GB total-device allowance on larger cards.

The literal v2 2000-step warmup is retained. The two-epoch four-chunk screen is
only about 2000 steps, so almost all of it is warmup. A shorter warmup would be a
separate recipe; do not silently alter it between arms.

## Architecture and evaluation

* A0 is the verbatim upstream `CaptionEncoder`, including initialization and the
  learned 8192×512 position table: **58,308,864 parameters**.
* A1–A3/A2g remove that table: **54,114,560 parameters**. The pre-norm GELU blocks,
  projection, dropout, and masked mean pooling retain the v2 widths. Attention
  uses SDPA and rotate-half q/k rotation with fp32 angles in every layer.
* A2 uses content 8 | slot 12 | anchor 12 pairs per 64-dimensional head. The slot
  ladder spans wavelengths 3–256; the simplex has 25 sentence addresses.
  Punctuation `.`, `!`, `?` belongs to its preceding sentence; the next token
  resets its slot and increments the sentence. [CLS] starts at (0,0). Sentence
  indices clamp at 24; manifests report rows that reach and exceed 24.
* A3 uses content 8 | slot 24 and no anchor. A2g keeps the A2 bands but uses the
  caption-wide token index on the slot stream.

Every run records held-out chunk-61 cosine, centered-variance participation ratio
(the original effective-rank definition), off-diagonal mean self-cosine, separate
>128-token gauges, timings, parameters, memory, and the original eight-task STS
suite. The STS harness keeps its **64-token** ceiling and test splits. Missing tasks
fail evaluation instead of inflating a partial-suite mean. Dataset revisions are
pinned once per experiment; no teacher models need to be downloaded.

Predictions are written to `predictions.json` before training. `summary.json`
compares paired seeds and reports their direction without declaring a settled
benefit from seed agreement alone; the applicable repeat/uncertainty evidence is
still required. Final models are evaluated consistently across arms; separate
best-by-heldout checkpoints are also retained. All arms ship, regardless of score.
Checkpoint resume restores model, Adam moments, scheduler, scaler, CPU/CUDA/NumPy/
Python RNG state, and the next batch cursor. Local saves are atomic.

## Load a trained arm

```python
from captionbert_rotary.inference import SentenceEncoder
encoder = SentenceEncoder.from_pretrained("runs/screen/A2-seed1")
vectors = encoder.encode(["A cat sits. It watches the street."])
# Hub: select an explicit experiment and arm/seed rather than guessing a winner.
encoder = SentenceEncoder.from_pretrained(
    "AbstractPhil/captionbert-8192-rotary",
    run="experiments/<config-id>/A2-seed1",
)
```

Code, tests, notebook, and build metadata live on GitHub. The Hub receives weights,
tokenizer assets, logs, configuration, predictions, and results, never Python code.
See `THIRD_PARTY.md` for the upstream revision and attribution.

## Rotary mechanism measurements

[Open the separate CPU diagnostics notebook](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.2.0/notebooks/captionbert_rotary_diagnostics.ipynb).
It reads completed frozen checkpoints from the existing campaign without changing
or restarting its training session. Existing task results and weights stay in
their original paths; additional measurements ship under each arm's
`analyses/captionbert-rotary-mechanisms-v1/` directory. Identical published or local
measurements are reused by checkpoint hash and probe identity. Run it again for
newly completed arms; it is not a persistent watcher.

The protocol measures address kernels, simplex coherence, the explicit orthogonal
lift, periodic aliases and clamping collisions; oracle address lookup with
identical content; frozen address/band interventions; and per-layer/head q/k score,
energy and attention-mass measurements. It includes an actual repeated frozen
baseline and controls where a band is absent. See [the measurement scope](docs/rotary_measurements.md).

```bash
python -m captionbert_rotary diagnostics --experiment 817daa9e66a8119dc6b18a9b63fad32490d1c891e5f26750e91125d9e5e14e6c --device cpu --threads 2
```

Future screen arms collect the same diagnostics automatically. The CPU notebook
uses its own environment and `requirements-cpu.txt`; it does not download the
consensus bank or train a new model. A0's rotary-band fields are inapplicable.

## Model README

The final Hub card follows the pinned
[CaptionBERT v2 format](https://huggingface.co/AbstractPhil/captionbert-8192-v2/blob/ed468fe5aed64a3d03a56a7f7811131d20f81ddb/README.md):
loading, the full eight-task table, geometry, rotary measurements, training recipe,
known limits, output convention and citation. Its numbers come from actual saved
arm/seed reports. This short screen is distinguished from the original full v2
training run. Loading uses this package's normalized tensor interface and selects
an explicit run and Hub revision.

The screen runner and diagnostics notebook finalize the card after shipping.
Manual finalization is also available:

```bash
python -m captionbert_rotary model-card --experiment 817daa9e66a8119dc6b18a9b63fad32490d1c891e5f26750e91125d9e5e14e6c --push
```

Publication requires all four arms and both seeds, finite complete task/geometry
results, matching data/benchmark identities and frozen reports tied to the actual
weight hashes. Missing evidence produces a local draft with `readiness.json`,
without replacing the public README. A parent-revision guard prevents concurrent
Hub updates from being silently overwritten.
