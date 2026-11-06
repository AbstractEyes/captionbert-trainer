# CaptionBERT rotary trainer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/v0.1.0/notebooks/captionbert_rotary_screen.ipynb)

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
in a durable queue and halt progression after at most one further active arm.
Resume checks the Hub for completed runs and verifies configuration/data identity.

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
compares paired seeds and leaves findings unsettled if a seed is missing or the
directions disagree. Final models are evaluated consistently across arms; separate
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
