---
license: mit
language: [en]
library_name: transformers
pipeline_tag: feature-extraction
tags: [sentence-similarity, feature-extraction, consensus-distillation, rotary-embeddings, custom_code, safetensors]
datasets: [AbstractPhil/conceptual-captions-12m-webdataset-berts, AbstractPhil/captionbert-8192-v2-consensus]
base_model: [google-bert/bert-base-uncased, answerdotai/ModernBERT-base, FacebookAI/roberta-base, albert/albert-base-v2, distilbert/distilbert-base-uncased]
---

# CaptionBERT 8192 rotary

A **54.1M-parameter** standalone caption sentence encoder distilled from the
published consensus of five BERT-family teachers. Inference uses this model
and its tokenizer, with **768-dimensional normalized sentence output**.

The default is **Relative 16 heads (N16), seed 1**, trained on
**4,000,000 paired caption rows** for **7,808 steps**. It is the
development-selected release from the completed structured-rotary campaign.
The two 4M architecture means are a near tie; the default follows the declared
preference for fewer additions. Both seed pairs and every other trained option
remain available.

12 layers · hidden width 512 · 16 heads · FFN2048 · trained input ceiling 256.
**8192 is the checked execution capacity; 8K semantic quality was not measured.**

[Plain-English report](STRUCTURED_REPORT.md) · [Technical companion](TECHNICAL_STRUCTURED.md) ·
[Training repository](https://github.com/AbstractEyes/captionbert-trainer) ·
[CPU TensorBoard viewer](https://colab.research.google.com/github/AbstractEyes/captionbert-trainer/blob/campaign-structured-20261106/notebooks/captionbert_tensorboard_viewer.ipynb)

## Load the model

~~~python
import torch
from transformers import AutoModel, AutoTokenizer

repo = "AbstractPhil/captionbert-8192-rotary"
revision = "ad14b77f4cc9017d513173f3b340e735af02ef14"
tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
model = AutoModel.from_pretrained(
    repo, revision=revision, trust_remote_code=True, use_safetensors=True
).eval()

batch = tokenizer(
    ["A cat watches from a windowsill.", "A feline looks out the window."],
    padding=True, truncation=True, max_length=256, return_tensors="pt"
)
with torch.inference_mode():
    output = model(**batch)

embeddings = output.pooler_output       # [2, 768], L2-normalized
tokens = output.last_hidden_state      # [2, sequence_length, 512]
similarity = (embeddings[0] @ embeddings[1]).item()
~~~

The saved tokenizer uses right padding, matching training. No teacher models
or training package are required. Pin the tested revision in the serving
verification report when reproducing exact results.

Weights are **model.safetensors**, saved with **safe_serialization=True** and
PyTorch header metadata **format: pt**. Self-contained configuration/model source
and AutoConfig/AutoModel mappings accompany them.

## Final sentence benchmarks

The default architecture and seed were selected using STS-B **validation**,
before reading these test outcomes. All six larger models are shown.
Spearman correlation, 64-token ceiling, same eight pinned task test splits;
the mean weights tasks equally.

| Recipe / option | Seed | STS-B | SICK-R | STS12 | STS13 | STS14 | STS15 | STS16 | BIOSSES | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4M ×4 epochs / Hierarchical | 1 | 0.5427 | 0.6536 | 0.4882 | 0.5523 | 0.5275 | 0.7023 | 0.6226 | 0.4753 | 0.5706 |
| 4M ×4 epochs / Hierarchical | 2 | 0.5540 | 0.6535 | 0.4932 | 0.5589 | 0.5285 | 0.6981 | 0.6243 | 0.5197 | 0.5788 |
| 4M ×4 epochs / Relative 16 heads | 1 | 0.5385 | 0.6451 | 0.4823 | 0.5466 | 0.5300 | 0.6997 | 0.6232 | 0.5169 | 0.5728 |
| 4M ×4 epochs / Relative 16 heads | 2 | 0.5448 | 0.6492 | 0.4884 | 0.5500 | 0.5316 | 0.7002 | 0.6249 | 0.4791 | 0.5710 |
| 1M ×16 epochs / Hierarchical | 1 | 0.5451 | 0.6553 | 0.4837 | 0.5260 | 0.5173 | 0.6950 | 0.6107 | 0.4657 | 0.5623 |
| 1M ×16 epochs / Hierarchical | 2 | 0.5410 | 0.6570 | 0.4913 | 0.5358 | 0.5244 | 0.7005 | 0.6046 | 0.4975 | 0.5690 |

Pair test means are hierarchical 4M **0.57467**, relative 4M **0.57191**, and hierarchical matched exposure **0.56567**. Hierarchical-minus-relative directions differ between seeds. The strongest individual test row is hierarchical seed 2; it does not replace the development-selected default.

The release is one checkpoint, with both seeds retained for comparison.
Training gradients use caption/consensus targets; STS-B validation labels
influence architecture and seed selection. The historical CaptionBERT v2
numbers are a separate experiment and are not treated as a matched rotary
comparison here.

### Development and geometry

| Recipe | Option | Seed | Dev STS-B | Dev teacher cosine | Dev participation rank |
| --- | --- | --- | --- | --- | --- |
| 4M ×4 epochs | Hierarchical | 1 | 0.68295 | 0.81017 | 96.60 |
| 4M ×4 epochs | Hierarchical | 2 | 0.68854 | 0.81066 | 96.51 |
| 4M ×4 epochs | Relative 16 heads | 1 | 0.68672 | 0.81769 | 97.20 |
| 4M ×4 epochs | Relative 16 heads | 2 | 0.68633 | 0.81811 | 97.13 |
| 1M ×16 epochs | Hierarchical | 1 | 0.68331 | 0.80801 | 95.63 |
| 1M ×16 epochs | Hierarchical | 2 | 0.67775 | 0.80879 | 95.51 |

| Recipe / option | Seed | Chunk060 sample | n | Teacher cosine | Participation rank | Self cosine |
| --- | --- | --- | --- | --- | --- | --- |
| 4M ×4 epochs / Hierarchical | 1 | All sampled rows | 5000 | 0.809758 | 95.952 | 0.003311 |
| 4M ×4 epochs / Hierarchical | 1 | >128-token rows | 5000 | 0.735772 | 75.211 | 0.008305 |
| 4M ×4 epochs / Hierarchical | 2 | All sampled rows | 5000 | 0.810098 | 95.854 | 0.003095 |
| 4M ×4 epochs / Hierarchical | 2 | >128-token rows | 5000 | 0.735145 | 75.173 | 0.008127 |
| 4M ×4 epochs / Relative 16 heads | 1 | All sampled rows | 5000 | 0.817975 | 96.337 | 0.003996 |
| 4M ×4 epochs / Relative 16 heads | 1 | >128-token rows | 5000 | 0.740089 | 76.073 | 0.010202 |
| 4M ×4 epochs / Relative 16 heads | 2 | All sampled rows | 5000 | 0.818179 | 96.243 | 0.003710 |
| 4M ×4 epochs / Relative 16 heads | 2 | >128-token rows | 5000 | 0.740014 | 76.053 | 0.009820 |
| 1M ×16 epochs / Hierarchical | 1 | All sampled rows | 5000 | 0.807663 | 94.974 | 0.003575 |
| 1M ×16 epochs / Hierarchical | 1 | >128-token rows | 5000 | 0.737254 | 73.677 | 0.011612 |
| 1M ×16 epochs / Hierarchical | 2 | All sampled rows | 5000 | 0.808123 | 94.868 | 0.003176 |
| 1M ×16 epochs / Hierarchical | 2 | >128-token rows | 5000 | 0.737987 | 73.511 | 0.010763 |

Caption geometry uses deterministic 5,000-row samples from the separate chunk 060,
plus a separate 5,000-row >128-token sample. Participation rank uses centered
covariance in FP64; self cosine is the exact off-diagonal mean.
Teacher cosine measures consensus reproduction, separately from capability.

## What the structured experiment found

The screen trained eight options under both weight seeds and repeated its
sixteen-head relative reference, for 18 screen models. The larger stage adds
the two4M architecture pairs and a1M matched-exposure hierarchical pair,
for 24 trained models in this campaign.

The complete whole-key model's designated lookup head found the correct earlier
record on all 10,146 sampled query positions under both seeds. Mean correct-record
probability was0.999510/0.999516, across1,935 target-bearing captions from 4,096
development rows. Its sentence-development result did not establish a benefit
from that accurate read. The full report separates word-record attention,
teacher reproduction, pooled sentence similarity and serving parity.

The adaptation uses a three-digit base-eight segment address, a zero-born value
path and, in complete binding options, whole-WordPiece key phases, record-only
norm 2.18 and lookup supervision. The default's actual rotary mode is
**NTK-scaled relative RoPE; 16 heads, scale 4; no hierarchical value path or lexical binding**. Complete binding models and controls remain under their
explicit experiment paths; they are not silently replaced by the default.

## Construction and continuation

- Screen:1M paired rows,4 epochs,1,952steps,200-step warmup.
- Larger4M: chunks000,001,002,003,004,006,009,010,4 epochs.
- Exposure control: original1M rows,16 epochs.
- Every larger recipe:7,808steps, batch 2,048,15,990,784 example presentations,
  400-step warmup, peak LR6e-4 and cosine decay to1e-6.
- Pure Adam, weight decay0, clipping1.0, symmetric InfoNCE at 0.07 plus MSE.
  Lookup loss weight0.1 only in its declared screen options.
- FP32 parameters/optimizer, BF16 training autocast, TF32 off.
- Caption/target pairing is exact flat row index. Chunk061 is development,
  chunk 060 caption test. No image-ID join, refit alignment or fallback teacher.

The [artifact census](https://huggingface.co/AbstractPhil/captionbert-8192-rotary/blob/main/campaigns/structured-20261106/artifact-census.json) pins every
run, final/best/midpoint weights, independent optimizer/RNG state, logs, raw
queries, data manifests and TensorBoard events. Original experiments retain their
paths. Recovery notebooks verify compatible identities, resume partial state and
skip only verified completed models.

## Numerical and shape verification

The selected default was tested from both its local export directory and the actual Hub repository at immutable revision **ad14b77f4cc9017d513173f3b340e735af02ef14**. Separate local and G4 processes compared it with the independently restored final training state and exact training implementation. Each default check contains twelve precision/shape cases plus two padding comparisons. Every tested native/AutoModel sentence and token output had maximum absolute difference zero. The G4 default check used Transformers 5.22.1, PyTorch 2.15.0+cu132 and Safetensors 0.8.0; the separate binding G4 check used Transformers 5.22.2. The local default checks stayed below 4.2 decimal GB total device use under an 18 GB guard. These are same-device, same-precision comparisons within each environment, not a claim that outputs are bitwise identical between different GPUs.

Measured reports: [local export](https://huggingface.co/AbstractPhil/captionbert-8192-rotary/blob/main/campaigns/structured-20261106/validation/deployment-local-parity.json), [Hub load on the local GPU](https://huggingface.co/AbstractPhil/captionbert-8192-rotary/blob/main/campaigns/structured-20261106/validation/deployment-remote-parity.json), [Hub load on G4](https://huggingface.co/AbstractPhil/captionbert-8192-rotary/blob/main/campaigns/structured-20261106/validation/deployment-colab-parity.json), and the [executed model-card example](https://huggingface.co/AbstractPhil/captionbert-8192-rotary/blob/main/campaigns/structured-20261106/validation/model-card-example.json).

Checks compare an independently loaded final training state with the separate
safe AutoModel export, including every state tensor and actual outputs.
They exercise multiple batch/length shapes, all-padding, right padding,
optional attention/pointer diagnostics and FP16/BF16 autocast with FP32 weights.

| Model / load | Transformers | PyTorch | Case entries | Max sentence / token delta | Peak observed device GB |
| --- | --- | --- | --- | --- | --- |
| Complete binding / local | not captured | 2.11.0+cu128 | 12 | 0 / 0 | 3.652 |
| Complete binding / Colab | 5.22.2 | 2.15.0+cu132 | 14 | 0 / 0 | 1.635 |
| Default / local directory | 4.57.6 | 2.11.0+cu128 | 14 | 0 / 0 | 4.120 |
| Default / Hub on local GPU | 4.57.6 | 2.11.0+cu128 | 14 | 0 / 0 | 4.384 |
| Default / Hub on Colab | 5.22.1 | 2.15.0+cu132 | 14 | 0 / 0 | 1.633 |

Outputs through 8,192 tokens are engineering shape/finite-value checks.
Long-context semantic quality,512-address retrieval in this encoder and
cross-hardware bitwise equality are not established by those checks.
Per-case timings are diagnostic wall times, not a throughput study.

## Limits

Training and caption evaluation use at most256 tokens; the sentence suite uses 64.
The data consists of caption rows with an English uncased WordPiece tokenizer;
image-level and near-duplicate independence across every chunk was not established.
Two weight seeds and the screen reference repeat do not provide a full-stage
population significance test.

Whole-word identity is exact after tokenization, including its uncasing.
The encoder is bidirectional and receives the complete input. A word-record read
here does not establish causal external-memory writing, coreference resolution,
or general binding-sensitive sentence quality. Frozen binding-path removal also
bypasses its norm operation; it does not isolate phase necessity for every
trainable head. The missing pointer-only trained control is a proposed next stage.

Use right padding for binding variants: shifting lexical word phases relative
to positional special-token phases can change their native output under left
padding. The serving verifier measures native/AutoModel agreement under both.

## Output convention

| Field | Shape | Meaning |
|---|---|---|
| last_hidden_state | [B,L,512] | Token states |
| pooler_output | [B,768] | L2-normalized sentence vector |
| hidden_states, when requested |13 tensors | Input plus 12 layers |
| attentions, when requested |12 tensors | [B,16,L,L] per layer |
| pointer_loss, when requested | scalar | Auxiliary diagnostic loss |

All-padding sentence vectors are zero. Token states are not promised to be zero
at padding positions. This API does not define an encode method or embedding
alias; use pooler_output for sentence similarity.

## Citation

~~~bibtex
@misc{abstractphil2026captionbertrotary,
  title = {Structured rotary CaptionBERT: precise word reads and the sentence-embedding test},
  author = {AbstractPhil},
  year = {2026},
  url = {https://huggingface.co/AbstractPhil/captionbert-8192-rotary}
}
~~~

MIT. Format follows the [CaptionBERT v2 card](https://huggingface.co/AbstractPhil/captionbert-8192-v2/blob/ed468fe5aed64a3d03a56a7f7811131d20f81ddb/README.md).
