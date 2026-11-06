# CaptionBERT with the multi-position rotary: a build brief for Codex (2026-11-05)

Prepared by Claude (Opus 5.5) from the research record for Phil to hand to Codex (GPT 6.1 sol). The goal: a variant of the
CaptionBERT v2 sentence encoder whose learned absolute position table is replaced by the house's multi-position anchored rotary,
trained from 2026-11-06, with the same data, objective and gauges as v2, so the result also advances the rotary research line.
Everything below is public on the Hugging Face Hub or written out here; nothing in this brief needs access to Phil's machines.

## 1. The base model: CaptionBERT v2 (shipped 2026-08-02)

Hub (public): `AbstractPhil/captionbert-8192-v2`. Its `trainers/pretrain_trainer.py` is the full v2 pipeline (stage 0 a parity
gate, 1 the alignment fit, 2 the consensus targets, 3 training); `modeling_captionbert.py` is the inference model;
`full_benchmark.py` and `early_eval.py` are the evaluation harness.

| part | v2 as shipped |
|---|---|
| tokenizer | bert-base-uncased (vocab 30,522), dynamic padding, ceiling 256 tokens (`max_tokens`), mean caption 48 tokens, length bucketing within windows of 64 batches |
| embeddings | token embedding + LEARNED absolute position embedding (`nn.Embedding(8192, 512)`, 4.2M parameters), then LayerNorm, then dropout 0.1 |
| trunk | 12 x `nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)`; head dim 64 |
| pooling | masked mean over tokens (`pooling="mean"`; `"cls"` exists as an option) |
| head | Linear(512, 512) -> GELU -> LayerNorm -> Linear(512, 768), then L2 normalization |
| size | 58,308,864 parameters |
| objective | InfoNCE (temperature 0.07, in-batch negatives) + per-sample MSE, between the student's embedding and a precomputed consensus target |
| optimizer | plain Adam, no weight decay; lr 6e-4, linear warmup from 1% over 2,000 steps, cosine to 1e-6 |
| batch | 2,048 (the batch size is the contrastive negative count) |
| run | 26.9M rows (54 of 66 chunks), 52,548 steps, 5 h 24 min on one RTX PRO 6000 (96 GB) |
| held out | chunks 60 and 61 |

THE TARGETS. `AbstractPhil/captionbert-8192-v2-consensus` (public, ~50.5 GB): `consensus_000.pt` ... `consensus_065.pt`, one
768-d fp16 vector per caption row. The consensus is the normalized centroid of five BERT-family teachers (bert-base,
ModernBERT-base, roberta-base, albert-base-v2, distilbert), each mapped once into bert-base's frame by a whitened Procrustes fit.
The captions are in `AbstractPhil/conceptual-captions-12m-webdataset-berts` (public; `captions_NNN.json`; CC BY-SA 4.0; three
fields per image: the LLaVA-NeXT long caption, its shortened form, the alt text). So training needs stage 3 only: pair each
chunk's caption rows with the same chunk's consensus rows by row index.
DATA CAVEAT: one caption row is missing in the bank (image 7,341,618 has no long caption), so pair captions and targets by row
index, never by `row // 3` (that join mis-pairs every image after it).

THE NUMBERS TO BEAT (v2): in-domain cosine to the consensus .8410 and effective rank 102.9 on the held-out chunk; the 8-task
harness mean .6077 (best teacher distilbert .5883; all-MiniLM-L6-v2 .7925 for scale); effective rank 36.6 on STS-B sentences.

## 2. The multi-position anchored rotary (the house's structure)

THE ADDRESS KERNEL. For rotate-half RoPE with pair frequencies w_j, identical unit content at positions m and n scores
    K(m - n) = (1/P) sum_j cos(w_j (m - n))
Its main lobe is the resolution, its sidelobes are what a mismatched position leaks, and the positions packed at |K| <= eps are
the capacity.

THE STRUCTURE. Every token carries, besides its sequence index, a structural address from the layout (compartment c, slot s).
A head's P rotary pairs are split by job:
- content pairs: w = 0 (not rotated; distance-free; owned by no position);
- slot pairs (the relative band): a geometric ladder of wavelengths on stream 0 (the slot coordinate; outside compartments the
  sequence index, so the band is an ordinary relative rotary there);
- anchor pairs: w_k = 2 pi k / (2 P_a + 1), k = 1..P_a, on stream 1 (the compartment index).
    q_m . k_n = sum_content <q_j, k_j> + sum_slot Re[q_j conj(k_j) e^{i w_j (s_m - s_n)}] + sum_anchor Re[q_k conj(k_k) e^{i w_k (c_m - c_n)}]
The anchor code on integer compartments is a regular simplex of 2 P_a + 1 anchors (kernel -1/(2 P_a) between distinct ones);
one plain dimension lifts it to 2 P_a + 1 orthonormal anchors. Beyond 2 P_a + 1 compartments the codes must overlap.

WHAT THE EXPERIMENTS ESTABLISHED (small synthetic retrieval beds, two seeds each; weights and results public at
`AbstractPhil/geolip-rotary-experiments`, `weights/` and `reports/`):

| finding | numbers |
|---|---|
| a query rotated to the structural address it asks for finds it exactly, at any length | 1.000 with five codes; absolute slot at 4x the trained length 1.000 |
| the multi-position rotary answers relative and fixed-position questions in one sequence | 1.000 / 1.000 |
| the anchor band carries a compartment's identity when content cannot | 1.000 against .17 without it |
| the anchor band's capacity is its coherence | 2 P_a + 1 orthogonal anchors exact; leak grows with overlap |
| the cost at scale | the rotation is 3.7-5.3% of a block's time at 4k tokens, 1.5-3.0% at 16k-64k |

NOT YET KNOWN: whether any of this helps a sentence encoder trained on a contrastive target. This screen is its first test.

THE REFERENCE IMPLEMENTATION (verbatim from the rotary bench; rotate-half convention, fp32 angles):

```python
import math
import torch
import torch.nn as nn


def rope_half(t, cos, sin):
    P = cos.shape[-1]
    a, b = t[..., :P], t[..., P:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


def freqs_rope(P, theta=10000.0):
    return theta ** (-torch.arange(P, dtype=torch.float64) / P)


def slot_ladder(n, lo=3.0, hi=64.0):
    """n pair frequencies with wavelengths geometric from lo to hi positions (the relative band)."""
    lam = lo * (hi / lo) ** (torch.arange(n, dtype=torch.float64) / max(1, n - 1))
    return 2 * math.pi / lam


def anchor_simplex(n):
    """n pairs at w_k = 2 pi k / (2n + 1): 2n + 1 equiangular anchors."""
    return 2 * math.pi * torch.arange(1, n + 1, dtype=torch.float64) / (2 * n + 1)


def mpar_bands(P, content, slot, anchor, lo=3.0, hi=64.0):
    """[content | slot | anchor] pairs: content unrotated, slot = the ladder on stream 0, anchor = the simplex code on stream 1."""
    assert content + slot + anchor == P
    w = torch.cat([torch.zeros(content, dtype=torch.float64), slot_ladder(slot, lo, hi), anchor_simplex(anchor)])
    stream = torch.cat([torch.zeros(content + slot, dtype=torch.long), torch.ones(anchor, dtype=torch.long)])
    return w, stream


class RopeMulti(nn.Module):
    """rotate-half RoPE with one position stream per pair: positions (B, S, n_streams); pair j turns at w_j on stream_j."""

    def __init__(self, w, stream):
        super().__init__()
        self.register_buffer("w", w.float(), persistent=False)
        self.register_buffer("stream", stream.long(), persistent=False)

    def forward(self, q, k, pq, pk):            # q, k: (B, H, S, head_dim); pq, pk: (B, S, n_streams)
        aq = (pq.float()[..., self.stream] * self.w)[:, None]
        ak = (pk.float()[..., self.stream] * self.w)[:, None]
        return rope_half(q, aq.cos(), aq.sin()), rope_half(k, ak.cos(), ak.sin())
```

The bench's split at head dim 32 (P = 16) was content 4 | slot 6 | anchor 6 (13 anchors).

## 3. The mapping onto CaptionBERT (settled 23:1x: the sentence number)

- Head dim 64, so P = 32 pairs per head; the rotation applies to q and k in every layer; the learned position table is removed
  (58.3M -> 54.1M parameters).
- Compartment = the sentence index inside the caption (0 for [CLS] and the first sentence; +1 after each token ".", "!" or "?";
  computed once at tokenization); slot = the token's offset inside its sentence (the house structure's own mapping). With 12
  anchor pairs there are 25 orthonormal sentence anchors; clamp longer captions at 24 and report how many rows reach it.
- Split at P = 32: content 8 | slot 12 | anchor 12 (the bench's 4 | 6 | 6 doubled); the slot ladder's wavelengths 3 -> 256
  tokens (the caption ceiling) instead of the bench's 3 -> 64.
- `nn.TransformerEncoderLayer` cannot take a rotary: replace it with an equivalent layer (pre-norm, GELU, dropout 0.1, the same
  widths) whose attention projects q, k, v, rotates q and k with `RopeMulti`, and calls
  `torch.nn.functional.scaled_dot_product_attention` with the key padding mask. Everything else stays as in v2.

## 4. The screen: arms, scale, gauges

Every arm sees the same rows in the same order, for the same steps, with two seeds (1 and 2):

| arm | positions |
|---|---|
| A0 replication | v2 verbatim: the learned position table (the record's recipe at the screen's scale) |
| A1 plain RoPE | all 32 pairs on the token index, `freqs_rope(32)` (theta 10,000); no table |
| A2 multi-position rotary | content 8, slot 12 (in-sentence offset, ladder 3 -> 256), anchor 12 (sentence index) |
| A3 anchor ablation | content 8, slot 24 (the same ladder on the in-sentence offset), no anchor band |
| A2g (optional) | A2 with stream 0 = the token index in the whole caption instead of the in-sentence offset |

Scale for a first read: four training chunks (about 2M rows) for two epochs at batch 2,048 (about 2,000 steps); v2 ran about
5,500 rows a second on the RTX PRO 6000, so roughly 15 minutes an arm and about 2 hours for eight runs. Validation on held-out
chunk 61; a run that reads well earns the full 54-chunk run.

Gauges, every arm: in-domain cosine to the consensus and effective rank on the held-out chunk; the 8-task harness mean
(`full_benchmark.py`); the mean self-cosine (isotropy); captions above 128 tokens scored separately (where the learned table is
thinnest); seconds per step and parameter count. Ship every arm's weights and results, the replication arm and any arm that
reads worse included.

Predictions are written down before any run (what each arm should read against A0 and why), then graded from the files.

## 5. House rules for the code

- Plain Adam, no weight decay; never AdamW.
- Two seeds per arm; a finding is not settled until both seeds agree.
- The replication arm (A0) runs beside the variants, verbatim.
- Long runs print the job's size, the time spent and the time left.
- Notebook cells must run when pasted whole into Colab: no argparse, no `__file__`, imports inside the cell.
- The code lives in its own GitHub repository with pinned requirements and tests; the Hugging Face Hub holds weights and results,
  never code.
- The notebook lives in that repository with an Open-in-Colab badge in its first cell; it installs the code from a tagged
  release; it reads the `HF_TOKEN` Colab secret for the pushes.
- Results ship as they land: a background upload with retries after each arm (written locally first), so a lost runtime
  costs one arm at most; a resume reads what already shipped.

## 6. Settled (Phil, 2026-11-05 23:1x)

1. POOLING: masked mean pooling in every arm, as v2 trains (its targets and its five teachers are mean-pooled); for this line
   it stands in place of the house's rule against averaging in geometric encoders.
2. THE COMPARTMENT: the sentence number inside the caption (section 3).
3. THE HUB DESTINATION: https://huggingface.co/AbstractPhil/captionbert-8192-rotary (every arm's weights, logs and results;
   code stays on GitHub).
4. THE GPU: Colab's G4 (RTX PRO 6000, 96 GB) at v2's settings; Codex writes the notebook (section 5's rules).

## 7. Later, if the screen reads well

- The key binding: a token's identity bound into some of a head's pairs by fixed random phases (in the record a told head
  found whole keys on twelve models of twelve this way, where a key code added to the input reached them on five of six
  seeds learned and three of six fixed); a candidate for word-level identity in captions.
- Members on anchor bands: attachments (adapters) that each own a band of anchors and stay quiet outside it, gated by the
  sinh / cosh form, not a softmax (merged members within .008 of solo, KL .0000, in the record).

## Pointers into the research record (for Phil and Claude)

repos/captionbert-v2.md, repos/captionbert.md, history/CAMPAIGN_RECORD_2026-08-02_captionbert.md; canon/rotary_catalogue.md
s.1-s.2 (the structure), s.8 (the multi-position rotary on mixed sequences), s.15 (the night in one page), s.16 (speeds), s.34
(the key binding); canon/trie_pillar.md (the binding generalized to every unit); the bench:
history/dockets/2026-10-22_rotary_night/tools/rotary_bench.py (`mpar_bands`, `BenchRopeMulti`).
