# Structured CaptionBERT: technical companion

This companion specifies the November 6, 2026 experiment, its exact model and
measurement definitions. [REPORT.md](STRUCTURED_REPORT.md) gives the plain-English analysis.
The [model card](HF_MODEL_CARD.md) gives loading, benchmark and output conventions.
The tables below are generated from the completed, revision-pinned reports.

## Implementation and immutable identities

| Item | Identity |
|---|---|
| Source rotary implementation | geolip-nano-beatrix-trainer v0.1.103; 3f174e3f391f983428171cefc6901b34b22274d8 |
| Source published result | geolip-rotary-experiments, f99d3cbadeb2a75ae27bdbcbb8fbc05fd3f6f076; six seeds at 512 addresses |
| Training package | captionbert-trainer v0.3.0; cb1388dd3d313313c2cbe59b28200e3738793067 |
| Training modeling source SHA256 | 094bf61ef954c61a2b200e5e8760348160b76a5fbfd186602dd992e4b62ccfa6 |
| Inference modeling source | v0.3.1; caeb0e24e0f68bd6e3a7090b2380af5717dab572; optional FP16 diagnostic-mask correction |
| Screen configuration | dc37215529c188be7b93804c4a21f125afa353e037b1a2a8fc11e5b2fc817d3a |
| Screen data identity | 659b21c1c6eb1074eeec7acdbfa9bd010965595a4669338690cf3ae38a4448d5 |
| Full4M configuration | aafe9c59b9d696107cb4393f3a26815e84714daddb7d6eb028b5d6137b88418e |
| Full4M data identity | 7adb10f8833fab44e3ebb44315b53cb65bd8010b038115d15350b18536b57e86 |
| Matched-exposure configuration | fad07b2a05857a7abef278004986cacc1e3b8eb89b70943668505ea90498525e |
| Screen selection lock | HF becdda8590d138a92e947976005357fd6ccb9de2, campaigns/structured-20261106/locked-screen-selection.json |
| Raw selection-lock SHA256 | bdbf782e86b29545ec4a0731a8669b69ed5794f82914bebd1236fab007820338 |
| Final frozen probe source | 59ec95344958120a328f65a0b056632db66ee088 |
| Final frozen output folder | frozen-followup-59ec95344958 |

The complete binding source is the final whole-key/every-position form. The
transfer changes byte keys into WordPiece keys and explicit source record frames
into natural repeated-word records. It is a tokenizer/task adaptation. The
source's causal 512-address result is not an encoder-length benchmark.

## Encoder and rotary formulas

The encoder has vocabulary 30,522, width 512,12 pre-norm transformer layers,
GELU feed-forward width 2,048 and dropout0.1. Its pooling/projection is

\[
h_{\mathrm{pool}}={\sum_t m_t h_t\over\max(1,\sum_t m_t)},\qquad
z=\operatorname{normalize}(W_2\operatorname{LN}(\operatorname{GELU}
(W_1h_{\mathrm{pool}}+b_1))+b_2).
\]

The projection widths are512→512→768. The implementation explicitly returns a
zero sentence vector for an all-padding input. Token outputs have width 512,
sentence outputs 768. Relative models have54,114,560 trainable parameters;
hierarchical/binding models have54,126,848. The 12,288 extra parameters form the
three8×512 value-address tables. They begin at zero.

For each 32-coordinate structured attention head, coordinates use the
half-split rotation convention: pair j consists of coordinates j and j+16.
For an angle θ,

\[
R_\theta(a,b)=(a\cos\theta-b\sin\theta,\ a\sin\theta+b\cos\theta).
\]

The sixteen rotation planes split into two relative-position planes, two
unrotated content planes and twelve address planes. Relative frequencies are
1 and 1/400, using base 10000·4²=160000. For each of three address levels,
the four anchor frequencies are2πk/9 for k=1,2,3,4. A valid address digit d
has angle(d−8)·2πk/9. Digit8 is null, giving zero anchor angle. Query and key
address streams are separate arguments; ordinary inference supplies each
token's own segment address to both.

A segment increments after the tokenized punctuation ".", "!" or "?".
The punctuation token remains in its preceding segment. Segment s<512 has
digits(floor(s/64) mod8, floor(s/8) mod8, s mod8). Overflow and padding receive
null digits. Punctuation segmentation is not linguistic sentence parsing.
The value path adds one learned vector from each address-level table before
embedding normalization. A higher digit constant within a caption contributes
a constant input offset; its equal query/key rotation cancels in dot products
when that band's address is shared. Short-caption results therefore exercise
a limited fraction of the nominal address range.

The R8 relative control has eight64-coordinate heads and ordinary base 10000
RoPE. The N16 control has sixteen32-coordinate heads and NTK-scaled base

\[
b=10000\cdot4^{32/(32-2)},\qquad \omega_j=b^{-j/16}.
\]

Their learned projection matrix sizes match; head division and frequency
scaling differ together. R8 versus N16 is a package comparison.

## Whole-key phases and record normalization

A word key is the complete sequence of WordPiece IDs available in the input.
Continuation markers define word grouping. Special tokens are excluded from
binding. A record is the first occurrence's word ending; later identical
occurrences supply eligible query positions. Exact token-tuple targets are
computed independently from the model's phase hash.

The phase replaces the earlier source's byte/offset table with a fixed31-bit
counter hash over token ID, reverse offset and pair index. Each available piece
contributes to its word's sum, modulo2³¹. Multiplication/xor mixing and integer
scatter accumulation are specified in the self-contained modeling source.
All 16 pair phases are obtained from these sums. Equal complete keys receive
equal angles. There is no additional key-length cutoff, while the 256-token
input ceiling can still truncate a word.

KEND substitutes these angles only at word endings. KALL and the supervised
whole-key variants substitute them at all active word pieces in the configured
head/layer. Other heads retain their ordinary hierarchical angles. At bound positions, the
lookup head replaces all 16 angles, including its relative/content/address planes.
Phase and record construction includes active non-special token groups; the
auxiliary target filter selects a subset. Unsupervised background records can
therefore include short/common words and punctuation. With
bidirectional input, the phase at an early piece can use the rest of its word.
This does not make the operation available before the word arrives in a
streaming model.

For a record-key pair(a,b), normalization uses

\[
(a',b')=(a,b)\,{\rho\over\sqrt{\max(a^2+b^2,10^{-12})}},\qquad \rho=2.18.
\]

Only record keys receive the norm hold. The denominator clamps squared length
before square root so padded zero pairs retain finite derivatives. This is a
numerical adaptation for padded inputs, preserving the forward floor of 10⁻⁶.
KALLP_N0 disables the norm hold during training. Frozen norm-off changes both
the serialized configuration and the live layer setting.

Binding and special-token phases make left padding a distinct input convention:
bound words use lexical angles while unbound special tokens retain positional
angles. The release uses right padding, matching training. Serving verification
compares each padded input against the native implementation and separately
requires right-padding invariance. It does not silently alter trained math to
force left-padding invariance.

## Objectives, schedules and data

Let Z be the normalized student batch and T the normalized consensus batch,
both with 768 coordinates. For B=2048 and temperature τ=.07:

\[
S=ZT^\mathsf{T}/\tau,\quad
L_{\mathrm{NCE}}=\tfrac12[\operatorname{CE}(S,i)+\operatorname{CE}
(S^\mathsf{T},i)],\quad
L_{\mathrm{MSE}}={1\over768B}\sum_{i,d}(Z_{id}-T_{id})^2,
\]
\[
L=L_{\mathrm{NCE}}+L_{\mathrm{MSE}}+\lambda L_{\mathrm{pointer}}.
\]

The pointer term is mean negative log probability on the exact earlier
first-occurrence ending at the designated head, with λ=.1 where enabled.
It excludes common function words, keys without an alphabetic character and keys shorter than
three characters. The main caption loss uses every example. Ordinary gradient
accumulation would change the batch contrastive problem and is not used.

| Property | Screen | Full4M | Matched exposure |
|---|---:|---:|---:|
| Available paired row indices |1,000,000|4,000,000|1,000,000|
| Epoch limit |4|4|16|
| Optimizer steps |1,952|7,808|7,808|
| Example presentations |3,997,696|15,990,784|15,990,784|
| Warmup steps |200|400|400|
| Batch size |2,048|2,048|2,048|
| Maximum input tokens |256|256|256|
| Weight seeds |1,2|1,2|1,2|

Pure Adam uses weight decay0, peak LR6e-4, minimum LR1e-6, cosine decay and
gradient clipping1.0. Parameters and optimizer states are FP32; training
autocast is BF16. TF32 is disabled. The exposure control matches examples and
optimizer steps, not exact processed tokens, FLOPs or length distributions.
Incomplete trailing batches are omitted by the registered row plan.

The screen uses chunks000–001. The 4M stage uses 000,001,002,003,004,006,009,010.
Chunk061 is caption development; chunk 060 is reserved caption test. Pairing is
strictly by flat row index against the fixed five-teacher consensus. Source
revisions and preparation hashes are embedded in the data identity; no image-ID
join, row/3 join, fallback teacher or refit alignment is used. The teachers are
BERT-base, ModernBERT-base, RoBERTa-base, ALBERT-base and DistilBERT.
Cross-chunk image-level/near-duplicate independence was not established here.

Actual G4 training uses Python3.13.16, torch2.15.0+cu132, CUDA13.2 and NVIDIA
RTX PRO6000 Blackwell Server Edition. Each run ships its environment and
pip-freeze record. Dependencies are floors inside an isolated environment;
the package source itself is pinned. Colab host packages are left intact.

## Selection and uncertainty

Screen ranking uses paired STS-B validation Spearman, with a64-token ceiling
and dataset revision96943a16ea6a35129e253c659081cb59daf81b30.
A structured pair is viable if its mean development teacher cosine is within.02
of the selected relative pair, mean development participation rank is at least.8
of that reference, and neither seed falls below.5 of its paired reference rank.

A comparison's descriptive variation band is the maximum of its two pair seed
spreads and the measured reference same-seed repeat. Near ties prefer fewer
enabled additions, then fewer parameters, then lower measured optimizer-step
time. Enabled additions count NTK scaling, hierarchy, value addresses, key
binding, record norm and pointer supervision. Both seeds remain in the pair.

The measured screen reference-repeat floor is.006177472234669801. The original
N16 pair mean is.6740236671082218; its repeated pair mean is.6709231783137373.
The repeated pair itself has seed spread.0118125561946757. The registered rule
uses the original pair and the same-seed repeat floor; it does not average four
reference runs or replace the original seed spread with the repeat pair's
spread. This sensitivity is reported rather than changed after selection.

The screen selects H16 and N16; KEND is the raw structured mean leader. The
full4M default comparison applies the same viability and near-tie preference
using development values. Within the chosen pair the better development seed
is deployed; exact ties choose seed 1. The matched-exposure pair remains a
scientific control. No final test score chooses architecture, seed or tasks.

A same-seed repeat at the full4M recipe was not measured. The screen repeat
is a conservative delivery preference, not a certified larger-stage noise
floor or a confidence interval. Final pair differences are descriptive.

The development rule selected **Relative 16 heads / seed 1** for the default4M release. The raw development mean leader was **Relative 16 heads**. Structured-pair viability: **True**. The complete deployment-selection JSON retains both pair means, seed spreads and the rule; final tests were not used.

## Measurements

Caption development and caption test each use5,000 deterministic rows from
their own chunk, plus a separate sample of up to5,000 inputs above128 tokens.
The latter is defined by tokenized input length, within the 256-token ceiling.
Teacher cosine measures reproduction of the frozen consensus.

Centered-covariance participation rank is computed in FP64:

\[
r={(\sum_i\lambda_i)^2\over\sum_i\lambda_i^2},\qquad
C=(X-\bar X)^\mathsf{T}(X-\bar X).
\]

This is not entropy-based effective rank. Off-diagonal self cosine is computed
exactly from normalized vectors without allocating a full pairwise matrix:

\[
\bar c={\|\sum_i x_i\|^2-\sum_i\|x_i\|^2\over n(n-1)}.
\]

The eight task test suite is STS-B, SICK-R, STS12–16 and BIOSSES, always with a
64-token ceiling. Spearman uses cosine similarity of normalized outputs.
The reported suite mean is unweighted across tasks. Dataset revisions are
locked before the larger jobs and checked by the collector. The dataset
builder may cache other splits internally; no such test scores enter selection.

Frozen lookup uses 4096 deterministic chunk 061 rows, sample seed 20261106 and row
hash e72547c65ba2f4d26131988027abd9004af0a350e3bd404242b0f843779eb8ed.
There are1935 target-bearing captions and 10,146 eligible query positions.
Ending queries number9483; non-ending663; continuation pieces663; queries in
above128-token captions3198. Non-ending and continuation are distinct subsets.

Each model is scored at its configured lookup layer/head, without searching
other heads. Metrics include correct-record probability/top-one, wrong-adjacent
probability, uniform probability, distribution tails and equally weighted
per-caption means. Query positions within a caption are correlated.
Per-query arrays and original sampled row IDs are preserved.

| Option | Top-one seed 1 (%) | Top-one seed 2 (%) | Record probability seed 1 | Record probability seed 2 |
| --- | --- | --- | --- | --- |
| Relative 8 heads | 0.444 | 0.394 | 0.013839 | 0.017438 |
| Relative 16 heads | 0.414 | 0.907 | 0.014163 | 0.020720 |
| Relative 16-head repeat | 0.670 | 0.838 | 0.015755 | 0.020573 |
| Hierarchical | 1.360 | 1.242 | 0.022474 | 0.021603 |
| Key at endings | 32.328 | 23.950 | 0.155960 | 0.104036 |
| Key at every piece | 11.482 | 12.054 | 0.056840 | 0.053741 |
| Whole key + lookup | 100.000 | 100.000 | 0.999510 | 0.999516 |
| Lookup without record norm | 99.093 | 99.024 | 0.975021 | 0.978533 |
| Lookup at later layer | 100.000 | 100.000 | 0.999720 | 0.999696 |

## Frozen interventions

The versioned follow-up preserves the original intact screen reports.
The KALLP pair gets intact, binding-off, norm-off, fixed-phase redraw and a
reconstructed fresh initialization under its registered weight seed.
Initialization has zero optimizer steps in the same frozen runtime; it is not
a saved original birth checkpoint. The trained interventions also receive
teacher and STS-B development measurements. Binding-off switches the rotary
mode to hierarchical and bypasses the record-norm operation inside that path;
it is a complete-path removal, not an isolated phase-only intervention.

H16 full4M and exposure controls get intact, anchor-rotation-off and
value-address-constant conditions. Anchor removal zeros only frequencies
whose kind is an address level, retaining relative frequencies. Value removal
sets every row in a level to that level's digit-zero row, retaining the
default offset while removing within-level address variation. Every modified
tensor/config/runtime setting is restored even after exceptions. N16 full4M
gets the intact reference probe.

Phase redraw retains the same key-equality construction. It changes a fixed
trained model's cross-key background, not the fact that equal keys share
angles. It is not a new phase draw used during training. Frozen removals
measure checkpoint dependence, not the performance of a newly trained
replacement. There is no trained pointer-only/no-phase arm in this matrix.

| Condition | Top-one seed 1 (%) | Top-one seed 2 (%) | Query mean probability 1 | Query mean probability 2 | Caption mean probability 1 | Caption mean probability 2 |
| --- | --- | --- | --- | --- | --- | --- |
| Intact trained head | 100.000 | 100.000 | 0.999510 | 0.999516 | 0.999637 | 0.999664 |
| All lexical phases identical; norm retained | 33.343 | 34.132 | 0.113691 | 0.096532 | 0.166191 | 0.145697 |
| Binding path off; also bypasses norm | 0.394 | 0.404 | 0.018799 | 0.017014 | 0.033147 | 0.029673 |
| Record norm off; phases retained | 40.173 | 39.651 | 0.349622 | 0.352166 | 0.405128 | 0.407843 |
| New fixed phase draw; key equality retained | 100.000 | 100.000 | 0.999478 | 0.999407 | 0.999604 | 0.999560 |
| Reconstructed fresh initialization | 2.070 | 1.961 | 0.014870 | 0.014196 | 0.021699 | 0.020848 |

| Condition | Dev STS-B seed 1 | Dev STS-B seed 2 | Teacher cosine seed 1 | Teacher cosine seed 2 |
| --- | --- | --- | --- | --- |
| Intact trained head | 0.66589 | 0.66698 | 0.766422 | 0.766831 |
| All lexical phases identical; norm retained | 0.67098 | 0.67107 | 0.765583 | 0.765771 |
| Binding path off; also bypasses norm | 0.67243 | 0.67309 | 0.765547 | 0.765810 |
| Record norm off; phases retained | 0.66730 | 0.66907 | 0.766346 | 0.766743 |
| New fixed phase draw; key equality retained | 0.66616 | 0.66702 | 0.766368 | 0.766798 |

| Recipe | Seed | Condition | Dev STS-B | Change from intact | Teacher cosine | Participation rank |
| --- | --- | --- | --- | --- | --- | --- |
| 4M ×4 | 1 | Intact | 0.68295 | 0.00000 | 0.810166 | 96.60 |
| 4M ×4 | 1 | Address rotations off | 0.68675 | 0.00381 | 0.813122 | 88.51 |
| 4M ×4 | 1 | Value address variation removed | 0.68294 | -0.00001 | 0.810102 | 96.59 |
| 4M ×4 | 2 | Intact | 0.68854 | 0.00000 | 0.810655 | 96.51 |
| 4M ×4 | 2 | Address rotations off | 0.69446 | 0.00591 | 0.805079 | 90.71 |
| 4M ×4 | 2 | Value address variation removed | 0.68864 | 0.00010 | 0.810677 | 96.50 |
| 1M ×16 | 1 | Intact | 0.68331 | 0.00000 | 0.808012 | 95.63 |
| 1M ×16 | 1 | Address rotations off | 0.69207 | 0.00876 | 0.808634 | 88.95 |
| 1M ×16 | 1 | Value address variation removed | 0.68325 | -0.00006 | 0.807947 | 95.62 |
| 1M ×16 | 2 | Intact | 0.67775 | 0.00000 | 0.808789 | 95.51 |
| 1M ×16 | 2 | Address rotations off | 0.68985 | 0.01210 | 0.805154 | 88.92 |
| 1M ×16 | 2 | Value address variation removed | 0.67776 | 0.00001 | 0.808771 | 95.52 |

## Safetensors, AutoModel and shape parity

Serving uses save_pretrained(..., safe_serialization=True), a file named
model.safetensors with header metadata {"format":"pt"}, the complete
CaptionBertConfig and self-contained model/configuration Python sources.
The configuration's auto_map names AutoConfig and AutoModel.
AutoModel returns last_hidden_state[B,L,512] and pooler_output[B,768].
The sentence vector has unit norm for nonempty input and is zero for all
padding. The API has no encode method or embedding alias.

Training-state replay and serving export are independent loads. Every state
tensor is compared with torch.equal, and the loader must report no missing,
unexpected or mismatched keys. Actual embeddings and token states are then
compared, with finite-value assertions and explicit padding cases.
The inference FP16 mask correction uses a finite value representable in the
actual diagnostic logits dtype. Its checked BF16 engineering loss and all
gradients matched the training source exactly; the training environments were
never upgraded mid-run.

The isolated serving verifier checks synthetic batch/length cases1×1,2×17,
3×128,1×512, real text and all-padding. Longer checks cover 1025 and 8192 tokens;
alternating punctuation crosses the 512-segment address boundary. AMP checks
cover FP16 and BF16 natural/all-padding inputs and optional pointer/attention
diagnostics. The output interface includes13 hidden-state tensors and 12
attention tensors, with the model's configured head count. Shape checks
establish execution, not semantic quality beyond the 256-token training range.

| Model / load | Transformers | PyTorch | Case entries | Max sentence / token delta | Peak observed device GB |
| --- | --- | --- | --- | --- | --- |
| Complete binding / local | not captured | 2.11.0+cu128 | 12 | 0 / 0 | 3.652 |
| Complete binding / Colab | 5.22.2 | 2.15.0+cu132 | 14 | 0 / 0 | 1.635 |
| Default / local directory | 4.57.6 | 2.11.0+cu128 | 14 | 0 / 0 | 4.120 |
| Default / Hub on local GPU | 4.57.6 | 2.11.0+cu128 | 14 | 0 / 0 | 4.384 |
| Default / Hub on Colab | 5.22.1 | 2.15.0+cu132 | 14 | 0 / 0 | 1.633 |

## Logs, continuation and artifact routes

Each arm/seed has a unique experiments/<config-id>/<arm>-seed<n>/ folder.
The artifact census contains all 24 models in this campaign and preserves
earlier experiments at their original paths. Every new model retains
model.safetensors, independent state.pt, best and trajectory-mid weights,
configuration, tokenizer, exact environment, training/evaluation logs,
development predictions and TensorBoard event files. Final tests, per-query
frozen arrays, sampled row IDs and complete operation consoles are retained.

state.pt preserves model, optimizer, scheduler, RNG and batch/epoch cursor.
Its internal step and identity are checked against the completed result.
resume.json can describe an earlier transport snapshot and is not the
authoritative final optimizer step. Fresh notebooks restore completed and
partial compatible runs from a pinned Hub revision, preserve newer local
state, and skip only verified completed runs. A disconnect can repeat work
after the latest durable boundary.

The follow-up notebooks restore the completed KALLP screen checkpoint even
when only larger-stage work has been recovered. This fresh-runtime path was
tested from an empty directory. Training progress includes step, losses, LR,
epoch, speed and ETA; JSONL retains every step. TensorBoard includes training,
development and frozen endpoint scalars. All event files remain downloadable
after GPU runtimes end.

The completed [artifact census](https://huggingface.co/AbstractPhil/captionbert-8192-rotary/blob/main/campaigns/structured-20261106/artifact-census.json) covers 18 screen and six larger trained models, with no missing required files. It records 86 native-checkpoint/AutoModel output comparisons across the frozen reports, plus the separate precision/shape serving checks above. Large experiment payloads are verified against revision-pinned Hub LFS SHA-256 and size metadata; the frozen checks actually load independent training states and compare tensors. This is not a claim that every large optimizer file was downloaded again to the local workstation.

The [TensorBoard verification](https://huggingface.co/AbstractPhil/captionbert-8192-rotary/blob/main/campaigns/structured-20261106/tensorboard-verification.json) read all 34 event files (52,871,434 bytes), required every training step through 1,952 or 7,808, matched final development values within 1e-6 and checked the frozen endpoints. The [prepared-data provenance](https://huggingface.co/AbstractPhil/captionbert-8192-rotary/blob/main/campaigns/structured-20261106/data-provenance.json) reconstructs the screen and full-data identities and preserves descriptive token/address counts. The CPU-only viewer downloads just these pinned events with readable screen/full4m/exposure1m paths; no GPU, checkpoint or training data is needed.

## References

- [Source rotary code](https://github.com/AbstractEyes/geolip-nano-beatrix-trainer/tree/3f174e3f391f983428171cefc6901b34b22274d8).
- [Source final rotary experiment](https://huggingface.co/AbstractPhil/geolip-rotary-experiments/blob/f99d3cbadeb2a75ae27bdbcbb8fbc05fd3f6f076/README.md).
- [CaptionBERT v2 construction and model-card format](https://huggingface.co/AbstractPhil/captionbert-8192-v2/blob/ed468fe5aed64a3d03a56a7f7811131d20f81ddb/README.md).
- [RoFormer](https://arxiv.org/abs/2104.09864v5).
- [Hugging Face custom models and AutoClass](https://huggingface.co/docs/transformers/en/custom_models).
- [Safetensors header metadata](https://huggingface.co/docs/safetensors/en/metadata_parsing).

## Exact data sources

| Source | Immutable revision |
| --- | --- |
| Caption rows | 4465c825e7f67abd5fdf1b681fb2052abbb06b82 |
| Published consensus | 566bfe623f68b0b3eebbe267ce8f98bd1248d0f8 |
| BERT uncased tokenizer | 86b5e0934494bd15c9632b12f734a8a67f723594 |
| mteb/biosses-sts | f4f1265a7a4b0c99e6da09859b887bdb1f36d8ed |
| mteb/sickr-sts | b93b4e08b707e9c9bc478e543495c3ced0b9d81e |
| mteb/sts12-sts | 3a51344839ec620dd3b753e0945bdf6c7e37af3e |
| mteb/sts13-sts | f475812ca75868066186343ebf60e01f95c75164 |
| mteb/sts14-sts | 8ab5e540b20b58d42fa6b944d3accee9df3fbaca |
| mteb/sts15-sts | ae8e1ecbd3b3bcf5c1c8fd798290033715d98b93 |
| mteb/sts16-sts | d8e3be175a943eecfea09c96795087771996197c |
| mteb/stsbenchmark-sts | 96943a16ea6a35129e253c659081cb59daf81b30 |

The isolated phase-constant control uses source 24fc4b6368f107d987b9b8894908cfec7dbb3252, probe SHA256 1cf1ba909939b1b9f95f100fc87cfe9c21e7698c689658fb125ac522ca2985b9, folder frozen-phase-constant-20261106. It zeroes only returned lexical angles after computing the original active/end/record masks. Record identities and rho 2.18 remain unchanged, as do all learned tensors and other heads. It is a frozen development intervention, not a trained pointer-only control.

The final development lock was published at 2026-11-06T20:21:55.092271+00:00, Hub 161db5761ca44d66ef084ea880b64b79eccb8758, raw SHA256 b1eaef6968f7059e42a0db240c21549118cb07679369a88c2017b960f1a709b4. It selected full4M/N16 seed 1 before final test tables were read. The final results collection comes from cf83c503bdc3a8087f189a7fca2670248fe74fb8.
