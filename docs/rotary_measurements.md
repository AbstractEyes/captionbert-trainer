# What the rotary screen measures

A0 is the required learned-position replication. A1 is plain token-index RoPE;
A2 splits 32 pairs into content 8, sentence-local slot 12 and sentence anchor 12;
A3 removes the anchor band while retaining 24 slot pairs. These are distinct
training arms, each with two seeds. The original held-out/STS evaluations answer
whether the changed positional representation helps the sentence encoder.

The frozen protocol adds mechanism measurements. It extends the implemented
`RopeMulti` and encoder using temporary hooks; it does not train a request head,
add an auxiliary loss, or implement the later key-binding/member systems.
The source experiments are archived in
[geolip-rotary-experiments](https://huggingface.co/AbstractPhil/geolip-rotary-experiments).

| Measurement | Question | Control and limit |
|---|---|---|
| Unit-content kernel | What resolution, side lobes and code collisions does the implemented frequency split produce? | Analytic code geometry, not learned-model attention |
| Simplex and explicit constant lift | What distinguishes the raw 25-code simplex from orthogonal anchors? | The encoder does not fix the constant coordinate required for the lift |
| Identical-content requested-address lookup | Can structural coordinates disambiguate repeated slots across compartments? | A3 has tied maxima; a plain RoPE given its exact global target also succeeds |
| Common slot/sentence shifts | Does a shared phase change cancel in q/k scores? | Report numerical residual beside the repeated frozen baseline |
| Sentence-ID permutation/collapse | Does the frozen encoder depend on the sentence stream? | A1/A3 have no anchor stream and should respond by zero |
| Single-sentence period alias | Does adding one complete anchor period to one group reproduce its phase? | Different from the collision created by clamping all later sentences at 24 |
| Uncapped sentence IDs | What happens when periodic codes replace the trained clamp on long probes? | An intervention, not evidence of usable extrapolation |
| Global versus sentence-reset slots | Is the output sensitive to the slot convention? | Same token IDs, order, mask, values and learned parameters |
| Content/slot/anchor q/k removal | How much does each band contribute to the frozen embedding? | Removes named q/k dimensions in every layer; values stay fixed; this is not a retrained ablation |
| Per-layer/head attention reconstruction | Where do band energy, score amplitude, within-sentence mass and repeated-slot leakage occur? | Explicit fp32 reconstruction, not native SDPA attention; opportunity counts accompany mass |

The record's exact fixed-position retrieval rotates a query to the address it
requests. This encoder rotates tokens to their own coordinates. It has no request
head. A successful oracle lookup or a large embedding response therefore does not
establish learned semantic retrieval, navigation, useful sentence meaning, AMOE
member protection, or task improvement.

The fixed probe texts include one to 32 short sentences and repeated content.
Reports retain their text, token IDs, clamped positions, maximum unclamped IDs,
checkpoint SHA-256, measurement source hash, device/dtype, repeated baseline,
all interventions and every layer/head. They use 16 texts by default and at most
four examples per forward. Head statistics are query-weighted means of per-batch
measurements; RMS ratios are not additive percentages.

Published task comparisons still need both seeds and a measured uncertainty bar.
The frozen numerical repeat characterizes this diagnostic call, not the training
recipe's hardware-repeat variation. No ratio is defined against a zero repeat.
Original synthetic retrieval experiments are referenced from their stored records;
they are not silently rerun or attributed to the CaptionBERT checkpoints.
