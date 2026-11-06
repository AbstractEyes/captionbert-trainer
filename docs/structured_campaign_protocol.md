# Structured CaptionBERT campaign: registered experiment

This protocol is fixed before scientific training. The earlier A0–A3 screen stays intact; its two-thousand-step warmup covered nearly its entire run. It does not decide whether the later whole-key mechanism improves CaptionBERT.

## Source and adaptation

The starting mechanism is the final entry form in the rotary line: whole-key phase binding at every address-writing position, a sixteen-pair lookup head, record-only key-pair norm 2.18, the hierarchical 8×3 address, the value-address path and lookup supervision. The source is [trainer v0.1.103](https://github.com/AbstractEyes/geolip-nano-beatrix-trainer/tree/3f174e3f391f983428171cefc6901b34b22274d8). Its final [published experiment](https://huggingface.co/AbstractPhil/geolip-rotary-experiments) documents **six seeds**, all forming the index at 512 addresses. Those results concern that causal byte model and its retrieval task.

CaptionBERT remains a full sentence encoder: width512, twelve layers, feed-forward2048, a512→512→768 projection and normalized masked-mean output. Structured arms use sixteen heads, preserving the source's32-dimensional head and its exact 8×3 frequencies. Relative controls retain both the original eight-head form and a sixteen-head NTK×4 form. Head count and NTK change together in that baseline comparison; it is not an isolated NTK experiment.

Natural text has no supplied record/index frames. Here a key is the exact WordPiece sequence of a word; its first occurrence's ending is a record, and later occurrences are potential queries. A fixed counter-generated phase per token ID and reverse offset replaces the byte table. All pieces contribute, with no length truncation. Equal words share the same phase; prefix or suffix agreement alone does not suffice. The phase is available at the ending or every piece, depending on the arm. Only the first occurrence's ending receives the pair-norm hold. This is an explicit tokenizer/task adaptation, not a reproduction of the original causal model.

Each token's structural address is its sentence number in three base-eight digits. Boundary punctuation belongs to the preceding sentence. Query/key streams are separate; the default caption input supplies each token's own address. Overflow past512 sentences is null, not an aliased address. Zero-initialized value-address tables carry the key stream's digits. No oracle request or teacher label enters inference. There is no address-writing decoder or modular-member claim for this encoder.

Source-function parity must show exact frequencies, separate query/key angles and record-only norm in fp32/fp64, plus a deliberately wrong rotation that fails. The WordPiece phase draw is declared different. Existing source research is cited within its scope; no transferred capability is assumed.

Padding also requires a numerical adaptation: clamp squared pair length at1e-12 before taking its square root. The forward norm floor stays1e-6, while zero padded pairs have finite derivatives. The source's dense byte rows do not establish that its padding-free training encountered this case.

## Data inspection

Before registration,4096 real rows from chunk000 were tokenized at256 tokens. Median length33,90th percentile118,99th178.1;272 rows exceeded128 and10 reached256. The sentence-count median was2,99th percentile9, maximum15. There were2026 rows with eligible repeated words,11120 supervised query positions, including707 continuation-piece queries. Exact token-tuple labels and the model's phase/record construction disagreed on zero queried positions in this sample. This is a descriptive first-row sample, not a representative estimate of the entire corpus.

Training pairs captions and frozen five-teacher consensus targets by exact row index, with the existing source revisions and hashes. Never join by image number or row divided by three. Screen chunks000–001 provide exactly1,000,000 rows. Chunk061 is development only. Chunk060 and the eight STS test splits stay out of architecture selection. STS-B validation uses revision96943a16ea6a35129e253c659081cb59daf81b30 and a64-token ceiling, matching the existing benchmark convention.

## Hypotheses and controls

| Question | Comparison | Evidence that would support it | Scope of a negative result |
|---|---|---|---|
| Does carrying a key across every subword help? | KALL versus KEND, neither with pointer loss | Better earlier-record attention on held-out multi-piece repeats; caption metrics reported separately | This WordPiece adaptation under the registered caption objective |
| Does direct lookup supervision make the mechanism useful? | KALLP versus KALL | Better exact-record mass with a positive sentence-development change | The chosen0.1 dose, lookup location and task |
| Does record-only norm impose a caption cost? | KALLP_N0 versus KALLP | Better sentence scores without a loss of useful key lookup | This hold length and model, not all normalization |
| Is lookup better placed later? | KALLP_L8 versus KALLP | A paired development improvement with preserved geometry | These two locations, not an optimal-depth law |
| Does structure help the encoder? | Best structured package versus relative controls | Paired development gains beyond measured repeat/seed noise, with held-out teacher/geometry viability | No automatic semantic claim from lookup success |

Changing the hierarchical bands and adding the value path is a package comparison (H16 versus N16), not an isolated attribution to one component. Pointer targets are exact token tuples and exclude the query's own occurrence. Common function words, nonalphabetic keys and words shorter than three characters receive no auxiliary target. The caption objective still trains on every input.

## Screen matrix and schedule

| Arm | Heads | Rotary/key binding | Pointer term | Other difference |
|---|---:|---|---:|---|
| R8 |8|plain global RoPE|0|relative baseline|
| N16 |16|NTK×4 global RoPE|0|relative baseline|
| H16 |16|hierarchical8×3|0|value-address path|
| KEND |16|hierarchy; key at endings|0|record-only norm2.18|
| KALL |16|hierarchy; key at every piece|0|same norm|
| KALLP |16|same complete binding|0.1|lookup at layer4, head0|
| KALLP_N0 |16|same complete binding|0.1|record norm disabled|
| KALLP_L8 |16|same complete binding|0.1|lookup at layer8, head0|

Each option uses seeds1 and2. N16 is repeated under both seeds to measure hardware repeat, for18 screen runs total. These explicit paired integer seeds are a recorded exception to the general run-name seed convention. Initial parameter draws, batches and objectives are matched where their shapes agree. All options use batch2048 without gradient accumulation,256-token ceiling, pure Adam at6e-4, clip1.0, symmetric InfoNCE at0.07 plus MSE, four epochs (1952 steps),200 warmup steps from1% and cosine to1e-6. Structured auxiliary supervision is added only where registered. Parameters remain fp32; CUDA autocast is bf16; TF32 is disabled.

Three independent lanes each run six models: R8/N16/N16-repeat; H16/KEND/KALLP_N0; KALL/KALLP/KALLP_L8. Each has an immutable configuration/data identity, per-arm/seed checkpoint and separate TensorBoard directory. Scalar logs include loss components, learning rate, epoch/step, gradient norm, throughput, memory and development measurements. TensorBoard event files, console/JSON logs, final/best/midpoint weights, optimizer/RNG/cursor state and environment are shipped. Interrupted runs resume compatible state; they do not restart completed work.

## Selection and larger experiments

Before opening tests, rank pairs by the mean STS-B validation Spearman. A structured pair is viable if both models are finite, its mean teacher cosine is within0.02 of the best relative pair and its mean effective rank is at least80% of that relative pair; no member may fall below50% of the corresponding control's rank. Report per-seed values, direction agreement and the measured noise bar (maximum hardware repeat and relevant seed spread). Ranking inside the noise bar is a descriptive choice, not a settled improvement. Near ties prefer the simpler/faster option; do not drop a weaker seed.

Promote the strongest viable structured pair and the strongest relative pair to exactly4,000,000 available paired rows: chunks000,001,002,003,004,006,009,010. Add a matched-exposure control: the selected structured architecture on1,000,000 rows for16 epochs. The4M recipes use four epochs; all three larger recipes are capped at7808 steps, batch2048 (15,990,784 examples),400 warmup steps and the same cosine horizon. This separates additional unique rows from merely taking more optimizer steps. It is not a model-width scaling experiment.

If no structured pair passes viability, explicitly label scaling the strongest finite structured option alongside the best relative baseline as an exploratory rescue, not a validated promotion. Do not force a favorable conclusion. Larger experiments use both registered seeds, for six final models. Their final weights are evaluated once on held-out chunk060 and the eight-task test harness. Architecture choice and root-seed choice use development metrics, never those tests.

After the larger runs, frozen development probes may disable key binding, disable the norm or change the fixed phase seed to distinguish equality lookup from dependence on a particular phase table. These are mechanism probes, not new trained improvements or evidence of a universally better rotary. Further training suggestions are written as hypotheses.

## Delivery and limits

The default deployed seed is the winning architecture's better **development** seed; both members remain available and all comparisons report the pair. Release AutoConfig/AutoModel source, tokenizer, complete config and `model.safetensors` through `save_pretrained(..., safe_serialization=True)`. Test isolated local and remote loading, absent/unexpected keys, safetensors metadata, padding/all-padding behavior, several batch/length shapes, finite values and parity with the training implementation. Publish raw tables, logs, parity/shape measurements, a plain-English report and updated model card. Preserve all earlier experiments.

Colab uses at most three G4/RTX PRO6000 runtimes with a92-decimal-GB total device ceiling. A full batch2048×256 forward/backward/Adam gate precedes training. Local engineering uses one4090 writer, allocator fraction at most0.73 and below20GB total; no local accuracy training. Estimated campaign envelope: about8–14 aggregate GPU hours and up to10 hours of active overnight operation including setup, analysis and publication, revised from measured throughput before the large stage. No hidden infinite queue or scheduled watcher. Local downloads stay on E: and within the available disk budget.

Long context is an engineering shape check only: training uses at most256 tokens. Neither the512-address source result nor an8192-position configuration proves8K semantic quality. Whole-key lookup, teacher reproduction, sentence capability, throughput and deployment parity are separate measurements.
