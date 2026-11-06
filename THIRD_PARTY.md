# Reference provenance

`reference/pretrain_trainer.py`, `modeling_captionbert.py`, `full_benchmark.py`,
`early_eval.py`, and `config.json` were downloaded unchanged from
[AbstractPhil/captionbert-8192-v2](https://huggingface.co/AbstractPhil/captionbert-8192-v2),
revision `ed468fe5aed64a3d03a56a7f7811131d20f81ddb`.
The model card declares MIT; the repository does not contain a separate LICENSE file.
Copyright belongs to AbstractPhil. The extracted `CaptionEncoder` class in
`captionbert_rotary/_v2.py` is verbatim and checked against that source by tests.

The caption dataset is CC BY-SA 4.0, as identified in the supplied build brief.
Downloaded captions, consensus targets, and trained weights are excluded from Git.
Source data revisions and file checksums are recorded in every prepared manifest.

## Structured rotary source

The structured address layout, whole-key binding and record-pair norm follow
[geolip-nano-beatrix-trainer v0.1.103](https://github.com/AbstractEyes/geolip-nano-beatrix-trainer/tree/3f174e3f391f983428171cefc6901b34b22274d8)
by AbstractPhil. The CaptionBERT implementation explicitly adapts its causal byte
keys/record frames to bidirectional WordPiece inputs and natural repeated-word
records, and changes the squared-norm floor for finite padded derivatives.
The [technical companion](docs/TECHNICAL_STRUCTURED.md) specifies the changes
and the scope of the source result. This model is not the source byte checkpoint.
