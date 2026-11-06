# Engineering validation

These files are setup checks, **not trained-model experiment results**.

* `local_cuda_smoke.json`: full-width models, 256 tokens, batch 8, two optimization
  steps, both seeds, all five supported arms. All losses and gradients were finite.
  Largest observed total device usage: 4.44 decimal GB, against a 20 GB ceiling.
* `tokenizer_inference.json`: 32 real caption rows from the pinned public source,
  the real pinned BERT tokenizer, sentence-boundary and anchor-clamp checks, and
  untrained A2 inference. Outputs are finite, normalized 768-dimensional vectors.
* `colab_py313_wheels.json`: pinned package resolution using Linux CPython 3.13
  wheel tags. This is a cross-platform dependency check run from Windows; pip's
  environment markers are still evaluated on the host. It does not claim an
  executed Colab session or test Linux-only CUDA runtime dependencies.

The local environment is Python 3.12.10, PyTorch 2.11.0+cu128 on RTX 4090.
The Colab notebook creates an isolated environment for these same pinned package
versions, keeping current Colab host packages separate. A wheel and source archive
were built locally. CPU regression tests also exercise exact attention/gradient
parity, row ordering, resumability, benchmark completeness, and upload retries.

The four-chunk, two-epoch scientific screen is reported separately in the
[Hub experiment artifacts](https://huggingface.co/AbstractPhil/captionbert-8192-rotary).
Its trained-model results are distinct from these engineering checks; no quality
comparison or improvement claim follows from the setup checks alone.
