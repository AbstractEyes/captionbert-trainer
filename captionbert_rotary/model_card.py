"""Data-driven CaptionBERT model card; incomplete screens remain local drafts."""
import json
import math
from pathlib import Path
from .data import atomic_json
from .diagnostics import PROTOCOL

TASK_NAMES = ("STS-B", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16", "BIOSSES")
FORMAT_REFERENCE = "https://huggingface.co/AbstractPhil/captionbert-8192-v2/blob/ed468fe5aed64a3d03a56a7f7811131d20f81ddb/README.md"


def number(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "n/a"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def render_model_card(results, diagnostics, config, repo, experiment, revision, checkpoint_hashes=None):
    arms, seeds = config.get("arms", ["A0", "A1", "A2", "A3"]), config.get("seeds", [1, 2])
    expected = [(arm, seed) for arm in arms for seed in seeds]
    missing, problems = [], []
    for key in expected:
        name = f"{key[0]}-seed{key[1]}"
        result = results.get(key)
        if result is None:
            missing.append(name)
            continue
        if (result.get("status") != "complete" or result.get("config_id") != experiment
                or result.get("arm") != key[0] or result.get("seed") != key[1]):
            problems.append(name + ": incomplete or wrong experiment")
        benchmark = result.get("benchmark") or {}
        tasks = benchmark.get("tasks", {})
        valid_tasks = set(tasks) == set(TASK_NAMES) and all(
            isinstance(tasks[t], dict) and finite(tasks[t].get("spearman")) for t in tasks)
        if not valid_tasks:
            problems.append(name + ": eight-task benchmark missing or nonfinite")
        if valid_tasks:
            mean = sum(tasks[t]["spearman"] for t in TASK_NAMES) / len(TASK_NAMES)
            if not finite(benchmark.get("mean")) or abs(mean - benchmark["mean"]) > 1e-6:
                problems.append(name + ": benchmark mean differs from task scores")
        if benchmark.get("max_tokens") != 64 or benchmark.get("split") != "test" or not benchmark.get("dataset_revisions"):
            problems.append(name + ": benchmark protocol or revision evidence missing")
        for slice_name in ("heldout", "above_128_tokens"):
            values = result.get("validation", {}).get(slice_name, {})
            for metric in ("cos_to_target", "erank", "self_cos"):
                if not finite(values.get(metric)):
                    problems.append(name + ": missing/nonfinite " + slice_name + " " + metric)
        report = diagnostics.get(key, {})
        identity = report.get("identity", {})
        if (report.get("protocol") != PROTOCOL or report.get("status") != "measured" or not report.get("baseline_finite")
                or identity.get("arm") != key[0] or identity.get("seed") != key[1]
                or identity.get("config_id") != experiment or identity.get("step") != result.get("steps")
                or identity.get("dataset_id") != result.get("dataset_id")
                or not (checkpoint_hashes or {}).get(key)
                or identity.get("checkpoint_sha256") != (checkpoint_hashes or {}).get(key)):
            problems.append(name + ": frozen rotary measurements missing or mismatched")
        if not finite(report.get("repeat_floor", {}).get("max_absolute_change")):
            problems.append(name + ": missing frozen repeat measurement")
        if key[0] != "A0" and any(not finite(report.get("interventions", {}).get(case, {}).get("mean_l2_change"))
                                 for case in ("sentence_ids_collapsed", "remove_anchor")):
            problems.append(name + ": incomplete rotary intervention measurements")
    probes = {r.get("probe_sha256") for r in diagnostics.values()}
    if len(probes) != 1 or None in probes:
        problems.append("Frozen probe identities differ or are missing")
    if set(seeds) != {1, 2} or not {"A0", "A1", "A2", "A3"}.issubset(arms):
        problems.append("The registered four-arm/two-seed screen is required for the final card")
    if len({r.get("steps") for r in results.values()}) > 1:
        problems.append("Training steps differ across arms/seeds")
    if len({r.get("dataset_id") for r in results.values()}) > 1:
        problems.append("Data identity differs across arms/seeds")
    if len({json.dumps((r.get("benchmark") or {}).get("dataset_revisions"), sort_keys=True) for r in results.values()}) > 1:
        problems.append("Benchmark dataset revisions differ across arms/seeds")
    if any(not r.get("dataset_id") or not isinstance(r.get("steps"), int) or r["steps"] <= 0 for r in results.values()):
        problems.append("Missing data identity or positive training step count")
    ready = not missing and not problems
    lines = ["---", "license: mit", "language: en", "library_name: pytorch", "pipeline_tag: sentence-similarity",
             "tags:", "- captionbert", "- rotary", "- sentence-similarity", "- consensus-distillation",
             "datasets:", "- AbstractPhil/conceptual-captions-12m-webdataset-berts",
             "- AbstractPhil/captionbert-8192-v2-consensus", "---", "", "# captionbert-8192-rotary", ""]
    if not ready:
        lines += ["> **Incomplete draft.** Final publication is gated on every arm/seed, all eight task scores, "
                  "and matching frozen rotary measurements. This draft is not published automatically.", ""]
    lines += ["A controlled positional screen of CaptionBERT v2: the learned-position replication (A0), "
              "plain token RoPE (A1), sentence-address rotary (A2), and its anchor ablation (A3). "
              "Every arm is trained from scratch against the same precomputed five-teacher consensus. "
              "Inference uses the tokenizer and one student; teacher models are not loaded.", "",
              "This is a short screen, separate from the original v2 full training run. "
              "Both seeds and all arms, including weaker results, are retained. No single winner is selected automatically.", "",
              "## Loading", "", "Install the tagged package in an isolated Python 3.11–3.13 environment:", "",
              "```bash", "pip install git+https://github.com/AbstractEyes/captionbert-trainer.git@v0.2.0", "```", "",
              "```python", "from captionbert_rotary.inference import SentenceEncoder",
              f'encoder = SentenceEncoder.from_pretrained("{repo}",',
              f'    run="experiments/{experiment}/A2-seed1",',
              f'    revision="{revision}")',
              'embeddings = encoder.encode(["A cat watches the street.", "A person walks nearby."])',
              "# embeddings: (2, 768), L2-normalized", "```", "",
              "Choose an explicit arm and seed. A2-seed1 above is a loading example, not a claim that it is best. "
              "Model code, requirements, tests and notebooks live in "
              "[captionbert-trainer](https://github.com/AbstractEyes/captionbert-trainer/tree/v0.2.0).", "",
              "The [technical companion](https://github.com/AbstractEyes/captionbert-trainer/blob/v0.2.0/docs/rotary_measurements.md) "
              "describes the measurement controls, report fields and limits.", "",
              "## Benchmark", "", "Spearman correlation on the same eight test tasks as the v2 harness; "
              "mean is the unweighted task average. The benchmark uses its original 64-token ceiling.", "",
              "| arm / seed | " + " | ".join(TASK_NAMES) + " | mean |",
              "|---|" + "---:|" * (len(TASK_NAMES) + 1)]
    for arm, seed in expected:
        result = results.get((arm, seed), {})
        bench = result.get("benchmark") or {}
        values = [number(bench.get("tasks", {}).get(t, {}).get("spearman")) for t in TASK_NAMES]
        lines.append(f"| {arm} / {seed} | " + " | ".join(values) + " | " + number(bench.get("mean")) + " |")
    lines += ["", "The published full-run v2 mean was 0.6077 after 26.9M rows and 52,548 steps. "
              "That is historical context, not the matched control for this screen; the matched control is A0 here.", "",
              "## Geometry", "", "Effective rank is the participation ratio of centered variance. "
              "Self-cosine is the mean off-diagonal cosine, an isotropy readout. "
              "Cosine to consensus measures mimicry, separate from task capability.", "",
              "The following caption-domain measurements use the fixed held-out chunk-61 sample, "
              "separately from the benchmark sentences. The >128-token slice is sampled separately from that chunk.", "",
              "| arm / seed | parameters | steps | held-out cosine | effective rank | self-cosine | seconds/step |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, seed in expected:
        r = results.get((arm, seed), {})
        held = r.get("validation", {}).get("heldout", {})
        long = r.get("validation", {}).get("above_128_tokens", {})
        params = f"{r['parameters']:,}" if "parameters" in r else "n/a"
        lines.append(f"| {arm} / {seed} | {params} | {r.get('steps', 'n/a')} | "
                     f"{number(held.get('cos_to_target'))} | {number(held.get('erank'), 2)} | "
                     f"{number(held.get('self_cos'))} | "
                     f"{number(r.get('seconds_per_step'), 3)} |")
    lines += ["", "| arm / seed | >128-token cosine | effective rank | self-cosine | measured rows |",
              "|---|---:|---:|---:|---:|"]
    for arm, seed in expected:
        long = results.get((arm, seed), {}).get("validation", {}).get("above_128_tokens", {})
        lines.append(f"| {arm} / {seed} | {number(long.get('cos_to_target'))} | "
                     f"{number(long.get('erank'), 2)} | {number(long.get('self_cos'))} | {long.get('n', 'n/a')} |")
    lines += ["", "## Rotary measurements", "",
              "The fixed frozen-model protocol measures address kernels, simplex coherence and its explicit constant lift, "
              "periodic aliases versus the sentence clamp, identical-content oracle address lookup, "
              "controlled address changes, q/k-band removals, and per-layer/head energy and attention-mass changes.", "",
              "The encoder rotates tokens to their own coordinates. It has no learned request head or AMOE member gate. "
              "Code-only oracle lookup and embedding sensitivity are not learned semantic-retrieval accuracy or task improvement. "
              "The raw anchor code is a regular simplex; the encoder does not fix the constant coordinate of the orthogonal lift.", "",
              "| arm / seed | repeated baseline max error | sentence-ID collapse, mean L2 change | anchor q/k removal, mean L2 change |",
              "|---|---:|---:|---:|"]
    for arm, seed in expected:
        d = diagnostics.get((arm, seed), {})
        cases = d.get("interventions", {})
        lines.append(f"| {arm} / {seed} | {number(d.get('repeat_floor', {}).get('max_absolute_change'), 8)} | "
                     f"{number(cases.get('sentence_ids_collapsed', {}).get('mean_l2_change'), 6)} | "
                     f"{number(cases.get('remove_anchor', {}).get('mean_l2_change'), 6)} |")
    lines += ["", "A0 has no rotary bands (n/a). A1/A3 have no anchor band, so anchor-removal effects are structural negative controls. "
              "Full probe inputs, checkpoint/source hashes, dtype/device, all interventions and every layer/head are in each run's "
              "`rotary_diagnostics.json` or additive `analyses/captionbert-rotary-mechanisms-v1/` report. "
              "The frozen repeat measures diagnostic-call precision, not training hardware-repeat uncertainty.", "",
              "## How it was built", "",
              "| part | screen recipe |", "|---|---|",
              f"| arms / seeds | {', '.join(arms)} / {', '.join(map(str, seeds))} |",
              f"| training chunks | {config.get('train_chunks', [0, 1, 2, 3])} |",
              f"| held-out chunk | {config.get('val_chunk', 61)} |",
              f"| epochs / contrastive batch | {config.get('epochs', 2)} / {config.get('batch_size', 2048)} |",
              f"| optimizer | pure Adam, weight decay zero; peak LR {config.get('lr', 6e-4)} |",
              f"| schedule | {config.get('warmup_steps', 2000)}-step linear warmup from 1%, then cosine |",
              f"| training token ceiling | {config.get('max_tokens', 256)} |",
              "| objective / pooling | symmetric InfoNCE (T=0.07) + MSE; masked mean; L2-normalized 768-d output |", "",
              "Captions and consensus tensors are paired by flat row index, with exact row-count checks. "
              "Sentence indices advance after `.`, `!` or `?`; the next token resets its sentence-local slot. "
              "A2 has content 8, slot 12 (wavelengths 3–256) and anchor 12 pairs per 64-d head; "
              "A3 has content 8 and slot 24. A1 uses all 32 pairs on global token index.", "",
              "## Known limits", "",
              "- The short screen spends most of its approximately 2,000 steps in the original 2,000-step warmup. "
              "It does not reproduce v2's full-run scale.",
              "- Training is capped at 256 tokens. The model name and A0's 8,192-position table do not establish 8k-context quality.",
              "- Sentence IDs clamp at 24. Unclamped rotary codes repeat every 25 addresses; neither is unlimited sentence capacity.",
              "- Caption-domain consensus agreement and STS capability are separate measurements. These five teachers also constrain the target's shared geometry.",
              "- Paired seeds expose seed dependence; a settled improvement additionally needs the applicable uncertainty/repeat evidence. "
              "A large band-removal response alone does not establish useful structure.", "",
              "## Output convention", "",
              "The package's `SentenceEncoder.encode` returns a `(batch, 768)` normalized tensor. "
              "This repository contains multiple explicit run artifacts; it is not a root-level Transformers `AutoModel` checkpoint.", "",
              "## Provenance", "",
              f"Experiment `{experiment}`; result files read at Hub revision `{revision}`. "
              f"Model-card organization follows the [v2 card]({FORMAT_REFERENCE}). "
              "Original results, every arm/seed, weights and analysis JSON remain under their immutable experiment paths.", "",
              "## Citation", "", "```bibtex", "@misc{captionbert_rotary_2026,",
              "  title = {CaptionBERT Sentence-Address Rotary Screen},", "  author = {AbstractPhil},",
              f"  year = {{2026}},", f"  url = {{https://huggingface.co/{repo}}}", "}", "```", ""]
    return "\n".join(lines), {"ready": ready, "missing_runs": missing, "problems": problems,
                              "format_reference": FORMAT_REFERENCE, "source_revision": revision}


def finalize_model_card(repo, experiment, output_dir="runs/model-card", push=False):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.model_info(repo, files_metadata=True)
    assets = {f.rfilename: f for f in info.siblings}
    files = set(assets)
    prefix = f"experiments/{experiment}/"
    config, results, diagnostics, checkpoint_hashes = {}, {}, {}, {}
    for filename in sorted(files):
        if not filename.startswith(prefix) or not filename.endswith("/result.json"):
            continue
        name = filename[len(prefix):].split("/")
        if len(name) != 2:
            continue
        result = json.loads(Path(hf_hub_download(repo, filename, revision=info.sha)).read_text())
        key = result["arm"], result["seed"]
        results[key] = result
        lfs = getattr(assets.get(prefix + name[0] + "/model.pt"), "lfs", None)
        checkpoint_hashes[key] = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        config_path = prefix + name[0] + "/experiment.json"
        if not config and config_path in files:
            config = json.loads(Path(hf_hub_download(repo, config_path, revision=info.sha)).read_text())["config"]
        candidates = (prefix + name[0] + "/analyses/captionbert-rotary-mechanisms-v1/rotary_diagnostics.json",
                      prefix + name[0] + "/rotary_diagnostics.json")
        for candidate in candidates:
            if candidate in files:
                diagnostics[key] = json.loads(Path(hf_hub_download(repo, candidate, revision=info.sha)).read_text())
                break
    text, readiness = render_model_card(results, diagnostics, config, repo, experiment, info.sha, checkpoint_hashes)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "README.md"
    path.write_text(text, encoding="utf-8")
    readiness["published"] = False
    if push and readiness["ready"]:
        commit = api.upload_file(repo_id=repo, repo_type="model", path_or_fileobj=str(path),
                                 path_in_repo="README.md", parent_commit=info.sha,
                                 commit_message=f"Document completed positional screen {experiment[:8]}")
        readiness.update(published=True, published_revision=commit.oid)
    atomic_json(root / "readiness.json", readiness)
    print(f"Model card: {'published' if readiness['published'] else 'ready locally' if readiness['ready'] else 'incomplete draft'}; "
          f"{len(readiness['missing_runs'])} missing runs, {len(readiness['problems'])} incomplete checks; {path}", flush=True)
    return readiness
