"""Read and verify the actual shipped TensorBoard events; CPU only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--census", default=str(ROOT / "docs/results/structured_artifact_census.json"))
    p.add_argument("--output", default=str(ROOT / "docs/results/structured_tensorboard_verification.json"))
    p.add_argument("--cache", default=str(ROOT / ".cache/tensorboard-events"))
    args = p.parse_args()
    census = json.loads(Path(args.census).read_text(encoding="utf8"))
    if census["status"] != "complete" or census["trained_models"] != 24:
        p.error("Verify the complete24-model artifact census first")
    files = [(key, run, name, meta) for key, run in census["runs"].items()
             for name, meta in run["files"].items() if "events.out.tfevents." in Path(name).name]
    if sum(meta["bytes"] for _, _, _, meta in files) > 256 << 20:
        p.error("Event downloads exceed the declared256MiB verification budget")
    from huggingface_hub import hf_hub_download
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    root = Path(args.cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hashes = {}
    for key, run, name, meta in files:
        destination = (root / "logs" / key / name).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("Invalid event destination")
        cached = hf_hub_download(census["repo"], run["prefix"] + name,
            revision=census["source_revision"], cache_dir=str(root / "hub"))
        raw = Path(cached).read_bytes()
        digest = hashlib.sha256(raw).hexdigest() if meta["lfs_sha256"] else hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if len(raw) != meta["bytes"] or digest != (meta["lfs_sha256"] or meta["git_blob_sha1"]):
            raise ValueError("An event file differs from the pinned artifact census")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        hashes[run["prefix"] + name] = hashlib.sha256(raw).hexdigest()

    def accumulator(path):
        return EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()

    def endpoint(reader, tag, step, expected):
        series = reader.Scalars(tag)
        if not series or series[-1].step != step or not math.isfinite(series[-1].value):
            raise ValueError(f"Missing/nonfinite final TensorBoard endpoint: {tag}")
        if abs(series[-1].value - expected) > 1e-6:
            raise ValueError(f"TensorBoard scalar differs from saved result: {tag}")
        return series[-1].value

    runs = {}
    for key, run in census["runs"].items():
        reader = accumulator(root / "logs" / key / "tensorboard")
        series = reader.Scalars("train/loss")
        if {x.step for x in series} != set(range(1, run["steps"]+1)):
            raise ValueError("Training-loss series does not cover every completed step: " + key)
        if any(not math.isfinite(x.value) for x in series):
            raise ValueError("Nonfinite TensorBoard training loss: " + key)
        endpoint(reader, "train/step", run["steps"], run["steps"])
        endpoint(reader, "final/stsb_development/spearman", run["steps"], run["development_spearman"])
        if not key.startswith("screen/"):
            for task in ("STS-B", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16", "BIOSSES"):
                tag = "final/test/tasks/" + task + "/spearman"
                if tag not in reader.Tags()["scalars"]:
                    raise ValueError("Missing final task TensorBoard scalar: " + key + "/" + task)
        frozen = {}
        for report in run["frozen_reports"]:
            folder = report["path"].split(run["prefix"], 1)[1].rsplit("/", 1)[0]
            if not any(name.startswith(folder + "/tensorboard/") for name in run["files"]):
                continue
            probe = accumulator(root / "logs" / key / folder / "tensorboard")
            observed = {}
            for condition, values in report["lookup_endpoints"].items():
                observed[condition] = {metric: endpoint(probe,
                    f"frozen/{condition}/lookup/all/{metric}", run["steps"], expected)
                    for metric, expected in values.items()}
            if report["initialization_lookup_endpoints"] is not None:
                observed["initialization"] = {metric: endpoint(probe,
                    f"frozen/initialization/lookup/all/{metric}", 0, expected)
                    for metric, expected in report["initialization_lookup_endpoints"].items()}
            frozen[folder] = observed
        runs[key] = dict(training_loss_points=len(series), unique_training_steps=run["steps"],
            last_training_step=series[-1].step, scalar_tags=reader.Tags()["scalars"],
            final_development_matches=True, frozen_endpoints=frozen)
        print("TensorBoard verified:", key, "through step", run["steps"], flush=True)
    report = dict(repo=census["repo"], artifact_revision=census["source_revision"],
        trained_models=24, event_files=len(files), downloaded_bytes=sum(meta["bytes"] for _, _, _, meta in files),
        file_sha256=hashes, runs=runs, seconds=time.monotonic()-started,
        training_performed=False, gpu_used=False, threads_limit=2,
        final_scalar_tolerance=1e-6, every_training_step_present=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")
    print("All shipped training and frozen TensorBoard endpoints verified:", output)


if __name__ == "__main__":
    main()
