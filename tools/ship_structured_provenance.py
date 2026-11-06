"""Preserve exact prepared-data identities and a bounded descriptive census.

CPU only, after the lane's optimizer and frozen queues. Reads cached tokens and
offsets, not model weights or consensus tensors. No training or new downloads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

TRAIN_CHUNKS = (0, 1, 2, 3, 4, 6, 9, 10)
SCREEN_DATA = "659b21c1c6eb1074eeec7acdbfa9bd010965595a4669338690cf3ae38a4448d5"
FULL_DATA = "7adb10f8833fab44e3ebb44315b53cb65bd8010b038115d15350b18536b57e86"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def histogram_summary(hist):
    total = int(hist.sum())
    cumulative = hist.cumsum()
    quantiles = {}
    for q in (.5, .9, .99, 1.):
        position = (total-1)*q
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        left, right = np.searchsorted(cumulative, (lo+1, hi+1))
        quantiles[str(q)] = float(left + (right-left)*(position-lo))
    return dict(rows=total, mean=float(hist @ np.arange(len(hist)) / total), quantiles=quantiles)


def maximum_segment_address(tokens, offsets, punctuation_ids):
    punctuation = tokens == punctuation_ids[0]
    for token in punctuation_ids[1:]:
        punctuation |= tokens == token
    result = np.add.reduceat(punctuation, offsets[:-1], dtype=np.int32)
    # Boundary tokens remain in the preceding segment.
    result -= punctuation[offsets[1:] - 1]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="/content/captionbert-data")
    p.add_argument("--root", default="/content/captionbert-structured")
    p.add_argument("--lane", choices=("A", "B"), default="A")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    data, root = Path(args.data), Path(args.root)
    seed = {"A": 1, "B": 2}[args.lane]
    for stage, arm in (("full4m", "H16"), ("full4m", "N16"), ("exposure1m", "H16")):
        result = json.loads((root / stage / f"{arm}-seed{seed}/result.json").read_text())
        if result["status"] != "complete" or result["steps"] != 7808:
            raise ValueError("Finish the registered optimizer queue before this descriptive census")
    config = json.loads((root / f"full4m/H16-seed{seed}/config.json").read_text())
    punctuation_ids = config["punctuation_ids"]
    chunks = (*TRAIN_CHUNKS, 60, 61)
    manifests, words = {}, {}
    source_bytes = 0
    for chunk in chunks:
        folder = data / f"chunk_{chunk:03d}_L256"
        raw = json.loads((folder / "manifest.json").read_text())
        manifests[chunk] = {k: v for k, v in raw.items() if k != "target_path"}
        if raw["max_tokens"] != 256 or raw["rows"] != 500000:
            raise ValueError("Unexpected input ceiling or row count")
        if chunk in TRAIN_CHUNKS:
            words[chunk] = json.loads((folder / "word_manifest.json").read_text())
        source_bytes += sum((folder / name).stat().st_size for name in ("tokens.npy", "offsets.npy"))
        if (folder / "tokens.npy").stat().st_size > 128 << 20:
            raise ValueError("A token chunk exceeds the declared128MiB per-file CPU scan budget")
    if source_bytes > 2 << 30:
        raise ValueError("Prepared token/offset inputs exceed the declared2GiB scan budget")
    validation = [manifests[61]]
    def dataset_id(selected):
        training = [{**manifests[chunk], "word_targets": words[chunk]} for chunk in selected]
        return digest({"train": training, "validation": validation})
    identities = dict(screen=dataset_id((0, 1)), full4m=dataset_id(TRAIN_CHUNKS))
    if identities != dict(screen=SCREEN_DATA, full4m=FULL_DATA):
        raise ValueError("Prepared manifests do not reconstruct the actual training data identities")
    stats = {}
    training_length_hist = np.zeros(257, dtype=np.int64)
    training_address_hist = np.zeros(257, dtype=np.int64)
    for chunk in chunks:
        folder = data / f"chunk_{chunk:03d}_L256"
        tokens = np.load(folder / "tokens.npy", mmap_mode="r")
        offsets = np.load(folder / "offsets.npy", mmap_mode="r")
        lengths = np.diff(offsets)
        max_address = maximum_segment_address(tokens, offsets, punctuation_ids)
        length_hist = np.bincount(lengths, minlength=257)
        address_hist = np.bincount(max_address, minlength=257)
        stats[chunk] = dict(length=histogram_summary(length_hist),
            maximum_segment_address=histogram_summary(address_hist),
            rows_above128=int(length_hist[129:].sum()), rows_at256=int(length_hist[256]),
            rows_reaching_address8=int(address_hist[8:].sum()),
            rows_reaching_address64=int(address_hist[64:].sum()),
            length_histogram=length_hist.tolist(), maximum_address_histogram=address_hist.tolist())
        if chunk in TRAIN_CHUNKS:
            training_length_hist += length_hist
            training_address_hist += address_hist
        print("Descriptive corpus census:", chunk, "rows", len(lengths), flush=True)
        del tokens, offsets, lengths, max_address
    training = dict(rows=int(training_length_hist.sum()),
        length=histogram_summary(training_length_hist),
        maximum_segment_address=histogram_summary(training_address_hist),
        rows_above128=int(training_length_hist[129:].sum()), rows_at256=int(training_length_hist[256]),
        rows_reaching_address8=int(training_address_hist[8:].sum()),
        rows_reaching_address64=int(training_address_hist[64:].sum()),
        rows_with_pointer_targets=sum(words[x]["rows_with_targets"] for x in TRAIN_CHUNKS),
        supervised_query_positions=sum(words[x]["supervised_query_positions"] for x in TRAIN_CHUNKS))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / ("prepared-provenance-" + args.lane + "-" + stamp)
    output.mkdir()
    report = dict(created_utc=stamp, lane=args.lane, data_identities=identities,
        training_chunks=TRAIN_CHUNKS, prepared_manifests=manifests, word_manifests=words,
        per_chunk=stats, full4m_training=training, source_token_offset_bytes=source_bytes,
        punctuation_ids=punctuation_ids, training_performed=False, gpu_used=False,
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        scope="Descriptive post-training corpus census; valid-token punctuation addresses, not linguistic sentence counts",
        identity_note="Matches Store: absolute target_path excluded; training adds the exact word_targets manifest",
        cache_note="Legacy positions.npy uses a24-address clamp; the structured encoder recomputes its512-address stream from tokens")
    (output / "provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    shutil.copyfile(__file__, output / "source.py")
    if args.push:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id="AbstractPhil/captionbert-8192-rotary", folder_path=output,
            path_in_repo=f"campaigns/structured-20261106/data-provenance/lane-{args.lane}/{stamp}",
            commit_message="Preserve prepared-data identities and full-corpus rotary address usage")
    print("Prepared-data provenance preserved:", output, flush=True)
    print(json.dumps(training, indent=2), flush=True)


if __name__ == "__main__":
    main()
