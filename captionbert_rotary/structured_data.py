"""Exact token-tuple pointer targets, separate from the model's phase hashes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .data import Store, atomic_json, sha256_file


WORD_CACHE_VERSION = "first-whole-word-record-v1"
STOP_WORDS = frozenset("the and for are with this that from was were not but has have into its his her their there here they them these those you your our out can all one two who what when where which will would could should".split())


def tokenizer_metadata(tokenizer):
    vocabulary = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
    return vocabulary, [i for i, word in enumerate(vocabulary) if word.startswith("##")]


def exact_word_targets(ids, vocabulary, special_ids):
    """Later occurrences query the first word's ending, excluding their own span.

    Keys are exact tuples, never phase hashes. Stop words and words shorter than
    three characters receive no auxiliary supervision. Every input piece is
    still available to the ordinary caption objective and the binding.
    """
    result = np.full(len(ids), -1, dtype=np.int16)
    words, current = [], []
    for i, value in enumerate(ids):
        value = int(value)
        if value in special_ids:
            if current:
                words.append(current)
                current = []
            continue
        if not vocabulary[value].startswith("##") and current:
            words.append(current)
            current = []
        current.append((i, value))
    if current:
        words.append(current)
    records = {}
    for word in words:
        key = tuple(value for _, value in word)
        if key not in records:
            records[key] = word[-1][0]
            continue
        text = "".join(vocabulary[value].removeprefix("##") for _, value in word)
        if len(text) >= 3 and text not in STOP_WORDS and any(c.isalpha() for c in text):
            for position, _ in word:
                result[position] = records[key]
    return result


def prepare_word_targets(data_dir, chunks, max_tokens=256):
    from transformers import AutoTokenizer
    root = Path(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    vocabulary, _ = tokenizer_metadata(tokenizer)
    vocab_hash = hashlib.sha256(json.dumps(vocabulary).encode()).hexdigest()
    specials = set(tokenizer.all_special_ids)
    for chunk in dict.fromkeys(chunks):
        directory = root / f"chunk_{chunk:03d}_L{max_tokens}"
        base = json.loads((directory / "manifest.json").read_text())
        manifest_path = directory / "word_manifest.json"
        target_path = directory / "word_targets.npy"
        expected = dict(version=WORD_CACHE_VERSION, vocab_sha256=vocab_hash,
                        tokens_sha256=base["prepared_sha256"]["tokens.npy"],
                        offsets_sha256=base["prepared_sha256"]["offsets.npy"])
        if manifest_path.exists():
            meta = json.loads(manifest_path.read_text())
            if any(meta.get(k) != v for k, v in expected.items()) or sha256_file(target_path) != meta["targets_sha256"]:
                raise ValueError(f"Stale or corrupt whole-word target cache: {directory}")
            print(f"Whole-word targets already verified: chunk {chunk:03d}", flush=True)
            continue
        tokens = np.load(directory / "tokens.npy", mmap_mode="r")
        offsets = np.load(directory / "offsets.npy", mmap_mode="r")
        labels = np.full(len(tokens), -1, dtype=np.int16)
        rows_with_targets = queries = 0
        for row in range(len(offsets) - 1):
            lo, hi = offsets[row:row + 2]
            value = exact_word_targets(tokens[lo:hi], vocabulary, specials)
            labels[lo:hi] = value
            count = int((value >= 0).sum())
            queries += count
            rows_with_targets += int(count > 0)
            if (row + 1) % 50000 == 0:
                print(f"Whole-word targets chunk {chunk:03d}: {row + 1:,}/{len(offsets) - 1:,} rows", flush=True)
        with open(target_path.with_suffix(".tmp"), "wb") as handle:
            np.save(handle, labels)
        target_path.with_suffix(".tmp").replace(target_path)
        atomic_json(manifest_path, {**expected, "targets_sha256": sha256_file(target_path),
                    "rows": len(offsets) - 1, "rows_with_targets": rows_with_targets,
                    "supervised_query_positions": queries})
        print(f"Whole-word targets chunk {chunk:03d}: {rows_with_targets:,} rows, {queries:,} query positions", flush=True)


class StructuredStore(Store):
    def __init__(self, directories):
        directories = list(directories)
        super().__init__(directories)
        self.word_targets = []
        for directory, meta in zip(directories, self.manifests):
            path = Path(directory)
            word_meta = json.loads((path / "word_manifest.json").read_text())
            if sha256_file(path / "word_targets.npy") != word_meta["targets_sha256"]:
                raise ValueError("Corrupt whole-word targets")
            self.word_targets.append(np.load(path / "word_targets.npy", mmap_mode="r"))
            meta["word_targets"] = word_meta

    def batch(self, indices):
        batch = super().batch(indices)
        labels = np.full(batch[0].shape, -1, dtype=np.int64)
        for row, index in enumerate(indices):
            part = int(np.searchsorted(self.starts, index, side="right") - 1)
            local = int(index - self.starts[part])
            offsets = self.parts[part][2]
            lo, hi = offsets[local:local + 2]
            labels[row, :hi - lo] = self.word_targets[part][lo:hi]
        return batch + (torch.from_numpy(labels),)

