"""Strict row-index pairing, ragged token storage, and v2 batch planning."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import CAPTION_REPO, TARGET_REPO, TOKENIZER_REPO

CAPTION_REVISION = "4465c825e7f67abd5fdf1b681fb2052abbb06b82"
TARGET_REVISION = "566bfe623f68b0b3eebbe267ce8f98bd1248d0f8"
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
ADDRESS_VERSION = 1


def sentence_positions(ids, punctuation_ids, max_compartment=24):
    """Boundary punctuation belongs to the preceding sentence; next token resets slot."""
    positions = np.empty((len(ids), 2), dtype=np.int16)
    compartment, slot, maximum = 0, 0, 0
    for i, token in enumerate(ids):
        positions[i] = (slot, min(compartment, max_compartment))
        maximum = max(maximum, compartment)
        if int(token) in punctuation_ids:
            compartment += 1
            slot = 0
        else:
            slot += 1
    return positions, maximum >= max_compartment, maximum > max_compartment


def read_captions(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or any(not isinstance(row, str) for row in raw):
        raise ValueError("Expected the upstream flat list of caption strings; never join on row // 3")
    return raw


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(data_dir="data", chunks=(0, 1, 2, 3, 61), max_tokens=256):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION,
                                      cache_dir=str(root / "tokenizer-cache"))
    tok.save_pretrained(root / "tokenizer")
    punctuation = set(tok.convert_tokens_to_ids([".", "!", "?"]))
    for number, chunk in enumerate(dict.fromkeys(chunks), 1):
        start = time.monotonic()
        out = root / f"chunk_{chunk:03d}_L{max_tokens}"
        manifest = out / "manifest.json"
        expected = {"caption_revision": CAPTION_REVISION, "target_revision": TARGET_REVISION,
                    "tokenizer_revision": TOKENIZER_REVISION, "max_tokens": max_tokens,
                    "chunk": chunk, "address_version": ADDRESS_VERSION}
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Stale prepared cache: {out}")
            Store([out])  # Verify the cache before accepting it.
            print(f"Prepared chunk {chunk:03d} already verified", flush=True)
            continue
        print(f"Preparing {number}/{len(chunks)}: chunk {chunk:03d}; ~0.8 GB targets + captions", flush=True)
        print(f"  Fetching captions_{chunk:03d}.json (download or cache)...", flush=True)
        cap = hf_hub_download(CAPTION_REPO, f"captions_{chunk:03d}.json", repo_type="dataset",
                              revision=CAPTION_REVISION, local_dir=root / "source-captions")
        print(f"  Captions ready; fetching consensus_{chunk:03d}.pt (download or cache)...", flush=True)
        target = hf_hub_download(TARGET_REPO, f"consensus_{chunk:03d}.pt", repo_type="dataset",
                                 revision=TARGET_REVISION, local_dir=root / "source-targets")
        print(f"  Targets ready; validating row counts and tokenizing...", flush=True)
        captions = read_captions(cap)
        targets = torch.load(target, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(targets, torch.Tensor) or targets.shape != (len(captions), 768):
            raise ValueError(f"Chunk {chunk}: exact caption/target row count mismatch; refusing truncation")
        out.mkdir(parents=True, exist_ok=True)
        offsets, flat, addresses = [0], [], []
        reaches, overflows = 0, 0
        for i in range(0, len(captions), 20000):
            enc = tok(captions[i:i + 20000], max_length=max_tokens, truncation=True,
                      padding=False)["input_ids"]
            for ids in enc:
                pos, reach, overflow = sentence_positions(ids, punctuation)
                offsets.append(offsets[-1] + len(ids))
                flat.append(np.asarray(ids, dtype=np.uint16))
                addresses.append(pos)
                reaches += int(reach)
                overflows += int(overflow)
            elapsed = time.monotonic() - start
            done = min(i + 20000, len(captions))
            print(f"  {done:,}/{len(captions):,} rows; elapsed {elapsed:.0f}s; "
                  f"ETA {elapsed * (len(captions) - done) / done:.0f}s", flush=True)
        print(f"  Saving token cache and verifying file hashes: {out}", flush=True)
        np.save(out / "tokens.npy", np.concatenate(flat))
        np.save(out / "positions.npy", np.concatenate(addresses))
        np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        # Consensus stays in its downloaded file, memory-mapped by Store.
        expected.update(rows=len(captions), target_path=str(Path(target).resolve()),
                        target_sha256=sha256_file(target), caption_sha256=sha256_file(cap),
                        reaches_anchor_24=reaches, overflows_anchor_24=overflows,
                        prepared_sha256={name: sha256_file(out / name) for name in
                                         ("tokens.npy", "positions.npy", "offsets.npy")})
        atomic_json(manifest, expected)
        print(f"Chunk {chunk:03d} prepared: {len(captions):,} rows; "
              f"anchor 24 reached by {reaches:,}, exceeded by {overflows:,}; "
              f"elapsed {time.monotonic() - start:.1f}s", flush=True)


class Store:
    def __init__(self, directories):
        self.parts, all_lengths, self.starts = [], [], [0]
        self.manifests = []
        for directory in directories:
            d = Path(directory)
            meta = json.loads((d / "manifest.json").read_text())
            for name, digest in meta["prepared_sha256"].items():
                if sha256_file(d / name) != digest:
                    raise ValueError(f"Corrupt prepared file: {d / name}")
            target = Path(meta["target_path"])
            if sha256_file(target) != meta["target_sha256"]:
                raise ValueError(f"Corrupt consensus file: {target}")
            tokens = np.load(d / "tokens.npy", mmap_mode="r")
            pos = np.load(d / "positions.npy", mmap_mode="r")
            offsets = np.load(d / "offsets.npy", mmap_mode="r")
            targets = torch.load(target, map_location="cpu", weights_only=True, mmap=True)
            rows = meta["rows"]
            if len(offsets) != rows + 1 or targets.shape[0] != rows or len(pos) != len(tokens):
                raise ValueError("Prepared row alignment mismatch")
            if offsets[0] != 0 or offsets[-1] != len(tokens) or np.any(np.diff(offsets) <= 0):
                raise ValueError("Invalid ragged token offsets")
            self.parts.append((tokens, pos, offsets, targets))
            self.starts.append(self.starts[-1] + rows)
            all_lengths.append(np.diff(offsets).astype(np.int32))
            # Exclude local absolute paths from experiment identity.
            self.manifests.append({k: v for k, v in meta.items() if k != "target_path"})
        self.lengths = np.concatenate(all_lengths)
        self.starts = np.asarray(self.starts)

    def __len__(self):
        return int(self.starts[-1])

    def batch(self, indices):
        indices = np.asarray(indices)
        length = int(self.lengths[indices].max())
        ids = np.zeros((len(indices), length), dtype=np.int64)
        mask = np.zeros_like(ids)
        pos = np.zeros((len(indices), length, 2), dtype=np.int64)
        targets = []
        for row, index in enumerate(indices):
            part = int(np.searchsorted(self.starts, index, side="right") - 1)
            local = int(index - self.starts[part])
            tokens, positions, offsets, tg = self.parts[part]
            lo, hi = offsets[local:local + 2]
            ids[row, :hi - lo] = tokens[lo:hi]
            mask[row, :hi - lo] = 1
            pos[row, :hi - lo] = positions[lo:hi]
            targets.append(tg[local])
        return tuple(map(torch.from_numpy, (ids, mask, pos))) + (torch.stack(targets),)


def plan_batches(lengths, batch_size, seed, window_batches=64):
    # Exact v2 algorithm: shuffle rows, stable length sort within windows, shuffle batches.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(lengths))
    batches = []
    window = batch_size * max(window_batches, 1)
    for i in range(0, len(perm), window):
        win = perm[i:i + window]
        win = win[np.argsort(lengths[win], kind="stable")]
        batches.extend(win[j:j + batch_size] for j in range(0, len(win) - batch_size + 1, batch_size))
    rng.shuffle(batches)
    return batches


def open_store(cfg, chunks):
    return Store([Path(cfg.data_dir) / f"chunk_{c:03d}_L{cfg.max_tokens}" for c in chunks])
