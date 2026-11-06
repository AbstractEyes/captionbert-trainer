"""Same eight task IDs, test splits, 64-token ceiling and gauges as v2."""
import numpy as np
import json
import time
from pathlib import Path
import torch
from scipy.stats import spearmanr
from .data import sentence_positions, atomic_json
from .metrics import geometry
from .model import embed

TASKS = (("STS-B", "mteb/stsbenchmark-sts"), ("SICK-R", "mteb/sickr-sts"),
         ("STS12", "mteb/sts12-sts"), ("STS13", "mteb/sts13-sts"),
         ("STS14", "mteb/sts14-sts"), ("STS15", "mteb/sts15-sts"),
         ("STS16", "mteb/sts16-sts"), ("BIOSSES", "mteb/biosses-sts"))


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device="cpu", batch_size=32, max_tokens=64):
    was_training = model.training
    model.eval()
    output = []
    punctuation = set(tokenizer.convert_tokens_to_ids([".", "!", "?"]))
    try:
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], max_length=max_tokens, truncation=True,
                              padding=True, return_tensors="pt")
            pos = np.zeros((*batch["input_ids"].shape, 2), dtype=np.int64)
            for row, (ids, mask) in enumerate(zip(batch["input_ids"], batch["attention_mask"])):
                n = int(mask.sum())
                pos[row, :n] = sentence_positions(ids[:n].tolist(), punctuation)[0]
            with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                value = embed(model, batch["input_ids"].to(device), batch["attention_mask"].to(device),
                              torch.from_numpy(pos).to(device))
            output.append(value.float().cpu())
        return torch.cat(output)
    finally:
        model.train(was_training)


def run_benchmark(model, tokenizer, cfg):
    started = time.monotonic()
    print(f"Benchmark: {len(TASKS)} tasks; 64-token ceiling; batch {cfg.eval_batch}", flush=True)
    from datasets import load_dataset
    from huggingface_hub import HfApi
    path = Path(cfg.output_dir) / "benchmark_revisions.json"
    if path.exists():
        pinned = json.loads(path.read_text())
    else:
        pinned = {repo: HfApi().dataset_info(repo).sha for _, repo in TASKS}
        atomic_json(path, pinned)
    scores, revisions = {}, {}
    for task_number, (name, repo) in enumerate(TASKS, 1):
        print(f"Benchmark {task_number}/{len(TASKS)} {name}: loading test split...", flush=True)
        # A failed task fails evaluation; never report a partial suite as an eight-task mean.
        ds = load_dataset(repo, split="test", revision=pinned[repo])
        columns = ds.column_names
        a = "sentence1" if "sentence1" in columns else columns[0]
        b = "sentence2" if "sentence2" in columns else columns[1]
        gold = "score" if "score" in columns else "similarity_score"
        print(f"Benchmark {name}: encoding {len(ds[a]):,} sentence pairs...", flush=True)
        ea = encode_texts(model, tokenizer, list(ds[a]), cfg.device, cfg.eval_batch)
        eb = encode_texts(model, tokenizer, list(ds[b]), cfg.device, cfg.eval_batch)
        cosine = torch.nn.functional.cosine_similarity(ea, eb).numpy()
        corr = float(spearmanr(cosine, np.asarray(ds[gold], dtype=float)).statistic)
        if not np.isfinite(corr):
            raise ValueError(f"Nonfinite benchmark score for {name}")
        scores[name] = {"spearman": corr, **geometry(torch.cat((ea, eb))[:2000])}
        revisions[name] = ds._fingerprint
        print(f"Benchmark {task_number}/{len(TASKS)} {name}: Spearman {corr:.4f}; "
              f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    print(f"Benchmark complete: mean {np.mean([r['spearman'] for r in scores.values()]):.4f}; "
          f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    return {"tasks": scores, "mean": float(np.mean([r["spearman"] for r in scores.values()])),
            "dataset_fingerprints": revisions, "dataset_revisions": pinned,
            "max_tokens": 64, "split": "test"}
