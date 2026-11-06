"""Seeded stage-3 screen; checkpoints include optimizer, cursor and all RNG states."""
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import importlib.metadata
import time
import numpy as np
import torch
from .config import ScreenConfig
from .data import atomic_json, open_store, plan_batches
from .hub import Uploader, pull_completed
from .memory import MemoryGuard
from .metrics import evaluate, objective
from .model import build_model, embed

PREDICTIONS = {
    "A0": "Reference at screen scale; no claim that two epochs reproduce full v2 scores.",
    "A1": "Similar to or slightly above A0 on >128 tokens; relative positions may help generalization.",
    "A2": "Above A0 and A3 on long multi-sentence captions if sentence identity helps consensus distillation.",
    "A3": "Near or below A0, and below A2, on multi-sentence captions: slots repeat without a sentence identity band.",
    "A2g": "Similar to or slightly above A0 on long captions; near A2, with less slot aliasing but weaker sentence alignment.",
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    ns = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (ns[0], torch.from_numpy(ns[1].astype(np.int64)), ns[2], ns[3], ns[4]),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(rng):
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"].cpu())
    name, array, index, gaussian, cached = rng["numpy"]
    np.random.set_state((name, array.cpu().numpy().astype(np.uint32), index, gaussian, cached))
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in rng["cuda"]])


def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def make_scheduler(opt, cfg, total):
    # Preserve the v2 schedule literally. A ~2k-step screen spends almost all its time warming up.
    return torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, cfg.warmup_steps),
              torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=max(total - cfg.warmup_steps, 1), eta_min=cfg.min_lr)],
        milestones=[cfg.warmup_steps])


def optimization_step(model, opt, scaler, batch, cfg):
    ids, mask, pos, target = (x.to(cfg.device) for x in batch)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.device.startswith("cuda")):
        value = embed(model, ids, mask, pos)
    loss, nce, mse = objective(value, target, cfg.temperature)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient norm")
    scaler.step(opt)
    scaler.update()
    return {"loss": float(loss.detach()), "nce": float(nce.detach()), "mse": float(mse.detach()),
            "grad_norm": float(norm)}


def vram_probe(model, cfg, guard):
    """Exercise worst-case length AND Adam moments without changing weights or RNG."""
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    rng = capture_rng()
    batch = cfg.batch_size
    ids = torch.randint(1, cfg.model.vocab_size, (batch, cfg.max_tokens))
    mask = torch.ones_like(ids)
    pos = torch.zeros(batch, cfg.max_tokens, 2, dtype=torch.long)
    pos[..., 0] = torch.arange(cfg.max_tokens)
    target = torch.randn(batch, cfg.model.output_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-9)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.device.startswith("cuda"))
    try:
        optimization_step(model, opt, scaler, (ids, mask, pos, target), cfg)
        guard.check()
        print(f"Worst-case probe passed: batch {batch} x {cfg.max_tokens} tokens", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("Worst-case batch exceeds the VRAM allowance. Use the Colab G4 for the "
                           "2048-negative screen, or explicitly reduce batch_size for diagnostics. "
                           "No silent batch reduction or gradient accumulation is applied.") from exc
    finally:
        opt.zero_grad(set_to_none=True)
        model.load_state_dict(state)
        restore_rng(rng)
        del opt, scaler, state
        gc.collect()
        if guard.enabled:
            torch.cuda.empty_cache()


def train_arm(cfg, arm, seed, train, val, dataset_id, guard, probe=True, on_step=None):
    name = f"{arm}-seed{seed}"
    directory = Path(cfg.output_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"config_id": cfg.fingerprint(), "dataset_id": dataset_id, "arm": arm, "seed": seed}
    if (directory / "result.json").exists():
        result = json.loads((directory / "result.json").read_text())
        if any(result.get(k) != v for k, v in identity.items()):
            raise ValueError(f"Refusing incompatible completed run: {name}")
        return result
    seed_all(seed)
    model = build_model(arm, cfg.model).to(cfg.device)
    model.train()
    steps_per_epoch = len(plan_batches(train.lengths, cfg.batch_size, seed, cfg.bucket_window))
    if not steps_per_epoch:
        raise ValueError("Fewer rows than one full contrastive batch")
    total = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total = min(total, cfg.max_steps)
    print(f"{name}: {len(train):,} rows x {cfg.epochs} epochs; {total:,} steps; "
          f"{sum(p.numel() for p in model.parameters()):,} params; batch {cfg.batch_size}", flush=True)
    if probe:
        vram_probe(model, cfg, guard)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # NO weight decay, NO AdamW
    scheduler = make_scheduler(opt, cfg, total)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.device.startswith("cuda"))
    step, next_epoch, next_batch, elapsed, best = 0, 0, 0, 0.0, -math.inf
    step_seconds, step_count = 0.0, 0
    state_path = directory / "state.pt"
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError(f"Refusing incompatible checkpoint: {name}")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        step, next_epoch, next_batch = saved["step"], saved["next_epoch"], saved["next_batch"]
        elapsed, best = saved["elapsed_seconds"], saved["best_cosine"]
        step_seconds, step_count = saved["step_seconds"], saved["step_count"]
        restore_rng(saved["rng"])
        print(f"Resumed {name} at step {step}/{total}", flush=True)
    start = time.monotonic()
    log_path = directory / "train.jsonl"
    # Drop log entries beyond the last committed checkpoint after a crash.
    if log_path.exists():
        kept = [line for line in log_path.read_text().splitlines() if json.loads(line)["step"] <= step]
        log_path.write_text("".join(line + "\n" for line in kept))

    def save():
        atomic_save(state_path, {"identity": identity, "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "rng": capture_rng(), "step": step,
                    "next_epoch": next_epoch, "next_batch": next_batch,
                    "elapsed_seconds": elapsed + time.monotonic() - start, "best_cosine": best,
                    "step_seconds": step_seconds, "step_count": step_count})

    try:
        for epoch in range(next_epoch, cfg.epochs):
            plan = plan_batches(train.lengths, cfg.batch_size, seed + epoch, cfg.bucket_window)
            begin = next_batch if epoch == next_epoch else 0
            for batch_index in range(begin, len(plan)):
                if step >= total:
                    break
                guard.check()
                if guard.enabled:
                    torch.cuda.synchronize()
                t = time.monotonic()
                row = optimization_step(model, opt, scaler, train.batch(plan[batch_index]), cfg)
                scheduler.step()
                if guard.enabled:
                    torch.cuda.synchronize()
                duration = time.monotonic() - t
                step_seconds += duration
                step_count += 1
                step += 1
                next_epoch, next_batch = epoch, batch_index + 1
                if next_batch == len(plan):
                    next_epoch, next_batch = epoch + 1, 0
                row.update(step=step, epoch=epoch, lr=opt.param_groups[0]["lr"], seconds=duration,
                           batch_sha256=hashlib.sha256(plan[batch_index].tobytes()).hexdigest())
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                if step % cfg.log_every == 0 or step == total:
                    spent = elapsed + time.monotonic() - start
                    print(f"{name} {step:,}/{total:,}; loss {row['loss']:.4f}; "
                          f"elapsed {spent / 60:.1f}m; ETA {(total - step) * step_seconds / step_count / 60:.1f}m",
                          flush=True)
                if step % cfg.eval_every == 0:
                    scores = evaluate(model, val, cfg)
                    atomic_json(directory / f"eval-{step:06d}.json", scores)
                    if scores["heldout"]["cos_to_target"] > best:
                        best = scores["heldout"]["cos_to_target"]
                        atomic_save(directory / "best_model.pt", model.state_dict())
                if step % cfg.checkpoint_every == 0 or step == total:
                    save()
                if on_step is not None:
                    on_step(step)
            if step >= total:
                break
        save()
        # Save final weights before expensive external evaluations; interrupted eval can retry.
        atomic_save(directory / "model.pt", model.state_dict())
        tokenizer_dir = Path(cfg.data_dir) / "tokenizer"
        if tokenizer_dir.exists():
            shutil.copytree(tokenizer_dir, directory / "tokenizer", dirs_exist_ok=True)
        atomic_json(directory / "model_config.json", {"arm": arm, **asdict(cfg.model)})
        scores = evaluate(model, val, cfg)
        if scores["heldout"]["cos_to_target"] > best:
            best = scores["heldout"]["cos_to_target"]
            atomic_save(directory / "best_model.pt", model.state_dict())
        benchmark = None
        if cfg.benchmark:
            from transformers import AutoTokenizer
            from .benchmark import run_benchmark
            tokenizer = AutoTokenizer.from_pretrained(Path(cfg.data_dir) / "tokenizer", local_files_only=True)
            benchmark = run_benchmark(model, tokenizer, cfg)
        result = {**identity, "steps": step, "parameters": sum(p.numel() for p in model.parameters()),
                  "seconds_per_step": step_seconds / max(step_count, 1),
                  "elapsed_seconds": elapsed + time.monotonic() - start,
                  "validation": scores, "benchmark": benchmark, "memory": guard.report(),
                  "model_config": asdict(cfg.model), "batch_size": cfg.batch_size,
                  "max_tokens": cfg.max_tokens, "status": "complete"}
        atomic_json(directory / "result.json", result)
        return result
    finally:
        del model, opt, scheduler, scaler
        gc.collect()
        if guard.enabled:
            torch.cuda.empty_cache()


def grade_results(root, arms=("A0", "A1", "A2", "A3"), seeds=(1, 2)):
    root = Path(root)
    results = {}
    for arm in arms:
        for seed in seeds:
            path = root / f"{arm}-seed{seed}" / "result.json"
            if path.exists():
                results[arm, seed] = json.loads(path.read_text())
    comparisons = {}
    for arm in arms:
        if arm == "A0":
            continue
        pairs = [(results.get((arm, s)), results.get(("A0", s))) for s in seeds]
        if len(seeds) < 2 or any(a is None or b is None for a, b in pairs):
            comparisons[arm] = {"settled": False, "reason": "Both paired seeds are required"}
            continue
        gauges = {}
        for gauge in ("cos_to_target", "erank", "self_cos"):
            deltas = [a["validation"]["heldout"][gauge] - b["validation"]["heldout"][gauge] for a, b in pairs]
            gauges[gauge] = {"seed_deltas": dict(zip(map(str, seeds), deltas)),
                             "same_direction": all(d > 0 for d in deltas) or all(d < 0 for d in deltas)}
        long_pairs = [(a["validation"]["above_128_tokens"], b["validation"]["above_128_tokens"])
                      for a, b in pairs]
        if all(a["n"] > 1 and b["n"] > 1 for a, b in long_pairs):
            for gauge in ("cos_to_target", "erank", "self_cos"):
                ds = [a[gauge] - b[gauge] for a, b in long_pairs]
                gauges[f"above_128_{gauge}"] = {"seed_deltas": dict(zip(map(str, seeds), ds)),
                    "same_direction": all(d > 0 for d in ds) or all(d < 0 for d in ds)}
        if all(a["benchmark"] is not None and b["benchmark"] is not None for a, b in pairs):
            ds = [a["benchmark"]["mean"] - b["benchmark"]["mean"] for a, b in pairs]
            gauges["benchmark_mean"] = {"seed_deltas": dict(zip(map(str, seeds), ds)),
                                        "same_direction": all(d > 0 for d in ds) or all(d < 0 for d in ds)}
        predicted_gauge = "above_128_cos_to_target" if arm in ("A1", "A2g") else "cos_to_target"
        expected_direction = -1 if arm == "A3" else 1
        predicted = gauges.get(predicted_gauge)
        verdict = None if predicted is None else all(
            d * expected_direction > 0 for d in predicted["seed_deltas"].values())
        comparisons[arm] = {"settled": gauges["cos_to_target"]["same_direction"], "gauges": gauges,
                            "prediction": PREDICTIONS[arm], "prediction_gauge": predicted_gauge,
                            "prediction_supported_both_seeds": verdict}
    anchor_test = {}
    if len(seeds) >= 2 and all((a, s) in results for a in ("A2", "A3") for s in seeds):
        ds = [results["A2", s]["validation"]["heldout"]["cos_to_target"] -
              results["A3", s]["validation"]["heldout"]["cos_to_target"] for s in seeds]
        anchor_test = {"A2_minus_A3_cosine": dict(zip(map(str, seeds), ds)),
                       "anchor_prediction_supported_both_seeds": all(d > 0 for d in ds)}
    report = {"completed_runs": len(results), "expected_runs": len(arms) * len(seeds),
              "comparisons_to_A0": comparisons, "anchor_ablation": anchor_test}
    atomic_json(root / "summary.json", report)
    return report


def run_screen(cfg=None):
    cfg = cfg or ScreenConfig()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "experiment.json").exists():
        old = json.loads((root / "experiment.json").read_text())
        if old["config_id"] != cfg.fingerprint():
            raise ValueError("Output directory belongs to a different experiment")
    atomic_json(root / "predictions.json", PREDICTIONS)  # Written before any run.
    train, val = open_store(cfg, cfg.train_chunks), open_store(cfg, [cfg.val_chunk])
    dataset_id = hashlib.sha256(json.dumps([train.manifests, val.manifests], sort_keys=True).encode()).hexdigest()
    if (root / "experiment.json").exists() and old["dataset_id"] != dataset_id:
        raise ValueError("Prepared data changed since this experiment began")
    atomic_json(root / "experiment.json", {"config_id": cfg.fingerprint(), "dataset_id": dataset_id,
                "config": asdict(cfg), "data": [train.manifests, val.manifests],
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "schedule_note": "v2 2000-step warmup retained even for the ~2000-step screen"})
    atomic_json(root / "environment.json", {"packages": {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}})
    pull_completed(cfg)
    uploader = Uploader(cfg)
    try:
        for arm in cfg.arms:
            for seed in cfg.seeds:
                guard = MemoryGuard(cfg.memory_limit_gb, cfg.device)
                train_arm(cfg, arm, seed, train, val, dataset_id, guard)
                grade_results(root, cfg.arms, cfg.seeds)
                uploader.submit(f"{arm}-seed{seed}")
    finally:
        uploader.close()
    return grade_results(root, cfg.arms, cfg.seeds)
