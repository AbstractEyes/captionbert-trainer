"""Registered two-seed structured screen and four-million-caption promotion."""
from dataclasses import asdict, dataclass
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from safetensors.torch import save_file
from .configuration_captionbert import CaptionBertConfig
from .modeling_captionbert import CaptionBertModel
from .data import Store, atomic_json, plan_batches, prepare, sha256_file
from .structured_data import StructuredStore, prepare_word_targets, tokenizer_metadata
from .train import seed_all, capture_rng, restore_rng, atomic_save, make_scheduler
from .metrics import objective, evaluate
from .benchmark import encode_texts, run_benchmark
from .hub import Uploader, pull_completed
from .memory import MemoryGuard


SOURCE_REVISION = "3f174e3f391f983428171cefc6901b34b22274d8"
STSB_REVISION = "96943a16ea6a35129e253c659081cb59daf81b30"
ARMS = {
    "R8": dict(rotary_mode="rope", num_attention_heads=8, pointer_weight=0.0, use_value_addresses=False),
    "N16": dict(rotary_mode="ntk", pointer_weight=0.0, use_value_addresses=False),
    "H16": dict(rotary_mode="hierarchical", pointer_weight=0.0),
    "KEND": dict(rotary_mode="key_end", pointer_weight=0.0, pointer_positions="end"),
    "KALL": dict(rotary_mode="key_all", pointer_weight=0.0),
    "KALLP": dict(rotary_mode="key_all", pointer_weight=0.1, pointer_positions="all"),
    "KALLP_N0": dict(rotary_mode="key_all", pointer_weight=0.1, pointer_positions="all", key_rho=0.0),
    "KALLP_L8": dict(rotary_mode="key_all", pointer_weight=0.1, pointer_positions="all", lookup_layer=7),
}
SCREEN_LANES = (("R8", "N16", "N16-repeat"), ("H16", "KEND", "KALLP_N0"), ("KALL", "KALLP", "KALLP_L8"))
FOUR_MILLION_CHUNKS = (0, 1, 2, 3, 4, 6, 9, 10)


@dataclass
class CampaignConfig:
    stage: str = "screen"
    arms: tuple = ("R8", "N16", "H16", "KEND", "KALL", "KALLP")
    seeds: tuple = (1, 2)
    train_chunks: tuple = (0, 1)
    val_chunk: int = 61
    epochs: int = 4
    batch_size: int = 2048
    max_tokens: int = 256
    bucket_window: int = 64
    lr: float = 6e-4
    min_lr: float = 1e-6
    warmup_steps: int = 200
    temperature: float = 0.07
    grad_clip: float = 1.0
    log_every: int = 25
    eval_every: int = 500
    checkpoint_every: int = 100
    resume_upload_minutes: float = 10.0
    eval_batch: int = 64
    eval_rows: int = 5000
    data_dir: str = "/content/captionbert-data"
    output_dir: str = "/content/captionbert-structured"
    hub_repo: str = "AbstractPhil/captionbert-8192-rotary"
    push: bool = True
    device: str = "cuda"
    memory_limit_gb: float = 92.0
    max_steps: int | None = None
    final_test: bool = False
    code_revision: str = "v0.3.1"

    @property
    def selected_names(self):
        return [f"{arm}-seed{seed}" for arm in self.arms for seed in self.seeds]

    def fingerprint(self):
        raw = asdict(self)
        for key in ("arms", "seeds", "data_dir", "output_dir", "push", "device", "memory_limit_gb"):
            raw.pop(key)
        raw.update(arm_definitions=ARMS, source_revision=SOURCE_REVISION,
                   stsb_revision=STSB_REVISION, word_feature_version="whole-word-counter31-reverse-v1")
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def model_config(arm, tokenizer):
    base = arm.removesuffix("-repeat")
    if base not in ARMS:
        raise ValueError(f"Unregistered arm: {arm}")
    _, continuation = tokenizer_metadata(tokenizer)
    return CaptionBertConfig(continuation_ids=continuation,
        punctuation_ids=tokenizer.convert_tokens_to_ids([".", "!", "?"]),
        special_ids=tokenizer.all_special_ids, **ARMS[base])


def export_model(model, directory, tokenizer=None):
    CaptionBertConfig.register_for_auto_class("AutoConfig")
    CaptionBertModel.register_for_auto_class("AutoModel")
    model.save_pretrained(directory, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)
    if not (Path(directory) / "model.safetensors").is_file():
        raise RuntimeError("Safe AutoModel export did not produce model.safetensors")


def tensorboard_scores(writer, prefix, value, step):
    for key, item in value.items():
        if isinstance(item, dict):
            tensorboard_scores(writer, f"{prefix}/{key}", item, step)
        elif isinstance(item, (int, float)) and np.isfinite(item):
            writer.add_scalar(f"{prefix}/{key}", item, step)


@torch.no_grad()
def dev_sts(model, tokenizer, cfg, directory):
    from datasets import load_dataset
    from scipy.stats import spearmanr
    was_training = model.training
    model.eval()
    try:
        ds = load_dataset("mteb/stsbenchmark-sts", split="validation", revision=STSB_REVISION,
                          cache_dir=str(Path(cfg.data_dir) / "benchmarks"))
        left = encode_texts(model, tokenizer, list(ds["sentence1"]), cfg.device, cfg.eval_batch)
        right = encode_texts(model, tokenizer, list(ds["sentence2"]), cfg.device, cfg.eval_batch)
        prediction = (left * right).sum(-1).numpy()
        gold = np.asarray(ds["score"], dtype=np.float64)
        score = float(spearmanr(prediction, gold).statistic)
        if not np.isfinite(score):
            raise ValueError("Nonfinite STS-B development score")
        np.savez(Path(directory) / "dev-sts-predictions.npz", prediction=prediction, gold=gold)
        return dict(spearman=score, n=len(gold), split="validation", revision=STSB_REVISION,
                    max_tokens=64, used_for_selection=True)
    finally:
        model.train(was_training)


def step(model, optimizer, batch, cfg):
    ids, mask, _, teacher, targets = (value.to(cfg.device) for value in batch)
    optimizer.zero_grad(set_to_none=True)
    pointer = targets if model.config.pointer_weight else None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(cfg.device).startswith("cuda")):
        output = model(ids, mask, pointer_targets=pointer)
    base, nce, mse = objective(output.pooler_output, teacher, cfg.temperature)
    loss = base + model.config.pointer_weight * output.pointer_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss; keep the last valid checkpoint")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient; optimizer step was not applied")
    optimizer.step()
    return dict(loss=float(loss.detach()), nce=float(nce.detach()), mse=float(mse.detach()),
                pointer=float(output.pointer_loss.detach()), grad_norm=float(norm),
                pointer_opportunities=int((targets >= 0).sum()),
                pointer_positions=int((targets >= 0).sum()) if pointer is not None else 0)


def run_one(cfg, arm, seed, training, validation, tokenizer, dataset_id, guard, uploader):
    from torch.utils.tensorboard import SummaryWriter
    name = f"{arm}-seed{seed}"
    directory = Path(cfg.output_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(config_id=cfg.fingerprint(), dataset_id=dataset_id, arm=arm, seed=seed)
    if (directory / "result.json").exists():
        result = json.loads((directory / "result.json").read_text())
        if any(result.get(k) != v for k, v in identity.items()):
            raise ValueError("Existing run identity differs")
        if sha256_file(directory / "model.safetensors") != result["checkpoint_sha256"]:
            raise ValueError("Completed model failed its hash check")
        print(f"{name}: completed and verified; skipping training", flush=True)
        uploader.submit(name)
        return result
    seed_all(seed)
    model = CaptionBertModel(model_config(arm, tokenizer)).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    planned = len(plan_batches(training.lengths, cfg.batch_size, seed, cfg.bucket_window))
    total = planned * cfg.epochs
    if cfg.max_steps is not None:
        total = min(total, cfg.max_steps)
    if not total or cfg.warmup_steps >= total:
        raise ValueError("The registered run must include training after warmup")
    scheduler = make_scheduler(optimizer, cfg, total)
    current, start_epoch, cursor = 0, 0, 0
    best_cos = -float("inf")
    state_path = directory / "state.pt"
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
        if saved["identity"] != identity or saved["total"] != total:
            raise ValueError("Refusing incompatible resume state")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        current, start_epoch, cursor = saved["step"], saved["epoch"], saved["cursor"]
        best_cos = saved.get("best_cos", -float("inf"))
        restore_rng(saved["rng"])
        del saved
        print(f"{name}: continuing step {current}/{total}, epoch {start_epoch}, batch {cursor}", flush=True)
    else:
        atomic_json(directory / "run.json", {**identity, "model_config": model.config.to_dict(), "total_steps": total,
                    "training_rows": len(training), "seed_derivation": "explicit paired integer seeds 1/2, flagged deviation from crc32-name default"})
    writer = SummaryWriter(str(directory / "tensorboard"), purge_step=current + 1, flush_secs=20)
    writer.add_text("protocol/identity", json.dumps(identity), current)
    print(f"{name}: {len(training):,} rows; {total:,} steps; warmup {cfg.warmup_steps}; "
          f"{sum(p.numel() for p in model.parameters()):,} parameters; TensorBoard={directory / 'tensorboard'}", flush=True)
    history_path = directory / "train.jsonl"
    if history_path.exists():
        keep = [line for line in history_path.read_text().splitlines() if json.loads(line)["step"] <= current]
        history_path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    started = time.monotonic()
    measured_steps = 0
    training_seconds = 0.0

    def save_state(epoch, next_cursor, force=False):
        writer.flush()
        atomic_save(state_path, dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=capture_rng(), step=current, epoch=epoch,
                    cursor=next_cursor, total=total, best_cos=best_cos))
        uploader.submit_checkpoint(name, current, identity, force=force)

    try:
        model.train()
        for epoch in range(start_epoch, cfg.epochs):
            batches = plan_batches(training.lengths, cfg.batch_size, seed + epoch, cfg.bucket_window)
            for index in range(cursor if epoch == start_epoch else 0, len(batches)):
                if current >= total:
                    break
                before = time.monotonic()
                lr_used = optimizer.param_groups[0]["lr"]
                values = step(model, optimizer, training.batch(batches[index]), cfg)
                scheduler.step()
                if str(cfg.device).startswith("cuda"):
                    torch.cuda.synchronize()
                duration = time.monotonic() - before
                current += 1
                measured_steps += 1
                training_seconds += duration
                values.update(step=current, epoch=epoch + (index + 1) / len(batches), lr=lr_used,
                              seconds=duration, rows_per_second=cfg.batch_size / max(duration, 1e-9))
                with history_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(values, allow_nan=False) + "\n")
                for key, value in values.items():
                    writer.add_scalar(f"train/{key}", value, current)
                if current == 1 or current % cfg.log_every == 0:
                    guard.check()
                    eta = training_seconds / measured_steps * (total - current)
                    if guard.enabled:
                        writer.add_scalar("memory/allocated_gb", torch.cuda.memory_allocated() / 1e9, current)
                        writer.add_scalar("memory/reserved_gb", torch.cuda.memory_reserved() / 1e9, current)
                    print(f"{name} | {current}/{total} | epoch {values['epoch']:.3f}/{cfg.epochs} | "
                          f"loss {values['loss']:.4f} nce {values['nce']:.4f} mse {values['mse']:.6f} "
                          f"pointer {values['pointer']:.4f} | lr {lr_used:.3g} | {duration:.3f}s/step | ETA {eta / 60:.1f}m", flush=True)
                if current % cfg.eval_every == 0 or current == total:
                    scores = evaluate(model, validation, cfg)
                    atomic_json(directory / f"eval-{current:06d}.json", scores)
                    tensorboard_scores(writer, "validation", scores, current)
                    if scores["heldout"]["cos_to_target"] > best_cos:
                        best_cos = scores["heldout"]["cos_to_target"]
                        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(directory / "best.safetensors"), metadata={"format": "pt"})
                    # Retain a trajectory midpoint and final, not an unbounded copy per log interval.
                    if current >= total // 2 and not (directory / "trajectory-mid.safetensors").exists():
                        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(directory / "trajectory-mid.safetensors"), metadata={"format": "pt"})
                    model.train()
                if current % cfg.checkpoint_every == 0 or current == total:
                    save_state(epoch, index + 1)
            cursor = 0
            if current >= total:
                break
        model.eval()
        export_model(model, directory, tokenizer)
        scores = evaluate(model, validation, cfg)
        development = dev_sts(model, tokenizer, cfg, directory)
        tensorboard_scores(writer, "final/validation", scores, current)
        tensorboard_scores(writer, "final/stsb_development", development, current)
        test = None
        test_evaluation = None
        if cfg.final_test:
            test_store = Store([Path(cfg.data_dir) / f"chunk_060_L{cfg.max_tokens}"])
            test_evaluation = evaluate(model, test_store, cfg)
            tensorboard_scores(writer, "final/heldout60", test_evaluation, current)
            test = run_benchmark(model, tokenizer, cfg)
            tensorboard_scores(writer, "final/test", test, current)
        writer.flush()
        result = {**identity, "status": "complete", "steps": current, "epochs": cfg.epochs,
                  "training_rows": len(training), "model_config": model.config.to_dict(),
                  "evaluation": scores, "development": development, "benchmark": test, "test_evaluation": test_evaluation,
                  "parameters": sum(p.numel() for p in model.parameters()), "seconds_per_step": training_seconds / max(measured_steps, 1),
                  "wall_seconds_this_session": time.monotonic() - started,
                  "checkpoint_sha256": sha256_file(directory / "model.safetensors"),
                  "tensorboard_event_files": [p.name for p in (directory / "tensorboard").glob("events.*")],
                  "memory": guard.report(), "source_rotary_revision": SOURCE_REVISION}
        atomic_json(directory / "result.json", result)
        print(f"{name} COMPLETE | dev STS {development['spearman']:.6f} | cosine {scores['heldout']['cos_to_target']:.6f}", flush=True)
        uploader.submit(name)
        return result
    except BaseException:
        if state_path.exists():
            safe = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
            uploader.submit_checkpoint(name, safe["step"], safe["identity"], force=True)
        raise
    finally:
        writer.close()
        del optimizer, scheduler, model
        gc.collect()
        if guard.enabled:
            torch.cuda.empty_cache()


def run_campaign(cfg):
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    chunks = tuple(dict.fromkeys((*cfg.train_chunks, cfg.val_chunk, *((60,) if cfg.final_test else ()))))
    if set(cfg.train_chunks) & {60, 61}:
        raise ValueError("Held-out data cannot enter training")
    if cfg.stage == "full4m" and cfg.train_chunks != FOUR_MILLION_CHUNKS:
        raise ValueError("The full stage requires the registered eight paired chunks")
    print(f"STAGE {cfg.stage}: code {cfg.code_revision}; ID {cfg.fingerprint()}; arms {cfg.arms}; seeds {cfg.seeds}", flush=True)
    guard = MemoryGuard(cfg.memory_limit_gb, cfg.device)
    prepare(cfg.data_dir, chunks=chunks, max_tokens=cfg.max_tokens)
    prepare_word_targets(cfg.data_dir, chunks=cfg.train_chunks, max_tokens=cfg.max_tokens)
    training = StructuredStore([Path(cfg.data_dir) / f"chunk_{c:03d}_L{cfg.max_tokens}" for c in cfg.train_chunks])
    validation = Store([Path(cfg.data_dir) / f"chunk_{cfg.val_chunk:03d}_L{cfg.max_tokens}"])
    expected_rows = 4000000 if cfg.stage == "full4m" else (1000000 if cfg.stage in ("screen", "followup", "exposure1m") else None)
    if expected_rows is not None and len(training) != expected_rows:
        raise ValueError(f"Registered {expected_rows:,} rows, found {len(training):,}; do not truncate or silently change scale")
    dataset_id = hashlib.sha256(json.dumps({"train": training.manifests, "validation": validation.manifests}, sort_keys=True).encode()).hexdigest()
    atomic_json(root / "experiment.json", {"config": asdict(cfg), "config_id": cfg.fingerprint(), "dataset_id": dataset_id,
                "arm_definitions": ARMS, "train_rows": len(training), "validation_rows": len(validation),
                "selection": "paired-seed STS-B validation; teacher/rank viability; test splits unopened for selection"})
    freeze = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
    (root / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    atomic_json(root / "environment.json", dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if guard.enabled else None, platform=platform.platform(), tf32=False,
                precision="bf16 autocast, fp32 parameters and optimizer", code_revision=cfg.code_revision))
    tokenizer = AutoTokenizer.from_pretrained(Path(cfg.data_dir) / "tokenizer", local_files_only=True)
    uploader = Uploader(cfg)
    results, failures = [], []
    try:
        pull_completed(cfg, dataset_id)
        for arm in cfg.arms:
            for seed in cfg.seeds:
                try:
                    results.append(run_one(cfg, arm, seed, training, validation, tokenizer, dataset_id, guard, uploader))
                except (RuntimeError, FloatingPointError) as error:
                    failures.append(dict(arm=arm, seed=seed, type=type(error).__name__, error=str(error)))
                    atomic_json(root / "failures.json", failures)
                    print(f"ARM FAILED {arm}-seed{seed}: {error}; prior artifacts retained", flush=True)
                atomic_json(root / "summary.json", dict(lane_local=True, results=results, failures=failures))
                uploader.flush(wait=False)
    finally:
        uploader.close()
    if failures:
        raise RuntimeError(f"{len(failures)} arm(s) failed; inspect failures.json before drawing conclusions")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "followup", "full4m", "exposure1m"), default="screen")
    parser.add_argument("--lane", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--output", default="/content/captionbert-structured")
    parser.add_argument("--data", default="/content/captionbert-data")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    if args.stage != "screen" and not args.arms:
        parser.error("Pass --arms explicitly after the registered selection; do not scale a default lane")
    cfg = CampaignConfig(stage=args.stage, arms=tuple(args.arms or SCREEN_LANES[args.lane]),
        seeds=tuple(args.seeds), output_dir=args.output, data_dir=args.data, push=not args.no_push)
    if args.stage == "full4m":
        cfg.train_chunks = FOUR_MILLION_CHUNKS
    if args.stage in ("full4m", "exposure1m"):
        cfg.epochs = 4 if args.stage == "full4m" else 16
        cfg.warmup_steps = 400
        cfg.max_steps = 7808
        cfg.final_test = True
    run_campaign(cfg)


if __name__ == "__main__":
    main()
