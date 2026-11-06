"""Read-only monitor for an existing screen, including v0.1.0. Standard library only."""
import argparse
import json
from pathlib import Path
import time


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def log_rows(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # The writer may currently be finishing the final line.
    return rows


def duration(seconds):
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def snapshot(root):
    root = Path(root)
    manifest = read_json(root / "experiment.json") or {}
    cfg = manifest.get("config", {})
    batch = cfg.get("batch_size", 0)
    train_manifests = (manifest.get("data") or [[]])[0]
    rows = sum(m.get("rows", 0) for m in train_manifests)
    steps_per_epoch = rows // batch if batch else 0
    total = steps_per_epoch * cfg.get("epochs", 0)
    if cfg.get("max_steps") is not None:
        total = min(total, cfg["max_steps"]) if total else cfg["max_steps"]
    names = [f"{a}-seed{s}" for a in cfg.get("arms", ("A0", "A1", "A2", "A3"))
             for s in cfg.get("seeds", (1, 2))]
    runs = []
    for name in names:
        directory = root / name
        records = log_rows(directory / "train.jsonl")
        last = records[-1] if records else None
        result = read_json(directory / "result.json")
        evaluations = sorted(directory.glob("eval-*.json")) if directory.exists() else []
        evaluation = read_json(evaluations[-1]) if evaluations else None
        checkpoint = directory / "state.pt"
        runs.append({"name": name, "last": last, "total": total, "batch": batch,
                     "epoch_total": cfg.get("epochs"), "result": result,
                     "evaluation": evaluation,
                     "training_compute_seconds": sum(r.get("seconds", 0) for r in records),
                     "checkpoint_mtime": checkpoint.stat().st_mtime if checkpoint.exists() else None})
    pending = len(list((root / ".upload_queue").glob("*/job.json")))
    return {"runs": runs, "pending_uploads": pending, "manifest_found": bool(manifest)}


def format_step(run):
    row = run["last"]
    numeric_total = run["total"] or row.get("total_steps", 0)
    total = numeric_total or "?"
    seconds = row.get("seconds", 0)
    step = row["step"]
    mean_seconds = run["training_compute_seconds"] / max(step, 1)
    eta = (numeric_total - step) * mean_seconds if numeric_total else None
    batch = run["batch"] or row.get("rows_seen", 0) // max(step, 1)
    rate = batch / seconds if seconds > 0 else 0
    epochs = run["epoch_total"] or row.get("total_epochs", "?")
    lr = row.get("lr_used", row.get("lr", 0))
    return (f"{run['name']} | epoch {row.get('epoch', 0) + 1}/{epochs} "
            f"| step {step}/{total} | loss {row['loss']:.5f} "
            f"NCE {row.get('nce', 0):.5f} MSE {row.get('mse', 0):.6f} "
            f"| lr {lr:.3e} grad {row.get('grad_norm', 0):.3f} "
            f"| {seconds:.3f}s/step {rate:,.0f} rows/s "
            f"| train compute {duration(run['training_compute_seconds'])} "
            f"ETA train ~{duration(eta)} (excludes evaluation/uploads)")


def format_metrics(name, scores):
    fields = []
    for label, key in (("heldout", "heldout"), (">128 tokens", "above_128_tokens")):
        metrics = scores.get(key, {})
        if metrics.get("n", 0):
            fields.append(f"{label} n={metrics['n']} cos={metrics.get('cos_to_target', 0):.5f} "
                          f"erank={metrics.get('erank')} self_cos={metrics.get('self_cos')}")
    return f"{name} | validation | " + " | ".join(fields)


def watch(root, interval=2.0, once=False):
    if interval <= 0:
        raise ValueError("Polling interval must be positive")
    print(f"Watching {Path(root)}; read-only, no GPU allocation. Ctrl+C stops only this monitor.", flush=True)
    seen = {}
    last_change = time.monotonic()
    while True:
        state = snapshot(root)
        changed = False
        complete = 0
        for run in state["runs"]:
            name = run["name"]
            if run["last"]:
                signature = (run["last"]["step"], run["last"].get("batch_sha256"))
                if seen.get((name, "step")) != signature:
                    print(format_step(run), flush=True)
                    seen[name, "step"] = signature
                    changed = True
            if run["checkpoint_mtime"] and seen.get((name, "checkpoint")) != run["checkpoint_mtime"]:
                print(f"{name} | checkpoint present: {Path(root) / name / 'state.pt'}", flush=True)
                seen[name, "checkpoint"] = run["checkpoint_mtime"]
                changed = True
            if run["evaluation"]:
                signature = json.dumps(run["evaluation"], sort_keys=True)
                if seen.get((name, "eval")) != signature:
                    print(format_metrics(name, run["evaluation"]), flush=True)
                    seen[name, "eval"] = signature
                    changed = True
            if run["result"]:
                complete += 1
                if not seen.get((name, "done")):
                    print(format_metrics(name, run["result"]["validation"]), flush=True)
                    benchmark = run["result"].get("benchmark")
                    print(f"{name} | completed | eight-task mean: "
                          f"{benchmark['mean'] if benchmark else 'not measured'}", flush=True)
                    seen[name, "done"] = True
                    changed = True
        if seen.get("uploads") != state["pending_uploads"]:
            print(f"Screen: {complete}/{len(state['runs'])} runs complete; "
                  f"{state['pending_uploads']} queued uploads", flush=True)
            seen["uploads"] = state["pending_uploads"]
            changed = True
        now = time.monotonic()
        if changed:
            last_change = now
        elif now - last_change >= 30:
            print("Monitor heartbeat: no new logged step in 30s. The training process may be "
                  "evaluating, saving, uploading, or idle; this monitor cannot inspect its process state.", flush=True)
            last_change = now
        if once:
            return state
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="/content/captionbert-screen")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        watch(args.directory, args.interval, args.once)
    except KeyboardInterrupt:
        print("Monitor stopped. Training was not changed.", flush=True)
