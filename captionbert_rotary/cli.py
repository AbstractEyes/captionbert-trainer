import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import torch
from .config import ScreenConfig, ModelConfig, ARMS
from .data import atomic_json, prepare
from .memory import MemoryGuard
from .model import build_model
from .train import grade_results, optimization_step, seed_all


def smoke(device="cuda", batch_size=8, steps=2, memory_limit_gb=20.0,
          output="runs/smoke.json", tiny=False):
    """Synthetic full-width training gate, never an experiment result."""
    model_cfg = (ModelConfig(vocab_size=128, max_len=256, d_model=64, n_heads=2,
                             n_layers=2, d_ff=128, output_dim=32) if tiny else ModelConfig())
    cfg = ScreenConfig(model=model_cfg, batch_size=batch_size, device=device,
                       memory_limit_gb=memory_limit_gb, benchmark=False)
    reports = []
    for arm in ARMS:
        for seed in (1, 2):
            guard = MemoryGuard(memory_limit_gb, device)
            seed_all(seed)
            model = build_model(arm, model_cfg).to(device).train()
            opt = torch.optim.Adam(model.parameters(), lr=6e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
            ids = torch.randint(1, model_cfg.vocab_size, (batch_size, 256))
            mask = torch.ones_like(ids)
            pos = torch.zeros(batch_size, 256, 2, dtype=torch.long)
            pos[..., 0] = torch.arange(256) % 16
            pos[..., 1] = torch.arange(256) // 16
            targets = torch.randn(batch_size, model_cfg.output_dim)
            start = time.monotonic()
            losses = []
            for _ in range(steps):
                row = optimization_step(model, opt, scaler, (ids, mask, pos, targets), cfg)
                losses.append(row["loss"])
                guard.check()
            if guard.enabled:
                torch.cuda.synchronize()
            report = {"arm": arm, "seed": seed, "losses": losses,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "seconds": time.monotonic() - start, **guard.report()}
            reports.append(report)
            print(json.dumps(report), flush=True)
            del model, opt, scaler
            gc.collect()
            if guard.enabled:
                torch.cuda.empty_cache()
    result = {"kind": "synthetic_training_gate", "comparable_to_v2": False,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "batch_size": batch_size, "sequence_length": 256, "steps": steps,
              "model_config": asdict(model_cfg), "runs": reports}
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="CaptionBERT rotary controlled screen")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--chunks", type=int, nargs="+", default=[0, 1, 2, 3, 61])
    p.add_argument("--max-tokens", type=int, default=256)
    p = sub.add_parser("screen")
    p.add_argument("--config", help="JSON ScreenConfig overrides")
    p = sub.add_parser("smoke")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--memory-limit-gb", type=float, default=20.0)
    p.add_argument("--output", default="runs/smoke.json")
    p.add_argument("--tiny", action="store_true")
    p = sub.add_parser("grade")
    p.add_argument("directory")
    p = sub.add_parser("diagnostics")
    p.add_argument("--repo", default="AbstractPhil/captionbert-8192-rotary")
    p.add_argument("--experiment", required=True)
    p.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "A3"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    p.add_argument("--device", default="cpu")
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--data-dir", default="data/diagnostics")
    p.add_argument("--output-dir", default="runs/diagnostics")
    p.add_argument("--memory-limit-gb", type=float, default=20.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--push", action="store_true")
    p = sub.add_parser("model-card")
    p.add_argument("--repo", default="AbstractPhil/captionbert-8192-rotary")
    p.add_argument("--experiment", required=True)
    p.add_argument("--output-dir", default="runs/model-card")
    p.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data_dir, tuple(args.chunks), args.max_tokens)
    elif args.command == "smoke":
        options = vars(args)
        options.pop("command")
        smoke(**options)
    elif args.command == "grade":
        manifest = Path(args.directory) / "experiment.json"
        options = json.loads(manifest.read_text())["config"] if manifest.exists() else {}
        print(json.dumps(grade_results(args.directory, options.get("arms", ("A0", "A1", "A2", "A3")),
                                       options.get("seeds", (1, 2))), indent=2))
    elif args.command == "diagnostics":
        from .analyse import analyse_available
        options = vars(args)
        options.pop("command")
        analyse_available(**options)
    elif args.command == "model-card":
        from .model_card import finalize_model_card
        options = vars(args)
        options.pop("command")
        finalize_model_card(**options)
    else:
        from .train import run_screen
        raw = json.loads(Path(args.config).read_text()) if args.config else {}
        if "model" in raw:
            raw["model"] = ModelConfig(**raw["model"])
        for name in ("arms", "seeds", "train_chunks"):
            if name in raw:
                raw[name] = tuple(raw[name])
        run_screen(ScreenConfig(**raw))
