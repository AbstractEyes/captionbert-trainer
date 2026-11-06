"""Bounded apparatus gates: gradients, full-width memory, export and resume math."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from .configuration_captionbert import CaptionBertConfig
from .modeling_captionbert import CaptionBertModel
from .structured_campaign import ARMS, CampaignConfig, export_model, step
from .train import capture_rng, restore_rng, seed_all
from .memory import MemoryGuard
from .data import atomic_json


def preflight(output, device="cuda", batch_size=8, length=256, full=False):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    guard = MemoryGuard(92.0 if full else 20.0, device)
    rows = []
    selected = ("KALLP",) if full else ("R8", "N16", "H16", "KEND", "KALL", "KALLP")
    for arm in selected:
        seed_all(1)
        model = CaptionBertModel(CaptionBertConfig(**ARMS[arm])).to(device)
        cfg = CampaignConfig(device=device, batch_size=batch_size, push=False)
        ids = torch.randint(104, model.config.vocab_size, (batch_size, length))
        mask = torch.ones_like(ids)
        pos = torch.zeros(batch_size, length, 2, dtype=torch.long)
        target = torch.randn(batch_size, 768)
        pointer = torch.zeros_like(ids)
        batch = (ids, mask, pos, target, pointer)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-9)
        started = time.monotonic()
        values = step(model, optimizer, batch, cfg)
        guard.check()
        row = dict(arm=arm, batch=batch_size, length=length, finite=True,
                   seconds=time.monotonic() - started, losses=values, memory=guard.report())
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer, batch
        gc.collect()
        if guard.enabled:
            torch.cuda.empty_cache()
    atomic_json(output, dict(passed=True, full_batch=full, rows=rows,
                note="Synthetic engineering only: no sentence-quality or research verdict"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/content/structured-preflight.json")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    preflight(args.output, args.device, 2048 if args.full else 8, full=args.full)


if __name__ == "__main__":
    main()
