"""Add mechanism measurements to immutable completed arms without retraining."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .config import ModelConfig, ScreenConfig
from .data import atomic_json, sha256_file
from .diagnostics import PROTOCOL, run_rotary_diagnostics, structural_probes
import hashlib
from .hub import Uploader
from .memory import MemoryGuard
from .model import build_model


def analyse_available(repo, experiment, arms=("A0", "A1", "A2", "A3"), seeds=(1, 2),
                      device="cpu", rows=16, batch_size=4, data_dir="data/diagnostics",
                      output_dir="runs/diagnostics", push=False, memory_limit_gb=20.0, threads=2):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    if not 1 <= rows <= 16 or not 1 <= threads <= 4:
        raise ValueError("Use 1..16 fixed probes and 1..4 CPU threads")
    torch.set_num_threads(threads)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(repo, files_metadata=True)
    files = {f.rfilename: f for f in info.siblings}
    cfg_upload = ScreenConfig(output_dir=str(root), hub_repo=repo, push=push)
    uploader = Uploader(cfg_upload)
    completed, awaiting = [], []
    try:
        for arm in arms:
            for seed in seeds:
                name = f"{arm}-seed{seed}"
                prefix = f"experiments/{experiment}/{name}"
                if prefix + "/result.json" not in files:
                    awaiting.append(name)
                    print(f"{name}: no completed result at this pinned Hub revision; leaving training untouched", flush=True)
                    continue
                folder = root / name
                folder.mkdir(parents=True, exist_ok=True)
                output = folder / "rotary_diagnostics.json"
                analysis_remote = prefix + "/analyses/" + PROTOCOL
                result_path = hf_hub_download(repo, prefix + "/result.json", revision=info.sha, local_dir=data_dir)
                result = json.loads(Path(result_path).read_text())
                if result["arm"] != arm or result["seed"] != seed or result["config_id"] != experiment:
                    raise ValueError(f"Source run identity mismatch: {name}")
                asset = files[prefix + "/model.pt"]
                lfs = getattr(asset, "lfs", None)
                expected_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                expected_probe = hashlib.sha256(json.dumps(structural_probes()[:rows]).encode()).hexdigest()
                # Reuse the exact stored protocol instead of remeasuring a completed analysis.
                existing = next((f for f in (analysis_remote + "/rotary_diagnostics.json",
                                            prefix + "/rotary_diagnostics.json") if f in files), None)
                if existing:
                    path = hf_hub_download(repo, existing, revision=info.sha, local_dir=data_dir)
                    saved = json.loads(Path(path).read_text())
                    identity = saved.get("identity", {})
                    if (saved.get("protocol") == PROTOCOL and saved.get("status") == "measured"
                            and saved.get("probe_sha256") == expected_probe
                            and identity.get("arm") == arm and identity.get("seed") == seed
                            and identity.get("step") == result["steps"]
                            and expected_sha and identity.get("checkpoint_sha256") == expected_sha):
                        atomic_json(output, saved)
                        completed.append({"name": name, "action": "reused published measurement", "path": str(output)})
                        print(f"{name}: reused published {PROTOCOL}", flush=True)
                        continue
                if getattr(asset, "size", 0) and asset.size > 300_000_000:
                    raise ValueError("Diagnostic checkpoint exceeds the registered 300 MB download budget")
                print(f"{name}: fetching final checkpoint ({getattr(asset, 'size', 0) / 1e6:.1f} MB) "
                      f"at pinned revision {info.sha}", flush=True)
                checkpoint = hf_hub_download(repo, prefix + "/model.pt", revision=info.sha, local_dir=data_dir)
                checkpoint_sha = sha256_file(checkpoint)
                if expected_sha and checkpoint_sha != expected_sha:
                    raise ValueError(f"Checkpoint content hash mismatch: {name}")
                if output.exists():
                    local = json.loads(output.read_text())
                    local_identity = local.get("identity", {})
                    if (local.get("protocol") == PROTOCOL and local.get("status") == "measured"
                            and local.get("probe_sha256") == expected_probe
                            and local_identity.get("checkpoint_sha256") == checkpoint_sha):
                        completed.append({"name": name, "action": "reused local measurement", "path": str(output)})
                        uploader.submit_analysis(name, folder, analysis_remote)
                        continue
                local_tokenizer = None
                for filename in files:
                    if filename.startswith(prefix + "/tokenizer/") and filename.endswith((".json", ".txt")):
                        path = hf_hub_download(repo, filename, revision=info.sha, local_dir=data_dir)
                        local_tokenizer = Path(path).parent
                if local_tokenizer is None:
                    raise ValueError(f"Tokenizer assets missing for {name}")
                tokenizer = AutoTokenizer.from_pretrained(local_tokenizer, local_files_only=True, trust_remote_code=False)
                model_cfg = ModelConfig(**result["model_config"])
                estimate = model_cfg.vocab_size * model_cfg.d_model + model_cfg.n_layers * (
                    4 * model_cfg.d_model ** 2 + 2 * model_cfg.d_model * model_cfg.d_ff)
                if estimate > 100_000_000:
                    raise ValueError("Source model exceeds the bounded CaptionBERT diagnostic size")
                guard = MemoryGuard(memory_limit_gb, device)
                model = build_model(arm, model_cfg)
                state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
                model.load_state_dict(state)
                del state
                model.to(device)
                cfg = SimpleNamespace(model=model_cfg, device=device, eval_batch=batch_size)
                try:
                    identity = {"config_id": experiment, "dataset_id": result["dataset_id"],
                                "arm": arm, "seed": seed, "step": result["steps"],
                                "checkpoint_sha256": checkpoint_sha,
                                "source_repo": repo, "source_revision": info.sha,
                                "source_checkpoint": prefix + "/model.pt"}
                    report = run_rotary_diagnostics(model, tokenizer, arm, cfg, output, identity, rows=rows)
                    report["memory"] = guard.report()
                    report["model_config"] = asdict(model_cfg)
                    atomic_json(output, report)
                finally:
                    del model
                    gc.collect()
                    if guard.enabled:
                        torch.cuda.empty_cache()
                atomic_json(folder / "source_result.json", result)
                uploader.submit_analysis(name, folder, analysis_remote)
                completed.append({"name": name, "action": "measured frozen checkpoint", "path": str(output)})
                if arm != "A0":
                    delta = report["interventions"]["remove_anchor"]["mean_l2_change"]
                    delta_text = f"{delta:.6g}" if delta is not None else "nonfinite"
                    print(f"{name}: frozen anchor-removal embedding L2 change {delta_text}; "
                          f"repeat max change {report['repeat_floor']['max_absolute_change']:.6g}", flush=True)
    finally:
        uploader.close()
    summary = {"protocol": PROTOCOL, "repo": repo, "source_revision": info.sha,
               "experiment": experiment, "completed": completed, "awaiting_completed_arms": awaiting,
               "training_steps_started": 0}
    if (root / "upload_status.json").exists():
        summary["shipping"] = json.loads((root / "upload_status.json").read_text())
    atomic_json(root / "summary.json", summary)
    return summary
