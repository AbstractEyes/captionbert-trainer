"""Bounded frozen phase-variation control on the completed binding seed pair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", RAYON_NUM_THREADS="2",
                  TOKENIZERS_PARALLELISM="false")
OUTPUT_NAME = "frozen-phase-constant-20261106"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/content/captionbert-serving-validation")
    p.add_argument("--data", default="/content/captionbert-data")
    p.add_argument("--probe-revision", required=True)
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    if len(args.probe_revision) != 40:
        p.error("Use a full immutable probe-source commit")
    from captionbert_rotary.structured_campaign import CampaignConfig
    from captionbert_rotary.hub import pull_completed
    from captionbert_rotary.data import prepare
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = CampaignConfig(stage="screen", arms=("KALLP",), seeds=(1, 2),
        output_dir=str(root / "screen"), data_dir=args.data)
    if cfg.fingerprint() != "dc37215529c188be7b93804c4a21f125afa353e037b1a2a8fc11e5b2fc817d3a":
        raise ValueError("Preserve the original v0.3.0 training environment")
    pull_completed(cfg, "659b21c1c6eb1074eeec7acdbfa9bd010965595a4669338690cf3ae38a4448d5")
    # One development chunk; no training chunks or new accuracy optimization.
    prepare(args.data, chunks=(61,), max_tokens=256)
    probe = root / ("phase-probe-" + args.probe_revision[:12] + ".py")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/AbstractEyes/captionbert-trainer/"
        + args.probe_revision + "/tools/probe_structured_campaign.py", probe)
    source_hash = hashlib.sha256(probe.read_bytes()).hexdigest()
    for seed in (1, 2):
        directory = root / f"screen/KALLP-seed{seed}"
        if not (directory / "result.json").is_file():
            raise ValueError("Restore both completed KALLP models before this control")
        result = json.loads((directory / "result.json").read_text())
        report = directory / OUTPUT_NAME / "report.json"
        if report.is_file():
            saved = json.loads(report.read_text())
            if (saved["probe_source_sha256"] != source_hash
                    or saved.get("checkpoint_sha256") != result["checkpoint_sha256"]
                    or saved["rows"] != 4096
                    or set(saved["measurements"]) != {"intact", "phase_constant"}):
                raise ValueError("Preserve the existing different measurement; use a new versioned folder")
            if args.push:
                from huggingface_hub import HfApi
                HfApi().upload_folder(repo_id=cfg.hub_repo, folder_path=report.parent,
                    path_in_repo=f"experiments/{result['config_id']}/{directory.name}/{OUTPUT_NAME}",
                    commit_message="Preserve compatible completed frozen phase control")
            print("Reusing compatible completed phase control:", seed, flush=True)
            continue
        command = [sys.executable, str(probe), "--root", str(root / "screen"), "--data", args.data,
            "--arms", "KALLP", "--seeds", str(seed), "--rows", "4096", "--batch", "8",
            "--sentence-ablations", "--phase-constant", "--output-name", OUTPUT_NAME,
            "--device", "cuda", "--memory-limit-gb", "92"]
        if args.push:
            command.append("--push")
        subprocess.run(command, check=True)
    print("Both frozen phase-variation controls complete; no optimizer or training", flush=True)


if __name__ == "__main__":
    main()
