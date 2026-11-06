"""Complete the registered screen on two already-connected Colab runtimes."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil

CONFIG_ID = "dc37215529c188be7b93804c4a21f125afa353e037b1a2a8fc11e5b2fc817d3a"
TRAIN_SOURCE_SHA256 = "094bf61ef954c61a2b200e5e8760348160b76a5fbfd186602dd992e4b62ccfa6"
JOBS = {"A": [("KALL", (1, 2)), ("KALLP", (1,))],
        "B": [("KALLP_L8", (1, 2)), ("KALLP", (2,))]}
ORIGINAL = {"A": ("R8", "N16", "N16-repeat"), "B": ("H16", "KEND", "KALLP_N0")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane", choices=("A", "B"), required=True)
    p.add_argument("--root", default="/content/captionbert-structured/screen")
    p.add_argument("--data", default="/content/captionbert-data")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ship-only", action="store_true")
    p.add_argument("--console")
    args = p.parse_args()
    root = Path(args.root)
    if args.dry_run:
        print(json.dumps(dict(lane=args.lane, jobs=JOBS[args.lane], config_id=CONFIG_ID,
                              training_release="v0.3.0", source_sha256=TRAIN_SOURCE_SHA256), indent=2))
        return
    if not args.ship_only:
        import captionbert_rotary
        from captionbert_rotary.modeling_captionbert import CaptionBertModel
        from captionbert_rotary.structured_campaign import CampaignConfig, run_campaign
        from captionbert_rotary.data import sha256_file
        source = Path(inspect.getsourcefile(CaptionBertModel))
        if captionbert_rotary.__version__ != "0.3.0" or hashlib.sha256(source.read_bytes()).hexdigest() != TRAIN_SOURCE_SHA256:
            raise RuntimeError("Restore the original v0.3.0 isolated environment with the tagged notebook setup cell before continuing")
        for arm in ORIGINAL[args.lane]:
            for seed in (1, 2):
                directory = root / f"{arm}-seed{seed}"
                result = json.loads((directory / "result.json").read_text())
                if result.get("status") != "complete" or result.get("config_id") != CONFIG_ID or result.get("steps") != 1952:
                    raise RuntimeError("Wait for all six original lane runs to complete before executing this continuation cell")
                if sha256_file(directory / "model.safetensors") != result["checkpoint_sha256"]:
                    raise RuntimeError("Original completed checkpoint failed its recorded hash")
        for arm, seeds in JOBS[args.lane]:
            cfg = CampaignConfig(stage="screen", arms=(arm,), seeds=seeds,
                                 output_dir=str(root), data_dir=args.data)
            if cfg.fingerprint() != CONFIG_ID:
                raise RuntimeError("Continuation configuration differs; restore the frozen v0.3.0 setup and this cell")
            print(f"CONTINUING {args.lane}: {arm}, seeds {seeds}, unchanged screen {CONFIG_ID}", flush=True)
            run_campaign(cfg)
    # The notebook calls ship-only after the child exits, capturing its entire
    # console as well as the original lane console. No token value is written.
    if args.ship_only:
        from huggingface_hub import HfApi
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = root / f"operation-{args.lane}-{stamp}"
        folder.mkdir(exist_ok=True)
        for name, value in (("original-console.log", root / "console.log"),
                            ("continuation-console.log", Path(args.console) if args.console else None)):
            if value and value.is_file():
                if not value.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Console must belong to this campaign output directory")
                shutil.copyfile(value, folder / name)
        shutil.copyfile(__file__, folder / "continuation-source.py")
        (folder / "operation.json").write_text(json.dumps(dict(lane=args.lane, jobs=JOBS[args.lane],
            training_release="v0.3.0", config_id=CONFIG_ID, time_utc=stamp), indent=2) + "\n")
        HfApi().upload_folder(repo_id="AbstractPhil/captionbert-8192-rotary", folder_path=folder,
            path_in_repo=f"experiments/{CONFIG_ID}/operations/lane-{args.lane}/{stamp}",
            commit_message=f"Preserve original and continuation consoles: lane {args.lane}")
        print("Lane console/source artifacts uploaded", flush=True)


if __name__ == "__main__":
    main()
