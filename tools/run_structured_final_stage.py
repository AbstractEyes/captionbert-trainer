"""Execute the locked six-model larger experiment on two existing G4s.

This orchestrator imports the original installed training release; it does not
replace it. Each lane owns one weight seed for all three registered recipes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil


TRAIN_VERSION = "0.3.0"
TRAIN_SOURCE_SHA256 = "094bf61ef954c61a2b200e5e8760348160b76a5fbfd186602dd992e4b62ccfa6"
SCREEN_ID = "dc37215529c188be7b93804c4a21f125afa353e037b1a2a8fc11e5b2fc817d3a"


def read_selection(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    analysis = value["analysis"]
    if analysis["status"] != "selected" or analysis["config_id"] != SCREEN_ID:
        raise ValueError("Finish the registered screen and lock a complete development-only selection first")
    selection = analysis["selection"]
    if selection["seeds"] != [1, 2]:
        raise ValueError("The larger experiment must preserve both registered seeds")
    if selection["relative"] not in ("R8", "N16"):
        raise ValueError("Invalid relative reference")
    if selection["structured"] not in ("H16", "KEND", "KALL", "KALLP", "KALLP_N0", "KALLP_L8"):
        raise ValueError("Invalid structured option")
    if not value.get("source_revision") or value.get("complete_runs", 0) < 18:
        raise ValueError("The selection needs a revision-pinned complete result census")
    revisions = value.get("final_benchmark_revisions", {})
    if len(revisions) != 8 or any(len(rev) != 40 for rev in revisions.values()):
        raise ValueError("Lock the same eight benchmark dataset revisions before running either final lane")
    return value


def jobs(selection, lane):
    selected = selection["analysis"]["selection"]
    seed = {"A": 1, "B": 2}[lane]
    return [("full4m", selected["structured"], seed),
            ("full4m", selected["relative"], seed),
            ("exposure1m", selected["structured"], seed)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane", choices=("A", "B"), required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--root", default="/content/captionbert-structured")
    p.add_argument("--data", default="/content/captionbert-data")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ship-only", action="store_true")
    p.add_argument("--console")
    args = p.parse_args()
    selection = read_selection(args.selection)
    work = jobs(selection, args.lane)
    root = Path(args.root)
    if args.dry_run:
        print(json.dumps(dict(lane=args.lane, jobs=work, training_release=TRAIN_VERSION,
                              selection=selection["analysis"]["selection"]), indent=2))
        return
    if args.ship_only:
        from huggingface_hub import HfApi
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = root / f"final-operation-{args.lane}-{stamp}"
        folder.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(args.selection, folder / "locked-selection.json")
        shutil.copyfile(__file__, folder / "orchestration-source.py")
        if args.console:
            console = Path(args.console)
            if not console.resolve().is_relative_to(root.resolve()):
                raise ValueError("Console must belong to this campaign")
            shutil.copyfile(console, folder / "console.log")
        (folder / "operation.json").write_text(json.dumps(dict(
            lane=args.lane, jobs=work, time_utc=stamp, training_release=TRAIN_VERSION
        ), indent=2) + "\n", encoding="utf8")
        HfApi().upload_folder(repo_id="AbstractPhil/captionbert-8192-rotary",
            folder_path=folder, path_in_repo=f"campaigns/structured-20261106/operations/lane-{args.lane}/{stamp}",
            commit_message=f"Preserve larger-stage lane {args.lane} console and locked selection")
        print("Full larger-stage console and operation source uploaded", flush=True)
        return
    import captionbert_rotary
    from captionbert_rotary.modeling_captionbert import CaptionBertModel
    from captionbert_rotary.structured_campaign import CampaignConfig, FOUR_MILLION_CHUNKS, run_campaign
    source = Path(inspect.getsourcefile(CaptionBertModel))
    if captionbert_rotary.__version__ != TRAIN_VERSION or hashlib.sha256(source.read_bytes()).hexdigest() != TRAIN_SOURCE_SHA256:
        raise RuntimeError("Restore the original v0.3.0 isolated environment with the tagged notebook setup cell")
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 1e9
    print(f"Larger-stage disk free: {free:.1f} GB; seed-owned jobs {work}", flush=True)
    if free < 25:
        raise RuntimeError("Less than25GB free: preserve uploaded screen artifacts and use a fresh G4 runtime with the original setup cell")
    for stage, arm, seed in work:
        directory = root / stage
        directory.mkdir(parents=True, exist_ok=True)
        cfg = CampaignConfig(stage=stage, arms=(arm,), seeds=(seed,),
            train_chunks=FOUR_MILLION_CHUNKS if stage == "full4m" else (0, 1),
            epochs=4 if stage == "full4m" else 16, warmup_steps=400,
            max_steps=7808, final_test=True, output_dir=str(directory), data_dir=args.data)
        if cfg.code_revision != "v0.3.0":
            raise RuntimeError("Training identity changed; restore the original v0.3.0 environment")
        pins = directory / "benchmark_revisions.json"
        expected = selection["final_benchmark_revisions"]
        if pins.exists() and json.loads(pins.read_text()) != expected:
            raise ValueError("Existing benchmark revisions differ; do not overwrite another run's pins")
        pins.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf8")
        shutil.copyfile(args.selection, directory / "locked-selection.json")
        print(f"LOCKED larger experiment: {stage}/{arm}-seed{seed}, {cfg.fingerprint()}", flush=True)
        run_campaign(cfg)


if __name__ == "__main__":
    main()
