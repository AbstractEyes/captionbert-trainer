"""Collect only small larger-stage results, preserving the locked selection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", default=str(ROOT / ".cache/structured-final-results"))
    p.add_argument("--repo", default="AbstractPhil/captionbert-8192-rotary")
    p.add_argument("--frozen-name", default="frozen-followup-59ec95344958")
    p.add_argument("--expected-full-data", default="7adb10f8833fab44e3ebb44315b53cb65bd8010b038115d15350b18536b57e86")
    args = p.parse_args()
    lock = json.loads(Path(args.selection).read_text(encoding="utf8"))
    if lock["analysis"]["status"] != "selected":
        p.error("Supply the complete locked screen selection")
    chosen = lock["analysis"]["selection"]
    from captionbert_rotary.structured_campaign import CampaignConfig, FOUR_MILLION_CHUNKS
    from huggingface_hub import HfApi, hf_hub_download
    output = Path(args.output).resolve()
    if output.drive.upper() != "E:":
        p.error("Keep local campaign artifacts/cache on E:")
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revision = api.model_info(args.repo).sha
    available = set(api.list_repo_files(args.repo, revision=revision))
    results, missing, identities = {}, [], {}
    for stage, arms in (("full4m", (chosen["structured"], chosen["relative"])),
                        ("exposure1m", (chosen["structured"],))):
        cfg = CampaignConfig(stage=stage, arms=arms, code_revision="v0.3.0",
            train_chunks=FOUR_MILLION_CHUNKS if stage == "full4m" else (0, 1),
            epochs=4 if stage == "full4m" else 16, warmup_steps=400,
            max_steps=7808, final_test=True)
        identities[stage] = cfg.fingerprint()
        for arm in arms:
            for seed in (1, 2):
                name = f"{arm}-seed{seed}"
                key = f"{stage}/{name}"
                prefix = f"experiments/{cfg.fingerprint()}/{name}/"
                if prefix + "result.json" not in available:
                    missing.append(key)
                    continue
                run = output / key
                run.mkdir(parents=True, exist_ok=True)
                small = ("result.json", "run.json", "train.jsonl", "eval.jsonl",
                         "dev-sts-predictions.npz", "environment.json", "pip_freeze.txt",
                         "experiment.json", "benchmark_revisions.json",
                         f"{args.frozen_name}/report.json")
                for filename in small:
                    remote = prefix + filename
                    if remote in available:
                        cached = hf_hub_download(args.repo, remote, revision=revision,
                                                 cache_dir=str(output / "hub-cache"))
                        destination = run / filename
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(Path(cached).read_bytes())
                result = json.loads((run / "result.json").read_text(encoding="utf8"))
                if any(result.get(k) != v for k, v in
                       dict(config_id=cfg.fingerprint(), arm=arm, seed=seed, status="complete",
                            steps=7808, training_rows=4000000 if stage == "full4m" else 1000000).items()):
                    raise ValueError(f"Wrong larger-stage identity/exposure: {key}")
                if result["benchmark"]["dataset_revisions"] != lock["final_benchmark_revisions"]:
                    raise ValueError(f"Benchmark revisions differ from the pre-test lock: {key}")
                expected_data = args.expected_full_data if stage == "full4m" else lock["analysis"]["dataset_id"]
                if result["dataset_id"] != expected_data:
                    raise ValueError(f"Prepared data differ from the recorded stage identity: {key}")
                results[key] = result
    report = dict(repo=args.repo, source_revision=revision, locked_selection=chosen,
                  stage_identities=identities, complete_runs=len(results), missing=missing, results=results)
    (output / "final-results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
