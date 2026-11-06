"""Download only small campaign results/logs; never optimizer/model payloads."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from captionbert_rotary.structured_analysis import SCREEN_NAMES, select_screen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="AbstractPhil/captionbert-8192-rotary")
    p.add_argument("--config-id", default="dc37215529c188be7b93804c4a21f125afa353e037b1a2a8fc11e5b2fc817d3a")
    p.add_argument("--output", default=str(ROOT / ".cache/structured-results"))
    p.add_argument("--results-only", action="store_true", help="Read just final JSON for a timely selection census")
    p.add_argument("--include-frozen", action="store_true", help="Also collect small frozen reports and query arrays")
    args = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    output = Path(args.output).resolve()
    if output.drive.upper() != "E:":
        p.error("Keep local campaign artifacts/cache on E:")
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revision = api.model_info(args.repo).sha
    prefix = f"experiments/{args.config_id}/"
    available = set(api.list_repo_files(args.repo, revision=revision))
    results = {}
    for name in (*SCREEN_NAMES, "R8-repeat-seed1", "R8-repeat-seed2"):
        remote = prefix + name + "/result.json"
        if remote not in available:
            continue
        run = output / name
        run.mkdir(exist_ok=True)
        files = ("result.json",) if args.results_only else (
            "result.json", "train.jsonl", "eval.jsonl", "run.json", "dev-sts-predictions.npz",
            "environment.json", "pip_freeze.txt", "experiment.json")
        if args.include_frozen:
            files += ("frozen-development/report.json", "frozen-development/intact-queries.npz",
                      "frozen-development/row-ids.npy")
        for file in files:
            path = prefix + name + "/" + file
            if path in available:
                cached = hf_hub_download(args.repo, path, revision=revision, cache_dir=str(output / "hub-cache"))
                destination = run / file
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(Path(cached).read_bytes())
        value = json.loads((run / "result.json").read_text(encoding="utf8"))
        if value.get("config_id") != args.config_id:
            raise ValueError(f"Wrong artifact identity for {name}")
        results[name] = value
    report = dict(repo=args.repo, source_revision=revision, config_id=args.config_id,
                  complete_runs=len(results), analysis=select_screen(results), results=results)
    path = output / "screen-analysis.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({k: report[k] for k in ("repo", "source_revision", "complete_runs")}))
    print(json.dumps({k: v for k, v in report["analysis"].items() if k not in ("pairs", "contrasts")}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
