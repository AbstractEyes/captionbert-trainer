"""Lock the completed development selection and dataset revisions before tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(ROOT / ".cache/structured-results/screen-analysis.json"))
    p.add_argument("--output", default=str(ROOT / "docs/results/structured_screen_selection.json"))
    args = p.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf8"))
    from captionbert_rotary.structured_analysis import select_screen
    analysis = select_screen(source["results"])
    if analysis["status"] != "selected":
        p.error(f"No completed selection: {analysis['status']}; missing {analysis.get('missing', [])}")
    output = Path(args.output)
    if output.exists():
        p.error("A locked selection already exists; preserve it rather than silently selecting again")
    from captionbert_rotary.benchmark import TASKS
    from captionbert_rotary.structured_campaign import STSB_REVISION
    from huggingface_hub import HfApi
    api = HfApi()
    revisions = {repo: STSB_REVISION if task == "STS-B" else api.dataset_info(repo).sha for task, repo in TASKS}
    census = {}
    for name, value in sorted(source["results"].items()):
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        census[name] = dict(result_canonical_sha256=hashlib.sha256(raw).hexdigest(),
            checkpoint_sha256=value["checkpoint_sha256"], steps=value["steps"],
            config_id=value["config_id"], dataset_id=value["dataset_id"])
    lock = dict(locked_at_utc=datetime.now(timezone.utc).isoformat(),
        repo=source["repo"], source_revision=source["source_revision"],
        config_id=source["config_id"], complete_runs=len(census), analysis=analysis,
        result_census=census, final_benchmark_revisions=revisions,
        selection_uses="complete paired development only; no final-test scores",
        training_release="v0.3.0")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(dict(path=str(output), locked_at_utc=lock["locked_at_utc"],
                          selection=analysis["selection"], hardware_repeat=analysis["hardware_repeat_floor"]), indent=2))


if __name__ == "__main__":
    main()
