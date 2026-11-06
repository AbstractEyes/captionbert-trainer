"""Choose the default four-million-row pair using development values only."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from captionbert_rotary.structured_analysis import choose_near_tie


def choose_deployment(lock, results):
    screen = lock["analysis"]
    if screen["status"] != "selected":
        raise ValueError("A complete locked screen selection is required")
    structured = screen["selection"]["structured"]
    relative = screen["selection"]["relative"]
    required = [f"{stage}/{arm}-seed{seed}"
                for stage, arm in (("full4m", structured), ("full4m", relative), ("exposure1m", structured))
                for seed in (1, 2)]
    if any(name not in results for name in required):
        raise ValueError("Finish all six registered larger models before choosing the default")
    for key in required:
        item = results[key]
        rows = 4000000 if key.startswith("full4m/") else 1000000
        if item["status"] != "complete" or item["steps"] != 7808 or item["training_rows"] != rows:
            raise ValueError("Incomplete or altered larger-stage exposure")
        if item["development"]["split"] != "validation":
            raise ValueError("Deployment selection requires the development split")
        name = key.split("/", 1)[1]
        if name != f"{item['arm']}-seed{item['seed']}":
            raise ValueError("Larger-stage artifact arm/seed differs from its run identity")
        if key.startswith("exposure1m/") and item["dataset_id"] != screen["dataset_id"]:
            raise ValueError("Matched-exposure dataset differs from the original screen")
    if len({(results[key]["config_id"], results[key]["dataset_id"])
            for key in required if key.startswith("exposure1m/")}) != 1:
        raise ValueError("Matched-exposure pair disagrees on configuration/data identity")
    pairs = {}
    identities = set()
    for arm in (structured, relative):
        records = [results[f"full4m/{arm}-seed{seed}"] for seed in (1, 2)]
        for seed, item in zip((1, 2), records):
            if item["arm"] != arm or item["seed"] != seed:
                raise ValueError("Wrong larger-stage arm/seed")
            identities.add((item["config_id"], item["dataset_id"]))
        scores = [item["development"]["spearman"] for item in records]
        teacher = [item["evaluation"]["heldout"]["cos_to_target"] for item in records]
        ranks = [item["evaluation"]["heldout"]["erank"] for item in records]
        if not all(math.isfinite(v) for v in (*scores, *teacher, *ranks)):
            raise ValueError("Nonfinite development values")
        pairs[arm] = dict(arm=arm, scores=scores, mean_score=mean(scores),
            seed_spread=abs(scores[1]-scores[0]), teacher=teacher, mean_teacher=mean(teacher),
            ranks=ranks, mean_rank=mean(ranks), model_config=records[0]["model_config"],
            parameters=records[0]["parameters"],
            mean_step_seconds=mean(item["seconds_per_step"] for item in records))
    if len(identities) != 1:
        raise ValueError("Four-million-row pairs disagree on configuration/data identity")
    candidate, reference = pairs[structured], pairs[relative]
    viable = (candidate["mean_teacher"] >= reference["mean_teacher"] - .02
              and candidate["mean_rank"] >= .8 * reference["mean_rank"]
              and all(a >= .5*b for a, b in zip(candidate["ranks"], reference["ranks"])))
    hardware = screen["hardware_repeat_floor"]
    chosen, leader = choose_near_tie([candidate, reference] if viable else [reference], hardware)
    seed = max((1, 2), key=lambda s: (chosen["scores"][s-1], -s))
    return dict(arm=chosen["arm"], seed=seed, stage="full4m",
        raw_mean_leader=leader["arm"], structured_viable=viable, pairs=pairs,
        source_config_id=next(iter(identities))[0], source_dataset_id=next(iter(identities))[1],
        rule="development-only pair means; conservative near-tie preference; better development seed",
        screen_reference_repeat=hardware, full_stage_repeat_measured=False,
        test_scores_used=False, claim_limit="descriptive two-seed final evidence")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", required=True)
    p.add_argument("--results", default=str(ROOT / ".cache/structured-final-results/final-results.json"))
    p.add_argument("--output", default=str(ROOT / "docs/results/structured_deployment_selection.json"))
    args = p.parse_args()
    lock = json.loads(Path(args.selection).read_text(encoding="utf8"))
    collected = json.loads(Path(args.results).read_text(encoding="utf8"))
    chosen = choose_deployment(lock, collected["results"])
    chosen["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    chosen["results_source_revision"] = collected["source_revision"]
    output = Path(args.output)
    if output.exists():
        p.error("Preserve the already locked deployment decision")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(chosen, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in chosen.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
