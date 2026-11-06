"""Build measured tables for the structured campaign after development selection.

Requires the locked six-model deployment decision. Final test values are read
only after that decision is checked against the unchanged development results.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_structured_final import choose_deployment

LABELS = {
    "R8": "Relative 8 heads", "N16": "Relative 16 heads",
    "N16-repeat": "Relative 16-head repeat", "H16": "Hierarchical",
    "KEND": "Key at endings", "KALL": "Key at every piece",
    "KALLP": "Whole key + lookup", "KALLP_N0": "Lookup without record norm",
    "KALLP_L8": "Lookup at later layer",
}
ORDER = ("R8", "N16", "N16-repeat", "H16", "KEND", "KALL", "KALLP", "KALLP_N0", "KALLP_L8")
TASKS = ("STS-B", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16", "BIOSSES")
FOLLOWUP = "frozen-followup-59ec95344958"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |"] +
        ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def f(value, digits=5):
    if not math.isfinite(value):
        raise ValueError("Public tables require finite actual measurements")
    return format(value, f".{digits}f")


def build_tables(screen, final, chosen):
    rows = []
    for arm in ORDER:
        pair = screen["analysis"]["pairs"][arm]
        rows.append([LABELS[arm], f(pair["scores"][0]), f(pair["scores"][1]),
            f(pair["mean_score"]), f(mean(pair["teacher"])), f(mean(pair["ranks"]), 2)])
    tables = {"SCREEN_TABLE": table(["Option", "Dev seed 1", "Dev seed 2", "Dev mean",
        "Mean teacher cosine", "Mean participation rank"], rows)}
    lookup = {}
    for arm in ORDER:
        lookup[arm] = [load(ROOT / f".cache/structured-results/{arm}-seed{s}/frozen-development/report.json")["measurements"]["intact"]
                       for s in (1, 2)]
    rows = [[LABELS[arm], *[f(100*x["all"]["exact_record_top1"], 3) for x in lookup[arm]],
             *[f(x["all"]["record_mass"], 6) for x in lookup[arm]]] for arm in ORDER]
    tables["LOOKUP_TABLE"] = table(["Option", "Top-one seed 1 (%)", "Top-one seed 2 (%)",
        "Record probability seed 1", "Record probability seed 2"], rows)
    rows = []
    for arm in ("KEND", "KALL", "KALLP", "KALLP_N0"):
        for seed, item in enumerate(lookup[arm], 1):
            rows.append([LABELS[arm], seed, *[f(100*item[group]["exact_record_top1"], 3)
                for group in ("ending", "non_ending", "continuation", "above128")]])
    tables["SUBWORD_TABLE"] = table(["Option", "Seed", "Ending (%)", "Non-ending (%)",
        "Continuation (%)", "Caption >128 tokens (%)"], rows)
    rows, tests, geometry, costs = [], [], [], []
    for key, item in final["results"].items():
        stage = "4M ×4 epochs" if key.startswith("full4m/") else "1M ×16 epochs"
        label = LABELS[item["arm"]]
        rows.append([stage, label, item["seed"], f(item["development"]["spearman"]),
            f(item["evaluation"]["heldout"]["cos_to_target"]),
            f(item["evaluation"]["heldout"]["erank"], 2)])
        # The caller already verified the development-only lock before reading
        # any of these test fields.
        benchmark = item["benchmark"]
        if set(benchmark["tasks"]) != set(TASKS) or benchmark["split"] != "test" or benchmark["max_tokens"] != 64:
            raise ValueError("An incomplete or altered final benchmark is not publishable")
        tests.append([stage + " / " + label, item["seed"],
            *[f(benchmark["tasks"][task]["spearman"], 4) for task in TASKS], f(benchmark["mean"], 4)])
        for subset in ("heldout", "above_128_tokens"):
            x = item["test_evaluation"][subset]
            geometry.append([stage + " / " + label, item["seed"],
                "All sampled rows" if subset == "heldout" else ">128-token rows", x["n"],
                f(x["cos_to_target"], 6), f(x["erank"], 3), f(x["self_cos"], 6)])
        costs.append([stage + " / " + label, item["seed"], item["steps"],
            f(item["seconds_per_step"], 3), f(item["wall_seconds_this_session"]/60, 2),
            f(item["memory"]["peak_allocated_gb"], 2)])
    tables["FINAL_DEVELOPMENT_TABLE"] = table(["Recipe", "Option", "Seed", "Dev STS-B",
        "Dev teacher cosine", "Dev participation rank"], rows)
    tables["TEST_TABLE"] = table(["Recipe / option", "Seed", *TASKS, "Mean"], tests)
    tables["TEST_GEOMETRY_TABLE"] = table(["Recipe / option", "Seed", "Chunk060 sample", "n",
        "Teacher cosine", "Participation rank", "Self cosine"], geometry)
    tables["COST_TABLE"] = table(["Recipe / option", "Seed", "Steps", "Seconds/optimizer step",
        "Session wall minutes", "Peak allocated GB"], costs)
    tables["DEPLOYMENT_SELECTION"] = (
        f"The development rule selected **{LABELS[chosen['arm']]} / seed{chosen['seed']}** "
        f"for the default4M release. The raw development mean leader was **{LABELS[chosen['raw_mean_leader']]}**. "
        f"Structured-pair viability: **{chosen['structured_viable']}**. "
        "The complete deployment-selection JSON retains both pair means, seed spreads and the rule; final tests were not used.")
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen", default=str(ROOT / ".cache/structured-results/screen-analysis.json"))
    p.add_argument("--final", default=str(ROOT / ".cache/structured-final-results/final-results.json"))
    p.add_argument("--selection", default=str(ROOT / "docs/results/structured_deployment_selection.json"))
    p.add_argument("--output", default=str(ROOT / ".cache/article-tables"))
    args = p.parse_args()
    screen, final, chosen = load(args.screen), load(args.final), load(args.selection)
    lock = load(ROOT / "docs/results/structured_screen_selection.json")
    rebuilt = choose_deployment(lock, final["results"])
    if any(chosen.get(k) != rebuilt[k] for k in rebuilt):
        raise ValueError("The development-only lock differs from the collected result identities/metrics")
    if len(screen["results"]) != 18 or len(final["results"]) != 6 or chosen["test_scores_used"]:
        raise ValueError("Finish all registered runs and lock development selection before building test tables")
    tables = build_tables(screen, final, chosen)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables.json").write_text(json.dumps(tables, indent=2)+"\n", encoding="utf8", newline="\n")
    for name, value in tables.items():
        (output / (name.lower()+".md")).write_text(value+"\n", encoding="utf8", newline="\n")
    print(json.dumps({"tables":list(tables), "default":chosen["arm"], "seed":chosen["seed"],
        "input_result_revision":final["source_revision"],"development_lock_reverified":True},indent=2))


if __name__ == "__main__":
    main()
