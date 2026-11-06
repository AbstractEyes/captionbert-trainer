"""Render the six larger outcomes after the development-only choice is locked."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
sys.path.insert(0, str(ROOT))
from tools.analyze_structured_final import choose_deployment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(ROOT / ".cache/structured-final-results/final-results.json"))
    p.add_argument("--selection", default=str(ROOT / "docs/results/structured_deployment_selection.json"))
    args = p.parse_args()
    collected = json.loads(Path(args.results).read_text(encoding="utf8"))
    chosen = json.loads(Path(args.selection).read_text(encoding="utf8"))
    lock = json.loads((ROOT / "docs/results/structured_screen_selection.json").read_text(encoding="utf8"))
    rebuilt = choose_deployment(lock, collected["results"])
    if len(collected["results"]) != 6 or any(chosen.get(k) != v for k, v in rebuilt.items()):
        raise ValueError("Lock the complete development-only decision before plotting final tests")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "captionbert-larger-20261106"})
    arms = lock["analysis"]["selection"]
    specifications = [
        ("full4m", arms["structured"], "4M rows × 4 epochs / hierarchical", "#0072b2"),
        ("full4m", arms["relative"], "4M rows × 4 epochs / relative", "#009e73"),
        ("exposure1m", arms["structured"], "1M rows × 16 epochs / hierarchical", "#64748b"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
    records = []
    for row, (stage, arm, label, color) in enumerate(specifications):
        pair = [collected["results"][f"{stage}/{arm}-seed{s}"] for s in (1, 2)]
        development = [x["development"]["spearman"] for x in pair]
        tests = [x["benchmark"]["mean"] for x in pair]
        records.append(dict(stage=stage, arm=arm, development=development, test_suite_mean=tests))
        for axis, values in zip(axes, (development, tests)):
            axis.plot(values, [row, row], color=color, alpha=.6)
            for seed, marker in ((1, "o"), (2, "s")):
                y = row + (-.065 if seed == 1 else .065)
                axis.scatter(values[seed-1], y, marker=marker, s=45, color=color, zorder=3)
                if stage == chosen["stage"] and arm == chosen["arm"] and seed == chosen["seed"]:
                    axis.scatter(values[seed-1], y, marker="*", s=155, color="#f59e0b",
                                 edgecolor="#334155", linewidth=.6, zorder=4)
            axis.scatter(sum(values)/2, row, marker="|", s=150, color=color, linewidth=1.8)
    axes[0].set_yticks(range(3), [x[2] for x in specifications])
    axes[0].invert_yaxis()
    axes[0].set_title("Development comparison")
    axes[0].set_xlabel("STS-B validation Spearman")
    axes[1].set_title("Final test outcomes")
    axes[1].set_xlabel("Unweighted mean Spearman across eight tasks")
    for axis in axes:
        axis.margins(x=.15, y=.28)
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
    handles = [
        Line2D([], [], color="#334155", marker="o", ls="", label="Seed 1"),
        Line2D([], [], color="#334155", marker="s", ls="", label="Seed 2"),
        Line2D([], [], color="#334155", marker="|", ls="", markersize=10, label="Pair mean"),
        Line2D([], [], color="#f59e0b", marker="*", ls="", markersize=10, label="Development-selected default"),
    ]
    fig.suptitle("Larger experiments: development choice and final tests", fontsize=14, y=.98)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, .92), ncol=4, frameon=False)
    fig.text(.02, .045, "Every recipe: 7,808 optimizer steps and 15,990,784 example presentations. Processed tokens and FLOPs are not exactly matched.", fontsize=8.5)
    fig.text(.02, .012, "Joined endpoints show the two seeds, not a confidence interval. The default is chosen before reading these test outcomes.", fontsize=8.5)
    fig.tight_layout(rect=(0, .1, 1, .83))
    output = ROOT / "docs/figures"
    for extension in ("png", "svg"):
        path = output / ("structured_larger." + extension)
        options = {"metadata": {"Date": None}} if extension == "svg" else {}
        fig.savefig(path, dpi=200, facecolor="white", **options)
        if extension == "svg":
            path.write_bytes(b"\n".join(line.rstrip() for line in path.read_bytes().splitlines()) + b"\n")
    plt.close(fig)
    summary = dict(results_source_revision=collected["source_revision"], records=records,
        selection=chosen["arm"], selected_seed=chosen["seed"], test_scores_used_for_selection=False,
        input_sha256={str(Path(x).name): hashlib.sha256(Path(x).read_bytes()).hexdigest()
                      for x in (args.results, args.selection)},
        figure_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in output.glob("structured_larger.*")})
    (ROOT / "docs/results/structured_larger_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")
    print("Larger-stage figures and source metrics written; development-only lock checked.")


if __name__ == "__main__":
    main()
