"""Render the measured screen as standalone scientific figures; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

LABELS = (
    ("R8", "Relative, 8 heads"),
    ("N16", "Relative, 16 heads (4M follow-up)"),
    ("N16-repeat", "16-head reference repeat"),
    ("H16", "Hierarchical (4M follow-up)"),
    ("KEND", "Key phase at word endings"),
    ("KALL", "Key phase at all pieces"),
    ("KALLP", "Whole key + lookup training"),
    ("KALLP_N0", "Lookup, record norm disabled"),
    ("KALLP_L8", "Lookup at later layer"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(ROOT / ".cache/structured-results"))
    p.add_argument("--output", default=str(ROOT / "docs/figures"))
    args = p.parse_args()
    source, output = Path(args.input), Path(args.output)
    collected = json.loads((source / "screen-analysis.json").read_text(encoding="utf8"))
    if collected["complete_runs"] < 18 or collected["analysis"]["status"] != "selected":
        p.error("Render only a complete matched screen")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "svg.hashsalt": "captionbert-structured-20261106"})
    rows, inputs, sample_hashes = [], {}, set()
    for arm, label in LABELS:
        measurements = []
        for seed in (1, 2):
            path = source / f"{arm}-seed{seed}" / "frozen-development/report.json"
            report = json.loads(path.read_text(encoding="utf8"))
            sample_hashes.add(report["row_ids_sha256"])
            inputs[path.relative_to(source).as_posix()] = sha(path)
            measurements.append(report)
        pair = collected["analysis"]["pairs"][arm]
        rows.append(dict(arm=arm, label=label, development=pair["scores"],
            mean_development=pair["mean_score"], teacher=pair["teacher"], ranks=pair["ranks"],
            lookup=[m["measurements"]["intact"] for m in measurements]))
    if len(sample_hashes) != 1:
        raise ValueError("The lookup plots must use the same development rows")
    sample = measurements[0]
    queries = sample["measurements"]["intact"]["all"]["queries"]
    captions = sample["measurements"]["intact"]["all"]["captions"]
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.7, 6.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1.]})
    y = np.arange(len(rows))
    reference = collected["analysis"]["pairs"]["N16"]["mean_score"]
    repeat = collected["analysis"]["hardware_repeat_floor"]
    axes[0].axvspan(reference-repeat, reference+repeat, color="#64748b", alpha=.12)
    axes[0].axvline(reference, color="#64748b", lw=1, ls="--")
    for i, item in enumerate(rows):
        color = "#64748b" if item["arm"] in ("R8", "N16", "N16-repeat") else "#0072b2"
        if item["arm"] in ("KALLP", "KALLP_N0", "KALLP_L8"):
            color = "#009e73"
        for ax in axes:
            if item["arm"] in ("N16", "H16"):
                ax.axhspan(i-.42, i+.42, color="#0072b2", alpha=.045)
        scores = item["development"]
        hits = [100*m["all"]["exact_record_top1"] for m in item["lookup"]]
        for ax, values in zip(axes, (scores, hits)):
            ax.plot(values, [i, i], color=color, alpha=.55, lw=1.5)
            ax.scatter(values[0], i-.075, marker="o", s=38, color=color, zorder=3)
            ax.scatter(values[1], i+.075, marker="s", s=34, color=color, zorder=3)
            ax.scatter(sum(values)/2, i, marker="|", s=130, color=color, linewidth=1.7, zorder=4)
    axes[0].set_yticks(y, [item["label"] for item in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlim(.655, .684)
    axes[0].set_xlabel("STS-B development Spearman")
    axes[0].set_title("Sentence similarity")
    axes[1].set_xlim(-2, 104)
    axes[1].set_xlabel("Correct earlier record as top attention position (%)")
    axes[1].set_title("Designated-head repeated-word lookup")
    for ax in axes:
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("One-million-row screen: sentence scores and lexical lookup", y=.98, fontsize=14)
    legend = [
        Line2D([], [], color="#334155", marker="o", ls="", label="Weight seed 1"),
        Line2D([], [], color="#334155", marker="s", ls="", label="Weight seed 2"),
        Line2D([], [], color="#334155", marker="|", ls="", markersize=10, label="Pair mean"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.61, .94), ncol=3, frameon=False)
    fig.text(.02, .04,
        f"Lookup: {queries:,} queried token positions in {captions:,} target-bearing captions from {sample['rows']:,} development rows. Only the configured head is scored.",
        fontsize=8.6)
    fig.text(.02, .015,
        "Left shading is the reference-repeat magnitude, not a confidence interval. Pale rows mark 4M follow-ups. Near ties follow the declared simplicity rule.",
        fontsize=8.6)
    fig.tight_layout(rect=(0, .09, 1, .9))
    for suffix in ("png", "svg"):
        path = output / f"structured_screen.{suffix}"
        options = {"metadata": {"Date": None}} if suffix == "svg" else {}
        fig.savefig(path, dpi=200, facecolor="white", **options)
        if suffix == "svg":
            path.write_bytes(b"\n".join(line.rstrip() for line in path.read_bytes().splitlines()) + b"\n")
    plt.close(fig)
    summary = dict(source_revision=collected["source_revision"], screen_config_id=collected["config_id"],
        hardware_repeat_floor=repeat, row_ids_sha256=next(iter(sample_hashes)), rows=rows,
        input_sha256=inputs, environment=dict(python=platform.python_version(),
            numpy=np.__version__, matplotlib=matplotlib.__version__),
        figures={p.name: sha(p) for p in output.glob("structured_screen.*")},
        claim_limit="Two weight seeds, one fixed phase draw; configured-head lookup and sentence capability are separate")
    path = ROOT / "docs/results/structured_screen_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")
    print(json.dumps(dict(figures=list(summary["figures"]), metrics=str(path), source_revision=collected["source_revision"]), indent=2))


if __name__ == "__main__":
    main()
