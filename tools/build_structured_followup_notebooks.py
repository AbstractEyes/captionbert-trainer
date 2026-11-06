"""Build the locked larger-stage/recovery notebooks without changing v0.3.0."""
import json
from pathlib import Path

from build_structured_notebooks import SETUP, PREFLIGHT, TENSORBOARD, RELAY, cell

ROOT = Path(__file__).resolve().parents[1]


def main():
    for lane, seed in (("A", 1), ("B", 2)):
        introduction = f"""# CaptionBERT four-million-row follow-up — seed {seed}

Use **G4 / RTX PRO6000** and enable notebook access to **HF_TOKEN**. Run setup,
the full-batch gate and TensorBoard, then the larger experiments and frozen
follow-up. The two training cells run sequentially; queue the frozen cell
behind the optimizer cell. Keep the v0.3.0 training environment.

This notebook restores the **locked, existing campaign**, including compatible
partial checkpoints. It verifies completed runs before skipping them. It runs
full4M hierarchical, full4M relative, then the hierarchical 1M matched-exposure
control, all under seed {seed}. Do not run another notebook with this seed
concurrently. The other notebook owns seed {3-seed}.

The final frozen cell restores the earlier completed binding checkpoint even
after a fresh runtime. It preserves the original intact reports, then measures
frozen interventions, reconstructed initialization, safe AutoModel parity and
synthetic shapes through 8192. Shape checks do not establish 8K semantic quality.

Training stays pinned to **v0.3.0**. The helper and probe source revisions are
explicit inside their cells. Interrupted sessions repeat setup/gate/TensorBoard
and these cells; they do not rerun the original screen queue.

Seed1 also preserves the prepared-data manifests and runs a bounded CPU-only
corpus/address census after the frozen probes. It changes no training recipe.

[Protocol](https://github.com/AbstractEyes/captionbert-trainer/blob/codex/structured-rotary-overnight/docs/structured_campaign_protocol.md)
· [Results and artifacts](https://huggingface.co/AbstractPhil/captionbert-8192-rotary)
"""
        sources = [
            ("markdown", introduction),
            ("code", SETUP.replace("RELEASE", "v0.3.0").replace("LANE", str(seed-1))),
            ("code", PREFLIGHT.replace("RELAY", RELAY)),
            ("code", TENSORBOARD),
            ("code", (ROOT / f"notebooks/cells/larger_{lane}.py").read_text(encoding="utf8")),
            ("code", (ROOT / f"notebooks/cells/frozen_final_{lane}.py").read_text(encoding="utf8")),
        ]
        if lane == "A":
            sources.append(("code", (ROOT / "notebooks/cells/ship_provenance_A.py").read_text(encoding="utf8")))
        cells = []
        for index, (kind, source) in enumerate(sources):
            if kind == "code":
                compile(source, f"followup-{lane}-cell{index}", "exec")
            cells.append(cell(kind, source, f"structured-followup-{lane}-{index}"))
        notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
            metadata={"accelerator": "GPU", "colab": {"name": f"CaptionBERT larger follow-up seed {seed}"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
        path = ROOT / f"notebooks/captionbert_structured_followup_seed{seed}.ipynb"
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n",
                        encoding="utf8", newline="\n")
        print(path)


if __name__ == "__main__":
    main()
