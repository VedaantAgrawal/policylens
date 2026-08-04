"""Build the ablation table from committed eval_results/*.json — no hand-typed numbers.

This is what `make eval` prints at the end and what the README's results
table is generated from. Every number here traces back to a JSON file
produced by run_stage.py / run_generation_stage.py.
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_RESULTS_DIR = Path("eval_results")

RETRIEVAL_STAGES = ["s0_bm25", "s1_dense", "s2_hybrid", "s3_rerank"]
RETRIEVAL_METRICS = ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
GENERATION_METRICS = ["citation_precision", "groundedness", "refusal_accuracy", "false_refusal_rate"]


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def render_retrieval_table() -> str:
    rows = []
    header = "| Stage | " + " | ".join(RETRIEVAL_METRICS) + " |"
    sep = "|---|" + "---|" * len(RETRIEVAL_METRICS)
    rows.append(header)
    rows.append(sep)

    for stage in RETRIEVAL_STAGES:
        result = _load(EVAL_RESULTS_DIR / f"{stage}.json")
        if result is None:
            rows.append(f"| {stage} | " + " | ".join(["(not run)"] * len(RETRIEVAL_METRICS)) + " |")
            continue
        cells = []
        for metric in RETRIEVAL_METRICS:
            stats = result["aggregate"][metric]
            cells.append(f"{stats['mean']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
        rows.append(f"| {stage} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def render_generation_table() -> str:
    rows = []
    header = "| Stage | " + " | ".join(GENERATION_METRICS) + " |"
    sep = "|---|" + "---|" * len(GENERATION_METRICS)
    rows.append(header)
    rows.append(sep)

    for stage in RETRIEVAL_STAGES:
        result = _load(EVAL_RESULTS_DIR / f"generation_{stage}.json")
        if result is None:
            rows.append(f"| {stage} | " + " | ".join(["(not run)"] * len(GENERATION_METRICS)) + " |")
            continue
        cells = []
        for metric in GENERATION_METRICS:
            stats = result["aggregate"][metric]
            if stats["mean"] is None:
                cells.append("n/a")
            else:
                cells.append(f"{stats['mean']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}] (n={stats['n']})")
        rows.append(f"| {stage} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def main() -> None:
    print("## Retrieval ablation (95% bootstrap CI)\n")
    print(render_retrieval_table())
    print("\n## Generation quality (95% bootstrap CI)\n")
    print(render_generation_table())


if __name__ == "__main__":
    main()
