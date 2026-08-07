"""Paired bootstrap significance test between retrieval stages.

The ablation table's per-stage bootstrap CIs (in run_stage.py) tell you the
uncertainty in each stage's own mean — they do NOT tell you whether stage B
is significantly better than stage A, because two overlapping CIs can still
hide a significant paired difference (or two non-overlapping ones can be
noise, in principle). This script runs the actual paired test — bootstrap_ci_delta
on the same aligned per-query scores — rather than eyeballing CI overlap,
which is what "statistically significant" claims in the README must be
backed by.

Usage: uv run python -m policylens.eval.stage_deltas
"""

from __future__ import annotations

import json
from pathlib import Path

from policylens.eval.metrics import bootstrap_ci_delta

EVAL_RESULTS_DIR = Path("eval_results")
METRICS = ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
STAGE_PAIRS = [
    ("s0_bm25", "s1_dense"),
    ("s0_bm25", "s2_hybrid"),
    ("s2_hybrid", "s3_rerank"),
    ("s0_bm25", "s3_rerank"),
]


def _load_per_query(stage: str) -> dict[str, dict]:
    data = json.loads((EVAL_RESULTS_DIR / f"{stage}.json").read_text())
    return {r["question_id"]: r for r in data["per_query"]}


def compute_deltas() -> dict:
    results = {}
    for stage_a, stage_b in STAGE_PAIRS:
        pq_a = _load_per_query(stage_a)
        pq_b = _load_per_query(stage_b)
        shared_ids = sorted(set(pq_a) & set(pq_b))
        if len(shared_ids) != len(pq_a) or len(shared_ids) != len(pq_b):
            raise ValueError(
                f"{stage_a} and {stage_b} were scored against different question sets "
                f"({len(pq_a)} vs {len(pq_b)}) — re-run both against the same golden set before comparing"
            )

        pair_key = f"{stage_a}->{stage_b}"
        results[pair_key] = {}
        for metric in METRICS:
            values_a = [pq_a[qid][metric] for qid in shared_ids]
            values_b = [pq_b[qid][metric] for qid in shared_ids]
            delta, lower, upper = bootstrap_ci_delta(values_a, values_b)
            results[pair_key][metric] = {
                "delta": delta,
                "ci_lower": lower,
                "ci_upper": upper,
                "significant": lower > 0 or upper < 0,
                "n": len(shared_ids),
            }
    return results


def main() -> None:
    results = compute_deltas()
    out_path = EVAL_RESULTS_DIR / "stage_deltas.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("## Paired bootstrap deltas (95% CI, 10k resamples, positive = stage B is better)\n")
    for pair_key, metrics in results.items():
        print(f"### {pair_key}")
        for metric, stats in metrics.items():
            sig = "significant" if stats["significant"] else "not significant"
            print(f"  {metric:15s} delta={stats['delta']:+.3f}  95% CI=[{stats['ci_lower']:+.3f}, {stats['ci_upper']:+.3f}]  ({sig})")
        print()

    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
