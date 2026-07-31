"""CLI: score one retrieval stage against the golden set and commit the result.

Usage: uv run python -m policylens.eval.run_stage s0_bm25
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from policylens.eval.golden_set import load_golden_set
from policylens.eval.runner import run_eval

EVAL_RESULTS_DIR = Path("eval_results")

_STAGE_FACTORIES = {}


def _register_stages():
    from policylens.retrieval.bm25 import Bm25Retriever

    _STAGE_FACTORIES["s0_bm25"] = Bm25Retriever


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("s0_bm25",):
        print(f"Usage: python -m policylens.eval.run_stage <stage>\nAvailable: s0_bm25")
        sys.exit(1)

    _register_stages()
    stage = sys.argv[1]
    retriever = _STAGE_FACTORIES[stage]()

    questions = load_golden_set()
    result = run_eval(retriever, questions)

    EVAL_RESULTS_DIR.mkdir(exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"{stage}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n{stage} results ({result['num_answerable_questions']} answerable questions):")
    for metric, stats in result["aggregate"].items():
        print(f"  {metric:15s} mean={stats['mean']:.3f}  95% CI=[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
