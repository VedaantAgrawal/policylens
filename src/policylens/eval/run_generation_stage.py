"""CLI: run the full generate+judge pipeline against the golden set and commit the result.

Usage: uv run python -m policylens.eval.run_generation_stage s2_hybrid
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from policylens.eval.generation_runner import run_generation_eval
from policylens.eval.golden_set import load_golden_set
from policylens.eval.judge import JUDGE_MODEL
from policylens.providers.anthropic_provider import AnthropicProvider
from policylens.retrieval.bm25 import Bm25Retriever
from policylens.retrieval.dense import DenseRetriever
from policylens.retrieval.hybrid import HybridRetriever

EVAL_RESULTS_DIR = Path("eval_results")

_RETRIEVER_FACTORIES = {
    "s0_bm25": Bm25Retriever,
    "s1_dense": DenseRetriever,
    "s2_hybrid": HybridRetriever,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _RETRIEVER_FACTORIES:
        print(f"Usage: python -m policylens.eval.run_generation_stage <stage>\nAvailable: {', '.join(_RETRIEVER_FACTORIES)}")
        sys.exit(1)

    stage = sys.argv[1]
    retriever = _RETRIEVER_FACTORIES[stage]()
    generation_provider = AnthropicProvider()  # defaults to claude-sonnet-5
    judge_provider = AnthropicProvider(model=JUDGE_MODEL)

    questions = load_golden_set()
    result = run_generation_eval(retriever, generation_provider, judge_provider, questions)

    EVAL_RESULTS_DIR.mkdir(exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"generation_{stage}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\ngeneration_{stage} results:")
    for metric, stats in result["aggregate"].items():
        if stats["mean"] is None:
            print(f"  {metric:20s} n=0 (no scoreable questions)")
        else:
            print(
                f"  {metric:20s} mean={stats['mean']:.3f}  "
                f"95% CI=[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]  n={stats['n']}"
            )
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
