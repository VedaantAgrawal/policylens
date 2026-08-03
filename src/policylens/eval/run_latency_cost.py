"""CLI: measure end-to-end retrieval+generation latency and cost per query.

Usage: uv run python -m policylens.eval.run_latency_cost s2_hybrid

Runs every answerable golden question through retrieval + generation (not
judging — judge cost is an eval-time cost, not something a real user's
query incurs) and reports p50/p95 latency and mean/total cost per query.
This is the "Ops" half of the ablation table the original project brief
called for and Day 1-3 didn't get to.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from policylens.eval.golden_set import load_golden_set
from policylens.eval.pricing import estimate_cost_usd
from policylens.generation.generate import generate_answer
from policylens.providers.anthropic_provider import AnthropicProvider, DEFAULT_MODEL
from policylens.retrieval.bm25 import Bm25Retriever
from policylens.retrieval.dense import DenseRetriever
from policylens.retrieval.hybrid import HybridRetriever

EVAL_RESULTS_DIR = Path("eval_results")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
RETRIEVAL_K = 5

_RETRIEVER_FACTORIES = {
    "s0_bm25": Bm25Retriever,
    "s1_dense": DenseRetriever,
    "s2_hybrid": HybridRetriever,
}


def _load_chunks_by_id() -> dict[str, dict]:
    chunks_by_id = {}
    with CHUNKS_PATH.open() as f:
        for line in f:
            chunk = json.loads(line)
            chunks_by_id[chunk["chunk_id"]] = chunk
    return chunks_by_id


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _RETRIEVER_FACTORIES:
        print(f"Usage: python -m policylens.eval.run_latency_cost <stage>\nAvailable: {', '.join(_RETRIEVER_FACTORIES)}")
        sys.exit(1)

    stage = sys.argv[1]
    retriever = _RETRIEVER_FACTORIES[stage]()
    generation_provider = AnthropicProvider()
    chunks_by_id = _load_chunks_by_id()

    questions = [q for q in load_golden_set() if q.answerable]

    per_query = []
    for q in questions:
        retrieval_start = time.monotonic()
        retrieved_ids = retriever.search(q.question, k=RETRIEVAL_K)
        retrieval_latency = time.monotonic() - retrieval_start

        chunks = [chunks_by_id[cid] for cid in retrieved_ids if cid in chunks_by_id]
        result = generate_answer(generation_provider, q.question, chunks)

        cost = estimate_cost_usd(DEFAULT_MODEL, result.input_tokens, result.output_tokens)
        per_query.append(
            {
                "question_id": q.question_id,
                "retrieval_latency_seconds": retrieval_latency,
                "generation_latency_seconds": result.latency_seconds,
                "total_latency_seconds": retrieval_latency + result.latency_seconds,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": cost,
            }
        )

    total_latencies = [r["total_latency_seconds"] for r in per_query]
    costs = [r["cost_usd"] for r in per_query]

    summary = {
        "stage": stage,
        "generation_model": DEFAULT_MODEL,
        "num_queries": len(per_query),
        "latency_seconds": {
            "p50": _percentile(total_latencies, 50),
            "p95": _percentile(total_latencies, 95),
            "mean": float(np.mean(total_latencies)),
            "retrieval_mean": float(np.mean([r["retrieval_latency_seconds"] for r in per_query])),
            "generation_mean": float(np.mean([r["generation_latency_seconds"] for r in per_query])),
        },
        "cost_usd": {
            "mean_per_query": float(np.mean(costs)),
            "total": float(np.sum(costs)),
        },
        "per_query": per_query,
    }

    EVAL_RESULTS_DIR.mkdir(exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"latency_cost_{stage}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\nlatency_cost_{stage} results (n={summary['num_queries']}):")
    print(f"  p50 latency        {summary['latency_seconds']['p50']:.2f}s")
    print(f"  p95 latency        {summary['latency_seconds']['p95']:.2f}s")
    print(f"  mean latency       {summary['latency_seconds']['mean']:.2f}s  "
          f"(retrieval {summary['latency_seconds']['retrieval_mean']:.3f}s + "
          f"generation {summary['latency_seconds']['generation_mean']:.2f}s)")
    print(f"  mean cost/query    ${summary['cost_usd']['mean_per_query']:.5f}")
    print(f"  total cost         ${summary['cost_usd']['total']:.4f}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
