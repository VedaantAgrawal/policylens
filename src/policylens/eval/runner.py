"""Score a retriever against the golden set and produce a committable result.

Unanswerable questions are excluded from these retrieval metrics by
construction (recall/MRR/nDCG are undefined without a relevant set) — they
exist for the refusal-accuracy metric at the generation layer instead.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from policylens.eval.golden_set import GoldenQuestion
from policylens.eval.metrics import bootstrap_ci, ndcg_at_k, reciprocal_rank, recall_at_k

if TYPE_CHECKING:
    from policylens.retrieval.base import Retriever

METRIC_NAMES = ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
RETRIEVAL_K = 10


def _score_one(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "recall_at_5": recall_at_k(retrieved, relevant, k=5),
        "recall_at_10": recall_at_k(retrieved, relevant, k=10),
        "mrr": reciprocal_rank(retrieved, relevant),
        "ndcg_at_10": ndcg_at_k(retrieved, relevant, k=10),
    }


def run_eval(retriever: "Retriever", questions: list[GoldenQuestion]) -> dict:
    answerable = [q for q in questions if q.answerable]
    if not answerable:
        raise ValueError("golden set has no answerable questions to score against")

    per_query = []
    for q in answerable:
        retrieved = retriever.search(q.question, k=RETRIEVAL_K)
        scores = _score_one(retrieved, set(q.relevant_chunk_ids))
        per_query.append({"question_id": q.question_id, "category": q.category, **scores})

    aggregate = {}
    for metric in METRIC_NAMES:
        values = [pq[metric] for pq in per_query]
        mean, lower, upper = bootstrap_ci(values)
        aggregate[metric] = {"mean": mean, "ci_lower": lower, "ci_upper": upper}

    return {
        "retriever": retriever.name,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "num_answerable_questions": len(answerable),
        "num_unanswerable_questions": len(questions) - len(answerable),
        "aggregate": aggregate,
        "per_query": per_query,
    }
