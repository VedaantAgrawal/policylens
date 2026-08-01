"""End-to-end generation eval: retrieve -> generate -> judge, over the full golden set.

Unlike runner.py (pure retrieval metrics on answerable questions only), this
scores the whole golden set — answerable questions get citation precision +
groundedness, unanswerable questions get refusal accuracy. Both subsets
matter for the same reason: a system that never refuses looks great on
groundedness and terrible on refusal accuracy, and vice versa.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from policylens.eval.golden_set import GoldenQuestion
from policylens.eval.judge import JUDGE_MODEL, judge_groundedness
from policylens.eval.metrics import bootstrap_ci
from policylens.generation.generate import generate_answer

if TYPE_CHECKING:
    from policylens.providers.base import ModelProvider
    from policylens.retrieval.base import Retriever

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
RETRIEVAL_K = 5  # chunks fed to generation; smaller than the RETRIEVAL_K used for pure retrieval metrics


def _load_chunks_by_id() -> dict[str, dict]:
    chunks_by_id = {}
    with CHUNKS_PATH.open() as f:
        for line in f:
            chunk = json.loads(line)
            chunks_by_id[chunk["chunk_id"]] = chunk
    return chunks_by_id


def run_generation_eval(
    retriever: "Retriever",
    generation_provider: "ModelProvider",
    judge_provider: "ModelProvider",
    questions: list[GoldenQuestion],
) -> dict:
    chunks_by_id = _load_chunks_by_id()

    per_query = []
    for q in questions:
        retrieved_ids = retriever.search(q.question, k=RETRIEVAL_K)
        chunks = [chunks_by_id[cid] for cid in retrieved_ids if cid in chunks_by_id]
        result = generate_answer(generation_provider, q.question, chunks)

        record = {
            "question_id": q.question_id,
            "category": q.category,
            "gold_answerable": q.answerable,
            "model_answerable": result.answerable,
            "citation_precision": result.citation_precision,
            "parse_error": result.parse_error,
            "grounded": None,
        }

        if q.answerable and result.answerable and result.citations and not result.parse_error:
            verdict = judge_groundedness(judge_provider, q.question, result, chunks_by_id)
            record["grounded"] = verdict.grounded

        per_query.append(record)

    unanswerable = [r for r in per_query if not r["gold_answerable"]]
    answerable = [r for r in per_query if r["gold_answerable"]]

    refusal_values = [1.0 if r["model_answerable"] is False else 0.0 for r in unanswerable]
    citation_precision_values = [r["citation_precision"] for r in answerable if r["citation_precision"] is not None]
    grounded_values = [1.0 if r["grounded"] else 0.0 for r in answerable if r["grounded"] is not None]
    false_refusal_values = [1.0 if r["model_answerable"] is False else 0.0 for r in answerable]

    aggregate = {}
    for name, values in [
        ("refusal_accuracy", refusal_values),
        ("citation_precision", citation_precision_values),
        ("groundedness", grounded_values),
        ("false_refusal_rate", false_refusal_values),
    ]:
        if values:
            mean, lower, upper = bootstrap_ci(values)
            aggregate[name] = {"mean": mean, "ci_lower": lower, "ci_upper": upper, "n": len(values)}
        else:
            aggregate[name] = {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0}

    return {
        "retriever": retriever.name,
        "generation_model": getattr(generation_provider, "_model", "unknown"),
        "judge_model": JUDGE_MODEL,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "num_answerable_questions": len(answerable),
        "num_unanswerable_questions": len(unanswerable),
        "aggregate": aggregate,
        "per_query": per_query,
    }
