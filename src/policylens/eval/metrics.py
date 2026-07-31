"""Retrieval metrics: recall@k, MRR, nDCG@k, and bootstrap confidence intervals.

All three ranking metrics assume binary relevance (a chunk either is or isn't in
a question's gold `relevant_chunk_ids` set) — the golden set has no graded
relevance judgments, so a graded-relevance nDCG would be inventing precision we
don't have.
"""

from __future__ import annotations

import math

import numpy as np


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        raise ValueError("recall_at_k is undefined for a query with no relevant chunks")
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        raise ValueError("reciprocal_rank is undefined for a query with no relevant chunks")
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        raise ValueError("ndcg_at_k is undefined for a query with no relevant chunks")

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(
    values: list[float], n_bootstrap: int = 10_000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) via percentile bootstrap over per-query scores."""
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")

    arr = np.array(values)
    rng = np.random.default_rng(seed)
    n = len(arr)
    resample_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        resample_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    lower = float(np.quantile(resample_means, alpha))
    upper = float(np.quantile(resample_means, 1 - alpha))
    return float(arr.mean()), lower, upper


def bootstrap_ci_delta(
    values_a: list[float], values_b: list[float], n_bootstrap: int = 10_000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Paired bootstrap CI for (mean(b) - mean(a)) over the same queries.

    Paired (not independent) resampling matters here: values_a and values_b
    come from scoring the *same* golden questions under two retrievers, so
    per-query performance is correlated and pairing preserves that instead of
    inflating the CI width as if the two runs were independent samples.
    """
    if len(values_a) != len(values_b):
        raise ValueError("paired bootstrap requires equal-length, aligned value lists")
    if not values_a:
        raise ValueError("bootstrap_ci_delta requires at least one value")

    a = np.array(values_a)
    b = np.array(values_b)
    n = len(a)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas[i] = b[idx].mean() - a[idx].mean()

    alpha = (1 - ci) / 2
    lower = float(np.quantile(deltas, alpha))
    upper = float(np.quantile(deltas, 1 - alpha))
    return float(b.mean() - a.mean()), lower, upper
