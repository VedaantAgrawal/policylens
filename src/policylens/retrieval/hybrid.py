"""S2: hybrid fusion of BM25 (S0) and dense (S1) rankings via Reciprocal Rank Fusion.

RRF combines two ranked lists using only rank position, not raw scores —
that sidesteps having to normalize BM25 scores (unbounded, corpus-frequency
dependent) against cosine similarities (bounded [-1, 1]), which a weighted
score-sum would require and which is easy to get wrong silently.
"""

from __future__ import annotations

from policylens.retrieval.bm25 import Bm25Retriever
from policylens.retrieval.dense import DenseRetriever

RRF_K = 60  # standard constant from the original RRF paper (Cormack et al. 2009)
CANDIDATE_POOL_SIZE = 50  # candidates pulled from each retriever before fusion


class HybridRetriever:
    name = "s2_hybrid"

    def __init__(self, bm25: Bm25Retriever | None = None, dense: DenseRetriever | None = None):
        self._bm25 = bm25 or Bm25Retriever()
        self._dense = dense or DenseRetriever()

    def search(self, query: str, k: int) -> list[str]:
        bm25_ranked = self._bm25.search(query, k=CANDIDATE_POOL_SIZE)
        dense_ranked = self._dense.search(query, k=CANDIDATE_POOL_SIZE)

        scores: dict[str, float] = {}
        for rank, chunk_id in enumerate(bm25_ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, chunk_id in enumerate(dense_ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:k]]
