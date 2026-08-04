"""S3: cross-encoder rerank on top of S2 hybrid candidates.

A bi-encoder (S1's dense retriever) embeds the query and each chunk
independently and compares vectors — fast, but the model never sees the
query and chunk together. A cross-encoder scores each (query, chunk) pair
jointly through one forward pass, which is far more accurate but too slow
to run over the whole corpus — hence "retrieve wide with something cheap,
then rerank a small candidate pool with something expensive," the standard
two-stage pattern this stage implements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from policylens.retrieval.hybrid import HybridRetriever

if TYPE_CHECKING:
    from policylens.retrieval.base import Retriever

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_POOL_SIZE = 20  # wide enough that reranking can recover a relevant chunk S2 ranked outside top-k


class CrossEncoderRerankRetriever:
    name = "s3_rerank"

    def __init__(
        self,
        base_retriever: "Retriever | None" = None,
        cross_encoder=None,
        chunks_path: Path = CHUNKS_PATH,
    ):
        self._base = base_retriever or HybridRetriever()
        self._cross_encoder = cross_encoder or CrossEncoder(MODEL_NAME)
        self._text_by_id: dict[str, str] = {}
        with chunks_path.open() as f:
            for line in f:
                chunk = json.loads(line)
                self._text_by_id[chunk["chunk_id"]] = chunk["text"]

    def search(self, query: str, k: int) -> list[str]:
        candidates = [c for c in self._base.search(query, k=CANDIDATE_POOL_SIZE) if c in self._text_by_id]
        if not candidates:
            return []
        pairs = [(query, self._text_by_id[cid]) for cid in candidates]
        scores = self._cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:k]]
