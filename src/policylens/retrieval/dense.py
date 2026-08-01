"""S1: dense retrieval over cached sentence-transformer embeddings.

Loads the embeddings built by `embed.py` rather than computing them here —
query-time cost is one embedding call plus a matrix multiply, not
re-embedding the corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from policylens.retrieval.embed import EMBEDDINGS_PATH, MODEL_NAME


class DenseRetriever:
    name = "s1_dense"

    def __init__(self, embeddings_path: Path = EMBEDDINGS_PATH):
        data = np.load(embeddings_path)
        self.chunk_ids: list[str] = list(data["chunk_ids"])
        self.vectors: np.ndarray = data["vectors"]  # pre-normalized, shape (N, dim)
        self._model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, k: int) -> list[str]:
        query_vec = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self.vectors @ query_vec
        top_indices = np.argsort(-scores)[:k]
        return [self.chunk_ids[i] for i in top_indices]
