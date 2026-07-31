"""S0 baseline: BM25 over chunk text. No embeddings, no LLM calls.

This is deliberately the same retrieval approach shipped in production at a
past employer — the point of the ablation is to show, with statistics, how
much each later upgrade (dense, hybrid, rerank) buys over what "the boring
thing that already works" gets you.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/processed/chunks.jsonl")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Bm25Retriever:
    name = "s0_bm25"

    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunk_ids: list[str] = []
        corpus_tokens: list[list[str]] = []
        with chunks_path.open() as f:
            for line in f:
                chunk = json.loads(line)
                self.chunk_ids.append(chunk["chunk_id"])
                corpus_tokens.append(tokenize(chunk["text"]))
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunk_ids[i] for i in ranked_indices]
