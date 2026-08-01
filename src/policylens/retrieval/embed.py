"""Precompute chunk embeddings once, cache to disk.

Embedding all 6,500+ chunks through a transformer takes real wall-clock
time, so this is a separate build step (`make embed`) rather than something
DenseRetriever redoes on every eval run — the eval harness should be
querying a cache, not recomputing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npz")

MODEL_NAME = "all-MiniLM-L6-v2"


def build_embeddings() -> None:
    chunk_ids = []
    texts = []
    with CHUNKS_PATH.open() as f:
        for line in f:
            chunk = json.loads(line)
            chunk_ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])

    print(f"Embedding {len(texts)} chunks with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBEDDINGS_PATH, chunk_ids=np.array(chunk_ids), vectors=vectors.astype(np.float32))
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_embeddings()
