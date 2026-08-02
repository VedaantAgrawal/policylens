"""Minimal FastAPI serving layer: query the RAG pipeline, and expose eval results.

Retriever/provider instances are built once at module load, not per-request
— DenseRetriever alone costs ~1s to load the embedding model, which would
otherwise tax every single query.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from policylens.generation.generate import generate_answer
from policylens.providers.anthropic_provider import AnthropicProvider
from policylens.retrieval.hybrid import HybridRetriever

app = FastAPI(title="PolicyLens API", description="Agentic RAG over insurance and regulatory documents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EVAL_RESULTS_DIR = Path("eval_results")
RETRIEVAL_K = 5

_retriever = HybridRetriever()
_generation_provider = AnthropicProvider()

_chunks_by_id: dict[str, dict] = {}
with CHUNKS_PATH.open() as f:
    for line in f:
        chunk = json.loads(line)
        _chunks_by_id[chunk["chunk_id"]] = chunk


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    chunk_id: str
    title: str
    url: str
    section_heading: str | None = None


class QueryResponse(BaseModel):
    answerable: bool
    answer: str
    citations: list[Citation]
    retrieved_chunk_ids: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    retrieved_ids = _retriever.search(request.question, k=RETRIEVAL_K)
    chunks = [_chunks_by_id[cid] for cid in retrieved_ids if cid in _chunks_by_id]
    result = generate_answer(_generation_provider, request.question, chunks)

    citations = [
        Citation(
            chunk_id=cid,
            title=_chunks_by_id[cid]["title"],
            url=_chunks_by_id[cid]["url"],
            section_heading=_chunks_by_id[cid].get("section_heading"),
        )
        for cid in result.citations
        if cid in _chunks_by_id
    ]

    return QueryResponse(
        answerable=result.answerable,
        answer=result.answer,
        citations=citations,
        retrieved_chunk_ids=retrieved_ids,
    )


@app.get("/eval")
def eval_results() -> dict:
    """Raw committed eval_results/*.json, for the static dashboard to render."""
    results = {}
    for path in EVAL_RESULTS_DIR.glob("*.json"):
        results[path.stem] = json.loads(path.read_text())
    return results
