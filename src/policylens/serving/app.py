"""Minimal FastAPI serving layer: query the RAG pipeline, and expose eval results.

Retriever/provider instances are built once at module load, not per-request
— DenseRetriever alone costs ~1s to load the embedding model, which would
otherwise tax every single query.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from policylens.agent.loop import run_agent
from policylens.agent.tools import CorpusTools
from policylens.eval.pricing import estimate_cost_usd
from policylens.generation.generate import generate_answer
from policylens.providers.anthropic_provider import AnthropicProvider, DEFAULT_MODEL
from policylens.retrieval.bm25 import Bm25Retriever
from policylens.retrieval.hybrid import HybridRetriever

app = FastAPI(title="PolicyLens API", description="Agentic RAG over insurance and regulatory documents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_response_time_header(request: Request, call_next):
    """Surfaces per-request latency for live demo purposes — same wall-clock
    metric `make eval-latency-cost` reports p50/p95 over, just visible on
    every single response instead of only in a committed eval run."""
    start = time.monotonic()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = f"{(time.monotonic() - start) * 1000:.1f}"
    return response

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EVAL_RESULTS_DIR = Path("eval_results")
RETRIEVAL_K = 5

_bm25 = Bm25Retriever()
_retriever = HybridRetriever(bm25=_bm25)
_generation_provider = AnthropicProvider()
_extraction_provider = AnthropicProvider(model="claude-haiku-4-5")
# Shares _bm25/_retriever with /query above rather than re-loading BM25 and the
# dense embedding model a second time — DenseRetriever alone costs real startup
# time and memory, not worth paying twice for the same corpus.
_corpus_tools = CorpusTools(bm25=_bm25, hybrid=_retriever)

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
    cost_usd: float


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

    cost_usd = estimate_cost_usd(DEFAULT_MODEL, result.input_tokens, result.output_tokens)

    return QueryResponse(
        answerable=result.answerable,
        answer=result.answer,
        citations=citations,
        retrieved_chunk_ids=retrieved_ids,
        cost_usd=cost_usd,
    )


class AgentQueryResponse(BaseModel):
    answer: str
    trace: list[dict]
    tool_call_count: int


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(request: QueryRequest) -> AgentQueryResponse:
    """Multi-step agent: plans tool calls (search_corpus, fetch_section,
    compare_provisions, extract_numeric_field) rather than single-shot
    retrieve-then-generate like /query. Returns the full structured trace of
    what it did, not just the final answer — see agent/loop.py."""
    result = run_agent(request.question, _generation_provider, _extraction_provider, _corpus_tools)
    return AgentQueryResponse(answer=result.answer, trace=result.trace, tool_call_count=result.tool_call_count)


@app.get("/eval")
def eval_results() -> dict:
    """Raw committed eval_results/*.json, for the static dashboard to render."""
    results = {}
    for path in EVAL_RESULTS_DIR.glob("*.json"):
        results[path.stem] = json.loads(path.read_text())
    return results
