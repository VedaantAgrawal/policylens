"""MCP server exposing PolicyLens retrieval as a tool.

Scope note: the full design called for search_corpus, fetch_section,
compare_provisions, and extract_numeric_field, with a planning agent loop on
top. Under a 3-day timeline, only search_corpus is implemented — a thin,
real wrapper over the same HybridRetriever the FastAPI /query endpoint uses,
not a stub that returns fake data. This makes the cut concrete: run it with
`uv run mcp dev src/policylens/agent/mcp_server.py`, or point a Claude
Desktop / Claude Code MCP config at `uv run python -m policylens.agent.mcp_server`.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from policylens.retrieval.hybrid import HybridRetriever

CHUNKS_PATH = Path("data/processed/chunks.jsonl")

server = MCPServer(
    name="policylens",
    instructions="Search PolicyLens's corpus of SEC 10-Ks, NAIC model laws, and state "
    "insurance department bulletins for passages relevant to a query.",
)

_retriever: HybridRetriever | None = None
_chunks_by_id: dict[str, dict] = {}


def _ensure_loaded() -> None:
    global _retriever
    if _retriever is not None:
        return
    _retriever = HybridRetriever()
    with CHUNKS_PATH.open() as f:
        for line in f:
            chunk = json.loads(line)
            _chunks_by_id[chunk["chunk_id"]] = chunk


@server.tool()
def search_corpus(query: str, k: int = 5) -> list[dict]:
    """Search the insurance/regulatory corpus and return the top-k matching chunks.

    Each result includes the chunk_id (needed to cite it), title, section
    heading, and the chunk text itself.
    """
    _ensure_loaded()
    assert _retriever is not None
    chunk_ids = _retriever.search(query, k=k)
    return [
        {
            "chunk_id": cid,
            "title": _chunks_by_id[cid]["title"],
            "section_heading": _chunks_by_id[cid].get("section_heading"),
            "text": _chunks_by_id[cid]["text"],
        }
        for cid in chunk_ids
        if cid in _chunks_by_id
    ]


if __name__ == "__main__":
    server.run()
