"""Shared tool implementations, used by both the agent loop and the MCP server.

Kept as plain functions/classes independent of any tool-use framework
(no @beta_tool, no MCP decorators here) so the same retrieval logic isn't
duplicated across the two call sites — agent/loop.py wraps these for the
Anthropic tool runner, agent/mcp_server.py wraps them for MCP clients.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from policylens.retrieval.bm25 import Bm25Retriever
from policylens.retrieval.hybrid import HybridRetriever

if TYPE_CHECKING:
    from policylens.providers.base import ModelProvider

CHUNKS_PATH = Path("data/processed/chunks.jsonl")

_EXTRACT_SYSTEM_PROMPT = """You extract a single numeric fact from regulatory/financial \
source text. Respond with ONLY a single JSON object, no markdown fences, no other text:
{"found": true or false, "value": "the number as it appears, e.g. \\"$3,078,000\\" or \\"60 days\\"", "chunk_id": "the chunk_id the number came from", "context": "the sentence containing the number"}
Set "found" to false and leave the other fields empty strings if the source text doesn't \
contain the requested number. Never guess or estimate — only report a number that is \
literally present in the source text.
"""


class CorpusTools:
    """Loads the corpus once, then exposes the 4 tools as plain methods."""

    def __init__(self, chunks_path: Path = CHUNKS_PATH, bm25=None, hybrid=None):
        # Injectable for tests — a fake retriever with a .search(query, k) method
        # avoids loading the real BM25 index / sentence-transformer embeddings.
        self._bm25 = bm25 if bm25 is not None else Bm25Retriever(chunks_path=chunks_path)
        self._hybrid = hybrid if hybrid is not None else HybridRetriever(bm25=self._bm25)
        self._chunks_by_id: dict[str, dict] = {}
        self._chunks_by_doc: dict[str, list[dict]] = {}
        with chunks_path.open() as f:
            for line in f:
                chunk = json.loads(line)
                self._chunks_by_id[chunk["chunk_id"]] = chunk
                self._chunks_by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    def search_corpus(self, query: str, k: int = 5) -> list[dict]:
        """Search the whole corpus and return the top-k matching chunks."""
        chunk_ids = self._hybrid.search(query, k=k)
        return [self._chunk_summary(cid) for cid in chunk_ids if cid in self._chunks_by_id]

    def fetch_section(self, doc_id: str, section_heading: str | None = None, page: int | None = None) -> list[dict]:
        """Fetch a specific section (by heading substring) or page from a known document.

        Use this after search_corpus has identified which document has the answer,
        to pull the full surrounding section rather than a single chunk.
        """
        chunks = self._chunks_by_doc.get(doc_id, [])
        if section_heading:
            needle = section_heading.lower()
            chunks = [c for c in chunks if c.get("section_heading") and needle in c["section_heading"].lower()]
        elif page is not None:
            chunks = [c for c in chunks if c.get("page") == page]
        chunks = sorted(chunks, key=lambda c: c["chunk_index"])
        return [self._chunk_summary(c["chunk_id"]) for c in chunks]

    def compare_provisions(self, doc_ids: list[str], aspect: str, k_per_doc: int = 3) -> dict[str, list[dict]]:
        """Retrieve the passages most relevant to `aspect` from each of several documents.

        Use this to compare how multiple model laws, bulletins, or filings treat
        the same topic (e.g. how two NAIC model acts each define a term) — runs a
        search scoped to each document independently rather than pooling results,
        so a document with weaker matches doesn't get crowded out by a stronger one.
        """
        results: dict[str, list[dict]] = {}
        for doc_id in doc_ids:
            doc_chunk_ids = {c["chunk_id"] for c in self._chunks_by_doc.get(doc_id, [])}
            if not doc_chunk_ids:
                results[doc_id] = []
                continue
            # Search the whole corpus, then keep only hits within this doc — the
            # underlying BM25/dense indices are corpus-wide, not rebuilt per doc.
            ranked = self._hybrid.search(aspect, k=max(k_per_doc * 20, 50))
            matches = [cid for cid in ranked if cid in doc_chunk_ids][:k_per_doc]
            results[doc_id] = [self._chunk_summary(cid) for cid in matches]
        return results

    def extract_numeric_field(self, provider: "ModelProvider", doc_id: str, field_description: str) -> dict:
        """Extract a specific numeric fact (a dollar amount, a day count, a percentage,
        etc.) from one document, with its source chunk_id, or report it wasn't found.
        """
        doc_chunk_ids = {c["chunk_id"] for c in self._chunks_by_doc.get(doc_id, [])}
        if not doc_chunk_ids:
            return {"found": False, "value": "", "chunk_id": "", "context": f"unknown doc_id: {doc_id}"}

        ranked = self._bm25.search(field_description, k=200)
        candidates = [cid for cid in ranked if cid in doc_chunk_ids][:5]
        if not candidates:
            return {"found": False, "value": "", "chunk_id": "", "context": "no matching passages in this document"}

        lines = [f"Looking for: {field_description}", "", "Source passages:"]
        for cid in candidates:
            chunk = self._chunks_by_id[cid]
            lines.append(f'<source chunk_id="{cid}">')
            lines.append(chunk["text"])
            lines.append("</source>")
        user_message = "\n".join(lines)

        completion = provider.complete(system=_EXTRACT_SYSTEM_PROMPT, user_message=user_message, max_tokens=300)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", completion.text.strip(), flags=re.MULTILINE)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"found": False, "value": "", "chunk_id": "", "context": "extraction parse error"}
        return {
            "found": bool(parsed.get("found")),
            "value": str(parsed.get("value", "")),
            "chunk_id": str(parsed.get("chunk_id", "")),
            "context": str(parsed.get("context", "")),
        }

    def _chunk_summary(self, chunk_id: str) -> dict:
        chunk = self._chunks_by_id[chunk_id]
        return {
            "chunk_id": chunk_id,
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "section_heading": chunk.get("section_heading"),
            "page": chunk.get("page"),
            "text": chunk["text"],
        }
