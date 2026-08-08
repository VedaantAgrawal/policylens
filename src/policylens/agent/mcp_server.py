"""MCP server exposing PolicyLens's 4 corpus tools plus the full agent loop.

All 4 originally-scoped tools (search_corpus, fetch_section,
compare_provisions, extract_numeric_field) are real implementations backed
by CorpusTools — the same code the FastAPI /agent/query endpoint and the
in-process agent loop use, not a separate reimplementation. Run it with
`uv run mcp dev src/policylens/agent/mcp_server.py`, or point a Claude
Desktop / Claude Code MCP config at `uv run python -m policylens.agent.mcp_server`.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from policylens.agent.tools import CorpusTools
from policylens.providers.anthropic_provider import AnthropicProvider

server = MCPServer(
    name="policylens",
    instructions="Search and read PolicyLens's corpus of SEC 10-Ks, NAIC model laws, "
    "and state insurance department bulletins. Start with search_corpus to find "
    "candidate documents, then fetch_section to read a specific section in full, "
    "compare_provisions to compare how several documents treat the same topic, or "
    "extract_numeric_field to pull one specific number out of a document.",
)

_tools: CorpusTools | None = None
_extraction_provider: AnthropicProvider | None = None


def _ensure_loaded() -> None:
    global _tools, _extraction_provider
    if _tools is not None:
        return
    _tools = CorpusTools()
    _extraction_provider = AnthropicProvider(model="claude-haiku-4-5")


@server.tool()
def search_corpus(query: str, k: int = 5) -> list[dict]:
    """Search the whole insurance/regulatory corpus and return the top-k matching chunks.

    Each result includes the chunk_id (needed to cite it), doc_id, title, section
    heading, and the chunk text itself.
    """
    _ensure_loaded()
    assert _tools is not None
    return _tools.search_corpus(query, k=k)


@server.tool()
def fetch_section(doc_id: str, section_heading: str | None = None, page: int | None = None) -> list[dict]:
    """Fetch a specific section (by heading substring) or page from a document already
    identified by search_corpus, to read its full surrounding context.
    """
    _ensure_loaded()
    assert _tools is not None
    return _tools.fetch_section(doc_id, section_heading=section_heading, page=page)


@server.tool()
def compare_provisions(doc_ids: list[str], aspect: str, k_per_doc: int = 3) -> dict[str, list[dict]]:
    """Retrieve the passages most relevant to a topic from each of several documents,
    for comparing how they each treat the same point.
    """
    _ensure_loaded()
    assert _tools is not None
    return _tools.compare_provisions(doc_ids, aspect, k_per_doc=k_per_doc)


@server.tool()
def extract_numeric_field(doc_id: str, field_description: str) -> dict:
    """Extract one specific numeric fact (a dollar amount, day count, percentage, or
    date) from a single document, with its source chunk_id.
    """
    _ensure_loaded()
    assert _tools is not None
    assert _extraction_provider is not None
    return _tools.extract_numeric_field(_extraction_provider, doc_id, field_description)


if __name__ == "__main__":
    server.run()
