import json

from policylens.agent.tools import CorpusTools
from policylens.providers.base import CompletionResult


class FakeRetriever:
    def __init__(self, ordered_ids):
        self.name = "fake"
        self._ordered_ids = ordered_ids

    def search(self, query: str, k: int) -> list[str]:
        return self._ordered_ids[:k]


class FakeProvider:
    name = "fake"

    def __init__(self, response: str):
        self._response = response

    def complete(self, *, system: str, user_message: str, max_tokens: int = 300) -> CompletionResult:
        return CompletionResult(text=self._response, input_tokens=50, output_tokens=20, latency_seconds=0.1)


def _write_chunks(tmp_path, chunks):
    path = tmp_path / "chunks.jsonl"
    with path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return path


CHUNKS = [
    {"chunk_id": "docA_0", "doc_id": "docA", "title": "Doc A", "section_heading": "Section 1. Purpose", "page": 1, "chunk_index": 0, "text": "Purpose text."},
    {"chunk_id": "docA_1", "doc_id": "docA", "title": "Doc A", "section_heading": "Section 2. Definitions", "page": 1, "chunk_index": 1, "text": "Definitions text."},
    {"chunk_id": "docA_2", "doc_id": "docA", "title": "Doc A", "section_heading": "Section 2. Definitions", "page": 2, "chunk_index": 2, "text": "More definitions, including a $3,078,000 assessment."},
    {"chunk_id": "docB_0", "doc_id": "docB", "title": "Doc B", "section_heading": "Section 1. Purpose", "page": 1, "chunk_index": 0, "text": "Doc B purpose text."},
]


def test_search_corpus_returns_chunk_summaries(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    hybrid = FakeRetriever(["docA_0", "docB_0"])
    tools = CorpusTools(chunks_path=chunks_path, bm25=FakeRetriever([]), hybrid=hybrid)

    results = tools.search_corpus("purpose", k=2)
    assert [r["chunk_id"] for r in results] == ["docA_0", "docB_0"]
    assert results[0]["doc_id"] == "docA"


def test_fetch_section_by_heading(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    tools = CorpusTools(chunks_path=chunks_path, bm25=FakeRetriever([]), hybrid=FakeRetriever([]))

    results = tools.fetch_section("docA", section_heading="Definitions")
    assert [r["chunk_id"] for r in results] == ["docA_1", "docA_2"]


def test_fetch_section_by_page(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    tools = CorpusTools(chunks_path=chunks_path, bm25=FakeRetriever([]), hybrid=FakeRetriever([]))

    results = tools.fetch_section("docA", page=2)
    assert [r["chunk_id"] for r in results] == ["docA_2"]


def test_fetch_section_unknown_doc_returns_empty(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    tools = CorpusTools(chunks_path=chunks_path, bm25=FakeRetriever([]), hybrid=FakeRetriever([]))

    assert tools.fetch_section("does_not_exist", section_heading="x") == []


def test_compare_provisions_scopes_results_per_doc(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    # Hybrid search returns a corpus-wide ranking mixing both docs.
    hybrid = FakeRetriever(["docB_0", "docA_1", "docA_0", "docA_2"])
    tools = CorpusTools(chunks_path=chunks_path, bm25=FakeRetriever([]), hybrid=hybrid)

    results = tools.compare_provisions(["docA", "docB"], aspect="purpose", k_per_doc=2)
    assert [r["chunk_id"] for r in results["docA"]] == ["docA_1", "docA_0"]
    assert [r["chunk_id"] for r in results["docB"]] == ["docB_0"]


def test_compare_provisions_unknown_doc_returns_empty_list(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    tools = CorpusTools(chunks_path=chunks_path, bm25=FakeRetriever([]), hybrid=FakeRetriever(["docA_0"]))

    results = tools.compare_provisions(["nonexistent"], aspect="x")
    assert results["nonexistent"] == []


def test_extract_numeric_field_found(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    bm25 = FakeRetriever(["docA_2", "docA_1"])
    tools = CorpusTools(chunks_path=chunks_path, bm25=bm25, hybrid=FakeRetriever([]))
    provider = FakeProvider(json.dumps({
        "found": True, "value": "$3,078,000", "chunk_id": "docA_2", "context": "a $3,078,000 assessment"
    }))

    result = tools.extract_numeric_field(provider, "docA", "assessment amount")
    assert result["found"] is True
    assert result["value"] == "$3,078,000"
    assert result["chunk_id"] == "docA_2"


def test_extract_numeric_field_not_found_in_doc(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    tools = CorpusTools(chunks_path=chunks_path, bm25=FakeRetriever([]), hybrid=FakeRetriever([]))
    provider = FakeProvider("irrelevant — should never be called")

    result = tools.extract_numeric_field(provider, "does_not_exist", "assessment amount")
    assert result["found"] is False


def test_extract_numeric_field_no_matching_passages(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    bm25 = FakeRetriever(["docB_0"])  # only matches outside docA
    tools = CorpusTools(chunks_path=chunks_path, bm25=bm25, hybrid=FakeRetriever([]))
    provider = FakeProvider("irrelevant — should never be called")

    result = tools.extract_numeric_field(provider, "docA", "assessment amount")
    assert result["found"] is False
    assert "no matching passages" in result["context"]


def test_extract_numeric_field_handles_malformed_json(tmp_path):
    chunks_path = _write_chunks(tmp_path, CHUNKS)
    bm25 = FakeRetriever(["docA_2"])
    tools = CorpusTools(chunks_path=chunks_path, bm25=bm25, hybrid=FakeRetriever([]))
    provider = FakeProvider("not json")

    result = tools.extract_numeric_field(provider, "docA", "assessment amount")
    assert result["found"] is False
    assert "parse error" in result["context"]
