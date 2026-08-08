import json

from policylens.agent.loop import _build_tools
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
    model = "fake-model"

    def __init__(self, response: str):
        self._response = response

    def complete(self, *, system: str, user_message: str, max_tokens: int = 300) -> CompletionResult:
        return CompletionResult(text=self._response, input_tokens=10, output_tokens=5, latency_seconds=0.05)


CHUNKS = [
    {"chunk_id": "docA_0", "doc_id": "docA", "title": "Doc A", "section_heading": "Section 1. Purpose", "page": 1, "chunk_index": 0, "text": "Purpose text."},
]


def _make_corpus_tools(tmp_path, bm25_ids=None, hybrid_ids=None):
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w") as f:
        for c in CHUNKS:
            f.write(json.dumps(c) + "\n")
    return CorpusTools(
        chunks_path=chunks_path,
        bm25=FakeRetriever(bm25_ids or []),
        hybrid=FakeRetriever(hybrid_ids or []),
    )


def test_build_tools_returns_four_named_tools(tmp_path):
    corpus_tools = _make_corpus_tools(tmp_path)
    trace = []
    tools = _build_tools(corpus_tools, FakeProvider("{}"), trace)
    assert [t.name for t in tools] == ["search_corpus", "fetch_section", "compare_provisions", "extract_numeric_field"]


def test_search_corpus_tool_records_trace(tmp_path):
    corpus_tools = _make_corpus_tools(tmp_path, hybrid_ids=["docA_0"])
    trace = []
    tools = _build_tools(corpus_tools, FakeProvider("{}"), trace)
    search_corpus = tools[0]

    raw = search_corpus(query="purpose", k=1)
    results = json.loads(raw)
    assert len(results) == 1
    assert trace == [{"type": "tool_call", "tool": "search_corpus", "input": {"query": "purpose", "k": 1}, "num_results": 1}]


def test_fetch_section_tool_records_trace(tmp_path):
    corpus_tools = _make_corpus_tools(tmp_path)
    trace = []
    tools = _build_tools(corpus_tools, FakeProvider("{}"), trace)
    fetch_section = tools[1]

    raw = fetch_section(doc_id="docA", section_heading="Purpose", page=None)
    results = json.loads(raw)
    assert len(results) == 1
    assert trace[0]["tool"] == "fetch_section"
    assert trace[0]["input"]["doc_id"] == "docA"


def test_compare_provisions_tool_records_trace(tmp_path):
    corpus_tools = _make_corpus_tools(tmp_path, hybrid_ids=["docA_0"])
    trace = []
    tools = _build_tools(corpus_tools, FakeProvider("{}"), trace)
    compare_provisions = tools[2]

    raw = compare_provisions(doc_ids=["docA"], aspect="purpose", k_per_doc=1)
    results = json.loads(raw)
    assert "docA" in results
    assert trace[0]["tool"] == "compare_provisions"
    assert trace[0]["num_results"] == {"docA": 1}


def test_extract_numeric_field_tool_records_trace(tmp_path):
    corpus_tools = _make_corpus_tools(tmp_path, bm25_ids=["docA_0"])
    trace = []
    provider = FakeProvider(json.dumps({"found": True, "value": "42", "chunk_id": "docA_0", "context": "42 days"}))
    tools = _build_tools(corpus_tools, provider, trace)
    extract_numeric_field = tools[3]

    raw = extract_numeric_field(doc_id="docA", field_description="number of days")
    result = json.loads(raw)
    assert result["found"] is True
    assert trace[0]["tool"] == "extract_numeric_field"
    assert trace[0]["found"] is True
