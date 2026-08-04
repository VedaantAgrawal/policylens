import json

from policylens.retrieval.rerank import CrossEncoderRerankRetriever


class FakeBaseRetriever:
    name = "fake_base"

    def __init__(self, ordered_ids):
        self._ordered_ids = ordered_ids

    def search(self, query: str, k: int) -> list[str]:
        return self._ordered_ids[:k]


class FakeCrossEncoder:
    """Scores by a lookup table keyed on chunk text, ignoring the query."""

    def __init__(self, score_by_text: dict[str, float]):
        self._score_by_text = score_by_text

    def predict(self, pairs):
        return [self._score_by_text[text] for _, text in pairs]


def _write_chunks(tmp_path, chunks):
    path = tmp_path / "chunks.jsonl"
    with path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return path


def test_reranks_by_cross_encoder_score_not_base_rank(tmp_path):
    # Base retriever ranks "a" first, but the cross-encoder should reorder
    # to put "b" first since it scores higher.
    chunks_path = _write_chunks(
        tmp_path,
        [
            {"chunk_id": "a", "text": "irrelevant text"},
            {"chunk_id": "b", "text": "highly relevant text"},
        ],
    )
    base = FakeBaseRetriever(["a", "b"])
    cross_encoder = FakeCrossEncoder({"irrelevant text": 0.1, "highly relevant text": 0.9})
    retriever = CrossEncoderRerankRetriever(base_retriever=base, cross_encoder=cross_encoder, chunks_path=chunks_path)

    result = retriever.search("query", k=2)
    assert result == ["b", "a"]


def test_respects_k_limit(tmp_path):
    chunks_path = _write_chunks(
        tmp_path,
        [
            {"chunk_id": "a", "text": "t1"},
            {"chunk_id": "b", "text": "t2"},
            {"chunk_id": "c", "text": "t3"},
        ],
    )
    base = FakeBaseRetriever(["a", "b", "c"])
    cross_encoder = FakeCrossEncoder({"t1": 0.5, "t2": 0.9, "t3": 0.1})
    retriever = CrossEncoderRerankRetriever(base_retriever=base, cross_encoder=cross_encoder, chunks_path=chunks_path)

    result = retriever.search("query", k=2)
    assert result == ["b", "a"]


def test_candidate_missing_from_chunk_store_is_skipped(tmp_path):
    chunks_path = _write_chunks(tmp_path, [{"chunk_id": "a", "text": "t1"}])
    base = FakeBaseRetriever(["a", "nonexistent_chunk"])
    cross_encoder = FakeCrossEncoder({"t1": 0.5})
    retriever = CrossEncoderRerankRetriever(base_retriever=base, cross_encoder=cross_encoder, chunks_path=chunks_path)

    result = retriever.search("query", k=5)
    assert result == ["a"]


def test_empty_candidate_pool_returns_empty_list(tmp_path):
    chunks_path = _write_chunks(tmp_path, [])
    base = FakeBaseRetriever([])
    cross_encoder = FakeCrossEncoder({})
    retriever = CrossEncoderRerankRetriever(base_retriever=base, cross_encoder=cross_encoder, chunks_path=chunks_path)

    assert retriever.search("query", k=5) == []
