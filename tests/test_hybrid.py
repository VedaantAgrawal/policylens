from policylens.retrieval.hybrid import HybridRetriever


class FakeRetriever:
    def __init__(self, ranking: list[str]):
        self._ranking = ranking

    def search(self, query: str, k: int) -> list[str]:
        return self._ranking[:k]


def test_agreement_boosts_shared_result_to_top():
    # "a" is ranked highly by both retrievers -> should win fusion even
    # though "z" is BM25's top result.
    bm25 = FakeRetriever(["z", "a", "b"])
    dense = FakeRetriever(["a", "y", "c"])
    hybrid = HybridRetriever(bm25=bm25, dense=dense)
    result = hybrid.search("query", k=3)
    assert result[0] == "a"


def test_union_of_both_rankings_present():
    bm25 = FakeRetriever(["a", "b"])
    dense = FakeRetriever(["c", "d"])
    hybrid = HybridRetriever(bm25=bm25, dense=dense)
    result = hybrid.search("query", k=10)
    assert set(result) == {"a", "b", "c", "d"}


def test_k_truncates_fused_results():
    bm25 = FakeRetriever(["a", "b", "c"])
    dense = FakeRetriever(["d", "e", "f"])
    hybrid = HybridRetriever(bm25=bm25, dense=dense)
    result = hybrid.search("query", k=2)
    assert len(result) == 2
