import json

from policylens.eval.judge import judge_groundedness
from policylens.generation.generate import GenerationResult
from policylens.providers.base import CompletionResult


class FakeProvider:
    name = "fake"

    def __init__(self, response: str):
        self._response = response

    def complete(self, *, system: str, user_message: str, max_tokens: int = 300) -> CompletionResult:
        return CompletionResult(text=self._response, input_tokens=80, output_tokens=20, latency_seconds=0.2)


CHUNKS_BY_ID = {"naic_808_1": {"chunk_id": "naic_808_1", "text": "60 days after default."}}


def _result(citations):
    return GenerationResult(
        question="q", answerable=True, answer="a", citations=citations, retrieved_chunk_ids=["naic_808_1"]
    )


def test_grounded_verdict_parsed():
    provider = FakeProvider(json.dumps({"grounded": True, "reasoning": "matches source"}))
    verdict = judge_groundedness(provider, "q", _result(["naic_808_1"]), CHUNKS_BY_ID)
    assert verdict.grounded is True
    assert not verdict.parse_error


def test_ungrounded_verdict_parsed():
    provider = FakeProvider(json.dumps({"grounded": False, "reasoning": "fabricated detail"}))
    verdict = judge_groundedness(provider, "q", _result(["naic_808_1"]), CHUNKS_BY_ID)
    assert verdict.grounded is False


def test_malformed_judge_response_defaults_to_ungrounded():
    provider = FakeProvider("not json")
    verdict = judge_groundedness(provider, "q", _result(["naic_808_1"]), CHUNKS_BY_ID)
    assert verdict.grounded is False
    assert verdict.parse_error is True


def test_citation_to_unknown_chunk_id_is_skipped_not_crashed():
    provider = FakeProvider(json.dumps({"grounded": True, "reasoning": "ok"}))
    verdict = judge_groundedness(provider, "q", _result(["unknown_chunk"]), CHUNKS_BY_ID)
    assert not verdict.parse_error
