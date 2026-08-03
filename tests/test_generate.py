import json

import pytest

from policylens.generation.generate import GenerationResult, generate_answer
from policylens.providers.base import CompletionResult


class FakeProvider:
    name = "fake"

    def __init__(self, response: str):
        self._response = response
        self.last_system = None
        self.last_user_message = None

    def complete(self, *, system: str, user_message: str, max_tokens: int = 1024) -> CompletionResult:
        self.last_system = system
        self.last_user_message = user_message
        return CompletionResult(text=self._response, input_tokens=100, output_tokens=50, latency_seconds=0.5)


CHUNKS = [
    {"chunk_id": "naic_808_1", "title": "Standard Nonforfeiture Law", "text": "60 days after default."},
    {"chunk_id": "naic_808_2", "title": "Standard Nonforfeiture Law", "text": "Other provisions."},
]


def test_parses_well_formed_json_response():
    response = json.dumps(
        {"answerable": True, "answer": "60 days [naic_808_1]", "citations": ["naic_808_1"]}
    )
    provider = FakeProvider(response)
    result = generate_answer(provider, "How many days?", CHUNKS)
    assert result.answerable is True
    assert result.citations == ["naic_808_1"]
    assert not result.parse_error


def test_strips_markdown_fences_before_parsing():
    response = "```json\n" + json.dumps({"answerable": True, "answer": "x", "citations": []}) + "\n```"
    provider = FakeProvider(response)
    result = generate_answer(provider, "q", CHUNKS)
    assert not result.parse_error
    assert result.answerable is True


def test_malformed_response_becomes_unanswerable_parse_error():
    provider = FakeProvider("not json at all")
    result = generate_answer(provider, "q", CHUNKS)
    assert result.parse_error is True
    assert result.answerable is False


def test_refusal_response_parsed_correctly():
    response = json.dumps({"answerable": False, "answer": "Sources don't cover this.", "citations": []})
    provider = FakeProvider(response)
    result = generate_answer(provider, "q", CHUNKS)
    assert result.answerable is False
    assert result.citations == []


def test_usage_and_latency_propagated_from_provider():
    response = json.dumps({"answerable": True, "answer": "a", "citations": []})
    provider = FakeProvider(response)
    result = generate_answer(provider, "q", CHUNKS)
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.latency_seconds == pytest.approx(0.5)


def test_retrieved_chunks_wrapped_as_source_tags_not_instructions():
    provider = FakeProvider(json.dumps({"answerable": True, "answer": "a", "citations": []}))
    generate_answer(provider, "q", CHUNKS)
    assert '<source chunk_id="naic_808_1"' in provider.last_user_message
    assert "</source>" in provider.last_user_message
    assert "not instructions" in provider.last_system


class TestCitationPrecision:
    def test_all_citations_valid(self):
        result = GenerationResult(
            question="q", answerable=True, answer="a",
            citations=["naic_808_1", "naic_808_2"],
            retrieved_chunk_ids=["naic_808_1", "naic_808_2"],
        )
        assert result.citation_precision == 1.0

    def test_some_citations_invalid(self):
        result = GenerationResult(
            question="q", answerable=True, answer="a",
            citations=["naic_808_1", "made_up_chunk"],
            retrieved_chunk_ids=["naic_808_1", "naic_808_2"],
        )
        assert result.citation_precision == pytest.approx(0.5)

    def test_no_citations_is_none_not_zero(self):
        result = GenerationResult(
            question="q", answerable=False, answer="refused",
            citations=[], retrieved_chunk_ids=["naic_808_1"],
        )
        assert result.citation_precision is None
