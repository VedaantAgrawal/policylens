import json

import pytest

from policylens.eval.golden_set import load_golden_set


def _write(tmp_path, questions, chunk_ids=None):
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(questions))

    chunks_path = tmp_path / "chunks.jsonl"
    if chunk_ids is not None:
        with chunks_path.open("w") as f:
            for cid in chunk_ids:
                f.write(json.dumps({"chunk_id": cid, "text": "x"}) + "\n")
    return golden_path, chunks_path


def test_loads_valid_set(tmp_path):
    questions = [
        {"question_id": "q1", "question": "?", "answerable": True, "relevant_chunk_ids": ["c1"], "category": "x"},
        {"question_id": "q2", "question": "?", "answerable": False, "relevant_chunk_ids": [], "category": "unanswerable"},
    ]
    golden_path, chunks_path = _write(tmp_path, questions, chunk_ids=["c1"])
    result = load_golden_set(golden_path, chunks_path)
    assert len(result) == 2


def test_answerable_without_chunk_ids_rejected(tmp_path):
    questions = [{"question_id": "q1", "question": "?", "answerable": True, "relevant_chunk_ids": [], "category": "x"}]
    golden_path, chunks_path = _write(tmp_path, questions, chunk_ids=[])
    with pytest.raises(ValueError, match="no relevant_chunk_ids"):
        load_golden_set(golden_path, chunks_path)


def test_unanswerable_with_chunk_ids_rejected(tmp_path):
    questions = [
        {"question_id": "q1", "question": "?", "answerable": False, "relevant_chunk_ids": ["c1"], "category": "unanswerable"}
    ]
    golden_path, chunks_path = _write(tmp_path, questions, chunk_ids=["c1"])
    with pytest.raises(ValueError, match="marked unanswerable but has relevant_chunk_ids"):
        load_golden_set(golden_path, chunks_path)


def test_duplicate_question_id_rejected(tmp_path):
    questions = [
        {"question_id": "q1", "question": "a", "answerable": True, "relevant_chunk_ids": ["c1"], "category": "x"},
        {"question_id": "q1", "question": "b", "answerable": True, "relevant_chunk_ids": ["c1"], "category": "x"},
    ]
    golden_path, chunks_path = _write(tmp_path, questions, chunk_ids=["c1"])
    with pytest.raises(ValueError, match="duplicate question_id"):
        load_golden_set(golden_path, chunks_path)


def test_chunk_id_not_in_corpus_rejected(tmp_path):
    questions = [
        {"question_id": "q1", "question": "?", "answerable": True, "relevant_chunk_ids": ["does_not_exist"], "category": "x"}
    ]
    golden_path, chunks_path = _write(tmp_path, questions, chunk_ids=["c1"])
    with pytest.raises(ValueError, match="not in the corpus"):
        load_golden_set(golden_path, chunks_path)


def test_missing_chunks_file_skips_corpus_check(tmp_path):
    # A fresh clone before `make setup` has no chunks.jsonl yet — the golden
    # set should still load (other validation still applies).
    questions = [{"question_id": "q1", "question": "?", "answerable": True, "relevant_chunk_ids": ["c1"], "category": "x"}]
    golden_path, chunks_path = _write(tmp_path, questions, chunk_ids=None)
    result = load_golden_set(golden_path, chunks_path)
    assert len(result) == 1


def test_real_golden_set_is_valid():
    """The committed golden set must always pass its own validation."""
    result = load_golden_set()
    assert len(result) >= 42
