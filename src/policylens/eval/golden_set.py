"""Load and validate the golden question set.

Validation is load-bearing, not a courtesy: a golden question whose
relevant_chunk_ids don't actually exist in the corpus would silently score
retrieval as "missed" forever, and there'd be no signal that the question
itself is broken rather than the retriever. Every load re-checks the set
against the current corpus rather than trusting it was correct once.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

GOLDEN_SET_PATH = Path("data/golden/golden_questions.json")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")


class GoldenQuestion(BaseModel):
    question_id: str
    question: str
    answerable: bool
    relevant_chunk_ids: list[str]
    category: str
    notes: str = ""


def _load_known_chunk_ids(chunks_path: Path) -> set[str] | None:
    if not chunks_path.exists():
        return None  # corpus not built yet (e.g. fresh clone before `make setup`) — skip this check
    with chunks_path.open() as f:
        return {json.loads(line)["chunk_id"] for line in f}


def load_golden_set(path: Path = GOLDEN_SET_PATH, chunks_path: Path = CHUNKS_PATH) -> list[GoldenQuestion]:
    questions = [GoldenQuestion(**q) for q in json.loads(path.read_text())]

    seen_ids: set[str] = set()
    known_chunk_ids = _load_known_chunk_ids(chunks_path)
    for q in questions:
        if q.question_id in seen_ids:
            raise ValueError(f"duplicate question_id: {q.question_id}")
        seen_ids.add(q.question_id)

        if q.answerable and not q.relevant_chunk_ids:
            raise ValueError(f"{q.question_id} is marked answerable but has no relevant_chunk_ids")
        if not q.answerable and q.relevant_chunk_ids:
            raise ValueError(f"{q.question_id} is marked unanswerable but has relevant_chunk_ids")

        if known_chunk_ids is not None:
            missing = [cid for cid in q.relevant_chunk_ids if cid not in known_chunk_ids]
            if missing:
                raise ValueError(f"{q.question_id} references chunk_id(s) not in the corpus: {missing}")
    return questions
