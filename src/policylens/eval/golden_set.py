"""Load and validate the golden question set."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

GOLDEN_SET_PATH = Path("data/golden/golden_questions.json")


class GoldenQuestion(BaseModel):
    question_id: str
    question: str
    answerable: bool
    relevant_chunk_ids: list[str]
    category: str
    notes: str = ""


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    questions = [GoldenQuestion(**q) for q in json.loads(path.read_text())]
    for q in questions:
        if q.answerable and not q.relevant_chunk_ids:
            raise ValueError(f"{q.question_id} is marked answerable but has no relevant_chunk_ids")
        if not q.answerable and q.relevant_chunk_ids:
            raise ValueError(f"{q.question_id} is marked unanswerable but has relevant_chunk_ids")
    return questions
