import pytest

from policylens.eval.golden_set import GoldenQuestion
from policylens.eval.runner import METRIC_NAMES, run_eval


class FakeRetriever:
    name = "fake"

    def __init__(self, answers: dict[str, list[str]]):
        self._answers = answers

    def search(self, query: str, k: int) -> list[str]:
        return self._answers.get(query, [])[:k]


def _q(qid, question, answerable=True, relevant=None, category="test"):
    return GoldenQuestion(
        question_id=qid,
        question=question,
        answerable=answerable,
        relevant_chunk_ids=relevant or [],
        category=category,
    )


def test_run_eval_perfect_retriever_scores_one_everywhere():
    questions = [_q("q1", "what is x", relevant=["a"]), _q("q2", "what is y", relevant=["b"])]
    retriever = FakeRetriever({"what is x": ["a"], "what is y": ["b"]})
    result = run_eval(retriever, questions)

    assert result["num_answerable_questions"] == 2
    for metric in METRIC_NAMES:
        assert result["aggregate"][metric]["mean"] == pytest.approx(1.0)


def test_run_eval_excludes_unanswerable_questions():
    questions = [
        _q("q1", "what is x", relevant=["a"]),
        _q("q2", "unanswerable one", answerable=False),
    ]
    retriever = FakeRetriever({"what is x": ["a"]})
    result = run_eval(retriever, questions)

    assert result["num_answerable_questions"] == 1
    assert result["num_unanswerable_questions"] == 1
    assert len(result["per_query"]) == 1


def test_run_eval_no_answerable_questions_raises():
    questions = [_q("q1", "unanswerable", answerable=False)]
    retriever = FakeRetriever({})
    with pytest.raises(ValueError):
        run_eval(retriever, questions)


def test_run_eval_zero_hits_scores_zero():
    questions = [_q("q1", "what is x", relevant=["a"])]
    retriever = FakeRetriever({"what is x": ["z", "y"]})
    result = run_eval(retriever, questions)
    for metric in METRIC_NAMES:
        assert result["aggregate"][metric]["mean"] == pytest.approx(0.0)


def test_per_query_scores_preserve_question_id_and_category():
    questions = [_q("q1", "what is x", relevant=["a"], category="naic_model_law")]
    retriever = FakeRetriever({"what is x": ["a"]})
    result = run_eval(retriever, questions)
    assert result["per_query"][0]["question_id"] == "q1"
    assert result["per_query"][0]["category"] == "naic_model_law"
