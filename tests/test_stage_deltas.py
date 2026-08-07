import json

import pytest

from policylens.eval.stage_deltas import compute_deltas


def _write_stage(tmp_path, name, per_query):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"retriever": name, "per_query": per_query}))
    return path


@pytest.fixture(autouse=True)
def _use_tmp_eval_dir(tmp_path, monkeypatch):
    import policylens.eval.stage_deltas as module

    monkeypatch.setattr(module, "EVAL_RESULTS_DIR", tmp_path)
    return tmp_path


def test_identical_stages_have_zero_delta(tmp_path):
    per_query = [
        {"question_id": f"q{i}", "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0}
        for i in range(10)
    ]
    _write_stage(tmp_path, "s0_bm25", per_query)
    _write_stage(tmp_path, "s1_dense", per_query)
    _write_stage(tmp_path, "s2_hybrid", per_query)
    _write_stage(tmp_path, "s3_rerank", per_query)

    results = compute_deltas()
    assert results["s0_bm25->s1_dense"]["recall_at_5"]["delta"] == 0.0
    assert results["s0_bm25->s1_dense"]["recall_at_5"]["significant"] is False


def test_clear_improvement_is_significant(tmp_path):
    # b is strictly better than a on every question — should register as significant.
    per_query_a = [
        {"question_id": f"q{i}", "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0}
        for i in range(30)
    ]
    per_query_b = [
        {"question_id": f"q{i}", "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0}
        for i in range(30)
    ]
    _write_stage(tmp_path, "s0_bm25", per_query_a)
    _write_stage(tmp_path, "s1_dense", per_query_a)
    _write_stage(tmp_path, "s2_hybrid", per_query_b)
    _write_stage(tmp_path, "s3_rerank", per_query_b)

    results = compute_deltas()
    stats = results["s0_bm25->s2_hybrid"]["recall_at_5"]
    assert stats["delta"] == pytest.approx(1.0)
    assert stats["significant"] is True


def test_mismatched_question_sets_raises(tmp_path):
    per_query_a = [{"question_id": "q1", "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0}]
    per_query_b = [
        {"question_id": "q1", "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0},
        {"question_id": "q2", "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0},
    ]
    _write_stage(tmp_path, "s0_bm25", per_query_a)
    _write_stage(tmp_path, "s1_dense", per_query_a)
    _write_stage(tmp_path, "s2_hybrid", per_query_b)
    _write_stage(tmp_path, "s3_rerank", per_query_b)

    with pytest.raises(ValueError, match="different question sets"):
        compute_deltas()
