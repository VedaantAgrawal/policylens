import math

import pytest

from policylens.eval.metrics import (
    bootstrap_ci,
    bootstrap_ci_delta,
    ndcg_at_k,
    reciprocal_rank,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_zero_recall(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_k_truncates_before_hit(self):
        assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_empty_relevant_raises(self):
        with pytest.raises(ValueError):
            recall_at_k(["a", "b"], set(), k=2)

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a"}, k=5) == 0.0


class TestReciprocalRank:
    def test_first_position_hit(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_third_position_hit(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_uses_first_of_multiple_relevant(self):
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_empty_relevant_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank(["a"], set())


class TestNdcgAtK:
    def test_ideal_ranking_is_one(self):
        # both relevant docs at the top -> nDCG should be exactly 1.0
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_worst_ranking_scores_below_ideal(self):
        # relevant docs pushed to the bottom -> DCG lower than IDCG
        worst = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
        best = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
        assert worst < best
        assert best == pytest.approx(1.0)

    def test_no_hits_is_zero(self):
        assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_k_smaller_than_relevant_set_still_normalizes_to_one_for_ideal(self):
        # only 2 slots available but 3 relevant docs exist; ideal DCG within
        # those 2 slots should still normalize to 1.0
        assert ndcg_at_k(["a", "b", "x"], {"a", "b", "c"}, k=2) == pytest.approx(1.0)

    def test_manual_value(self):
        # relevant doc at rank 2: DCG = 1/log2(3); ideal DCG (rank 1) = 1/log2(2) = 1
        retrieved = ["x", "a", "y"]
        expected = (1 / math.log2(3)) / 1.0
        assert ndcg_at_k(retrieved, {"a"}, k=3) == pytest.approx(expected)

    def test_empty_relevant_raises(self):
        with pytest.raises(ValueError):
            ndcg_at_k(["a"], set(), k=3)


class TestBootstrapCi:
    def test_mean_matches_sample_mean(self):
        values = [0.2, 0.4, 0.6, 0.8, 1.0]
        mean, lower, upper = bootstrap_ci(values, n_bootstrap=2000, seed=1)
        assert mean == pytest.approx(sum(values) / len(values))
        assert lower <= mean <= upper

    def test_constant_values_gives_zero_width_ci(self):
        values = [0.5] * 10
        mean, lower, upper = bootstrap_ci(values, n_bootstrap=1000, seed=1)
        assert mean == pytest.approx(0.5)
        assert lower == pytest.approx(0.5)
        assert upper == pytest.approx(0.5)

    def test_deterministic_given_seed(self):
        values = [0.1, 0.9, 0.3, 0.7, 0.5]
        run1 = bootstrap_ci(values, n_bootstrap=500, seed=7)
        run2 = bootstrap_ci(values, n_bootstrap=500, seed=7)
        assert run1 == run2

    def test_empty_values_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])

    def test_wider_spread_gives_wider_ci(self):
        tight = bootstrap_ci([0.5, 0.5, 0.51, 0.49], n_bootstrap=2000, seed=1)
        wide = bootstrap_ci([0.0, 1.0, 0.0, 1.0], n_bootstrap=2000, seed=1)
        assert (wide[2] - wide[1]) > (tight[2] - tight[1])


class TestBootstrapCiDelta:
    def test_identical_distributions_have_zero_mean_delta(self):
        values = [0.3, 0.5, 0.7, 0.9]
        delta, lower, upper = bootstrap_ci_delta(values, values, n_bootstrap=2000, seed=1)
        assert delta == pytest.approx(0.0)
        assert lower <= 0.0 <= upper

    def test_uniform_improvement_gives_positive_delta(self):
        baseline = [0.2, 0.3, 0.4, 0.5]
        improved = [0.4, 0.5, 0.6, 0.7]
        delta, lower, upper = bootstrap_ci_delta(baseline, improved, n_bootstrap=2000, seed=1)
        assert delta == pytest.approx(0.2)
        assert lower > 0.0  # improvement is consistent across every paired query -> CI excludes zero

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            bootstrap_ci_delta([0.1, 0.2], [0.1], n_bootstrap=100)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci_delta([], [])
