import pytest

from policylens.eval.pricing import estimate_cost_usd


def test_estimate_cost_sonnet():
    # 1M input tokens @ $3, 1M output tokens @ $15
    cost = estimate_cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(18.00)


def test_estimate_cost_scales_linearly():
    cost = estimate_cost_usd("claude-haiku-4-5", input_tokens=500, output_tokens=100)
    expected = (500 * 1.00 + 100 * 5.00) / 1_000_000
    assert cost == pytest.approx(expected)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        estimate_cost_usd("not-a-real-model", input_tokens=10, output_tokens=10)
