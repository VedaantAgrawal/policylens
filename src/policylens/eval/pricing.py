"""Per-token pricing for cost-per-query reporting.

Standard published sticker prices, not the claude-sonnet-5 introductory rate
($2/$10 through 2026-08-31) — this project's cost numbers are meant to hold
up as a durable unit-economics artifact, not go stale the day the intro
period ends.
"""

from __future__ import annotations

PRICE_PER_MTOK_USD = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICE_PER_MTOK_USD:
        raise ValueError(f"No pricing for model {model!r} — add it to PRICE_PER_MTOK_USD")
    rates = PRICE_PER_MTOK_USD[model]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
