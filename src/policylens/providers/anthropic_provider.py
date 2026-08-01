"""Anthropic API provider — the only provider with live credentials.

Default model is claude-sonnet-5, not claude-opus-5, and thinking is
disabled with effort held low: this project set an explicit <$20 total
budget, and extractive Q&A / judging tasks don't need Opus-tier reasoning
depth. Override the model at call time for anything that does.
"""

from __future__ import annotations

import anthropic

from policylens import config  # noqa: F401  (loads .env as a side effect)

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL):
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, *, system: str, user_message: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": user_message}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Anthropic refused the request: {response.stop_details}")
        return next(block.text for block in response.content if block.type == "text")
