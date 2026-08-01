"""Model-provider interface.

Generation, LLM-judge, and (later) the agent layer all call through this
Protocol instead of importing an SDK directly — that's what lets the same
business logic run against Anthropic or Bedrock without touching call sites.
"""

from __future__ import annotations

from typing import Protocol


class ModelProvider(Protocol):
    name: str

    def complete(self, *, system: str, user_message: str, max_tokens: int = 1024) -> str:
        """Return the model's text response to a single-turn request."""
        ...
