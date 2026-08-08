"""Model-provider interface.

Generation, LLM-judge, and (later) the agent layer all call through this
Protocol instead of importing an SDK directly — that's what lets the same
business logic run against Anthropic or Bedrock without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class CompletionResult:
    """Provider response, plus what it costs and how long it took to get it.

    Token counts and latency are carried on every response, not just fetched
    on demand, so callers (generation, judge, the /query endpoint) can report
    cost-per-query and p50/p95 latency without a second measurement path.
    """

    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class ModelProvider(Protocol):
    name: str
    model: str

    def complete(self, *, system: str, user_message: str, max_tokens: int = 1024) -> CompletionResult:
        """Return the model's response to a single-turn request, with usage + timing."""
        ...

    @property
    def client(self):
        """Raw underlying SDK client — an escape hatch for callers (the agent
        loop) that need tool use or multi-turn conversations `complete()` can't
        express. Both AnthropicProvider and BedrockProvider expose the same
        messages.create/tool_runner surface here, so agent code never branches
        on which provider it's holding."""
        ...
