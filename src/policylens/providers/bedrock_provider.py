"""Bedrock provider — implements the same interface as AnthropicProvider.

Untested: no AWS credentials are available in this environment. Written
against the documented AnthropicBedrockMantle client so the interface is
real, not a stub, but treat this as unverified until it's run against an
actual AWS account.
"""

from __future__ import annotations

import time

from anthropic import AnthropicBedrockMantle

from policylens.providers.base import CompletionResult

DEFAULT_MODEL = "anthropic.claude-sonnet-5"


class BedrockProvider:
    name = "bedrock"

    def __init__(self, model: str = DEFAULT_MODEL, aws_region: str = "us-east-1"):
        self._client = AnthropicBedrockMantle(aws_region=aws_region)
        self._model = model

    @property
    def client(self) -> AnthropicBedrockMantle:
        """Raw SDK client — same escape hatch as AnthropicProvider.client, for
        the agent loop's tool-use needs. AnthropicBedrockMantle exposes the same
        messages.create/tool_runner surface, so agent code doesn't need to
        branch on provider."""
        return self._client

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, user_message: str, max_tokens: int = 1024) -> CompletionResult:
        start = time.monotonic()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        latency_seconds = time.monotonic() - start
        text = next(block.text for block in response.content if block.type == "text")
        return CompletionResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_seconds=latency_seconds,
        )
