"""Bedrock provider — implements the same interface as AnthropicProvider.

Untested: no AWS credentials are available in this environment. Written
against the documented AnthropicBedrockMantle client so the interface is
real, not a stub, but treat this as unverified until it's run against an
actual AWS account.
"""

from __future__ import annotations

from anthropic import AnthropicBedrockMantle

DEFAULT_MODEL = "anthropic.claude-sonnet-5"


class BedrockProvider:
    name = "bedrock"

    def __init__(self, model: str = DEFAULT_MODEL, aws_region: str = "us-east-1"):
        self._client = AnthropicBedrockMantle(aws_region=aws_region)
        self._model = model

    def complete(self, *, system: str, user_message: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return next(block.text for block in response.content if block.type == "text")
