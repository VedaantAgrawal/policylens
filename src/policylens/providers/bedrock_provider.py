"""Bedrock provider — implements the same interface as AnthropicProvider.

Goes through the bedrock-runtime endpoint (`AnthropicBedrock`, SigV4-signed),
verified against a real AWS account. The newer bedrock-mantle endpoint was
tried first but doesn't carry any Anthropic models on this account — its
`/v1/models` listing returned 49 models across a dozen providers and zero
`anthropic.*` entries.
"""

from __future__ import annotations

import os
import time

from anthropic import AnthropicBedrock

from policylens.providers.base import CompletionResult

# Cross-region inference profile ID — this account's Claude models are
# INFERENCE_PROFILE-only (no on-demand throughput) and aren't exposed via
# the bedrock-mantle endpoint at all, so we go through bedrock-runtime.
# Sonnet 5 itself (`us.anthropic.claude-sonnet-5`) is listed as ACTIVE but
# returns 403 "not available for this account" — an AWS-side entitlement
# gap on new-model rollout, not a config issue. Sonnet 4.6 is the newest
# Sonnet actually entitled on this account.
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"


class BedrockProvider:
    name = "bedrock"

    def __init__(self, model: str = DEFAULT_MODEL, aws_region: str | None = None):
        region = aws_region or os.environ.get("AWS_REGION", "us-east-2")
        self._client = AnthropicBedrock(aws_region=region)
        self._model = model

    @property
    def client(self) -> AnthropicBedrock:
        """Raw SDK client — same escape hatch as AnthropicProvider.client, for
        the agent loop's tool-use needs. AnthropicBedrock exposes the same
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
