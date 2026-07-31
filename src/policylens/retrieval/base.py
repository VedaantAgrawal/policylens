"""Retriever interface every ablation stage (S0-S3) implements identically.

Keeping this interface tiny is what makes the ablation table fair: the eval
runner in policylens.eval.runner calls every stage through the exact same
`search` method, so a score difference reflects the retrieval method, not a
difference in how it was harnessed.
"""

from __future__ import annotations

from typing import Protocol


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[str]:
        """Return up to k chunk_ids, ranked most relevant first."""
        ...
