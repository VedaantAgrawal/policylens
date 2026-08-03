"""LLM-as-judge for groundedness: is the answer actually supported by its cited sources?

Runs on claude-haiku-4-5 — a binary rubric check against short source text
is a simple classification task, not one that needs Sonnet-level reasoning,
and judge calls run once per generated answer so cost adds up fastest here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from policylens.generation.generate import GenerationResult
    from policylens.providers.base import ModelProvider

JUDGE_MODEL = "claude-haiku-4-5"

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker. You will be shown a QUESTION, an ANSWER, \
and the SOURCE TEXT the answer's citations point to. Your only job is to judge whether the \
ANSWER's factual claims are actually supported by the SOURCE TEXT.

Rubric — mark grounded=true only if ALL of these hold:
1. Every specific fact, number, or quote in the ANSWER appears in or is a fair paraphrase of the SOURCE TEXT.
2. The ANSWER does not add information that isn't in the SOURCE TEXT, even if that information is \
plausible or generally true.
3. The ANSWER does not contradict the SOURCE TEXT.

Mark grounded=false if any claim is unsupported, fabricated, or contradicts the source — even a single \
unsupported detail is enough to fail. Treat the SOURCE TEXT as the only permitted evidence; your own \
outside knowledge of insurance regulation is irrelevant to this judgment.

Respond with ONLY a single JSON object, no markdown fences, no other text:
{"grounded": true or false, "reasoning": "one sentence citing the specific supported or unsupported claim"}
"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class GroundednessVerdict:
    grounded: bool
    reasoning: str
    parse_error: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0


def _build_user_message(question: str, answer: str, cited_chunks: list[dict]) -> str:
    lines = [f"QUESTION: {question}", "", f"ANSWER: {answer}", "", "SOURCE TEXT:"]
    for c in cited_chunks:
        lines.append(f'<source chunk_id="{c["chunk_id"]}">')
        lines.append(c["text"])
        lines.append("</source>")
    return "\n".join(lines)


def judge_groundedness(
    provider: "ModelProvider", question: str, result: "GenerationResult", chunks_by_id: dict[str, dict]
) -> GroundednessVerdict:
    cited_chunks = [chunks_by_id[cid] for cid in result.citations if cid in chunks_by_id]
    user_message = _build_user_message(question, result.answer, cited_chunks)
    completion = provider.complete(system=JUDGE_SYSTEM_PROMPT, user_message=user_message, max_tokens=300)
    usage = {
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "latency_seconds": completion.latency_seconds,
    }

    try:
        cleaned = _FENCE_RE.sub("", completion.text.strip())
        parsed = json.loads(cleaned)
        return GroundednessVerdict(
            grounded=bool(parsed["grounded"]), reasoning=str(parsed["reasoning"]), **usage
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return GroundednessVerdict(
            grounded=False, reasoning=f"[judge parse error] {completion.text[:300]}", parse_error=True, **usage
        )
