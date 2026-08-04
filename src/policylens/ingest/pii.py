"""Regex-based PII redaction, applied to every chunk before it's written.

Public regulatory/financial documents (10-Ks, model laws, bulletins) rarely
carry real PII, but the guardrail is defensive-by-default: a state bulletin
quoting a consumer complaint, or future corpus expansion into complaint
records, could easily include one. Patterns are deliberately conservative
(exact separator shapes) to avoid clobbering statute citations, model law
numbers ("MO-808-2"), dates, and dollar amounts, which all share the
"digits with punctuation" shape PII does.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Phone: parens around the area code, or dash-only separators — NOT bare
    # spaces. 10-K actuarial loss-development tables are columns of
    # space-separated 3-4 digit numbers (years, dollar amounts) that
    # false-positive on a space-tolerant 3-3-4 pattern; real phone numbers
    # in this corpus's "contact us" sections are always parenthesized or
    # dash-separated, never bare-space, so this loses no real matches.
    # No leading \b before "(" — a word boundary can't match between two
    # non-word characters (e.g. a space followed by "("), so anchoring
    # there would silently fail to match "(555) 123-4567" after a space.
    ("PHONE", re.compile(r"(?:\+?1[-.\s]?)?\(\d{3}\)[-.\s]?\d{3}-\d{4}\b|\b\d{3}-\d{3}-\d{4}\b")),
    # Credit card: dash-separated 4-4-4-4 only, for the same reason — a
    # space-separated run is indistinguishable from a financial table here.
    ("CREDIT_CARD", re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{4}\b")),
]


def redact_pii(text: str) -> tuple[str, int]:
    """Replace PII matches with a [REDACTED-<TYPE>] marker. Returns (text, count)."""
    count = 0
    for label, pattern in _PATTERNS:

        def _sub(_match: re.Match[str], label: str = label) -> str:
            nonlocal count
            count += 1
            return f"[REDACTED-{label}]"

        text = pattern.sub(_sub, text)
    return text, count
