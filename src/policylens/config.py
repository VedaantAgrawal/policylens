"""Loads .env once on import so every entrypoint sees ANTHROPIC_API_KEY.

Anthropic's SDK reads credentials from the process environment, not from
.env files directly — this is the one place that bridges the two, so
scripts never need to remember to call load_dotenv() themselves.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
