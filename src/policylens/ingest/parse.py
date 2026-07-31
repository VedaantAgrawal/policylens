"""Extract normalized, heading-annotated text from raw PDF/HTML documents.

Both extractors produce the same intermediate representation: plain text with
inline markers — `\x01HEADING\x01{text}\x01` for detected section headings and
`\x02PAGE\x02{n}\x02` for page boundaries (PDFs only) — so chunk.py can treat
PDF and HTML sources uniformly instead of needing per-format chunking logic.
Control characters are used as markers specifically because they cannot appear
in extracted document text, so there's no risk of a heading/page marker being
confused with real content.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

HEADING_MARK = "\x01HEADING\x01"
PAGE_MARK = "\x02PAGE\x02"

# Heuristics for "this line is a section heading", tuned against NAIC model
# laws (which use "Section 1.", "ARTICLE I") and SEC 10-Ks (which use
# "Item 1.", "Item 1A."). Not a layout parser — just line-shape rules applied
# after whitespace normalization.
_HEADING_PATTERNS = [
    re.compile(r"^Item\s+\d+[A-Z]?\.?\s*.{0,120}$", re.I),
    re.compile(r"^Section\s+\d+[A-Za-z]?\.?\s*.{0,120}$", re.I),
    re.compile(r"^ARTICLE\s+[IVXLC0-9]+\.?\s*.{0,120}$", re.I),
    re.compile(r"^Drafting Note", re.I),
    re.compile(r"^PART\s+[IVXLC0-9]+\.?\s*.{0,120}$", re.I),
    re.compile(r"^[A-Z][A-Z0-9 ,\-'&/]{6,100}$"),  # short all-caps line
]


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not (3 <= len(line) <= 120):
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _annotate_headings(raw_lines: list[str]) -> str:
    out = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        if _looks_like_heading(line):
            out.append(f"{HEADING_MARK}{line}{HEADING_MARK}")
        else:
            out.append(line)
    return "\n".join(out)


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page_num, page in enumerate(reader.pages, start=1):
        parts.append(f"{PAGE_MARK}{page_num}{PAGE_MARK}")
        text = page.extract_text() or ""
        parts.append(_annotate_headings(text.split("\n")))
    return "\n".join(parts)


def extract_html_text(path: Path) -> str:
    with open(path, "rb") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Promote heading tags and bold/strong lines to explicit markers before
    # falling back to line-shape heuristics on the rest of the text.
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = tag.get_text(strip=True)
        if text:
            tag.replace_with(f"\n{HEADING_MARK}{text}{HEADING_MARK}\n")

    raw_text = soup.get_text(separator="\n")
    lines = [l.strip() for l in raw_text.split("\n")]
    return _annotate_headings(lines)
