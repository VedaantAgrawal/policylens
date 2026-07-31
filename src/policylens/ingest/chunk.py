"""Turn parsed, heading-annotated document text into retrieval chunks.

Chunking is section-aware: a heading/page transition always starts a new span,
and a span only gets word-window split if it's too long to be one chunk. This
keeps a chunk's `section_heading` and `page` metadata accurate to what's
actually in it, which is what inline citations point back to at generation
time — a citation is only as trustworthy as this metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from policylens.ingest.parse import HEADING_MARK, PAGE_MARK, extract_html_text, extract_pdf_text

TARGET_CHUNK_WORDS = 300
OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 20  # spans shorter than this are dropped (nav debris, empty headers)

MANIFEST_PATH = Path("data/manifest.jsonl")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")


@dataclass
class Span:
    page: int | None
    heading: str | None
    text: str


def _split_into_spans(marked_text: str) -> list[Span]:
    spans: list[Span] = []
    current_page: int | None = None
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = " ".join(buffer).strip()
        if text:
            spans.append(Span(page=current_page, heading=current_heading, text=text))
        buffer.clear()

    for line in marked_text.split("\n"):
        if line.startswith(PAGE_MARK):
            flush()
            current_page = int(line.replace(PAGE_MARK, ""))
        elif line.startswith(HEADING_MARK):
            flush()
            current_heading = line.replace(HEADING_MARK, "")
        else:
            if line.strip():
                buffer.append(line.strip())
    flush()
    return spans


def _window_split(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    windows = []
    step = size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if len(window) >= MIN_CHUNK_WORDS:
            windows.append(window)
        if start + size >= len(words):
            break
    return windows


def chunk_document(doc: dict) -> list[dict]:
    local_path = Path(doc["local_path"])
    if local_path.suffix == ".pdf":
        marked_text = extract_pdf_text(local_path)
    else:
        marked_text = extract_html_text(local_path)

    spans = _split_into_spans(marked_text)
    chunks = []
    chunk_index = 0
    for span in spans:
        words = span.text.split()
        if len(words) < MIN_CHUNK_WORDS:
            continue
        for window in _window_split(words, TARGET_CHUNK_WORDS, OVERLAP_WORDS):
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_{chunk_index}",
                    "doc_id": doc["doc_id"],
                    "source_type": doc["source_type"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "section_heading": span.heading,
                    "page": span.page,
                    "chunk_index": chunk_index,
                    "text": " ".join(window),
                    "extra": doc.get("extra", {}),
                }
            )
            chunk_index += 1
    return chunks


def main() -> None:
    docs = [json.loads(line) for line in MANIFEST_PATH.open()]
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with CHUNKS_PATH.open("w") as out:
        for doc in docs:
            try:
                chunks = chunk_document(doc)
            except Exception as e:
                print(f"  ! failed to chunk {doc['doc_id']}: {e}")
                continue
            for c in chunks:
                out.write(json.dumps(c) + "\n")
            total_chunks += len(chunks)
            print(f"  {doc['doc_id']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {total_chunks} from {len(docs)} documents")
    print(f"Written to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
