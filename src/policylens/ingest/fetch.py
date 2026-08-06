"""Download the corpus defined in sources.py into data/raw/, with a manifest.

Idempotent and re-runnable: `make ingest` calls this fresh, so the manifest is
overwritten each run rather than appended to. This is the reproducibility
boundary for the corpus — data/raw/ itself is gitignored (large binary files),
but sources.py + this script + the committed manifest let anyone regenerate the
exact same document set.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from policylens.ingest.sources import (
    ISSUERS,
    MAX_BULLETINS_PER_STATE,
    NAIC_MODEL_LAWS,
    STATE_BULLETIN_LISTING_PAGES,
)

USER_AGENT = "PolicyLens research project (vedaantagrawal17@gmail.com)"
REQUEST_DELAY_SECONDS = 0.4  # polite rate limit against public/government sites

DATA_RAW = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.jsonl")  # committed provenance trail; data/raw/ itself is not


@dataclass
class ManifestEntry:
    doc_id: str
    source_type: str
    title: str
    url: str
    local_path: str
    fetched_at: str
    extra: dict = field(default_factory=dict)


def _get(url: str, **kwargs) -> requests.Response:
    time.sleep(REQUEST_DELAY_SECONDS)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


SEC_10KS_PER_ISSUER = 3  # most recent N annual filings — real content, same real companies


def fetch_sec_10ks() -> list[ManifestEntry]:
    entries = []
    for issuer in ISSUERS:
        submissions = _get(f"https://data.sec.gov/submissions/CIK{issuer.cik}.json").json()
        recent = submissions["filings"]["recent"]
        forms = recent["form"]
        indices = [i for i, f in enumerate(forms) if f == "10-K"][:SEC_10KS_PER_ISSUER]
        if not indices:
            print(f"  ! no 10-K found for {issuer.ticker}, skipping")
            continue

        cik_int = str(int(issuer.cik))
        for position, idx in enumerate(indices):
            accession = recent["accessionNumber"][idx].replace("-", "")
            primary_doc = recent["primaryDocument"][idx]
            filing_date = recent["filingDate"][idx]
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{primary_doc}"

            resp = _get(doc_url)
            local_path = DATA_RAW / "sec_10k" / f"{issuer.ticker}_{filing_date}.html"
            _write(local_path, resp.content)

            # The most recent filing keeps the original bare doc_id
            # (sec10k_{ticker}) — the existing golden set's chunk_ids
            # (e.g. sec10k_MET_66) assume that exact id and would silently
            # break if it grew a date suffix. Older filings get one, since
            # they're new documents with no existing references to preserve.
            doc_id = f"sec10k_{issuer.ticker}" if position == 0 else f"sec10k_{issuer.ticker}_{filing_date}"

            entries.append(
                ManifestEntry(
                    doc_id=doc_id,
                    source_type="sec_10k",
                    title=f"{issuer.name} 10-K ({filing_date})",
                    url=doc_url,
                    local_path=str(local_path),
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    extra={"ticker": issuer.ticker, "cik": issuer.cik, "filing_date": filing_date},
                )
            )
            print(f"  fetched {issuer.ticker} 10-K ({filing_date})")
    return entries


def fetch_naic_model_laws() -> list[ManifestEntry]:
    entries = []
    for number, title in NAIC_MODEL_LAWS.items():
        url = f"https://content.naic.org/sites/default/files/model-law-{number}.pdf"
        resp = _get(url)
        local_path = DATA_RAW / "naic_model_law" / f"model-law-{number}.pdf"
        _write(local_path, resp.content)
        entries.append(
            ManifestEntry(
                doc_id=f"naic_{number}",
                source_type="naic_model_law",
                title=title,
                url=url,
                local_path=str(local_path),
                fetched_at=datetime.now(timezone.utc).isoformat(),
                extra={"model_number": number},
            )
        )
        print(f"  fetched NAIC model law {number}: {title}")
    return entries


def _extract_ny_bulletin_links(listing_html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(listing_html, "lxml")
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"/circular[_-]letters?/[a-z0-9_-]+$", href, re.I) and "withdrawn" not in href:
            full_url = urljoin(base_url, href)
            if full_url not in seen and not href.rstrip("/").endswith("circular_letters"):
                seen.add(full_url)
                links.append((full_url, a.get_text(strip=True) or full_url))
    return links


def _extract_ca_bulletin_links(listing_html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(listing_html, "lxml")
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        is_pdf_bulletin = href.lower().endswith(".pdf") and "bulletin" in href.lower()
        is_cfm_bulletin = re.search(r"/bulletin-\d{4}-\d+\.cfm$", href, re.I)
        if is_pdf_bulletin or is_cfm_bulletin:
            full_url = urljoin(base_url, href)
            if full_url not in seen:
                seen.add(full_url)
                links.append((full_url, a.get_text(strip=True) or full_url))
    return links


def _extract_tx_bulletin_links(listing_html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(listing_html, "lxml")
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"b-\d{4}-\d+\.html$", href, re.I):
            full_url = urljoin(base_url, href)
            if full_url not in seen:
                seen.add(full_url)
                links.append((full_url, a.get_text(strip=True) or full_url))
    return links


def fetch_state_bulletins() -> list[ManifestEntry]:
    entries = []
    extractors = {
        "NY": _extract_ny_bulletin_links,
        "CA": _extract_ca_bulletin_links,
        "TX": _extract_tx_bulletin_links,
    }
    # TX only lists one year per index page; pull a few years to reach the cap.
    extra_listing_pages = {
        "TX": [
            "https://www.tdi.texas.gov/bulletins/2025/index.html",
            "https://www.tdi.texas.gov/bulletins/2024/index.html",
        ]
    }

    for state, listing_url in STATE_BULLETIN_LISTING_PAGES.items():
        listing_urls = [listing_url] + extra_listing_pages.get(state, [])
        links: list[tuple[str, str]] = []
        for url in listing_urls:
            resp = _get(url)
            links.extend(extractors[state](resp.text, url))
            if len(links) >= MAX_BULLETINS_PER_STATE:
                break

        links = links[:MAX_BULLETINS_PER_STATE]
        for i, (doc_url, doc_title) in enumerate(links):
            resp = _get(doc_url)
            is_pdf = doc_url.lower().endswith(".pdf") or resp.headers.get("Content-Type", "").startswith(
                "application/pdf"
            )
            ext = "pdf" if is_pdf else "html"
            local_path = DATA_RAW / "state_bulletin" / state / f"{state.lower()}_{i:03d}.{ext}"
            _write(local_path, resp.content)
            entries.append(
                ManifestEntry(
                    doc_id=f"state_{state.lower()}_{i:03d}",
                    source_type="state_bulletin",
                    title=doc_title[:200],
                    url=doc_url,
                    local_path=str(local_path),
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    extra={"state": state},
                )
            )
        print(f"  fetched {len(links)} bulletins for {state}")
    return entries


def main() -> None:
    print("Fetching SEC 10-Ks...")
    sec_entries = fetch_sec_10ks()
    print("Fetching NAIC model laws...")
    naic_entries = fetch_naic_model_laws()
    print("Fetching state DOI bulletins...")
    state_entries = fetch_state_bulletins()

    all_entries = sec_entries + naic_entries + state_entries
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry.__dict__) + "\n")

    print(f"\nTotal documents fetched: {len(all_entries)}")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
