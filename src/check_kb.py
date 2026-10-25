"""
check_kb.py — Quick sanity-check for the knowledge base files.

Run this before ingestion to verify that every file listed in sources.csv
actually exists and has enough text content. PDFs that were copy-pasted from
accordion-heavy pages often end up nearly empty after text extraction.

Usage:
    python -m src.check_kb

Output example:
    OK (14)
      cabin_baggage.pdf: 12,450 chars
      ...
    TOO SHORT (2)  ← re-copy these; accordions probably weren't expanded
      extra_baggage.pdf: 312 chars
    MISSING (1)
      skywards_miles.pdf
"""

import csv
from pathlib import Path

# Paths are relative to the project root, so run from there.
RAW = Path("data/raw")
SOURCES = Path("data/sources.csv")

# Files with fewer than this many characters are flagged — they likely weren't
# copy-pasted correctly (JavaScript-rendered accordions collapse to near-empty text).
MIN_CHARS = 500

missing: list[str] = []
too_short: list[tuple[str, int]] = []
ok: list[tuple[str, int]] = []

with open(SOURCES, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        path = RAW / row["file"]

        if not path.exists():
            missing.append(row["file"])
            continue

        # Extract raw text length — PDFs need pypdf, markdown/text can be read directly.
        if path.suffix == ".pdf":
            from pypdf import PdfReader
            text = "".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        bucket = too_short if len(text) < MIN_CHARS else ok
        bucket.append((row["file"], len(text)))

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\nOK ({len(ok)})")
for name, n in ok:
    print(f"  {name}: {n:,} chars")

print(f"\nTOO SHORT ({len(too_short)})  ← re-copy these, accordions probably not expanded")
for name, n in too_short:
    print(f"  {name}: {n:,} chars")

print(f"\nMISSING ({len(missing)})")
for name in missing:
    print(f"  {name}")
