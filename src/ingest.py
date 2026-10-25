"""
ingest.py — Knowledge-base ingestion pipeline.

Pipeline:
  raw PDF / Markdown  →  clean  →  chunk  →  embed  →  FAISS index

The output (index/) is a pre-built vector store that the agent loads at
runtime. Running ingestion once takes a few minutes; answering questions
then costs only an embedding call per query (< 100 ms).

Key design decisions:
  - Structure-aware PDF splitting: the Emirates Conditions of Carriage uses
    article headings (ARTICLE 7 — REFUSAL OF CARRIAGE). Splitting on these
    boundaries keeps each chunk self-contained and improves citation accuracy.
  - Boilerplate removal: page numbers, repeated headers/footers, and TOC
    dot-leaders are stripped before chunking so they don't pollute embeddings.
  - stats.json: written alongside the index so the Phase 8 experiment runner
    can read the chunk count and settings without re-loading the index.

Usage:
    python -m src.ingest                # build / rebuild the main index
    python -m src.ingest --headings     # debug: list detected CoC sections

Experiment overrides (used by eval/run_experiments.py via subprocess env vars):
    CHUNK_SIZE=500 STRUCTURE_AWARE=false INDEX_DIR=index_exp/c500-fixed python -m src.ingest
"""
import argparse
import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("ingest")

# ── Heading pattern for the Emirates Conditions of Carriage ──────────────────
# Matches lines like "ARTICLE 7 — REFUSAL AND LIMITATION OF CARRIAGE" and
# "ARTICLE 12 — LIABILITY FOR DAMAGE". These become section labels in metadata
# so the agent can cite "Article 7" in its answer.
ARTICLE_RE = re.compile(
    r"^\s*(?:ARTICLE\s+)?(\d{1,2})\s*(?:[.\s—–\-]+)\s*([A-Z][A-Z0-9 ,&/'()\-]{3,})\s*$"
)
# Lines that appear on almost every PDF page (page numbers, footers).
# A line is treated as boilerplate if it appears on > 30% of pages.
PAGE_NUM_RE = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
NAV_NOISE = (
    "skip to the main content",
    "accessibility information",
    "opens a pdf in a new tab",
    "user-provided source text page",
)

# RecursiveCharacterTextSplitter tries larger separators first (paragraphs,
# then lines, then sentences) so chunks break at natural boundaries.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def clean_text(text: str) -> str:
    """
    Normalise and de-noise raw extracted text.

    These PDFs are extracted with a `word\n \nword` layout (a newline + single
    space between every token). The first step collapses that pattern into normal
    prose so downstream splitting and embedding work correctly.

    Also removes hyphenated line-breaks, TOC dot-leaders, repeated navigation
    fragments, and excess whitespace.
    """
    text = text.replace("\r", "")

    # Fix the single-space-per-token extraction layout: "word\n \nword" → "word word"
    # This is the most common quality issue in these Emirates FAQs.
    text = re.sub(r"(\w)\n +(\w)", r"\1 \2", text)
    # Second pass: collapse remaining isolated newlines between words (no leading space)
    text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)

    # Remove trailing RAG Metadata block if present in document
    if "RAG Metadata" in text:
        text = text.split("RAG Metadata")[0]
    # Remove meta prompts
    text = re.sub(r"Answering instruction:.*", "", text, flags=re.IGNORECASE)
    # Rejoin hyphenated line-breaks: "bagga-\nge" → "baggage"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Remove TOC dot leaders (e.g. .............13)
    text = re.sub(r"\.{3,}\s*\d+", "", text)
    # Remove common navigation noise
    text = "\n".join(
        l for l in text.split("\n") if not any(n in l.lower() for n in NAV_NOISE)
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, meta: dict, structured: bool) -> list[Document]:
    pages = [p.extract_text() or "" for p in PdfReader(path).pages]
    lines_per_page = [[l.strip() for l in t.splitlines()] for t in pages]

    # Repeated short lines across many pages are headers/footers.
    counts = Counter(l for lines in lines_per_page for l in set(lines) if l)
    threshold = max(3, int(0.3 * len(pages)))
    boilerplate = {l for l, c in counts.items() if c >= threshold and len(l) < 100}

    segments: list[tuple[str, int, str]] = []   # (section, page, text)
    section = "General"
    for pno, lines in enumerate(lines_per_page, start=1):
        # Ignore TOC page in CoC (page 2) for heading discovery to avoid duplicate headings
        is_toc_page = structured and pno == 2
        buf: list[str] = []
        for line in lines:
            if line in boilerplate or PAGE_NUM_RE.match(line):
                continue
            m = ARTICLE_RE.match(line) if (structured and not is_toc_page) else None
            if m:
                if buf:
                    segments.append((section, pno, "\n".join(buf)))
                    buf = []
                section = f"Article {m.group(1)}: {m.group(2).title()}"
            buf.append(line)
        if buf:
            segments.append((section, pno, "\n".join(buf)))

    chunks = []
    for sec, pno, text in segments:
        cleaned = clean_text(text)
        # Prepend the source filename (without extension) to each chunk so the
        # embedding captures the document topic, not just the passage content.
        # This helps FAISS distinguish "cabin_baggage" chunks from "checked_baggage"
        # chunks when a query like "cabin baggage allowance" is ambiguous.
        source_label = meta.get("source", "").replace("_", " ").replace(".pdf", "")
        for piece in splitter.split_text(cleaned):
            if len(piece) > 40:
                chunks.append(
                    Document(
                        page_content=f"[{source_label}] {piece}",
                        metadata={**meta, "section": sec, "page": pno},
                    )
                )
    return chunks


def load_markdown(path: Path, meta: dict) -> list[Document]:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    source_label = meta.get("source", "").replace("_", " ").replace(".md", "")
    if not settings.structure_aware:      # "fixed" baseline: plain recursive splitting
        return [
            Document(
                page_content=f"[{source_label}] {p}",
                metadata={**meta, "section": "General", "page": None},
            )
            for p in splitter.split_text(text) if len(p) > 40
        ]
    md_split = MarkdownHeaderTextSplitter(
        [("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
    )
    chunks = []
    for part in md_split.split_text(text):
        section = " > ".join(part.metadata[k] for k in ("h1", "h2", "h3") if k in part.metadata)
        for piece in splitter.split_text(part.page_content):
            if len(piece) > 40:
                chunks.append(
                    Document(
                        page_content=f"[{source_label}] {piece}",
                        metadata={**meta, "section": section or "General", "page": None},
                    )
                )
    return chunks


def load_all(headings_only: bool = False) -> list[Document]:
    all_chunks: list[Document] = []
    with open(settings.sources_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = settings.raw_dir / row["file"]
            if not path.exists():
                log.warning("Missing, skipped: %s", row["file"])
                continue
            meta = {
                "source": path.name,
                "url": row["url"],
                "market": row["market"],
                "doc_type": row["doc_type"],
                "date_collected": row["date_collected"],
            }
            if path.suffix.lower() == ".pdf":
                structured = settings.structure_aware and "conditions_of_carriage" in path.name
                chunks = load_pdf(path, meta, structured)
            else:
                chunks = load_markdown(path, meta)

            if headings_only:
                if "conditions_of_carriage" in path.name:
                    seen = []
                    for c in chunks:
                        if c.metadata["section"] not in seen:
                            seen.append(c.metadata["section"])
                    print(f"\n{len(seen)} sections detected in {path.name}:")
                    for s in seen:
                        print("  ", s)
                continue

            log.info("%-40s -> %3d chunks", row["file"], len(chunks))
            all_chunks.extend(chunks)

    for i, c in enumerate(all_chunks):
        c.metadata["chunk_id"] = f"{c.metadata['source']}::{i}"
    return all_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headings", action="store_true", help="print detected CoC sections and exit")
    args = ap.parse_args()

    chunks = load_all(headings_only=args.headings)
    if args.headings:
        return
    if not chunks:
        raise SystemExit("No chunks produced. Check data/sources.csv and data/raw/.")

    log.info("Embedding %d chunks with %s ...", len(chunks), settings.embedding_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    # Rebuilt from scratch each run, so ingestion is idempotent.
    store = FAISS.from_documents(chunks, embeddings)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.index_dir))

    sizes = [len(c.page_content) for c in chunks]
    (settings.index_dir / "stats.json").write_text(json.dumps({
        "chunks": len(chunks),
        "avg_chars": sum(sizes) // len(sizes),
        "max_chars": max(sizes),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "structure_aware": settings.structure_aware,
        "embedding_model": settings.embedding_model,
    }, indent=2))
    log.info("Done. %d chunks, avg %d chars, max %d. Index saved to %s",
             len(chunks), sum(sizes) // len(sizes), max(sizes), settings.index_dir)


if __name__ == "__main__":
    main()
