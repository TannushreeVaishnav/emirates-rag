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
  - Layout-preserving extraction keeps headings, numbers, and repeated policy
    words intact; only explicitly labelled page numbers and navigation are removed.
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
# Bare numbers may be policy limits, not page numbers.
PAGE_NUM_RE = re.compile(r"^page\s+\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
PDF_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][^\n.!?]{0,100}$")
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
    """Normalise extracted tokens without flattening real paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", text)
    # Token-per-line PDFs use a whitespace-only line between every token.
    # Do this before trimming lines, which would make these indistinguishable
    # from genuine blank paragraph separators.
    text = re.sub(r"(?<=\S)\n[ \t]+\n(?=\S)", " ", text)
    text = re.sub(
        r"(?m)^(?:[^\s]+\n){2,}[^\s]+$",
        lambda match: " ".join(match.group().split()),
        text,
    )
    # Also handle runs of bare, blank-line-separated tokens. Multiword
    # paragraphs/headings delimit runs and retain their original boundaries.
    paragraphs = re.split(r"\n[ \t]*\n", text)
    normalised: list[str] = []
    tokens: list[str] = []
    for paragraph in paragraphs:
        if paragraph.strip() and len(paragraph.split()) == 1:
            tokens.append(paragraph.strip())
        else:
            if tokens:
                normalised.append(" ".join(tokens))
                tokens = []
            normalised.append(paragraph.strip())
    if tokens:
        normalised.append(" ".join(tokens))
    text = "\n\n".join(normalised)
    # Remove TOC dot leaders (e.g. .............13).
    text = re.sub(r"\.{3,}\s*\d+", "", text)
    # Remove common navigation noise
    text = "\n".join(
        l for l in text.split("\n") if not any(n in l.lower() for n in NAV_NOISE)
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, meta: dict, structured: bool) -> list[Document]:
    # Layout extraction reconstructs physical lines from PDF coordinates instead
    # of treating every drawn token as a paragraph.
    pages = [p.extract_text(extraction_mode="layout") or "" for p in PdfReader(path).pages]
    segments: list[tuple[str, int, str]] = []   # (section, page, text)
    section = "General"
    heading_path: list[tuple[str, str]] = []
    for pno, page in enumerate(pages, start=1):
        is_toc_page = structured and pno == 2
        buf: list[str] = []
        for line in clean_text(page).splitlines():
            line = line.strip()
            if PAGE_NUM_RE.fullmatch(line):
                continue
            article = ARTICLE_RE.match(line) if (structured and not is_toc_page) else None
            is_heading = article is not None or (
                not structured and PDF_HEADING_RE.fullmatch(line) is not None
            )
            if is_heading:
                if any(buf) and "\n".join(buf).strip() != section:
                    segments.append((section, pno, "\n".join(buf)))
                if article:
                    section = f"Article {article.group(1)}: {article.group(2).title()}"
                else:
                    number = re.match(r"^(\d+(?:\.\d+)*)", line).group(1)
                    heading_path = [
                        (parent_number, title) for parent_number, title in heading_path
                        if number.startswith(parent_number + ".")
                    ]
                    heading_path.append((number, line))
                    section = " > ".join(title for _, title in heading_path)
                buf = [section]
            else:
                buf.append(line)
        if any(buf) and "\n".join(buf).strip() != section:
            segments.append((section, pno, "\n".join(buf)))

    chunks = []
    for sec, pno, text in segments:
        cleaned = clean_text(text)
        # Repeat the section heading on every continuation chunk so allowances
        # remain attributable to their cabin class even across page boundaries.
        prefix = "" if sec == "General" else f"{sec}\n"
        # Prepend the source filename (without extension) to each chunk so the
        # embedding captures the document topic, not just the passage content.
        # This helps FAISS distinguish "cabin_baggage" chunks from "checked_baggage"
        # chunks when a query like "cabin baggage allowance" is ambiguous.
        source_label = meta.get("source", "").replace("_", " ").replace(".pdf", "")
        available = max(1, settings.chunk_size - len(prefix))
        section_splitter = RecursiveCharacterTextSplitter(
            chunk_size=available,
            chunk_overlap=min(settings.chunk_overlap, available - 1),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for piece in section_splitter.split_text(cleaned):
            if piece.strip():
                chunks.append(
                    Document(
                        page_content=(
                            f"[{source_label}] "
                            f"{'' if piece.startswith(sec) else prefix}{piece}"
                        ),
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
            for p in splitter.split_text(text) if p.strip()
        ]
    md_split = MarkdownHeaderTextSplitter(
        [("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
    )
    chunks = []
    for part in md_split.split_text(text):
        section = " > ".join(part.metadata[k] for k in ("h1", "h2", "h3") if k in part.metadata)
        for piece in splitter.split_text(part.page_content):
            if piece.strip():
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
