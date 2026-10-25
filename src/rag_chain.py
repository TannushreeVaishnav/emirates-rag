"""
rag_chain.py — Simple (Phase 3) RAG chain — baseline for comparison.

This is the "before" system that the LangGraph agent is measured against.
It does three things:

  1. Retrieve the top-k most similar chunks from the FAISS index.
  2. Format them as numbered passages and inject them into the generation prompt.
  3. Parse the model's response and attach source citations.

There's no routing, grading, rewriting, or output check here — those are
LangGraph-only features. Running this baseline gives honest numbers to compare
against the full agent in the eval report.

Usage:
    from src.rag_chain import ask
    result = ask("What is the cabin baggage allowance for Economy?")
    print(result["answer"])
    print(result["citations"])
"""

import time
import re
from urllib.parse import urlsplit

from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.llm_gateway import get_llm, start_tracking
from src.prompts import GENERATION_PROMPT, NO_ANSWER_MSG
from src.retriever import retrieve


def format_context(docs: list) -> str:
    """
    Format retrieved documents as numbered passages for the generation prompt.

    Each passage gets a number [1], [2], ... that the model uses for citations.
    A short metadata header (source file and section) is prepended so the model
    knows where each passage came from.

    Example output:
        [1] cabin_baggage.pdf — Cabin Allowances
        Economy passengers may carry one piece of cabin baggage ...

        [2] checked_baggage.pdf — Baggage Policy
        ...
    """
    lines = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        lines.append(f"[{i}] {m['source']} — {m.get('section', 'General')}")
        lines.append(doc.page_content)
        lines.append("")  # blank line between passages
    return "\n".join(lines)


def normalize_citations(answer: str) -> str:
    """Canonicalize numeric citation markers without inventing references."""
    return re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", answer)


def citation_numbers(answer: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[(\d+)\]", normalize_citations(answer))]


def citation_markdown(answer: str, citations: list[dict]) -> str:
    """Link only references backed by a structured source with an HTTP(S) URL."""
    urls = {}
    for citation in citations:
        url = citation.get("url", "")
        parsed = urlsplit(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            if citation.get("page") and parsed.path.lower().endswith(".pdf") and not parsed.fragment:
                url += f"#page={citation['page']}"
            urls[citation["ref"]] = url.replace("(", "%28").replace(")", "%29")
    return re.sub(
        r"\[(\d+)\]",
        lambda match: f"[[{match[1]}]]({urls[int(match[1])]})"
        if int(match[1]) in urls else match[0],
        normalize_citations(answer),
    )


def build_citations(docs: list, answer: str) -> list[dict]:
    """
    Build the citations list from documents whose index was referenced in the answer.

    The model is instructed to cite passages as [1], [2], etc. This function
    checks which numbers appear in the answer and returns those documents as
    structured citation objects.

    Returns:
        List of dicts with keys: ref, source, section, page, url, date_collected, text.
    """
    citations = []
    references = set(citation_numbers(answer))
    for i, doc in enumerate(docs, start=1):
        # Only include this doc if its citation number actually appears in the answer
        if i in references:
            m = doc.metadata
            citations.append({
                "ref": i,
                "source": m["source"],
                "section": m.get("section", "General"),
                "page": m.get("page"),
                "url": m.get("url", ""),
                "date_collected": m.get("date_collected", ""),
                "text": doc.page_content,
            })
    return citations


def ask(question: str) -> dict:
    """
    Run the baseline RAG pipeline: retrieve → format → generate → cite.

    Args:
        question: The user's question (plain string, no history).

    Returns:
        Dict with keys:
            answer      — the generated answer string
            citations   — list of citation dicts (empty if NO_ANSWER)
            docs        — raw retrieved Document objects (for eval hit-rate check)
            context_docs — same as docs for baseline (no grading step)
            latency     — wall-clock seconds for the full call
    """
    t0 = time.perf_counter()
    start_tracking()
    docs = retrieve(question)

    chain = GENERATION_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({
        "context": format_context(docs),
        "question": question,
    }).strip()

    refused = answer.startswith(NO_ANSWER_MSG[:40])
    return {
        "answer": answer,
        "citations": [] if refused else build_citations(docs, answer),
        "docs": docs,
        "context_docs": docs,   # baseline has no grader; all retrieved docs are "context"
        "latency": time.perf_counter() - t0,
    }
