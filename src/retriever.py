"""
retriever.py — Deterministic lexical + FAISS hybrid retrieval.

Loads the pre-built FAISS index from disk (built by src/ingest.py) and exposes
a single `retrieve()` function used by the LangGraph agent and the eval runner.

Design choices:
  - @lru_cache on _store() so the embedding model and index are loaded only once
    per process — important for Streamlit where every request hits the same server.
  - BM25 over all indexed chunks recovers exact policy terms missed by dense
    search; reciprocal-rank fusion combines it with similarity search or MMR.
  - Source-label prefix: during ingestion each chunk is prefixed with its source
    filename (e.g. "[cabin baggage] Economy passengers...") so FAISS embeddings
    capture the document topic as well as the content. This prefix is stripped
    from retrieved documents before they reach the grader or generator, so
    downstream prompts always receive clean prose.
"""

import math
import re
from collections import Counter
from copy import deepcopy
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

# Pattern that matches the source-label prepended during ingestion:
# "[cabin baggage] " or "[faq/dangerous goods] " etc.
_LABEL_RE = re.compile(r"^\[[^\]]+\]\s*")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _document_key(doc: Document) -> tuple[str, str]:
    """Collapse repeated chunks without conflating independent sources."""
    return (
        str(doc.metadata.get("source", "")),
        " ".join(_LABEL_RE.sub("", doc.page_content).split()),
    )


@lru_cache(maxsize=1)
def _lexical_index(store: FAISS) -> tuple:
    """Build local BM25 statistics once for the cached store, including labels."""
    documents = []
    seen = set()
    for index in sorted(store.index_to_docstore_id):
        doc = store.docstore.search(store.index_to_docstore_id[index])
        if not isinstance(doc, Document):
            continue
        key = _document_key(doc)
        if key not in seen:
            seen.add(key)
            documents.append(doc)
    terms = [Counter(_TOKEN_RE.findall(doc.page_content.casefold())) for doc in documents]
    frequencies = Counter(token for counts in terms for token in counts)
    lengths = [sum(counts.values()) for counts in terms]
    average = sum(lengths) / len(lengths) if lengths else 1.0
    return documents, terms, frequencies, lengths, average


def _lexical_search(store: FAISS, query: str, k: int) -> list[Document]:
    documents, terms, frequencies, lengths, average = _lexical_index(store)
    total = len(documents)
    query_weights = {
        token: math.log(
            1 + (total - frequencies[token] + 0.5) / (frequencies[token] + 0.5)
        )
        for token in sorted(set(_TOKEN_RE.findall(query.casefold())))
        if frequencies[token]
    }
    ranked = []
    for index, counts in enumerate(terms):
        score = 0.0
        for token, inverse_frequency in query_weights.items():
            frequency = counts.get(token, 0)
            if frequency:
                score += inverse_frequency * frequency * 2.5 / (
                    frequency + 1.5 * (0.25 + 0.75 * lengths[index] / average)
                )
        if score > 0:
            ranked.append((score, index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [documents[index] for _, index in ranked[:k]]


@lru_cache(maxsize=1)
def _store() -> FAISS:
    """
    Load and cache the FAISS index.

    lru_cache(maxsize=1) means this runs exactly once per process, regardless
    of how many questions are asked. The embedding model (~90 MB) stays in RAM.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},  # needed for cosine similarity
    )
    return FAISS.load_local(
        str(settings.index_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # required by LangChain for local indexes
    )


def retrieve(query: str, k: int | None = None, mmr: bool | None = None) -> list[Document]:
    """
    Return top-k chunks from reciprocal-rank fused dense and lexical rankings.

    Labels participate in both rankings. Returned documents are independent
    copies with labels removed; cached indexed text and metadata remain intact.

    Args:
        query: The user's (possibly rewritten) question.
        k:     Number of chunks to return. Defaults to settings.top_k.
        mmr:   Whether to use Maximal Marginal Relevance search.
               Defaults to settings.use_mmr; pass True/False to override.

    Returns:
        List of LangChain Document objects with clean page_content and metadata.
    """
    k = settings.top_k if k is None else k
    if k <= 0:
        return []
    mmr = settings.use_mmr if mmr is None else mmr
    store = _store()
    candidate_count = max(20, k * 5)

    if mmr:
        dense = store.max_marginal_relevance_search(
            query, k=candidate_count, fetch_k=candidate_count * 5
        )
    else:
        dense = store.similarity_search(query, k=candidate_count)
    lexical = _lexical_search(store, query, candidate_count)

    scores = {}
    documents = {}
    for ranking in (dense, lexical):
        seen = set()
        rank = 0
        for doc in ranking:
            key = _document_key(doc)
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)

    # Stable key tie-breaking is independent of hash/set iteration order.
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    result = []
    for key in ordered[:k]:
        doc = deepcopy(documents[key])
        doc.page_content = _LABEL_RE.sub("", doc.page_content)
        result.append(doc)
    return result
