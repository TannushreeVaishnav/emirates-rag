"""
retriever.py — Hybrid search over the local FAISS document corpus.

Vector search and BM25 keyword search contribute candidates, merged with
reciprocal-rank fusion. This helps exact policy terms compete with semantic
matches without treating rank or vector distance as answer confidence.
The store and lexical postings are cached once per process. Returned documents
are copies so cleaning legacy source-label prefixes never mutates the index.
"""

import re
import math
from collections import Counter
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

# Pattern that matches the source-label prepended during ingestion:
# "[cabin baggage] " or "[faq/dangerous goods] " etc.
_LABEL_RE = re.compile(r"^\[[^\]]+\]\s*")


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


_STOPWORDS = frozenset("a an the is are was were do does can i my me to of in on for and or what how about emirates".split())


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


@lru_cache(maxsize=1)
def _lexical_index():
    """Cache BM25 postings over the same indexed chunks; no second document corpus."""
    store = _store()
    docs = [store.docstore.search(key) for key in store.index_to_docstore_id.values()]
    postings = {}
    lengths = []
    for index, doc in enumerate(docs):
        terms = _terms(doc.metadata.get("source", "") + " " + doc.page_content)
        lengths.append(len(terms))
        for term, count in Counter(terms).items():
            postings.setdefault(term, []).append((index, count))
    return docs, postings, lengths, sum(lengths) / max(len(lengths), 1)


def _keyword_search(query: str, k: int) -> list[Document]:
    docs, postings, lengths, average = _lexical_index()
    scores = Counter()
    for term in set(_terms(query)):
        matches = postings.get(term, [])
        weight = math.log(1 + (len(docs) - len(matches) + 0.5) / (len(matches) + 0.5))
        for index, count in matches:
            norm = 1.2 * (0.25 + 0.75 * lengths[index] / max(average, 1))
            scores[index] += weight * count * 2.2 / (count + norm)
    return [docs[index] for index, _ in scores.most_common(k)]


def _key(doc: Document) -> tuple:
    return (doc.metadata.get("source"), doc.metadata.get("page"), _LABEL_RE.sub("", doc.page_content))


def retrieve(query: str, k: int | None = None, mmr: bool | None = None) -> list[Document]:
    """
    Return the top-k most relevant document chunks for a query.

    The source-label prefix added during ingestion (e.g. "[cabin baggage] ...")
    is stripped from each document's page_content before returning so that the
    grader, generator, and output_check receive clean prose without the label.
    The label only exists in the FAISS index to improve embedding discrimination.

    Args:
        query: The user's (possibly rewritten) question.
        k:     Number of chunks to return. Defaults to settings.top_k.
        mmr:   Whether to use Maximal Marginal Relevance search.
               Defaults to settings.use_mmr; pass True/False to override.

    Returns:
        List of LangChain Document objects with clean page_content and metadata.
    """
    k = k or settings.top_k
    mmr = settings.use_mmr if mmr is None else mmr
    store = _store()

    candidates = max(k * 5, 20)
    if mmr:
        dense = store.max_marginal_relevance_search(query, k=candidates, fetch_k=candidates * 2)
    else:
        dense = store.similarity_search(query, k=candidates)
    lexical = _keyword_search(query, candidates)
    # Reciprocal-rank fusion avoids treating vector distances as calibrated confidence.
    scores = Counter()
    documents = {}
    for ranking in (dense, lexical):
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            key = _key(doc)
            if key in seen:
                continue
            seen.add(key)
            documents[key] = doc
            scores[key] += 1 / (60 + rank)
    return [
        Document(page_content=_LABEL_RE.sub("", documents[key].page_content),
                 metadata=dict(documents[key].metadata))
        for key, _ in scores.most_common(k)
    ]
