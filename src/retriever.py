"""
retriever.py — FAISS vector-store wrapper.

Loads the pre-built FAISS index from disk (built by src/ingest.py) and exposes
a single `retrieve()` function used by the LangGraph agent and the eval runner.

Design choices:
  - @lru_cache on _store() so the embedding model and index are loaded only once
    per process — important for Streamlit where every request hits the same server.
  - MMR (Maximal Marginal Relevance) is an alternative search mode that trades
    off a bit of relevance for more diverse results. I added it for Phase 8
    experiments but it's off by default because similarity search performed better.
  - Source-label prefix: during ingestion each chunk is prefixed with its source
    filename (e.g. "[cabin baggage] Economy passengers...") so FAISS embeddings
    capture the document topic as well as the content. This prefix is stripped
    from retrieved documents before they reach the grader or generator, so
    downstream prompts always receive clean prose.
"""

import re
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

    if mmr:
        # fetch_k=k*5 gives MMR a larger candidate pool to pick diverse results from
        docs = store.max_marginal_relevance_search(query, k=k, fetch_k=k * 5)
    else:
        docs = store.similarity_search(query, k=k)

    # Strip the source-label prefix — it served its purpose at embedding time.
    for doc in docs:
        doc.page_content = _LABEL_RE.sub("", doc.page_content)

    return docs
