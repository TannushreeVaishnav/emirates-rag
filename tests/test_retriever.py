"""Offline consumer-level hybrid retrieval regressions; no models are loaded."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src import retriever


def make_doc(text, source="cabin_baggage.pdf", chunk_id="cabin::1", **metadata):
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "chunk_id": chunk_id,
            "page": 1,
            "section": "Economy Class",
            "url": "https://www.emirates.com/policy",
            **metadata,
        },
    )


class FakeStore:
    def __init__(self, documents, dense):
        self.index_to_docstore_id = {index: str(index) for index in range(len(documents))}
        self.documents = dict(zip(self.index_to_docstore_id.values(), documents))
        self.docstore = SimpleNamespace(search=self.documents.__getitem__)
        self.dense = dense
        self.calls = []

    def similarity_search(self, query, k):
        self.calls.append(("similarity", query, k))
        return self.dense[:k]

    def max_marginal_relevance_search(self, query, k, fetch_k):
        self.calls.append(("mmr", query, k, fetch_k))
        return self.dense[:k]


@pytest.fixture(autouse=True)
def isolate_cache_and_models(monkeypatch):
    retriever._lexical_index.cache_clear()
    monkeypatch.setattr(
        retriever,
        "HuggingFaceEmbeddings",
        lambda **kwargs: pytest.fail("offline retrieval tests must not load an embedding model"),
    )
    yield
    retriever._lexical_index.cache_clear()


def use_store(monkeypatch, documents, dense):
    store = FakeStore(documents, dense)
    monkeypatch.setattr(retriever, "_store", lambda: store)
    return store


@pytest.mark.parametrize("mmr", [False, True])
def test_exact_cabin_policy_is_recovered_when_dense_search_misses(monkeypatch, mmr):
    checked = make_doc(
        "[checked baggage] Checked bags are delivered at the airport.",
        source="checked_baggage.pdf",
        chunk_id="checked::1",
    )
    cabin = make_doc(
        "[cabin baggage] Economy Class allowance: 7 kg. Dimensions: 55 × 38 × 22 cm."
    )
    store = use_store(monkeypatch, [checked, cabin], [checked])
    docs = retriever.retrieve("Economy cabin baggage allowance dimensions", k=2, mmr=mmr)
    assert any("Economy Class allowance: 7 kg" in doc.page_content for doc in docs)
    assert len(docs) == 2
    assert store.calls[0][0] == ("mmr" if mmr else "similarity")
    assert all(not doc.page_content.startswith("[") for doc in docs)


def test_source_labels_remain_searchable_but_are_not_returned(monkeypatch):
    cabin = make_doc("[cabin baggage] Allowance: 7 kg; dimensions: 55 × 38 × 22 cm.")
    use_store(monkeypatch, [cabin], [])
    docs = retriever.retrieve("cabin", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "Allowance: 7 kg; dimensions: 55 × 38 × 22 cm."
    assert cabin.page_content.startswith("[cabin baggage]")


def test_duplicate_chunks_and_ranking_overlap_are_deduplicated(monkeypatch):
    cabin = make_doc("[cabin baggage] Economy cabin baggage allowance is 7 kg.")
    duplicate = make_doc(cabin.page_content, chunk_id="cabin::duplicate", page=2)
    banks = make_doc(
        "[dangerous goods] One power bank per passenger, maximum 100 Wh, cabin only.",
        source="dangerous_goods.pdf",
        chunk_id="dangerous::1",
    )
    use_store(monkeypatch, [cabin, duplicate, banks], [cabin, duplicate, cabin, banks])
    docs = retriever.retrieve("cabin power bank", k=5)
    assert len(docs) == 2
    assert sum("7 kg" in doc.page_content for doc in docs) == 1
    assert any("100 Wh" in doc.page_content for doc in docs)


def test_returned_documents_and_nested_metadata_do_not_mutate_cached_index(monkeypatch):
    cached = make_doc(
        "[cabin baggage] Economy cabin allowance is 7 kg.",
        provenance={"pages": [1, 2]},
        market="IN",
    )
    original = deepcopy(cached)
    use_store(monkeypatch, [cached], [cached])
    first = retriever.retrieve("cabin allowance", k=1)[0]
    assert first is not cached
    assert first.metadata == original.metadata
    first.page_content = "changed"
    first.metadata["provenance"]["pages"].append(99)
    first.metadata["source"] = "changed.pdf"
    second = retriever.retrieve("cabin allowance", k=1)[0]
    assert cached == original
    assert second.page_content == "Economy cabin allowance is 7 kg."
    assert second.metadata == original.metadata
    assert second is not first


def test_reciprocal_rank_fusion_favours_evidence_in_both_rankings(monkeypatch):
    dense_only = make_doc("[other] Unrelated airport transport.", "other.pdf", "other::1")
    both = make_doc("[cabin baggage] Economy cabin allowance is 7 kg.")
    lexical_only = make_doc(
        "[cabin baggage] Economy cabin allowance Economy cabin allowance.",
        chunk_id="cabin::2",
    )
    use_store(monkeypatch, [dense_only, both, lexical_only], [dense_only, both])
    docs = retriever.retrieve("Economy cabin allowance", k=3)
    assert docs[0].metadata["chunk_id"] == "cabin::1"


def test_fusion_ties_have_repeatable_order(monkeypatch):
    alpha = make_doc("[alpha] Policy alpha.", "alpha.pdf", "alpha::1")
    beta = make_doc("[beta] Policy beta.", "beta.pdf", "beta::1")
    use_store(monkeypatch, [alpha, beta], [beta, alpha])
    first = retriever.retrieve("policy", k=2)
    second = retriever.retrieve("policy", k=2)
    assert [doc.metadata["source"] for doc in first] == ["alpha.pdf", "beta.pdf"]
    assert [doc.metadata for doc in first] == [doc.metadata for doc in second]


def test_explicit_empty_result_limit_does_not_load_index(monkeypatch):
    monkeypatch.setattr(retriever, "_store", lambda: pytest.fail("index should not load for k=0"))
    assert retriever.retrieve("cabin allowance", k=0) == []
