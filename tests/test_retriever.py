from types import SimpleNamespace

from langchain_core.documents import Document

import src.retriever as r


def test_keyword_evidence_recovers_policy_missed_by_dense_search(monkeypatch):
    unrelated = Document(page_content="Online check-in opens before departure.", metadata={"source": "checkin.pdf"})
    policy = Document(page_content="[dangerous goods] Portable chargers and power banks must be carried in cabin baggage.", metadata={"source": "dangerous_goods.pdf"})
    documents = {"one": unrelated, "two": policy}
    store = SimpleNamespace(
        index_to_docstore_id={0: "one", 1: "two"},
        docstore=SimpleNamespace(search=documents.get),
        similarity_search=lambda query, k: [unrelated, unrelated],
    )
    monkeypatch.setattr(r, "_store", lambda: store)
    r._lexical_index.cache_clear()
    try:
        result = r.retrieve("power banks portable chargers", k=2)
        assert any("must be carried in cabin" in doc.page_content for doc in result)
        assert len({doc.page_content for doc in result}) == len(result)
        retrieved_policy = next(doc for doc in result if doc.metadata["source"] == "dangerous_goods.pdf")
        assert retrieved_policy.page_content == policy.page_content.removeprefix("[dangerous goods] ")
        assert policy.page_content.startswith("[dangerous goods]")
    finally:
        r._lexical_index.cache_clear()
