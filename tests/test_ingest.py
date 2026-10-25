"""Offline regressions for policy-preserving extraction and chunking."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import ingest


ROOT = Path(__file__).resolve().parents[1]
META = {
    "source": "cabin_baggage.pdf",
    "url": "https://www.emirates.com/cabin-baggage",
    "market": "IN",
    "doc_type": "faq",
    "date_collected": "2026-10-25",
}


def stub_pdf(monkeypatch, pages):
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self, *, extraction_mode):
            assert extraction_mode == "layout"
            return self.text

    monkeypatch.setattr(
        ingest, "PdfReader", lambda path: SimpleNamespace(pages=[Page(text) for text in pages])
    )


@pytest.mark.parametrize("separator", ["\n \n", "\n\n", "\n"])
def test_token_layout_preserves_repeated_words_negation_and_numbers(separator):
    text = separator.join(["Passengers", "must", "not", "not", "carry", "7", "kg"])
    assert ingest.clean_text(text) == "Passengers must not not carry 7 kg"


def test_clean_text_retains_real_section_and_paragraph_boundaries():
    text = (
        "2.1 Economy Class\n\nAllowance: 7 kg\nMaximum dimensions: 55 × 38 × 22 cm"
        "\n\n2.2 Premium Economy\n\nAllowance: 10 kg"
    )
    assert ingest.clean_text(text) == text
    assert ingest.clean_text("Cabin bagga-\nge must not be checked.") == (
        "Cabin baggage must not be checked."
    )


def test_pdf_repeated_policy_words_and_bare_limits_are_not_boilerplate(monkeypatch):
    stub_pdf(monkeypatch, [
        f"Policy for passengers\nPassengers must not not carry\n7\nkg\nPage {page}"
        for page in range(1, 5)
    ])
    docs = ingest.load_pdf(Path("unused.pdf"), META, structured=False)
    assert len(docs) == 4
    for page, doc in enumerate(docs, start=1):
        assert "Passengers must not not carry" in doc.page_content
        assert "7" in doc.page_content and "kg" in doc.page_content
        assert "Page " not in doc.page_content
        assert doc.metadata == {**META, "section": "General", "page": page}
        assert doc.page_content.startswith("[cabin baggage] ")


def test_short_policy_chunks_are_retained(monkeypatch):
    stub_pdf(monkeypatch, ["7"])
    docs = ingest.load_pdf(Path("unused.pdf"), META, structured=False)
    assert len(docs) == 1
    assert docs[0].page_content == "[cabin baggage] 7"


def test_heading_is_repeated_with_allowance_after_page_boundary(monkeypatch):
    stub_pdf(monkeypatch, [
        "3.1 Economy Class",
        "Allowance: 15 lb\nMaximum dimensions: 22 × 15 × 8 inches",
    ])
    docs = ingest.load_pdf(Path("unused.pdf"), META, structured=False)
    allowance = next(doc for doc in docs if "15 lb" in doc.page_content)
    assert "3.1 Economy Class" in allowance.page_content
    assert allowance.metadata["section"] == "3.1 Economy Class"
    assert allowance.metadata["page"] == 2


def test_continuation_chunks_keep_the_cabin_class(monkeypatch):
    monkeypatch.setattr(ingest.settings, "chunk_size", 160)
    monkeypatch.setattr(ingest.settings, "chunk_overlap", 15)
    stub_pdf(monkeypatch, [
        "2.1 Economy Class\n\nAllowance: 7 kg\n\n"
        + "Maximum dimensions: 55 × 38 × 22 cm. " * 10
    ])
    docs = ingest.load_pdf(Path("unused.pdf"), META, structured=False)
    assert len(docs) > 1
    assert all("2.1 Economy Class" in doc.page_content for doc in docs)
    assert any("Allowance: 7 kg" in doc.page_content for doc in docs)
    assert all(doc.metadata["section"] == "2.1 Economy Class" for doc in docs)


def test_real_cabin_pdf_keeps_economy_allowance_coherent():
    docs = ingest.load_pdf(ROOT / "data/raw/faq/cabin_baggage.pdf", META, structured=False)
    assert any(
        "2.1 Economy Class" in doc.page_content
        and "Allowance: 7 kg" in doc.page_content
        and "55 × 38 × 22 cm" in doc.page_content
        for doc in docs
    )
    assert all(all(doc.metadata[key] == value for key, value in META.items()) for doc in docs)
    assert all(doc.metadata["page"] >= 1 and doc.metadata["section"] for doc in docs)


def test_real_power_bank_and_checkin_policy_limits_survive():
    dangerous = ingest.load_pdf(
        ROOT / "data/raw/faq/dangerous_goods.pdf",
        {**META, "source": "dangerous_goods.pdf"},
        structured=False,
    )
    text = " ".join(doc.page_content for doc in dangerous)
    assert "Each passenger may carry one power bank." in text
    assert "maximum capacity of 100 Wh" in text
    assert "permitted in cabin baggage only" in text
    assert "prohibited in checked baggage" in text
    checkin = ingest.load_pdf(
        ROOT / "data/raw/faq/checkin_options.pdf",
        {**META, "source": "checkin_options.pdf"},
        structured=False,
    )
    assert any("48 hours before the flight" in doc.page_content for doc in checkin)


def test_short_markdown_policy_is_retained(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# Cabin\n\nNot allowed: 2.", encoding="utf-8")
    docs = ingest.load_markdown(path, {**META, "source": "policy.md"})
    assert any("Not allowed: 2." in doc.page_content for doc in docs)
    assert all(doc.metadata["url"] == META["url"] for doc in docs)


def test_regional_refund_clause_keeps_country_scope_on_every_chunk():
    docs = ingest.load_pdf(
        ROOT / "data/raw/core/rules_and_notices.pdf",
        {**META, "source": "rules_and_notices.pdf"},
        structured=False,
    )
    clauses = [doc for doc in docs if "five business days" in doc.page_content]
    assert clauses
    assert all("Colombia Notices" in doc.page_content for doc in clauses)
    assert all("Colombia Notices" in doc.metadata["section"] for doc in clauses)
