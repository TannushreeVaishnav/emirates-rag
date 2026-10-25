from pathlib import Path
from types import SimpleNamespace

import src.ingest as ingest


def test_pdf_cleanup_preserves_repeated_words_negation_and_numeric_limits(monkeypatch):
    text = 'Power\nbanks\nmust\nnot\nexceed\n100\nWh.\nRefund\nrequests\nwithin\n48\nhours\napply\nonly\nwhen\nconditions\nare\nmet.\nPage 1'
    pages = [SimpleNamespace(extract_text=lambda: text) for _ in range(4)]
    monkeypatch.setattr(ingest, 'PdfReader', lambda _: SimpleNamespace(pages=pages))
    chunks = ingest.load_pdf(Path('policy.pdf'), {'source': 'policy.pdf'}, structured=False)
    for chunk in chunks:
        assert 'must not exceed 100 Wh' in chunk.page_content
        assert 'within 48 hours' in chunk.page_content
        assert 'Page 1' not in chunk.page_content
