from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_app_answers_a_question(monkeypatch):
    if not (ROOT / "index" / "index.faiss").exists():
        pytest.skip("build the index first: python -m src.ingest")

    import src.graph as g

    def fake_chat(message, thread_id="default"):
        return {
            "answer": "According to cabin_baggage.pdf, Economy cabin baggage is up to 7 kg [1].",
            "citations": [
                {
                    "ref": 1,
                    "source": "cabin_baggage.pdf",
                    "section": "Cabin baggage",
                    "page": None,
                    "url": "https://example.com",
                    "date_collected": "2026-10-25",
                    "text": "Economy cabin baggage is one piece up to 7 kg.",
                }
            ],
            "route": "policy",
            "provider_used": "gemini",
            "latency": 1.2,
            "error": None,
            "masked_question": message,
            "context_docs": [],
            "refused": False,
            "citations": [
                {
                    "ref": 1,
                    "source": "cabin_baggage.pdf",
                    "section": "Cabin baggage",
                    "page": None,
                    "url": "https://example.com",
                    "date_collected": "2026-10-25",
                    "text": "Economy cabin baggage is one piece up to 7 kg.",
                }
            ],
        }

    monkeypatch.setattr(g, "chat", fake_chat)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception

    at.chat_input[0].set_value("What is the cabin baggage allowance?").run()
    assert not at.exception
    assert any("7 kg" in m.value for m in at.markdown)
    assert any("[[1]](https://example.com)" in m.value for m in at.markdown)
