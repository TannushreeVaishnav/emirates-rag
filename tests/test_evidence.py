import json
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

import src.graph as g
from src.llm_gateway import AllProvidersFailed
from src.prompts import NO_ANSWER_MSG, SERVICE_MSG
from src.rag_chain import build_citations, citation_markdown, normalize_citations


def document():
    return Document(page_content="Economy cabin baggage allowance is 7 kg.", metadata={
        "source": "cabin_baggage.pdf", "page": 2,
        "url": "https://www.emirates.com/baggage.pdf", "section": "Economy",
    })


@pytest.mark.parametrize("stage", ["router", "grader", "check"])
def test_provider_outage_is_not_missing_evidence(monkeypatch, stage):
    def respond(prompt):
        text = prompt.to_string()
        if "You classify" in text:
            if stage == "router":
                raise AllProvidersFailed("429")
            return json.dumps({"route": "policy", "is_followup": False})
        if "You filter" in text:
            if stage == "grader":
                raise AllProvidersFailed("429")
            return '{"relevant": [1]}'
        if "You verify" in text:
            raise AllProvidersFailed("429")
        return "Economy cabin baggage is 7 kg [1]."

    monkeypatch.setattr(g, "llm", lambda: RunnableLambda(respond))
    monkeypatch.setattr(g, "fast_llm", lambda: RunnableLambda(respond))
    monkeypatch.setattr(g, "retrieve", lambda _: [document()])
    monkeypatch.setattr(g, "log_request", lambda _: None)
    monkeypatch.setattr(g.settings, "use_grader", True)
    monkeypatch.setattr(g.settings, "use_output_check", True)
    result = g.chat("What is Economy cabin baggage?", uuid.uuid4().hex)
    assert result["answer"] == SERVICE_MSG
    assert result["error"] == "llm_unavailable"
    assert result["outcome"] != "no_evidence"
    assert result["citations"] == []


@pytest.mark.parametrize("answer", ["7 kg", "7 kg [0]", "7 kg [2]"])
def test_missing_and_out_of_range_references_fail_before_llm(monkeypatch, answer):
    monkeypatch.setattr(g, "llm", lambda: pytest.fail("invalid citations must not call a provider"))
    result = g.output_check_node({"answer": answer, "context_docs": [document()]})
    assert result["supported"] is False


def test_malformed_support_assessment_cannot_publish_answer(monkeypatch):
    monkeypatch.setattr(g, "llm", lambda: RunnableLambda(lambda _: '{"supported": "false"}'))
    monkeypatch.setattr(g.settings, "use_output_check", True)
    with pytest.raises(g.EvidenceCheckFailed):
        g.output_check_node({"answer": "7 kg [1]", "context_docs": [document()]})


def test_numeric_decorative_citations_link_to_original_pdf_page():
    answer = normalize_citations("Economy allowance is 7 kg【1†】.")
    citations = build_citations([document()], answer)
    assert citations[0]["source"] == "cabin_baggage.pdf"
    assert citation_markdown(answer, citations) == (
        "Economy allowance is 7 kg[[1]](https://www.emirates.com/baggage.pdf#page=2)."
    )


def test_non_web_source_cannot_become_clickable():
    assert citation_markdown("Policy [1]", [{"ref": 1, "url": "javascript:alert(1)"}]) == "Policy [1]"


def test_standalone_question_does_not_inherit_router_rewrite(monkeypatch):
    question = "What is the cabin baggage allowance for Economy?"
    monkeypatch.setattr(g, "fast_llm", lambda: RunnableLambda(lambda _: json.dumps({
        "route": "policy", "is_followup": False,
        "standalone_question": "What is the refund for Economy?",
    })))
    state = {"question": question, "messages": [HumanMessage(content="refund?"), HumanMessage(content=question)]}
    assert g.router_node(state)["question"] == question


def test_followup_uses_resolved_question(monkeypatch):
    resolved = "What is the cabin baggage allowance for Business Class?"
    monkeypatch.setattr(g, "fast_llm", lambda: RunnableLambda(lambda _: json.dumps({
        "route": "policy", "is_followup": True, "standalone_question": resolved,
    })))
    assert g.router_node({"question": "What about Business Class?", "messages": []})["question"] == resolved


def test_grader_sees_evidence_after_seven_hundred_characters(monkeypatch):
    doc = document()
    doc.page_content = "Policy background. " * 50 + "Economy cabin baggage allowance is 7 kg."
    def grade(prompt):
        return json.dumps({"relevant": [1] if "allowance is 7 kg" in prompt.to_string() else []})
    monkeypatch.setattr(g, "fast_llm", lambda: RunnableLambda(grade))
    monkeypatch.setattr(g.settings, "use_grader", True)
    assert g.grade_node({"question": "Economy cabin baggage?", "docs": [doc]})["context_docs"] == [doc]
