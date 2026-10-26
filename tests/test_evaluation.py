"""Evaluation math and HTTP caps: fake transports only, never live providers."""
import asyncio
import json
import math
import sys
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.documents import Document

from eval import run_eval as ev
from eval.metrics import (
    answer_metrics, citation_metrics, retrieval_metrics, source_ranking,
    summarize, validate_judge_scores,
)


def doc(source="a.pdf", text="7 kg"):
    return Document(page_content=text, metadata={"source": source, "section": "Allowance"})


def case(identifier="q1", **changes):
    item = {"id": identifier, "category": "direct", "question": "Allowance?", "should_refuse": False,
            "source_doc": ["a.pdf"], "expected_answer": "7 kg", "history": []}
    return {**item, **changes}


def setup_eval(tmp_path, monkeypatch, items, **options):
    dataset = tmp_path / "qa.json"
    dataset.write_text(json.dumps(items), encoding="utf-8")
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "results")
    return SimpleNamespace(**{"qa": str(dataset), "no_table": True, "sleep": 0, **options})


def test_source_metrics_real_math_and_duplicate_rankings():
    docs = [doc("x.pdf"), doc("a.pdf"), doc("a.pdf"), doc("b.pdf"), doc("outside.pdf")]
    assert source_ranking(docs, 4) == ["x.pdf", "a.pdf", "b.pdf"]
    scores = retrieval_metrics(["a.pdf", "b.pdf"], docs, 4)
    assert scores["precision_at_k"] == pytest.approx(2 / 3)
    assert scores["recall_at_k"] == 1
    assert scores["f1_at_k"] == pytest.approx(0.8)
    assert scores["mrr"] == 0.5
    assert scores["average_precision"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert scores["ndcg"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    # Duplicate chunks consume retrieval slots; a later source outside k is NOT rescued.
    assert retrieval_metrics(["a.pdf", "b.pdf"], docs, 3)["recall_at_k"] == 0.5


def test_retrieval_boundaries_and_unavailable_labels():
    assert set(retrieval_metrics(["a.pdf"], [], 4).values()) == {0.0}
    assert set(retrieval_metrics([], [doc()], 4).values()) == {None}
    assert set(retrieval_metrics(None, [doc()], 4).values()) == {None}
    assert set(retrieval_metrics(["a.pdf", "a.pdf"], [doc()], 1).values()) == {1.0}
    assert ev.hit(case(source_doc=[]), []) is None
    assert ev.hit(case(gold_keyword="23 kg"), [doc()]) is False
    with pytest.raises(ValueError, match="positive integer"):
        retrieval_metrics(["a.pdf"], [], 0)


def test_reference_metrics_normalize_and_multiset_overlap():
    scores = answer_metrics("The 7 kg [1].", "7 kg")
    assert set(scores.values()) == {1.0}
    scores = answer_metrics("kg kg kg", "kg bag")
    assert scores["token_precision"] == pytest.approx(1 / 3)
    assert scores["token_recall"] == 0.5
    assert scores["token_f1"] == pytest.approx(0.4)
    assert set(answer_metrics("anything", "  ").values()) == {None}
    assert set(answer_metrics("", "7 kg").values()) == {0.0}


def test_citations_check_references_not_semantic_proof():
    result = {"answer": "7 kg [1] [9]", "docs": [doc()], "context_docs": [doc()],
              "citations": [{"ref": 1, "source": "a.pdf"}]}
    assert citation_metrics(case(), result) == {"citation_reference_validity": 0.5, "citation_source_overlap": 1.0}
    result["context_docs"] = []
    assert citation_metrics(case(), result) == {"citation_reference_validity": 0.0, "citation_source_overlap": 0.0}
    result.update(answer="7 kg", citations=[])
    assert citation_metrics(case(source_doc=[]), result)["citation_source_overlap"] is None


@pytest.mark.parametrize("bad", [None, "0.5", True, -0.1, 1.1, float("nan"), float("inf")])
def test_judge_scores_strictly_bounded(bad):
    output = {"faithfulness": bad, "relevance": 1, "correctness": 1, "citation_support": 1}
    with pytest.raises(ValueError):
        validate_judge_scores(output, True, True)


def test_missing_judge_scores_are_not_fabricated():
    with pytest.raises(ValueError, match="correctness"):
        validate_judge_scores({"faithfulness": 1, "relevance": 1}, True, False)
    scores = validate_judge_scores({"faithfulness": 0.3, "relevance": 0.8}, False, False)
    assert scores["correctness"] is None and scores["citation_support"] is None


def test_judge_uses_full_actual_context_and_handles_invalid_output():
    class Judge:
        def invoke(self, payload):
            self.payload = payload
            return {"faithfulness": 1, "relevance": 1, "correctness": 1}
    judge = Judge()
    result = {"answer": "7 kg", "docs": [doc(text="UNTRUSTED RAW")], "context_docs": []}
    scores = ev.judge_answer(judge, case(), result)
    assert scores["faithfulness"] == 1
    assert "UNTRUSTED RAW" not in judge.payload["context"]
    result["context_docs"] = [doc(text="x" * 7000 + "END EVIDENCE")]
    ev.judge_answer(judge, case(), result)
    assert "END EVIDENCE" in judge.payload["context"]
    judge.invoke = lambda _: {"faithfulness": "1", "relevance": 1}
    scores = ev.judge_answer(judge, case(), result)
    assert scores["faithfulness"] is None and scores["judge_error"]


def test_sync_budget_caps_failed_sends_and_restores_hooks():
    calls = []
    def send(request):
        calls.append(request)
        raise httpx.ConnectError("fake transport outage", request=request)
    original = httpx.Client._send_single_request
    with httpx.Client(transport=httpx.MockTransport(send)) as client:
        with ev.HTTPBudget(2, sleep=0) as budget:
            for _ in range(2):
                with pytest.raises(httpx.ConnectError):
                    client.get("https://api.groq.com/test")
            with pytest.raises(ev.BudgetExhausted):
                client.get("https://api.groq.com/test")
            assert budget.count == len(calls) == 2
    assert httpx.Client._send_single_request is original


def test_sync_redirects_are_separate_dispatches():
    calls = []
    def send(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "/again"})
    with httpx.Client(transport=httpx.MockTransport(send), follow_redirects=True) as client:
        with ev.HTTPBudget(2, sleep=0) as budget:
            with pytest.raises(ev.BudgetExhausted):
                client.get("https://api.groq.com/start")
            assert len(calls) == budget.count == 2


def test_async_budget_never_exceeds_cap_and_restores():
    calls = []
    async def send(request):
        calls.append(request)
        return httpx.Response(200, json={"ok": True})
    original = httpx.AsyncClient._send_single_request
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            with ev.HTTPBudget(2, sleep=0) as budget:
                await client.get("https://api.groq.com/one")
                await client.get("https://api.groq.com/two")
                with pytest.raises(ev.BudgetExhausted):
                    await client.get("https://api.groq.com/three")
                assert budget.count == len(calls) == 2
    asyncio.run(exercise())
    assert httpx.AsyncClient._send_single_request is original


def test_aiohttp_send_guard_is_counted_and_restored(monkeypatch):
    aiohttp = pytest.importorskip("aiohttp")
    calls = []
    async def fake_send(self, connection):
        calls.append(connection)
        return "ok"
    monkeypatch.setattr(aiohttp.ClientRequest, "send", fake_send)
    async def exercise():
        with ev.HTTPBudget(1, sleep=0) as budget:
            request = SimpleNamespace(url="https://generativelanguage.googleapis.com/v1beta/models/fake")
            assert await aiohttp.ClientRequest.send(request, "fake") == "ok"
            with pytest.raises(ev.BudgetExhausted):
                await aiohttp.ClientRequest.send(request, "blocked")
            assert budget.count == 1
    asyncio.run(exercise())
    assert calls == ["fake"]
    assert aiohttp.ClientRequest.send is fake_send


def test_unrelated_http_never_consumes_provider_budget():
    calls = []
    def send(request):
        calls.append(request)
        return httpx.Response(200)
    with httpx.Client(transport=httpx.MockTransport(send)) as client:
        with ev.HTTPBudget(0, sleep=0) as budget:
            client.get("https://huggingface.co/fake-cached-model")
            client.get("https://api.groq.com.attacker.invalid/not-a-provider")
            with pytest.raises(ev.BudgetExhausted):
                client.get("https://generativelanguage.googleapis.com/v1beta/models/fake")
            assert budget.count == 0
    assert len(calls) == 2


def test_provider_requests_are_paced_before_dispatch(monkeypatch):
    clock = [100.0]
    waits = []
    monkeypatch.setattr(ev.time, "monotonic", lambda: clock[0])
    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(ev.time, "sleep", sleep)
    budget = ev.HTTPBudget(2, sleep=6)
    budget.before_send()
    budget.before_send()
    assert waits == [6]
    assert budget.count == 2


def test_evaluate_budget_counts_history_retries_fallback_and_judge(tmp_path, monkeypatch):
    from src.llm_gateway import LLMGateway
    from langchain_core.messages import AIMessage
    calls = []
    def transport(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"ok": True})
    client = httpx.Client(transport=httpx.MockTransport(transport))
    class Model:
        def __init__(self, fail=False):
            self.fail = fail
        def invoke(self, input, config=None):
            client.get("https://api.groq.com/fake")
            if self.fail:
                error = RuntimeError("fake provider failure")
                error.status_code = 500
                raise error
            return AIMessage(content="7 kg [1]")
    gateway = LLMGateway([("gemini", Model(True)), ("groq", Model())], attempts=2, base_delay=0)
    def run(question, history):
        # Every history turn goes through the same retry/fallback gateway.
        for previous in history:
            gateway.invoke(previous)
        gateway.invoke(question)
        return {"answer": "7 kg [1]", "docs": [doc()], "context_docs": [doc()], "citations": [{"ref": 1, "source": "a.pdf"}]}
    monkeypatch.setitem(ev.RUNNERS, "baseline", run)
    class Judge:
        def invoke(self, payload):
            gateway.invoke("judge")
            return {"faithfulness": 1, "relevance": 1, "correctness": 1, "citation_support": 1}
    monkeypatch.setattr(ev, "build_judge", lambda: Judge())
    args = setup_eval(tmp_path, monkeypatch, [case(history=["Earlier?"]), case("q2")], max_api_calls=7)
    try:
        report = ev.evaluate(args)
    finally:
        client.close()
    # History=3 sends, final=3 sends, judge's first failed send=1; no retry dispatch.
    assert len(calls) == report["metadata"]["request_count"] == 7
    assert report["rows"][0]["request_count"] == 7
    assert report["rows"][0]["correctness"] is None
    assert report["rows"][0]["judge_error"]
    assert report["rows"][1]["status"] == "unevaluated"
    assert report["metadata"]["partial_count"] == 1
    assert report["metadata"]["unevaluated_count"] == 1


def test_mid_generation_budget_stops_remaining_cases(tmp_path, monkeypatch):
    sends = []
    client = httpx.Client(transport=httpx.MockTransport(lambda request: (sends.append(request), httpx.Response(200))[1]))
    runs = []
    def run(question, history):
        runs.append(question)
        for _ in range(3):
            client.get("https://api.groq.com/fake")
        pytest.fail("runner must have been interrupted")
    monkeypatch.setitem(ev.RUNNERS, "baseline", run)
    args = setup_eval(tmp_path, monkeypatch, [case(), case("q2")], max_api_calls=1, no_judge=True)
    try:
        report = ev.evaluate(args)
    finally:
        client.close()
    assert len(sends) == len(runs) == 1
    assert report["rows"][0]["status"] == "budget_exhausted"
    assert report["summary"]["decision_accuracy"]["value"] == 0
    assert report["metadata"]["partial_count"] == 1
    assert report["metadata"]["unevaluated_count"] == 1
    assert report["summary"]["success_coverage"]["value"] == 0


def test_service_errors_cannot_improve_accuracy(tmp_path, monkeypatch):
    def fail(question, history):
        raise RuntimeError("fake outage")
    monkeypatch.setitem(ev.RUNNERS, "baseline", fail)
    args = setup_eval(tmp_path, monkeypatch, [case(), case("q2", should_refuse=True, source_doc=[])], no_judge=True)
    report = ev.evaluate(args)
    assert report["summary"]["decision_accuracy"]["value"] == 0
    assert report["summary"]["decision_accuracy"]["n"] == 2
    assert report["summary"]["service_error_rate"]["value"] == 1
    assert report["summary"]["refusal_recall"]["value"] == 0
    assert report["summary"]["refusal_unknown"]["value"] == 2
    assert report["summary"]["exact_match"]["value"] == 0
    assert report["summary"]["faithfulness"]["value"] is None


def test_retrieval_only_zero_http_and_legacy_summary(tmp_path, monkeypatch):
    args = setup_eval(tmp_path, monkeypatch, [case(expected_answer=""), case("q2", source_doc=[])],
                      retrieval_only=True, max_api_calls=0, json_out=str(tmp_path / "legacy.json"))
    monkeypatch.setattr(ev, "retrieve", lambda question, k: [doc()])
    monkeypatch.setattr(ev, "build_judge", lambda: pytest.fail("retrieval-only must never build a judge"))
    monkeypatch.setitem(ev.RUNNERS, "baseline", lambda *a: pytest.fail("must not run generation"))
    original = httpx.Client._send_single_request
    report = ev.evaluate(args)
    assert httpx.Client._send_single_request is original
    assert report["schema_version"] == 1
    assert report["metadata"]["request_count"] == 0
    assert report["summary"]["hit"] == {"value": 1.0, "n": 1, "description": report["summary"]["hit"]["description"]}
    assert report["summary"]["correctness"]["value"] is None
    assert report["summary"]["exact_match"]["value"] is None
    assert report["rows"][1]["recall_at_k"] is None
    legacy = json.loads((tmp_path / "legacy.json").read_text(encoding="utf-8"))
    assert legacy["hit"] == 1 and legacy["hit_n"] == 1 and legacy["misses"] == []
    assert "schema_version" not in legacy and legacy["hit_by_cat"] == {"direct": 1.0}
    versioned = list((tmp_path / "results").glob("*.json"))
    assert len(versioned) == 1
    assert json.loads(versioned[0].read_text(encoding="utf-8"))["schema_version"] == 1


def test_retrieval_mode_enforces_zero_provider_sends(tmp_path, monkeypatch):
    sends = []
    client = httpx.Client(transport=httpx.MockTransport(lambda request: (sends.append(request), httpx.Response(200))[1]))
    def retrieve(question, k):
        client.get("https://api.groq.com/unexpected")
        return [doc()]
    monkeypatch.setattr(ev, "retrieve", retrieve)
    args = setup_eval(tmp_path, monkeypatch, [case(), case("q2")], retrieval_only=True)
    try:
        report = ev.evaluate(args)
    finally:
        client.close()
    assert sends == []
    assert report["metadata"]["request_count"] == 0
    assert report["rows"][0]["status"] == "budget_exhausted"
    assert report["rows"][1]["status"] == "unevaluated"


def test_no_judge_preserves_unavailable_reference_scores(tmp_path, monkeypatch):
    sends = []
    client = httpx.Client(transport=httpx.MockTransport(lambda request: (sends.append(request), httpx.Response(200))[1]))
    def runner(question, history):
        client.get("https://api.groq.com/fake")
        return {"answer": "7 kg [1]", "docs": [doc()], "context_docs": [doc()],
                "citations": [{"ref": 1, "source": "a.pdf"}]}
    monkeypatch.setitem(ev.RUNNERS, "baseline", runner)
    monkeypatch.setattr(ev, "build_judge", lambda: pytest.fail("--no-judge must not construct judge"))
    args = setup_eval(tmp_path, monkeypatch, [case(expected_answer="")], no_judge=True, max_api_calls=1)
    try:
        report = ev.evaluate(args)
    finally:
        client.close()
    assert len(sends) == report["metadata"]["request_count"] == 1
    assert report["rows"][0]["status"] == "success"
    assert report["rows"][0]["decision_correct"] is True
    for key in ("correctness", "faithfulness", "token_f1", "exact_match"):
        assert report["summary"][key]["value"] is None


def test_empty_dataset_saves_report_and_csv(tmp_path, monkeypatch):
    args = setup_eval(tmp_path, monkeypatch, [], retrieval_only=True)
    report = ev.evaluate(args)
    assert report["rows"] == [] and report["categories"] == {}
    assert report["summary"]["hit"]["value"] is None
    assert report["summary"]["request_count"]["value"] == 0
    assert len(list((tmp_path / "results").glob("*.csv"))) == 1


def test_zero_budget_never_attempts_a_runner(tmp_path, monkeypatch):
    args = setup_eval(tmp_path, monkeypatch, [case()], max_api_calls=0)
    monkeypatch.setitem(ev.RUNNERS, "baseline", lambda *a: pytest.fail("no budget"))
    report = ev.evaluate(args)
    assert report["metadata"]["request_count"] == report["metadata"]["attempted_count"] == 0
    assert report["metadata"]["unevaluated_count"] == 1


def test_refusal_confusion_and_latency_boundaries():
    rows = []
    for i, (gold, refused) in enumerate(((True, True), (True, False), (False, True), (False, False))):
        row = ev._row(case(str(i), should_refuse=gold))
        row.update(status="success", evaluated=True, refused=refused, decision_correct=gold == refused,
                   answer="refusal" if refused else "7 kg", latency=i + 1, request_count=1)
        rows.append(row)
    scores = summarize(rows, mode="graph")
    for key in ("refusal_tp", "refusal_fp", "refusal_fn", "refusal_tn"):
        assert scores[key]["value"] == 1
    for key in ("refusal_precision", "refusal_recall", "refusal_f1", "decision_accuracy", "false_refusal_rate", "answer_coverage"):
        assert scores[key]["value"] == 0.5
    assert scores["p50"]["value"] == 2
    assert scores["p95"]["value"] == scores["p99"]["value"] == 4
    assert ev.pct([], 99) is None and ev.pct([9], 99) == 9


@pytest.mark.parametrize("change", [{"max_api_calls": -1}, {"max_api_calls": 1.5}, {"sleep": -1}, {"sleep": float("nan")}, {"sleep": float("inf")}, {"limit": 0}, {"k": 0}, {"config": "unknown"}, {"label": "../escape"}])
def test_bad_parameters_fail_clearly(tmp_path, monkeypatch, change):
    args = setup_eval(tmp_path, monkeypatch, [], **change)
    with pytest.raises(ValueError):
        ev.evaluate(args)


def test_bad_dataset_fails_clearly(tmp_path, monkeypatch):
    args = setup_eval(tmp_path, monkeypatch, [case(should_refuse="false")], retrieval_only=True)
    with pytest.raises(ValueError, match="should_refuse"):
        ev.evaluate(args)


def test_graph_history_failure_is_not_hidden(monkeypatch):
    calls = []
    def chat(question, thread):
        calls.append(question)
        return {"error": "history failed"}
    monkeypatch.setitem(sys.modules, "src.graph", SimpleNamespace(chat=chat))
    with pytest.raises(RuntimeError, match="history failed"):
        ev.run_graph("Final?", ["Earlier?"])
    assert calls == ["Earlier?"]


def test_retrieval_reports_preserve_references_without_scoring_answer_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "retrieve", lambda *args, **kwargs: [doc()])
    items = [case(), case("refuse", should_refuse=True, expected_answer="", source_doc=[])]
    report = ev.evaluate(setup_eval(tmp_path, monkeypatch, items, retrieval_only=True, max_api_calls=0))
    for key in ("refusal_recall", "false_refusal_rate", "answer_coverage", "scored_coverage", "refusal_unknown"):
        assert report["summary"][key]["value"] is None
        assert report["summary"][key]["n"] == 0
    assert report["rows"][0]["expected_answer"] == "7 kg"
    assert report["rows"][0]["source_doc"] == ["a.pdf"]
