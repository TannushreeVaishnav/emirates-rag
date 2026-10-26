"""Offline report-viewer regressions; no index, provider or production fixtures."""

import builtins
import copy
import csv
import io
import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from eval.report_view import (
    RESULTS_ENV,
    ReportError,
    case_scored,
    load_report,
    metric_group,
    metric_value,
    rows_csv,
    visible_metrics,
)


PAGE = Path(__file__).resolve().parent.parent / "pages" / "1_Evaluation.py"


def metric(value, n, description="Measured eligible cases only"):
    return {"value": value, "n": n, "description": description}


@pytest.fixture
def report():
    return {
        "schema_version": 1,
        "metadata": {
            "mode": "baseline", "k": 4, "dataset": "eval/test_questions.json",
            "created_at": "2026-10-26T12:00:00+00:00", "provider": {"name": "recorded-provider"},
            "judge": {"enabled": False}, "request_count": 2,
        },
        "summary": {
            "hit": metric(1.0, 1, "Any expected source found; source labels may be alternatives"),
            "recall_at_k": metric(0.5, 1, "Source-label recall proxy; not answer accuracy"),
            "exact_match": metric(None, 0, "No reference answers in this test report"),
            "faithfulness": metric(None, 0, "Judging disabled"),
            "decision_accuracy": metric(1.0, 2), "errors": metric(1, 3),
            "request_count": metric(2, 3),
        },
        "categories": {
            "direct": {"hit": metric(1.0, 1), "exact_match": metric(None, 0), "errors": metric(1, 2)},
            "refusal": {"decision_accuracy": metric(1.0, 1), "errors": metric(0, 1)},
        },
        "rows": [
            {"id": "d01", "category": "direct", "question": "What is the cabin allowance?",
             "answer": "One cabin bag up to 7 kg [1].", "expected_answer": "", "hit": True,
             "recall_at_k": 0.5, "exact_match": None, "decision_correct": True,
             "judge_reason": "", "error": "", "judge_error": "", "status": "success"},
            {"id": "d02", "category": "direct", "question": "When does check-in open?",
             "answer": "", "expected_answer": "", "hit": None, "recall_at_k": None,
             "exact_match": None, "decision_correct": None, "judge_reason": "",
             "error": "Recorded provider failure", "judge_error": "", "status": "error"},
            {"id": "r01", "category": "refusal", "question": "Give unrelated medical advice",
             "answer": "I can help with travel policies.", "expected_answer": "", "hit": None,
             "exact_match": None, "decision_correct": True, "judge_reason": "Not judged",
             "error": "", "judge_error": "", "status": "success"},
        ],
    }


@pytest.fixture
def isolated_reports(tmp_path, monkeypatch):
    monkeypatch.setenv(RESULTS_ENV, str(tmp_path))
    original_import = builtins.__import__

    def forbid_backend(name, *args, **kwargs):
        if name == "src" or name.startswith("src."):
            raise AssertionError(f"Dashboard imported backend module: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_backend)
    return tmp_path


def save_report(directory, report, name="baseline_20261026_120000.json"):
    path = directory / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def widget(at, label):
    return next(element for element in at.selectbox if element.label == label)


def case_table(at):
    return next(element.value for element in at.dataframe if "Case" in element.value.columns)


def test_empty_directory_has_safe_offline_instructions(isolated_reports):
    at = AppTest.from_file(str(PAGE)).run()
    assert not at.exception
    assert any("No readable evaluation reports" in element.value for element in at.info)
    instructions = "\n".join(element.value for element in at.code)
    assert "--retrieval-only --no-judge --max-api-calls 0" in instructions
    assert "streamlit run pages/1_Evaluation.py" in instructions
    assert not at.button


def test_csv_only_folder_explains_migration(isolated_reports):
    (isolated_reports / "legacy.csv").write_text("id,hit\nd01,True\n", encoding="utf-8")
    at = AppTest.from_file(str(PAGE)).run()
    assert not at.exception
    assert any("legacy CSV" in element.value for element in at.info)
    assert any("denominators" in element.value for element in at.info)


@pytest.mark.parametrize("contents", ["{bad json", "[]", '{"schema_version": 99}', '{"schema_version": 1, "metadata": []}'])
def test_malformed_report_is_visible_without_page_crash(isolated_reports, contents):
    (isolated_reports / "bad.json").write_text(contents, encoding="utf-8")
    at = AppTest.from_file(str(PAGE)).run()
    assert not at.exception
    assert any("could not be displayed safely" in element.value for element in at.warning)
    assert any("bad.json" in element.value for element in at.text)


def test_realistic_report_renders_nulls_samples_filters_and_downloads(isolated_reports, report):
    save_report(isolated_reports, report)
    at = AppTest.from_file(str(PAGE)).run()
    assert not at.exception
    captions = "\n".join(element.value for element in at.caption)
    assert "Mode: baseline" in captions
    assert "eval/test_questions.json" in captions
    assert "n = 0 measured samples" in captions
    assert any(element.value == "Not measured" for element in at.metric)
    assert set(case_table(at)["Case"]) == {"d01", "d02", "r01"}
    assert len(at.get("download_button")) == 2
    assert any("Per-category metrics" == element.value for element in at.subheader)

    widget(at, "Category").set_value("direct").run()
    assert not at.exception
    assert set(case_table(at)["Case"]) == {"d01", "d02"}
    widget(at, "Case status").set_value("Failed (execution or judge)").run()
    assert not at.exception
    assert list(case_table(at)["Case"]) == ["d02"]
    assert any("Recorded provider failure" == element.value for element in at.text)

    widget(at, "Case status").set_value("Scored").run()
    assert list(case_table(at)["Case"]) == ["d01"]
    widget(at, "Case status").set_value("Unscored").run()
    assert list(case_table(at)["Case"]) == ["d02"]
    widget(at, "Category").set_value("refusal").run()
    assert not at.exception
    assert any("No cases match" in element.value for element in at.info)


def test_latest_saved_report_selected_and_runs_can_be_compared(isolated_reports, report):
    old = save_report(isolated_reports, report, "z_old.json")
    latest = save_report(isolated_reports, report, "a_latest.json")
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))
    os.utime(latest, ns=(2_000_000_000, 2_000_000_000))
    (isolated_reports / "malformed.json").write_text("null", encoding="utf-8")
    at = AppTest.from_file(str(PAGE)).run()
    assert not at.exception
    assert widget(at, "Saved report (newest saved first)").value == "a_latest.json"
    assert any(element.label == "Compare saved runs" for element in at.expander)


def test_retrieval_snapshot_never_displays_answer_accuracy(isolated_reports, report):
    report["metadata"]["mode"] = "retrieval"
    # Even inconsistent imported snapshots must not imply generated-answer measurements.
    report["summary"]["accuracy"] = metric(1.0, 1)
    save_report(isolated_reports, report)
    at = AppTest.from_file(str(PAGE)).run()
    assert not at.exception
    assert any("Offline retrieval only" in element.value for element in at.info)
    assert {element.label for element in at.get("tab")} == {"Retrieval", "Answer", "Refusal", "Citation", "Operational"}
    assert any("Not measured · n = 0" in element.value for element in at.caption)
    labels = [element.label.lower() for element in at.metric]
    assert not any("accuracy" in label or "exact match" in label for label in labels)
    for element in at.dataframe:
        if "Metric" in element.value.columns:
            assert not element.value["Metric"].str.contains("accuracy|exact match", case=False).any()


def test_partial_budget_run_keeps_coverage_prominent(isolated_reports, report):
    report["metadata"].update({
        "selected_count": 41, "completed_count": 2, "partial_count": 1,
        "unevaluated_count": 38, "budget_exhausted": True,
    })
    save_report(isolated_reports, report)
    at = AppTest.from_file(str(PAGE)).run()
    assert not at.exception
    captions = "\n".join(element.value for element in at.caption)
    assert "2 completed · 41 selected · 1 partial · 38 unevaluated" in captions
    assert any("partial run" in element.value for element in at.warning)


@pytest.mark.parametrize("invalid", [
    {"value": float("nan"), "n": 1, "description": "bad"},
    {"value": float("inf"), "n": 1, "description": "bad"},
    {"value": "0.5", "n": 1, "description": "bad"},
    {"value": True, "n": 1, "description": "bad"},
    {"value": 0.5, "n": True, "description": "bad"},
    {"value": None, "n": -1, "description": "bad"},
    {"value": None, "n": 0},
])
def test_metric_schema_rejects_unsafe_or_ambiguous_values(tmp_path, report, invalid):
    report["summary"]["hit"] = invalid
    path = save_report(tmp_path, report)
    with pytest.raises(ReportError):
        load_report(path)


def test_null_and_real_zero_remain_distinct(tmp_path, report):
    report["summary"]["request_count"] = metric(0, 0)
    parsed = load_report(save_report(tmp_path, report))
    assert metric_value(parsed["summary"]["request_count"]) == "0"
    assert metric_value(parsed["summary"]["exact_match"]) == "Not measured"


def test_csv_preserves_case_fields_and_nested_values(report):
    rows = copy.deepcopy(report["rows"])
    rows[0]["sources"] = ["cabin_baggage.pdf", "policy.pdf"]
    rows[0]["answer"] = 'One bag, up to 7 kg.\nUse "cabin" rules.'
    exported = list(csv.DictReader(io.StringIO(rows_csv(rows))))
    assert exported[0]["answer"] == rows[0]["answer"]
    assert json.loads(exported[0]["sources"]) == rows[0]["sources"]
    assert exported[1]["error"] == "Recorded provider failure"


@pytest.mark.parametrize("name", ["precision_at_k", "recall_at_k", "f1_at_k", "average_precision", "mrr", "ndcg"])
def test_source_proxies_are_grouped_as_retrieval(name):
    assert metric_group(name) == "Retrieval"
    assert name in visible_metrics({name: metric(0.0, 1)}, "retrieval")
    assert case_scored({name: 0.0})
