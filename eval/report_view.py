"""Read-only, provider-free helpers for the evaluation dashboard."""

import csv
import io
import json
import math
import os
from pathlib import Path


RESULTS_ENV = "EMIRATES_EVAL_RESULTS_DIR"
MAX_REPORT_BYTES = 20 * 1024 * 1024


class ReportError(ValueError):
    """A report cannot safely be displayed as schema version 1."""


def results_directory() -> Path:
    """Allow isolated local report collections without importing app settings."""
    return Path(os.environ.get(RESULTS_ENV, Path(__file__).resolve().parent / "results"))


def report_paths(directory: Path) -> list[Path]:
    """Newest saved reports first; never follow report symlinks."""
    if not directory.exists():
        return []
    try:
        return sorted(
            (p for p in directory.glob("*.json") if p.is_file() and not p.is_symlink()),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
            reverse=True,
        )
    except OSError as exc:
        raise ReportError(f"Cannot list the reports directory: {exc}") from exc


def _reject_constant(value: str):
    raise ReportError(f"Non-finite JSON number is not supported: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ReportError("Non-finite JSON numbers are not supported.")
    return number


def _metric_mapping(value, location: str) -> None:
    if not isinstance(value, dict):
        raise ReportError(f"{location} must be a metric mapping.")
    for name, metric in value.items():
        if not isinstance(name, str) or not isinstance(metric, dict):
            raise ReportError(f"{location} must contain named metric objects.")
        score = metric.get("value")
        if "value" not in metric or (
            score is not None
            and (type(score) not in (int, float) or (isinstance(score, float) and not math.isfinite(score)))
        ):
            raise ReportError(f"{location}.{name}.value must be a finite number or null.")
        n = metric.get("n")
        if type(n) is not int or n < 0:
            raise ReportError(f"{location}.{name}.n must be a non-negative integer.")
        if not isinstance(metric.get("description"), str):
            raise ReportError(f"{location}.{name}.description must be text.")


def load_report(path: Path) -> dict:
    """Parse JSON only, validating shapes before any UI code touches them."""
    try:
        if path.is_symlink():
            raise ReportError("Report symlinks are not supported.")
        with path.open("rb") as stream:
            raw = stream.read(MAX_REPORT_BYTES + 1)
        if len(raw) > MAX_REPORT_BYTES:
            raise ReportError("Report exceeds the 20 MiB display limit.")
        report = json.loads(
            raw.decode("utf-8-sig"), parse_constant=_reject_constant, parse_float=_finite_float
        )
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ReportError(f"Cannot read JSON report: {exc}") from exc
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int:
        raise ReportError("Expected a versioned report object with schema_version=1.")
    if report["schema_version"] != 1:
        raise ReportError(f"Unsupported report schema version: {report['schema_version']}.")
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise ReportError("metadata must be an object.")
    if metadata.get("mode") not in ("retrieval", "baseline", "graph"):
        raise ReportError("metadata.mode must be retrieval, baseline or graph.")
    if not isinstance(metadata.get("dataset"), str) or not isinstance(metadata.get("created_at"), str):
        raise ReportError("metadata.dataset and metadata.created_at must be text.")
    if type(metadata.get("k")) is not int or metadata["k"] < 1:
        raise ReportError("metadata.k must be a positive integer.")
    _metric_mapping(report.get("summary"), "summary")
    categories = report.get("categories")
    if not isinstance(categories, dict):
        raise ReportError("categories must be an object.")
    for category, metrics in categories.items():
        _metric_mapping(metrics, f"categories.{category}")
    rows = report.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ReportError("rows must be a list of per-case objects.")
    for row in rows:
        if not isinstance(row.get("category"), str) or not isinstance(row.get("question"), str):
            raise ReportError("Every row needs a text category and question.")
        for field in ("answer", "error", "judge_reason", "judge_error"):
            if field in row and row[field] is not None and not isinstance(row[field], str):
                raise ReportError(f"Row {field} must be text or null.")
    return report


def metric_group(name: str) -> str:
    key = name.lower()
    if key in {
        "hit", "mrr", "map", "ndcg", "recall", "precision", "precision_at_k",
        "recall_at_k", "f1_at_k", "average_precision",
    } or key.startswith(("source_", "retrieval_")):
        return "Retrieval"
    if key in {"answer_coverage", "scored_coverage", "success_coverage"}:
        return "Operational"
    if "refus" in key or key in {"false_ref", "decision_accuracy", "decision_acc"}:
        return "Refusal"
    if "cit" in key:
        return "Citation"
    if key in {"faith", "rel", "em", "f1"} or "accuracy" in key or any(
        token in key for token in ("answer", "faith", "relevance", "exact_match", "token_", "correctness")
    ):
        return "Answer"
    return "Operational"


def visible_metrics(metrics: dict, mode: str) -> dict:
    """Retrieval snapshots cannot make claims about generated answers."""
    return {
        name: metric for name, metric in metrics.items()
        if mode != "retrieval" or metric_group(name) in ("Retrieval", "Operational")
    }


def metric_label(name: str) -> str:
    labels = {
        "hit": "Source hit proxy", "source_hit_rate": "Source hit proxy",
        "precision_at_k": "Source precision@k proxy", "recall_at_k": "Source recall@k proxy",
        "f1_at_k": "Source F1@k proxy", "mrr": "Source reciprocal rank proxy",
        "average_precision": "Source average precision proxy", "ndcg": "Source nDCG proxy",
        "exact_match": "Answer exact match (reference)", "token_f1": "Answer token F1 (reference)",
        "decision_accuracy": "Answer/refuse decision accuracy",
        "judge_correctness": "Answer correctness (judge)",
        "correctness": "Answer correctness (judge)", "citation_support": "Citation support (judge)",
        "faithfulness": "Faithfulness (judge)", "relevance": "Relevance (judge)",
        "citation_source_overlap": "Citation source overlap proxy",
        "p99": "Answer latency p99 (s)",
        "faith": "Faithfulness (judge)", "rel": "Relevance (judge)",
        "refusal_acc": "Refusal decision accuracy", "false_ref": "False refusal rate",
        "cite": "Citation source-match proxy", "p50": "Answer latency p50 (s)",
        "p95": "Answer latency p95 (s)", "llm_calls": "LLM calls per case",
    }
    return labels.get(name, name.replace("_", " ").capitalize())


def metric_value(metric: dict) -> str:
    value = metric["value"]
    if value is None:
        return "Not measured"
    return f"{value:,.3f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"


def case_failed(row: dict) -> bool:
    """Failure means an execution/judge error, not an invented score threshold."""
    return bool(row.get("error") or row.get("judge_error") or row.get("status") in ("error", "budget_exhausted"))


def case_scored(row: dict) -> bool:
    fields = (
        "hit", "source_hit", "source_hit_rate", "source_recall", "source_recall_at_k",
        "source_precision", "source_precision_at_k", "source_mrr", "source_mrr_at_k",
        "decision_correct", "citation_ok", "faithfulness", "relevance", "exact_match",
        "answer_exact_match", "token_f1", "answer_token_f1", "answer_correctness",
        "precision_at_k", "recall_at_k", "f1_at_k", "mrr", "average_precision", "ndcg",
        "token_precision", "token_recall", "correctness", "citation_support",
        "citation_reference_validity", "citation_source_overlap",
    )
    return any(row.get(field) is not None for field in fields)


def filter_rows(rows: list[dict], category: str | None, status: str) -> list[dict]:
    return [
        row for row in rows
        if (category is None or row["category"] == category)
        and (
            status == "All cases"
            or (status == "Failed (execution or judge)" and case_failed(row))
            or (status == "Scored" and case_scored(row))
            or (status == "Unscored" and not case_scored(row))
        )
    ]


def rows_csv(rows: list[dict]) -> str:
    """Export every recorded field, including nested data as JSON cells."""
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        return ""
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        })
    return stream.getvalue()
