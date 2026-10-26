"""Offline metrics. Source labels are incomplete relevance proxies, not chunk truth.

Rank the first k retrieved chunks, then collapse repeated source names, preserving
first occurrence. Precision uses that unique-source ranking's length; recall/AP
and ideal DCG use the supplied source labels (which may be alternatives, not an
exhaustive relevance set). No source labels means unavailable, never a zero.
"""
import math
import re
import string
from collections import Counter

RETRIEVAL_KEYS = ("precision_at_k", "recall_at_k", "f1_at_k", "mrr", "average_precision", "ndcg")
ANSWER_KEYS = ("exact_match", "token_precision", "token_recall", "token_f1")
JUDGE_KEYS = ("faithfulness", "relevance", "correctness", "citation_support")


def source_ranking(docs, k):
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    return list(dict.fromkeys(
        d.metadata.get("source") for d in docs[:k] if d.metadata.get("source")
    ))


def retrieval_metrics(labels, docs, k):
    ranking = source_ranking(docs, k)
    gold = set(labels or [])
    if not gold:
        return dict.fromkeys(RETRIEVAL_KEYS)
    relevant = [int(source in gold) for source in ranking]
    found = sum(relevant)
    precision = found / len(ranking) if ranking else 0.0
    recall = found / len(gold)
    ap = sum(sum(relevant[:i]) / i for i, rel in enumerate(relevant, 1) if rel) / len(gold)
    dcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(relevant, 1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return {
        "precision_at_k": precision, "recall_at_k": recall,
        "f1_at_k": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mrr": next((1 / i for i, rel in enumerate(relevant, 1) if rel), 0.0),
        "average_precision": ap, "ndcg": dcg / ideal,
    }


def normalize_answer(text):
    text = re.sub(r"\[\d+\]|【\d+(?:†[^】]*)?】", "", text.lower())
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(answer, reference):
    if not isinstance(reference, str) or not reference.strip():
        return dict.fromkeys(ANSWER_KEYS)
    prediction, target = normalize_answer(answer), normalize_answer(reference)
    tokens, gold = prediction.split(), target.split()
    common = sum((Counter(tokens) & Counter(gold)).values())
    precision = common / len(tokens) if tokens else float(not gold)
    recall = common / len(gold) if gold else float(not tokens)
    return {
        "exact_match": float(prediction == target), "token_precision": precision,
        "token_recall": recall,
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def context_docs(res):
    """An explicitly empty graded context MUST NOT fall back to raw retrieval."""
    return res.get("context_docs", []) if "context_docs" in res else res.get("docs", [])


def citation_metrics(item, res):
    answer = res.get("answer", "")
    normalized = re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", answer)
    refs = set(int(n) for n in re.findall(r"\[(\d+)\]", normalized))
    docs = context_docs(res)
    structured = {}
    for citation in res.get("citations", []):
        ref = citation.get("ref")
        if isinstance(ref, int) and not isinstance(ref, bool):
            structured[ref] = citation.get("source")
    valid = {n for n in refs if 1 <= n <= len(docs)
             and structured.get(n) == docs[n - 1].metadata.get("source")
             and structured.get(n)}
    labels = set(item.get("source_doc") or [])
    cited = {structured[n] for n in valid}
    return {
        "citation_reference_validity": len(valid) / len(refs) if refs else 0.0,
        "citation_source_overlap": float(bool(cited & labels)) if labels else None,
    }


def validate_judge_scores(out, has_reference, has_citations):
    if not isinstance(out, dict):
        raise ValueError("judge output must be an object")
    result = {}
    for key in JUDGE_KEYS:
        applicable = key not in {"correctness", "citation_support"} or (
            has_reference if key == "correctness" else has_citations
        )
        if not applicable:
            result[key] = None
            continue
        score = out.get(key)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"judge {key} must be a finite number between 0 and 1")
        result[key] = float(score)
    return result


DESCRIPTIONS = {
    "hit": "Source-label and optional gold-keyword hit among top-k chunks; relevance proxy only.",
    "precision_at_k": "Labeled-source fraction of unique sources in the first k chunks; incomplete-label proxy.",
    "recall_at_k": "Fraction of supplied source labels retrieved in top-k chunks; labels may be alternatives.",
    "f1_at_k": "Harmonic mean of source-label precision and recall proxies.",
    "mrr": "Reciprocal first labeled-source rank after first-occurrence deduplication within top-k chunks.",
    "average_precision": "Sum of precision at labeled-source ranks divided by supplied label count; truncated source AP proxy.",
    "ndcg": "Binary source-label DCG / ideal DCG for min(k,label count); source relevance proxy.",
    "exact_match": "Normalized exact match against nonblank reference; strips numeric citations, punctuation and English articles.",
    "token_precision": "Multiset normalized token overlap / predicted tokens; requires reference.",
    "token_recall": "Multiset normalized token overlap / reference tokens; requires reference.",
    "token_f1": "Harmonic mean of normalized token precision and recall; requires reference.",
    "faithfulness": "LLM judge support of all answer claims using full actual generation context, not correctness.",
    "relevance": "LLM judge relevance of answer to question.",
    "correctness": "LLM judge correctness relative to a nonblank supplied reference only.",
    "citation_support": "LLM judge semantic support of cited claims by their numbered passages, not deterministic proof.",
    "citation_reference_validity": "Fraction of distinct inline references resolving to actual generation passages and matching structured citation sources; no refs scores zero.",
    "citation_source_overlap": "Any valid cited source overlaps supplied source labels; NOT semantic proof.",
    "decision_accuracy": "Correct answer/refuse decisions / attempted cases; service and interrupted generation failures score zero.",
    "false_refusal_rate": "False refusals / attempted answerable cases; read alongside error and success coverage.",
    "refusal_precision": "TP / (TP+FP) for observed successful refusal decisions; undefined denominator is unavailable.",
    "refusal_recall": "TP / all attempted should-refuse cases, including failed decisions.",
    "refusal_f1": "Harmonic mean of refusal precision and recall; unavailable if either denominator is undefined.",
    "service_error_rate": "Cases with runner/service errors / attempted cases; budget interruption is reported separately.",
    "answer_coverage": "Successful non-refused nonempty answers / attempted answerable cases.",
    "success_coverage": "Completed successful runner cases / all selected cases, including unevaluated cases.",
    "scored_coverage": "Cases with valid faithfulness and relevance judge scores / selected answerable cases.",
    "evaluation_coverage": "Attempted cases / all selected cases.",
    "p50": "Nearest-rank successful runner latency median in seconds (includes history replay, excludes judge).",
    "p95": "Nearest-rank successful runner latency 95th percentile in seconds.",
    "p99": "Nearest-rank successful runner latency 99th percentile in seconds.",
    "request_count": "Observed provider HTTP dispatch attempts, including failed sends, history, retries and judge; unrelated HTTP excluded.",
    "llm_calls": "Mean observed provider HTTP dispatch attempts per attempted case; includes judge/history/retries.",
    "errors": "Number of runner/service errors (budget interruption reported separately).",
}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summarize(rows, *, mode):
    summary = {}

    def put(key, value, n, description=None):
        summary[key] = {"value": value, "n": n, "description": description or DESCRIPTIONS[key]}

    for key in ("hit", *RETRIEVAL_KEYS, *ANSWER_KEYS, *JUDGE_KEYS,
                "citation_reference_validity", "citation_source_overlap", "decision_accuracy"):
        field = "decision_correct" if key == "decision_accuracy" else key
        values = [r.get(field) for r in rows if r.get(field) is not None]
        put(key, mean(values), len(values))
    attempted = [r for r in rows if r.get("evaluated")]
    decisions = [r for r in attempted if r.get("refused") is not None]
    answerable = [r for r in attempted if not r["should_refuse"]]
    should_refuse = [r for r in attempted if r["should_refuse"]]
    tp = sum(r["should_refuse"] and r["refused"] for r in decisions)
    fp = sum(not r["should_refuse"] and r["refused"] for r in decisions)
    fn = sum(r["should_refuse"] and not r["refused"] for r in decisions)
    tn = sum(not r["should_refuse"] and not r["refused"] for r in decisions)
    for key, value in (("refusal_tp", tp), ("refusal_fp", fp), ("refusal_fn", fn), ("refusal_tn", tn)):
        put(key, value, len(decisions), "Observed successful decision confusion count; refusal is the positive class.")
    put("refusal_unknown", len(attempted) - len(decisions), len(attempted), "Attempted cases without an observable refusal decision.")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / len(should_refuse) if should_refuse else None
    put("refusal_precision", precision, tp + fp)
    put("refusal_recall", recall, len(should_refuse))
    f1 = None if precision is None or recall is None else (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    put("refusal_f1", f1, len(attempted))
    put("false_refusal_rate", fp / len(answerable) if answerable else None, len(answerable))
    errors = sum(r.get("status") == "error" for r in attempted)
    put("errors", errors, len(attempted))
    put("service_error_rate", errors / len(attempted) if attempted else None, len(attempted))
    put("answer_coverage", sum(bool(r.get("answer")) and r.get("refused") is False for r in answerable) / len(answerable) if answerable else None, len(answerable))
    put("success_coverage", sum(r.get("status") == "success" for r in rows) / len(rows) if rows else None, len(rows))
    eligible = [r for r in rows if not r["should_refuse"]]
    put("scored_coverage", sum(r.get("faithfulness") is not None and r.get("relevance") is not None for r in eligible) / len(eligible) if eligible else None, len(eligible))
    put("evaluation_coverage", len(attempted) / len(rows) if rows else None, len(rows))
    latencies = [r["latency"] for r in rows if r.get("latency") is not None]
    for p in (50, 95, 99):
        put(f"p{p}", percentile(latencies, p), len(latencies))
    counts = [r["request_count"] for r in attempted if r.get("request_count") is not None]
    put("request_count", sum(counts), len(counts))
    put("llm_calls", mean(counts), len(counts))
    if mode == "retrieval":
        for key in (
            "refusal_tp", "refusal_fp", "refusal_fn", "refusal_tn", "refusal_unknown",
            "refusal_precision", "refusal_recall", "refusal_f1", "false_refusal_rate",
            "answer_coverage", "scored_coverage",
        ):
            summary[key]["value"] = None
            summary[key]["n"] = 0
    for alias, key in {"faith": "faithfulness", "rel": "relevance", "refusal_acc": "refusal_recall", "false_ref": "false_refusal_rate", "cite": "citation_source_overlap"}.items():
        summary[alias] = dict(summary[key])
    return summary
