"""Saved evaluation snapshots only: no backend imports or provider requests."""

import json

import streamlit as st

from eval.report_view import (
    ReportError,
    case_failed,
    case_scored,
    filter_rows,
    load_report,
    metric_group,
    metric_label,
    metric_value,
    report_paths,
    results_directory,
    rows_csv,
    visible_metrics,
)


st.set_page_config(page_title="Evaluation · Emirates Policy Assistant", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 1.5rem; max-width: 1200px;}
    h1 {font-size: 1.75rem !important;}
    [data-testid="stMetricValue"] {font-size: 1.65rem; font-variant-numeric: tabular-nums;}
    @media (max-width: 640px) {
      .block-container {padding-left: 1rem; padding-right: 1rem;}
    }
    </style>""",
    unsafe_allow_html=True,
)
st.title("Evaluation reports")
st.caption("Emirates Policy Assistant · Saved snapshots · Read-only, no API requests")


def empty_state(directory, legacy_csv=False):
    if legacy_csv:
        st.info(
            "This folder contains legacy CSV results but no readable versioned JSON reports. "
            "CSV files lack metric denominators and run metadata, so they are not treated as "
            "complete dashboard reports. Keep them for historical analysis and create a new report."
        )
    else:
        st.info("No readable evaluation reports yet. Create an offline retrieval snapshot to get started.")
    st.caption(f"Report directory: {directory}")
    st.markdown(
        "From the repository root, with an existing local index and embedding model cache, "
        "these commands generate a retrieval-only report with a zero-request budget and open "
        "this read-only viewer. They do not generate or judge answers."
    )
    st.code(
        "python -m eval.run_eval --retrieval-only --no-judge --max-api-calls 0 --label retrieval_offline\n"
        "streamlit run pages/1_Evaluation.py",
        language="bash",
    )
    st.caption(
        "The evaluator writes to eval/results by default. If viewing a separate report folder, "
        "copy the generated JSON there or unset EMIRATES_EVAL_RESULTS_DIR. "
        "Live generation/judging is not started from this page."
    )


def metric_table(metrics, mode):
    return [
        {"Metric": metric_label(name), "Value": metric_value(metric), "n": metric["n"],
         "Definition": metric["description"]}
        for name, metric in visible_metrics(metrics, mode).items()
    ]


def render():
    directory = results_directory()
    try:
        paths = report_paths(directory)
    except ReportError as exc:
        st.error(str(exc))
        return
    reports = {}
    invalid = []
    for path in paths:
        try:
            reports[path.name] = (path, load_report(path))
        except ReportError as exc:
            invalid.append(f"{path.name}: {exc}")
    if invalid:
        st.warning(f"{len(invalid)} report file(s) could not be displayed safely.")
        with st.expander("Report validation errors"):
            for error in invalid:
                st.text(error)
    if not reports:
        try:
            legacy_csv = any(directory.glob("*.csv"))
        except OSError:
            legacy_csv = False
        empty_state(directory, legacy_csv)
        return

    selected = st.selectbox("Saved report (newest saved first)", list(reports))
    path, report = reports[selected]
    metadata = report["metadata"]
    mode = metadata["mode"]
    st.caption(
        f"Mode: {mode} · Created: {metadata['created_at']} · "
        f"Input data: {metadata['dataset']} · Retrieval k: {metadata['k']}"
    )
    coverage_fields = (
        ("completed_count", "completed"), ("selected_count", "selected"),
        ("partial_count", "partial"), ("unevaluated_count", "unevaluated"),
        ("request_count", "provider requests"),
    )
    coverage = [f"{metadata[key]} {label}" for key, label in coverage_fields if key in metadata]
    if coverage:
        st.caption("Run coverage: " + " · ".join(coverage))
    if metadata.get("budget_exhausted"):
        st.warning("Request budget exhausted: this is a partial run. Read coverage before comparing scores.")
    if mode == "retrieval":
        st.info(
            "Offline retrieval only: no answers were generated or judged. Source-label metrics "
            "are retrieval proxies, not answer accuracy. Expected source labels may be alternatives, "
            "not an exhaustive list of every relevant document."
        )
    else:
        st.info(
            "This snapshot records a live generation run; judging is measured only where recorded. "
            "Viewing it sends no requests. Reference-based answer scores apply only to cases with "
            "nonblank expected answers; missing references or judge scores are Not measured. "
            "Source retrieval and citation source-match metrics are proxies, not answer accuracy."
        )
    st.caption(
        "Each n is that metric's measured eligible sample count. Not measured is not zero. "
        "Answer/refuse decision accuracy measures the decision only, not whether the answer is correct. "
        "Expected source labels may be alternatives rather than exhaustive relevance judgments."
    )
    with st.expander("Run metadata and measurement limits"):
        st.json(metadata)
        st.caption(
            "Not measured is not zero. Each metric's n is its own eligible measured sample count, "
            "not necessarily the dataset size. Judge scores are model assessments, not verified "
            "ground truth. References describe the indexed policy snapshot, not current-web verification. "
            "Failed, partial and unevaluated cases must be read alongside coverage and errors."
        )

    category_options = sorted(set(report["categories"]) | {row["category"] for row in report["rows"]})
    category = st.selectbox("Category", ["All categories", *category_options])
    category_filter = None if category == "All categories" else category
    metrics = report["summary"] if category_filter is None else report["categories"].get(category, {})
    display_metrics = visible_metrics(metrics, mode)
    st.subheader("Run metrics" if category_filter is None else f"Category metrics: {category}")
    st.caption("Metric scope follows Category. The case-status filter below affects case inspection and its CSV only.")
    lead_names = [
        name for name in ("hit", "recall_at_k", "exact_match", "decision_accuracy", "service_error_rate")
        if name in display_metrics
    ][:3]
    if not lead_names:
        lead_names = list(display_metrics)[:3]
    if lead_names:
        for column, name in zip(st.columns(len(lead_names)), lead_names):
            metric = display_metrics[name]
            with column:
                st.metric(metric_label(name), metric_value(metric), help=metric["description"])
                st.caption(f"n = {metric['n']:,} measured samples")
    groups = ["Retrieval", "Answer", "Refusal", "Citation", "Operational"]
    for tab, group in zip(st.tabs(groups), groups):
        with tab:
            group_metrics = {name: metric for name, metric in display_metrics.items() if metric_group(name) == group}
            if group_metrics:
                st.dataframe(metric_table(group_metrics, mode), hide_index=True, use_container_width=True)
            else:
                st.caption(
                    "Not measured · n = 0. Retrieval-only mode generates and judges no answers."
                    if mode == "retrieval" and group in ("Answer", "Refusal", "Citation")
                    else "Not measured in this report."
                )

    st.subheader("Per-category metrics")
    category_rows = []
    for name, values in report["categories"].items():
        if category_filter is None or name == category_filter:
            for entry in metric_table(values, mode):
                category_rows.append({"Category": name, **entry})
    if category_rows:
        st.dataframe(category_rows, hide_index=True, use_container_width=True, height=240)
    else:
        st.caption("No category aggregates were recorded for this selection.")

    st.subheader("Case inspection")
    status = st.selectbox("Case status", ["All cases", "Failed (execution or judge)", "Scored", "Unscored"])
    rows = filter_rows(report["rows"], category_filter, status)
    st.caption(
        f"{len(rows):,} of {len(report['rows']):,} cases shown · "
        "Failed means a recorded execution/judge error, not a low metric score. "
        "Scored means at least one retrieval, answer, citation or refusal measurement is present."
    )
    if rows:
        st.dataframe(
            [{"Case": str(row.get("id", index + 1)), "Category": row["category"],
              "Status": str(row.get("status", "error" if case_failed(row) else "recorded")),
              "Scored": case_scored(row), "Question": row["question"], "Error": row.get("error") or ""}
             for index, row in enumerate(rows)],
            hide_index=True, use_container_width=True, height=240,
        )
        case_index = st.selectbox(
            "Inspect case", range(len(rows)),
            format_func=lambda index: f"{rows[index].get('id', index + 1)} · {rows[index]['question'][:100]}",
        )
        row = rows[case_index]
        for field, label in (("question", "Question"), ("answer", "Answer"), ("expected_answer", "Reference answer"),
                             ("error", "Execution error"), ("judge_error", "Judge error"), ("judge_reason", "Judge reason")):
            st.markdown(f"**{label}**")
            st.text(row.get(field) or "Not recorded")
        with st.expander("All recorded case fields"):
            st.json(row)
    else:
        st.info("No cases match these filters.")

    st.subheader("Download snapshot")
    st.caption("JSON contains the complete saved report; CSV contains the cases in the current filters.")
    left, right = st.columns(2)
    with left:
        st.download_button("Download report JSON", json.dumps(report, ensure_ascii=False, indent=2),
                           file_name=path.name, mime="application/json")
    with right:
        st.download_button("Download filtered cases CSV", rows_csv(rows),
                           file_name=f"{path.stem}_filtered.csv", mime="text/csv")

    if len(reports) > 1:
        with st.expander("Compare saved runs"):
            st.caption(
                "Run-level snapshots, unaffected by case filters. Compare only equivalent datasets, "
                "retrieval k, modes and eligible samples; different coverage can change a mean."
            )
            comparison = []
            for filename, (_, other) in reports.items():
                meta = other["metadata"]
                for entry in metric_table(other["summary"], meta["mode"]):
                    comparison.append({"Run": filename, "Created": meta["created_at"], "Mode": meta["mode"],
                                       "Dataset": meta["dataset"], "k": meta["k"], **entry})
            st.dataframe(comparison, hide_index=True, use_container_width=True)


render()
