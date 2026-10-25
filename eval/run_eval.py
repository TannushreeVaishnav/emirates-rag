"""Evaluate a RAG configuration on eval/qa_set.json.

python -m eval.run_eval --config baseline
python -m eval.run_eval --retrieval-only          # no LLM calls, checks hit rate only
python -m eval.run_eval --config baseline --no-judge --limit 5
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import ROOT, settings
from src.llm_gateway import get_llm
from src.retriever import retrieve

QA_PATH = ROOT / "eval" / "qa_set.json"
RESULTS = ROOT / "eval" / "results"

REFUSAL_MARKERS = (
    "i couldn't find",
    "i could not find",
    "i can help with emirates travel policies",
    "please check emirates.com",
)


def is_refusal(answer: str) -> bool:
    a = answer.strip().lower().replace("\u2019", "'")
    return any(m in a[:200] for m in REFUSAL_MARKERS)


# ---- runners: each takes (question, history) and returns a dict -----------------
def run_baseline(question: str, history: list[str]) -> dict:
    from src.rag_chain import ask
    return ask(question)  # baseline has no memory, history is ignored


def run_graph(question: str, history: list[str]) -> dict:
    import uuid
    from src.graph import chat
    tid = f"eval-{uuid.uuid4()}"
    for earlier in history:  # replay earlier turns so follow-ups have memory
        chat(earlier, tid)
    res = chat(question, tid)  # latency reflects the final turn only
    if res.get("error"):
        raise RuntimeError(res["error"])
    return res


RUNNERS = {"baseline": run_baseline, "graph": run_graph}


# ---- LLM-as-judge ---------------------------------------------------------------
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict evaluator of a retrieval-augmented QA system.
Score the ANSWER using only the QUESTION, CONTEXT and (if given) REFERENCE.

faithfulness: 1 = every factual claim in the ANSWER is supported by the CONTEXT;
0.5 = mostly supported but with a minor unsupported detail; 0 = contains claims that are
absent from or contradicted by the CONTEXT.
relevance: 1 = directly answers the QUESTION (and agrees with REFERENCE if given);
0.5 = partially answers; 0 = does not answer or contradicts the REFERENCE.

Return JSON only: {{"faithfulness": 0 or 0.5 or 1, "relevance": 0 or 0.5 or 1, "reason": "one short sentence"}}"""),
    ("human", "QUESTION: {question}\n\nREFERENCE: {reference}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}"),
])


def build_judge():
    return JUDGE_PROMPT | get_llm() | JsonOutputParser()


def judge_answer(judge, item: dict, res: dict):
    # Prefer context_docs (graded) over docs (raw retrieved) for faithfulness
    docs = res.get("context_docs") or res["docs"]
    context = "\n\n".join(d.page_content for d in docs)[:6000]
    try:
        out = judge.invoke({
            "question": item["question"],
            "reference": item.get("expected_answer") or "(none provided)",
            "context": context,
            "answer": res["answer"],
        })
        return float(out["faithfulness"]), float(out["relevance"]), str(out.get("reason", ""))
    except Exception as e:
        return None, None, f"judge failed: {str(e)[:80]}"


# ---- metrics --------------------------------------------------------------------
def hit(item: dict, docs: list) -> bool:
    gold = set(item["source_doc"])
    kw = item.get("gold_keyword", "").lower()
    return any(
        d.metadata["source"] in gold and (not kw or kw in d.page_content.lower())
        for d in docs
    )


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def fmt(x, digits=2):
    return "n/a" if x is None else f"{x:.{digits}f}"


def validate(items):
    sources_path = settings.sources_csv if hasattr(settings, "sources_csv") else ROOT / "data" / "sources.csv"
    known = {r["file"].split("/")[-1] for r in csv.DictReader(open(sources_path, encoding="utf-8"))}
    for it in items:
        for s in it["source_doc"]:
            if s not in known:
                print(f"WARNING {it['id']}: source_doc '{s}' not in sources.csv — check label")


# ---- main -----------------------------------------------------------------------
def evaluate(args):
    items = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    if args.category:
        items = [i for i in items if i["category"] == args.category]
    if args.exclude_category:
        items = [i for i in items if i["category"] not in args.exclude_category]
    if args.limit:
        items = items[: args.limit]
    validate(items)

    judge = None if (args.no_judge or args.retrieval_only) else build_judge()
    runner = RUNNERS[args.config]
    rows = []

    for n, item in enumerate(items, start=1):
        err = ""
        try:
            if args.retrieval_only:
                t0 = time.perf_counter()
                docs = retrieve(item["question"])
                res = {"answer": "", "citations": [], "docs": docs, "latency": time.perf_counter() - t0,
                       "llm_calls": 0, "llm_seconds": 0.0}
            else:
                res = runner(item["question"], item.get("history", []))
        except Exception as e:
            err = str(e)[:200]
            res = {"answer": "", "citations": [], "docs": [], "latency": 0.0,
                   "llm_calls": 0, "llm_seconds": 0.0}

        answerable = not item["should_refuse"]
        refused = (
            bool(res.get("refused", is_refusal(res["answer"])))
            if res["answer"]
            else False
        )
        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "should_refuse": item["should_refuse"],
            "refused": None if args.retrieval_only else refused,
            "decision_correct": (
                None
                if (args.retrieval_only or err)
                else (refused == item["should_refuse"])
            ),
            "hit": hit(item, res["docs"]) if (answerable and item["source_doc"]) else None,
            "citation_ok": None,
            "faithfulness": None,
            "relevance": None,
            "latency": None if (err or args.retrieval_only) else round(res["latency"], 2),
            "llm_calls": res.get("llm_calls"),
            "llm_seconds": res.get("llm_seconds"),
            "top_sources": " | ".join(
                f"{d.metadata['source']}::{d.metadata['section'][:30]}" for d in res["docs"]
            ),
            "cited_sources": " | ".join(sorted({c["source"] for c in res["citations"]})),
            "judge_reason": "",
            "answer": res["answer"].replace("\n", " "),
            "error": err,
        }

        if answerable and not args.retrieval_only and not err and item["source_doc"]:
            row["citation_ok"] = (not refused) and any(
                c["source"] in item["source_doc"] for c in res["citations"]
            )

        if judge and answerable and not refused and res["answer"]:
            row["faithfulness"], row["relevance"], row["judge_reason"] = judge_answer(
                judge, item, res
            )

        tag = "ERR " if err else ("HIT " if row["hit"] else "MISS" if row["hit"] is False else "    ")
        print(f"[{n:02d}/{len(items)}] {item['id']} {tag} {item['question'][:70]}")
        if args.retrieval_only and row["hit"] is False:
            print(f"        gold={item['source_doc']}  got={row['top_sources'][:120]}")
        if err:
            print(f"        error: {err}")

        rows.append(row)
        if not args.retrieval_only:
            time.sleep(args.sleep)

    write_outputs(rows, args)


def write_outputs(rows, args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or ("retrieval" if args.retrieval_only else args.config)

    csv_path = RESULTS / f"{label}_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    answerable = [r for r in rows if not r["should_refuse"]]
    should_ref = [r for r in rows if r["should_refuse"]]
    lat = [r["latency"] for r in rows if r["latency"] is not None]

    summary = {
        "hit": mean([1.0 if r["hit"] else 0.0 for r in rows if r["hit"] is not None]),
        "faith": mean([r["faithfulness"] for r in rows]),
        "rel": mean([r["relevance"] for r in rows]),
        "refusal_acc": mean(
            [1.0 if r["refused"] else 0.0 for r in should_ref if r["decision_correct"] is not None]
        ),
        "false_ref": mean(
            [1.0 if r["refused"] else 0.0 for r in answerable if r["decision_correct"] is not None]
        ),
        "cite": mean([1.0 if r["citation_ok"] else 0.0 for r in rows if r["citation_ok"] is not None]),
        "p50": pct(lat, 50),
        "p95": pct(lat, 95),
        "errors": sum(1 for r in rows if r["error"]),
        "llm_calls": mean([r["llm_calls"] for r in rows]),
    }

    chunk = getattr(settings, "chunk_size", "?")
    print(f"\n=== {label}  (n={len(rows)}, k={settings.top_k}, chunk={chunk}) ===")
    print(f"Hit rate@{settings.top_k}:        {fmt(summary['hit'])}")
    print(f"Faithfulness:        {fmt(summary['faith'])}")
    print(f"Answer relevance:    {fmt(summary['rel'])}")
    print(f"Refusal accuracy:    {fmt(summary['refusal_acc'])}   (should-refuse correctly refused)")
    print(f"False refusal rate:  {fmt(summary['false_ref'])}   (answerable wrongly refused)")
    print(f"Citation correct:    {fmt(summary['cite'])}")
    print(f"Latency p50 / p95:   {fmt(summary['p50'], 1)}s / {fmt(summary['p95'], 1)}s")
    print(f"Errors:              {summary['errors']}")
    print(f"LLM calls/question:  {fmt(summary['llm_calls'], 1)}")

    print("\nBy category (decision correct = refused when it should, answered when it should):")
    for cat in dict.fromkeys(r["category"] for r in rows):
        sub = [r for r in rows if r["category"] == cat and r["decision_correct"] is not None]
        if sub:
            print(f"  {cat:12s} {sum(r['decision_correct'] for r in sub)}/{len(sub)}")

    if args.json_out:
        summary["n"] = len(rows)
        summary["hit_n"] = len([r for r in rows if r["hit"] is not None])
        summary["misses"] = [r["id"] for r in rows if r["hit"] is False]
        summary["hit_by_cat"] = {
            c: mean([1.0 if r["hit"] else 0.0 for r in rows if r["category"] == c and r["hit"] is not None])
            for c in dict.fromkeys(r["category"] for r in rows)
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_table:
        md_path = RESULTS / "summary.md"
        if not md_path.exists():
            md_path.write_text(
                "| Date | Config | N | k | Chunk | Hit@k | Faithfulness | Relevance |"
                " Refusal acc. | False refusals | Citation | p50 | p95 |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
                encoding="utf-8",
            )
        with open(md_path, "a", encoding="utf-8") as f:
            f.write(
                f"| {stamp[:8]} | {label} | {len(rows)} | {settings.top_k} | {chunk} | "
                f"{fmt(summary['hit'])} | {fmt(summary['faith'])} | {fmt(summary['rel'])} | "
                f"{fmt(summary['refusal_acc'])} | {fmt(summary['false_ref'])} | {fmt(summary['cite'])} | "
                f"{fmt(summary['p50'], 1)}s | {fmt(summary['p95'], 1)}s |\n"
            )
        print(f"\nSummary table:    {md_path}")
    print(f"Per-question CSV: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline", choices=list(RUNNERS))
    ap.add_argument("--label", help="name for this run in the summary table")
    ap.add_argument("--retrieval-only", action="store_true", help="skip LLM; measure hit rate only")
    ap.add_argument("--no-judge", action="store_true", help="skip faithfulness/relevance scoring")
    ap.add_argument("--category", help="run a single category, e.g. direct")
    ap.add_argument("--exclude-category", action="append", default=[], help="exclude a category (repeatable)")
    ap.add_argument("--limit", type=int, help="run only the first N items")
    ap.add_argument("--json-out", help="write the summary as JSON to this path")
    ap.add_argument("--no-table", action="store_true", help="don't append a row to summary.md")
    ap.add_argument("--qa", default=str(QA_PATH), help="eval file (default: eval/qa_set.json)")
    ap.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="seconds between questions (free-tier rate limits)",
    )
    evaluate(ap.parse_args())
