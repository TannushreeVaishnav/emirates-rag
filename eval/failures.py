"""Group failures from an eval CSV.   python -m eval.failures [path]"""
import csv
import json
import sys
from pathlib import Path

from src.config import ROOT

gold = {
    i["id"]: i["source_doc"]
    for i in json.loads((ROOT / "eval" / "qa_set.json").read_text(encoding="utf-8"))
}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def newest() -> Path:
    files = sorted(
        (ROOT / "eval" / "results").glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        raise SystemExit("No result CSVs yet. Run an eval first.")
    return files[-1]


path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest()
rows = list(csv.DictReader(open(path, encoding="utf-8")))
print(f"{path.name}: {len(rows)} rows\n")

BUCKETS = [
    (
        "RETRIEVAL MISS (gold doc not in top-k)",
        lambda r: r["hit"] == "False",
        lambda r: f"gold={gold.get(r['id'])}  got: {r['top_sources'][:200]}",
    ),
    (
        "FALSE REFUSAL (answerable but refused)",
        lambda r: r["should_refuse"] == "False" and r["refused"] == "True",
        lambda r: f"hit={r['hit']}  {r['answer'][:140]}",
    ),
    (
        "MISSED REFUSAL (should have refused)",
        lambda r: r["should_refuse"] == "True" and r["refused"] == "False",
        lambda r: r["answer"][:160],
    ),
    (
        "WRONG CITATION",
        lambda r: r["citation_ok"] == "False" and r["refused"] == "False",
        lambda r: f"gold={gold.get(r['id'])}  cited={r['cited_sources']}",
    ),
    (
        "NOT FAITHFUL",
        lambda r: (num(r.get("faithfulness")) or 1) < 1,
        lambda r: r.get("judge_reason", ""),
    ),
    (
        "NOT RELEVANT",
        lambda r: (num(r.get("relevance")) or 1) < 1,
        lambda r: r.get("judge_reason", ""),
    ),
    (
        "ERROR",
        lambda r: bool(r.get("error")),
        lambda r: r.get("error", ""),
    ),
]

for title, test, detail in BUCKETS:
    hits = [r for r in rows if test(r)]
    print(f"== {title}: {len(hits)}")
    for r in hits:
        print(f"  {r['id']:4s} {r['question'][:70]}\n       {detail(r)}")
    print()
