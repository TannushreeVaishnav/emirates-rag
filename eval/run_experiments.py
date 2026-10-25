"""Retrieval-only experiments (no LLM calls).

python -m eval.run_experiments chunking
python -m eval.run_experiments retriever --index index_exp/c900
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.config import ROOT

EXP = ROOT / "index_exp"
OUT = ROOT / "eval" / "results"

CHUNKING = {
    "c500":        {"CHUNK_SIZE": "500",  "CHUNK_OVERLAP": "75"},
    "c900":        {"CHUNK_SIZE": "900",  "CHUNK_OVERLAP": "150"},
    "c1000":       {"CHUNK_SIZE": "1000", "CHUNK_OVERLAP": "150"},
    "c500-fixed":  {"CHUNK_SIZE": "500",  "CHUNK_OVERLAP": "75",  "STRUCTURE_AWARE": "false"},
    "c900-fixed":  {"CHUNK_SIZE": "900",  "CHUNK_OVERLAP": "150", "STRUCTURE_AWARE": "false"},
    "c1000-fixed": {"CHUNK_SIZE": "1000", "CHUNK_OVERLAP": "150", "STRUCTURE_AWARE": "false"},
}
GRID = {   # name -> (k, mmr)
    "k3-sim": (3, False), "k4-sim": (4, False), "k6-sim": (6, False),
    "k4-mmr": (4, True),  "k6-mmr": (6, True),
}


def run(args: list[str], env_extra: dict) -> None:
    p = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env={**os.environ, **env_extra},
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-1500:])
        raise SystemExit(f"failed: {' '.join(str(a) for a in args)}")
    print(p.stdout[-600:])


def retrieval_eval(label: str, env: dict) -> dict:
    out = OUT / f"exp_{label}.json"
    run(
        ["-m", "eval.run_eval", "--retrieval-only", "--exclude-category", "follow_up",
         "--label", f"exp-{label}", "--json-out", str(out), "--no-table"],
        env,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def report(title: str, results: dict, base: str, path: Path) -> None:
    base_miss = set(results[base][1]["misses"])
    lines = [
        f"### {title}", "",
        f"| Config | Chunks | Avg chars | Hit@k | Direct | Multi | Fixed vs {base} | Broken vs {base} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, (stats, res) in results.items():
        miss = set(res["misses"])
        n = res["hit_n"]
        hit = res["hit"]
        cat = res["hit_by_cat"]
        lines.append(
            f"| {name} | {stats['chunks']} | {stats['avg_chars']} | {fmt(hit)} "
            f"({round((hit or 0) * n)}/{n}) | "
            f"{fmt(cat.get('direct'))} | {fmt(cat.get('multi'))} | "
            f"{', '.join(sorted(base_miss - miss)) or '-'} | "
            f"{', '.join(sorted(miss - base_miss)) or '-'} |"
        )
    text = "\n".join(lines)
    print("\n" + text)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"\nReport saved to {path}")


def cmd_chunking() -> None:
    results = {}
    for name, cfg in CHUNKING.items():
        idx = EXP / name
        env = {
            **cfg,
            "INDEX_DIR": str(idx),
            "TOP_K": "4",
            "USE_MMR": "false",
            "HF_HOME": os.environ.get("HF_HOME", "D:/hf_cache"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        print(f"\n{'='*60}")
        print(f"[{name}] ingesting ...")
        run(["-m", "src.ingest"], env)
        stats = json.loads((idx / "stats.json").read_text(encoding="utf-8"))
        print(f"[{name}] {stats['chunks']} chunks, avg {stats['avg_chars']} chars — evaluating ...")
        results[name] = (stats, retrieval_eval(name, env))
    OUT.mkdir(parents=True, exist_ok=True)
    report(
        "Chunking comparison (similarity search, k=4)",
        results,
        "c900",
        OUT / "experiments_chunking.md",
    )


def cmd_retriever(index: str) -> None:
    idx = Path(index)
    idx = idx if idx.is_absolute() else ROOT / idx
    stats_file = idx / "stats.json"
    if not stats_file.exists():
        raise SystemExit(
            f"stats.json not found in {idx}. Re-run ingestion: python -m src.ingest"
        )
    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    base_env = {
        "INDEX_DIR": str(idx),
        "CHUNK_SIZE": str(stats["chunk_size"]),
        "CHUNK_OVERLAP": str(stats["chunk_overlap"]),
        "STRUCTURE_AWARE": str(stats["structure_aware"]).lower(),
        "HF_HOME": os.environ.get("HF_HOME", "D:/hf_cache"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    results = {}
    for name, (k, mmr) in GRID.items():
        print(f"\n[{name}] k={k}, mmr={mmr} — evaluating ...")
        results[name] = (
            stats,
            retrieval_eval(name, {**base_env, "TOP_K": str(k), "USE_MMR": str(mmr).lower()}),
        )
    OUT.mkdir(parents=True, exist_ok=True)
    report(
        f"Retriever comparison on {idx.name}",
        results,
        "k4-sim",
        OUT / "experiments_retriever.md",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chunking")
    r = sub.add_parser("retriever")
    r.add_argument("--index", default="index_exp/c900")
    args = ap.parse_args()
    if args.cmd == "chunking":
        cmd_chunking()
    else:
        cmd_retriever(args.index)
