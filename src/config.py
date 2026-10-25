"""
config.py — Central configuration for the Emirates RAG Bot.

All tuneable values (model names, chunk sizes, paths, feature flags) live here
so that I only need to change one file — or set an environment variable — to
tweak behaviour across the whole project.

pydantic-settings automatically reads .env values and maps them to the typed
fields below, which makes it easy to override anything without touching code.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Make sure heavy model caches stay on D: (C: is full on my dev machine) ──────
os.environ.setdefault("HF_HOME", "D:/hf_cache")
# OpenBLAS crashes with multi-threading on Windows when FAISS loads; pin to 1.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Project root is two levels up from this file (src/config.py → project root).
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Single source of truth for every tunable parameter.

    Values are read from .env first, then fall back to the defaults below.
    Environment variable names match the field names in UPPER_CASE
    (e.g. GEMINI_MODEL, TOP_K, CHUNK_SIZE).
    """

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # ── LLM provider keys & model names ──────────────────────────────────────
    google_api_key: str = ""
    groq_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    groq_model: str = "openai/gpt-oss-120b"      # set in .env; free-tier Groq

    # ── Embedding & retrieval ─────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"   # fast, good quality, free
    top_k: int = 4           # number of chunks returned per query
    temperature: float = 0.1 # low temp → more deterministic policy answers
    market: str = "IN"       # controls which market-specific rules apply

    # ── Chunking (changed via env vars during Phase 8 experiments) ────────────
    chunk_size: int = 900
    chunk_overlap: int = 150
    structure_aware: bool = True   # False → plain fixed-size splits (experiment baseline)
    use_mmr: bool = False          # True → Maximal Marginal Relevance for diverse retrieval

    # ── Graph feature flags (can be toggled off for ablation studies) ─────────
    use_grader: bool = True        # grade retrieved chunks before generating
    use_output_check: bool = True  # verify answer is grounded in context
    max_rewrites: int = 1          # max query rewrites before returning NO_ANSWER

    # ── LLM gateway settings ──────────────────────────────────────────────────
    primary_provider: str = "gemini"
    fallback_provider: str = "groq"
    llm_timeout: float = 25.0      # seconds; free-tier APIs can be slow
    llm_attempts: int = 2          # retries on transient errors before failover
    force_primary_fail: bool = False  # test switch: forces Gemini to fail → Groq

    # ── Phase 8: fast helper model (router / grader / rewriter only) ──────────
    fast_model: str = ""    # e.g. "llama-3.1-8b-instant"; empty = use main model

    # ── Paths ─────────────────────────────────────────────────────────────────
    log_dir: Path = ROOT / "logs"
    raw_dir: Path = ROOT / "data" / "raw"
    sources_csv: Path = ROOT / "data" / "sources.csv"
    index_dir: Path = ROOT / "index"


# Module-level singleton — import `settings` everywhere instead of re-creating it.
settings = Settings()
