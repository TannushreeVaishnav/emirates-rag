# Emirates Policy Assistant — RAG Chatbot

> **Unofficial demo project.** Not affiliated with Emirates Airline.

A conversational AI that answers questions about Emirates airline travel policies
(baggage, check-in, refunds, special assistance, dangerous goods) using
Retrieval-Augmented Generation (RAG). Built as a final-year AI Engineering capstone.

---

## What it does

- Answers policy questions from a 17-document knowledge base (PDFs scraped from emirates.com)
- Grounds answers in retrieved passages and shows source passages and page numbers
- Reports low confidence when the documents do not provide enough supporting evidence
- Resolves follow-up questions using recent conversation context
- Refuses off-topic, price, or booking requests with a polite explanation
- Detects and blocks prompt-injection attempts
- Strips PII (emails, card numbers, passports) before it reaches any LLM or log
- Falls back from Gemini to Groq automatically when rate limits hit
- Remembers context across turns in a multi-turn conversation

Retrieval combines vector similarity with BM25 keyword matching over the same
indexed chunks, merging ranks and removing duplicate passages. Confidence is an
evidence decision, not a calibrated percentage: insufficient evidence, invalid
citations, or failed answer verification produce an explicit low-confidence
response rather than an unverified policy answer. Keep `USE_GRADER` and
`USE_OUTPUT_CHECK` enabled for normal use; disabling them weakens these checks.
PDF cleanup preserves standalone numbers and repeated words, including negations:
token-per-line extraction must not turn a policy limit or exception into missing text.

---

## Architecture

```
User question
     │
     ▼
┌─────────────┐   block   ┌──────────────┐
│  Guardrail  │──────────▶│ Refuse node  │
│  (PII mask  │           └──────────────┘
│  + injection│
│  detection) │
└──────┬──────┘
       │ pass
       ▼
┌─────────────┐   route   ┌──────────────────────┐
│   Router    │──────────▶│ flight_status / chat  │
│   (LLM)     │           │ / out_of_scope        │
└──────┬──────┘           └──────────────────────┘
       │ policy
       ▼
┌─────────────┐
│  FAISS      │  vector similarity / MMR
│  Retriever  │
└──────┬──────┘
       │ top-k chunks
       ▼
┌─────────────┐   irrelevant   ┌─────────────┐   max retries
│   Grader    │───────────────▶│  Rewriter   │──────────────▶ NO_ANSWER
│   (LLM)     │                │  (LLM)      │
└──────┬──────┘                └──────┬──────┘
       │ relevant                     │ rewritten query → retriever
       ▼
┌─────────────┐
│  Generator  │  cites [1][2] from context
│   (LLM)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐   fail   ┌──────────────┐
│ Output check│─────────▶│ NO_ANSWER    │
│  (rules +   │           └──────────────┘
│   LLM)      │
└──────┬──────┘
       │ pass
       ▼
   Answer + citations
```

**Provider failover:** every LLM call goes through `LLMGateway` which tries
Gemini first, classifies any error (rate-limit, transient, fatal), and falls
over to Groq with exponential backoff on transient errors only.

---

## Project structure

```
emirates-rag-bot/
├── app.py                  # Streamlit chat UI
├── requirements.txt
├── pytest.ini
│
├── src/                    # Core library (importable as `src.*`)
│   ├── config.py           # All settings; reads .env
│   ├── ingest.py           # PDF/MD → clean → chunk → embed → FAISS
│   ├── retriever.py        # FAISS similarity / MMR search
│   ├── rag_chain.py        # Baseline RAG chain (Phase 3)
│   ├── graph.py            # LangGraph agent (Phase 5+)
│   ├── prompts.py          # All LangChain prompt templates
│   ├── llm_gateway.py      # Gemini → Groq failover gateway
│   ├── guardrails.py       # PII masking, injection detection, output rules
│   ├── tools.py            # Mock flight-status tool
│   ├── logging_utils.py    # JSONL request logger
│   ├── cli_ask.py          # One-shot terminal tester (baseline)
│   ├── cli_chat.py         # Multi-turn terminal tester (agent)
│   └── check_kb.py         # Pre-ingestion sanity check
│
├── eval/                   # Evaluation framework
│   ├── qa_set.json         # 40-question eval set (direct, multi, refusal)
│   ├── qa_holdout.json     # 10-question holdout (run once, after tuning)
│   ├── redteam.json        # 15 adversarial test cases
│   ├── run_eval.py         # Main eval runner (retrieval-only or full graph)
│   ├── run_experiments.py  # Phase 8: chunking & retriever grid search
│   ├── run_redteam.py      # Red-team runner
│   └── failures.py         # Post-eval failure analysis by bucket
│
├── tests/                  # Offline pytest suite
│   ├── test_guardrails.py  # PII, injection, output rules (31 tests)
│   ├── test_gateway.py     # Failover, retry, forced-fail (7 tests)
│   ├── test_graph_routes.py# Graph routing, memory, security, evidence checks
│   ├── test_retriever.py   # Hybrid retrieval and cached-document isolation
│   ├── test_ingest.py      # Preserve policy numbers and negations during PDF cleanup
│   └── test_app.py         # Streamlit AppTest smoke test (requires built index)
│
├── data/
│   ├── raw/                # Source PDFs (gitignored)
│   ├── sources.csv         # Metadata: file, url, market, doc_type, date
│   └── mock/flights.json   # Mock flight data for the status tool
│
├── index/                  # FAISS index (gitignored, built by ingest.py)
├── logs/                   # Request & feedback logs (gitignored)
├── docs/                   # Eval report (eval_report.md)
└── .streamlit/config.toml  # Emirates red theme
```

---

## Setup

**Prerequisites:** Python 3.11+, Git, ~2 GB free on a non-C drive (for model cache).

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/<you>/emirates-rag-bot
cd emirates-rag-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env template and add your API keys
copy .env.example .env               # Windows
# cp .env.example .env               # macOS / Linux
# Edit .env: add GOOGLE_API_KEY and GROQ_API_KEY

# 4. Check the knowledge base files are present
python -m src.check_kb

# 5. Build the FAISS index (takes 3–5 min on first run)
python -m src.ingest

# 6. Launch the Streamlit UI
streamlit run app.py
```

---

## API keys

| Key | Where to get it | Free tier |
|:----|:----------------|:----------|
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | 1 M tokens/day |
| `GROQ_API_KEY`   | [console.groq.com](https://console.groq.com) | 14 400 req/day |

---

## Running the tests

The suite makes no live LLM API calls. The UI test requires a built knowledge-base index.

```bash
pytest tests/ -v
```

---

## Evaluation

```bash
# Retrieval-only (no API cost): compare 6 chunking configurations
python -m eval.run_experiments chunking

# Full graph eval with LLM judge
python -m eval.run_eval --config graph --label final --sleep 8

# Red-team: 16 adversarial inputs
python -m eval.run_redteam

# Analyse failures from the most recent CSV
python -m eval.failures
```

Results are saved to `eval/results/summary.md` and per-question CSVs.

---

## Environment variables (`.env`)

| Variable | Default | Description |
|:---------|:--------|:------------|
| `GOOGLE_API_KEY` | — | Gemini API key |
| `GROQ_API_KEY` | — | Groq API key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Primary LLM |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Fallback LLM |
| `TOP_K` | `4` | Chunks retrieved per query |
| `CHUNK_SIZE` | `900` | Characters per chunk |
| `USE_GRADER` | `true` | Enable relevance grading |
| `USE_OUTPUT_CHECK` | `true` | Enable output verification |
| `FORCE_PRIMARY_FAIL` | `false` | Force Groq fallback (testing) |
| `FAST_MODEL` | `` | Small helper model (e.g. `llama-3.1-8b-instant`) |

---

## Technical highlights (for portfolio / interviews)

- **LangGraph state machine** with 11 nodes, conditional edges, and per-session
  `MemorySaver` for multi-turn memory.
- **Three-layer input guardrail**: regex PII masking → regex injection detection
  with Unicode normalisation → LLM attack classifier.
- **Deterministic output rules** checked before the LLM output-check call to
  catch invented monetary amounts and guarantee claims cheaply.
- **Custom LangChain Runnable** (`LLMGateway`) with per-error-class retry
  strategy and `ContextVar`-based call tracking inside LangGraph threads.
- **Retrieval evaluation framework** with hit-rate@k, LLM-as-judge
  faithfulness/relevance scoring, refusal accuracy, and a grid-search
  experiment runner across 6 chunking configs × 5 retriever configs.
- **Offline tests** using monkeypatching so the full graph, guardrails,
  and gateway are tested without any API calls.

---

## Limitations

- This is an **unofficial demo** using publicly available documents. Always
  verify important travel information on emirates.com.
- The knowledge base was collected on a specific date and may be out of date.
- The LLM judge used for evaluation is the same model family as the generator,
  which may inflate faithfulness scores.
- Free-tier rate limits affect latency; p95 may exceed 6 s under load.

Policy freshness means whether the local documents still match Emirates' current
published rules. `data/sources.csv` records collection dates, not policy effective
dates or proof that a source is still current. Refresh manually: check each original
URL, replace changed files in `data/raw`, update collection dates only after actually
collecting the source, run `python -m src.ingest`, and restart the app so cached
retrieval data reloads. Rebuilding the index alone does not fetch newer policies.
There is no automatic source-refresh job.

---

*Built as part of an AI Engineering capstone project — 2026.*
