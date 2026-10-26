# Emirates Policy Assistant — RAG Chatbot

> **Unofficial demo project.** Not affiliated with Emirates Airline.

A conversational AI that answers questions about Emirates airline travel policies
(baggage, check-in, refunds, special assistance, dangerous goods) using
Retrieval-Augmented Generation (RAG). Built as a final-year AI Engineering capstone.

---

## What it does

- Answers policy questions from a 17-document knowledge base (PDFs scraped from emirates.com)
- Shows clickable inline citations linked to original sources, with source passages and page numbers
- Refuses off-topic, price, or booking requests with a polite explanation
- Detects and blocks prompt-injection attempts
- Strips PII (emails, card numbers, passports) before it reaches any LLM or log
- Falls back from Gemini to Groq automatically when rate limits hit
- Remembers context across turns in a multi-turn conversation

Retrieval combines vector similarity with BM25 keyword matching over the same
indexed chunks, using reciprocal-rank fusion and passage deduplication. PDF cleanup
preserves policy numbers, negations, and section ancestry, including regional
qualifications. Standalone questions are not rewritten using unrelated chat history.

Provider outages and failed evidence assessment produce a service/verification
error, not a claim that the policy is absent. A rate-limited provider/model is
skipped for the rest of that turn; a new turn may try it again. No policy answer
passes through merely because its grader or support checker is unavailable.

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
│   ├── retriever.py        # Hybrid FAISS/BM25 retrieval
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
├── tests/                  # Offline regression suite; no live provider calls
│   ├── test_guardrails.py  # PII, injection, output rules
│   ├── test_gateway.py     # Failover, retry, rate-limit suppression/recovery
│   ├── test_graph_routes.py# Routing, memory, repeated questions, security
│   ├── test_ingest.py      # Policy facts, section ancestry, PDF cleanup
│   ├── test_retriever.py   # Hybrid retrieval and cached-document isolation
│   ├── test_evidence.py    # Evidence failures and clickable citations
│   └── test_app.py         # Streamlit AppTest smoke test
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

**Prerequisites:** Python 3.11+, Git, and space for the local embedding model.
Set `HF_HOME` if you need a custom cache location; the app otherwise uses the
Hugging Face default.

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

After changing ingestion or replacing policy documents, rebuild the index and
restart Streamlit so its cached retriever reloads. Rebuilding does not download
newer policies. Collection dates in `data/sources.csv` are not proof that a
policy is still current.

The Streamlit file watcher is disabled to avoid introspection of Transformers'
optional vision modules. Restart the server after Python/configuration changes;
browser refresh alone does not reload the cached backend. `torchvision` is not
required for this text-only application.

---

## API keys

| Key | Where to get it | Free tier |
|:----|:----------------|:----------|
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | 1 M tokens/day |
| `GROQ_API_KEY`   | [console.groq.com](https://console.groq.com) | 14 400 req/day |

---

## Running the tests

The regression suite makes no live LLM API calls. The UI smoke test requires a
built index. Tests cover policy-preserving extraction, hybrid retrieval, repeated
questions, rate-limit suppression/recovery, evidence-check failures, and citations.

```bash
pytest tests/ -v
```

---

## Evaluation

Open **Evaluation** in the Streamlit sidebar to inspect saved runs. The dashboard
is read-only: viewing, filtering, comparing, or downloading results never calls
an LLM.

Run commands from the repository root with the virtual environment activated
(`.\.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on
macOS/Linux). Install `requirements.txt` and build the local index with
`python -m src.ingest` first. The first embedding-model download needs internet;
subsequent retrieval-only runs can use the cached model.

For live generation/judging, configure `GOOGLE_API_KEY` and/or `GROQ_API_KEY` in
`.env`. No provider key is needed for retrieval-only evaluation or viewing reports.

```bash
# Offline retrieval benchmark; no generation/judging requests
python -m eval.run_eval --retrieval-only --no-judge --max-api-calls 0 --label retrieval_offline

# Explicit live evaluation: capped across generation, history replay, retries,
# fallbacks and judging. A small budget normally produces a partial graph run.
python -m eval.run_eval --config graph --label graph_judged --max-api-calls 10 --sleep 25

# Generate answers without the extra LLM judge calls (still uses the API)
python -m eval.run_eval --config baseline --no-judge --limit 6 --max-api-calls 10 --sleep 25

# View reports directly, or use the Evaluation page in the main app
streamlit run pages/1_Evaluation.py

# Existing tools remain available
python -m eval.run_experiments chunking
python -m eval.run_redteam
python -m eval.failures
```

Runs save versioned JSON reports, per-question CSVs and `summary.md` under
`eval/results/`. The dashboard shows sample sizes, categories, failures, missing
scores and incomplete-budget coverage. Legacy CSV-only results are retained but
cannot provide all denominators or metadata.
For the main application, run `streamlit run app.py` and select **Evaluation** in
the sidebar. Restart Streamlit after updating this code because file watching is
disabled. New CLI-generated reports appear after refreshing/rerunning the page.

**Metric interpretation**

- Retrieval: hit rate, source-label precision/recall/F1@k, reciprocal rank,
  average precision and nDCG. These are **source-level proxies**, not passage-level
  relevance judgments or answer accuracy. Existing source labels are not an
  exhaustive relevance set; alternate valid sources can affect these estimates.
- Answers: normalized exact match and token precision/recall/F1 against available
  references. Paraphrases can score poorly despite being correct.
- LLM judgment: faithfulness, relevance and reference-based correctness estimates.
  Judge scores are not ground truth; using the generation provider as judge can
  introduce bias.
- Refusals: decision accuracy, refusal precision/recall/F1 and false-refusal rate.
  Correctly choosing to answer does **not** establish that the answer is correct.
- Citations: numeric-reference validity and overlap with labeled sources; source
  overlap alone does not prove that an individual claim is supported.
- Operations: errors, completion/scoring coverage, latency percentiles and observed
  provider request counts. Failed or unassessed cases must not masquerade as success.

Six questions in `eval/qa_set.json` have reference answers curated from the local
policy documents: Economy cabin baggage, power banks, online check-in, wheelchair
requests, infants and refunds. Other references remain unannotated. Missing scores
are **Not measured**, never zero or inferred accuracy. Local references and collection
dates do not prove that a policy is current. Retrieval-only runs do not measure
generated-answer quality.
Retrieval-only runs search the question as written without replaying conversation
history. Add `--exclude-category follow_up` to benchmark standalone retrieval;
use a graph run to evaluate contextual follow-up resolution.


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
| `TEMPERATURE` | `0.0` | Reduce generation/assessment variation; does not guarantee determinism |
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
- **Offline regression tests** using monkeypatching so the graph, guardrails,
  gateway, citation handling, and UI can be exercised without live provider calls.

---

## Limitations

- This is an **unofficial demo** using publicly available documents. Always
  verify important travel information on emirates.com.
- The knowledge base was collected on a specific date and may be out of date.
- The LLM judge used for evaluation is the same model family as the generator,
  which may inflate faithfulness scores.
- Free-tier rate limits affect latency; p95 may exceed 6 s under load.

---

*Built as part of an AI Engineering capstone project — 2026.*
