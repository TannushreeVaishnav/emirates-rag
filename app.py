"""
app.py — Streamlit chat UI for the Emirates Policy Assistant.

This is the entry point users interact with. It wraps the LangGraph agent
(src/graph.py) in a web UI with:
  - Multi-turn chat with session memory (one MemorySaver thread per browser tab)
  - Expandable Sources section for every answer (shows the exact passage used)
  - Thumbs up/down feedback saved to logs/feedback.jsonl
  - Sample questions in the sidebar and main area for first-time users
  - A per-session turn limit (MAX_USER_TURNS) to protect free-tier API quota
  - Friendly error messages when both LLM providers are rate-limited

Design notes:
  - @st.cache_resource on load_backend() means the FAISS index and embedding
    model are loaded only once, regardless of how many users are active.
  - The env vars at the top MUST be set before any src.* import triggers the
    embedding model, which is why they come before everything else.
  - $ signs in answers are escaped to prevent Streamlit rendering them as LaTeX.

Run:
    streamlit run app.py
"""
# ── IMPORTANT: must be set before any src.* import loads the embedding model ──
import os
os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# ───────────────────────────────────────────────────────────────────────────────

import csv
import json
import logging
import uuid
from datetime import datetime, timezone

import streamlit as st

from src.config import settings
from src.logging_utils import hash_id

st.set_page_config(
    page_title="Emirates Policy Assistant (unofficial)",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="auto",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("app")

# Shown in the sidebar and main area; updated when the providers answer.
DISCLAIMER = "Unofficial demo. Not affiliated with Emirates. Verify important details on emirates.com."
WELCOME = (
    "Hi! I answer questions about Emirates policies (baggage, check-in, refunds and changes, "
    "special assistance, travel documents) using the documents in my knowledge base, "
    "and I show the sources for every answer."
)
# Error messages — rate-limit gets a more specific hint to wait
ERROR_MSG = "Sorry, something went wrong on my side. Please try again in a moment."
RATE_LIMIT_MSG = (
    "⏳ Both AI providers are currently at their free-tier rate limit. "
    "Please wait 20–30 seconds and try again."
)
MAX_USER_TURNS = 25   # per browser session — protects free-tier API quota on a public demo

SAMPLES = [
    "What is the cabin baggage allowance for Economy?",
    "Can I carry a power bank?",
    "Can I cancel my ticket and get a refund?",
    "When does online check-in open?",
    "How do I request a wheelchair?",
    "What are the rules for infants?",
]

# Best-effort styling: full-width buttons, tighter padding on phones.
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 6rem; max-width: 760px;}
[data-testid="stSidebar"] .stButton button, .stButton button {width: 100%; text-align: left;}
@media (max-width: 640px) {.block-container {padding-left: .8rem; padding-right: .8rem;}}
</style>
""",
    unsafe_allow_html=True,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def md(text: str) -> str:
    """Escape dollar signs so prices render as text, not LaTeX."""
    return text.replace("$", r"\$")


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("thread_id", uuid.uuid4().hex)
    ss.setdefault("messages", [])
    ss.setdefault("pending", None)
    ss.setdefault("last_provider", "")


def new_chat() -> None:
    st.session_state.thread_id = uuid.uuid4().hex   # new thread = fresh LangGraph memory
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.last_provider = ""


def ask_sample(question: str) -> None:
    st.session_state.pending = question


@st.cache_data
def kb_info() -> tuple[str, int]:
    try:
        with open(settings.sources_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        dates = [r["date_collected"] for r in rows if r.get("date_collected")]
        return (max(dates) if dates else "unknown"), len(rows)
    except Exception:
        return "unknown", 0


@st.cache_resource(show_spinner="Loading the knowledge base — first run takes a moment…")
def load_backend():
    """
    Import the agent and warm up the FAISS index.

    @st.cache_resource means this runs exactly once per server process, so the
    ~90 MB embedding model stays in RAM across all user sessions.
    """
    from src.graph import chat
    from src.retriever import _store
    _store()   # warm up FAISS and the embedding model; subsequent calls use the cache
    return chat


def save_feedback(msg_id: str) -> None:
    """
    Persist a thumbs rating to logs/feedback.jsonl.

    Only the masked question (PII already removed), the answer snippet, and
    the route are logged — never the raw user text. Called by st.feedback()
    on_change, so it runs in the same thread as the widget callback.
    """
    value = st.session_state.get(f"fb_{msg_id}")   # 1 = thumbs up, 0 = thumbs down
    if value is None:
        return
    msg = next((m for m in st.session_state.messages if m["id"] == msg_id), None)
    if not msg:
        return
    try:
        settings.log_dir.mkdir(exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session": hash_id(st.session_state.thread_id),
            "message_id": msg_id,
            "rating": "up" if value == 1 else "down",
            "question_masked": msg.get("question_masked", "")[:200],
            "answer": msg["content"][:300],
            "route": msg.get("route", ""),
            "provider": msg.get("provider", ""),
            "sources": msg.get("sources", []),
        }
        with open(settings.log_dir / "feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        log.exception("could not save feedback")


def render_sources(citations: list[dict]) -> None:
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            page = f", p.{c['page']}" if c.get("page") else ""
            st.markdown(f"**[{c['ref']}]** `{c['source']}` · {md(c['section'])}{page}")
            snippet = " ".join(c["text"].split())
            if len(snippet) > 400:
                snippet = snippet[:400].rsplit(" ", 1)[0] + "…"
            st.markdown("> " + md(snippet))
            st.markdown(f"[Open original]({c['url']}) · collected {c['date_collected']}")


def render_history() -> None:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(md(m["content"]))
            if m["role"] != "assistant":
                continue
            if m.get("citations"):
                render_sources(m["citations"])
            meta = [
                x for x in (
                    m.get("provider"),
                    f"{m['latency']}s" if m.get("latency") else "",
                )
                if x
            ]
            if meta:
                st.caption(" · ".join(meta))
            if hasattr(st, "feedback") and not m.get("error"):
                st.feedback(
                    "thumbs",
                    key=f"fb_{m['id']}",
                    on_change=save_feedback,
                    args=(m["id"],),
                )


def answer_question(prompt: str) -> None:
    user_msg_id = uuid.uuid4().hex[:8]
    st.session_state.messages.append(
        {"id": user_msg_id, "role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(md(prompt))

def answer_question(prompt: str) -> None:
    user_msg_id = uuid.uuid4().hex[:8]
    st.session_state.messages.append(
        {"id": user_msg_id, "role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(md(prompt))

    res = None
    rate_limited = False
    with st.chat_message("assistant"):
        with st.spinner("Searching the Emirates documents…"):
            try:
                res = load_backend()(prompt, st.session_state.thread_id)
            except Exception as exc:
                # Detect both-providers-exhausted separately so we can give
                # a more specific "wait 30 seconds" message instead of generic error.
                from src.llm_gateway import AllProvidersFailed
                rate_limited = isinstance(exc, AllProvidersFailed)
                log.exception("chat() raised unexpectedly")

    msg: dict = {
        "id": uuid.uuid4().hex[:8],
        "role": "assistant",
        # Give a more specific message when both providers hit their free-tier quota
        "content": RATE_LIMIT_MSG if rate_limited else ERROR_MSG,
        "error": True,
    }
    if res is not None:
        msg.update(
            content=res["answer"],
            citations=res.get("citations", []),
            route=res.get("route", ""),
            provider=res.get("provider_used", ""),
            latency=round(res.get("latency", 0), 1),
            question_masked=res.get("masked_question", ""),
            sources=sorted({d.metadata["source"] for d in res.get("context_docs", [])}),
            error=bool(res.get("error")),
        )
        if res.get("provider_used"):
            st.session_state.last_provider = res["provider_used"]
    st.session_state.messages.append(msg)


# ── page ───────────────────────────────────────────────────────────────────────

init_state()

if not (settings.index_dir / "index.faiss").exists():
    st.error(
        "The knowledge base index was not found. "
        "Run `python -m src.ingest` first, then restart the app."
    )
    st.stop()

kb_date, kb_docs = kb_info()

pending = st.session_state.pending  # a sample question clicked on the previous run
st.session_state.pending = None

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.button("➕ New chat", on_click=new_chat)
    st.markdown("**Try asking**")
    for i, q in enumerate(SAMPLES):
        st.button(q, key=f"side_{i}", on_click=ask_sample, args=(q,))
    st.caption('Follow-ups work too — e.g. *"What about Business Class?"*')
    st.divider()
    st.markdown("**Knowledge base**")
    st.caption(
        f"Last updated: {kb_date}  \n"
        f"{kb_docs} documents · market {settings.market}"
    )
    st.markdown("**Model**")
    st.caption(
        f"Primary: {settings.primary_provider} ({settings.gemini_model})  \n"
        f"Fallback: {settings.fallback_provider} ({settings.groq_model})  \n"
        f"Last answer by: {st.session_state.last_provider or '–'}"
    )
    st.divider()
    st.caption(DISCLAIMER)

# ── main ───────────────────────────────────────────────────────────────────────
st.markdown("## ✈️ Emirates Policy Assistant")
st.info(DISCLAIMER)
st.caption(f"Knowledge base last updated: {kb_date}")

if not st.session_state.messages and not pending:
    with st.chat_message("assistant"):
        st.markdown(WELCOME)
    st.markdown("**Try one of these:**")
    for i, q in enumerate(SAMPLES[:4]):
        st.button(q, key=f"main_{i}", on_click=ask_sample, args=(q,))

render_history()

user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
capped = user_turns >= MAX_USER_TURNS
if capped:
    st.info(
        "This demo session has reached its limit. "
        "Click **New chat** in the sidebar to continue."
    )

typed = st.chat_input(
    "Ask about baggage, check-in, refunds, assistance…",
    disabled=capped,
)
prompt = pending or typed
if prompt and not capped:
    answer_question(prompt)
    st.rerun()   # redraw from history so Sources and feedback widgets appear uniformly
