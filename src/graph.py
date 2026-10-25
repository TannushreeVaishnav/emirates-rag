"""
graph.py — LangGraph agent: the main brain of the chatbot.

This is the most complex module in the project. It implements an 11-node
state machine that processes each user message through:

  guardrail → router → (policy path)
                            retrieve → grade → generate → output_check
                                         └→ rewrite → retrieve (max 1 retry)
                       → flight_status_tool
                       → chitchat_reply
                       → refuse

Why LangGraph instead of a plain chain?
  - Conditional edges: I can route policy questions through retrieval while
    chitchat goes directly to a canned response, without awkward if-else in
    the chain itself.
  - Cycles: the rewrite-and-retrieve loop can't be expressed as a DAG.
  - MemorySaver: each thread_id gets its own memory slot, giving the chatbot
    multi-turn context with zero extra code.
  - Observability: every step's state is recorded in the checkpointer, which
    makes debugging much easier than a flat chain.

Each node is a plain Python function that takes a GraphState dict and returns
a partial dict of updates. LangGraph merges updates into the state after each
node. The `messages` field uses add_messages to append rather than replace.
"""
import logging
import re
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.config import settings
from src.guardrails import MAX_INPUT_CHARS, check_output_rules, detect_injection, mask_pii
from src.llm_gateway import AllProvidersFailed, get_fast_llm, get_llm, start_tracking
from src.logging_utils import hash_id, log_request
from src.prompts import (
    CHECK_PROMPT,
    CHITCHAT_MSG,
    GENERATION_PROMPT,
    GRADER_PROMPT,
    NO_ANSWER_MSG,
    OUT_OF_SCOPE_MSG,
    REWRITE_PROMPT,
    ROUTER_PROMPT,
    SERVICE_MSG,
)
from src.rag_chain import build_citations, format_context
from src.retriever import retrieve
from src.tools import flight_status_reply

log = logging.getLogger("graph")
# All valid route labels the router can emit; anything else defaults to "policy".
ROUTES = {"policy", "flight_status", "chitchat", "out_of_scope"}


class GraphState(TypedDict, total=False):
    """
    The shared state dict that flows through every node.

    total=False means every key is optional, so nodes only need to return
    the keys they actually update. LangGraph merges each return dict into
    the existing state.
    """
    messages: Annotated[list[AnyMessage], add_messages]  # full conversation history
    question: str               # current (standalone, possibly rewritten) query
    masked_question: str        # PII-masked version, safe for logs and the checkpointer
    route: str                  # policy | flight_status | chitchat | out_of_scope
    docs: list                  # raw retrieved Documents (before grading)
    context_docs: list          # docs that survived grading (used for generation)
    docs_relevant: bool         # True if at least one graded doc is relevant
    retries: int                # how many times the query has been rewritten so far
    answer: str                 # final generated text
    citations: list[dict]       # structured citations derived from the answer
    supported: bool             # True if the output check passed
    provider_used: str          # which LLM provider gave the final answer
    pii_types: list[str]        # PII types found and masked (logged, never stored raw)
    blocked_reason: str | None  # set by guardrail or router if the message is blocked


# Module-level singletons for the LLM gateways.
# Lazy-initialised so the models don't load until the first request,
# which keeps import time fast during testing.
_llm = None
_fast_llm = None


def llm():
    """Return the main LLM gateway (Gemini → Groq failover)."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def fast_llm():
    """
    Return a fast helper gateway for low-complexity nodes (router, grader, rewriter).

    If FAST_MODEL is set in .env, uses a small Groq model that responds faster.
    Falls back to the main gateway when FAST_MODEL is empty.
    """
    global _fast_llm
    if settings.fast_model and settings.groq_api_key:
        if _fast_llm is None:
            _fast_llm = get_fast_llm()
        return _fast_llm
    return llm()


def _final(text: str) -> dict:
    """Convenience helper: create a terminal state update with just an answer string."""
    return {"answer": text, "citations": [], "messages": [AIMessage(content=text)]}


def _history(messages: list, limit: int = 6) -> str:
    """
    Format recent conversation history as a compact text block for the router prompt.

    Only the last `limit` turns are included to avoid blowing up the context
    window for long conversations. 6 turns is usually enough for follow-up resolution.
    """
    lines = []
    for m in messages[-limit:]:
        who = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{who}: {str(m.content)[:1200]}")
    return "\n".join(lines) or "(none)"


# ── nodes ──────────────────────────────────────────────────────────────────────

def guardrail_node(state: GraphState) -> dict:
    """
    Node 1: Input validation — runs before any LLM call.

    Three checks in order:
      1. Length limit: reject very long inputs that may be automated attacks.
      2. Regex injection detection: catches most known jailbreak patterns.
      3. PII masking: replaces emails, card numbers etc. with [LABEL] tokens.

    The masked question replaces the raw message in the LangGraph state so PII
    is never written to the MemorySaver checkpointer or any log file.
    """
    last = state["messages"][-1]
    text = last.content if isinstance(last.content, str) else str(last.content)

    masked, found = mask_pii(text)
    blocked = None
    if len(text) > MAX_INPUT_CHARS:
        blocked = "too_long"
    elif detect_injection(text):
        blocked = "prompt_injection"

    update: dict = {
        "question": masked,
        "masked_question": masked,
        "blocked_reason": blocked,
        "pii_types": sorted(set(found)),
    }
    if masked != text:
        # Replace the raw message in state so PII never reaches the checkpointer.
        update["messages"] = [HumanMessage(content=masked, id=last.id)]
    return update


def router_node(state: GraphState) -> dict:
    """
    Node 2: Classify the question and resolve follow-ups.

    The LLM returns a JSON object with:
      - route: one of {policy, flight_status, chitchat, out_of_scope}
      - standalone_question: the question rewritten to be self-contained
        (e.g. "What about Business Class?" → "What is the cabin baggage
         allowance for Business Class?")
      - attack: True if the message looks like a prompt-injection attempt
        that the regex guardrail missed (e.g. written in another language).

    Defaults to "policy" on parse errors so a broken router doesn't
    permanently block legitimate questions.
    """
    question = state["question"]
    try:
        out = (ROUTER_PROMPT | fast_llm() | JsonOutputParser()).invoke(
            {"history": _history(state["messages"][:-1]), "question": question}
        )
        if not isinstance(out, dict):
            raise ValueError("router returned non-dict")
        route = out.get("route")
        standalone = (out.get("standalone_question") or question).strip()
        attack = bool(out.get("attack", False))
        if route not in ROUTES:
            route = "policy"
    except Exception as e:
        log.warning("router failed (%s); defaulting to policy", str(e)[:80])
        route, standalone, attack = "policy", question, False

    if attack:
        # LLM-level injection detection caught something the regex missed
        return {
            "route": "out_of_scope",
            "question": standalone,
            "blocked_reason": "prompt_injection_llm",
        }
    return {"route": route, "question": standalone}


def retrieve_node(state: GraphState) -> dict:
    """
    Node 3: Retrieve relevant document chunks from the FAISS index.

    Also scans each retrieved chunk for injection text. This guards against
    "document-side injection" where an attacker embeds instructions in a web
    page that ends up in the knowledge base.
    """
    docs = retrieve(state["question"])
    # Drop any retrieved chunk that contains injection text (document-side injection)
    clean = [d for d in docs if not detect_injection(d.page_content)]
    if len(clean) < len(docs):
        log.warning(
            "dropped %d retrieved chunk(s) that look like prompt injection",
            len(docs) - len(clean),
        )
    return {"docs": clean}


def grade_node(state: GraphState) -> dict:
    """
    Node 4: Filter retrieved chunks to only the relevant ones.

    The grader sends all retrieved passages to the LLM and asks it to return
    a list of passage numbers worth keeping. This step removes noise (e.g. a
    baggage-weight chunk retrieved for a check-in question) before generation.

    Gated on settings.use_grader so it can be disabled for ablation studies.
    If grading fails, fail closed: never treat unassessed passages as reliable
    evidence for a policy answer.
    """
    docs = state.get("docs", [])
    if not docs:
        return {"docs_relevant": False, "context_docs": []}
    if not settings.use_grader:
        return {"docs_relevant": True, "context_docs": docs}

    passages = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1))
    try:
        out = (GRADER_PROMPT | fast_llm() | JsonOutputParser()).invoke(
            {"question": state["question"], "passages": passages}
        )
        keep_ids = {int(n) for n in out.get("relevant", []) if str(n).strip().isdigit()}
        keep = [d for i, d in enumerate(docs, start=1) if i in keep_ids]
    except Exception as e:
        log.warning("grader failed (%s); treating evidence as insufficient", str(e)[:80])
        keep = []
    return {"docs_relevant": bool(keep), "context_docs": keep}


def rewrite_node(state: GraphState) -> dict:
    """
    Node 5: Rewrite the query and increment the retry counter.

    Triggered when the grader returns no relevant docs. A more specific
    question often retrieves better chunks — for example, "baggage allowance"
    might become "hand luggage weight limit Economy class".

    The retry counter is checked by after_grade() to enforce max_rewrites.
    """
    new_q = (REWRITE_PROMPT | fast_llm() | StrOutputParser()).invoke({"question": state["question"]})
    return {"question": new_q.strip().strip('"'), "retries": state.get("retries", 0) + 1}


def generate_node(state: GraphState) -> dict:
    """
    Node 6: Generate the grounded answer with inline citations.

    Uses the main LLM (not the fast helper) because answer quality matters here.
    The generation prompt instructs the model to cite passages by number [1][2]
    and to return exactly NO_ANSWER_MSG if the context doesn't contain the answer.
    """
    docs = state["context_docs"]
    answer = (GENERATION_PROMPT | llm() | StrOutputParser()).invoke(
        {"context": format_context(docs), "question": state["question"]}
    ).strip()
    refused = answer.startswith(NO_ANSWER_MSG[:40])
    return {"answer": answer, "citations": [] if refused else build_citations(docs, answer)}


def output_check_node(state: GraphState) -> dict:
    """
    Node 7: Verify the generated answer is actually grounded in the context.

    Two-stage check:
      1. Deterministic rules (check_output_rules): catches invented amounts,
         guarantee claims, and prompt-leakage. Cheap and always runs.
      2. LLM check (CHECK_PROMPT): asks the model whether every factual claim
         in the answer is supported by the retrieved passages. Only runs if
         the deterministic rules pass AND settings.use_output_check is True.

    If either check fails, after_check() routes to refuse_no_answer.
    """
    answer = state["answer"]
    ok = True
    if not answer.startswith(NO_ANSWER_MSG[:40]):
        context = "\n\n".join(d.page_content for d in state["context_docs"])
        # Step 1: fast deterministic rules (no LLM call needed)
        violations = check_output_rules(answer, context)
        references = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
        if not references or any(n < 1 or n > len(state["context_docs"]) for n in references):
            violations.append("missing_or_invalid_citations")
        if violations:
            log.warning("output rules violated: %s", violations)
            ok = False
        elif settings.use_output_check:
            # Step 2: LLM-based faithfulness check
            try:
                out = (CHECK_PROMPT | llm() | JsonOutputParser()).invoke(
                    {"context": context, "answer": answer}
                )
                ok = out.get("supported") is True
            except Exception as e:
                log.warning("output check failed (%s); evidence unverified", str(e)[:80])
                ok = False
    update: dict = {"supported": ok}
    if ok:
        update["messages"] = [AIMessage(content=answer)]
    return update


# ── Terminal nodes (no LLM call) ──────────────────────────────────────────────────

def flight_status_node(state: GraphState) -> dict:
    return _final(flight_status_reply(state["question"]))


def chitchat_node(state: GraphState) -> dict:
    return _final(CHITCHAT_MSG)


def refuse_node(state: GraphState) -> dict:
    return _final(OUT_OF_SCOPE_MSG)


def refuse_no_answer_node(state: GraphState) -> dict:
    return _final(NO_ANSWER_MSG)


# ── edges ──────────────────────────────────────────────────────────────────────

def after_guardrail(state: GraphState) -> str:
    return "refuse" if state.get("blocked_reason") else "router"


def after_router(state: GraphState) -> str:
    return {
        "policy": "retrieve",
        "flight_status": "flight_status_tool",
        "chitchat": "chitchat_reply",
    }.get(state["route"], "refuse")


def after_grade(state: GraphState) -> str:
    if state["docs_relevant"]:
        return "generate"
    return "rewrite" if state.get("retries", 0) < settings.max_rewrites else "refuse_no_answer"


def after_check(state: GraphState) -> str:
    return "end" if state.get("supported", True) else "refuse_no_answer"


# ── build ──────────────────────────────────────────────────────────────────────

def build_graph():
    b = StateGraph(GraphState)

    b.add_node("guardrail", guardrail_node)
    b.add_node("router", router_node)
    b.add_node("retrieve", retrieve_node)
    b.add_node("grade", grade_node)
    b.add_node("rewrite", rewrite_node)
    b.add_node("generate", generate_node)
    b.add_node("output_check", output_check_node)
    b.add_node("flight_status_tool", flight_status_node)
    b.add_node("chitchat_reply", chitchat_node)
    b.add_node("refuse", refuse_node)
    b.add_node("refuse_no_answer", refuse_no_answer_node)

    b.add_edge(START, "guardrail")
    b.add_conditional_edges(
        "guardrail", after_guardrail, {"refuse": "refuse", "router": "router"}
    )
    b.add_conditional_edges(
        "router", after_router, {
            "retrieve": "retrieve",
            "flight_status_tool": "flight_status_tool",
            "chitchat_reply": "chitchat_reply",
            "refuse": "refuse",
        },
    )
    b.add_edge("retrieve", "grade")
    b.add_conditional_edges(
        "grade", after_grade, {
            "generate": "generate",
            "rewrite": "rewrite",
            "refuse_no_answer": "refuse_no_answer",
        },
    )
    b.add_edge("rewrite", "retrieve")
    b.add_edge("generate", "output_check")
    b.add_conditional_edges(
        "output_check", after_check, {"end": END, "refuse_no_answer": "refuse_no_answer"}
    )
    for node in ("flight_status_tool", "chitchat_reply", "refuse", "refuse_no_answer"):
        b.add_edge(node, END)

    return b.compile(checkpointer=MemorySaver())


graph = build_graph()

_REFUSAL_MARKERS = (
    "my confidence is low",
    "i couldn't find",
    "i could not find",
    "i can help with emirates travel policies",
    "please check emirates.com",
)


def _is_refusal(answer: str) -> bool:
    a = answer.strip().lower().replace("\u2019", "'")
    return any(m in a[:200] for m in _REFUSAL_MARKERS)


def chat(message: str, thread_id: str = "default") -> dict:
    """
    Public API: process one conversational turn and return a result dict.

    Args:
        message:   The raw user message (may contain PII; the guardrail masks it).
        thread_id: Identifies the conversation. The same thread_id reuses
                   MemorySaver state so follow-up questions have context.

    Returns a dict with:
        answer         — the final answer string
        citations      — list of source citation dicts
        route          — the route chosen by the router node
        retries        — number of query rewrites performed
        refused        — True if the question was refused for any reason
        provider_used  — which LLM provider gave the final answer
        llm_calls      — total LLM calls made (useful for latency analysis)
        llm_seconds    — total LLM wall-clock time
        latency        — end-to-end wall-clock seconds
        error          — error string if something failed, otherwise None
    """
    t0 = time.perf_counter()
    tracker = start_tracking()  # fresh tracker for this turn; attached to context var
    error = None
    try:
        state = graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                # per-turn resets (checkpointer keeps everything else between turns)
                "question": "",
                "route": "",
                "docs": [],
                "context_docs": [],
                "docs_relevant": False,
                "retries": 0,
                "answer": "",
                "citations": [],
                "supported": True,
                "blocked_reason": None,
                "pii_types": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except AllProvidersFailed as e:
        log.error("%s", e)
        state, error = {"answer": SERVICE_MSG}, "llm_unavailable"

    latency = time.perf_counter() - t0
    answer = state.get("answer", "")
    result = {
        "answer": answer,
        "citations": state.get("citations", []),
        "route": state.get("route", ""),
        "retries": state.get("retries", 0),
        "blocked_reason": state.get("blocked_reason"),
        "pii_types": state.get("pii_types", []),
        "docs": state.get("docs", []),
        "context_docs": state.get("context_docs", []),
        "masked_question": state.get("masked_question", ""),
        "llm_seconds": round(sum(c.seconds for c in tracker.calls), 2),
        "refused": (
            bool(state.get("blocked_reason"))
            or state.get("route") == "out_of_scope"
            or _is_refusal(answer)
        ),
        "provider_used": tracker.last_ok,
        "llm_calls": len(tracker.calls),
        "llm_failures": tracker.failures,
        "error": error,
        "latency": latency,
    }
    log.info(
        "route=%s retries=%d blocked=%s provider=%s latency=%.1fs",
        result["route"],
        result["retries"],
        result["blocked_reason"],
        result["provider_used"],
        latency,
    )
    log_request({
        "session": hash_id(thread_id),
        "question_masked": state.get("masked_question", "")[:200],
        "route": result["route"],
        "blocked_reason": result["blocked_reason"],
        "pii_types": result["pii_types"],
        "retries": result["retries"],
        "sources": sorted({d.metadata["source"] for d in result["context_docs"]}),
        "provider_used": result["provider_used"],
        "llm_calls": result["llm_calls"],
        "llm_failures": result["llm_failures"],
        "refused": result["refused"],
        "error": error,
        "latency_s": round(latency, 2),
    })
    return result
