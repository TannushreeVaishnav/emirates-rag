"""
llm_gateway.py — Fault-tolerant LLM provider gateway.

The free-tier APIs I'm using (Gemini, Groq) have strict rate limits and
occasional outages. This module wraps both providers in a single LangChain
Runnable so the rest of the code never has to think about failover.

Design:
  - Primary provider (Gemini) is tried first.
  - On rate-limit (429) or fatal errors, it fails over to Groq immediately
    without retrying — retrying an exhausted quota just wastes time.
  - On transient errors (5xx, timeout, connection reset), it retries with
    exponential backoff before failing over.
  - A ContextVar (UsageTracker) records every LLM call made during a single
    request, including which provider answered and how long it took. LangGraph
    copies the context into worker threads, so calls made inside nodes are
    still recorded.
  - FORCE_PRIMARY_FAIL=true is a test switch that simulates a Gemini outage
    to verify the fallback path works end-to-end.
  - get_fast_llm() returns a gateway that tries a small, fast Groq model first
    for low-complexity helper calls (router, grader, rewriter), then falls back
    to the normal providers. It's gated on the FAST_MODEL setting.
"""

import contextvars
import logging
import random
import time
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from src.config import settings

log = logging.getLogger("gateway")


class AllProvidersFailed(RuntimeError):
    """Raised when every configured provider has failed."""
    pass


@dataclass
class CallRecord:
    """Stores the outcome of one LLM call attempt."""
    provider: str   # "gemini", "groq", "groq-fast"
    ok: bool        # True if the call succeeded
    seconds: float  # wall-clock time for the call
    error: str = ""
    role: str = "main"  # "main" or "fast"; used to track which provider answered the user


class UsageTracker:
    """
    Collects every LLM call made during one request/turn.

    Stored in a ContextVar so each LangGraph invocation gets its own tracker
    without global state conflicts. The tracker is read back in chat() to
    populate the response dict and the request log.
    """

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    @property
    def last_ok(self) -> str:
        """The provider that gave the last successful *main* answer (not a helper call)."""
        for c in reversed(self.calls):
            if c.ok and c.role == "main":
                return c.provider
        return ""

    @property
    def failures(self) -> int:
        """Total number of failed call attempts (across all providers)."""
        return sum(1 for c in self.calls if not c.ok)


# ContextVar: set at the start of each chat() call, read inside graph nodes.
_current: contextvars.ContextVar[UsageTracker | None] = contextvars.ContextVar(
    "tracker", default=None
)


def start_tracking() -> UsageTracker:
    """Create a fresh tracker for this request and install it in the ContextVar."""
    tracker = UsageTracker()
    _current.set(tracker)
    return tracker


def _http_status(exc: Exception) -> int | None:
    """Extract an HTTP status code from various exception types."""
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return getattr(getattr(exc, "response", None), "status_code", None)


def classify(exc: Exception) -> str:
    """
    Categorise an exception so the retry logic knows what to do.

    Returns:
        "rate_limit"  → fail over immediately (retrying a 429 wastes quota)
        "transient"   → retry with backoff, then fail over
        "fatal"       → fail over immediately (auth error, bad request, etc.)
    """
    status = _http_status(exc)
    msg = str(exc).lower()

    if status == 429 or any(k in msg for k in ("429", "rate limit", "quota", "resource_exhausted")):
        return "rate_limit"
    if (
        status in (500, 502, 503, 504)
        or isinstance(exc, (TimeoutError, ConnectionError))
        or any(k in msg for k in ("timeout", "timed out", "unavailable", "overloaded", "connection"))
    ):
        return "transient"
    return "fatal"


class LLMGateway(Runnable[LanguageModelInput, BaseMessage]):
    """
    A LangChain Runnable that wraps multiple providers and handles failover.

    Usage (just like any LangChain chat model):
        result = gateway.invoke("your prompt")
        chain  = prompt | gateway | parser
    """

    def __init__(
        self,
        providers: list,     # list of (name: str, model: BaseChatModel)
        attempts: int = 2,
        base_delay: float = 1.0,
        force_primary_fail: bool = False,
        role: str = "main",
    ) -> None:
        self.providers = providers
        self.attempts = attempts
        self.base_delay = base_delay
        self.force_primary_fail = force_primary_fail
        self.role = role  # "main" or "fast" — recorded in each CallRecord

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> BaseMessage:
        """
        Try each provider in order. Retry transient errors, fail over on others.
        Raises AllProvidersFailed if every provider has been exhausted.
        """
        tracker = _current.get()
        last_err: Exception | None = None

        for idx, (name, model) in enumerate(self.providers):
            for attempt in range(1, self.attempts + 1):
                t0 = time.perf_counter()
                try:
                    # Test switch: simulate primary outage to verify Groq fallback
                    if idx == 0 and self.force_primary_fail:
                        raise RuntimeError("429 forced primary failure (FORCE_PRIMARY_FAIL=true)")

                    out = model.invoke(input, config)
                    if tracker:
                        tracker.calls.append(
                            CallRecord(name, True, time.perf_counter() - t0, role=self.role)
                        )
                    return out

                except Exception as e:
                    kind = classify(e)
                    last_err = e
                    if tracker:
                        tracker.calls.append(
                            CallRecord(
                                name, False,
                                time.perf_counter() - t0,
                                f"{kind}: {str(e)[:120]}",
                                role=self.role,
                            )
                        )
                    log.warning(
                        "provider=%s attempt=%d kind=%s error=%s",
                        name, attempt, kind, str(e)[:120],
                    )
                    # Only transient errors are worth retrying; everything else → fail over
                    if kind != "transient" or attempt == self.attempts:
                        break
                    # Exponential backoff with jitter
                    time.sleep(
                        self.base_delay * 2 ** (attempt - 1)
                        + random.uniform(0, 0.25 * self.base_delay)
                    )

        raise AllProvidersFailed(
            f"All providers failed ({', '.join(n for n, _ in self.providers)}): "
            f"{str(last_err)[:150]}"
        ) from last_err



# ── Builder functions ─────────────────────────────────────────────────────────

def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.temperature,
        google_api_key=settings.google_api_key,
        max_retries=0,    # retries are handled by LLMGateway, not the client
        timeout=settings.llm_timeout,
    )


def _build_groq():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.temperature,
        api_key=settings.groq_api_key,
        max_retries=0,
        timeout=settings.llm_timeout,
    )


_BUILDERS = {"gemini": _build_gemini, "groq": _build_groq}
_API_KEYS = {
    "gemini": lambda: settings.google_api_key,
    "groq":   lambda: settings.groq_api_key,
}


@lru_cache(maxsize=1)
def get_llm() -> LLMGateway:
    """
    Build and cache the main LLM gateway (Gemini → Groq).

    lru_cache means the models are instantiated once and reused. The provider
    order follows primary_provider / fallback_provider in settings.
    """
    providers = []
    for name in dict.fromkeys([settings.primary_provider, settings.fallback_provider]):
        if name not in _BUILDERS:
            log.warning("Unknown provider %r — skipped", name)
        elif not _API_KEYS[name]():
            log.warning("No API key for %s — skipped", name)
        else:
            providers.append((name, _BUILDERS[name]()))

    if not providers:
        raise RuntimeError("No LLM providers configured. Add API keys to .env")

    return LLMGateway(
        providers,
        attempts=settings.llm_attempts,
        force_primary_fail=settings.force_primary_fail,
        role="main",
    )


@lru_cache(maxsize=1)
def get_fast_llm() -> LLMGateway:
    """
    Build and cache a fast-helper gateway for low-complexity tasks.

    Puts a small, fast Groq model (FAST_MODEL in .env) at the front of the
    provider list so router / grader / rewriter calls return faster.
    Falls back to the normal providers if the fast model also fails.
    If FAST_MODEL is empty, this just returns the main gateway.
    """
    if not settings.fast_model:
        return get_llm()

    from langchain_groq import ChatGroq
    fast_model = ChatGroq(
        model=settings.fast_model,
        temperature=0,    # classification tasks don't need creativity
        api_key=settings.groq_api_key,
        max_retries=0,
        timeout=settings.llm_timeout,
    )
    return LLMGateway(
        [("groq-fast", fast_model), *get_llm().providers],
        attempts=settings.llm_attempts,
        role="fast",
    )
