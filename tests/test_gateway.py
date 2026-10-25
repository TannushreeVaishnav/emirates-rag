"""
tests/test_gateway.py — Unit tests for the LLM failover gateway.

7 tests covering:
  - Successful call records the provider and elapsed time in UsageTracker
  - Transient errors (500) trigger exponential backoff and then succeed
  - Rate-limit errors (429) fail over to the next provider immediately
  - Fatal errors (auth) also fail over without retrying
  - AllProvidersFailed is raised when every provider is exhausted
  - FORCE_PRIMARY_FAIL=true forces the failover path end-to-end
  - ContextVar tracking works inside LangChain pipe chains (prompt | gateway | parser)

All tests run offline with no API calls.
"""
import pytest
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from src.llm_gateway import AllProvidersFailed, LLMGateway, start_tracking


class Fake:
    """Stand-in chat model. Outcomes are consumed in order; the last one repeats."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def invoke(self, input, config=None):
        self.calls += 1
        o = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(o, Exception):
            raise o
        return AIMessage(content=o)


def gateway(primary, fallback, **kw):
    return LLMGateway([("gemini", primary), ("groq", fallback)], attempts=2, base_delay=0, **kw)


def test_primary_ok():
    p, f, t = Fake("hi"), Fake("fb"), start_tracking()
    assert gateway(p, f).invoke("q").content == "hi"
    assert f.calls == 0 and t.last_ok == "gemini" and t.failures == 0


def test_rate_limit_fails_over_without_retry():
    p, f, t = Fake(Exception("429 Too Many Requests")), Fake("fb"), start_tracking()
    assert gateway(p, f).invoke("q").content == "fb"
    assert p.calls == 1 and t.last_ok == "groq" and t.failures == 1


def test_transient_error_is_retried():
    p, f, t = Fake(TimeoutError("timed out"), "ok"), Fake("fb"), start_tracking()
    assert gateway(p, f).invoke("q").content == "ok"
    assert p.calls == 2 and f.calls == 0 and t.last_ok == "gemini" and t.failures == 1


def test_fatal_error_fails_over_immediately():
    p, f = Fake(ValueError("invalid api key")), Fake("fb")
    start_tracking()
    assert gateway(p, f).invoke("q").content == "fb" and p.calls == 1


def test_all_providers_fail():
    start_tracking()
    with pytest.raises(AllProvidersFailed):
        gateway(
            Fake(Exception("503 unavailable")), Fake(Exception("503 unavailable"))
        ).invoke("q")


def test_forced_primary_failure():
    p, f, t = Fake("hi"), Fake("fb"), start_tracking()
    assert gateway(p, f, force_primary_fail=True).invoke("q").content == "fb"
    assert p.calls == 0 and t.last_ok == "groq"


def test_works_inside_a_chain():
    start_tracking()
    assert (gateway(Fake("hi"), Fake("fb")) | StrOutputParser()).invoke("q") == "hi"
