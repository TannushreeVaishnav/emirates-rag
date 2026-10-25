"""
tests/test_gateway.py — Unit tests for the LLM failover gateway.

Coverage includes:
  - Successful call records the provider and elapsed time in UsageTracker
  - Transient errors (500) trigger exponential backoff and then succeed
  - Rate-limit errors (429) fail over to the next provider immediately
  - Fatal errors (auth) also fail over without retrying
  - AllProvidersFailed is raised when every provider is exhausted
  - FORCE_PRIMARY_FAIL=true forces the failover path end-to-end
  - ContextVar tracking works inside LangChain pipe chains (prompt | gateway | parser)
  - Rate-limited endpoints stay suppressed across stages until a new turn
  - Groq main/fast models have distinct identities, but shared models do not
  - Suppressed exhaustion still raises; transient and fatal failures are not suppressed

All tests run offline with no API calls.
"""
import pytest
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from src.llm_gateway import AllProvidersFailed, LLMGateway, start_tracking


class Fake:
    """Stand-in chat model. Outcomes are consumed in order; the last one repeats."""

    def __init__(self, *outcomes, model_name=None):
        self.outcomes, self.calls = list(outcomes), 0
        self.model_name = model_name

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
    p = Fake(Exception("503 unavailable"))
    f = Fake(Exception("503 unavailable"))
    start_tracking()
    g = gateway(p, f)
    for _ in range(2):
        with pytest.raises(AllProvidersFailed):
            g.invoke("q")
    # Retries remain bounded per invocation; transient failures aren't suppressed.
    assert p.calls == f.calls == 4


def test_forced_primary_failure():
    p, f, t = Fake("hi"), Fake("fb"), start_tracking()
    g = gateway(p, f, force_primary_fail=True)
    assert g.invoke("q").content == "fb"
    assert g.invoke("another stage").content == "fb"
    assert p.calls == 0 and f.calls == 2 and t.last_ok == "groq"
    assert t.failures == 1


def test_works_inside_a_chain():
    start_tracking()
    assert (gateway(Fake("hi"), Fake("fb")) | StrOutputParser()).invoke("q") == "hi"


def test_rate_limit_suppresses_endpoint_across_gateway_invocations():
    p = Fake(Exception("429 Too Many Requests"), model_name="gemini-main")
    same_endpoint = Fake("must not be called", model_name="gemini-main")
    f = Fake("fallback")
    tracker = start_tracking()

    assert gateway(p, f).invoke("first stage").content == "fallback"
    assert gateway(same_endpoint, f, role="fast").invoke("next stage").content == "fallback"

    assert p.calls == 1 and same_endpoint.calls == 0 and f.calls == 2
    assert tracker.failures == 1 and len(tracker.calls) == 3


def test_new_turn_resets_rate_limit_suppression():
    p = Fake(Exception("429 Too Many Requests"), "recovered")
    f = Fake("fallback")
    g = gateway(p, f)
    previous_turn = start_tracking()

    assert g.invoke("first stage").content == "fallback"
    assert g.invoke("next stage").content == "fallback"
    assert p.calls == 1

    new_turn = start_tracking()
    assert g.invoke("new turn").content == "recovered"
    assert p.calls == 2 and f.calls == 2
    assert new_turn is not previous_turn
    assert new_turn.failures == 0 and new_turn.last_ok == "gemini"
    assert previous_turn.failures == 1 and len(previous_turn.calls) == 3


@pytest.mark.parametrize(
    ("limited_name", "limited_model", "healthy_name", "healthy_model"),
    [
        ("groq-fast", "small-model", "groq", "large-model"),
        ("groq", "large-model", "groq-fast", "small-model"),
        ("groq", "small-model", "groq", "large-model"),
    ],
)
def test_rate_limit_does_not_suppress_other_groq_models(
    limited_name, limited_model, healthy_name, healthy_model
):
    limited = Fake(Exception("429 Too Many Requests"), model_name=limited_model)
    healthy = Fake("answer", model_name=healthy_model)
    limited_gateway = LLMGateway([(limited_name, limited)], base_delay=0)
    healthy_gateway = LLMGateway([(healthy_name, healthy)], base_delay=0)
    tracker = start_tracking()

    with pytest.raises(AllProvidersFailed):
        limited_gateway.invoke("limited stage")
    assert healthy_gateway.invoke("other model").content == "answer"
    with pytest.raises(AllProvidersFailed):
        limited_gateway.invoke("limited again")

    assert limited.calls == healthy.calls == 1
    assert tracker.failures == 1


def test_same_groq_endpoint_is_suppressed_across_main_and_fast_roles():
    limited = Fake(Exception("429 Too Many Requests"), model_name="shared-model")
    same_endpoint = Fake("must not be called", model_name="shared-model")
    main = LLMGateway([("groq", limited)], base_delay=0)
    fast = LLMGateway([("groq-fast", same_endpoint)], base_delay=0, role="fast")
    start_tracking()

    with pytest.raises(AllProvidersFailed):
        main.invoke("main stage")
    with pytest.raises(AllProvidersFailed):
        fast.invoke("helper stage")
    assert limited.calls == 1 and same_endpoint.calls == 0


def test_both_rate_limited_providers_stay_exhausted_until_new_turn():
    p = Fake(Exception("429 Gemini quota"), "recovered")
    f = Fake(Exception("429 Groq quota"))
    g = gateway(p, f)
    tracker = start_tracking()

    for _ in range(2):
        with pytest.raises(AllProvidersFailed, match="All providers failed") as failure:
            g.invoke("q")
        assert failure.value.__cause__ is f.outcomes[0]

    assert p.calls == f.calls == 1
    assert tracker.failures == 2 and tracker.last_ok == ""
    start_tracking()
    assert g.invoke("next turn").content == "recovered"
    assert p.calls == 2 and f.calls == 1


def test_fatal_error_is_not_suppressed_for_later_invocations():
    p = Fake(ValueError("invalid api key"))
    f = Fake("fallback")
    g = gateway(p, f)
    tracker = start_tracking()

    assert g.invoke("first stage").content == "fallback"
    assert g.invoke("next stage").content == "fallback"
    assert p.calls == f.calls == tracker.failures == 2
