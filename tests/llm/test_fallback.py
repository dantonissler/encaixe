import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel

from encaixe.llm import fallback as fb
from encaixe.llm.breaker import BreakerRegistry


async def test_uses_fallback_when_primary_fails(settings, monkeypatch):
    def boom(messages, info):
        raise ModelHTTPError(503, "primary")

    primary = FunctionModel(boom)
    real = fb.build_model
    monkeypatch.setattr(fb, "build_model", lambda s, st: primary if s == "bad" else real(s, st))
    reg = BreakerRegistry()
    model, eff = fb.build_chain("bad", ["test"], settings, reg)
    assert eff == ["bad", "test"]
    result = await Agent(model).run("oi")
    assert result.output
    assert reg.get("bad").failures == 1
    assert reg.get("test").state == "closed"


def test_open_breaker_removes_primary_from_chain(settings):
    reg = BreakerRegistry()
    for _ in range(3):
        reg.get("ollama:a").record_failure()
    _, eff = fb.build_chain("ollama:a", ["test"], settings, reg)
    assert eff == ["test"]


def test_all_open_keeps_last(settings):
    reg = BreakerRegistry()
    for s in ("ollama:a", "ollama:b"):
        for _ in range(3):
            reg.get(s).record_failure()
    _, eff = fb.build_chain("ollama:a", ["ollama:b"], settings, reg)
    assert eff == ["ollama:b"]


from pydantic_ai.models.test import TestModel

from encaixe.llm.breaker import CircuitBreaker


def _open_registry(spec, now):
    reg = BreakerRegistry(clock=lambda: now[0])
    for _ in range(3):
        reg.get(spec).record_failure()
    return reg


def test_primary_returns_to_chain_after_reset(settings):
    now = [0.0]
    reg = _open_registry("ollama:a", now)
    assert fb.build_chain("ollama:a", ["test"], settings, reg)[1] == ["test"]
    now[0] = 31.0
    assert fb.build_chain("ollama:a", ["test"], settings, reg)[1] == ["ollama:a", "test"]


def test_uncalled_half_open_primary_stays_in_chain(settings):
    now = [0.0]
    reg = _open_registry("ollama:a", now)
    now[0] = 31.0
    for _ in range(3):  # build_chain sozinho nunca prende o breaker em half_open
        assert "ollama:a" in fb.build_chain("ollama:a", ["test"], settings, reg)[1]
    assert reg.get("ollama:a").state == "open"
    reg.get("ollama:a").try_acquire()  # teste reivindicado e nunca concluído
    assert not reg.get("ollama:a").available()
    now[0] = 62.0
    assert "ollama:a" in fb.build_chain("ollama:a", ["test"], settings, reg)[1]


async def test_half_open_trial_success_closes_and_failure_reopens():
    now = [0.0]
    b = CircuitBreaker(clock=lambda: now[0])
    for _ in range(3):
        b.record_failure()
    now[0] = 31.0
    bad = fb.BreakerModel(FunctionModel(lambda m, i: (_ for _ in ()).throw(ModelHTTPError(503, "x"))), b)
    with pytest.raises(ModelHTTPError):
        await Agent(bad).run("oi")
    assert b.state == "open"
    now[0] = 62.0
    good = fb.BreakerModel(TestModel(), b)
    await Agent(good).run("oi")
    assert b.state == "closed"


async def test_stream_success_entry_failure_and_body_error():
    b = CircuitBreaker()
    ok = fb.BreakerModel(TestModel(), b)
    async with Agent(ok).run_stream("oi") as r:
        await r.get_output()
    assert b.state == "closed" and b.failures == 0

    async def boom(messages, info):
        raise ModelHTTPError(503, "x")
        yield ""  # pragma: no cover

    b2 = CircuitBreaker()
    bad = fb.BreakerModel(FunctionModel(stream_function=boom), b2)
    with pytest.raises(ModelHTTPError):
        async with Agent(bad).run_stream("oi") as r:
            await r.get_output()
    assert b2.failures == 1

    b3 = CircuitBreaker()
    ok3 = fb.BreakerModel(TestModel(), b3)
    with pytest.raises(ValueError):
        async with ok3.request_stream(
            [ModelRequest.user_text_prompt("oi")], None, ModelRequestParameters()
        ):
            raise ValueError("consumidor")
    assert b3.failures == 0
