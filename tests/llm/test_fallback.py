from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
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
    assert reg.get("bad")._failures == 1
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
