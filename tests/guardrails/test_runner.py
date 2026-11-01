
from encaixe.core.registry import register
from encaixe.core.types import GuardResult
from encaixe.guardrails import runner
from encaixe.output.schemas import RespostaBase

CALLS: list[str] = []


@register("guard", "t_boom")
def _boom(**_):
    class G:
        async def check(self, text, ctx):
            raise RuntimeError("x")

    return G()


@register("guard", "t_block")
def _blk(**_):
    class G:
        async def check(self, text, ctx):
            CALLS.append("block")
            return GuardResult("t_block", "block", "r")

    return G()


@register("guard", "t_after")
def _after(**_):
    class G:
        async def check(self, text, ctx):
            CALLS.append("after")
            return GuardResult("t_after", "pass")

    return G()


async def test_fail_closed_on_exception(settings, make_ctx):
    res = await runner.run_guards(["t_boom"], "x", make_ctx(), settings=settings, fail_closed=True)
    assert res == [GuardResult("t_boom", "block", "unavailable")]
    assert runner.blocked(res) is res[0]


async def test_fail_open_when_configured(settings, make_ctx):
    res = await runner.run_guards(["t_boom"], "x", make_ctx(), settings=settings, fail_closed=False)
    assert res[0].verdict == "pass" and runner.blocked(res) is None


async def test_stops_at_first_block(settings, make_ctx):
    CALLS.clear()
    res = await runner.run_guards(
        ["t_block", "t_after"], "x", make_ctx(), settings=settings, fail_closed=True
    )
    assert len(res) == 1 and CALLS == ["block"]


async def test_output_guards_use_citations_and_text(settings, make_ctx):
    out = RespostaBase(resposta="oi", citacoes=["zz"])
    res = await runner.run_output_guards(
        ["canary", "citations"], out, make_ctx(), settings=settings, fail_closed=True
    )
    assert [r.guard for r in res] == ["canary", "citations"] and res[-1].verdict == "block"
