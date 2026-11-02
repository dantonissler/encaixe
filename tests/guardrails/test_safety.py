"""Guard `content_safety` em camadas (ruling do D2 da Task 15): ShieldGemma 2B como principal
(todo veredito conta) e Llama Guard 3 1B como secundário opcional, cujo "unsafe" só conta com
alguma categoria fora da lista ignorada (padrão S1 e S6)."""

import json

import httpx
import pytest
import respx

from encaixe.guardrails.runner import run_guards
from encaixe.guardrails.safety import ContentSafetyGuard, parse_verdict
from encaixe.settings import Settings

BASE = "http://ollama.test"
PRIMARY = "shieldgemma-2b-cpu"
SECONDARY = "llama-guard3-1b-cpu"


def settings(**kw) -> Settings:
    return Settings(_env_file=None, ollama_base_url=BASE, **kw)


def _reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def route_by_model(replies: dict[str, object]):
    """Mock de /api/chat que responde conforme o modelo pedido; registra os corpos."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        reply = replies[body["model"]]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return _reply(str(reply))

    respx.post(f"{BASE}/api/chat").mock(side_effect=handler)
    return seen


@pytest.mark.parametrize(
    "content,expected",
    [
        ("safe", ("pass", [])),
        ("unsafe\nS1,S10", ("block", ["S1", "S10"])),
        ("\n  unsafe", ("block", [])),
        ("unsafe\nS1, S6", ("block", ["S1", "S6"])),
        ("Yes. The prompt asks for", ("block", [])),
        ("No, the human question does not", ("pass", [])),
        ("  No  ", ("pass", [])),
    ],
)
def test_parse_verdict_llama_guard_and_yes_no(content, expected):
    assert parse_verdict(content) == expected


@pytest.mark.parametrize("content", ["", "hmm", "Talvez"])
def test_parse_verdict_rejects_garbage(content):
    with pytest.raises(ValueError):
        parse_verdict(content)


@respx.mock
async def test_primary_block_counts_whatever_the_secondary_says(make_ctx):
    seen = route_by_model({PRIMARY: "Yes", SECONDARY: "safe"})
    r = await ContentSafetyGuard(settings()).check("texto", make_ctx())
    assert r.verdict == "block" and r.reason.startswith(PRIMARY)
    assert sorted(b["model"] for b in seen) == sorted([PRIMARY, SECONDARY])


@respx.mock
@pytest.mark.parametrize("categories", ["S1", "S6", "S1,S6"])
async def test_secondary_ignored_categories_only_means_pass(make_ctx, categories):
    route_by_model({PRIMARY: "No", SECONDARY: f"unsafe\n{categories}"})
    r = await ContentSafetyGuard(settings()).check("Crie uma tarefa: contrato 9f8e7d", make_ctx())
    assert r.verdict == "pass"


@respx.mock
async def test_secondary_category_outside_ignore_list_blocks(make_ctx):
    route_by_model({PRIMARY: "No", SECONDARY: "unsafe\nS1,S11"})
    r = await ContentSafetyGuard(settings()).check("texto", make_ctx())
    assert r.verdict == "block" and r.reason == f"{SECONDARY}:S11"


@respx.mock
async def test_secondary_unsafe_without_categories_blocks(make_ctx):
    """ "unsafe" sem categoria não tem o que ignorar: conta e bloqueia (fail-closed; decisão
    mantida na rodada de correções 2). Uma resposta truncada não pode virar "seguro"."""
    route_by_model({PRIMARY: "No", SECONDARY: "unsafe"})
    assert (await ContentSafetyGuard(settings()).check("x", make_ctx())).verdict == "block"


@respx.mock
async def test_ignore_list_is_configurable(make_ctx):
    route_by_model({PRIMARY: "No", SECONDARY: "unsafe\nS6"})
    guard = ContentSafetyGuard(settings(safety_secondary_ignore_categories=["S1"]))
    assert (await guard.check("x", make_ctx())).verdict == "block"


@respx.mock
async def test_secondary_disabled_only_primary_is_asked(make_ctx):
    seen = route_by_model({PRIMARY: "No"})
    guard = ContentSafetyGuard(settings(safety_secondary_model=""))
    assert (await guard.check("x", make_ctx())).verdict == "pass"
    assert [b["model"] for b in seen] == [PRIMARY]


@respx.mock
async def test_output_uses_only_the_secondary_by_default(make_ctx):
    """Ruling da rodada de correções 2: o ShieldGemma levava ~7 s na saída (CPU); a saída
    fica com o Llama Guard e a lista de ignoradas. O ShieldGemma só olha a entrada."""
    seen = route_by_model({SECONDARY: "safe"})  # o principal nem é consultado
    r = await ContentSafetyGuard(settings()).check_response("resp", make_ctx(user_message="perg"))
    assert r.verdict == "pass"
    assert [b["model"] for b in seen] == [SECONDARY]
    assert [m["role"] for m in seen[0]["messages"]] == ["user", "assistant"]
    assert [m["content"] for m in seen[0]["messages"]] == ["perg", "resp"]
    assert seen[0]["stream"] is False


@respx.mock
@pytest.mark.parametrize("reply,verdict", [("unsafe\nS1", "pass"), ("unsafe\nS1,S2", "block")])
async def test_output_secondary_keeps_its_ignore_list(make_ctx, reply, verdict):
    route_by_model({SECONDARY: reply})
    r = await ContentSafetyGuard(settings()).check_output_pair("salário?", "R$ 72.000,00")
    assert r.verdict == verdict


@respx.mock
async def test_output_unsafe_without_categories_blocks(make_ctx):
    route_by_model({SECONDARY: "unsafe"})
    r = await ContentSafetyGuard(settings()).check_output_pair("p", "r")
    assert r.verdict == "block"


@respx.mock
async def test_output_primary_opt_in_runs_both(make_ctx):
    seen = route_by_model({PRIMARY: "Yes", SECONDARY: "safe"})
    guard = ContentSafetyGuard(settings(safety_output_primary=True))
    r = await guard.check_response("resp", make_ctx(user_message="perg"))
    assert r.verdict == "block" and r.reason.startswith(PRIMARY)
    assert sorted(b["model"] for b in seen) == sorted([PRIMARY, SECONDARY])


@respx.mock
async def test_output_falls_back_to_primary_without_secondary(make_ctx):
    """Sem secundário a saída nunca fica sem checagem: usa o principal (todo veredito conta)."""
    seen = route_by_model({PRIMARY: "Yes"})
    guard = ContentSafetyGuard(settings(safety_secondary_model=""))
    r = await guard.check_output_pair("p", "r")
    assert r.verdict == "block"
    assert [b["model"] for b in seen] == [PRIMARY]


@respx.mock
async def test_input_still_runs_both(make_ctx):
    seen = route_by_model({PRIMARY: "No", SECONDARY: "safe"})
    await ContentSafetyGuard(settings()).check("pergunta", make_ctx())
    assert sorted(b["model"] for b in seen) == sorted([PRIMARY, SECONDARY])


@respx.mock
@pytest.mark.parametrize(
    "replies",
    [
        {PRIMARY: "No", SECONDARY: httpx.ConnectError("fora")},
        {PRIMARY: httpx.Response(500), SECONDARY: "safe"},
        {PRIMARY: "hmm", SECONDARY: "safe"},  # resposta ilegível
    ],
)
async def test_backend_error_is_unavailable(make_ctx, replies):
    route_by_model(replies)
    with pytest.raises((httpx.HTTPError, ValueError)):
        await ContentSafetyGuard(settings()).check("x", make_ctx())


@respx.mock
async def test_a_definite_block_wins_over_the_other_backend_error(make_ctx):
    route_by_model({PRIMARY: "Yes", SECONDARY: httpx.ConnectError("fora")})
    r = await ContentSafetyGuard(settings()).check("x", make_ctx())
    assert r.verdict == "block"


@respx.mock
async def test_runner_keeps_unavailable_semantics(make_ctx):
    route_by_model({PRIMARY: "No", SECONDARY: httpx.ConnectError("fora")})
    cfg = settings()
    closed = await run_guards(["content_safety"], "x", make_ctx(), settings=cfg, fail_closed=True)
    opened = await run_guards(["content_safety"], "x", make_ctx(), settings=cfg, fail_closed=False)
    assert (closed[0].verdict, closed[0].reason) == ("block", "unavailable")
    assert (opened[0].verdict, opened[0].reason) == ("pass", "unavailable")
