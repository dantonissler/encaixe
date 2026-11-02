from __future__ import annotations

import asyncio
import time

import pytest
from fastmcp import FastMCP
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.fallback import FallbackModel

from encaixe.core.config import McpServerConfig
from encaixe.core.pii import mask_pii
from encaixe.core.registry import register
from encaixe.core.types import ChatMessage, GuardResult, Principals
from encaixe.llm.breaker import BreakerRegistry
from encaixe.llm.fallback import build_chain
from encaixe.messages import (
    ALL_FAILED,
    GUARD_UNAVAILABLE,
    LIMIT_REACHED,
    NOT_FOUND,
    RAG_UNAVAILABLE,
    REFUSAL_INPUT,
    RETRACTION,
)
from encaixe.output.schemas import TEXT_ANSWER_RETRY
from encaixe.pipeline.chat import ChatPipeline
from encaixe.pipeline.events import Retraction, Status, TextDelta
from encaixe.settings import Settings

from .helpers import (
    Call,
    FailingModel,
    FakeApprovals,
    FakeCache,
    FakeHistory,
    FakeRetriever,
    Raw,
    Script,
    answer,
    chunk,
    collect,
    final_of,
    make_app,
    make_ctx,
    make_harness,
    text_of,
)

# --- guards de teste (registrados no registry, como os reais) ---------------------------------

LIVE_EVENTS: list = []
SEEN_AT_OUTPUT_CHECK: list[list] = []


@register("guard", "tp_block_in")
def _block_in(**_):
    class G:
        name = "tp_block_in"

        async def check(self, text, ctx):
            return GuardResult(self.name, "block", "unsafe")

    return G()


@register("guard", "tp_boom")
def _boom(**_):
    class G:
        name = "tp_boom"

        async def check(self, text, ctx):
            raise ConnectionError("llama guard fora do ar")

    return G()


@register("guard", "tp_block_out")
def _block_out(**_):
    class G:
        name = "tp_block_out"

        async def check(self, text, ctx):
            return GuardResult(self.name, "block", "unsafe")

    return G()


@register("guard", "tp_spy_out")
def _spy_out(**_):
    class G:
        name = "tp_spy_out"

        async def check(self, text, ctx):
            SEEN_AT_OUTPUT_CHECK.append(list(LIVE_EVENTS))
            return GuardResult(self.name, "pass")

    return G()


# --- 1. guards de entrada --------------------------------------------------------------------


async def test_input_block_returns_refusal_and_skips_agent():
    app = make_app(input_guards=["tp_block_in"])
    h = make_harness(app)
    events = await collect(ChatPipeline(h.deps), make_ctx(app, identity=True))
    assert events[:-1] == [TextDelta(REFUSAL_INPUT)]
    assert final_of(events).output.resposta == REFUSAL_INPUT
    assert h.script.calls == 0 and h.chains == [] and h.retriever.calls == []
    assert h.history.appended == []  # mensagem bloqueada não entra no histórico
    assert h.cache.gets == 0 and h.enqueued == []


async def test_guard_unavailable_fail_closed_message():
    app = make_app(input_guards=["tp_boom"], on_unavailable="block")
    h = make_harness(app)
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert text_of(events) == GUARD_UNAVAILABLE
    assert final_of(events).output.resposta == GUARD_UNAVAILABLE
    assert h.script.calls == 0


async def test_guard_unavailable_fail_open_when_configured():
    app = make_app(input_guards=["tp_boom"], on_unavailable="allow")
    h = make_harness(app, Script(answer("Trinta dias.")))
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert text_of(events) == "Trinta dias."
    assert h.script.calls == 1


async def test_normalized_text_replaces_message():
    app = make_app(input_guards=["normalize"], cache=False)
    h = make_harness(app)
    ctx = make_ctx(app, "Férias\u200b de quantos dias?")
    await collect(ChatPipeline(h.deps), ctx)
    assert ctx.user_message == "Férias de quantos dias?"
    assert "Férias de quantos dias?" in h.script.prompt()
    assert h.retriever.calls[0][2] == "Férias de quantos dias?"


async def test_persists_masked_user_message_when_pii():
    app = make_app(input_guards=["normalize", "pii"])
    h = make_harness(app, Script(answer("Anotado.")))
    msg = "Meu CPF é 529.982.247-25, quantos dias de férias tenho?"
    ctx = make_ctx(app, msg, identity=True)
    events = await collect(ChatPipeline(h.deps), ctx)
    assert final_of(events).output.resposta == "Anotado."
    assert h.history.appended == [
        ("user", "Meu CPF é [CPF], quantos dias de férias tenho?"),
        ("assistant", "Anotado."),
    ]
    assert "529.982.247-25" in h.script.prompt()  # o modelo recebe o texto original (normalizado)
    assert h.cache.sets == []  # dado pessoal nunca vai para o cache


async def test_persists_masked_assistant_reply_when_pii():
    app = make_app(input_guards=["normalize", "pii"])
    reply = "Anotei o CPF 529.982.247-25 e o e-mail ana@aurora.com.br."
    h = make_harness(app, Script(answer(reply)))
    msg = "Meu CPF é 529.982.247-25 e meu e-mail é ana@aurora.com.br."
    events = await collect(ChatPipeline(h.deps), make_ctx(app, msg, identity=True))
    assert final_of(events).output.resposta == reply  # o usuário recebe a resposta como é
    assert text_of(events) == reply
    (_, persisted) = h.history.appended[-1]
    assert "529.982.247-25" not in persisted and "ana@aurora.com.br" not in persisted
    assert persisted == mask_pii(reply)


async def test_assistant_reply_not_masked_without_pii_in_turn():
    app = make_app(input_guards=["normalize", "pii"])
    h = make_harness(app, Script(answer("O telefone do RH é (11) 3333-4444.")))
    await collect(ChatPipeline(h.deps), make_ctx(app, "Qual o telefone do RH?", identity=True))
    assert h.history.appended[-1] == ("assistant", "O telefone do RH é (11) 3333-4444.")


# --- 2. aprovações ---------------------------------------------------------------------------


async def test_approval_result_goes_to_extra_context_with_identity():
    app = make_app()
    h = make_harness(app, approvals=FakeApprovals("RESULTADO-DA-ACAO"))
    ctx = make_ctx(app, "sim", identity=True, conversation_id="c1", principals=Principals("ana"))
    await collect(ChatPipeline(h.deps), ctx)
    assert len(h.approvals.calls) == 1 and h.approvals.calls[0].startswith("DADOS-")
    assert ctx.extra_context == ["RESULTADO-DA-ACAO"]
    assert "RESULTADO-DA-ACAO" in h.script.prompt()
    assert h.cache.gets == 0 and h.cache.sets == []  # resposta depende da ação: sem cache


@pytest.mark.parametrize(
    "identity,conversation_id,user_ref",
    [
        (False, "c1", "ana"),
        (True, None, "ana"),
        # conversa anônima continuada pelo id: quem repete o X-Conversation-Id não é um usuário
        (True, "c1", None),
    ],
)
async def test_approval_not_resolved_without_identity(identity, conversation_id, user_ref):
    app = make_app()
    h = make_harness(app, approvals=FakeApprovals("RESULTADO-DA-ACAO"))
    ctx = make_ctx(
        app,
        "sim",
        identity=identity,
        conversation_id=conversation_id,
        principals=Principals(user_ref),
    )
    await collect(ChatPipeline(h.deps), ctx)
    assert h.approvals.calls == []
    assert ctx.extra_context == []
    assert "RESULTADO-DA-ACAO" not in h.script.prompt()


# --- 3. cache exato --------------------------------------------------------------------------


async def test_cache_hit_skips_agent():
    app = make_app(rag=False)
    h = make_harness(app, Script(answer("Trinta dias corridos.", [])))
    pipeline = ChatPipeline(h.deps)
    first = await collect(pipeline, make_ctx(app))
    assert h.script.calls == 1 and len(h.cache.sets) == 1
    ctx2 = make_ctx(app)
    second = await collect(pipeline, ctx2)
    assert h.script.calls == 1  # o agente não roda de novo
    assert ctx2.cache_layer_hit == "exact"
    assert text_of(second) == text_of(first) == "Trinta dias corridos."
    assert final_of(second).output.resposta == "Trinta dias corridos."
    assert h.history.appended[-2:] == [
        ("user", "Qual é a política de férias?"),
        ("assistant", "Trinta dias corridos."),
    ]


async def test_cache_key_uses_override_model_and_principals():
    app = make_app(rag=False)
    h = make_harness(app, Script(answer("Ok.", [])))
    pipeline = ChatPipeline(h.deps)
    await collect(pipeline, make_ctx(app))
    await collect(pipeline, make_ctx(app, principals=Principals("u1", ("rh",))))
    await collect(pipeline, make_ctx(app, override="ollama:outro"))  # env dev: override vale
    assert h.script.calls == 3 and len(set(h.cache.sets)) == 3


async def test_cache_error_fails_open():
    app = make_app(rag=False)
    h = make_harness(app, Script(answer("Trinta dias.", [])), cache=FakeCache(fail=True))
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert text_of(events) == "Trinta dias."
    assert final_of(events).output.resposta == "Trinta dias."
    assert h.cache.gets == 1 and h.script.calls == 1


async def test_corrupt_cache_entry_is_a_miss():
    app = make_app(rag=False)
    h = make_harness(app, Script(answer("Do modelo.", [])))
    pipeline = ChatPipeline(h.deps)
    await collect(pipeline, make_ctx(app))
    (key,) = h.cache.data
    h.cache.data[key] = {"sem": "resposta"}
    events = await collect(pipeline, make_ctx(app))
    assert text_of(events) == "Do modelo." and h.script.calls == 2


# --- 4/5. retrieval e contexto ---------------------------------------------------------------


async def test_retrieved_chunks_reach_model_and_ctx():
    app = make_app()
    h = make_harness(app, Script(answer("São 30 dias.", ["k1"])))
    ctx = make_ctx(app)
    events = await collect(ChatPipeline(h.deps), ctx)
    assert [s.chunk.id for s in ctx.retrieved] == ["k1"]
    assert "Política de férias: 30 dias corridos." in h.script.prompt()
    assert final_of(events).output.citacoes == ["k1"]


async def test_rag_disabled_skips_retrieval():
    app = make_app(rag=False)
    h = make_harness(app, Script(answer("Oi!", [])))
    events = await collect(ChatPipeline(h.deps), make_ctx(app, "oi"))
    assert h.retriever.calls == [] and text_of(events) == "Oi!"


@pytest.mark.parametrize("cites", [[], ["inventado-9"]])
async def test_empty_retrieval_yields_not_found(cites):
    app = make_app()
    h = make_harness(
        app,
        Script(answer("Inventei uma resposta. Com duas frases.", cites)),
        retriever=FakeRetriever([]),
    )
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert text_of(events) == NOT_FOUND  # nada do texto inventado foi liberado
    assert not any(isinstance(e, Retraction) for e in events)
    out = final_of(events).output
    assert out.resposta == NOT_FOUND and out.citacoes == []
    assert h.cache.sets == []


async def test_retrieval_failure_says_documents_could_not_be_consulted():
    """Revisão final M-T13a (spec §6.11.5): RAG fora do ar não é "não encontrei"; a resposta
    avisa que os documentos não puderam ser consultados e não vai para o cache."""
    app = make_app()
    h = make_harness(
        app, Script(answer("Inventei algo.", ["k1"])), retriever=FakeRetriever(fail=True)
    )
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert text_of(events) == RAG_UNAVAILABLE
    out = final_of(events).output
    assert out.resposta == RAG_UNAVAILABLE and out.citacoes == []
    assert h.cache.sets == []


async def test_canary_identical_across_requests_of_same_app():
    app = make_app(rag=False, cache=False)
    h = make_harness(app)
    pipeline = ChatPipeline(h.deps)
    c1, c2 = make_ctx(app, canary=""), make_ctx(app, canary="cnr-outro")
    await collect(pipeline, c1)
    await collect(pipeline, c2)
    canary = h.deps.canaries["a"]
    assert canary.startswith("cnr-") and c1.canary == c2.canary == canary
    first, second = h.script.infos[0].instructions, h.script.infos[1].instructions
    assert first == second and canary in first


async def test_canary_from_deps_is_used():
    app = make_app(rag=False, cache=False)
    h = make_harness(app, canaries={"a": "cnr-fixo123"})
    await collect(ChatPipeline(h.deps), make_ctx(app, canary="cnr-outro"))
    assert "cnr-fixo123" in h.script.infos[0].instructions


async def test_server_history_and_summary_with_identity():
    app = make_app(rag=False, cache=False)
    history = FakeHistory(
        messages=[
            ChatMessage("user", "pergunta antiga", seq=3),
            ChatMessage("assistant", "resposta antiga", seq=4),
        ],
        summary=("RESUMO-DA-CONVERSA", 2),
    )
    h = make_harness(app, Script(answer("Nova.", [])), history=history)
    ctx = make_ctx(app, "pergunta nova", identity=True, history=[ChatMessage("user", "IGNORAR")])
    await collect(ChatPipeline(h.deps), ctx)
    assert history.listed == 1
    prompt = h.script.prompt()
    assert "pergunta antiga" in prompt and "IGNORAR" not in prompt
    assert any("RESUMO-DA-CONVERSA" in p for p in h.script.system_parts())
    assert h.enqueued == [("summarize_conversation", "a", "c1")]
    assert h.history.appended == [("user", "pergunta nova"), ("assistant", "Nova.")]
    assert h.history.append_kwargs[1]["prompt_version"] == "v1"


async def test_client_history_without_identity():
    app = make_app(rag=False, cache=False)
    h = make_harness(app, Script(answer("Nova.", [])))
    client = [
        ChatMessage("user", "pergunta do cliente"),
        ChatMessage("assistant", "resposta do cliente"),
    ]
    await collect(ChatPipeline(h.deps), make_ctx(app, "pergunta nova", history=client))
    assert h.history.listed == 0
    assert "pergunta do cliente" in h.script.prompt()
    assert h.history.appended == [("user", "pergunta nova"), ("assistant", "Nova.")]
    assert h.enqueued == []  # sem identidade de conversa o resumo nunca seria usado


async def test_nothing_persisted_without_conversation_id():
    app = make_app(rag=False, cache=False)
    h = make_harness(app)
    events = await collect(ChatPipeline(h.deps), make_ctx(app, conversation_id=None))
    assert final_of(events).conversation_id is None
    assert h.history.appended == [] and h.enqueued == []


# --- 6. agente, ferramentas e streaming ------------------------------------------------------


async def test_buffered_mode_releases_sentence_by_sentence():
    app = make_app(output_guards=["pii_leak", "canary"], require_citations=False)
    h = make_harness(
        app, Script(answer("Primeira frase. Segunda frase! Terceira", ["k1"]), chunk=3)
    )
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas == ["Primeira frase. ", "Segunda frase! ", "Terceira"]
    assert final_of(events).output.resposta == "Primeira frase. Segunda frase! Terceira"


async def test_buffered_mode_holds_sentence_with_canary_leak():
    app = make_app(output_guards=["canary"], require_citations=False)
    h = make_harness(
        app,
        Script(answer("Tudo certo. O código é cnr-abc123def456. Fim.", ["k1"]), chunk=3),
        canaries={"a": "cnr-abc123def456"},
    )
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Tudo certo. "]
    assert Retraction(RETRACTION) in events
    assert final_of(events).output.resposta == RETRACTION
    assert "cnr-abc123def456" not in str(events)


async def test_tool_call_emits_status_and_registers_retrieval():
    app = make_app(native=["buscar_documentos"])
    script = Script(
        Call("buscar_documentos", {"consulta": "férias"}), answer("São 30 dias.", ["k1"])
    )
    # a pré-busca não acha nada; a busca feita pela ferramenta acha o trecho
    h = make_harness(app, script, retriever=FakeRetriever(sequence=[[], [chunk()]]))
    ctx = make_ctx(app)
    events = await collect(ChatPipeline(h.deps), ctx)
    assert Status("tool", {"name": "buscar_documentos"}) in events
    assert ctx.tools_called == ["buscar_documentos"]
    assert [s.chunk.id for s in ctx.retrieved] == ["k1"]
    assert final_of(events).output.resposta == "São 30 dias."  # não vira NOT_FOUND
    # só a busca nos documentos: continua cacheável (a chave leva corpus e principais)
    assert len(h.cache.sets) == 1


async def test_loop_detected_returns_limit_message():
    app = make_app(native=["buscar_documentos"])
    h = make_harness(app, Script(Call("buscar_documentos", {"consulta": "x"})))
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    out = final_of(events).output
    assert out.resposta.startswith(LIMIT_REACHED) and "buscar_documentos" in out.resposta
    assert text_of(events) == out.resposta


def tarefas_server() -> tuple[FastMCP, list[McpServerConfig]]:
    server = FastMCP("tarefas")

    @server.tool
    def listar_tarefas() -> str:
        """Lista tarefas."""
        return "nenhuma"

    @server.tool
    def criar_tarefa(titulo: str) -> str:
        """Cria tarefa."""
        return "criada"

    mcp = [
        McpServerConfig(
            name="tarefas",
            url="http://mcp.invalido/mcp",
            allow=["listar_tarefas", "criar_tarefa"],
            require_approval=["criar_tarefa"],
        )
    ]
    return server, mcp


async def test_write_tools_absent_without_identity():
    server, mcp = tarefas_server()
    app = make_app(native=["buscar_documentos"], mcp=mcp, rag=False, cache=False)
    h = make_harness(app, mcp_source=lambda cfg, headers: server)
    pipeline = ChatPipeline(h.deps)
    ana = Principals("ana")
    await collect(pipeline, make_ctx(app, identity=False, principals=ana))
    await collect(pipeline, make_ctx(app, identity=True, conversation_id=None, principals=ana))
    await collect(pipeline, make_ctx(app, identity=True, conversation_id="c1", principals=ana))
    assert h.script.tool_names(0) == {"buscar_documentos", "listar_tarefas"}
    assert h.script.tool_names(1) == {"buscar_documentos", "listar_tarefas"}
    assert h.script.tool_names(2) == {"buscar_documentos", "listar_tarefas", "criar_tarefa"}


async def test_anonymous_conversation_continued_by_id_gets_no_write_tools():
    """Sem user_ref, repetir o X-Conversation-Id recebido não dá ferramentas de escrita."""
    server, mcp = tarefas_server()
    app = make_app(native=["buscar_documentos"], mcp=mcp, rag=False, cache=False)
    h = make_harness(
        app, approvals=FakeApprovals("RESULTADO"), mcp_source=lambda cfg, headers: server
    )
    pipeline = ChatPipeline(h.deps)
    anonymous = make_ctx(app, identity=True, conversation_id="c1", principals=Principals(None))
    await collect(pipeline, anonymous)
    assert h.script.tool_names(0) == {"buscar_documentos", "listar_tarefas"}
    assert h.approvals.calls == [] and anonymous.extra_context == []
    group_only = Principals(None, ("rh",))  # grupo sem usuário também não é identidade
    await collect(
        pipeline, make_ctx(app, identity=True, conversation_id="c1", principals=group_only)
    )
    assert h.script.tool_names(1) == {"buscar_documentos", "listar_tarefas"}
    assert h.approvals.calls == []


async def test_mcp_server_down_degrades_to_no_mcp_tools():
    mcp = [McpServerConfig(name="tarefas", url="http://127.0.0.1:9/mcp", allow=["listar_tarefas"])]
    app = make_app(native=["buscar_documentos"], mcp=mcp, rag=False)
    h = make_harness(
        app, Script(answer("Sem tarefas.", [])), mcp_source=lambda cfg, headers: cfg.url
    )
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert text_of(events) == "Sem tarefas."
    assert h.script.tool_names(0) == {"buscar_documentos"}


async def test_model_override_only_in_dev():
    app = make_app(rag=False, cache=False, fallbacks=["test:fallback"])
    prod = make_harness(app, settings=Settings(env="prod"))
    await collect(ChatPipeline(prod.deps), make_ctx(app, override="ollama:outro"))
    assert prod.chains == [("test:primary", ["test:fallback"])]
    dev = make_harness(app, settings=Settings(env="dev"))
    await collect(ChatPipeline(dev.deps), make_ctx(app, override="ollama:outro"))
    assert dev.chains == [("ollama:outro", [])]


# --- fallback e falhas -----------------------------------------------------------------------


async def test_fallback_model_marks_fallback_used():
    app = make_app(rag=False, fallbacks=["test:fallback"])
    script = Script(answer("Respondi pelo reserva.", []), name="fallback")

    def chain(primary, fallbacks):
        model = FallbackModel(
            FailingModel.build("primary"), script.model(), fallback_on=(ModelAPIError,)
        )
        return model, [primary, *fallbacks]

    h = make_harness(app, script, chain=chain)
    ctx = make_ctx(app)
    events = await collect(ChatPipeline(h.deps), ctx)
    assert Status("fallback", {"model": "fallback"}) in events
    assert ctx.fallback_used is True
    assert text_of(events) == "Respondi pelo reserva."
    assert h.cache.sets == []  # resposta do reserva não vai para o cache do principal


async def test_breaker_open_reports_fallback_before_streaming():
    app = make_app(rag=False, fallbacks=["test:fallback"])
    script = Script(answer("Pelo reserva.", []), name="fallback")
    h = make_harness(
        app, script, chain=lambda primary, fallbacks: (script.model(), list(fallbacks))
    )
    ctx = make_ctx(app)
    events = await collect(ChatPipeline(h.deps), ctx)
    assert events[0] == Status("fallback", {"model": "test:fallback", "reason": "breaker_open"})
    assert ctx.fallback_used is True and text_of(events) == "Pelo reserva."


async def test_output_retry_after_release_emits_retraction():
    app = make_app(rag=False)
    invalid = '{"resposta": "Primeira versão. Ainda escrevendo", "outro": '  # JSON truncado
    h = make_harness(app, Script(Raw(invalid), answer("Segunda versão.", []), chunk=3))
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert h.script.calls == 2  # nova tentativa de saída (run_stream não faria)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Primeira versão. "]
    assert events[-2] == Retraction("Segunda versão.")
    assert final_of(events).output.resposta == "Segunda versão."


async def test_all_models_fail_returns_all_failed():
    app = make_app(rag=False, fallbacks=["test:fallback"])

    def chain(primary, fallbacks):
        model = FallbackModel(
            FailingModel.build("primary"),
            FailingModel.build("fallback"),
            fallback_on=(ModelAPIError,),
        )
        return model, [primary, *fallbacks]

    h = make_harness(app, chain=chain)
    events = await collect(ChatPipeline(h.deps), make_ctx(app, identity=True))
    assert text_of(events) == ALL_FAILED
    assert final_of(events).output.resposta == ALL_FAILED
    assert h.cache.sets == []  # falha nunca vai para o cache
    assert h.history.appended == [
        ("user", "Qual é a política de férias?"),
        ("assistant", ALL_FAILED),
    ]


async def test_ollama_down_with_real_chain_returns_all_failed():
    app = make_app(rag=False, fallbacks=["ollama:reserva"])
    settings = Settings(ollama_base_url="http://127.0.0.1:9")
    breakers = BreakerRegistry()

    def chain(primary, fallbacks):
        return build_chain("ollama:principal", fallbacks, settings, breakers)

    h = make_harness(app, chain=chain, settings=settings)
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert text_of(events) == ALL_FAILED
    assert final_of(events).output.resposta == ALL_FAILED


async def test_timeout_returns_all_failed():
    app = make_app(rag=False, timeout_s=1)
    h = make_harness(app, Script(answer("Tarde demais.", []), delay=10))
    start = time.monotonic()
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert time.monotonic() - start < 5
    assert text_of(events) == ALL_FAILED


class DriverCrash(BaseException):
    """Escapa do `except Exception` do driver (não é CancelledError do consumidor)."""


async def test_agent_driver_dying_without_terminal_update_does_not_hang(monkeypatch):
    import encaixe.pipeline.chat as chat_module

    async def crashing_driver(agent, built, deps, limits, timeout_s, queue):
        await asyncio.sleep(0)
        raise DriverCrash("morreu sem avisar")

    monkeypatch.setattr(chat_module, "drive_agent", crashing_driver)
    app = make_app(rag=False)
    h = make_harness(app)
    events = await asyncio.wait_for(collect(ChatPipeline(h.deps), make_ctx(app)), 5)
    assert text_of(events) == ALL_FAILED
    assert final_of(events).output.resposta == ALL_FAILED
    assert h.cache.sets == []


async def test_unexpected_error_still_ends_with_final(monkeypatch):
    app = make_app(rag=False)
    h = make_harness(app)
    pipeline = ChatPipeline(h.deps)

    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(pipeline, "_build_context", boom)
    events = await collect(pipeline, make_ctx(app))
    assert text_of(events) == ALL_FAILED
    assert final_of(events).output.resposta == ALL_FAILED


# --- 7. guards de saída ----------------------------------------------------------------------


async def test_output_block_emits_retraction():
    app = make_app(output_guards=["tp_block_out"], require_citations=False)
    h = make_harness(app, Script(answer("Frase um. Frase dois.", ["k1"]), chunk=3))
    events = await collect(ChatPipeline(h.deps), make_ctx(app, identity=True))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Frase um. "]
    assert events[-2] == Retraction(RETRACTION)
    out = final_of(events).output
    assert out.resposta == RETRACTION and out.citacoes == []
    assert h.cache.sets == []
    assert h.history.appended[-1] == ("assistant", RETRACTION)


@pytest.mark.parametrize("mode", ["buffered_sentences", "strict"])
async def test_output_guard_unavailable_is_guard_unavailable_not_retraction(mode):
    # tp_boom levanta: o runner dá block com reason "unavailable" (Llama Guard fora do ar)
    app = make_app(output_guards=["tp_boom"], on_unavailable="block", rag=False, mode=mode)
    h = make_harness(app, Script(answer("Frase um. Frase dois.", []), chunk=3))
    events = await collect(ChatPipeline(h.deps), make_ctx(app, identity=True))
    out = final_of(events).output
    assert out.resposta == GUARD_UNAVAILABLE and out.citacoes == []
    assert events[-2] == Retraction(GUARD_UNAVAILABLE)  # o texto já liberado deixa de valer
    assert RETRACTION not in str(events)
    assert h.cache.sets == []
    assert h.history.appended[-1] == ("assistant", GUARD_UNAVAILABLE)


async def test_strict_mode_releases_nothing_before_check():
    LIVE_EVENTS.clear()
    SEEN_AT_OUTPUT_CHECK.clear()
    app = make_app(mode="strict", output_guards=["tp_spy_out"])
    h = make_harness(app, Script(answer("Frase um. Frase dois. Frase três.", ["k1"]), chunk=3))
    await collect(ChatPipeline(h.deps), make_ctx(app), sink=LIVE_EVENTS)
    assert len(SEEN_AT_OUTPUT_CHECK) == 1
    assert not any(isinstance(e, TextDelta) for e in SEEN_AT_OUTPUT_CHECK[0])
    deltas = [e.text for e in LIVE_EVENTS if isinstance(e, TextDelta)]
    assert deltas == ["Frase um. Frase dois. Frase três."]


async def test_strict_mode_block_releases_only_retraction():
    app = make_app(mode="strict", output_guards=["tp_block_out"])
    h = make_harness(app, Script(answer("Frase um. Frase dois.", ["k1"])))
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert not any(isinstance(e, TextDelta) for e in events)
    assert events[:-1] == [Retraction(RETRACTION)]
    assert final_of(events).output.resposta == RETRACTION


# --- citações: ids curtos (T1…), normalização e NOT_FOUND --------------------------------------

TWO_CHUNKS = [
    chunk("k1", "Férias: 30 dias corridos."),
    chunk("k2", "Divisão em até três períodos."),
]


@pytest.mark.parametrize(
    "cites,expected",
    [
        (["T1"], ["k1"]),
        (["t1"], ["k1"]),
        (["[T2]"], ["k2"]),
        (["id: T1"], ["k1"]),
        (["k2"], ["k2"]),  # o id real do trecho também vale
        (["T2", "t1", "k2"], ["k2", "k1"]),
    ],
)
async def test_citation_aliases_map_to_chunk_ids(cites, expected):
    app = make_app(output_guards=["citations"])
    h = make_harness(
        app, Script(answer("São 30 dias.", cites)), retriever=FakeRetriever(TWO_CHUNKS)
    )
    ctx = make_ctx(app)
    events = await collect(ChatPipeline(h.deps), ctx)
    assert ctx.citation_aliases == {"T1": "k1", "T2": "k2"}
    prompt = h.script.prompt()
    assert "T1" in prompt and "T2" in prompt and "k1" not in prompt and "k2" not in prompt
    out = final_of(events).output
    assert out.resposta == "São 30 dias." and out.citacoes == expected
    assert not any(isinstance(e, Retraction) for e in events)


async def test_invented_citation_is_dropped_not_retracted():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from encaixe.observability.spans import use_tracer_provider

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app = make_app(output_guards=["citations"])
    cites = ["T1", "T7", "id: d135c7b6-1111-4222-8333-444455556666", "k-falso"]
    h = make_harness(app, Script(answer("Segundo o manual.", cites)))
    use_tracer_provider(provider)
    try:
        events = await collect(ChatPipeline(h.deps), make_ctx(app))
    finally:
        use_tracer_provider(None)
    out = final_of(events).output
    assert out.resposta == "Segundo o manual." and out.citacoes == ["k1"]
    assert not any(isinstance(e, Retraction) for e in events)
    (span,) = [
        s for s in exporter.get_finished_spans() if s.name == "encaixe.pipeline.output_guards"
    ]
    assert span.attributes["citations.dropped"] == 3


@pytest.mark.parametrize("cites", [["T9", "id: inventado"], []])
async def test_zero_valid_citations_is_not_found_never_retraction(cites):
    app = make_app(output_guards=["citations"])
    h = make_harness(app, Script(answer("Acho que são 45 dias. Confie.", cites), chunk=3))
    events = await collect(ChatPipeline(h.deps), make_ctx(app, identity=True))
    assert not any(isinstance(e, Retraction) for e in events)
    assert text_of(events) == NOT_FOUND  # nada do texto sem base foi liberado antes
    out = final_of(events).output
    assert out.resposta == NOT_FOUND and out.citacoes == []
    assert RETRACTION not in str(events)
    assert h.cache.sets == []
    assert h.history.appended[-1] == ("assistant", NOT_FOUND)


async def test_without_require_citations_uncited_answer_stands():
    app = make_app(output_guards=["citations"], require_citations=False)
    h = make_harness(app, Script(answer("Oi! Como posso ajudar?", [])))
    events = await collect(ChatPipeline(h.deps), make_ctx(app, "oi"))
    assert final_of(events).output.resposta == "Oi! Como posso ajudar?"


async def test_tool_added_chunks_get_aliases():
    app = make_app(native=["buscar_documentos"], output_guards=["citations"])
    script = Script(
        Call("buscar_documentos", {"consulta": "divisão"}), answer("Até três períodos.", ["T2"])
    )
    retriever = FakeRetriever(sequence=[[chunk("k1")], [chunk("k1"), chunk("k2", "Divisão.")]])
    h = make_harness(app, script, retriever=retriever)
    ctx = make_ctx(app)
    events = await collect(ChatPipeline(h.deps), ctx)
    # pré-busca: k1 = T1; a ferramenta reaproveita T1 e continua a numeração em T2
    assert ctx.citation_aliases == {"T1": "k1", "T2": "k2"}
    tool_return = next(
        p.content
        for m in h.script.messages[1]
        for p in getattr(m, "parts", [])
        if getattr(p, "part_kind", "") == "tool-return"
    )
    assert "T1" in tool_return and "T2" in tool_return and "k2" not in tool_return
    out = final_of(events).output
    assert out.resposta == "Até três períodos." and out.citacoes == ["k2"]


async def test_require_citations_holds_release_until_final_output():
    LIVE_EVENTS.clear()
    SEEN_AT_OUTPUT_CHECK.clear()
    app = make_app(output_guards=["tp_spy_out"])  # buffered_sentences + RAG + require_citations
    h = make_harness(app, Script(answer("Frase um. Frase dois. Frase três.", ["T1"]), chunk=3))
    await collect(ChatPipeline(h.deps), make_ctx(app), sink=LIVE_EVENTS)
    assert len(SEEN_AT_OUTPUT_CHECK) == 1
    assert not any(isinstance(e, TextDelta) for e in SEEN_AT_OUTPUT_CHECK[0])
    deltas = [e.text for e in LIVE_EVENTS if isinstance(e, TextDelta)]
    assert deltas == ["Frase um. Frase dois. Frase três."]


async def test_answer_grounded_on_mcp_tool_needs_no_citation():
    server, mcp = tarefas_server()
    app = make_app(mcp=mcp, output_guards=["citations"], cache=False)
    script = Script(Call("listar_tarefas", {}), answer("Você não tem tarefas.", []))
    h = make_harness(app, script, mcp_source=lambda cfg, headers: server)
    ctx = make_ctx(app, "quais são minhas tarefas?")
    events = await collect(ChatPipeline(h.deps), ctx)
    assert ctx.tools_called == ["listar_tarefas"]
    assert final_of(events).output.resposta == "Você não tem tarefas."


async def test_closing_stream_early_cancels_agent():
    app = make_app(rag=False)
    h = make_harness(app, Script(answer("Primeira. Segunda. Terceira.", []), chunk=2, delay=0))
    pipeline = ChatPipeline(h.deps)
    stream = pipeline.stream(make_ctx(app))
    first = await anext(stream)
    assert isinstance(first, TextDelta)
    await stream.aclose()  # cliente desconectou
    assert h.history.appended == []
    leftovers = [
        t for t in asyncio.all_tasks() if t.get_name().startswith("encaixe.agent") and not t.done()
    ]
    assert leftovers == []


# --- resposta em texto puro (fora da ferramenta `responder`) --------------------------------


async def test_text_answer_after_mcp_tool_is_accepted_and_released_by_sentence():
    """Turno apoiado numa ferramenta MCP não precisa de citação: o texto puro vale como
    resposta (sem nova tentativa) e sai frase a frase, token a token."""
    server, mcp = tarefas_server()
    app = make_app(mcp=mcp, output_guards=["citations"], cache=False)  # RAG + require_citations
    text = "Você tem duas tarefas. A primeira vence hoje. A segunda, amanhã."
    script = Script(Call("listar_tarefas", {}), text, chunk=3)
    h = make_harness(app, script, mcp_source=lambda cfg, headers: server)
    events = await collect(ChatPipeline(h.deps), make_ctx(app, "quais são minhas tarefas?"))
    assert script.calls == 2
    out = final_of(events).output
    assert out.resposta == text and out.citacoes == []
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas[:2] == ["Você tem duas tarefas. ", "A primeira vence hoje. "]
    assert "".join(deltas) == text


async def test_text_answer_without_rag_is_accepted():
    app = make_app(rag=False)
    h = make_harness(app, Script("Primeira frase. Segunda frase.", chunk=3))
    events = await collect(ChatPipeline(h.deps), make_ctx(app, "oi"))
    assert h.script.calls == 1
    assert final_of(events).output.resposta == "Primeira frase. Segunda frase."
    assert [e.text for e in events if isinstance(e, TextDelta)] == [
        "Primeira frase. ",
        "Segunda frase.",
    ]


async def test_text_answer_in_document_turn_is_retried_through_responder():
    """Turno de documentos: texto puro perderia as citações, então o modelo é mandado de volta
    à ferramenta `responder` (TEXT_ANSWER_RETRY) e a resposta citada é a que vale."""
    app = make_app()  # RAG + require_citations, sem ferramentas
    script = Script("São 30 dias corridos.", answer("São 30 dias corridos.", ["T1"]))
    h = make_harness(app, script)
    events = await collect(ChatPipeline(h.deps), make_ctx(app))
    assert script.calls == 2
    assert TEXT_ANSWER_RETRY in str(script.messages[1][-1].parts[-1].content)
    out = final_of(events).output
    assert out.resposta == "São 30 dias corridos." and out.citacoes == ["k1"]


async def test_tool_call_markup_in_text_never_reaches_the_stream():
    server, mcp = tarefas_server()
    app = make_app(mcp=mcp, cache=False)
    leaked = "Você tem duas tarefas. A primeira vence hoje.\n<tool_use>\n<parameter=tool_name>\n"
    final = answer("Você tem duas tarefas. A primeira vence hoje. A segunda, amanhã.", [])
    script = Script(Call("listar_tarefas", {}), leaked, final, chunk=3)
    h = make_harness(app, script, mcp_source=lambda cfg, headers: server)
    events = await collect(ChatPipeline(h.deps), make_ctx(app, "quais são minhas tarefas?"))
    assert script.calls == 3
    assert "<tool_use>" not in str(events) and "<parameter" not in str(events)
    assert final_of(events).output.resposta == final["resposta"]
    # o já liberado ("…hoje.\n") não é prefixo da resposta nova: retratação com o texto certo
    assert text_of(events) == "Você tem duas tarefas. A primeira vence hoje.\n"
    assert Retraction(final["resposta"]) in events
