"""Peças auxiliares do pipeline de chat: estado do turno, liberação por frase, montagem do
modelo e das ferramentas, execução do agente numa tarefa própria (com prazo) e toolsets MCP
tolerantes a servidor fora do ar."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import structlog
from opentelemetry import trace
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.usage import UsageLimits

from encaixe.agent.loop_guard import LoopDetected, wrap_toolsets
from encaixe.context.builder import BuiltContext
from encaixe.core.types import ChatMessage, GuardResult, RequestContext
from encaixe.guardrails.runner import blocked, run_guards
from encaixe.output.schemas import RespostaBase
from encaixe.pipeline.events import ChatEvent, Retraction, TextDelta
from encaixe.settings import Settings
from encaixe.tools.mcp import build_mcp_toolsets
from encaixe.tools.native import SEARCH_TOOL, AgentDeps, build_native_toolset

log = structlog.get_logger(__name__)

# Fim de frase: pontuação seguida de espaço, ou quebra de linha.
SENTENCE_END = re.compile(r"[.!?…]\s|\n")
# Checagens rápidas (regex) aplicadas antes de liberar cada frase no modo buffered_sentences.
QUICK_GUARDS = ("pii_leak", "canary")
RAG_TOOL = SEARCH_TOOL


def span_attr(key: str, value: str | bool | int) -> None:
    trace.get_current_span().set_attribute(key, value)


def current_trace_id() -> str | None:
    sc = trace.get_current_span().get_span_context()
    return format(sc.trace_id, "032x") if sc.is_valid else None


def static_output(text: str) -> RespostaBase:
    """Resposta fixa (recusa, falha, NOT_FOUND...): nunca leva citações."""
    return RespostaBase(resposta=text, citacoes=[])


def release_point(text: str, start: int) -> int:
    """Índice logo depois do último fim de frase em `text[start:]` (ou `start`)."""
    end = start
    for match in SENTENCE_END.finditer(text, start):
        end = match.end()
    return end


def grounded_elsewhere(ctx: RequestContext) -> bool:
    """A resposta se apoia em algo além dos documentos (ação aprovada ou ferramenta MCP)."""
    return bool(ctx.extra_context) or any(t != RAG_TOOL for t in ctx.tools_called)


def text_answer_allowed(ctx: RequestContext) -> bool:
    """Resposta em texto puro (fora da ferramenta `responder`) vale, com `citacoes` vazia,
    quando o turno não precisa de citação: app sem RAG, ou resposta apoiada numa ferramenta
    MCP ou numa ação aprovada (`grounded_elsewhere`). Avaliado na hora da resposta."""
    return not ctx.app.config.rag.enabled or grounded_elsewhere(ctx)


def may_become_not_found(ctx: RequestContext) -> bool:
    """A resposta final ainda pode virar NOT_FOUND (então nada é liberado antes do fim).

    Com RAG ligado e sem outra base (`grounded_elsewhere`): nada recuperado, ou
    `require_citations` (sem citação válida no fim, vira NOT_FOUND). Por isso apps com
    `require_citations` ficam, na prática, em modo `strict` nos turnos de documentos; o
    `buffered_sentences` volta a liberar por frase quando a resposta passa a se apoiar
    numa ferramenta MCP ou numa ação aprovada.
    """
    rag = ctx.app.config.rag
    if not rag.enabled or grounded_elsewhere(ctx):
        return False
    return not ctx.retrieved or rag.require_citations


def not_found_reason(ctx: RequestContext, output: RespostaBase) -> str | None:
    """Por que a resposta (com citações já resolvidas) vira NOT_FOUND, ou None."""
    if not may_become_not_found(ctx):
        return None
    if not ctx.retrieved:
        return "no_retrieval"  # qualquer resposta (e citação) seria inventada
    if not output.citacoes:
        return "no_valid_citation"  # require_citations: resposta sem base nos trechos
    return None


def may_write(ctx: RequestContext) -> bool:
    """Ferramentas de escrita e resolução de aprovações: só numa conversa com identidade de
    servidor e de um usuário identificado. Sem `user_ref`, quem repete o `X-Conversation-Id`
    recebido (cliente anônimo) não ganha escrita."""
    return (
        ctx.has_conversation_identity
        and ctx.conversation_id is not None
        and ctx.principals.user_ref is not None
    )


def primary_model_name(model: Model) -> str:
    inner = model.models[0] if isinstance(model, FallbackModel) else model
    return inner.model_name


class SentenceReleaser:
    """Decide quanto do texto parcial pode sair antes dos guards de saída.

    Só libera frases completas, só quando `may_release()` e só se o texto liberado até
    ali passa em `quick_check`. Uma reprovação segura o resto até os guards de saída.
    No modo `strict`, `may_release` é sempre falso.
    """

    def __init__(
        self,
        quick_check: Callable[[str], Awaitable[bool]] | None,
        may_release: Callable[[], bool],
    ) -> None:
        self.released = ""
        self._check = quick_check
        self._may_release = may_release
        self._held = False

    async def feed(self, text: str) -> str:
        """Recebe o `resposta` parcial atual; devolve o trecho novo a emitir (ou "")."""
        if self._held or not text.startswith(self.released) or not self._may_release():
            return ""
        cut = release_point(text, len(self.released))
        if cut <= len(self.released):
            return ""
        candidate = text[:cut]
        if self._check is not None and not await self._check(candidate):
            self._held = True
            return ""
        delta, self.released = candidate[len(self.released) :], candidate
        return delta

    def closing(self, final_text: str, *, retract: bool) -> list[ChatEvent]:
        """Eventos que fecham o texto: o restante, ou uma retratação se o já liberado não vale."""
        if not retract and final_text.startswith(self.released):
            rest = final_text[len(self.released) :]
            return [TextDelta(rest)] if rest else []
        return [Retraction(final_text)]


def make_releaser(ctx: RequestContext, settings: Settings) -> SentenceReleaser:
    if ctx.app.config.streaming.mode == "strict":
        return SentenceReleaser(None, lambda: False)
    quick = [n for n in ctx.app.config.guardrails.output if n in QUICK_GUARDS]

    async def check(text: str) -> bool:
        # O texto inteiro até o corte: um canário partido entre frases também é pego.
        if not quick:
            return True
        results = await run_guards(quick, text, ctx, settings=settings, fail_closed=True)
        return blocked(results) is None

    return SentenceReleaser(check, lambda: not may_become_not_found(ctx))


@dataclass
class Turn:
    """Estado de uma requisição que não pertence ao RequestContext."""

    releaser: SentenceReleaser
    model_spec: str
    fallbacks: list[str]
    persist_user: str = ""
    pii_flagged: bool = False
    summary: str | None = None
    history: list[ChatMessage] = field(default_factory=list)
    cache_key: str | None = None
    output: RespostaBase | None = None
    synthetic: bool = False  # resposta fixa: sem guards de saída e sem cache
    model_name: str | None = None
    output_results: list[GuardResult] = field(default_factory=list)

    def finish_static(self, text: str) -> None:
        self.output, self.synthetic = static_output(text), True


def model_specs(ctx: RequestContext, settings: Settings) -> tuple[str, list[str]]:
    """(principal, reservas). O override só vale em dev e dispensa a cadeia (bake-off)."""
    models = ctx.app.config.models
    if ctx.model_override:
        if settings.env == "dev":
            return ctx.model_override, []
        log.warning("model_override_ignored", model=ctx.model_override)
    return models.primary, list(models.fallbacks)


def build_toolsets(ctx: RequestContext, mcp_source: Any = None) -> list[AbstractToolset[Any]]:
    """Nativas + MCP (escrita só com `may_write`), cada uma com um LoopGuard novo."""
    tools = ctx.app.config.tools
    toolsets: list[AbstractToolset[Any]] = []
    if tools.native:
        toolsets.append(build_native_toolset(tools.native))
    mcp = build_mcp_toolsets(ctx.app, ctx, allow_write=may_write(ctx), source=mcp_source)
    toolsets += [TolerantToolset(ts, s.name) for ts, s in zip(mcp, tools.mcp_servers, strict=True)]
    return wrap_toolsets(toolsets)


class TolerantToolset(WrapperToolset[Any]):
    """Servidor MCP fora do ar não derruba a execução: o toolset fica sem ferramentas."""

    def __init__(self, wrapped: AbstractToolset[Any], server: str) -> None:
        super().__init__(wrapped)
        self.server = server
        self._down = False
        self._depth = 0  # entradas bem-sucedidas ainda não fechadas

    def _mark_down(self, stage: str, exc: Exception) -> None:
        self._down = True
        log.warning(
            "mcp_server_unavailable", server=self.server, stage=stage, error=type(exc).__name__
        )
        trace.get_current_span().add_event("mcp_server_unavailable", {"server": self.server})

    async def __aenter__(self) -> Self:
        if not self._down:
            try:
                await self.wrapped.__aenter__()
                self._depth += 1
            except Exception as exc:  # noqa: BLE001 - qualquer falha de conexão degrada
                self._mark_down("connect", exc)
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        if self._depth == 0:
            return None
        self._depth -= 1
        try:
            return await self.wrapped.__aexit__(*args)
        except Exception as exc:  # noqa: BLE001
            log.warning("mcp_server_close_failed", server=self.server, error=type(exc).__name__)
            return None

    async def get_tools(self, ctx: Any) -> dict[str, Any]:
        if self._down:
            return {}
        try:
            return await self.wrapped.get_tools(ctx)
        except Exception as exc:  # noqa: BLE001
            self._mark_down("list_tools", exc)
            return {}

    async def get_instructions(self, ctx: Any) -> Any:
        if self._down:
            return None
        try:
            return await super().get_instructions(ctx)
        except Exception as exc:  # noqa: BLE001
            self._mark_down("instructions", exc)
            return None


@dataclass
class AgentUpdate:
    kind: Literal["partial", "model", "tool", "done", "limit", "timeout", "error"]
    value: Any = None


def post_terminal_if_missing(task: asyncio.Task[None], queue: asyncio.Queue[AgentUpdate]) -> None:
    """Callback de término da tarefa do agente: se ela morreu sem publicar o desfecho
    (BaseException fora de `Exception`, cancelamento externo), publica um erro para o
    consumidor não esperar `queue.get()` para sempre segurando a vaga de geração."""
    if task.cancelled():
        queue.put_nowait(AgentUpdate("error", asyncio.CancelledError()))
    elif (exc := task.exception()) is not None:
        log.error("agent_driver_crashed", error=type(exc).__name__)
        queue.put_nowait(AgentUpdate("error", exc))


async def drive_agent(
    agent: Agent[Any, RespostaBase],
    built: BuiltContext,
    deps: AgentDeps,
    usage_limits: UsageLimits,
    timeout_s: float,
    queue: asyncio.Queue[AgentUpdate],
) -> None:
    """Roda o agente e publica o andamento em `queue`; termina sempre com done/limit/timeout/error.

    Roda numa tarefa própria para que o `asyncio.timeout` cubra só o agente (e não o
    consumidor do stream) e para que o cancelamento pelo consumidor seja limpo.
    Usa `iter()` com `node.stream()` (e não `run_stream`) para manter as novas tentativas
    de saída inválida: `run_stream` não as suporta.
    """
    try:
        async with asyncio.timeout(timeout_s):
            async with agent.iter(
                built.user_prompt,
                message_history=built.history,
                instructions=built.system_prompt,
                deps=deps,
                usage_limits=usage_limits,
            ) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as stream:
                            try:
                                async for partial in stream.stream_output(debounce_by=None):
                                    queue.put_nowait(AgentUpdate("partial", partial.resposta))
                            except (ValidationError, UnexpectedModelBehavior):
                                pass  # saída inválida: o grafo pede nova tentativa ao modelo
                            queue.put_nowait(AgentUpdate("model", stream.response.model_name))
                    elif Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as events:
                            async for event in events:
                                if isinstance(event, FunctionToolCallEvent):
                                    queue.put_nowait(AgentUpdate("tool", event.part.tool_name))
                result = run.result
        if result is None:
            raise UnexpectedModelBehavior("execução terminou sem resultado")
        queue.put_nowait(AgentUpdate("done", result.output))
    except (UsageLimitExceeded, LoopDetected) as exc:
        queue.put_nowait(AgentUpdate("limit", exc))
    except TimeoutError:
        queue.put_nowait(AgentUpdate("timeout"))
    except Exception as exc:  # noqa: BLE001 - inclui FallbackExceptionGroup (todos falharam)
        queue.put_nowait(AgentUpdate("error", exc))
