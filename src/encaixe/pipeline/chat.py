"""Pipeline de chat por requisição (spec §6.1): guards → aprovação → histórico → cache →
RAG → contexto → agente em streaming → guards de saída → cache → persistência."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress
from typing import TYPE_CHECKING

import structlog

from encaixe.agent.factory import build_agent, run_outcome_on_limit, usage_limits_for
from encaixe.cache.exact import exact_key, is_cacheable
from encaixe.context.aliases import resolve_citations
from encaixe.context.builder import BuiltContext, ContextBuilder, conversation_messages
from encaixe.context.untrusted import new_boundary
from encaixe.core.pii import mask_pii
from encaixe.core.types import ChatMessage, RequestContext
from encaixe.guardrails.canary import make_canary
from encaixe.guardrails.runner import blocked, run_guards, run_output_guards
from encaixe.messages import ALL_FAILED, GUARD_UNAVAILABLE, NOT_FOUND, REFUSAL_INPUT, RETRACTION
from encaixe.observability.spans import current_ctx, set_current_ctx, traced
from encaixe.output.schemas import RespostaBase
from encaixe.pipeline.events import ChatEvent, Final, Retraction, Status, TextDelta
from encaixe.pipeline.stages import (
    AgentUpdate,
    Turn,
    build_toolsets,
    current_trace_id,
    drive_agent,
    make_releaser,
    may_write,
    model_specs,
    not_found_reason,
    post_terminal_if_missing,
    primary_model_name,
    span_attr,
    static_output,
)
from encaixe.tools.native import AgentDeps

if TYPE_CHECKING:
    from encaixe.wiring import Deps

log = structlog.get_logger(__name__)


class ChatPipeline:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.builder = ContextBuilder(deps.counter)

    async def stream(self, ctx: RequestContext) -> AsyncIterator[ChatEvent]:
        """Eventos da resposta; o último é sempre `Final`. Nunca levanta (exceto cancelamento)."""
        # Canário estável por app: o system prompt fica idêntico entre requisições (prefix cache).
        ctx.canary = self.deps.canaries.setdefault(ctx.app.config.id, ctx.canary or make_canary())
        token = set_current_ctx(ctx)
        emitted_text = finished = False
        try:
            async with aclosing(self._run(ctx)) as events:
                async for event in events:
                    emitted_text = emitted_text or isinstance(event, TextDelta)
                    finished = isinstance(event, Final)
                    yield event
        except Exception:  # rede de segurança: o stream termina bem formado
            log.exception("pipeline_failed", request_id=ctx.request_id)
            if not finished:
                yield Retraction(ALL_FAILED) if emitted_text else TextDelta(ALL_FAILED)
                yield Final(static_output(ALL_FAILED), ctx.conversation_id, ctx.request_id)
        finally:
            with suppress(ValueError):  # fechado em outro contexto (GC): nada a restaurar
                current_ctx.reset(token)

    @traced("pipeline.chat")
    async def _run(self, ctx: RequestContext) -> AsyncIterator[ChatEvent]:
        trace_id = current_trace_id() or ctx.request_id
        spec, fallbacks = model_specs(ctx, self.deps.settings)
        turn = Turn(make_releaser(ctx, self.deps.settings), spec, fallbacks)

        refusal = await self._input_guards(ctx, turn)
        if refusal is not None:
            yield TextDelta(refusal)
            yield Final(static_output(refusal), ctx.conversation_id, trace_id)
            return

        await self._resolve_approval(ctx)
        await self._load_history(ctx, turn)
        cached = await self._cache_lookup(ctx, turn)
        if cached is not None:
            turn.output = cached
            if cached.resposta:
                yield TextDelta(cached.resposta)
        else:
            await self._retrieve(ctx)
            built = self._build_context(ctx, turn)
            # aclosing: se o cliente sair, o agente é cancelado agora, não na coleta de lixo.
            async with aclosing(self._agent(ctx, built, turn)) as agent_events:
                async for event in agent_events:
                    yield event
            for event in await self._check_output(ctx, turn):
                yield event
            await self._cache_store(ctx, turn)

        await self._persist(ctx, turn, trace_id)
        assert turn.output is not None
        yield Final(turn.output, ctx.conversation_id, trace_id)

    # --- 1. guards de entrada ---------------------------------------------------------------

    @traced("pipeline.input_guards")
    async def _input_guards(self, ctx: RequestContext, turn: Turn) -> str | None:
        g = ctx.app.config.guardrails
        results = await run_guards(
            g.input,
            ctx.user_message,
            ctx,
            settings=self.deps.settings,
            fail_closed=g.on_guard_unavailable == "block",
        )
        ctx.guard_results.extend(results)
        hit = blocked(results)
        if hit is not None:
            span_attr("guard.blocked", hit.guard)
            return GUARD_UNAVAILABLE if hit.reason == "unavailable" else REFUSAL_INPUT
        for r in results:
            if r.guard == "normalize" and r.redacted_text is not None:
                ctx.user_message = r.redacted_text
        # O modelo recebe o texto normalizado; o histórico guarda a versão mascarada.
        turn.persist_user = ctx.user_message
        for r in results:
            if r.guard == "pii" and r.verdict == "flag" and r.redacted_text is not None:
                turn.persist_user, turn.pii_flagged = r.redacted_text, True
        return None

    # --- 2. aprovação pendente --------------------------------------------------------------

    @traced("pipeline.approvals")
    async def _resolve_approval(self, ctx: RequestContext) -> None:
        if not may_write(ctx):  # mesma regra das ferramentas de escrita
            return
        try:
            result = await self.deps.approvals.resolve_turn(ctx, new_boundary())
        except Exception as exc:  # noqa: BLE001 - a aprovação fica pendente; o turno segue
            log.warning("approval_resolve_failed", error=type(exc).__name__)
            span_attr("approval.error", type(exc).__name__)
            return
        if result is not None:
            ctx.extra_context.append(result)

    # --- histórico (servidor com identidade de conversa; senão, o do cliente) ---------------

    @traced("pipeline.history")
    async def _load_history(self, ctx: RequestContext, turn: Turn) -> None:
        if not (ctx.has_conversation_identity and ctx.conversation_id):
            turn.history = conversation_messages(ctx.client_history)
            return
        store, app_id = self.deps.history_store, ctx.app.config.id
        try:
            stored = await store.get_summary(app_id, ctx.conversation_id)
            covers = 0
            if stored is not None:
                turn.summary, covers = stored
            turn.history = await store.list_messages(app_id, ctx.conversation_id, after_seq=covers)
        except Exception as exc:  # noqa: BLE001 - responde sem histórico
            log.warning("history_load_failed", error=type(exc).__name__)
            span_attr("history.error", type(exc).__name__)
            turn.summary, turn.history = None, []

    # --- 3. cache exato (fail-open) ---------------------------------------------------------

    @traced("pipeline.cache_lookup")
    async def _cache_lookup(self, ctx: RequestContext, turn: Turn) -> RespostaBase | None:
        cache = self.deps.cache
        if cache is None or not ctx.app.config.cache.exact.enabled:
            return None
        if ctx.extra_context or turn.pii_flagged:
            return None  # nunca seria gravado: is_cacheable recusa
        history = conversation_messages(turn.history)
        if turn.summary:
            history = [ChatMessage("system", turn.summary), *history]
        try:
            turn.cache_key = exact_key(
                app_id=ctx.app.config.id,
                prompt_version=ctx.app.prompt_version,
                model=turn.model_spec,
                principals=ctx.principals,
                history=history,
                user_message=ctx.user_message,
                corpus_version=await self.deps.corpus_version(ctx.app.config.id),
            )
            raw = await cache.get(turn.cache_key)
            output = None if raw is None else ctx.app.output_model.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - Redis/JSON/validação: vira miss
            log.warning("cache_error", op="get", error=type(exc).__name__)
            span_attr("cache.error", type(exc).__name__)
            return None
        span_attr("cache.hit", output is not None)
        if output is not None:
            ctx.cache_layer_hit = "exact"
            span_attr("cache.layer_hit", "exact")  # atributo do spec §6.13
        return output

    # --- 4. retrieval -----------------------------------------------------------------------

    @traced("pipeline.retrieve")
    async def _retrieve(self, ctx: RequestContext) -> None:
        rag = ctx.app.config.rag
        if not rag.enabled:
            return
        try:
            ctx.retrieved = await self.deps.retriever.retrieve(
                ctx.app.config.id, ctx.principals, ctx.user_message, rag.top_k, rag.rerank_top_n
            )
        except Exception as exc:  # noqa: BLE001 - RAG fora: segue sem trechos (vira NOT_FOUND)
            log.warning("retrieval_failed", error=type(exc).__name__)
            span_attr("rag.error", type(exc).__name__)
            ctx.retrieved = []
        span_attr("rag.retrieved", len(ctx.retrieved))

    # --- 5. contexto ------------------------------------------------------------------------

    @traced("pipeline.context")
    def _build_context(self, ctx: RequestContext, turn: Turn) -> BuiltContext:
        built = self.builder.build(
            ctx.app,
            canary=ctx.canary,
            summary=turn.summary,
            history=turn.history,
            retrieved=ctx.retrieved,
            extra_context=ctx.extra_context,
            user_message=ctx.user_message,
        )
        # A ferramenta buscar_documentos continua a numeração (T3, T4…) neste mesmo mapa.
        ctx.citation_aliases = dict(built.citation_aliases)
        return built

    # --- 6. agente em streaming -------------------------------------------------------------

    @traced("pipeline.agent")
    async def _agent(
        self, ctx: RequestContext, built: BuiltContext, turn: Turn
    ) -> AsyncIterator[ChatEvent]:
        try:
            model, effective = self.deps.chain_factory(turn.model_spec, turn.fallbacks)
            agent = build_agent(ctx.app, model, build_toolsets(ctx, self.deps.mcp_source))
        except Exception as exc:  # noqa: BLE001 - modelo/ferramenta mal configurado
            log.error("agent_setup_failed", error=type(exc).__name__, detail=str(exc))
            turn.finish_static(ALL_FAILED)
            return
        primary_name = primary_model_name(model)
        if effective[0] != turn.model_spec:  # breaker aberto: a cadeia já começa no reserva
            ctx.fallback_used = True
            yield Status("fallback", {"model": effective[0], "reason": "breaker_open"})
        deps = AgentDeps(
            ctx=ctx,
            retriever=self.deps.retriever,
            injection=self.deps.injection,
            approvals=self.deps.approvals,
            boundary=built.boundary,
        )
        queue: asyncio.Queue[AgentUpdate] = asyncio.Queue()
        task = asyncio.create_task(
            drive_agent(
                agent, built, deps, usage_limits_for(ctx.app), ctx.app.config.agent.timeout_s, queue
            ),
            name=f"encaixe.agent.{ctx.request_id}",
        )
        task.add_done_callback(lambda t: post_terminal_if_missing(t, queue))
        try:
            while True:
                update = await queue.get()
                if update.kind == "partial":
                    if delta := await turn.releaser.feed(update.value):
                        yield TextDelta(delta)
                elif update.kind == "tool":
                    yield Status("tool", {"name": update.value})
                elif update.kind == "model":
                    turn.model_name = update.value
                    if update.value != primary_name and not ctx.fallback_used:
                        ctx.fallback_used = True
                        yield Status("fallback", {"model": update.value})
                else:
                    self._agent_outcome(ctx, turn, update)
                    break
        finally:
            if not task.done():  # consumidor saiu (cliente desconectou): para o agente
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        span_attr("fallback.used", ctx.fallback_used)

    def _agent_outcome(self, ctx: RequestContext, turn: Turn, update: AgentUpdate) -> None:
        if update.kind == "done":
            turn.output = update.value
        elif update.kind == "limit":
            turn.output, turn.synthetic = run_outcome_on_limit(update.value, ctx.tools_called), True
            span_attr("agent.limit", type(update.value).__name__)
        else:
            error = "timeout" if update.kind == "timeout" else type(update.value).__name__
            log.warning("agent_failed", error=error, detail=str(update.value or ""))
            span_attr("agent.error", error)
            turn.finish_static(ALL_FAILED)

    # --- 7. guards de saída -----------------------------------------------------------------

    @traced("pipeline.output_guards")
    async def _check_output(self, ctx: RequestContext, turn: Turn) -> list[ChatEvent]:
        assert turn.output is not None
        if not turn.synthetic:
            self._resolve_citations(ctx, turn)
            if (reason := not_found_reason(ctx, turn.output)) is not None:
                span_attr("not_found", reason)
                turn.finish_static(NOT_FOUND)
        retract = False
        if not turn.synthetic:
            g = ctx.app.config.guardrails
            turn.output_results = await run_output_guards(
                g.output,
                turn.output,
                ctx,
                settings=self.deps.settings,
                fail_closed=g.on_guard_unavailable == "block",
            )
            ctx.guard_results.extend(turn.output_results)
            if (hit := blocked(turn.output_results)) is not None:
                span_attr("guard.blocked", hit.guard)
                # Guard fora do ar (fail-closed) não é violação de política.
                turn.finish_static(GUARD_UNAVAILABLE if hit.reason == "unavailable" else RETRACTION)
                retract = True
        return turn.releaser.closing(turn.output.resposta, retract=retract)

    def _resolve_citations(self, ctx: RequestContext, turn: Turn) -> None:
        """Aliases (T1…) e variantes ("t1", "[T1]", "id: T1", UUID) → ids reais dos trechos
        recuperados; o que não resolve sai (antes dos guards: citação ruim não é RETRACTION)."""
        assert turn.output is not None
        cited, dropped = resolve_citations(
            turn.output.citacoes, ctx.citation_aliases, (s.chunk.id for s in ctx.retrieved)
        )
        span_attr("citations.dropped", dropped)
        if cited != turn.output.citacoes:
            turn.output = turn.output.model_copy(update={"citacoes": cited})

    # --- 8. gravação no cache (fail-open) ---------------------------------------------------

    @traced("pipeline.cache_store")
    async def _cache_store(self, ctx: RequestContext, turn: Turn) -> None:
        if self.deps.cache is None or turn.cache_key is None or turn.synthetic:
            return
        # A chave usa o modelo principal: resposta do reserva não fica servida pelo TTL inteiro.
        if ctx.fallback_used or not is_cacheable(ctx, turn.output_results):
            return
        try:
            await self.deps.cache.set(turn.cache_key, turn.output, ctx.app.config.cache.exact.ttl_s)
        except Exception as exc:  # noqa: BLE001
            log.warning("cache_error", op="set", error=type(exc).__name__)
            span_attr("cache.error", type(exc).__name__)

    # --- 9. persistência e resumo assíncrono ------------------------------------------------

    @traced("pipeline.persist")
    async def _persist(self, ctx: RequestContext, turn: Turn, trace_id: str) -> None:
        if ctx.conversation_id is None or turn.output is None:
            return
        app_id, cid, count = ctx.app.config.id, ctx.conversation_id, self.deps.counter.count
        # Turno com dado pessoal: a resposta gravada (lida depois pelo resumidor) também vai
        # mascarada; a que o usuário recebeu não muda.
        answer = mask_pii(turn.output.resposta) if turn.pii_flagged else turn.output.resposta
        tools = [{"name": t} for t in ctx.tools_called] or None
        meta = {"trace_id": trace_id, "prompt_version": ctx.app.prompt_version}
        try:
            user = ChatMessage("user", turn.persist_user, tokens=count(turn.persist_user))
            await self.deps.history_store.append(app_id, cid, user, **meta)
            reply = ChatMessage("assistant", answer, tool_calls=tools, tokens=count(answer))
            await self.deps.history_store.append(app_id, cid, reply, **meta, model=turn.model_name)
        except Exception as exc:  # noqa: BLE001 - a resposta já foi dada; registra e segue
            log.error("persist_failed", error=type(exc).__name__)
            span_attr("persist.error", type(exc).__name__)
            return
        # Sem identidade de conversa o histórico vem do cliente: um resumo nunca seria lido.
        if ctx.has_conversation_identity and self.deps.enqueue is not None:
            try:
                await self.deps.enqueue("summarize_conversation", app_id, cid)
            except Exception as exc:  # noqa: BLE001
                log.warning("summary_enqueue_failed", error=type(exc).__name__)
