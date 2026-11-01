"""Fakes e construtores compartilhados pelos testes do pipeline."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import jinja2
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.profiles import ModelProfile

from encaixe.core.config import (
    AgentConfig,
    AppConfig,
    CacheConfig,
    ExactCacheConfig,
    GuardrailsConfig,
    LoadedApp,
    McpServerConfig,
    ModelsConfig,
    RagConfig,
    StreamingConfig,
    ToolsConfig,
)
from encaixe.core.types import (
    ChatMessage,
    Chunk,
    GuardResult,
    Principals,
    RequestContext,
    ScoredChunk,
)
from encaixe.output.schemas import RespostaBase
from encaixe.pipeline.events import Final, TextDelta
from encaixe.settings import Settings
from encaixe.wiring import Deps

PROFILE = ModelProfile(supports_json_schema_output=True)
PRIMARY = "test:primary"


def make_app(
    *,
    app_id: str = "a",
    input_guards: list[str] | None = None,
    output_guards: list[str] | None = None,
    on_unavailable: str = "block",
    mode: str = "buffered_sentences",
    rag: bool = True,
    native: list[str] | None = None,
    mcp: list[McpServerConfig] | None = None,
    cache: bool = True,
    fallbacks: list[str] | None = None,
    timeout_s: int = 30,
) -> LoadedApp:
    cfg = AppConfig(
        id=app_id,
        models=ModelsConfig(primary=PRIMARY, fallbacks=fallbacks or []),
        rag=RagConfig(enabled=rag, top_k=10, rerank_top_n=3),
        tools=ToolsConfig(native=native or [], mcp_servers=mcp or []),
        agent=AgentConfig(timeout_s=timeout_s),
        cache=CacheConfig(exact=ExactCacheConfig(enabled=cache, ttl_s=60)),
        guardrails=GuardrailsConfig(
            input=input_guards or [],
            output=output_guards or [],
            on_guard_unavailable=on_unavailable,
        ),
        streaming=StreamingConfig(mode=mode),
    )
    return LoadedApp(
        config=cfg,
        root=None,  # type: ignore[arg-type]
        system_prompt=jinja2.Template("Você é o assistente. Código de controle: {{ canary }}."),
        prompt_version="v1",
        output_model=RespostaBase,
    )


def make_ctx(
    app: LoadedApp,
    message: str = "Qual é a política de férias?",
    *,
    identity: bool = False,
    conversation_id: str | None = "c1",
    history: list[ChatMessage] | None = None,
    principals: Principals | None = None,
    canary: str = "",
    override: str | None = None,
) -> RequestContext:
    return RequestContext(
        request_id="r1",
        app=app,
        principals=principals or Principals(None),
        conversation_id=conversation_id,
        has_conversation_identity=identity,
        user_message=message,
        client_history=history or [],
        canary=canary,
        model_override=override,
    )


def chunk(id_: str = "k1", text: str = "Política de férias: 30 dias corridos.") -> ScoredChunk:
    return ScoredChunk(Chunk(id_, "a", "d1", text, ("RH",), ("app:*",), {"title": "RH"}), 0.9)


def answer(text: str, cites: list[str] | None = None) -> dict:
    return {"resposta": text, "citacoes": cites if cites is not None else ["k1"]}


@dataclass
class Call:
    """Passo do roteiro em que o modelo chama uma ferramenta."""

    tool: str
    args: dict


class Script:
    """Modelo de teste (FunctionModel em streaming) que segue um roteiro e registra o que viu."""

    def __init__(self, *steps: Any, name: str = "primary", chunk: int = 4, delay: float = 0.0):
        self.steps = list(steps) or [answer("Resposta padrão.")]
        self.name = name
        self.chunk = chunk
        self.delay = delay
        self.infos: list[Any] = []
        self.messages: list[list[Any]] = []

    def model(self) -> FunctionModel:
        return FunctionModel(stream_function=self._stream, profile=PROFILE, model_name=self.name)

    async def _stream(self, messages, info):
        self.infos.append(info)
        self.messages.append(list(messages))
        step = self.steps[min(len(self.infos), len(self.steps)) - 1]
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(step, Call):
            yield {0: DeltaToolCall(name=step.tool, json_args=json.dumps(step.args))}
            return
        # str: texto cru (ex.: JSON inválido); dict: saída estruturada
        text = step if isinstance(step, str) else json.dumps(step, ensure_ascii=False)
        for i in range(0, len(text), self.chunk):
            yield text[i : i + self.chunk]

    @property
    def calls(self) -> int:
        return len(self.infos)

    def prompt(self, call: int = 0) -> str:
        """Todo o texto de usuário (histórico + mensagem atual) enviado na chamada `call`."""
        return "\n".join(
            p.content
            for m in self.messages[call]
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )

    def system_parts(self, call: int = 0) -> list[str]:
        return [
            p.content
            for m in self.messages[call]
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, SystemPromptPart)
        ]

    def tool_names(self, call: int = 0) -> set[str]:
        return {t.name for t in self.infos[call].function_tools}


class FailingModel:
    """Modelo que sempre falha como um Ollama fora do ar."""

    @staticmethod
    def build(name: str) -> FunctionModel:
        from pydantic_ai.exceptions import ModelHTTPError

        async def stream(messages, info):
            raise ModelHTTPError(503, name, "fora do ar")
            yield ""  # pragma: no cover

        return FunctionModel(stream_function=stream, profile=PROFILE, model_name=name)


class FakeHistory:
    def __init__(self, messages: list[ChatMessage] | None = None, summary=None):
        self.messages = messages or []
        self.summary = summary
        self.appended: list[tuple[str, str]] = []
        self.append_kwargs: list[dict] = []
        self.listed = 0

    async def create_conversation(self, app_id, user_ref):
        raise AssertionError("o pipeline não cria conversas")

    async def get_owner(self, app_id, conversation_id):
        raise AssertionError("o pipeline não valida dono")

    async def append(
        self, app_id, conversation_id, msg, *, trace_id=None, prompt_version=None, model=None
    ):
        self.appended.append((msg.role, msg.content))
        self.append_kwargs.append(
            {"trace_id": trace_id, "prompt_version": prompt_version, "model": model}
        )
        return len(self.appended)

    async def list_messages(self, app_id, conversation_id, after_seq=0):
        self.listed += 1
        return [m for m in self.messages if (m.seq or 0) > after_seq]

    async def get_summary(self, app_id, conversation_id):
        return self.summary

    async def save_summary(self, app_id, conversation_id, summary, covers_until_seq):
        raise AssertionError("o pipeline não grava resumo")


class FakeRetriever:
    """Devolve `chunks` em toda chamada, ou `sequence[i]` na i-ésima chamada."""

    def __init__(
        self,
        chunks: list[ScoredChunk] | None = None,
        *,
        fail: bool = False,
        sequence: list[list[ScoredChunk]] | None = None,
    ):
        self.chunks = [chunk()] if chunks is None else chunks
        self.fail = fail
        self.sequence = sequence
        self.calls: list[tuple] = []

    async def retrieve(self, app_id, principals, query, top_k, rerank_top_n):
        self.calls.append((app_id, principals, query, top_k, rerank_top_n))
        if self.fail:
            raise ConnectionError("pgvector fora do ar")
        if self.sequence is not None:
            return list(self.sequence[min(len(self.calls), len(self.sequence)) - 1])
        return list(self.chunks)


class FakeCache:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.data: dict[str, dict] = {}
        self.gets = 0
        self.sets: list[str] = []

    async def get(self, key):
        self.gets += 1
        if self.fail:
            raise ConnectionError("redis fora do ar")
        return self.data.get(key)

    async def set(self, key, output, ttl_s):
        if self.fail:
            raise ConnectionError("redis fora do ar")
        self.sets.append(key)
        self.data[key] = output.model_dump(mode="json")


class FakeApprovals:
    def __init__(self, result: str | None = None):
        self.result = result
        self.calls: list[str] = []

    async def resolve_turn(self, ctx, boundary):
        self.calls.append(boundary)
        return self.result

    async def request(self, ctx, server, tool, args):
        return "ap1"


class FakeInjection:
    async def check_text(self, text):
        return GuardResult("prompt_injection", "pass")


@dataclass
class Harness:
    deps: Deps
    script: Script
    history: FakeHistory
    retriever: FakeRetriever
    cache: FakeCache
    approvals: FakeApprovals
    enqueued: list[tuple] = field(default_factory=list)
    chains: list[tuple] = field(default_factory=list)


def make_harness(
    app: LoadedApp,
    script: Script | None = None,
    *,
    chain=None,
    history: FakeHistory | None = None,
    retriever: FakeRetriever | None = None,
    cache: FakeCache | None = None,
    approvals: FakeApprovals | None = None,
    settings: Settings | None = None,
    mcp_source=None,
    canaries: dict[str, str] | None = None,
) -> Harness:
    script = script or Script()
    h = Harness(
        deps=None,  # type: ignore[arg-type]
        script=script,
        history=history or FakeHistory(),
        retriever=retriever or FakeRetriever(),
        cache=cache or FakeCache(),
        approvals=approvals or FakeApprovals(),
    )

    def chain_factory(primary, fallbacks):
        h.chains.append((primary, list(fallbacks)))
        if chain is not None:
            return chain(primary, fallbacks)
        return script.model(), [primary]

    async def enqueue(function, *args):
        h.enqueued.append((function, *args))

    async def corpus_version(app_id):
        return 7

    h.deps = Deps(
        settings=settings or Settings(),
        apps={app.config.id: app},
        history_store=h.history,
        retriever=h.retriever,
        injection=FakeInjection(),
        approvals=h.approvals,
        chain_factory=chain_factory,
        corpus_version=corpus_version,
        cache=h.cache,
        enqueue=enqueue,
        canaries=canaries if canaries is not None else {},
        mcp_source=mcp_source,
    )
    return h


async def collect(pipeline, ctx, sink: list | None = None) -> list:
    events = sink if sink is not None else []
    async for event in pipeline.stream(ctx):
        events.append(event)
    return events


def text_of(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def final_of(events: list) -> Final:
    assert isinstance(events[-1], Final), events
    assert sum(isinstance(e, Final) for e in events) == 1
    return events[-1]
