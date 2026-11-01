import jinja2
import pytest
from pydantic_ai.models.test import TestModel

from encaixe.core.config import AppConfig, BudgetConfig, ContextConfig, LoadedApp, ModelsConfig
from encaixe.core.types import ChatMessage
from encaixe.output.schemas import RespostaBase
from encaixe.settings import Settings
from encaixe.workers.summarize import summarize_conversation


class FakeStore:
    def __init__(self, msgs, summary=None):
        self.msgs = msgs
        self.summary = summary
        self.saved = None

    async def get_summary(self, app_id, conversation_id):
        return self.summary

    async def list_messages(self, app_id, conversation_id, after_seq=0):
        return [m for m in self.msgs if m.seq > after_seq]

    async def save_summary(self, app_id, conversation_id, summary, covers_until_seq):
        self.saved = (summary, covers_until_seq)


def make_ctx(store, history_budget=45):
    app = LoadedApp(
        config=AppConfig(
            id="a",
            models=ModelsConfig(primary="test"),
            context=ContextConfig(budget=BudgetConfig(history=history_budget)),
        ),
        root=None,  # type: ignore[arg-type]
        system_prompt=jinja2.Template("x"),
        prompt_version="v",
        output_model=RespostaBase,
    )
    seen = []

    def factory(spec):
        seen.append(spec)
        return TestModel(custom_output_text="RESUMO NOVO")

    return {
        "history_store": store,
        "apps": {"a": app},
        "settings": Settings(),
        "model_factory": factory,
    }, seen


def turns(n):
    out = []
    for i in range(n):
        out.append(ChatMessage("user", f"{i}" * 30, seq=2 * i + 1))
        out.append(ChatMessage("assistant", f"{i}" * 30, seq=2 * i + 2))
    return out


@pytest.mark.asyncio
async def test_summarizes_when_over_budget():
    store = FakeStore(turns(4), summary=("resumo velho", 0))
    ctx, seen = make_ctx(store)
    assert await summarize_conversation(ctx, "a", "c1") is True
    # sobram os 2 turnos mais novos (seq 5..8); resumo cobre até seq 4
    assert store.saved == ("RESUMO NOVO", 4)
    assert seen == ["test"]


@pytest.mark.asyncio
async def test_noop_when_within_budget():
    store = FakeStore(turns(2))
    ctx, _ = make_ctx(store, history_budget=1000)
    assert await summarize_conversation(ctx, "a", "c1") is False
    assert store.saved is None


from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel


def capturing_ctx(store, history_budget=45):
    ctx, _ = make_ctx(store, history_budget)
    prompts = []

    def fn(messages, info):
        text = "\n".join(
            p.content
            for m in messages
            for p in m.parts
            if getattr(p, "part_kind", "") == "user-prompt"
        )
        prompts.append(text)
        return ModelResponse(parts=[TextPart(f"R{len(prompts)}")])

    ctx["model_factory"] = lambda spec: FunctionModel(fn)
    return ctx, prompts


@pytest.mark.asyncio
async def test_previous_summary_reaches_model():
    store = FakeStore(turns(4), summary=("RESUMO VELHO", 0))
    ctx, prompts = capturing_ctx(store)
    await summarize_conversation(ctx, "a", "c1")
    assert "RESUMO VELHO" in prompts[0]
    assert "0" * 30 in prompts[0]


@pytest.mark.asyncio
async def test_idempotent_rerun_returns_false():
    store = FakeStore(turns(4))
    ctx, _ = make_ctx(store)
    assert await summarize_conversation(ctx, "a", "c1") is True
    summary, covers = store.saved
    store.summary = (summary, covers)
    assert await summarize_conversation(ctx, "a", "c1") is False


@pytest.mark.asyncio
async def test_large_backlog_is_batched():
    msgs = []
    for i in range(10):
        msgs.append(ChatMessage("user", "u" * 3000, seq=2 * i + 1))  # 1000 tokens
        msgs.append(ChatMessage("assistant", "a" * 3000, seq=2 * i + 2))
    store = FakeStore(msgs)
    ctx, prompts = capturing_ctx(store, history_budget=4000)
    assert await summarize_conversation(ctx, "a", "c1") is True
    assert len(prompts) > 1
    assert "R1" in prompts[1]  # resumo corrente avança para o próximo lote
    assert store.saved[0] == f"R{len(prompts)}"
    assert store.saved[1] == 16  # mantém 2 turnos (4000 tokens); cobre seq 1..16


@pytest.mark.asyncio
async def test_tool_and_system_messages_ignored():
    msgs = turns(4)
    msgs.insert(3, ChatMessage("tool", "FERRAMENTA", seq=100))
    msgs.insert(3, ChatMessage("system", "SISTEMA", seq=101))
    store = FakeStore(msgs)
    ctx, prompts = capturing_ctx(store)
    await summarize_conversation(ctx, "a", "c1")
    assert "FERRAMENTA" not in prompts[0] and "SISTEMA" not in prompts[0]
    assert store.saved[1] == 4


@pytest.mark.asyncio
async def test_summarizer_disables_reasoning():
    store = FakeStore(turns(4))
    ctx, _ = make_ctx(store)
    seen = []

    def fn(messages, info):
        seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart("R")])

    ctx["model_factory"] = lambda spec: FunctionModel(fn)
    assert await summarize_conversation(ctx, "a", "c1") is True
    # o Qwen3.5 pensa por padrão; sem isso o /v1 gasta o orçamento em `reasoning`
    assert seen and all(s["extra_body"]["reasoning_effort"] == "none" for s in seen)
