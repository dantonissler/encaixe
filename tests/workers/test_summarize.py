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
