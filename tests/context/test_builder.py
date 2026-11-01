import jinja2
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from encaixe.context.builder import ContextBuilder
from encaixe.context.tokens import HeuristicTokenCounter
from encaixe.core.config import AppConfig, BudgetConfig, ContextConfig, LoadedApp, ModelsConfig
from encaixe.core.types import ChatMessage, Chunk, ScoredChunk
from encaixe.output.schemas import RespostaBase


def make_app(**budget) -> LoadedApp:
    cfg = AppConfig(
        id="a",
        models=ModelsConfig(primary="test"),
        context=ContextConfig(budget=BudgetConfig(**budget)),
    )
    tpl = jinja2.Template("Sistema {{ canary }} / {{ boundary }}")
    return LoadedApp(
        config=cfg,
        root=None,
        system_prompt=tpl,
        prompt_version="v",
        output_model=RespostaBase,  # type: ignore[arg-type]
    )


def chunk(i: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(Chunk(i, "a", "d", text, (), ("app:*",), {"title": f"T{i}"}), score)


def build(app=None, **kw):
    args = {
        "canary": "CANARIO123",
        "summary": None,
        "history": [],
        "retrieved": [],
        "extra_context": [],
        "user_message": "oi",
    }
    args.update(kw)
    return ContextBuilder(HeuristicTokenCounter()).build(app or make_app(), **args)


def test_order_system_then_history_then_rag_in_user_prompt():
    hist = [ChatMessage("user", "p1", seq=1), ChatMessage("assistant", "r1", seq=2)]
    ctx = build(
        summary="resumo anterior",
        history=hist,
        retrieved=[chunk("c1", "texto do trecho", 0.9)],
        extra_context=["extra info"],
        user_message="pergunta final",
    )
    assert "CANARIO123" in ctx.system_prompt
    assert ctx.boundary in ctx.system_prompt
    assert ctx.user_prompt.endswith("pergunta final")
    up = ctx.user_prompt
    assert up.index("tipo=contexto") < up.index("tipo=trechos") < up.index("pergunta final")
    assert ctx.boundary in up
    first = ctx.history[0]
    assert isinstance(first, ModelRequest)
    assert isinstance(first.parts[0], SystemPromptPart)
    assert first.parts[0].content == "resumo anterior"
    assert isinstance(ctx.history[1], ModelRequest)
    assert isinstance(ctx.history[1].parts[0], UserPromptPart)
    assert isinstance(ctx.history[2], ModelResponse)
    assert isinstance(ctx.history[2].parts[0], TextPart)
    assert set(ctx.token_usage) == {"system", "summary", "history", "rag", "user"}


def test_history_skips_tool_and_system_roles():
    hist = [
        ChatMessage("user", "p1", seq=1),
        ChatMessage("tool", "x", seq=2),
        ChatMessage("system", "y", seq=3),
        ChatMessage("assistant", "r1", seq=4),
    ]
    ctx = build(history=hist)
    assert len(ctx.history) == 2


def test_history_trimmed_keeps_most_recent_within_budget():
    # cada mensagem de 30 chars = 10 tokens; turno = 20 tokens; orçamento 45 => 2 turnos
    hist = []
    for i in range(4):
        hist.append(ChatMessage("user", f"{i}" * 30, seq=2 * i + 1))
        hist.append(ChatMessage("assistant", f"{i}" * 30, seq=2 * i + 2))
    ctx = build(make_app(history=45), history=hist)
    assert len(ctx.history) == 4
    assert ctx.history[0].parts[0].content == "2" * 30
    assert ctx.token_usage["history"] == 40


def test_history_never_splits_turn():
    hist = [
        ChatMessage("user", "a" * 30, seq=1),
        ChatMessage("assistant", "b" * 30, seq=2),
    ]
    ctx = build(make_app(history=15), history=hist)
    assert ctx.history == []


def test_rag_budget_drops_lowest_scores():
    chunks = [
        chunk("low", "b" * 300, 0.1),
        chunk("high", "a" * 300, 0.9),
        chunk("mid", "c" * 300, 0.5),
    ]
    ctx = build(make_app(rag=250), retrieved=chunks)
    assert "high" in ctx.user_prompt and "mid" in ctx.user_prompt
    assert ctx.user_prompt.index("high") < ctx.user_prompt.index("mid")
    assert "low" not in ctx.user_prompt.replace("tipo", "")  # id "low" ausente
    assert ctx.token_usage["rag"] <= 250


def test_long_user_message_truncated_to_budget():
    ctx = build(make_app(user=10), user_message="x" * 1000)
    assert ctx.user_prompt.endswith(" [mensagem truncada]")
    assert ctx.user_prompt.count("x") == 30
