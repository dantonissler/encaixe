from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from encaixe.context.untrusted import new_boundary, wrap_untrusted
from encaixe.core.config import LoadedApp
from encaixe.core.protocols import TokenCounter
from encaixe.core.types import ChatMessage, ScoredChunk
from encaixe.observability.spans import traced

TRUNCATED_SUFFIX = " [mensagem truncada]"
CHARS_PER_TOKEN = 3


def conversation_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    return [m for m in messages if m.role in ("user", "assistant")]


def group_turns(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    """Agrupa em turnos: cada mensagem do usuário abre um turno novo."""
    turns: list[list[ChatMessage]] = []
    for m in conversation_messages(messages):
        if m.role == "user" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    return turns


def split_history(
    messages: list[ChatMessage], budget: int, counter: TokenCounter
) -> tuple[list[ChatMessage], list[ChatMessage], int]:
    """Devolve (mais antigas descartadas, mais recentes mantidas, tokens mantidos).

    Mantém turnos inteiros, do mais novo para o mais velho, enquanto couberem.
    """
    turns = group_turns(messages)
    used = 0
    keep_from = len(turns)
    for i in range(len(turns) - 1, -1, -1):
        cost = sum(counter.count(m.content) for m in turns[i])
        if used + cost > budget:
            break
        used += cost
        keep_from = i
    older = [m for t in turns[:keep_from] for m in t]
    kept = [m for t in turns[keep_from:] for m in t]
    return older, kept, used


def truncate_to_tokens(text: str, tokens: int, suffix: str = "") -> str:
    limit = max(tokens, 0) * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    return text[:limit] + suffix


@dataclass
class BuiltContext:
    system_prompt: str
    history: list[ModelMessage]
    user_prompt: str
    boundary: str
    token_usage: dict[str, int]


class ContextBuilder:
    def __init__(self, counter: TokenCounter) -> None:
        self.counter = counter

    @traced("context.build")
    def build(
        self,
        app: LoadedApp,
        *,
        canary: str,
        summary: str | None,
        history: list[ChatMessage],
        retrieved: list[ScoredChunk],
        extra_context: list[str],
        user_message: str,
    ) -> BuiltContext:
        budget = app.config.context.budget
        count = self.counter.count
        boundary = new_boundary()
        system_prompt = app.system_prompt.render(canary=canary, boundary=boundary)

        model_history: list[ModelMessage] = []
        summary_text = ""
        if summary:
            summary_text = truncate_to_tokens(summary, budget.summary)
            model_history.append(ModelRequest(parts=[SystemPromptPart(content=summary_text)]))

        _, kept, history_tokens = split_history(history, budget.history, self.counter)
        for m in kept:
            if m.role == "user":
                model_history.append(ModelRequest(parts=[UserPromptPart(content=m.content)]))
            else:
                model_history.append(ModelResponse(parts=[TextPart(content=m.content)]))

        rag_items: list[dict] = []
        rag_tokens = 0
        for sc in sorted(retrieved, key=lambda s: s.score, reverse=True):
            item = {
                "id": sc.chunk.id,
                "titulo": sc.chunk.metadata.get("title", ""),
                "texto": sc.chunk.text,
            }
            cost = count(item["texto"])
            if rag_tokens + cost > budget.rag:
                break
            rag_items.append(item)
            rag_tokens += cost

        user_text = truncate_to_tokens(user_message, budget.user, TRUNCATED_SUFFIX)
        blocks: list[str] = []
        if extra_context:
            blocks.append(
                wrap_untrusted("contexto", [{"texto": s} for s in extra_context], boundary)
            )
        if rag_items:
            blocks.append(wrap_untrusted("trechos", rag_items, boundary))
        blocks.append(user_text)

        return BuiltContext(
            system_prompt=system_prompt,
            history=model_history,
            user_prompt="\n\n".join(blocks),
            boundary=boundary,
            token_usage={
                "system": count(system_prompt),
                "summary": count(summary_text),
                "history": history_tokens,
                "rag": rag_tokens,
                "user": count(user_text),
            },
        )
