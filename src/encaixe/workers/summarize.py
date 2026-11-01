from __future__ import annotations

from pydantic_ai import Agent

from encaixe.context.builder import split_history
from encaixe.context.tokens import HeuristicTokenCounter
from encaixe.llm.models import build_model
from encaixe.observability.spans import traced

SUMMARY_PROMPT = (
    "Resuma a conversa abaixo em português, em até 200 palavras, "
    "preservando fatos, decisões e pendências."
)


async def summarize_conversation(ctx: dict, app_id: str, conversation_id: str) -> bool:
    with traced("summary.run", app_id=app_id):
        store = ctx["history_store"]
        app = ctx["apps"][app_id]
        counter = HeuristicTokenCounter()

        existing = await store.get_summary(app_id, conversation_id)
        previous, covers = existing if existing else ("", 0)
        messages = await store.list_messages(app_id, conversation_id, after_seq=covers)

        older, _, _ = split_history(messages, app.config.context.budget.history, counter)
        if not older:
            return False
        last_seq = older[-1].seq
        if last_seq is None:
            return False

        lines = []
        if previous:
            lines.append(f"Resumo anterior:\n{previous}\n")
        lines.extend(f"{m.role}: {m.content}" for m in older)

        factory = ctx.get("model_factory") or (lambda spec: build_model(spec, ctx["settings"]))
        model = factory(app.config.models.primary)
        agent = Agent(model, output_type=str, instructions=SUMMARY_PROMPT)
        result = await agent.run("\n".join(lines))

        await store.save_summary(app_id, conversation_id, result.output, last_seq)
        return True
