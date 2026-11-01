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
BATCH_TOKENS = 3000


async def summarize_conversation(ctx: dict, app_id: str, conversation_id: str) -> bool:
    with traced("summary.run", app_id=app_id):
        store = ctx["history_store"]
        app = ctx["apps"][app_id]
        counter = HeuristicTokenCounter()

        existing = await store.get_summary(app_id, conversation_id)
        previous, covers = existing if existing else ("", 0)
        messages = await store.list_messages(app_id, conversation_id, after_seq=covers)

        older, _, _ = split_history(messages, app.config.context.budget.history, counter)
        older = [m for m in older if m.seq is not None]
        if not older:
            return False

        # lotes do mais antigo ao mais novo, cada um com até BATCH_TOKENS
        batches: list[list] = []
        current: list = []
        used = 0
        for m in older:
            cost = counter.count(m.content)
            if current and used + cost > BATCH_TOKENS:
                batches.append(current)
                current, used = [], 0
            current.append(m)
            used += cost
        if current:
            batches.append(current)

        factory = ctx.get("model_factory") or (lambda spec: build_model(spec, ctx["settings"]))
        agent = Agent(
            factory(app.config.models.primary), output_type=str, instructions=SUMMARY_PROMPT
        )
        running = previous
        for batch in batches:
            lines = []
            if running:
                lines.append(f"Resumo anterior:\n{running}\n")
            lines.extend(f"{m.role}: {m.content}" for m in batch)
            running = (await agent.run("\n".join(lines))).output

        await store.save_summary(app_id, conversation_id, running, older[-1].seq)
        return True
