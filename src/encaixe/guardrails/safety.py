from __future__ import annotations

import httpx

from encaixe.core.registry import register
from encaixe.core.types import GuardResult, RequestContext
from encaixe.observability.spans import traced


class LlamaGuard:
    name = "content_safety"

    def __init__(self, settings, client: httpx.AsyncClient | None = None) -> None:
        self.model = settings.safety_model
        self.base_url = settings.ollama_base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def _ask(self, messages: list[dict]) -> GuardResult:
        resp = await self._client.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        content = str(resp.json()["message"]["content"]).strip()
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        verdict = lines[0].lower() if lines else ""
        if verdict == "safe":
            return GuardResult(self.name, "pass")
        if verdict == "unsafe":
            categories = lines[1].replace(" ", "") if len(lines) > 1 else ""
            return GuardResult(self.name, "block", categories)
        raise ValueError(f"resposta inesperada do Llama Guard: {content[:80]!r}")

    async def check_output_pair(self, user_text: str, assistant_text: str) -> GuardResult:
        with traced("guard.content_safety") as span:
            result = await self._ask(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            span.set_attribute("verdict", result.verdict)
            return result

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        with traced("guard.content_safety") as span:
            result = await self._ask([{"role": "user", "content": text}])
            span.set_attribute("verdict", result.verdict)
            return result

    async def check_response(self, text: str, ctx: RequestContext) -> GuardResult:
        """Modo saída: ctx.user_message é o lado do usuário."""
        return await self.check_output_pair(ctx.user_message, text)


@register("guard", "content_safety")
def _factory(settings, **_):
    return LlamaGuard(settings)
