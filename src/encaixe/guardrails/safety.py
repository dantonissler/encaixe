"""Guard de conteúdo (`content_safety`) em camadas, rodando no Ollama.

- Principal (`settings.safety_model`, padrão ShieldGemma 2B): todo "unsafe"/"Yes" bloqueia.
- Secundário opcional (`settings.safety_secondary_model`, padrão Llama Guard 3 1B; vazio
  desliga): o "unsafe" só conta se alguma categoria devolvida estiver fora de
  `settings.safety_secondary_ignore_categories` (padrão S1 e S6, os falsos positivos medidos
  em pedidos com códigos e em respostas de salário). Sem categoria, conta.

Os dois rodam ao mesmo tempo; bloqueia se qualquer um bloquear. Erro de qualquer modelo (fora
do ar, resposta ilegível) vira exceção, e o runner aplica `on_guard_unavailable`; um bloqueio
certo de um dos modelos vale mesmo com erro no outro. Medição em `evals/guard_probe.py`.
"""

from __future__ import annotations

import asyncio

import httpx

from encaixe.core.registry import register
from encaixe.core.types import GuardResult, RequestContext
from encaixe.observability.spans import traced

# Modelos de Sim/Não (ShieldGemma) explicariam "passo a passo" sem limite; só a 1ª palavra
# importa no principal. O secundário não tem limite: as categorias decidem.
PRIMARY_NUM_PREDICT = 8


def parse_verdict(content: str) -> tuple[str, list[str]]:
    """("pass" | "block", categorias) a partir da resposta do modelo.

    Llama Guard: "safe" ou "unsafe" e, na linha seguinte, as categorias ("S1,S10").
    ShieldGemma / Granite Guardian: "Yes" (viola a política) ou "No", seguido de explicação.
    """
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    words = lines[0].split() if lines else []
    first = words[0].strip(".,:;!").lower() if words else ""
    if first in ("safe", "no"):
        return "pass", []
    if first == "unsafe":
        cats = lines[1] if len(lines) > 1 else ""
        return "block", [c.strip() for c in cats.split(",") if c.strip()]
    if first == "yes":
        return "block", []
    raise ValueError(f"resposta inesperada do guard de conteúdo: {content[:80]!r}")


class OllamaSafetyModel:
    """Um modelo de segurança no Ollama (`/api/chat`, sem streaming)."""

    def __init__(
        self, model: str, base_url: str, client: httpx.AsyncClient, num_predict: int | None
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._client = client
        self._num_predict = num_predict

    async def ask(self, messages: list[dict]) -> tuple[str, list[str]]:
        body: dict = {"model": self.model, "messages": messages, "stream": False}
        if self._num_predict is not None:
            body["options"] = {"num_predict": self._num_predict}
        with traced("guard.content_safety.model", model=self.model) as span:
            resp = await self._client.post(f"{self.base_url}/api/chat", json=body)
            resp.raise_for_status()
            verdict, categories = parse_verdict(str(resp.json()["message"]["content"]))
            span.set_attribute("verdict", verdict)
            span.set_attribute("categories", ",".join(categories))
            return verdict, categories


class ContentSafetyGuard:
    name = "content_safety"

    def __init__(self, settings, client: httpx.AsyncClient | None = None) -> None:
        base = settings.ollama_base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self.primary = OllamaSafetyModel(
            settings.safety_model, base, self._client, PRIMARY_NUM_PREDICT
        )
        secondary = settings.safety_secondary_model
        self.secondary = (
            OllamaSafetyModel(secondary, base, self._client, None) if secondary else None
        )
        self.ignore = set(settings.safety_secondary_ignore_categories)

    def models(self) -> list[str]:
        return [m.model for m in (self.primary, self.secondary) if m is not None]

    async def _decide(self, messages: list[dict]) -> GuardResult:
        backends = [m for m in (self.primary, self.secondary) if m is not None]
        answers = await asyncio.gather(*(m.ask(messages) for m in backends), return_exceptions=True)
        blocks: list[str] = []
        for backend, answer in zip(backends, answers, strict=True):
            if isinstance(answer, BaseException):
                continue
            verdict, categories = answer
            if verdict != "block":
                continue
            if backend is self.primary:
                blocks.append(f"{backend.model}:{','.join(categories) or 'unsafe'}")
                continue
            counted = [c for c in categories if c not in self.ignore]
            if counted or not categories:  # sem categoria não há o que ignorar
                blocks.append(f"{backend.model}:{','.join(counted) or 'unsafe'}")
        if blocks:
            return GuardResult(self.name, "block", "; ".join(blocks))
        for answer in answers:
            if isinstance(answer, BaseException):
                raise answer  # o runner transforma em "unavailable"
        return GuardResult(self.name, "pass")

    async def _checked(self, messages: list[dict]) -> GuardResult:
        with traced("guard.content_safety") as span:
            result = await self._decide(messages)
            span.set_attribute("verdict", result.verdict)
            return result

    async def check_output_pair(self, user_text: str, assistant_text: str) -> GuardResult:
        return await self._checked(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        return await self._checked([{"role": "user", "content": text}])

    async def check_response(self, text: str, ctx: RequestContext) -> GuardResult:
        """Modo saída: ctx.user_message é o lado do usuário."""
        return await self.check_output_pair(ctx.user_message, text)


@register("guard", "content_safety")
def _factory(settings, **_):
    return ContentSafetyGuard(settings)
