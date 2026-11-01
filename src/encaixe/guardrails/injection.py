from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from encaixe.core.registry import register
from encaixe.core.types import GuardResult, RequestContext
from encaixe.observability.spans import traced

MALICIOUS = {"injection", "malicious", "jailbreak", "label_1"}
SAFE = {"safe", "benign", "label_0"}
WINDOW = 512
OVERLAP = 32


def _default_factory(model_id: str) -> Any:
    from transformers import pipeline

    return pipeline(
        "text-classification", model=model_id, truncation=True, max_length=WINDOW, device=-1
    )


class InjectionGuard:
    name = "prompt_injection"

    def __init__(
        self,
        model_id: str,
        threshold: float,
        pipeline_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self._factory = pipeline_factory or _default_factory
        self._pipe: Any = None
        self._lock = asyncio.Lock()

    async def _get_pipe(self) -> Any:
        if self._pipe is None:
            async with self._lock:
                if self._pipe is None:
                    self._pipe = await asyncio.to_thread(self._factory, self.model_id)
        return self._pipe

    def _windows(self, pipe: Any, text: str) -> list[str]:
        tok = getattr(pipe, "tokenizer", None)
        if tok is None:
            return [text]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        size = WINDOW - 8  # folga para tokens especiais
        if len(ids) <= size:
            return [text]
        step = size - OVERLAP
        return [
            tok.decode(ids[i : i + size])
            for i in range(0, len(ids), step)
            if ids[i : i + size]
        ]

    def _classify(self, pipe: Any, text: str) -> GuardResult:
        windows = self._windows(pipe, text)
        outputs = pipe(windows)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for out in outputs:
            if isinstance(out, list):
                out = out[0]
            label = str(out["label"]).lower()
            if label in MALICIOUS and float(out["score"]) >= self.threshold:
                return GuardResult(self.name, "block", "injection")
        return GuardResult(self.name, "pass")

    async def check_text(self, text: str) -> GuardResult:
        with traced("guard.prompt_injection") as span:
            pipe = await self._get_pipe()
            result = await asyncio.to_thread(self._classify, pipe, text)
            span.set_attribute("verdict", result.verdict)
            return result

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        return await self.check_text(text)


@register("guard", "prompt_injection")
def _factory(settings, pipeline_factory=None, **_):
    return InjectionGuard(settings.injection_model_id, settings.injection_threshold, pipeline_factory)
