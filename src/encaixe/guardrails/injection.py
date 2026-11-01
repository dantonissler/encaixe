from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any, Literal

from encaixe.core.registry import register
from encaixe.core.types import GuardResult, RequestContext
from encaixe.guardrails.heuristics import detect
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
        backend: Literal["heuristic", "model"] = "heuristic",
    ) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.backend = backend
        self._factory = pipeline_factory or _default_factory
        self._pipe: Any = None
        self._lock = asyncio.Lock()
        self._model_lock = threading.Lock()  # tokenizer/pipeline HF não são thread-safe

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
        return [tok.decode(ids[i : i + size]) for i in range(0, len(ids), step)]

    def _classify(self, pipe: Any, text: str) -> GuardResult:
        with self._model_lock:
            windows = self._windows(pipe, text)
            outputs = pipe(windows, top_k=None)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for out in outputs:
            scores = [out] if isinstance(out, dict) else out
            malicious = 0.0
            for item in scores:
                label = str(item["label"]).lower()
                if label in MALICIOUS:
                    malicious = max(malicious, float(item["score"]))
                elif label not in SAFE:
                    raise ValueError(f"rótulo desconhecido do classificador: {item['label']!r}")
            if malicious >= self.threshold:
                return GuardResult(self.name, "block", "injection")
        return GuardResult(self.name, "pass")

    async def check_text(self, text: str) -> GuardResult:
        with traced("guard.prompt_injection") as span:
            hit = detect(text)
            if hit:
                result = GuardResult(self.name, "block", f"heuristic:{hit}")
            elif self.backend == "model":
                pipe = await self._get_pipe()
                result = await asyncio.to_thread(self._classify, pipe, text)
            else:
                result = GuardResult(self.name, "pass")
            span.set_attribute("verdict", result.verdict)
            return result

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        return await self.check_text(text)


@register("guard", "prompt_injection")
def _factory(settings, pipeline_factory=None, **_):
    return InjectionGuard(
        settings.injection_model_id,
        settings.injection_threshold,
        pipeline_factory,
        backend=settings.injection_backend,
    )
