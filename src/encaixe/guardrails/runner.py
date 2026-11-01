from __future__ import annotations

from typing import Any

from encaixe.core.registry import create
from encaixe.core.types import GuardResult, RequestContext
from encaixe.guardrails import canary, citations, injection, normalize, pii_guard, safety  # noqa: F401
from encaixe.output.schemas import RespostaBase

_CACHE: dict[tuple[str, int], Any] = {}


def get_guard(name: str, settings: Any) -> Any:
    key = (name, id(settings))
    if key not in _CACHE:
        _CACHE[key] = create("guard", name, settings=settings)
    return _CACHE[key]


async def _run(names, settings, fail_closed, call) -> list[GuardResult]:
    results: list[GuardResult] = []
    for name in names:
        try:
            guard = get_guard(name, settings)
            result = await call(guard)
        except Exception:  # noqa: BLE001 - falha do guard segue a política fail_closed
            result = GuardResult(name, "block" if fail_closed else "pass", "unavailable")
        results.append(result)
        if result.verdict == "block":
            break
    return results


async def run_guards(
    names: list[str], text: str, ctx: RequestContext, *, settings: Any, fail_closed: bool
) -> list[GuardResult]:
    current = text

    async def call(guard):
        nonlocal current
        r = await guard.check(current, ctx)
        if r.redacted_text is not None:
            current = r.redacted_text
        return r

    return await _run(names, settings, fail_closed, call)


async def run_output_guards(
    names: list[str],
    output: RespostaBase,
    ctx: RequestContext,
    *,
    settings: Any,
    fail_closed: bool,
) -> list[GuardResult]:
    async def call(guard):
        if hasattr(guard, "check_output"):
            return await guard.check_output(output, ctx)
        if hasattr(guard, "check_response"):
            return await guard.check_response(output.resposta, ctx)
        return await guard.check(output.resposta, ctx)

    return await _run(names, settings, fail_closed, call)


def blocked(results: list[GuardResult]) -> GuardResult | None:
    return next((r for r in results if r.verdict == "block"), None)
