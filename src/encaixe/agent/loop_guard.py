from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from opentelemetry import trace
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset


class LoopDetected(Exception):
    """A mesma ferramenta foi chamada com os mesmos args `max_repeats` vezes."""

    def __init__(self, tool_name: str, repeats: int) -> None:
        super().__init__(f"Loop detectado: '{tool_name}' chamada {repeats}x com os mesmos argumentos")
        self.tool_name = tool_name
        self.repeats = repeats


class LoopGuard(WrapperToolset[Any]):
    """Conta chamadas por (ferramenta, JSON canônico dos args). Uma instância por execução."""

    def __init__(self, wrapped: AbstractToolset[Any], max_repeats: int = 3) -> None:
        super().__init__(wrapped)
        self.max_repeats = max_repeats
        self._counts: dict[tuple[str, str], int] = {}

    @staticmethod
    def _key(name: str, tool_args: dict[str, Any]) -> tuple[str, str]:
        return name, json.dumps(tool_args, sort_keys=True, ensure_ascii=False, default=str)

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: Any, tool: Any) -> Any:
        key = self._key(name, tool_args)
        count = self._counts[key] = self._counts.get(key, 0) + 1
        if count >= self.max_repeats:
            span = trace.get_current_span()
            span.add_event("loop_detected", {"tool": name, "repeats": count})
            span.set_attribute("agent.loop_detected", True)
            raise LoopDetected(name, count)
        return await super().call_tool(name, tool_args, ctx, tool)


def wrap_toolsets(toolsets: Sequence[AbstractToolset[Any]], max_repeats: int = 3) -> list[AbstractToolset[Any]]:
    """Embrulha cada toolset num LoopGuard novo; chamar uma vez por execução do agente."""
    return [LoopGuard(ts, max_repeats=max_repeats) for ts in toolsets]
