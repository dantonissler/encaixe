"""Limite de tamanho do resultado de ferramenta (spec §6.6; revisão final m1)."""

from __future__ import annotations

from encaixe.core.types import GuardResult
from encaixe.settings import Settings
from encaixe.tools.results import DEFAULT_MAX_CHARS, TRUNCATED, sanitize_tool_result


class SeenInjection:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def check_text(self, text):
        self.seen.append(text)
        return GuardResult("injection", "pass")


async def test_long_result_is_truncated_with_visible_marker_before_check_and_wrap():
    inj = SeenInjection()
    out = await sanitize_tool_result(inj, "listar", "x" * 50 + "CAUDA", "B", max_chars=20)
    assert TRUNCATED in out and "CAUDA" not in out
    assert "x" * 20 in out and "x" * 21 not in out
    assert inj.seen == ["x" * 20 + "\n" + TRUNCATED]  # o guard vê o mesmo texto que o modelo
    assert out.startswith("<<B tipo=ferramenta>>")


async def test_result_within_limit_is_unchanged():
    inj = SeenInjection()
    out = await sanitize_tool_result(inj, "listar", "tarefa a", "B", max_chars=8)
    assert TRUNCATED not in out and inj.seen == ["tarefa a"]


def test_default_limit_comes_from_settings():
    assert Settings(_env_file=None).tool_result_max_chars == DEFAULT_MAX_CHARS == 4000
