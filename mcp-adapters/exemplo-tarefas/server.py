"""Adaptador MCP de exemplo: tarefas por usuário, em memória.

A identidade vem do cabeçalho X-User-Ref, que a plataforma define; sem ele, "anon".
"""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

mcp = FastMCP("exemplo-tarefas")

_TAREFAS: dict[str, list[dict]] = {}


def _user_ref() -> str:
    try:
        headers = get_http_headers()
    except Exception:  # noqa: BLE001  fora de uma requisição HTTP (ex.: transporte em memória)
        headers = {}
    return headers.get("x-user-ref") or "anon"


def listar_tarefas_de(user_ref: str) -> list[dict]:
    return list(_TAREFAS.get(user_ref, []))


def criar_tarefa_para(user_ref: str, titulo: str, prazo: str | None) -> dict:
    tarefa = {"id": len(_TAREFAS.get(user_ref, [])) + 1, "titulo": titulo, "prazo": prazo}
    _TAREFAS.setdefault(user_ref, []).append(tarefa)
    return tarefa


@mcp.tool(description="Lista as tarefas do usuário atual.")
def listar_tarefas() -> list[dict]:
    return listar_tarefas_de(_user_ref())


@mcp.tool(description="Cria uma tarefa para o usuário atual, com título e prazo opcional (AAAA-MM-DD).")
def criar_tarefa(titulo: str, prazo: str | None = None) -> dict:
    return criar_tarefa_para(_user_ref(), titulo, prazo)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8765, path="/mcp")
