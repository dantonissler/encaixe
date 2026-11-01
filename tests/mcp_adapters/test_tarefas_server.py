from __future__ import annotations

import importlib.util
from pathlib import Path

from fastmcp import Client

SERVER = Path(__file__).resolve().parents[2] / "mcp-adapters" / "exemplo-tarefas" / "server.py"


def load_server():
    spec = importlib.util.spec_from_file_location("tarefas_server", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def test_create_and_list_per_user():
    mod = load_server()
    async with Client(mod.mcp) as c:
        # in-memory sem HTTP: sem cabeçalho cai em "anon"
        await c.call_tool("criar_tarefa", {"titulo": "pagar boleto", "prazo": "2026-12-01"})
        res = await c.call_tool("listar_tarefas", {})
        assert [t["titulo"] for t in res.data] == ["pagar boleto"]
        assert res.data[0]["prazo"] == "2026-12-01"


def test_store_is_keyed_by_user_ref():
    mod = load_server()
    mod.criar_tarefa_para("ana", "a1", None)
    mod.criar_tarefa_para("bia", "b1", None)
    assert [t["titulo"] for t in mod.listar_tarefas_de("ana")] == ["a1"]
    assert [t["titulo"] for t in mod.listar_tarefas_de("bia")] == ["b1"]
    assert mod.listar_tarefas_de("zed") == []
