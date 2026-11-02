"""Casos de escrita (2 turnos) e preparo das tarefas dos casos de leitura.

O promptfoo não encadeia turnos, e a ferramenta de escrita (`criar_tarefa`) só é oferecida com
um `X-Conversation-Id` de um turno anterior. Cada caso aqui:
  1. pede a criação da tarefa -> a resposta precisa pedir confirmação ("confirma...");
  2. responde "sim" na mesma conversa -> a resposta diz que criou, e `listar_tarefas` (direto
     no adaptador MCP) mostra a tarefa.
O resultado entra em `evals/results/<app>.json` na seção `extra.multiturn`, que o
compare_baseline.py soma ao pass rate do promptfoo (rode depois do promptfoo, que reescreve o
arquivo).

`--seed-only` cria as tarefas do usuário `eval-leitor` (idempotente) usadas pelos casos de
`listar_tarefas` do tests.yaml; rode antes do promptfoo. O adaptador guarda tudo em memória.

Ambiente: ENCAIXE_EVAL_API_KEY (obrigatória), ENCAIXE_API_URL (http://localhost:8100),
ENCAIXE_MCP_URL (http://localhost:8765/mcp), ENCAIXE_EVAL_MODEL (override, só com ENCAIXE_ENV=dev).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evals" / "results"
READER = "eval-leitor"
READER_TASKS = ["Enviar relatório mensal", "Renovar crachá"]
CASES = [
    {
        "id": "escrita-1",
        "user": "eval-escritor-1",
        "prompt": "Crie uma tarefa: revisar contrato, prazo sexta",
        "needle": "revisar contrato",
    },
    {
        "id": "escrita-2",
        "user": "eval-escritor-2",
        "prompt": "Crie uma tarefa: renovar certificado digital, prazo 2026-12-15",
        "needle": "renovar certificado digital",
    },
]
CREATED = re.compile(r"\b(cri(ei|ada|ado)|criou|registr\w+|adicion\w+)\b", re.IGNORECASE)


def check_turns(ask: str, answer: str, listing: str, needle: str) -> list[str]:
    """Falhas de um caso (lista vazia = passou)."""
    failures = []
    if "confirma" not in ask.lower():
        failures.append("turno 1: a resposta não pede confirmação")
    if not CREATED.search(answer):
        failures.append("turno 2: a resposta não diz que a tarefa foi criada")
    if needle.lower() not in listing.lower():
        failures.append("listar_tarefas não mostra a tarefa criada")
    return failures


def merge_extra(path: Path, summary: dict[str, Any]) -> None:
    """Grava `extra.multiturn` no JSON de resultados, preservando o resto."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault("extra", {})["multiturn"] = summary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "passed": sum(1 for c in cases if not c["failures"]),
        "total": len(cases),
        "cases": cases,
    }


def _headers(user: str, conversation_id: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {os.environ['ENCAIXE_EVAL_API_KEY']}",
        "X-User-Ref": user,
    }
    if conversation_id:
        headers["X-Conversation-Id"] = conversation_id
    if os.environ.get("ENCAIXE_EVAL_MODEL"):
        headers["X-Encaixe-Model"] = os.environ["ENCAIXE_EVAL_MODEL"]
    return headers


async def _chat(
    client: httpx.AsyncClient, user: str, text: str, conversation_id: str | None = None
) -> tuple[str, str | None]:
    resp = await client.post(
        "/v1/chat/completions",
        headers=_headers(user, conversation_id),
        json={
            "model": "exemplo",
            "messages": [{"role": "user", "content": text}],
            "stream": False,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], resp.headers.get("X-Conversation-Id")


async def _mcp(user: str, tool: str, args: dict[str, Any] | None = None) -> str:
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    url = os.environ.get("ENCAIXE_MCP_URL", "http://localhost:8765/mcp")
    async with Client(StreamableHttpTransport(url, headers={"X-User-Ref": user})) as mcp:
        result = await mcp.call_tool(tool, args or {})
    return json.dumps(result.structured_content or result.data, ensure_ascii=False, default=str)


async def seed_reader_tasks() -> None:
    existing = await _mcp(READER, "listar_tarefas")
    for title in READER_TASKS:
        if title not in existing:
            await _mcp(READER, "criar_tarefa", {"titulo": title})
    print(f"tarefas de {READER} prontas: {', '.join(READER_TASKS)}")


async def run_cases() -> dict[str, Any]:
    base = os.environ.get("ENCAIXE_API_URL", "http://localhost:8100")
    results = []
    async with httpx.AsyncClient(base_url=base, timeout=300) as client:
        for case in CASES:
            failures: list[str] = []
            try:
                ask, cid = await _chat(client, case["user"], case["prompt"])
                if not cid:
                    raise RuntimeError("a API não devolveu X-Conversation-Id")
                answer, _ = await _chat(client, case["user"], "sim", cid)
                listing = await _mcp(case["user"], "listar_tarefas")
                failures = check_turns(ask, answer, listing, case["needle"])
            except Exception as exc:  # noqa: BLE001 - um caso quebrado não derruba os outros
                ask = answer = listing = ""
                failures = [f"erro: {type(exc).__name__}: {exc}"]
            results.append({"id": case["id"], "failures": failures, "turn1": ask, "turn2": answer})
            print(f"{case['id']}: {'ok' if not failures else 'FALHOU ' + '; '.join(failures)}")
    return summarize(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--app", default="exemplo")
    parser.add_argument("--seed-only", action="store_true", help="só cria as tarefas de leitura")
    args = parser.parse_args(argv)
    if not os.environ.get("ENCAIXE_EVAL_API_KEY"):
        print("defina ENCAIXE_EVAL_API_KEY (veja `uv run encaixe seed --app exemplo`).")
        return 2
    if args.seed_only:
        asyncio.run(seed_reader_tasks())
        return 0
    summary = asyncio.run(run_cases())
    merge_extra(RESULTS_DIR / f"{args.app}.json", summary)
    print(f"multiturn: {summary['passed']}/{summary['total']}")
    return 0  # a regressão é julgada pelo compare_baseline.py


if __name__ == "__main__":
    sys.exit(main())
