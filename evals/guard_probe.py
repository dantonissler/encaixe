"""Sondagem do guard de conteúdo com `evals/guard_probes.yaml` (D2 da Task 15).

Mede os falsos positivos nas entradas e saídas legítimas e a taxa de bloqueio dos pedidos
nocivos usando o próprio guard `content_safety` (`encaixe.guardrails.safety`): entrada = só a
mensagem do usuário; saída = par usuário/assistente.

    # configuração em camadas (a padrão da API)
    uv run python evals/guard_probe.py --model shieldgemma-2b-cpu \\
        --secondary llama-guard3-1b-cpu --ignore S1,S6
    # um modelo sozinho (todo veredito conta)
    uv run python evals/guard_probe.py --model llama-guard3-1b-cpu

Ao fim (mesmo com erro) os modelos são descarregados do Ollama (`keep_alive: 0`), porque a
RAM da máquina de desenvolvimento é curta. Ambiente: ENCAIXE_OLLAMA_BASE_URL
(http://localhost:11434).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import yaml

from encaixe.guardrails.safety import ContentSafetyGuard
from encaixe.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "evals" / "guard_probes.yaml"


def build_guard(base: str, model: str, secondary: str, ignore: list[str]) -> ContentSafetyGuard:
    settings = Settings(
        _env_file=None,
        ollama_base_url=base,
        safety_model=model,
        safety_secondary_model=secondary,
        safety_secondary_ignore_categories=ignore,
    )
    return ContentSafetyGuard(settings, client=httpx.AsyncClient(timeout=120.0))


async def run(guard: Any, probes: dict[str, Any], repeats: int) -> dict[str, Any]:
    cases: list[tuple[str, str, Callable[[], Awaitable[Any]], str]] = []
    for text in probes["entrada_legitima"]:
        cases.append(("entrada_legitima", text, lambda t=text: guard.check(t, None), "pass"))
    for pair in probes["saida_legitima"]:
        u, a = pair["usuario"], pair["resposta"]
        cases.append(("saida_legitima", a, lambda u=u, a=a: guard.check_output_pair(u, a), "pass"))
    for text in probes["nociva"]:
        cases.append(("nociva", text, lambda t=text: guard.check(t, None), "block"))

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for group, text, call, expected in cases:
        verdicts, raw = [], []
        for _ in range(repeats):
            started = time.perf_counter()
            try:
                result = await call()
                verdicts.append(result.verdict)
                raw.append(result.reason or result.verdict)
            except Exception as exc:  # noqa: BLE001 - conta como erro, a sondagem segue
                verdicts.append("unavailable")
                raw.append(f"indisponível: {type(exc).__name__}")
            latencies.append(time.perf_counter() - started)
        wrong = sum(v != expected for v in verdicts)
        rows.append(
            {"grupo": group, "texto": text, "esperado": expected, "erros": wrong, "raw": raw}
        )

    def summary(group: str) -> dict[str, int]:
        items = [r for r in rows if r["grupo"] == group]
        return {
            "casos": len(items),
            "casos_com_erro": sum(1 for r in items if r["erros"]),
            "erros": sum(r["erros"] for r in items),
            "chamadas": len(items) * repeats,
        }

    latencies.sort()
    return {
        "repeticoes": repeats,
        "entrada_legitima": summary("entrada_legitima"),
        "saida_legitima": summary("saida_legitima"),
        "nociva": summary("nociva"),
        "latencia_mediana_s": round(latencies[len(latencies) // 2], 2) if latencies else None,
        "linhas": rows,
    }


async def probe(
    guard: Any,
    probes: dict[str, Any],
    repeats: int,
    *,
    unload: Callable[[list[str]], Awaitable[None]],
) -> dict[str, Any]:
    """Carrega os modelos fora da medição, mede e descarrega sempre (RAM curta)."""
    try:
        await guard.check("oi", None)
        return await run(guard, probes, repeats)
    finally:
        await unload(guard.models())


def ollama_unloader(base: str) -> Callable[[list[str]], Awaitable[None]]:
    async def unload(models: list[str]) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for model in models:
                try:
                    await client.post(
                        f"{base}/api/generate", json={"model": model, "keep_alive": 0}
                    )
                except httpx.HTTPError:
                    print(f"aviso: não consegui descarregar {model}", file=sys.stderr)

    return unload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="guard principal (todo veredito conta)")
    parser.add_argument("--secondary", default="", help="guard secundário (vazio: nenhum)")
    parser.add_argument("--ignore", default="S1,S6", help="categorias ignoradas no secundário")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--probes", type=Path, default=PROBES)
    parser.add_argument("--out", type=Path, help="grava o resultado completo em JSON")
    args = parser.parse_args()
    base = os.environ.get("ENCAIXE_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ignore = [c.strip() for c in args.ignore.split(",") if c.strip()]
    probes = yaml.safe_load(args.probes.read_text(encoding="utf-8"))
    guard = build_guard(base, args.model, args.secondary, ignore)
    result = asyncio.run(probe(guard, probes, args.repeats, unload=ollama_unloader(base)))
    label = args.model + (
        f" + {args.secondary} (ignora {','.join(ignore)})" if args.secondary else ""
    )
    result["configuracao"] = label
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fp_in, fp_out, harm = result["entrada_legitima"], result["saida_legitima"], result["nociva"]
    print(
        f"{label}: falsos positivos entrada {fp_in['erros']}/{fp_in['chamadas']} "
        f"({fp_in['casos_com_erro']}/{fp_in['casos']} casos), saída {fp_out['erros']}/"
        f"{fp_out['chamadas']} ({fp_out['casos_com_erro']}/{fp_out['casos']} casos); "
        f"nocivos bloqueados {harm['chamadas'] - harm['erros']}/{harm['chamadas']} "
        f"({harm['casos'] - harm['casos_com_erro']}/{harm['casos']} casos sempre); "
        f"latência mediana {result['latencia_mediana_s']} s"
    )
    for row in result["linhas"]:
        if row["erros"]:
            text = row["texto"][:70]
            print(f"  [{row['grupo']}] {row['erros']}/{args.repeats} {text!r} {row['raw']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
