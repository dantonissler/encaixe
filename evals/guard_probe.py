"""Sondagem do guard de conteúdo com `evals/guard_probes.yaml` (D2 da Task 15).

Mede, para um modelo do Ollama, os falsos positivos nas entradas e saídas legítimas e a taxa
de bloqueio dos pedidos nocivos, com o mesmo formato de chamada do guard `content_safety`:
entrada = só a mensagem do usuário; saída = par usuário/assistente.

    uv run python evals/guard_probe.py --model llama-guard3-1b-cpu --repeats 3

Respostas aceitas: Llama Guard ("safe"/"unsafe\\nS1") e ShieldGemma/Granite Guardian
("Yes" = viola a política, "No" = não viola). Ao fim o modelo é descarregado do Ollama
(`keep_alive: 0`), porque a RAM da máquina de desenvolvimento é curta.

Ambiente: ENCAIXE_OLLAMA_BASE_URL (http://localhost:11434).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "evals" / "guard_probes.yaml"


def parse_verdict(content: str) -> str:
    """'block', 'pass' ou 'unknown' a partir da 1ª palavra da resposta do modelo."""
    words = content.strip().split()
    first = words[0].strip(".,:;!").lower() if words else ""
    if first in ("unsafe", "yes"):
        return "block"
    if first in ("safe", "no"):
        return "pass"
    return "unknown"


def ask(client: httpx.Client, base: str, model: str, messages: list[dict[str, str]]) -> str:
    # Só o veredito (1ª palavra) importa; o ShieldGemma explicaria "passo a passo" sem limite.
    body = {"model": model, "messages": messages, "stream": False, "options": {"num_predict": 10}}
    resp = client.post(f"{base}/api/chat", json=body)
    resp.raise_for_status()
    return str(resp.json()["message"]["content"])


def run(model: str, repeats: int, base: str, probes: dict[str, Any]) -> dict[str, Any]:
    cases: list[tuple[str, str, list[dict[str, str]], str]] = []
    for text in probes["entrada_legitima"]:
        cases.append(("entrada_legitima", text, [{"role": "user", "content": text}], "pass"))
    for pair in probes["saida_legitima"]:
        messages = [
            {"role": "user", "content": pair["usuario"]},
            {"role": "assistant", "content": pair["resposta"]},
        ]
        cases.append(("saida_legitima", pair["resposta"], messages, "pass"))
    for text in probes["nociva"]:
        cases.append(("nociva", text, [{"role": "user", "content": text}], "block"))

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    with httpx.Client(timeout=120.0) as client:
        ask(client, base, model, [{"role": "user", "content": "oi"}])  # carga fora da medição
        for group, text, messages, expected in cases:
            verdicts, raw = [], []
            for _ in range(repeats):
                t0 = time.perf_counter()
                content = ask(client, base, model, messages)
                latencies.append(time.perf_counter() - t0)
                verdicts.append(parse_verdict(content))
                raw.append(content.strip()[:40])
            wrong = sum(v != expected for v in verdicts)
            rows.append(
                {"grupo": group, "texto": text, "esperado": expected, "erros": wrong, "raw": raw}
            )
        client.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0})

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
        "modelo": model,
        "repeticoes": repeats,
        "entrada_legitima": summary("entrada_legitima"),
        "saida_legitima": summary("saida_legitima"),
        "nociva": summary("nociva"),
        "latencia_mediana_s": round(latencies[len(latencies) // 2], 2) if latencies else None,
        "linhas": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--probes", type=Path, default=PROBES)
    parser.add_argument("--out", type=Path, help="grava o resultado completo em JSON")
    args = parser.parse_args()
    base = os.environ.get("ENCAIXE_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    probes = yaml.safe_load(args.probes.read_text(encoding="utf-8"))
    result = run(args.model, args.repeats, base, probes)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fp_in, fp_out, harm = result["entrada_legitima"], result["saida_legitima"], result["nociva"]
    print(
        f"{args.model}: falsos positivos entrada {fp_in['erros']}/{fp_in['chamadas']} "
        f"({fp_in['casos_com_erro']}/{fp_in['casos']} casos), saída {fp_out['erros']}/"
        f"{fp_out['chamadas']} ({fp_out['casos_com_erro']}/{fp_out['casos']} casos); "
        f"nocivos bloqueados {harm['chamadas'] - harm['erros']}/{harm['chamadas']} "
        f"({harm['casos'] - harm['casos_com_erro']}/{harm['casos']} casos sempre); "
        f"latência mediana {result['latencia_mediana_s']} s"
    )
    for row in result["linhas"]:
        if row["erros"]:
            print(
                f"  [{row['grupo']}] {row['erros']}/{args.repeats} {row['texto'][:70]!r} {row['raw']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
