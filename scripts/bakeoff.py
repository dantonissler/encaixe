#!/usr/bin/env python3
"""Disputa de modelos: roda o promptfoo uma vez por modelo e compara numa tabela markdown.

Cada modelo entra pelo header X-Encaixe-Model (ENCAIXE_EVAL_MODEL=ollama:<modelo>), que o
servidor só aceita com ENCAIXE_ENV=dev, e dispensa o fallback. Modelos ausentes do Ollama
(`ollama list`, ou GET /api/tags quando o CLI não está instalado) são pulados com uma nota.
Saída: evals/results/bakeoff-<data>.md. Requer ENCAIXE_EVAL_API_KEY e a API de pé.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evals" / "results"
DEFAULT_MODELS = ["qwen3.5:4b", "llama3.2:3b", "gemma4:e4b"]


def installed_models() -> set[str]:
    """Nomes dos modelos baixados no Ollama."""
    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True, timeout=30
        ).stdout
        return {line.split()[0] for line in out.splitlines()[1:] if line.strip()}
    except (OSError, subprocess.SubprocessError):
        import httpx

        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        tags = httpx.get(f"{url}/api/tags", timeout=10).json()
        return {m["name"] for m in tags.get("models", [])}


def summarize_promptfoo(data: dict[str, Any]) -> dict[str, Any]:
    """Pass rate, latência média (ms) e pass rate por categoria de um JSON do promptfoo."""
    results = (data.get("results") or {}).get("results") or []
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        category = ((r.get("testCase") or {}).get("metadata") or {}).get("category", "outros")
        by_cat[category].append(bool(r.get("success")))
    latencies = [r["latencyMs"] for r in results if r.get("latencyMs") is not None]
    passed = sum(1 for r in results if r.get("success"))
    total = len(results)
    multi = (data.get("extra") or {}).get("multiturn") or {}
    if multi.get("total"):  # casos de escrita (2 turnos), fora do promptfoo
        by_cat["escrita"] = [i < multi["passed"] for i in range(multi["total"])]
        passed += multi["passed"]
        total += multi["total"]
    return {
        "pass_rate": passed / total if total else 0.0,
        "latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "cases": total,
        "categories": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
    }


def build_table(rows: dict[str, dict[str, Any]], skipped: dict[str, str]) -> str:
    categories = sorted({c for r in rows.values() for c in r["categories"]})
    lines = [
        "| modelo | pass rate | latência média (s) | " + " | ".join(categories) + " |",
        "|---|---|---|" + "---|" * len(categories),
    ]
    for model, r in rows.items():
        cells = [f"{r['categories'].get(c, 0):.0%}" for c in categories]
        lines.append(
            f"| {model} | {r['pass_rate']:.0%} | {r['latency_ms'] / 1000:.1f} | "
            + " | ".join(cells)
            + " |"
        )
    if skipped:
        lines += ["", "Modelos pulados:"] + [f"- {m}: {why}" for m, why in skipped.items()]
    return "\n".join(lines) + "\n"


def run_model(app: str, model: str) -> dict[str, Any]:
    out = RESULTS_DIR / f"bakeoff-{app}-{model.replace(':', '_').replace('/', '_')}.json"
    env = {**os.environ, "ENCAIXE_EVAL_MODEL": f"ollama:{model}"}
    out.unlink(missing_ok=True)  # nunca resumir o resultado de uma execução anterior
    # promptfoo sai com código != 0 quando há casos reprovados; o JSON ainda é gravado.
    subprocess.run(
        [str(ROOT / "scripts" / "promptfoo.sh"), "eval", "-c",
         str(ROOT / "apps" / app / "evals" / "promptfooconfig.yaml"), "-o", str(out)],
        env=env, cwd=ROOT, check=False,
    )  # fmt: skip
    if not out.exists():
        raise RuntimeError(f"promptfoo não gerou {out} para {model}")
    subprocess.run(
        [sys.executable, str(ROOT / "evals" / "multiturn.py"), "--app", app, "--results", str(out)],
        env=env, cwd=ROOT, check=False,
    )  # fmt: skip
    return summarize_promptfoo(json.loads(out.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--app", default="exemplo")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args(argv)
    if not os.environ.get("ENCAIXE_EVAL_API_KEY"):
        print("defina ENCAIXE_EVAL_API_KEY.", file=sys.stderr)
        return 2
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    present = installed_models()
    rows: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for model in args.models:
        if model not in present:
            skipped[model] = "não está no `ollama list` (rode `ollama pull`)"
            print(f"pulando {model}: não baixado")
            continue
        print(f"== {model}")
        try:
            rows[model] = run_model(args.app, model)
        except RuntimeError as exc:
            skipped[model] = str(exc)
            print(f"falhou {model}: {exc}")
    table = build_table(rows, skipped)
    path = RESULTS_DIR / f"bakeoff-{dt.date.today().isoformat()}.md"
    path.write_text(table, encoding="utf-8")
    print(table)
    print(f"tabela em {path}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
