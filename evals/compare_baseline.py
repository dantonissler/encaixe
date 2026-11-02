"""Compara os resultados da avaliação com a baseline e falha se houver regressão.

Lê `evals/results/<app>.json` (promptfoo, com `extra.multiturn` do multiturn.py) e
`evals/results/<app>-retrieval.json`; a baseline fica em `evals/baselines/<app>.json`.
Sem baseline: passa e avisa (use `--update` para gravá-la).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("pass_rate", "recall@5", "mrr")


def compare(current: dict, baseline: dict, max_drop: float = 0.05) -> list[str]:
    """Regressões: métricas que caíram mais de `max_drop` (absoluto) abaixo da baseline."""
    problems = []
    for name in METRICS:
        if name not in baseline:
            continue
        before, now = baseline[name], current.get(name, 0.0)
        if before - now > max_drop + 1e-9:
            problems.append(f"{name}: {before:.3f} -> {now:.3f} (queda de {before - now:.3f})")
    return problems


def pass_rate(promptfoo: dict[str, Any]) -> float:
    """Taxa de aprovação: casos do promptfoo mais os do multiturn (`extra.multiturn`)."""
    stats = (promptfoo.get("results") or {}).get("stats") or {}
    passed = stats.get("successes", 0)
    total = passed + stats.get("failures", 0) + stats.get("errors", 0)
    extra = (promptfoo.get("extra") or {}).get("multiturn") or {}
    passed += extra.get("passed", 0)
    total += extra.get("total", 0)
    return passed / total if total else 0.0


def collect(app: str, results_dir: Path) -> dict[str, float]:
    promptfoo = json.loads((results_dir / f"{app}.json").read_text(encoding="utf-8"))
    retrieval = json.loads((results_dir / f"{app}-retrieval.json").read_text(encoding="utf-8"))
    return {
        "pass_rate": pass_rate(promptfoo),
        "recall@5": retrieval["recall@5"],
        "mrr": retrieval["mrr"],
    }


def main(
    argv: list[str] | None = None,
    *,
    results_dir: Path = ROOT / "evals" / "results",
    baselines_dir: Path = ROOT / "evals" / "baselines",
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--update", action="store_true", help="grava os resultados como baseline")
    parser.add_argument("--max-drop", type=float, default=0.05)
    args = parser.parse_args(argv)

    try:
        current = collect(args.app, results_dir)
    except FileNotFoundError as exc:
        print(f"resultado ausente: {exc.filename}. Rode `make eval APP={args.app}`.")
        return 1
    print(" ".join(f"{k}={v:.3f}" for k, v in current.items()))
    path = baselines_dir / f"{args.app}.json"
    if args.update:
        baselines_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"baseline gravada em {path}")
        return 0
    if not path.exists():
        print(f"sem baseline ({path}); passando. Use --update para gravá-la.")
        return 0
    problems = compare(current, json.loads(path.read_text(encoding="utf-8")), args.max_drop)
    for problem in problems:
        print(f"REGRESSÃO {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
