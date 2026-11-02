"""Qualidade da recuperação (recall@k e MRR) medida direto no retriever, sem o LLM.

Os casos vêm de `apps/<app>/evals/tests.yaml` (os que têm `metadata.expected_docs`, títulos
de documentos). Cada caso roda com os principais das suas vars (`user_ref`, `groups`), então a
ACL conta. Saída: `evals/results/<app>-retrieval.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evals" / "results"


def recall_at_k(expected: set[str], got: list[str], k: int) -> float:
    """Fração dos esperados presentes nos `k` primeiros resultados."""
    if not expected:
        return 0.0
    return len(expected & set(got[:k])) / len(expected)


def mrr(expected: set[str], got: list[str]) -> float:
    """1 / posição (a partir de 1) do primeiro resultado esperado; 0 se nenhum aparece."""
    for rank, item in enumerate(got, start=1):
        if item in expected:
            return 1 / rank
    return 0.0


def count_leaks(forbidden: set[str], got: list[str]) -> int:
    """Quantos documentos proibidos (ACL) apareceram na recuperação."""
    return len(forbidden & set(got))


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Casos com `expected_docs` e/ou `forbidden_docs` (checagem negativa de ACL)."""
    cases = []
    for test in yaml.safe_load(path.read_text(encoding="utf-8")):
        meta = test.get("metadata") or {}
        expected, forbidden = meta.get("expected_docs"), meta.get("forbidden_docs")
        if not expected and not forbidden:
            continue
        vars_ = test.get("vars") or {}
        groups = [g.strip() for g in str(vars_.get("groups") or "").split(",") if g.strip()]
        cases.append(
            {
                "prompt": vars_["prompt"],
                "user_ref": vars_.get("user_ref") or None,
                "groups": tuple(groups),
                "expected_docs": list(expected or []),
                "forbidden_docs": list(forbidden or []),
            }
        )
    return cases


def summarize(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    scored = [r for r in rows if r["expected"]]  # os casos só de ACL não entram nas médias
    n = len(scored) or 1
    return {
        f"recall@{k}": sum(r["recall"] for r in scored) / n,
        "mrr": sum(r["mrr"] for r in scored) / n,
        "cases": len(scored),
        "acl_cases": sum(1 for r in rows if r.get("forbidden")),
        "acl_leaks": sum(r.get("leaks", 0) for r in rows),
        "details": rows,
    }


async def _run(app_id: str, k: int) -> dict[str, Any]:
    from encaixe.core.config import load_apps
    from encaixe.core.types import Principals
    from encaixe.db.pool import app_scope, create_pool
    from encaixe.embeddings.ollama import OllamaEmbedder
    from encaixe.rag.rerank import CrossEncoderReranker
    from encaixe.rag.retriever import HybridRetriever
    from encaixe.settings import get_settings
    from encaixe.vectorstore.pgvector import PgVectorStore
    from encaixe.wiring import set_runtime_env

    settings = get_settings()
    set_runtime_env()
    app = load_apps(settings.apps_dir)[app_id]
    cases = load_cases(settings.apps_dir / app_id / "evals" / "tests.yaml")
    pool = await create_pool(settings.database_url)
    embedder = OllamaEmbedder(settings)
    try:
        async with app_scope(pool, app_id, None, system=True) as conn:
            rows = await conn.fetch(
                "SELECT id::text, title FROM documents WHERE app_id = $1", app_id
            )
        title_by_id = {r["id"]: r["title"] for r in rows}
        retriever = HybridRetriever(
            embedder, PgVectorStore(pool), CrossEncoderReranker(settings.reranker_model_id)
        )
        details = []
        for case in cases:
            principals = Principals(case["user_ref"], case["groups"])
            chunks = await retriever.retrieve(
                app_id, principals, case["prompt"], app.config.rag.top_k, k
            )
            got: list[str] = []
            for scored in chunks:  # por documento, na ordem do ranking
                title = title_by_id.get(scored.chunk.document_id, scored.chunk.document_id)
                if title not in got:
                    got.append(title)
            expected = set(case["expected_docs"])
            forbidden = set(case["forbidden_docs"])
            details.append(
                {
                    "prompt": case["prompt"],
                    "expected": sorted(expected),
                    "forbidden": sorted(forbidden),
                    "leaks": count_leaks(forbidden, got),
                    "got": got,
                    "recall": recall_at_k(expected, got, k),
                    "mrr": mrr(expected, got),
                }
            )
    finally:
        await embedder.aclose()
        await pool.close()
    return summarize(details, k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    result = asyncio.run(_run(args.app, args.k))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.app}-retrieval.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{args.app}: recall@{args.k}={result[f'recall@{args.k}']:.3f} "
        f"mrr={result['mrr']:.3f} ({result['cases']} casos) "
        f"acl_leaks={result['acl_leaks']} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
