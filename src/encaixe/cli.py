from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path

from encaixe.settings import get_settings


async def _migrate(args: argparse.Namespace) -> int:
    from encaixe.db.migrate import apply_migrations

    applied = await apply_migrations(get_settings().database_admin_url)
    print("migrações aplicadas: " + (", ".join(applied) if applied else "nenhuma"))
    return 0


async def _keys_create(args: argparse.Namespace) -> int:
    from encaixe.db.pool import create_pool
    from encaixe.db.repos.keys import create_key

    pool = await create_pool(get_settings().database_url)
    try:
        secret = await create_key(pool, args.app)
    finally:
        await pool.close()
    print(secret)
    return 0


def _client(key: str):
    import httpx

    from encaixe.seed import api_url

    return httpx.Client(base_url=api_url(), headers={"Authorization": f"Bearer {key}"}, timeout=60)


def _send(app_id: str, key: str, files: list[Path], acl: list[str] | None) -> int:
    import httpx

    from encaixe import seed

    with _client(key) as client:
        try:
            docs = [seed.upload(client, app_id, f, acl) for f in files]
            rows = seed.wait_all(client, app_id, docs)
        except httpx.HTTPError as exc:
            print(f"falha ao falar com a API ({seed.api_url()}): {exc}", file=sys.stderr)
            return 1
    print(seed.format_table(rows))
    pending = [r for r in rows if r["status"] not in seed.TERMINAL]
    failed = [r for r in rows if r["status"] == "failed"]
    if pending:
        print(f"{len(pending)} documento(s) não terminaram em 300 s (o worker está de pé?)")
    return 1 if pending or failed else 0


async def _seed(args: argparse.Namespace) -> int:
    key = os.environ.get("ENCAIXE_EVAL_API_KEY")
    if not key:
        from encaixe.db.pool import create_pool
        from encaixe.db.repos.keys import create_key

        pool = await create_pool(get_settings().database_url)
        try:
            key = await create_key(pool, args.app)
        finally:
            await pool.close()
        print(f"chave criada (guarde-a): export ENCAIXE_EVAL_API_KEY={key}")
    else:
        print("usando a chave de ENCAIXE_EVAL_API_KEY")
    docs_dir = get_settings().apps_dir / args.app / "docs"
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        print(f"nenhum .md em {docs_dir}", file=sys.stderr)
        return 1
    return await asyncio.to_thread(_send, args.app, key, files, None)


async def _ingest(args: argparse.Namespace) -> int:
    key = args.key or os.environ.get("ENCAIXE_API_KEY") or os.environ.get("ENCAIXE_EVAL_API_KEY")
    if not key:
        print("informe --key ou defina ENCAIXE_API_KEY.", file=sys.stderr)
        return 2
    acl = [a.strip() for a in args.acl.split(",") if a.strip()] if args.acl is not None else None
    return await asyncio.to_thread(_send, args.app, key, [Path(args.file)], acl)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="encaixe")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("migrate", help="aplica as migrações do banco")
    p.set_defaults(func=_migrate)

    keys = sub.add_parser("keys", help="gerencia chaves de API")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)
    create = keys_sub.add_parser("create", help="cria uma chave para um app")
    create.add_argument("--app", required=True)
    create.set_defaults(func=_keys_create)

    p = sub.add_parser("seed", help="envia o corpus de apps/<app>/docs pela API")
    p.add_argument("--app", required=True)
    p.set_defaults(func=_seed)

    p = sub.add_parser("ingest", help="envia um arquivo pela API")
    p.add_argument("--app", required=True)
    p.add_argument("--acl", help="CSV: app:*, user:<id>, group:<nome> (padrão: front-matter)")
    p.add_argument("--key", help="chave da API (padrão: ENCAIXE_API_KEY)")
    p.add_argument("file")
    p.set_defaults(func=_ingest)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    func: Callable = args.func
    sys.exit(asyncio.run(func(args)))
