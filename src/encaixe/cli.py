from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable

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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    func: Callable = args.func
    sys.exit(asyncio.run(func(args)))
