from __future__ import annotations

from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_ID = 7_426_001


async def apply_migrations(admin_url: str) -> list[str]:
    """Aplica em ordem as migrações pendentes; devolve as versões aplicadas (idempotente)."""
    conn = await asyncpg.connect(admin_url)
    applied: list[str] = []
    try:
        await conn.execute("SELECT pg_advisory_lock($1)", _LOCK_ID)
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")}
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if path.stem in done:
                    continue
                async with conn.transaction():
                    await conn.execute(path.read_text(encoding="utf-8"))
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)", path.stem
                    )
                applied.append(path.stem)
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_ID)
    finally:
        await conn.close()
    return applied
