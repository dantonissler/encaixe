"""Quem chama: chave de API (Bearer) → app. A chave nunca vai ao navegador (spec §6.4a)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Request

from encaixe.api.errors import ApiError, service_unavailable
from encaixe.core.config import LoadedApp
from encaixe.db.repos.keys import resolve_key
from encaixe.wiring import Deps

KeyResolver = Callable[[str], Awaitable[str | None]]

_CHALLENGE = {"WWW-Authenticate": "Bearer"}


async def get_deps(request: Request) -> Deps:
    return request.app.state.deps


async def key_resolver(deps: Annotated[Deps, Depends(get_deps)]) -> KeyResolver:
    """Segredo → app_id (ou None). Substituível nos testes via `dependency_overrides`."""

    async def resolve(secret: str) -> str | None:
        return await resolve_key(deps.pool, secret)

    return resolve


async def require_app(
    request: Request, resolve: Annotated[KeyResolver, Depends(key_resolver)]
) -> str:
    """app_id da chave em `Authorization: Bearer <chave>`; 401 sem chave ou com chave inválida."""
    scheme, _, secret = request.headers.get("authorization", "").partition(" ")
    secret = secret.strip()
    if scheme.lower() != "bearer" or not secret:
        raise ApiError(
            401,
            "Envie a chave de API no header 'Authorization: Bearer <chave>'.",
            code="missing_api_key",
            headers=_CHALLENGE,
        )
    async with service_unavailable("postgres"):
        app_id = await resolve(secret)
    if app_id is None:
        raise ApiError(
            401, "Chave de API inválida ou revogada.", code="invalid_api_key", headers=_CHALLENGE
        )
    return app_id


def resolve_app(deps: Deps, key_app: str, requested: str | None) -> LoadedApp:
    """A app da chave; 404 se o cliente pediu outra (`model` do corpo ou header `X-App-Id`)."""
    if requested and requested != key_app:
        raise ApiError(404, f"App '{requested}' não encontrada.", code="model_not_found")
    app = deps.apps.get(key_app)
    if app is None:
        raise ApiError(404, f"App '{key_app}' não está carregada.", code="model_not_found")
    return app
