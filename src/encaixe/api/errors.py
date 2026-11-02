"""Erros HTTP no formato da OpenAI: `{"error": {"message", "type", "code"}}`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = structlog.get_logger(__name__)

RETRY_AFTER = {"Retry-After": "5"}
UNAVAILABLE = "Serviço temporariamente indisponível. Tente novamente em instantes."

_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "not_found_error",
    409: "conflict_error",
    413: "invalid_request_error",
    429: "rate_limit_error",
}


class ApiError(HTTPException):
    """HTTPException com o `code` da OpenAI (ex.: `invalid_api_key`, `model_not_found`)."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code, message, headers)
        self.code = code


def error_body(status: int, message: str, code: str | None = None) -> dict[str, Any]:
    kind = _TYPES.get(status, "server_error" if status >= 500 else "invalid_request_error")
    return {"error": {"message": message, "type": kind, "code": code}}


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    body = error_body(exc.status_code, message, getattr(exc, "code", None))
    return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    problems = "; ".join(
        f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}" for e in exc.errors()
    )
    return JSONResponse(error_body(400, f"Requisição inválida: {problems}"), status_code=400)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)


@asynccontextmanager
async def service_unavailable(resource: str) -> AsyncIterator[None]:
    """Falha de dependência (Postgres, Redis, fila) vira 503 em vez de 500."""
    try:
        yield
    except HTTPException:
        raise
    except Exception as exc:
        log.error("dependency_failed", resource=resource, error=type(exc).__name__)
        raise ApiError(503, UNAVAILABLE, code="service_unavailable", headers=RETRY_AFTER) from exc
