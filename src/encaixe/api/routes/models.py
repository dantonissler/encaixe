"""`GET /v1/models`: as apps que a chave enxerga, no formato da OpenAI (o Open WebUI usa)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from encaixe.api.auth import get_deps, require_app
from encaixe.api.errors import ApiError
from encaixe.wiring import Deps

router = APIRouter()


def _card(request: Request, app_id: str) -> dict[str, Any]:
    return {
        "id": app_id,
        "object": "model",
        "created": request.app.state.started_at,
        "owned_by": "encaixe",
    }


@router.get("/v1/models")
async def list_models(
    request: Request,
    key_app: Annotated[str, Depends(require_app)],
    deps: Annotated[Deps, Depends(get_deps)],
) -> dict[str, Any]:
    data = [_card(request, key_app)] if key_app in deps.apps else []
    return {"object": "list", "data": data}


@router.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    request: Request,
    key_app: Annotated[str, Depends(require_app)],
    deps: Annotated[Deps, Depends(get_deps)],
) -> dict[str, Any]:
    if model_id != key_app or key_app not in deps.apps:
        raise ApiError(404, f"App '{model_id}' não encontrada.", code="model_not_found")
    return _card(request, key_app)
