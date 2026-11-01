import pytest
from fastapi import HTTPException
from starlette.requests import Request

from encaixe.api.identity import resolve_identity
from encaixe.api.routes.chat import ChatCompletionRequest


def request(**headers: str) -> Request:
    raw = [(k.replace("_", "-").lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": raw})


def body(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "a", "messages": [{"role": "user", "content": "oi"}], **fields}
    )


def test_precedence_body_user_over_header():
    req = request(X_User_Ref="do-header")
    meta = {"user_ref": "do-metadata"}
    principals, _, _ = resolve_identity(req, body(user="do-corpo", metadata=meta))
    assert principals.user_ref == "do-corpo"
    principals, _, _ = resolve_identity(req, body(metadata=meta))
    assert principals.user_ref == "do-metadata"
    principals, _, _ = resolve_identity(req, body())
    assert principals.user_ref == "do-header"
    # valor em branco não conta: passa para a próxima fonte
    principals, _, _ = resolve_identity(req, body(user="  ", metadata={"user_ref": ""}))
    assert principals.user_ref == "do-header"


def test_groups_from_metadata_list_over_header_csv():
    req = request(X_User_Groups="financeiro, rh,,rh")
    principals, _, _ = resolve_identity(req, body(metadata={"groups": ["diretoria"]}))
    assert principals.groups == ("diretoria",)
    principals, _, _ = resolve_identity(req, body())
    assert principals.groups == ("financeiro", "rh")
    principals, _, _ = resolve_identity(request(), body(metadata={"groups": "a, b"}))
    assert principals.groups == ("a", "b")


def test_conversation_id_header_over_metadata_and_model_override():
    req = request(X_Conversation_Id="cid-header", X_Encaixe_Model="ollama:outro")
    _, cid, override = resolve_identity(req, body(metadata={"conversation_id": "cid-meta"}))
    assert (cid, override) == ("cid-header", "ollama:outro")
    _, cid, override = resolve_identity(request(), body(metadata={"conversation_id": "cid-meta"}))
    assert (cid, override) == ("cid-meta", None)


def test_without_identity_only_public_principal():
    principals, cid, override = resolve_identity(request(), body())
    assert principals.user_ref is None and principals.groups == ()
    assert principals.as_list() == ["app:*"]
    assert cid is None and override is None


def test_works_without_body_for_get_routes():
    principals, cid, _ = resolve_identity(request(X_User_Ref="ana", X_Conversation_Id="c1"), None)
    assert principals.user_ref == "ana" and cid == "c1"


@pytest.mark.parametrize(
    "req,fields",
    [
        (request(), {"user": "ana,group:rh"}),
        (request(), {"metadata": {"user_ref": "ana,bruno"}}),
        (request(X_User_Ref="ana,app:*"), {}),
        (request(), {"metadata": {"groups": ["rh,diretoria"]}}),
        (request(), {"metadata": {"groups": [1]}}),
        (request(), {"metadata": {"user_ref": 7}}),
    ],
)
def test_invalid_identity_is_400(req, fields):
    with pytest.raises(HTTPException) as exc:
        resolve_identity(req, body(**fields))
    assert exc.value.status_code == 400
