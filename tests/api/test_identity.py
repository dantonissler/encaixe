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


def _status(req, b) -> int:
    with pytest.raises(HTTPException) as exc:
        resolve_identity(req, b)
    return exc.value.status_code


def test_header_user_ref_wins_and_body_only_fills_in():
    """Revisão final M-T13b: o header (posto pelo backend) vence; o corpo só vale sem header."""
    req = request(X_User_Ref="ana")
    principals, _, _ = resolve_identity(req, body())
    assert principals.user_ref == "ana"
    # corpo que concorda com o header é aceito
    principals, _, _ = resolve_identity(req, body(user="ana", metadata={"user_ref": " ana "}))
    assert principals.user_ref == "ana"
    # sem header, o corpo vale: `user` antes de `metadata.user_ref`
    meta = {"user_ref": "do-metadata"}
    principals, _, _ = resolve_identity(request(), body(user="do-corpo", metadata=meta))
    assert principals.user_ref == "do-corpo"
    principals, _, _ = resolve_identity(request(), body(metadata=meta))
    assert principals.user_ref == "do-metadata"
    # valor em branco não conta (nem no header nem no corpo)
    principals, _, _ = resolve_identity(req, body(user="  ", metadata={"user_ref": ""}))
    assert principals.user_ref == "ana"
    # header presente e em branco também é declaração do backend (anônimo): corpo não o troca
    assert _status(request(X_User_Ref=" "), body(user="bia")) == 400
    principals, _, _ = resolve_identity(request(X_User_Ref=" "), body(user=" "))
    assert principals.user_ref is None


@pytest.mark.parametrize(
    "fields",
    [
        {"user": "bruno"},
        {"metadata": {"user_ref": "bruno"}},
        {"user": "ana", "metadata": {"user_ref": "bruno"}},
    ],
)
def test_body_user_ref_disagreeing_with_header_is_400(fields):
    """O navegador não troca de usuário pelo corpo quando o backend declarou o header."""
    assert _status(request(X_User_Ref="ana"), body(**fields)) == 400


def test_header_groups_win_and_body_must_agree():
    req = request(X_User_Groups="financeiro, rh,,rh")
    principals, _, _ = resolve_identity(req, body())
    assert principals.groups == ("financeiro", "rh")
    # mesma lista em outra ordem concorda
    principals, _, _ = resolve_identity(req, body(metadata={"groups": ["rh", "financeiro"]}))
    assert principals.groups == ("financeiro", "rh")
    assert _status(req, body(metadata={"groups": ["diretoria"]})) == 400
    assert _status(req, body(metadata={"groups": []})) == 400
    # header vazio declara "sem grupos": o corpo não acrescenta grupos
    assert _status(request(X_User_Groups=""), body(metadata={"groups": ["rh"]})) == 400
    # sem header, o corpo vale (lista ou CSV)
    principals, _, _ = resolve_identity(request(), body(metadata={"groups": ["diretoria"]}))
    assert principals.groups == ("diretoria",)
    principals, _, _ = resolve_identity(request(), body(metadata={"groups": "a, b"}))
    assert principals.groups == ("a", "b")


def test_conversation_id_header_wins_body_must_agree_and_model_override():
    req = request(X_Conversation_Id="cid-header", X_Encaixe_Model="ollama:outro")
    _, cid, override = resolve_identity(req, body(metadata={"conversation_id": "cid-header"}))
    assert (cid, override) == ("cid-header", "ollama:outro")
    _, cid, override = resolve_identity(req, body())
    assert cid == "cid-header"
    assert _status(req, body(metadata={"conversation_id": "cid-meta"})) == 400
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
