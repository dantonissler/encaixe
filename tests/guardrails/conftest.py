from types import SimpleNamespace

import pytest

from encaixe.core.types import Chunk, Principals, RequestContext, ScoredChunk


def _make_chunk(id_: str, text: str = "x") -> ScoredChunk:
    return ScoredChunk(Chunk(id_, "a", "d", text, (), (), {}), 1.0)


@pytest.fixture
def make_chunk():
    return _make_chunk


@pytest.fixture
def make_ctx():
    def _make(user_message="oi", retrieved=(), canary="cnr-abc", require_citations=True):
        app = SimpleNamespace(
            config=SimpleNamespace(id="a", rag=SimpleNamespace(require_citations=require_citations)),
            prompt_version="v1",
        )
        return RequestContext(
            request_id="r", app=app, principals=Principals(None), conversation_id=None,
            has_conversation_identity=False, user_message=user_message, client_history=[],
            canary=canary, retrieved=list(retrieved),
        )

    return _make
