from encaixe.cache.exact import ExactCache, clear_response_cache, exact_key, is_cacheable
from encaixe.core.types import ChatMessage, GuardResult, Principals
from encaixe.output.schemas import RespostaBase


def _key(**over):
    kw = {
        "app_id": "a",
        "app_version": "v1",
        "model": "m",
        "principals": Principals(user_ref="u1"),
        "history": [ChatMessage(role="user", content="oi")],
        "user_message": "Qual o prazo?",
        "corpus_version": 1,
    }
    kw.update(over)
    return exact_key(**kw)


def test_key_differs_by_principals_and_corpus_version():
    base = _key()
    assert base.startswith("enc:resp:")
    assert _key(principals=Principals(user_ref="u2")) != base
    assert _key(corpus_version=2) != base
    assert _key(app_id="b") != base
    assert _key(app_version="v2") != base


def test_key_normalizes_whitespace_and_case():
    assert _key(user_message="  QUAL   o\n prazo? ") == _key()


async def test_set_get_with_ttl(fake_redis):
    cache = ExactCache(fake_redis)
    assert await cache.get("k") is None
    out = RespostaBase(resposta="ok", citacoes=["c1"])
    await cache.set("k", out, 60)
    assert await cache.get("k") == out.model_dump(mode="json")
    assert 0 < await fake_redis.ttl("k") <= 60


class _Cfg:
    def __init__(self, enabled=True):
        self.cache = type("C", (), {"exact": type("E", (), {"enabled": enabled, "ttl_s": 10})()})()


class _Ctx:
    def __init__(self, **kw):
        self.app = type("A", (), {"config": _Cfg(kw.pop("enabled", True))})()
        self.tools_called = kw.get("tools_called", [])
        self.guard_results = kw.get("guard_results", [])
        self.extra_context = kw.get("extra_context", [])


def test_not_cacheable_when_tool_called_or_pii():
    assert is_cacheable(_Ctx(), [])
    assert not is_cacheable(_Ctx(enabled=False), [])
    assert not is_cacheable(_Ctx(tools_called=["t"]), [])
    assert not is_cacheable(_Ctx(guard_results=[GuardResult("pii", "flag")]), [])
    assert not is_cacheable(_Ctx(guard_results=[GuardResult("injection", "block")]), [])
    assert not is_cacheable(_Ctx(), [GuardResult("grounding", "block")])
    assert not is_cacheable(_Ctx(extra_context=["x"]), [])
    assert is_cacheable(
        _Ctx(guard_results=[GuardResult("pii", "pass")]), [GuardResult("g", "flag")]
    )


def test_document_search_tool_does_not_prevent_caching():
    """Com a saída por ferramenta o modelo passou a chamar buscar_documentos em ~metade dos
    turnos de documentos. Ela só lê o corpus (a chave já leva a versão do corpus e os
    principais), então a resposta continua cacheável; ferramentas MCP (dados por usuário
    que mudam) continuam fora."""
    assert is_cacheable(_Ctx(tools_called=["buscar_documentos"]), [])
    assert is_cacheable(_Ctx(tools_called=["buscar_documentos", "buscar_documentos"]), [])
    assert not is_cacheable(_Ctx(tools_called=["buscar_documentos", "listar_tarefas"]), [])
    assert not is_cacheable(_Ctx(tools_called=["criar_tarefa:pendente"]), [])


async def test_clear_response_cache_removes_only_responses(fake_redis):
    """`make eval` limpa as respostas antes de rodar (revisão final I5)."""
    for i in range(3):
        await fake_redis.set(f"enc:resp:{i}", "{}")
    await fake_redis.set("enc:emb:bge:abc", "[]")
    await fake_redis.set("arq:queue", "x")
    assert await clear_response_cache(fake_redis) == 3
    assert sorted(await fake_redis.keys("*")) == ["arq:queue", "enc:emb:bge:abc"]
    assert await clear_response_cache(fake_redis) == 0
