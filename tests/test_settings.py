from encaixe.settings import Settings


def test_defaults_and_env_override(monkeypatch):
    monkeypatch.setenv("ENCAIXE_MAX_QUEUE", "7")
    s = Settings()
    assert s.max_queue == 7
    assert s.redis_url.endswith(":56379/0")
    assert s.embedding_dims == 1024
