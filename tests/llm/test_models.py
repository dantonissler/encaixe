import pytest
from pydantic_ai.models.test import TestModel

from encaixe.llm.models import build_model


def test_parse_ollama_spec(settings):
    assert build_model("ollama:qwen3.5:4b", settings).model_name == "qwen3.5:4b"


def test_test_spec(settings):
    assert isinstance(build_model("test", settings), TestModel)


def test_unknown_prefix_raises(settings):
    with pytest.raises(ValueError):
        build_model("openai:gpt", settings)


async def test_hung_ollama_fails_after_one_request_timeout():
    """Sem novas tentativas no cliente: cada tentativa esperaria o timeout por requisição
    inteiro e um principal travado comeria o prazo que sobra para o reserva."""
    import asyncio
    import time

    from pydantic_ai import Agent
    from pydantic_ai.exceptions import ModelAPIError
    from pydantic_ai.settings import ModelSettings

    from encaixe.settings import Settings

    connections = 0

    async def hang(reader, writer):  # aceita a conexão e nunca responde
        nonlocal connections
        connections += 1
        await asyncio.sleep(3600)

    server = await asyncio.start_server(hang, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    model = build_model("ollama:x", Settings(ollama_base_url=f"http://127.0.0.1:{port}"))
    agent = Agent(model, model_settings=ModelSettings(timeout=0.5))
    started = time.monotonic()
    try:
        with pytest.raises(ModelAPIError):
            await agent.run("oi")
    finally:
        server.close()
    assert connections == 1
    assert time.monotonic() - started < 1.5
