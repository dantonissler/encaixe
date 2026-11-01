import httpx
import respx

from encaixe.guardrails.safety import LlamaGuard


def _reply(content):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


@respx.mock
async def test_parses_unsafe_categories(settings, make_ctx):
    route = respx.post(f"{settings.ollama_base_url}/api/chat").mock(return_value=_reply("unsafe\nS1,S10"))
    r = await LlamaGuard(settings).check("texto", make_ctx())
    assert r.verdict == "block" and r.reason == "S1,S10"
    assert route.called


@respx.mock
async def test_safe_passes(settings, make_ctx):
    respx.post(f"{settings.ollama_base_url}/api/chat").mock(return_value=_reply("safe"))
    assert (await LlamaGuard(settings).check("oi", make_ctx())).verdict == "pass"


@respx.mock
async def test_output_sends_pair_and_garbage_raises(settings, make_ctx):
    import json

    route = respx.post(f"{settings.ollama_base_url}/api/chat").mock(return_value=_reply("safe"))
    await LlamaGuard(settings).check_response("resp", make_ctx(user_message="perg"))
    body = json.loads(route.calls[0].request.content)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["stream"] is False
    respx.post(f"{settings.ollama_base_url}/api/chat").mock(return_value=_reply("hmm"))
    import pytest

    with pytest.raises(ValueError):
        await LlamaGuard(settings).check("x", make_ctx())
