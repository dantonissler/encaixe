from encaixe.guardrails.pii_guard import PiiGuard, PiiLeakGuard

CPF = "529.982.247-25"


async def test_input_flags_and_redacts(make_ctx):
    r = await PiiGuard().check(f"meu cpf {CPF}", make_ctx())
    assert r.verdict == "flag" and r.redacted_text == "meu cpf [CPF]"
    assert (await PiiGuard().check("oi", make_ctx())).verdict == "pass"


async def test_leak_blocks_unknown_cpf_but_allows_user_provided(make_ctx, make_chunk):
    g = PiiLeakGuard()
    assert (await g.check(f"é {CPF}", make_ctx())).verdict == "block"
    assert (await g.check(f"é {CPF}", make_ctx(user_message="52998224725"))).verdict == "pass"
    ctx = make_ctx(retrieved=[make_chunk("c1", f"titular {CPF}")])
    assert (await g.check("cpf 52998224725", ctx)).verdict == "pass"
