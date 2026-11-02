from encaixe.guardrails.canary import CanaryGuard, make_canary


async def test_detects_canary(make_ctx):
    c = make_canary()
    assert c.startswith("cnr-") and len(c) == 16
    ctx = make_ctx(canary=c)
    assert (await CanaryGuard().check(f"vazou {c}", ctx)).verdict == "block"
    assert (await CanaryGuard().check("ok", ctx)).verdict == "pass"


async def test_detects_obfuscated_canary(make_ctx):
    ctx = make_ctx(canary="cnr-abcdef123456")
    text = "c n r-ab" + chr(0x200B) + "cdef 123456"
    assert (await CanaryGuard().check(text, ctx)).verdict == "block"
