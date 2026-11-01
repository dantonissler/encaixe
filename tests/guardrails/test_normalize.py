from encaixe.guardrails.normalize import NormalizeGuard


async def test_strips_unicode_tags_and_zero_width(make_ctx):
    text = "ol" + chr(0x200B) + "a" + chr(0xE0041) + chr(0x202E) + " mundo"
    r = await NormalizeGuard(100).check(text, make_ctx())
    assert r.verdict == "pass" and r.redacted_text == "ola mundo"


async def test_too_long_blocks(make_ctx):
    r = await NormalizeGuard(5).check("123456", make_ctx())
    assert r.verdict == "block"
