from encaixe.guardrails.normalize import NormalizeGuard


async def test_strips_unicode_tags_and_zero_width(make_ctx):
    text = "ol" + chr(0x200B) + "a" + chr(0xE0041) + chr(0x202E) + " mundo"
    r = await NormalizeGuard(100).check(text, make_ctx())
    assert r.verdict == "pass" and r.redacted_text == "ola mundo"


async def test_too_long_blocks(make_ctx):
    r = await NormalizeGuard(5).check("123456", make_ctx())
    assert r.verdict == "block"


import pytest


@pytest.mark.parametrize(
    "cp", [0x061C, 0x206A, 0x206F, 0x034F, 0x180E, 0xFE0F, 0xE0100, 0x115F, 0x1160, 0x3164, 0xFFF9, 0xFFFB, 0x0007, 0x0085]
)
async def test_strips_extra_invisible_classes(make_ctx, cp):
    r = await NormalizeGuard(100).check("a" + chr(cp) + "b", make_ctx())
    assert r.redacted_text == "ab"


async def test_keeps_newline_and_tab(make_ctx):
    r = await NormalizeGuard(100).check("a\nb\tc", make_ctx())
    assert r.redacted_text == "a\nb\tc"


async def test_length_checked_after_nfkc_expansion(make_ctx):
    r = await NormalizeGuard(5).check(chr(0xFDFA), make_ctx())  # expande para várias letras
    assert r.verdict == "block"
