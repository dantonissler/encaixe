from encaixe.guardrails.citations import CitationsGuard
from encaixe.output.schemas import RespostaBase


async def test_rejects_unknown_id(make_ctx, make_chunk):
    ctx = make_ctx(retrieved=[make_chunk("c1")])
    g = CitationsGuard()
    assert (await g.check_output(RespostaBase(resposta="x", citacoes=["c9"]), ctx)).verdict == "block"
    assert (await g.check_output(RespostaBase(resposta="x", citacoes=["c1"]), ctx)).verdict == "pass"


async def test_empty_retrieval_and_empty_citations_pass(make_ctx):
    r = await CitationsGuard().check_output(RespostaBase(resposta="x"), make_ctx())
    assert r.verdict == "pass"
