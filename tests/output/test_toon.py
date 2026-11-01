from pydantic import BaseModel

from encaixe.output.toon import from_toon, to_toon


def test_roundtrip_list_of_dicts():
    data = [{"id": "a", "texto": "um"}, {"id": "b", "texto": "dois"}]
    assert from_toon(to_toon(data)) == data


def test_basemodel():
    class M(BaseModel):
        resposta: str
        citacoes: list[str]

    m = M(resposta="oi", citacoes=["x", "y"])
    assert from_toon(to_toon(m)) == {"resposta": "oi", "citacoes": ["x", "y"]}
