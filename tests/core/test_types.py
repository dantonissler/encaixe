from encaixe.core.types import Principals


def test_principals_as_list_and_fingerprint():
    assert Principals("u1", ("rh",)).as_list() == ["app:*", "user:u1", "group:rh"]
    assert Principals(None).as_list() == ["app:*"]
    a = Principals("u1", ("rh", "ti")).fingerprint()
    b = Principals("u1", ("ti", "rh")).fingerprint()
    assert a == b and len(a) == 16


def test_principals_reject_commas_and_empty():
    import pytest

    for args in (("x,group:rh",), (None, ("a,group:rh",)), ("",), (None, ("",))):
        with pytest.raises(ValueError):
            Principals(*args)


def test_fingerprint_is_unambiguous():
    """Revisão final M-T12: juntar com "|" fazia o grupo `rh|user:x` colidir com o usuário x
    no grupo rh (mesma chave do cache exato)."""
    forged = Principals(None, ("rh|user:x",))
    real = Principals("x", ("rh",))
    assert forged.fingerprint() != real.fingerprint()
