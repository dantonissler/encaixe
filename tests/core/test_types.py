from encaixe.core.types import Principals


def test_principals_as_list_and_fingerprint():
    assert Principals("u1", ("rh",)).as_list() == ["app:*", "user:u1", "group:rh"]
    assert Principals(None).as_list() == ["app:*"]
    a = Principals("u1", ("rh", "ti")).fingerprint()
    b = Principals("u1", ("ti", "rh")).fingerprint()
    assert a == b and len(a) == 16
