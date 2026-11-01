from encaixe.core.pii import find_pii, mask_pii


def test_valid_cpf_masked_invalid_ignored():
    assert mask_pii("cpf 529.982.247-25") == "cpf [CPF]"
    assert mask_pii("cpf 52998224725") == "cpf [CPF]"
    assert find_pii("111.111.111-12") == []
    assert find_pii("529.982.247-24") == []


def test_cnpj_email_phone():
    assert mask_pii("cnpj 11.222.333/0001-81") == "cnpj [CNPJ]"
    assert mask_pii("cnpj 11222333000181") == "cnpj [CNPJ]"
    assert find_pii("11.222.333/0001-82") == []
    assert mask_pii("a@b.com.br") == "[EMAIL]"
    for tel in ("(11) 98765-4321", "11 3456-7890", "+55 11 98765-4321"):
        assert mask_pii(f"fone {tel}") == "fone [TELEFONE]", tel
    kinds = [m.kind for m in find_pii("x@y.io 529.982.247-25")]
    assert kinds == ["email", "cpf"]
