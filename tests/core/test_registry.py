import pytest

from encaixe.core import registry


def test_unknown_impl_lists_available():
    @registry.register("kind_x", "alfa")
    def make_alfa(**kw):
        return "alfa"

    assert registry.create("kind_x", "alfa") == "alfa"
    with pytest.raises(registry.UnknownImplementationError) as exc:
        registry.create("kind_x", "beta")
    assert "alfa" in str(exc.value)
