import pytest

from encaixe.context.aliases import alias_for, resolve_citations

UUID1 = "b1337089-7397-5397-91d2-dc9a0b5bdedb"
UUID2 = "0d7aab7f-625b-5721-ade0-a01f262d7add"
ALIASES = {"T1": UUID1, "T2": UUID2}
VALID = {UUID1, UUID2}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("T1", [UUID1]),
        ("t1", [UUID1]),
        ("[T2]", [UUID2]),
        ("id: T1", [UUID1]),
        ("ID:T2", [UUID2]),
        (" T2 ", [UUID2]),
        ('"T1"', [UUID1]),
        ("T01", [UUID1]),
        (UUID1, [UUID1]),
        (UUID1.upper(), [UUID1]),
        (f"id: {UUID2}", [UUID2]),
        ("T1, T2", [UUID1, UUID2]),
    ],
)
def test_citation_variants_resolve_to_chunk_ids(raw, expected):
    assert resolve_citations([raw], ALIASES, VALID) == (expected, 0)


@pytest.mark.parametrize(
    "raw",
    [
        "T9",  # alias que não foi dado
        "d135c7b6-1111-4222-8333-444455556666",  # UUID inventado
        "-d135c7b6-…-",
        "",
        "manual de férias",
    ],
)
def test_invented_citations_are_dropped(raw):
    assert resolve_citations([raw], ALIASES, VALID) == ([], 1)


def test_dedup_keeps_first_order_and_counts_drops():
    cited, dropped = resolve_citations(["T2", UUID2, "t1", "T7", "[T2]"], ALIASES, VALID)
    assert cited == [UUID2, UUID1]
    assert dropped == 1


def test_alias_to_chunk_not_retrieved_is_dropped():
    assert resolve_citations(["T1"], {"T1": "x"}, set()) == ([], 1)


def test_exact_non_uuid_chunk_id_accepted():
    assert resolve_citations(["k1"], {}, {"k1"}) == (["k1"], 0)


def test_alias_for_reuses_and_continues_numbering():
    aliases = {"T1": "a"}
    assert alias_for(aliases, "a") == "T1"
    assert alias_for(aliases, "b") == "T2"
    assert alias_for(aliases, "b") == "T2"
    assert aliases == {"T1": "a", "T2": "b"}
