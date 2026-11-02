from encaixe.vectorstore.pgvector import rrf_fuse


def test_fuses_two_lists():
    assert rrf_fuse([["a", "b"], ["b", "c"]])[0][0] == "b"


def test_empty_lists():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_score_and_stable_order():
    out = rrf_fuse([["a"], ["b"]])
    assert [i for i, _ in out] == ["a", "b"]
    assert out[0][1] == 1 / 61
