from retrieval import mrr, recall_at_k


def test_recall_and_mrr():
    assert recall_at_k({"a", "b"}, ["x", "a", "y"], 2) == 0.5
    assert mrr({"a"}, ["x", "a"]) == 0.5


def test_edge_cases():
    assert recall_at_k(set(), ["a"], 5) == 0.0
    assert recall_at_k({"a"}, ["x", "y", "a"], 2) == 0.0
    assert mrr({"a"}, ["x", "y"]) == 0.0
    assert mrr({"a", "b"}, ["x", "b", "a"]) == 0.5  # primeiro acerto vale
