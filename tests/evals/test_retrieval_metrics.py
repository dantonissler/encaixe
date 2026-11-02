from retrieval import count_leaks, mrr, recall_at_k, summarize


def test_recall_and_mrr():
    assert recall_at_k({"a", "b"}, ["x", "a", "y"], 2) == 0.5
    assert mrr({"a"}, ["x", "a"]) == 0.5


def test_edge_cases():
    assert recall_at_k(set(), ["a"], 5) == 0.0
    assert recall_at_k({"a"}, ["x", "y", "a"], 2) == 0.0
    assert mrr({"a"}, ["x", "y"]) == 0.0
    assert mrr({"a", "b"}, ["x", "b", "a"]) == 0.5  # primeiro acerto vale


def test_acl_leak_counting_and_summary():
    assert count_leaks({"salarios"}, ["a", "salarios"]) == 1
    assert count_leaks({"salarios"}, ["a", "b"]) == 0
    rows = [
        {"expected": ["a"], "recall": 1.0, "mrr": 1.0, "leaks": 0},
        {"expected": [], "forbidden": ["s"], "recall": 0.0, "mrr": 0.0, "leaks": 1},
        {"expected": [], "forbidden": ["s"], "recall": 0.0, "mrr": 0.0, "leaks": 0},
    ]
    s = summarize(rows, 5)
    assert s["acl_leaks"] == 1 and s["acl_cases"] == 2
    assert s["recall@5"] == 1.0 and s["cases"] == 1  # casos de ACL fora das médias
