from encaixe.context.tokens import HeuristicTokenCounter


def test_heuristic():
    c = HeuristicTokenCounter()
    assert c.count("") == 0
    assert c.count("abc") == 1
    assert c.count("abcd") == 2
    assert c.count("a" * 30) == 10
