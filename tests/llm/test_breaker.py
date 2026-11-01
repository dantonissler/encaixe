from encaixe.llm.breaker import CircuitBreaker


def test_opens_after_three_failures_and_half_opens_after_reset():
    now = [0.0]
    b = CircuitBreaker(clock=lambda: now[0])
    assert b.state == "closed" and b.allow()
    for _ in range(3):
        b.record_failure()
    assert b.state == "open" and not b.allow()
    now[0] = 31.0
    assert b.allow() and b.state == "half_open"
    assert not b.allow()  # só uma tentativa
    b.record_failure()
    assert b.state == "open"
    now[0] = 62.0
    assert b.allow()
    b.record_success()
    assert b.state == "closed" and b.allow()
