from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

State = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """closed -> open após `failure_threshold` falhas seguidas; após `reset_after_s`
    permite uma tentativa (half_open): sucesso fecha, falha reabre."""

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_after_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._clock = clock
        self._failures = 0
        self._opened_at = 0.0
        self._state: State = "closed"

    @property
    def state(self) -> State:
        return self._state

    def allow(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open" and self._clock() - self._opened_at >= self.reset_after_s:
            self._state = "half_open"
            return True  # única tentativa de teste
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failures += 1
        if self._state == "half_open" or self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = self._clock()


class BreakerRegistry:
    """Um breaker por spec de modelo, em processo."""

    def __init__(self, **breaker_kwargs) -> None:
        self._kwargs = breaker_kwargs
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, spec: str) -> CircuitBreaker:
        if spec not in self._breakers:
            self._breakers[spec] = CircuitBreaker(**self._kwargs)
        return self._breakers[spec]
