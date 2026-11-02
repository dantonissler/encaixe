"""Vaga de geração: limita gerações simultâneas (GPU) com uma fila curta e limitada."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class QueueFull(Exception):
    """Sem vaga livre e a fila de espera está cheia."""


class QueueTimeout(Exception):
    """Esperou na fila além de `timeout_s`."""


class GenerationGate:
    def __init__(self, max_concurrent: int, max_queue: int, timeout_s: float) -> None:
        if max_concurrent < 1 or max_queue < 0:
            raise ValueError("max_concurrent >= 1 e max_queue >= 0")
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.timeout_s = timeout_s
        self._sem = asyncio.Semaphore(max_concurrent)
        self._active = 0
        self.waiting = 0

    @property
    def available(self) -> int:
        """Vagas livres agora."""
        return self.max_concurrent - self._active

    @property
    def full(self) -> bool:
        """Um `slot()` pedido agora seria recusado com `QueueFull`."""
        return self._sem.locked() and self.waiting >= self.max_queue

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if self._sem.locked():
            if self.waiting >= self.max_queue:
                raise QueueFull()
            self.waiting += 1
            try:
                async with asyncio.timeout(self.timeout_s):
                    await self._sem.acquire()
            except TimeoutError:
                raise QueueTimeout() from None
            finally:
                self.waiting -= 1
        else:
            await self._sem.acquire()
        self._active += 1
        try:
            yield
        finally:
            # Roda também no cancelamento (cliente desconectou): sem isso o bot trava.
            self._active -= 1
            self._sem.release()
