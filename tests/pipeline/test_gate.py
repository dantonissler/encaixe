from __future__ import annotations

import asyncio

import pytest

from encaixe.pipeline.gate import GenerationGate, QueueFull, QueueTimeout


async def _hold(gate: GenerationGate, entered: asyncio.Event, release: asyncio.Event) -> None:
    async with gate.slot():
        entered.set()
        await release.wait()


async def _until(cond) -> None:
    for _ in range(200):
        if cond():
            return
        await asyncio.sleep(0)
    raise AssertionError("condição não atingida")


async def test_free_slot_is_taken_immediately():
    gate = GenerationGate(max_concurrent=2, max_queue=0, timeout_s=1)
    assert gate.available == 2 and gate.waiting == 0
    async with gate.slot():
        assert gate.available == 1
        async with gate.slot():
            assert gate.available == 0
    assert gate.available == 2


async def test_queue_full_raises():
    gate = GenerationGate(max_concurrent=1, max_queue=1, timeout_s=5)
    entered, release = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(_hold(gate, entered, release))
    await entered.wait()
    waiter = asyncio.create_task(_hold(gate, asyncio.Event(), release))
    await _until(lambda: gate.waiting == 1)
    assert gate.full
    with pytest.raises(QueueFull):
        async with gate.slot():
            pass
    assert gate.waiting == 1  # a recusa não mexe na fila
    release.set()
    await asyncio.gather(holder, waiter)
    assert gate.available == 1 and gate.waiting == 0 and not gate.full


async def test_queue_timeout():
    gate = GenerationGate(max_concurrent=1, max_queue=3, timeout_s=0.05)
    entered, release = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(_hold(gate, entered, release))
    await entered.wait()
    with pytest.raises(QueueTimeout):
        async with gate.slot():
            pass
    assert gate.waiting == 0
    release.set()
    await holder
    assert gate.available == 1


async def test_slot_released_on_cancel():
    gate = GenerationGate(max_concurrent=1, max_queue=1, timeout_s=5)
    entered = asyncio.Event()
    task = asyncio.create_task(_hold(gate, entered, asyncio.Event()))
    await entered.wait()
    assert gate.available == 0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.available == 1
    async with gate.slot():  # a vaga liberada pode ser usada de novo
        assert gate.available == 0


async def test_cancel_while_waiting_leaves_no_leak():
    gate = GenerationGate(max_concurrent=1, max_queue=2, timeout_s=5)
    entered, release = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(_hold(gate, entered, release))
    await entered.wait()
    waiter = asyncio.create_task(_hold(gate, asyncio.Event(), asyncio.Event()))
    await _until(lambda: gate.waiting == 1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert gate.waiting == 0
    release.set()
    await holder
    assert gate.available == 1


def test_rejects_invalid_limits():
    with pytest.raises(ValueError):
        GenerationGate(max_concurrent=0, max_queue=1, timeout_s=1)
    with pytest.raises(ValueError):
        GenerationGate(max_concurrent=1, max_queue=-1, timeout_s=1)
