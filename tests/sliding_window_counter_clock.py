"""Deterministic virtual clock used to freeze time around window boundaries.

The production in-memory storages and both strategy flavours read the wall
clock through the ``time`` module imported into their own namespace, i.e. they
call ``time.time()``. :func:`frozen_time` swaps the module-level ``time`` name
in each relevant module for a tiny shim whose ``time()`` answers from a
scripted clock, while every other attribute (``monotonic``, ``sleep`` ...) is
delegated to the real ``time`` module. No real window ever has to elapse.

Only the modules reachable from the public sliding window counter API are
patched; pytest itself and third-party libraries keep the real wall clock.
"""

from __future__ import annotations

import contextlib
import time as real_time
from types import ModuleType
from typing import Iterator

import limits.aio.storage.memory as aio_memory_storage
import limits.aio.strategies as aio_strategies
import limits.storage.memory as sync_memory_storage
import limits.strategies as sync_strategies

PATCHED_MODULES = (
    sync_memory_storage,
    sync_strategies,
    aio_memory_storage,
    aio_strategies,
)


class FrozenClock:
    """Mutable holder of the single virtual timestamp currently in effect."""

    def __init__(self, start: float) -> None:
        self.now = float(start)

    def set(self, value: float) -> None:
        self.now = float(value)

    def __call__(self) -> float:
        return self.now


class _FrozenTimeModule(ModuleType):
    """``time`` module shim: frozen ``time()``, everything else delegated."""

    def __init__(self, clock: FrozenClock) -> None:
        super().__init__("frozen_time")
        self._clock = clock

    def time(self) -> float:
        return self._clock()

    def __getattr__(self, name: str) -> object:
        return getattr(real_time, name)


@contextlib.contextmanager
def frozen_time(start: float) -> Iterator[FrozenClock]:
    clock = FrozenClock(start)
    shim = _FrozenTimeModule(clock)
    originals: dict[ModuleType, ModuleType] = {
        module: getattr(module, "time") for module in PATCHED_MODULES
    }
    for module in PATCHED_MODULES:
        setattr(module, "time", shim)
    try:
        yield clock
    finally:
        for module, original in originals.items():
            setattr(module, "time", original)
