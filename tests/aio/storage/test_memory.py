from __future__ import annotations

import asyncio
import pickle
import time

import pytest

import limits.aio.storage.memory
from limits.aio.storage import MemoryStorage


class TestSerialization:
    async def test_pickle(self):
        storage = MemoryStorage()
        assert 1 == await storage.incr("test", 60)
        assert await storage.acquire_entry("moving_test", 2, 60)
        dump = pickle.dumps(storage)
        restored = pickle.loads(dump)
        assert 2 == await restored.incr("test", 60)
        assert await restored.acquire_entry("moving_test", 2, 60)
        assert not await restored.acquire_entry("moving_test", 2, 60)
        await restored.close()
        await storage.close()


class FakeClock:
    """stand-in for the ``time`` module with a manually advanced clock"""

    def __init__(self, start: float = 1_000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(limits.aio.storage.memory, "time", fake)
    return fake


@pytest.fixture
async def storage(clock):
    instance = MemoryStorage()
    # drive the cleanup explicitly instead of relying on the background task
    await instance.close()
    yield instance
    await instance.close()


def structure_sizes(storage: MemoryStorage) -> dict[str, int]:
    return {
        "counters": len(storage.storage),
        "expirations": len(storage.expirations),
        "events": len(storage.events),
        "locks": len(storage.locks),
        "expiry_index": len(storage._expiry_index),
        "event_expiry_index": len(storage._event_expiry_index),
    }


def due(storage: MemoryStorage, now: float) -> bool:
    return (storage._expiry_index and storage._expiry_index[0][0] <= now) or (
        storage._event_expiry_index and storage._event_expiry_index[0][0] <= now
    )


async def drain(storage: MemoryStorage, now: float, max_keys: int | None = None) -> int:
    passes = 0
    while due(storage, now):
        await storage._expire_events(now=now, max_keys=max_keys)
        passes += 1
        assert passes < 10_000, "cleanup did not converge"
    return passes


@pytest.mark.memory
class TestReclamation:
    async def test_short_lived_keys_fully_reclaimed(self, storage, clock):
        for i in range(5000):
            await storage.incr(f"fixed/{i}", expiry=10)
            await storage.acquire_entry(f"moving/{i}", 5, 10)

        before = structure_sizes(storage)
        assert before == {
            "counters": 5000,
            "expirations": 5000,
            "events": 5000,
            "locks": 10000,
            "expiry_index": 5000,
            "event_expiry_index": 5000,
        }

        now = clock.advance(11)
        await drain(storage, now)

        after = structure_sizes(storage)
        assert all(count == 0 for count in after.values()), after

    async def test_cleanup_pass_is_bounded(self, storage, clock):
        for i in range(2000):
            await storage.incr(f"fixed/{i}", expiry=10)

        now = clock.advance(11)
        reclaimed = await storage._expire_events(now=now, max_keys=100)
        assert reclaimed == 100
        assert len(storage.storage) == 1900
        assert len(storage.locks) == 1900

        passes = await drain(storage, now, max_keys=100)
        assert passes == 19
        assert all(count == 0 for count in structure_sizes(storage).values())

    async def test_persistent_keys_unaffected(self, storage, clock):
        for i in range(2000):
            await storage.incr(f"attacker/{i}", expiry=10)

        assert await storage.incr("persistent", expiry=10) == 1
        for _ in range(5):
            now = clock.advance(10)
            await drain(storage, now)
            # one-shot keys are gone, the active key starts a fresh window
            assert await storage.incr("persistent", expiry=10) == 1
            # the moving window limit is still enforced across cleanups
            for _ in range(3):
                assert await storage.acquire_entry("active", 3, 10)
            assert not await storage.acquire_entry("active", 3, 10)

        assert await storage.get("persistent") == 1
        assert len(storage.storage) == 1
        assert set(storage.events) == {"active"}
        assert len(storage.locks) == 2

    async def test_get_does_not_leak_locks(self, storage):
        for i in range(1000):
            assert await storage.get(f"unknown/{i}") == 0
        assert len(storage.locks) == 0

    async def test_decr_unknown_key_leaves_no_state(self, storage):
        assert await storage.decr("unknown") == 0
        assert "unknown" not in storage.storage
        assert len(storage.expirations) == 0
        assert len(storage.locks) == 0

    async def test_decr_at_expiry_boundary(self, storage, clock):
        assert await storage.incr("key", expiry=10, amount=5) == 5
        clock.advance(11)
        # expired: the decrement must not go negative or resurrect stale state
        assert await storage.decr("key") == 0
        assert "key" not in storage.storage
        # reusing the same key right after expiry restarts consistently
        assert await storage.incr("key", expiry=10) == 1
        assert await storage.get("key") == 1
        assert await storage.decr("key") == 0
        assert "key" not in storage.storage

    async def test_pickle_preserves_expiry_indexes(self, storage):
        await storage.incr("test", 60)
        await storage.acquire_entry("moving_test", 2, 60)
        restored = pickle.loads(pickle.dumps(storage))
        assert len(restored._expiry_index) == 1
        assert len(restored._event_expiry_index) == 1
        await restored.close()


@pytest.mark.memory
class TestCleanupConcurrency:
    async def test_cleanup_collides_with_held_and_waiting_lock(self, storage, clock):
        assert await storage.incr("key", expiry=10) == 1

        async with storage._locked("key"):
            lock_entry = storage.locks["key"]

            waiter = asyncio.create_task(storage.incr("key", expiry=10))
            for _ in range(10):
                await asyncio.sleep(0)
                if storage.locks["key"].holders == 2:
                    break
            assert storage.locks["key"].holders == 2

            clock.advance(11)
            janitor = asyncio.create_task(storage._expire_events(now=clock.time()))
            for _ in range(10):
                await asyncio.sleep(0)
                if storage.locks["key"].holders == 3:
                    break

            # the cleanup and the waiter pile up on the very same lock:
            # it must not be deleted or replaced while held or waited on
            assert storage.locks["key"].holders == 3
            assert storage.locks["key"] is lock_entry
            assert storage.storage["key"] == 1

        incr_result, _ = await asyncio.gather(waiter, janitor)

        # the increment was applied exactly once, on a fresh window
        assert incr_result == 1
        assert await storage.get("key") == 1
        assert storage.locks["key"] is lock_entry

    async def test_no_over_admission_while_cleanup_runs(self, storage, clock):
        limit, workers = 10, 40
        start = asyncio.Event()
        stop = False

        async def janitor():
            while not stop:
                await storage._expire_events(now=clock.time())
                await asyncio.sleep(0)

        async def worker():
            await start.wait()
            return await storage.acquire_entry("key", limit, 60)

        janitor_task = asyncio.create_task(janitor())

        for _ in range(2):
            start.clear()
            tasks = [asyncio.create_task(worker()) for _ in range(workers)]
            await asyncio.sleep(0)
            start.set()
            admitted = await asyncio.gather(*tasks)
            # never more than the limit, and the full limit is admitted
            assert sum(admitted) == limit
            now = clock.advance(61)
            await drain(storage, now)

        stop = True
        await janitor_task
        assert all(
            count == 0 for count in structure_sizes(storage).values()
        ), structure_sizes(storage)

    async def test_concurrent_incr_decr_with_cleanup(self, storage, clock):
        workers, iterations = 20, 100
        start = asyncio.Event()
        stop = False

        async def janitor():
            while not stop:
                await storage._expire_events(now=clock.time())
                await asyncio.sleep(0)

        async def worker():
            await start.wait()
            for _ in range(iterations):
                await storage.incr("key", expiry=3600)
                await storage.decr("key")

        janitor_task = asyncio.create_task(janitor())
        tasks = [asyncio.create_task(worker()) for _ in range(workers)]
        await asyncio.sleep(0)
        start.set()
        await asyncio.gather(*tasks)
        stop = True
        await janitor_task

        # every increment was matched by its decrement: no lost update
        assert await storage.get("key") == 0
        assert "key" not in storage.storage


@pytest.mark.memory
class TestReclamationBenchmark:
    async def test_reclamation_benchmark(self, storage, clock, capsys):
        keys = 20_000

        start = time.perf_counter()
        for i in range(keys):
            await storage.incr(f"fixed/{i}", expiry=10)
            await storage.acquire_entry(f"moving/{i}", 5, 10)
        populate = time.perf_counter() - start

        before = structure_sizes(storage)
        now = clock.advance(11)

        start = time.perf_counter()
        passes = await drain(storage, now)
        reclaim = time.perf_counter() - start

        after = structure_sizes(storage)
        with capsys.disabled():
            print(
                f"\npopulated {keys} key pairs in {populate:.3f}s"
                f"\nsizes before reclamation: {before}"
                f"\nreclaimed in {reclaim:.3f}s over {passes} bounded passes"
                f"\nsizes after reclamation: {after}"
            )

        assert all(count == 0 for count in after.values())
        assert reclaim < 5


@pytest.mark.memory
class TestShutdown:
    async def test_background_cleanup_reclaims_without_requests(self):
        storage = MemoryStorage()
        try:
            await storage.incr("key", expiry=0.2)
            await storage.acquire_entry("moving", 5, 1)
            deadline = time.time() + 5
            while time.time() < deadline and (
                storage.storage or storage.events or storage.locks
            ):
                await asyncio.sleep(0.02)
            assert not storage.storage
            assert not storage.expirations
            assert not storage.events
            assert not storage.locks
        finally:
            await storage.close()

    async def test_close_stops_background_cleanup(self):
        storage = MemoryStorage()
        try:
            await storage.incr("key", expiry=0.2)
            assert storage.timer is not None and not storage.timer.done()

            await storage.close()
            assert storage.timer.done()

            # requests after close do not restart the background cleanup
            await storage.incr("key", expiry=0.2)
            await asyncio.sleep(0.1)
            assert storage.timer.done()

            # lazy eviction still applies after close
            await asyncio.sleep(0.3)
            assert await storage.get("key") == 0
        finally:
            await storage.close()
