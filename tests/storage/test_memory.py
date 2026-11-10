from __future__ import annotations

import pickle
import threading
import time

import pytest

import limits.storage.memory
from limits.storage import MemoryStorage


class TestSerialization:
    def test_pickle(self):
        storage = MemoryStorage()
        assert 1 == storage.incr("test", 60)
        assert storage.acquire_entry("moving_test", 2, 60)
        dump = pickle.dumps(storage)
        restored = pickle.loads(dump)
        assert 2 == restored.incr("test", 60)
        assert restored.acquire_entry("moving_test", 2, 60)
        assert not restored.acquire_entry("moving_test", 2, 60)
        restored.close()
        storage.close()


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
    monkeypatch.setattr(limits.storage.memory, "time", fake)
    return fake


@pytest.fixture
def storage(clock):
    instance = MemoryStorage()
    # drive the cleanup explicitly instead of relying on the background timer
    instance.close()
    yield instance
    instance.close()


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


def drain(storage: MemoryStorage, now: float, max_keys: int | None = None) -> int:
    passes = 0
    while due(storage, now):
        storage._expire_events(now=now, max_keys=max_keys)
        passes += 1
        assert passes < 10_000, "cleanup did not converge"
    return passes


def wait_for_holders(
    storage: MemoryStorage, key: str, holders: int, timeout: float = 5
):
    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = storage.locks.get(key)
        if entry is not None and entry.holders == holders:
            return entry
        time.sleep(0.001)
    raise AssertionError(f"lock for {key!r} never reached {holders} holders")


@pytest.mark.memory
class TestReclamation:
    def test_short_lived_keys_fully_reclaimed(self, storage, clock):
        for i in range(5000):
            storage.incr(f"fixed/{i}", expiry=10)
            storage.acquire_entry(f"moving/{i}", 5, 10)

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
        drain(storage, now)

        after = structure_sizes(storage)
        assert all(count == 0 for count in after.values()), after

    def test_cleanup_pass_is_bounded(self, storage, clock):
        for i in range(2000):
            storage.incr(f"fixed/{i}", expiry=10)

        now = clock.advance(11)
        reclaimed = storage._expire_events(now=now, max_keys=100)
        assert reclaimed == 100
        assert len(storage.storage) == 1900
        assert len(storage.locks) == 1900

        passes = drain(storage, now, max_keys=100)
        assert passes == 19
        assert all(count == 0 for count in structure_sizes(storage).values())

    def test_persistent_keys_unaffected(self, storage, clock):
        for i in range(2000):
            storage.incr(f"attacker/{i}", expiry=10)

        assert storage.incr("persistent", expiry=10) == 1
        for _ in range(5):
            now = clock.advance(10)
            drain(storage, now)
            # one-shot keys are gone, the active key starts a fresh window
            assert storage.incr("persistent", expiry=10) == 1
            # the moving window limit is still enforced across cleanups
            for _ in range(3):
                assert storage.acquire_entry("active", 3, 10)
            assert not storage.acquire_entry("active", 3, 10)

        assert storage.get("persistent") == 1
        assert len(storage.storage) == 1
        assert set(storage.events) == {"active"}
        assert len(storage.locks) == 2

    def test_get_does_not_leak_locks(self, storage):
        for i in range(1000):
            assert storage.get(f"unknown/{i}") == 0
        assert len(storage.locks) == 0

    def test_decr_unknown_key_leaves_no_state(self, storage):
        assert storage.decr("unknown") == 0
        assert "unknown" not in storage.storage
        assert len(storage.expirations) == 0
        assert len(storage.locks) == 0

    def test_decr_at_expiry_boundary(self, storage, clock):
        assert storage.incr("key", expiry=10, amount=5) == 5
        clock.advance(11)
        # expired: the decrement must not go negative or resurrect stale state
        assert storage.decr("key") == 0
        assert "key" not in storage.storage
        # reusing the same key right after expiry restarts consistently
        assert storage.incr("key", expiry=10) == 1
        assert storage.get("key") == 1
        assert storage.decr("key") == 0
        assert "key" not in storage.storage

    def test_pickle_preserves_expiry_indexes(self, storage):
        storage.incr("test", 60)
        storage.acquire_entry("moving_test", 2, 60)
        restored = pickle.loads(pickle.dumps(storage))
        assert len(restored._expiry_index) == 1
        assert len(restored._event_expiry_index) == 1
        restored.close()


@pytest.mark.memory
class TestCleanupConcurrency:
    def test_cleanup_collides_with_held_and_waiting_lock(self, storage, clock):
        assert storage.incr("key", expiry=10) == 1
        results = {}

        def waiter():
            results["incr"] = storage.incr("key", expiry=10)

        def janitor():
            results["reclaimed"] = storage._expire_events(now=clock.time())

        with storage._locked("key"):
            lock_entry = storage.locks["key"]

            wait_thread = threading.Thread(target=waiter)
            wait_thread.start()
            wait_for_holders(storage, "key", 2)

            clock.advance(11)
            janitor_thread = threading.Thread(target=janitor)
            janitor_thread.start()
            wait_for_holders(storage, "key", 3)

            # the cleanup and the waiter pile up on the very same lock:
            # it must not be deleted or replaced while held or waited on
            time.sleep(0.01)
            assert storage.locks["key"] is lock_entry
            assert storage.storage["key"] == 1

        wait_thread.join(timeout=5)
        janitor_thread.join(timeout=5)
        assert not wait_thread.is_alive()
        assert not janitor_thread.is_alive()

        # the increment was applied exactly once, on a fresh window
        assert results["incr"] == 1
        assert storage.get("key") == 1
        assert storage.locks["key"] is lock_entry

    def test_no_over_admission_while_cleanup_runs(self, storage, clock):
        limit, workers = 10, 40
        barrier = threading.Barrier(workers)
        stop = threading.Event()
        admitted = []

        def janitor():
            while not stop.is_set():
                storage._expire_events(now=clock.time())
                time.sleep(0.0005)

        def worker():
            barrier.wait()
            if storage.acquire_entry("key", limit, 60):
                admitted.append(1)

        janitors = [threading.Thread(target=janitor) for _ in range(2)]
        for thread in janitors:
            thread.start()

        for _ in range(2):
            threads = [threading.Thread(target=worker) for _ in range(workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            # never more than the limit, and the full limit is admitted
            assert len(admitted) == limit
            admitted.clear()
            now = clock.advance(61)
            drain(storage, now)
            barrier.reset()

        stop.set()
        for thread in janitors:
            thread.join()
        assert all(
            count == 0 for count in structure_sizes(storage).values()
        ), structure_sizes(storage)

    def test_concurrent_incr_decr_with_cleanup(self, storage, clock):
        workers, iterations = 20, 100
        barrier = threading.Barrier(workers)
        stop = threading.Event()

        def janitor():
            while not stop.is_set():
                storage._expire_events(now=clock.time())
                time.sleep(0.0005)

        def worker():
            barrier.wait()
            for _ in range(iterations):
                storage.incr("key", expiry=3600)
                storage.decr("key")

        janitor_thread = threading.Thread(target=janitor)
        janitor_thread.start()
        threads = [threading.Thread(target=worker) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        stop.set()
        janitor_thread.join()

        # every increment was matched by its decrement: no lost update
        assert storage.get("key") == 0
        assert "key" not in storage.storage


@pytest.mark.memory
class TestReclamationBenchmark:
    def test_reclamation_benchmark(self, storage, clock, capsys):
        keys = 20_000

        start = time.perf_counter()
        for i in range(keys):
            storage.incr(f"fixed/{i}", expiry=10)
            storage.acquire_entry(f"moving/{i}", 5, 10)
        populate = time.perf_counter() - start

        before = structure_sizes(storage)
        now = clock.advance(11)

        start = time.perf_counter()
        passes = drain(storage, now)
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
    def test_background_cleanup_reclaims_without_requests(self):
        storage = MemoryStorage()
        try:
            storage.incr("key", expiry=0.2)
            storage.acquire_entry("moving", 5, 1)
            deadline = time.time() + 5
            while time.time() < deadline and (
                storage.storage or storage.events or storage.locks
            ):
                time.sleep(0.02)
            assert not storage.storage
            assert not storage.expirations
            assert not storage.events
            assert not storage.locks
        finally:
            storage.close()

    def test_close_stops_background_cleanup(self):
        storage = MemoryStorage()
        try:
            storage.incr("key", expiry=0.2)
            deadline = time.time() + 1
            while not storage.timer.is_alive() and time.time() < deadline:
                time.sleep(0.005)
            assert storage.timer.is_alive()

            storage.close()
            assert not storage.timer.is_alive()

            # requests after close do not restart the background cleanup
            storage.incr("key", expiry=0.2)
            time.sleep(0.2)
            assert not storage.timer.is_alive()

            # lazy eviction still applies after close
            time.sleep(0.3)
            assert storage.get("key") == 0
        finally:
            storage.close()
