from __future__ import annotations

import asyncio
import pickle
import time

import hiro
import pytest

from limits.aio.storage import MemoryStorage, Reservation, ReservationSupport
from limits.aio.storage.base import Storage
from limits.aio.storage.memory import _RESERVED


class PhaseBarrier:
    """One-shot rendezvous point injected at an await boundary.

    A fresh :class:`asyncio.Future` gate is armed for every rendezvous, so a
    barrier released by one operation cannot leak into the next. Cancelling a
    task parked on the gate propagates through :func:`asyncio.wait_for`; the
    storage's own cancellation rollback is then responsible for releasing the
    half-finished hold.
    """

    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.reached = asyncio.Event()
        self._gate: asyncio.Future[None] | None = None

    def _ensure_gate(self) -> asyncio.Future[None]:
        if self._gate is None:
            self._gate = asyncio.get_running_loop().create_future()
        return self._gate

    async def wait(self, phase: str, reservation: Reservation | None) -> None:
        if phase != self.phase:
            return
        self.reached.set()
        await asyncio.wait_for(self._ensure_gate(), timeout=None)

    def release_next(self) -> None:
        gate = self._ensure_gate()
        if not gate.done():
            gate.set_result(None)
        self._gate = None
        self.reached = asyncio.Event()

    def uninstall(self, storage: MemoryStorage) -> None:
        storage._reservation_event = None


def ledger(storage: MemoryStorage, key: str) -> tuple[int, int, int]:
    """Return (committed, held by active reservations, tracked records)."""
    now = time.time()
    committed = storage.storage.get(key, 0)
    held = sum(
        reservation.amount
        for reservation, state, expires_at in storage.reservations.values()
        if reservation.key == key and state == _RESERVED and expires_at > now
    )
    return committed, held, len(storage.reservations)


class TestReservationProtocol:
    async def test_capacity_held_then_committed(self):
        storage = MemoryStorage()
        reservation = await storage.reserve("key", limit=2, expiry=60)
        assert isinstance(reservation, Reservation)
        assert ledger(storage, "key") == (0, 1, 1)
        # A second request must see the hold as consumed capacity.
        assert await storage.reserve("key", limit=2, expiry=60, amount=2) is None
        assert await storage.commit_reservation(reservation) is True
        assert await storage.get("key") == 1
        assert ledger(storage, "key") == (1, 0, 0)

    async def test_cancel_releases_only_its_own_hold(self):
        storage = MemoryStorage()
        first = await storage.reserve("key", limit=2, expiry=60)
        second = await storage.reserve("key", limit=2, expiry=60)
        assert ledger(storage, "key") == (0, 2, 2)
        assert await storage.reserve("key", limit=2, expiry=60) is None
        assert await storage.cancel_reservation(first) is True
        assert ledger(storage, "key") == (0, 1, 1)
        freed = await storage.reserve("key", limit=2, expiry=60)
        assert freed is not None and freed.id != first.id
        assert await storage.commit_reservation(second) is True
        assert await storage.get("key") == 1

    async def test_handles_are_unique(self):
        storage = MemoryStorage()
        handles = {
            (await storage.reserve(f"k{i}", limit=1, expiry=60)).id
            for i in range(25)
        }
        assert len(handles) == 25

    async def test_amount_over_limit_is_refused(self):
        storage = MemoryStorage()
        assert await storage.reserve("key", limit=2, expiry=60, amount=3) is None
        assert ledger(storage, "key") == (0, 0, 0)

    async def test_hit_remains_atomic_immediate_consumption(self):
        storage = MemoryStorage()
        limiter_storage = MemoryStorage()
        # Existing incr path still consumes immediately, no reservation ledger.
        assert await limiter_storage.incr("key", 60) == 1
        assert await limiter_storage.incr("key", 60) == 2
        assert limiter_storage.reservations == {}
        # A committed reservation and an immediate hit share the same ledger.
        reservation = await storage.reserve("key", limit=2, expiry=60)
        assert await storage.incr("key", 60) == 1
        assert await storage.reserve("key", limit=2, expiry=60) is None
        assert await storage.commit_reservation(reservation) is True
        assert await storage.get("key") == 2
        assert await storage.incr("key", 60) == 3

    async def test_repeat_commit_and_cancel_are_idempotent(self):
        storage = MemoryStorage()
        reservation = await storage.reserve("key", limit=1, expiry=60)
        assert await storage.commit_reservation(reservation) is True
        assert await storage.commit_reservation(reservation) is False
        assert await storage.cancel_reservation(reservation) is False
        assert await storage.get("key") == 1

        other = await storage.reserve("other", limit=1, expiry=60)
        assert await storage.cancel_reservation(other) is True
        assert await storage.cancel_reservation(other) is False
        assert await storage.commit_reservation(other) is False
        assert await storage.get("other") == 0

    async def test_unknown_handle_changes_nothing(self):
        storage = MemoryStorage()
        ghost = Reservation(id="nope", key="key", amount=1, expiry=60, created_at=0.0)
        assert await storage.commit_reservation(ghost) is False
        assert await storage.cancel_reservation(ghost) is False
        assert ledger(storage, "key") == (0, 0, 0)

    async def test_multi_amount_reservation_shares_the_ledger(self):
        storage = MemoryStorage()
        bulk = await storage.reserve("key", limit=5, expiry=60, amount=3)
        assert bulk is not None
        assert ledger(storage, "key") == (0, 3, 1)
        # Only two units remain for both holds and immediate consumption.
        assert await storage.reserve("key", limit=5, expiry=60, amount=3) is None
        small = await storage.reserve("key", limit=5, expiry=60, amount=2)
        assert small is not None
        assert await storage.reserve("key", limit=5, expiry=60) is None
        assert await storage.cancel_reservation(bulk) is True
        assert ledger(storage, "key") == (0, 2, 1)
        assert await storage.commit_reservation(small) is True
        assert await storage.get("key") == 2

    async def test_reservation_interleaves_with_real_fixed_window_limiter(self):
        from limits import RateLimitItemPerMinute
        from limits.aio.strategies import FixedWindowRateLimiter

        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(3)
        key = item.key_for("tenant")

        hold = await storage.reserve(key, limit=3, expiry=60)
        assert hold is not None
        # The reservation ledger accounts for the outstanding hold.
        committed, held, _ = ledger(storage, key)
        assert (committed, held) == (0, 1)
        # Immediate hits remain atomic and consume the committed ledger
        # independently of outstanding reservations.
        assert await limiter.hit(item, "tenant") is True
        assert await storage.get(key) == 1
        # Further reservations see both the hold and the committed hit.
        assert await storage.reserve(key, limit=3, expiry=60, amount=2) is None
        assert await storage.commit_reservation(hold) is True
        assert await storage.get(key) == 2
        # Canceling after commit cannot refund the consumed quota.
        assert await storage.cancel_reservation(hold) is False
        assert await storage.get(key) == 2


class TestReservationRaces:
    async def test_commit_wins_against_cancel(self):
        storage = MemoryStorage()
        reservation = await storage.reserve("key", limit=1, expiry=60)
        barrier = PhaseBarrier("commit.completed")
        storage._reservation_event = barrier.wait

        commit = asyncio.create_task(storage.commit_reservation(reservation))
        await barrier.reached.wait()
        # The commit has released the lock and parked before returning. The
        # cancel now enters the same critical section and must observe the
        # terminal (committed) state.
        cancel = asyncio.create_task(storage.cancel_reservation(reservation))
        await asyncio.sleep(0)
        barrier.release_next()
        assert await commit is True
        assert await cancel is False
        assert ledger(storage, "key") == (1, 0, 0)

    async def test_cancel_wins_against_commit(self):
        storage = MemoryStorage()
        reservation = await storage.reserve("key", limit=1, expiry=60)
        barrier = PhaseBarrier("cancel.completed")
        storage._reservation_event = barrier.wait

        cancel = asyncio.create_task(storage.cancel_reservation(reservation))
        await barrier.reached.wait()
        commit = asyncio.create_task(storage.commit_reservation(reservation))
        await asyncio.sleep(0)
        barrier.release_next()
        assert await cancel is True
        assert await commit is False
        assert ledger(storage, "key") == (0, 0, 0)
        # Freed capacity is immediately reusable.
        assert await storage.reserve("key", limit=1, expiry=60) is not None

    async def test_concurrent_reserves_serialize_on_the_lock(self):
        storage = MemoryStorage()
        entered = PhaseBarrier("reserve.entered")
        recorded = PhaseBarrier("reserve.recorded")
        first_seen = False

        async def hook(phase, reservation):
            nonlocal first_seen
            if phase == "reserve.entered":
                # Only the first (lock-holding) task parks; the second task
                # must be allowed to enter once it acquires the lock.
                if not first_seen:
                    first_seen = True
                    await entered.wait(phase, reservation)
            else:
                await recorded.wait(phase, reservation)

        storage._reservation_event = hook

        first = asyncio.create_task(storage.reserve("key", limit=1, expiry=60))
        # The first task holds the lock and parks at the in-lock barrier.
        await entered.reached.wait()
        second = asyncio.create_task(storage.reserve("key", limit=1, expiry=60))
        for _ in range(10):
            await asyncio.sleep(0)
        assert not second.done()
        entered.release_next()
        await recorded.reached.wait()
        assert ledger(storage, "key") == (0, 1, 1)
        recorded.release_next()
        winner = await first
        loser = await second
        assert winner is not None
        assert loser is None
        assert ledger(storage, "key") == (0, 1, 1)
        assert await storage.cancel_reservation(winner) is True

    async def test_cancel_after_storage_hold_but_before_handle_returned(self):
        storage = MemoryStorage()
        barrier = PhaseBarrier("reserve.recorded")
        storage._reservation_event = barrier.wait

        task = asyncio.create_task(storage.reserve("key", limit=1, expiry=60))
        await barrier.reached.wait()
        # The hold exists in storage while the caller still has no handle.
        assert ledger(storage, "key") == (0, 1, 1)
        task.cancel()
        with pytest.raises((asyncio.CancelledError, asyncio.TimeoutError)):
            await task
        barrier.uninstall(storage)
        # No permanently leaked occupancy: capacity is available again.
        assert ledger(storage, "key") == (0, 0, 0)
        retry = await storage.reserve("key", limit=1, expiry=60)
        assert retry is not None
        assert await storage.cancel_reservation(retry) is True


class TestReservationExpiry:
    def test_expired_reservation_cannot_commit_and_releases_capacity(self):
        storage = MemoryStorage()
        with hiro.Timeline().freeze() as timeline:
            reservation = asyncio.run(storage.reserve("key", limit=1, expiry=10))
            assert ledger(storage, "key") == (0, 1, 1)
            timeline.forward(11)
            # Sweep via the same path the background timer uses.
            asyncio.run(storage._MemoryStorage__expire_events())
            assert ledger(storage, "key") == (0, 0, 0)
            assert asyncio.run(storage.commit_reservation(reservation)) is False
            assert asyncio.run(storage.get("key")) == 0
            # The released quota can be reserved again.
            replacement = asyncio.run(storage.reserve("key", limit=1, expiry=10))
            assert replacement is not None
            assert asyncio.run(storage.commit_reservation(replacement)) is True

    def test_cancel_never_refunds_a_later_window(self):
        storage = MemoryStorage()
        with hiro.Timeline().freeze() as timeline:
            reservation = asyncio.run(storage.reserve("key", limit=5, expiry=10))
            assert asyncio.run(storage.commit_reservation(reservation)) is True
            assert asyncio.run(storage.get("key")) == 1
            timeline.forward(11)
            # The fixed window has rolled over; the counter is gone.
            assert asyncio.run(storage.get("key")) == 0
            asyncio.run(storage.incr("key", 10))
            # Cancelling an old handle must not manufacture credit in the
            # new window: cancel only ever removes this handle's hold.
            stale = Reservation(
                id=reservation.id,
                key="key",
                amount=reservation.amount,
                expiry=reservation.expiry,
                created_at=reservation.created_at,
            )
            assert asyncio.run(storage.cancel_reservation(stale)) is False
            assert asyncio.run(storage.get("key")) == 1

    def test_commit_after_expiry_is_refused_but_cancel_is_clean(self):
        storage = MemoryStorage()
        with hiro.Timeline().freeze() as timeline:
            reservation = asyncio.run(storage.reserve("key", limit=1, expiry=10))
            timeline.forward(5)
            assert asyncio.run(storage.commit_reservation(reservation)) is True
            timeline.forward(20)
            assert asyncio.run(storage.get("key")) == 0


class TestReservationHousekeeping:
    async def test_reset_clears_reservations(self):
        storage = MemoryStorage()
        await storage.reserve("key", limit=2, expiry=60)
        await storage.reserve("key", limit=2, expiry=60)
        assert len(storage.reservations) == 2
        await storage.reset()
        assert storage.reservations == {}
        assert await storage.reserve("key", limit=2, expiry=60) is not None

    async def test_clear_drops_matching_reservations(self):
        storage = MemoryStorage()
        keep = await storage.reserve("other", limit=1, expiry=60)
        drop = await storage.reserve("key", limit=1, expiry=60)
        await storage.clear("key")
        assert await storage.commit_reservation(drop) is False
        assert await storage.commit_reservation(keep) is True

    async def test_pickle_preserves_active_reservation(self):
        storage = MemoryStorage()
        reservation = await storage.reserve("key", limit=2, expiry=60)
        restored = pickle.loads(pickle.dumps(storage))
        assert isinstance(restored, ReservationSupport)
        assert await restored.commit_reservation(reservation) is True
        assert await restored.get("key") == 1


class TestUnsupportedBackends:
    def test_base_storage_rejects_the_protocol(self):
        class DummyStorage(Storage):
            STORAGE_SCHEME = ["dummy"]

            @property
            def base_exceptions(self):
                return ValueError

            async def incr(self, key, expiry, amount=1):
                return 0

            async def get(self, key):
                return 0

            async def get_expiry(self, key):
                return 0.0

            async def check(self):
                return True

            async def reset(self):
                return 0

            async def clear(self, key):
                return None

        storage = DummyStorage()
        handle = Reservation(id="h", key="key", amount=1, expiry=60, created_at=0.0)

        async def exercise():
            with pytest.raises(NotImplementedError):
                await storage.reserve("key", 1, 60)
            with pytest.raises(NotImplementedError):
                await storage.commit_reservation(handle)
            with pytest.raises(NotImplementedError):
                await storage.cancel_reservation(handle)

        asyncio.run(exercise())
