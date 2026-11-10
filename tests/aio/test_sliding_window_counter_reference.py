"""Reference-model tests for the asynchronous sliding window counter.

This is the asynchronous counterpart of
``tests/test_sliding_window_counter_reference.py``. It replays the *same*
scripted event sequences and compares against the *same* independent oracle,
but drives the public ``await``-based API of
:class:`limits.aio.strategies.SlidingWindowCounterRateLimiter` on top of the
async in-memory storage. No real time is waited on: the shared frozen clock
patches the async modules' view of ``time.time()`` exactly as for the sync
storage.
"""

from __future__ import annotations

import math

import pytest

from limits.aio.storage.memory import MemoryStorage as AsyncMemoryStorage
from limits.aio.strategies import (
    SlidingWindowCounterRateLimiter as AsyncSlidingWindowCounterRateLimiter,
)
from limits.limits import RateLimitItemPerSecond
import limits.aio.storage.memory as aio_memory_module
from tests.sliding_window_counter_clock import frozen_time
from tests.sliding_window_counter_reference import (
    Event,
    ReferenceStats,
    expected_trace,
)
from tests.sliding_window_counter_scenarios import (
    BOUNDARY,
    EXPIRY,
    LIMIT,
    all_scenarios,
)

FLOAT_TOL = 1e-9


def _make_limiter(storage=None) -> AsyncSlidingWindowCounterRateLimiter:
    return AsyncSlidingWindowCounterRateLimiter(storage or AsyncMemoryStorage())


def _is_exact_boundary(at: float) -> bool:
    return math.isclose(at % EXPIRY, 0.0, abs_tol=1e-12)


def _as_reference_stats(value: object) -> ReferenceStats:
    assert isinstance(value, ReferenceStats)
    return value


async def _replay_and_compare(limiter, item, events, *, strict: bool) -> int:
    oracle_trace = expected_trace(events, LIMIT, EXPIRY)
    boundary_differences = 0
    with frozen_time(events[0].at) as clock:
        for event, expected in zip(events, oracle_trace):
            clock.set(event.at)
            at_boundary = _is_exact_boundary(event.at)
            if event.kind == "hit":
                got = await limiter.hit(item, "swc-ref", cost=event.cost)
                if got != expected:
                    if not strict and at_boundary and expected is True and got is False:
                        boundary_differences += 1
                        continue
                    raise AssertionError(f"{event}: oracle={expected} got={got}")
            elif event.kind == "test":
                got = await limiter.test(item, "swc-ref", cost=event.cost)
                if got != expected:
                    if not strict and at_boundary and expected is True and got is False:
                        boundary_differences += 1
                        continue
                    raise AssertionError(f"{event}: oracle={expected} got={got}")
            elif event.kind == "stats":
                got = await limiter.get_window_stats(item, "swc-ref")
                expected_stats = _as_reference_stats(expected)
                if not strict and at_boundary:
                    assert 0 <= got.remaining <= LIMIT
                    assert got.remaining <= expected_stats.remaining
                    if got.remaining != expected_stats.remaining:
                        boundary_differences += 1
                    continue
                assert got.reset_time == pytest.approx(
                    expected_stats.reset_time, abs=FLOAT_TOL
                )
                if got.remaining != expected_stats.remaining:
                    raise AssertionError(
                        f"{event}: oracle remaining={expected_stats.remaining} "
                        f"got={got.remaining}"
                    )
            else:
                await limiter.clear(item, "swc-ref")
                assert expected is None
    return boundary_differences


@pytest.mark.parametrize("name, events", all_scenarios(), ids=lambda value: value)
async def test_async_scenarios_match_independent_oracle(name, events):
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        differences = await _replay_and_compare(limiter, item, events, strict=False)
        if name in (
            "mixed_costs_before_on_and_after_boundary",
            "boundary_ownership_half_open_epoch",
        ):
            assert differences > 0
    finally:
        if limiter.storage.timer and not limiter.storage.timer.done():
            limiter.storage.timer.cancel()


async def test_async_test_is_read_only_and_rejections_do_not_count():
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        with frozen_time(BOUNDARY - 3.0):
            assert await limiter.hit(item, "ro", cost=4)
            before = await limiter.get_window_stats(item, "ro")
            for cost in (1, 3, 6, LIMIT + 1):
                await limiter.test(item, "ro", cost=cost)
            after_probes = await limiter.get_window_stats(item, "ro")
            assert after_probes.remaining == before.remaining
            assert after_probes.reset_time == pytest.approx(
                before.reset_time, abs=FLOAT_TOL
            )
            assert not await limiter.hit(item, "ro", cost=LIMIT + 1)
            assert not await limiter.hit(item, "ro", cost=LIMIT)
            after_refusals = await limiter.get_window_stats(item, "ro")
            assert after_refusals.remaining == before.remaining
            assert await limiter.hit(item, "ro", cost=before.remaining)
            assert (await limiter.get_window_stats(item, "ro")).remaining == 0
    finally:
        if limiter.storage.timer and not limiter.storage.timer.done():
            limiter.storage.timer.cancel()


async def test_async_floor_rounding_and_overlimit_cost():
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        with frozen_time(BOUNDARY - 5.0) as clock:
            clock.set(BOUNDARY - 5.0)
            assert await limiter.hit(item, "floor", cost=9)
            clock.set(BOUNDARY + 2.5)
            assert await limiter.hit(item, "floor", cost=4)  # w = 10.75
            assert (await limiter.get_window_stats(item, "floor")).remaining == 0
            assert not await limiter.test(item, "floor", cost=1)
        # Over-limit cost on an empty window reserves nothing.
        with frozen_time(BOUNDARY - 1.0):
            assert not await limiter.hit(item, "big", cost=LIMIT + 1)
            assert not await limiter.test(item, "big", cost=LIMIT + 1)
            assert (await limiter.get_window_stats(item, "big")).remaining == LIMIT
    finally:
        if limiter.storage.timer and not limiter.storage.timer.done():
            limiter.storage.timer.cancel()


async def test_async_clear_then_reuse_and_empty_stats():
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        with frozen_time(BOUNDARY - 2.0) as clock:
            empty = await limiter.get_window_stats(item, "wipe")
            assert empty.remaining == LIMIT
            assert empty.reset_time == pytest.approx(clock(), abs=FLOAT_TOL)
            assert await limiter.hit(item, "wipe", cost=LIMIT)
            clock.set(BOUNDARY)
            await limiter.clear(item, "wipe")
            fresh = await limiter.get_window_stats(item, "wipe")
            assert fresh.remaining == LIMIT
            assert fresh.reset_time == pytest.approx(BOUNDARY, abs=FLOAT_TOL)
            assert await limiter.hit(item, "wipe", cost=7)
            assert (await limiter.get_window_stats(item, "wipe")).remaining == 3
    finally:
        if limiter.storage.timer and not limiter.storage.timer.done():
            limiter.storage.timer.cancel()


# --- Async negative controls: the same mutants must be detected here too. ---


class _AsyncRoundingMutantStorage(AsyncMemoryStorage):
    """Rounds the weighted count to nearest instead of flooring it."""

    async def acquire_sliding_window_entry(self, key, limit, expiry, amount=1):
        if amount > limit:
            return False
        now = aio_memory_module.time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        (
            previous_count,
            previous_ttl,
            current_count,
            _,
        ) = await self._get_sliding_window_info(previous_key, current_key, expiry, now)
        weighted = previous_count * previous_ttl / expiry + current_count
        if round(weighted) + amount > limit:  # mutant: round() over floor()
            return False
        current_count = await self.incr(current_key, 2 * expiry, amount=amount)
        weighted = previous_count * previous_ttl / expiry + current_count
        if round(weighted) > limit:  # mutant: round() over floor()
            await self.decr(current_key, amount)
            return False
        return True


async def test_async_rounding_mutant_is_detected():
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    events = [
        Event(BOUNDARY - 5.0, "hit", 9),
        Event(BOUNDARY + 2.5, "hit", 4),
        Event(BOUNDARY + 2.5, "stats"),
    ]
    oracle_trace = expected_trace(events, LIMIT, EXPIRY)
    mutant = _make_limiter(_AsyncRoundingMutantStorage())
    try:
        disagreements = 0
        with frozen_time(events[0].at) as clock:
            for event, expected in zip(events, oracle_trace):
                clock.set(event.at)
                if event.kind == "hit":
                    got = await mutant.hit(item, "m", cost=event.cost)
                    if got != expected:
                        disagreements += 1
                else:
                    got = await mutant.get_window_stats(item, "m")
                    if got.remaining != expected.remaining:
                        disagreements += 1
        assert disagreements > 0, "async rounding mutant was not detected"
    finally:
        if mutant.storage.timer and not mutant.storage.timer.done():
            mutant.storage.timer.cancel()


def _half_open_previous_ttl(now, expiry, previous_count):
    if previous_count == 0:
        return 0.0
    offset = now - math.floor(now / expiry) * expiry
    if offset <= 0:
        return 0.0
    return max(0.0, expiry - offset)


class _AsyncHalfOpenBoundaryMutantStorage(AsyncMemoryStorage):
    """Async variant honouring the documented half-open bucket convention."""

    async def _info(self, previous_key, current_key, expiry, now):
        previous_count = await self.get(previous_key)
        current_count = await self.get(current_key)
        previous_ttl = _half_open_previous_ttl(now, expiry, previous_count)
        current_ttl = (1 - ((now / expiry) % 1)) * expiry + expiry
        return previous_count, previous_ttl, current_count, current_ttl

    async def get_sliding_window(self, key, expiry):
        now = aio_memory_module.time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        return await self._info(previous_key, current_key, expiry, now)

    async def acquire_sliding_window_entry(self, key, limit, expiry, amount=1):
        if amount > limit:
            return False
        now = aio_memory_module.time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        previous_count, previous_ttl, current_count, _ = await self._info(
            previous_key, current_key, expiry, now
        )
        weighted = previous_count * previous_ttl / expiry + current_count
        if math.floor(weighted) + amount > limit:
            return False
        current_count = await self.incr(current_key, 2 * expiry, amount=amount)
        weighted = previous_count * previous_ttl / expiry + current_count
        if math.floor(weighted) > limit:
            await self.decr(current_key, amount)
            return False
        return True


async def test_async_half_open_boundary_mutant_diverges_at_boundary():
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    events = [
        Event(BOUNDARY - 2.0, "hit", LIMIT),
        Event(BOUNDARY, "stats"),
        Event(BOUNDARY, "hit", 1),
    ]
    oracle_trace = expected_trace(events, LIMIT, EXPIRY)
    production = _make_limiter()
    mutant = _make_limiter(_AsyncHalfOpenBoundaryMutantStorage())
    try:
        prod_outcomes, mutant_outcomes = [], []
        with frozen_time(events[0].at) as clock:
            for event in events:
                clock.set(event.at)
                if event.kind == "hit":
                    prod_outcomes.append(
                        await production.hit(item, "p", cost=event.cost)
                    )
                    mutant_outcomes.append(await mutant.hit(item, "m", cost=event.cost))
                else:
                    prod_outcomes.append(
                        (await production.get_window_stats(item, "p")).remaining
                    )
                    mutant_outcomes.append(
                        (await mutant.get_window_stats(item, "m")).remaining
                    )
        assert prod_outcomes[0] == mutant_outcomes[0] is True
        assert prod_outcomes[1] == 0
        assert mutant_outcomes[1] == oracle_trace[1].remaining == LIMIT
        assert prod_outcomes[2] is False
        assert mutant_outcomes[2] is oracle_trace[2] is True
    finally:
        for limiter in (production, mutant):
            if limiter.storage.timer and not limiter.storage.timer.done():
                limiter.storage.timer.cancel()
