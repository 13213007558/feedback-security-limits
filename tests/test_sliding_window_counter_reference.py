"""Reference-model tests for the synchronous sliding window counter.

The production :class:`~limits.strategies.SlidingWindowCounterRateLimiter` is
exercised only through its public ``hit`` / ``test`` / ``get_window_stats`` /
``clear`` interface against the in-memory storage, with time fully frozen. Its
answers are compared event by event with an independent oracle
(:mod:`tests.sliding_window_counter_reference`) that re-derives the algorithm
from the documented weight formula instead of reusing production code.

Two boundary-adjacent rules are pinned down explicitly:

* **Rounding rule.** Consumed capacity is ``floor(weighted_count)``; the tests
  engineer fractional weights (e.g. 10.75 -> 10) so truncation or
  nearest-rounding variants cannot pass.
* **Exact boundary ownership.** The documentation defines half-open buckets in
  which the previous bucket's weight is zero the instant a shift happens. The
  timestamp-keyed memory storage instead keeps the previous bucket at *full*
  weight for that single exact-boundary instant (its TTL formula yields a full
  period when ``now`` is an exact multiple of the window). The replay below
  documents both answers and asserts the storage is never more permissive than
  the half-open model there.

The identical scenarios are replayed against the async storage in
``tests/aio/test_sliding_window_counter_reference.py``.
"""

from __future__ import annotations

import math

import pytest

from limits.limits import RateLimitItemPerSecond
import limits.storage.memory as memory_storage_module
from limits.storage.memory import MemoryStorage
from limits.strategies import SlidingWindowCounterRateLimiter
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


def _make_limiter(storage=None) -> SlidingWindowCounterRateLimiter:
    return SlidingWindowCounterRateLimiter(storage or MemoryStorage())


def _is_exact_boundary(at: float) -> bool:
    return math.isclose(at % EXPIRY, 0.0, abs_tol=1e-12)


def _as_reference_stats(value: object) -> ReferenceStats:
    assert isinstance(value, ReferenceStats)
    return value


def _replay_and_compare(limiter, item, events, *, strict: bool) -> int:
    """Replay scripted events and compare against the independent oracle.

    ``strict=False`` tolerates the single documented exact-boundary convention
    of the fixed-epoch memory storage: at that instant it may be stricter than
    the half-open oracle (refusing an admission or reporting less remaining),
    but never more permissive. Returns how many such boundary differences were
    observed, so individual tests can prove the quirk is actually exercised.
    """
    oracle_trace = expected_trace(events, LIMIT, EXPIRY)
    boundary_differences = 0
    with frozen_time(events[0].at) as clock:
        for event, expected in zip(events, oracle_trace):
            clock.set(event.at)
            at_boundary = _is_exact_boundary(event.at)
            if event.kind in ("hit", "test"):
                got = (
                    limiter.hit(item, "swc-ref", cost=event.cost)
                    if event.kind == "hit"
                    else limiter.test(item, "swc-ref", cost=event.cost)
                )
                if got != expected:
                    if not strict and at_boundary and expected is True and got is False:
                        boundary_differences += 1
                        continue
                    raise AssertionError(f"{event}: oracle={expected} got={got}")
            elif event.kind == "stats":
                got = limiter.get_window_stats(item, "swc-ref")
                expected_stats = _as_reference_stats(expected)
                if not strict and at_boundary:
                    # Documented exact-boundary convention: production keeps the
                    # previous bucket fully weighted for this single instant, so
                    # both remaining and reset_time can differ from the
                    # half-open oracle. Remaining must never be over-reported.
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
                limiter.clear(item, "swc-ref")
                assert expected is None
    return boundary_differences


@pytest.mark.parametrize("name, events", all_scenarios(), ids=lambda value: value)
def test_scenarios_match_independent_oracle(name, events):
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        differences = _replay_and_compare(limiter, item, events, strict=False)
        # Every tolerated difference is asserted inside the replay to occur on
        # an exact boundary and to be stricter (never more permissive) than the
        # half-open oracle. The two hand-built sequences are additionally
        # required to exercise that convention explicitly.
        if name in (
            "mixed_costs_before_on_and_after_boundary",
            "boundary_ownership_half_open_epoch",
        ):
            assert differences > 0
    finally:
        limiter.storage.timer.cancel()


def test_half_open_boundary_is_documented_oracle_behaviour():
    """The oracle itself: previous weight is zero exactly on the boundary."""
    events = [
        Event(BOUNDARY - 2.0, "hit", LIMIT),
        Event(BOUNDARY, "stats"),
        Event(BOUNDARY, "hit", 1),
        Event(BOUNDARY, "stats"),
    ]
    trace = expected_trace(events, LIMIT, EXPIRY)
    assert trace[1].remaining == LIMIT  # full capacity restored at the boundary
    assert trace[1].reset_time == pytest.approx(BOUNDARY, abs=FLOAT_TOL)
    assert trace[2] is True
    assert trace[3].remaining == LIMIT - 1


def test_weighted_count_uses_floor_rounding():
    """10.75 weighted consumes 10 (floor); 11 would require nearest/ceil."""
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        with frozen_time(BOUNDARY - 5.0) as clock:
            clock.set(BOUNDARY - 5.0)
            assert limiter.hit(item, "floor", cost=9)
            # ttl 7.5 at BOUNDARY+2.5 -> w = 9*7.5/10 + 4 = 10.75
            clock.set(BOUNDARY + 2.5)
            assert limiter.hit(item, "floor", cost=4)
            assert limiter.get_window_stats(item, "floor").remaining == 0
            assert not limiter.test(item, "floor", cost=1)
            assert not limiter.hit(item, "floor", cost=1)
    finally:
        limiter.storage.timer.cancel()


def test_test_is_read_only_and_rejections_do_not_count():
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        with frozen_time(BOUNDARY - 3.0):
            assert limiter.hit(item, "ro", cost=4)
            before = limiter.get_window_stats(item, "ro")
            for cost in (1, 3, 6, LIMIT + 1):
                limiter.test(item, "ro", cost=cost)
            after_probes = limiter.get_window_stats(item, "ro")
            assert after_probes.remaining == before.remaining
            assert after_probes.reset_time == pytest.approx(
                before.reset_time, abs=FLOAT_TOL
            )
            assert not limiter.hit(item, "ro", cost=LIMIT + 1)
            assert not limiter.hit(item, "ro", cost=LIMIT)
            after_refusals = limiter.get_window_stats(item, "ro")
            assert after_refusals.remaining == before.remaining
            assert limiter.hit(item, "ro", cost=before.remaining)
            assert limiter.get_window_stats(item, "ro").remaining == 0
    finally:
        limiter.storage.timer.cancel()


def test_reset_time_semantics_and_empty_window():
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        with frozen_time(BOUNDARY - 4.0) as clock:
            empty = limiter.get_window_stats(item, "reset")
            assert empty.remaining == LIMIT
            assert empty.reset_time == pytest.approx(clock(), abs=FLOAT_TOL)

            assert limiter.hit(item, "reset", cost=6)
            stats = limiter.get_window_stats(item, "reset")
            assert stats.reset_time == pytest.approx(BOUNDARY, abs=FLOAT_TOL)
            assert stats.remaining == 4

            clock.set(BOUNDARY + 1.0)
            stats = limiter.get_window_stats(item, "reset")
            ttl = EXPIRY - 1.0
            expected_wait = math.fmod(ttl, EXPIRY / 6)
            assert stats.reset_time == pytest.approx(
                clock() + expected_wait, abs=FLOAT_TOL
            )
    finally:
        limiter.storage.timer.cancel()


def test_clear_then_reuse_across_boundary():
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        with frozen_time(BOUNDARY - 2.0) as clock:
            assert limiter.hit(item, "wipe", cost=LIMIT)
            assert limiter.get_window_stats(item, "wipe").remaining == 0
            clock.set(BOUNDARY)
            limiter.clear(item, "wipe")
            fresh = limiter.get_window_stats(item, "wipe")
            assert fresh.remaining == LIMIT
            assert fresh.reset_time == pytest.approx(BOUNDARY, abs=FLOAT_TOL)
            assert limiter.hit(item, "wipe", cost=7)
            assert limiter.get_window_stats(item, "wipe").remaining == 3
    finally:
        limiter.storage.timer.cancel()


def test_cost_larger_than_limit_never_admits_or_reserves():
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    limiter = _make_limiter()
    try:
        with frozen_time(BOUNDARY - 1.0):
            assert not limiter.hit(item, "big", cost=LIMIT + 1)
            assert not limiter.test(item, "big", cost=LIMIT + 1)
            stats = limiter.get_window_stats(item, "big")
            assert stats.remaining == LIMIT
            assert limiter.hit(item, "big", cost=LIMIT)
    finally:
        limiter.storage.timer.cancel()


# --- Negative controls: each temporary mutant must be caught by the suite. ---


class _RoundingMutantStorage(MemoryStorage):
    """Rounds the weighted count to nearest instead of flooring it."""

    def acquire_sliding_window_entry(self, key, limit, expiry, amount=1):
        if amount > limit:
            return False
        now = memory_storage_module.time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        previous_count, previous_ttl, current_count, _ = self._get_sliding_window_info(
            previous_key, current_key, expiry, now
        )
        weighted = previous_count * previous_ttl / expiry + current_count
        if round(weighted) + amount > limit:  # mutant: round() replaces floor()
            return False
        current_count = self.incr(current_key, 2 * expiry, amount=amount)
        weighted = previous_count * previous_ttl / expiry + current_count
        if round(weighted) > limit:  # mutant: round() replaces floor()
            self.decr(current_key, amount)
            return False
        return True


def test_rounding_mutant_is_detected():
    """Nearest-rounding diverges from the floor oracle on a fractional weight."""
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    events = [
        Event(BOUNDARY - 5.0, "hit", 9),
        Event(BOUNDARY + 2.5, "hit", 4),  # oracle: floor(6.75)+4=10 admit
        Event(BOUNDARY + 2.5, "stats"),
    ]
    oracle_trace = expected_trace(events, LIMIT, EXPIRY)
    mutant = _make_limiter(_RoundingMutantStorage())
    try:
        disagreements = 0
        with frozen_time(events[0].at) as clock:
            for event, expected in zip(events, oracle_trace):
                clock.set(event.at)
                if event.kind == "hit":
                    if mutant.hit(item, "m", cost=event.cost) != expected:
                        disagreements += 1
                else:
                    got = mutant.get_window_stats(item, "m")
                    if got.remaining != expected.remaining:
                        disagreements += 1
        assert disagreements > 0, "nearest-rounding mutant was not detected"
    finally:
        mutant.storage.timer.cancel()


def _half_open_previous_ttl(now: float, expiry: int, previous_count: int) -> float:
    """Documented half-open weight lifetime: zero exactly at the shift instant."""
    if previous_count == 0:
        return 0.0
    offset = now - math.floor(now / expiry) * expiry
    if offset <= 0:
        return 0.0
    return max(0.0, expiry - offset)


class _HalfOpenBoundaryMutantStorage(MemoryStorage):
    """Variant honouring the documented half-open bucket convention.

    The only behavioural change versus production is the previous-window TTL:
    it is exactly zero on the boundary instant and grows back to (almost) the
    full period immediately afterwards. Both the stats read path and the
    admission path are overridden so the change is observable everywhere.
    """

    def _info(self, previous_key, current_key, expiry, now):
        previous_count = self.get(previous_key)
        current_count = self.get(current_key)
        previous_ttl = _half_open_previous_ttl(now, expiry, previous_count)
        current_ttl = (1 - ((now / expiry) % 1)) * expiry + expiry
        return previous_count, previous_ttl, current_count, current_ttl

    def get_sliding_window(self, key, expiry):
        now = memory_storage_module.time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        return self._info(previous_key, current_key, expiry, now)

    def acquire_sliding_window_entry(self, key, limit, expiry, amount=1):
        if amount > limit:
            return False
        now = memory_storage_module.time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        previous_count, previous_ttl, current_count, _ = self._info(
            previous_key, current_key, expiry, now
        )
        weighted = previous_count * previous_ttl / expiry + current_count
        if math.floor(weighted) + amount > limit:
            return False
        current_count = self.incr(current_key, 2 * expiry, amount=amount)
        weighted = previous_count * previous_ttl / expiry + current_count
        if math.floor(weighted) > limit:
            self.decr(current_key, amount)
            return False
        return True


def test_half_open_boundary_mutant_diverges_from_production_at_boundary():
    """The corrected half-open variant must disagree exactly at the boundary."""
    item = RateLimitItemPerSecond(LIMIT, EXPIRY)
    boundary_events = [
        Event(BOUNDARY - 2.0, "hit", LIMIT),
        Event(BOUNDARY, "stats"),
        Event(BOUNDARY, "hit", 1),
    ]
    oracle_trace = expected_trace(boundary_events, LIMIT, EXPIRY)
    production = _make_limiter()
    mutant = _make_limiter(_HalfOpenBoundaryMutantStorage())
    try:
        prod_outcomes, mutant_outcomes = [], []
        with frozen_time(boundary_events[0].at) as clock:
            for event in boundary_events:
                clock.set(event.at)
                if event.kind == "hit":
                    prod_outcomes.append(production.hit(item, "p", cost=event.cost))
                    mutant_outcomes.append(mutant.hit(item, "m", cost=event.cost))
                else:
                    prod_outcomes.append(
                        production.get_window_stats(item, "p").remaining
                    )
                    mutant_outcomes.append(mutant.get_window_stats(item, "m").remaining)
        # Pre-boundary saturation is identical.
        assert prod_outcomes[0] == mutant_outcomes[0] is True
        # At the exact boundary the half-open mutant restores the full quota and
        # admits a unit hit, while production keeps the old bucket fully weighted.
        assert prod_outcomes[1] == 0
        assert mutant_outcomes[1] == oracle_trace[1].remaining == LIMIT
        assert prod_outcomes[2] is False
        assert mutant_outcomes[2] is oracle_trace[2] is True
    finally:
        production.storage.timer.cancel()
        mutant.storage.timer.cancel()

    # With no boundary hit, an off-boundary instant must agree across models:
    # both then observe the same decaying previous bucket.
    decay_events = [
        Event(BOUNDARY - 2.0, "hit", LIMIT),
        Event(BOUNDARY + 0.5, "stats"),
    ]
    decay_oracle = expected_trace(decay_events, LIMIT, EXPIRY)
    production = _make_limiter()
    mutant = _make_limiter(_HalfOpenBoundaryMutantStorage())
    try:
        with frozen_time(decay_events[0].at) as clock:
            clock.set(decay_events[0].at)
            assert production.hit(item, "p", cost=LIMIT)
            assert mutant.hit(item, "m", cost=LIMIT)
            clock.set(decay_events[1].at)
            prod_remaining = production.get_window_stats(item, "p").remaining
            mutant_remaining = mutant.get_window_stats(item, "m").remaining
        assert prod_remaining == mutant_remaining == decay_oracle[1].remaining
    finally:
        production.storage.timer.cancel()
        mutant.storage.timer.cancel()
