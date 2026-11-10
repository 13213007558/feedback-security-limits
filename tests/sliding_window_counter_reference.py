"""Independent reference model for the sliding window counter strategy.

This module deliberately re-derives the algorithm from first principles instead
of importing or reusing :class:`limits.strategies.SlidingWindowCounterRateLimiter`
or any storage helper, so the production implementation cannot act as its own
oracle.  Time is supplied explicitly to every method: nothing in here reads the
wall clock, which lets the tests pin virtual time around window boundaries.

Model (window length ``W`` seconds, capacity ``L`` weighted units):

* The timeline is partitioned into fixed, half-open epochs of length ``W``:
  epoch ``k`` covers ``[k*W, (k+1)*W)``. A request exactly on a boundary
  ``k*W`` belongs to the new epoch ``k`` (boundary ownership = new window).
* Two counters are kept: the count acquired in the previous epoch ``P`` and in
  the current epoch ``C``. Both are integer totals of the ``cost`` of accepted
  hits; rejected hits never touch either counter.
* At time ``t`` the elapsed fraction of the previous epoch that is still inside
  the sliding look-back window is the remaining TTL of the previous epoch
  divided by ``W``. The *weighted* count is therefore

      w(t) = P * previous_ttl(t) / W + C

  Rounding rule for admission/quota decisions: the consumed integer capacity is
  ``floor(w)`` (round towards minus infinity), never plain ``int()``/truncation
  or rounding to nearest. A hit of cost ``c`` is admitted iff
  ``floor(w) + c <= L``; the read-only probe ``test`` uses the same predicate
  but performs no state change. A cost larger than the whole capacity can never
  be admitted even on an empty window.
* ``remaining`` is ``max(0, L - floor(w))``.
* ``reset_time`` is the earliest virtual timestamp at which one unit of
  capacity becomes usable again:
    - empty window -> the current time itself (nothing is held);
    - otherwise the earlier of (a) the next current-epoch boundary and
      (b) the time at which one more weighted unit of the previous epoch has
      decayed, i.e. ``t + previous_ttl mod (W / P)`` when ``P > 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ReferenceStats:
    reset_time: float
    remaining: int


class ReferenceSlidingWindowCounter:
    """Deterministic, clock-free oracle for one rate limit key."""

    def __init__(self, limit: int, expiry: int) -> None:
        if expiry <= 0:
            raise ValueError("expiry must be positive")
        self.limit = int(limit)
        self.expiry = int(expiry)
        self._initialized = False
        self._previous_epoch = 0
        self._current_epoch = 0
        self._previous_count = 0
        self._current_count = 0

    def _epoch_index(self, at: float) -> int:
        return math.floor(at / self.expiry)

    def _roll(self, at: float) -> None:
        """Advance the epoch pair to ``at`` without consulting any clock."""
        epoch = self._epoch_index(at)
        if not self._initialized:
            self._initialized = True
            self._current_epoch = epoch
            self._previous_epoch = epoch - 1
            return
        if epoch == self._current_epoch:
            return
        if epoch == self._current_epoch + 1:
            self._previous_epoch = self._current_epoch
            self._previous_count = self._current_count
            self._current_epoch = epoch
            self._current_count = 0
            return
        # Jumped more than one window: every counter has aged out.
        self._previous_epoch = epoch - 1
        self._previous_count = 0
        self._current_epoch = epoch
        self._current_count = 0

    def _previous_ttl(self, at: float) -> float:
        """Remaining weight lifetime of the previous epoch at time ``at``.

        Half-open convention: at the exact boundary ``at == current_epoch*W``
        the previous epoch has just fully elapsed, so its TTL (and therefore
        its weight) is zero. It then counts down from ``W - offset`` as ``at``
        moves into the new epoch, reaching zero again at the next boundary.
        """
        if self._previous_count == 0:
            return 0.0
        boundary = self._current_epoch * self.expiry
        offset_into_current_epoch = at - boundary
        if offset_into_current_epoch <= 0:
            return 0.0
        return max(0.0, float(self.expiry - offset_into_current_epoch))

    def weighted_count(self, at: float) -> float:
        self._roll(at)
        return (
            self._previous_count * self._previous_ttl(at) / self.expiry
            + self._current_count
        )

    def consumed(self, at: float) -> int:
        """Integer capacity held at time ``at`` (floor of the weighted count)."""
        return math.floor(self.weighted_count(at))

    def can_acquire(self, at: float, cost: int) -> bool:
        if cost <= 0 or cost > self.limit:
            return False
        return self.consumed(at) + cost <= self.limit

    def hit(self, at: float, cost: int = 1) -> bool:
        if not self.can_acquire(at, cost):
            return False
        self._roll(at)
        self._current_count += cost
        return True

    def test(self, at: float, cost: int = 1) -> bool:
        # Read-only: intentionally no mutation, mirroring the public contract.
        return self.can_acquire(at, cost)

    def stats(self, at: float) -> ReferenceStats:
        self._roll(at)
        remaining = max(0, self.limit - math.floor(self.weighted_count(at)))
        if self._previous_count == 0 and self._current_count == 0:
            return ReferenceStats(at, remaining)

        next_boundary = (self._current_epoch + 1) * self.expiry
        wait_until_free = [float(next_boundary) - at]
        if self._previous_count > 0:
            decay_step = self.expiry / self._previous_count
            wait_until_free.append(math.fmod(self._previous_ttl(at), decay_step))
        return ReferenceStats(at + min(wait_until_free), remaining)

    def clear(self) -> None:
        self._initialized = False
        self._previous_epoch = 0
        self._current_epoch = 0
        self._previous_count = 0
        self._current_count = 0


@dataclass(frozen=True)
class Event:
    """One step of a scripted scenario at a fixed virtual time."""

    at: float
    kind: str  # "hit" | "test" | "stats" | "clear"
    cost: int = 1


def expected_trace(events: list[Event], limit: int, expiry: int) -> list[object]:
    """Run a scenario through the independent oracle and hand back expectations.

    ``hit``/``test`` yield a bool, ``stats`` yields a ``ReferenceStats`` and
    ``clear`` yields ``None``. This is the single source of truth the sync and
    async production limiters are both compared against.
    """
    oracle = ReferenceSlidingWindowCounter(limit, expiry)
    trace: list[object] = []
    for event in events:
        if event.kind == "hit":
            trace.append(oracle.hit(event.at, event.cost))
        elif event.kind == "test":
            trace.append(oracle.test(event.at, event.cost))
        elif event.kind == "stats":
            trace.append(oracle.stats(event.at))
        elif event.kind == "clear":
            oracle.clear()
            trace.append(None)
        else:  # pragma: no cover - defensive, keeps the oracle honest
            raise ValueError(f"unknown event kind: {event.kind!r}")
    return trace
