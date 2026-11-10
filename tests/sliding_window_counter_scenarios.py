"""Scripted scenarios for the sliding window counter boundary tests.

The same event sequences are replayed against the synchronous and the
asynchronous in-memory storages, each exclusively through their public
``hit`` / ``test`` / ``get_window_stats`` / ``clear`` interfaces. Every event
carries its own virtual timestamp, so no real waiting on a window is required.

Two kinds of scenarios are provided:

* short sequences with the expected numbers worked out by hand and annotated
  inline (the reference model is still run, but a human can re-derive each
  line with pencil and paper);
* a longer pseudo-random sequence generated from a fixed seed whose expected
  outputs come only from the independent oracle in
  :mod:`tests.sliding_window_counter_reference`.
"""

from __future__ import annotations

import random

from tests.sliding_window_counter_reference import Event

# Window length and capacity used by every scenario. Epoch boundaries therefore
# land exactly on ..., 990, 1000, 1010, 1020, ... with VIRTUAL_EPOCH_START being
# itself a boundary, which makes "exactly on the boundary" easy to express.
EXPIRY = 10
LIMIT = 10
VIRTUAL_EPOCH_START = 1000.0
BOUNDARY = VIRTUAL_EPOCH_START + EXPIRY  # 1010.0: the boundary under test
NEXT_BOUNDARY = BOUNDARY + EXPIRY  # 1020.0


def boundary_hand_sequences() -> list[tuple[str, list[Event]]]:
    """Hand-computed sequences straddling the 1010.0 window boundary.

    Notation: P = previous epoch weighted contribution, C = current epoch
    integer count, w = P*ttl/W + C, consumed = floor(w). Capacity L = 10.

    Boundary instants (t = 1010.0, 1020.0) are only *observed* through
    ``stats``/``test``: an admitted write exactly on a boundary would make the
    two models diverge in stored state for the rest of the window, because the
    fixed-epoch memory storage keeps the previous bucket fully weighted on that
    single instant while the half-open oracle gives it zero weight. The
    admission consequence of that convention is covered in isolation by the
    half-open mutant test. Every actual ``hit`` below happens strictly inside an
    epoch, where the two models are mathematically identical.
    """
    return [
        (
            "mixed_costs_before_on_and_after_boundary",
            [
                # --- previous epoch [1000, 1010) ---
                # t=1005: empty window, cost 3 -> admitted; C=3, remaining 7
                Event(1005.0, "hit", 3),
                # t=1005: read-only probe for cost 7 must not mutate anything
                Event(1005.0, "test", 7),
                # t=1005: 3 + 8 = 11 > 10 -> refused, counters stay at 3
                Event(1005.0, "hit", 8),
                Event(1005.0, "stats"),
                # t=1008: 3 + 6 = 9 <= 10 -> admitted; C=9, remaining 1
                Event(1008.0, "hit", 6),
                Event(1008.0, "stats"),
                # t=1008: floor(9) + 2 = 11 -> refused, still 9 consumed
                Event(1008.0, "test", 2),
                Event(1008.0, "hit", 2),
                # --- exactly on the boundary 1010.0 (observed, never written) ---
                # Half-open model: previous bucket weight 0 -> remaining 10.
                # Memory storage: previous bucket still fully weighted ->
                # remaining 1. Both numbers are asserted by the replay harness.
                Event(1010.0, "stats"),
                # A probe at the boundary differs the same way; it must not
                # mutate either model.
                Event(1010.0, "test", 1),
                # --- just after the boundary, previous weight is decaying ---
                # t=1012.5: previous ttl = 7.5, w = 9*7.5/10 + 0 = 6.75,
                # floor = 6 -> remaining 4; cost 4 admitted, C becomes 4.
                Event(1012.5, "test", 4),
                Event(1012.5, "hit", 4),
                Event(1012.5, "stats"),
                # t=1017: ttl = 3, w = 9*3/10 + 4 = 2.7 + 4 = 6.7, floor = 6
                # remaining 4; cost 4 -> 6 + 4 = 10 admitted; C becomes 8.
                Event(1017.0, "hit", 4),
                Event(1017.0, "stats"),
                # cost 5 now: floor(2.7 + 8) = 10, +5 > 10 -> refused.
                Event(1017.0, "hit", 5),
                # --- next boundary 1020.0: only observation again ---
                Event(1020.0, "stats"),
                # The first write strictly inside the new epoch sees the old
                # buckets fully gone and gets the whole capacity back.
                Event(1020.5, "hit", 9),
                Event(1020.5, "stats"),
            ],
        ),
        (
            "boundary_ownership_half_open_epoch",
            [
                # Fill the previous epoch to the limit in two weighted hits.
                Event(1002.0, "hit", 5),
                Event(1007.0, "hit", 5),
                Event(1009.9, "stats"),
                # One nanosecond before the boundary the previous window still
                # owns the request: w ~= 9.9999..., floor = 9, no room left.
                Event(1009.999999999, "test", 1),
                # Exactly on 1010.0: half-open model reports a fresh window
                # (probe True), memory storage reports it saturated (probe
                # False). The probe mutates nothing in either case.
                Event(1010.0, "test", 1),
                Event(1010.0, "stats"),
                # Immediately after, both models agree again: the previous full
                # window still weighs 9.9/10, so the unit request is refused.
                Event(1010.1, "test", 1),
                Event(1010.1, "stats"),
                # Halfway through the new epoch the decayed weight is 5, so a
                # cost-5 hit exactly uses the freed capacity.
                Event(1015.0, "hit", 5),
                Event(1015.0, "stats"),
            ],
        ),
        (
            "overlimit_cost_empty_window_and_clear",
            [
                # Empty window: a cost above capacity is rejected without
                # reserving anything, so the following unit hit must succeed.
                Event(1001.0, "hit", LIMIT + 1),
                Event(1001.0, "stats"),
                Event(1001.0, "hit", 1),
                Event(1001.0, "test", LIMIT),
                # Saturate and confirm rejected hits do not inflate the count.
                Event(1001.5, "hit", LIMIT - 1),
                Event(1001.6, "hit", 1),
                Event(1001.7, "hit", 1),  # refused: already at floor 10
                Event(1001.8, "stats"),
                # Wipe the key; it must behave like a brand new window again.
                Event(1002.0, "clear"),
                Event(1002.0, "stats"),
                Event(1002.0, "hit", LIMIT),
                Event(1002.1, "stats"),
                Event(1002.2, "hit", 1),  # refused, still 10 consumed
                # Clearing at the boundary then reusing the key works as well.
                Event(1010.0, "clear"),
                Event(1010.0, "stats"),
                Event(1010.4, "hit", 2),
                Event(1010.4, "stats"),
            ],
        ),
    ]


def seeded_random_events(seed: int = 20261110) -> list[Event]:
    """Fixed-seed multi-window sequence mixing every public operation.

    The sequence deliberately crosses several boundaries (1010, 1020, 1030) and
    includes timestamps landing exactly on boundaries, fractional offsets and
    costs that occasionally exceed the remaining capacity or even the limit.

    Boundary instants are only observed (``stats``/``test``). Any randomly
    generated write that would land exactly on a boundary is nudged a small,
    fixed offset into the new epoch, so the half-open oracle and the fixed-epoch
    memory storage never diverge in stored state during the replay. Boundary
    *admission* behaviour is exercised separately by the mutant tests.
    """
    rng = random.Random(seed)
    events: list[Event] = []
    # Deterministic, monotonically non-decreasing virtual timestamps.
    clock = VIRTUAL_EPOCH_START + 1.0
    for step in range(120):
        clock += rng.choice([0.0, 0.25, 0.5, 1.0, 2.5, 5.0, 9.75])
        at = round(clock, 4)
        kind = rng.choices(
            ["hit", "test", "stats", "clear"], weights=[55, 20, 20, 5], k=1
        )[0]
        cost = rng.choice([1, 1, 1, 2, 3, 5, 8, LIMIT + 1])
        if kind == "hit" and at % EXPIRY == 0:
            at = round(at + 0.37, 4)  # keep writes strictly inside an epoch
        events.append(Event(at, kind, cost))
    # Guarantee exact-boundary observation and over-limit coverage regardless
    # of the random draw. None of these mutate state at the boundary.
    events.extend(
        [
            Event(1010.0, "stats"),
            Event(1010.0, "test", 1),
            Event(1020.0, "test", LIMIT + 1),
            Event(1030.0, "clear"),
            Event(1030.0, "stats"),
            Event(1030.2, "hit", 2),
        ]
    )
    events.sort(key=lambda event: event.at)
    return events


def all_scenarios() -> list[tuple[str, list[Event]]]:
    scenarios = boundary_hand_sequences()
    scenarios.append(("seeded_random_sequence", seeded_random_events()))
    return scenarios
