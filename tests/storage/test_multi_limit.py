from __future__ import annotations

import threading
import time

import pytest

from limits.errors import CapabilityError
from limits.storage import (
    MemoryStorage,
    MultiLimitEntry,
    MultiLimitResult,
    merge_multi_limit_entries,
)
from limits.storage.base import Storage


def login_entries(global_limit: int = 1, cost: int = 1) -> list[MultiLimitEntry]:
    return [
        MultiLimitEntry("login/ip/10.0.0.1", 5, 60, cost),
        MultiLimitEntry("login/account/alice", 5, 60, cost),
        MultiLimitEntry("login/global", global_limit, 60, cost),
    ]


class TestMergeMultiLimitEntries:
    def test_merge_sums_weighted_cost_for_duplicate_keys(self):
        merged = merge_multi_limit_entries(
            [
                MultiLimitEntry("k", 10, 60, 2),
                MultiLimitEntry("k", 10, 60, 3),
                MultiLimitEntry("other", 4, 30, 1),
            ]
        )
        assert merged == [
            MultiLimitEntry("k", 10, 60, 5),
            MultiLimitEntry("other", 4, 30, 1),
        ]

    def test_conflicting_expiry_for_same_key_is_rejected(self):
        with pytest.raises(ValueError):
            merge_multi_limit_entries(
                [MultiLimitEntry("k", 10, 60), MultiLimitEntry("k", 10, 30)]
            )

    def test_conflicting_limit_for_same_key_is_rejected(self):
        with pytest.raises(ValueError):
            merge_multi_limit_entries(
                [MultiLimitEntry("k", 10, 60), MultiLimitEntry("k", 20, 60)]
            )

    @pytest.mark.parametrize(
        "entries",
        [
            [],
            [MultiLimitEntry("k", 0, 60)],
            [MultiLimitEntry("k", 5, 0)],
            [MultiLimitEntry("k", 5, 60, 0)],
            [MultiLimitEntry("k", -1, 60)],
            [MultiLimitEntry("k", 5, 60, True)],
            [MultiLimitEntry("", 5, 60)],
            [("not", "an", "entry")],
        ],
    )
    def test_invalid_entries_raise_value_error(self, entries):
        with pytest.raises(ValueError):
            merge_multi_limit_entries(entries)


class TestIncrMany:
    def test_all_rules_deducted_together(self):
        storage = MemoryStorage()
        result = storage.incr_many(login_entries())
        assert result == MultiLimitResult(
            True,
            {
                "login/ip/10.0.0.1": 1,
                "login/account/alice": 1,
                "login/global": 1,
            },
            (),
        )
        assert storage.get("login/global") == 1

    def test_one_rule_rejected_leaves_every_counter_untouched(self):
        storage = MemoryStorage()
        assert storage.incr_many(login_entries(global_limit=1)).allowed
        # The global dimension is exhausted; ip and account still have room.
        result = storage.incr_many(login_entries(global_limit=1))
        assert not result.allowed
        assert result.rejected == ("login/global",)
        assert result.counters == {
            "login/ip/10.0.0.1": 1,
            "login/account/alice": 1,
            "login/global": 1,
        }
        # No partial increment leaked into the first two dimensions.
        assert storage.get("login/ip/10.0.0.1") == 1
        assert storage.get("login/account/alice") == 1
        assert storage.get("login/global") == 1

    def test_several_rules_can_be_reported_as_rejected(self):
        storage = MemoryStorage()
        storage.incr_many(
            [
                MultiLimitEntry("a", 1, 60),
                MultiLimitEntry("b", 1, 60),
                MultiLimitEntry("c", 5, 60),
            ]
        )
        result = storage.incr_many(
            [
                MultiLimitEntry("a", 1, 60),
                MultiLimitEntry("b", 1, 60),
                MultiLimitEntry("c", 5, 60),
            ]
        )
        assert not result.allowed
        assert result.rejected == ("a", "b")
        assert storage.get("a") == 1
        assert storage.get("b") == 1
        assert storage.get("c") == 1

    def test_duplicate_keys_merge_their_weighted_cost(self):
        storage = MemoryStorage()
        result = storage.incr_many(
            [
                MultiLimitEntry("k", 10, 60, 2),
                MultiLimitEntry("k", 10, 60, 3),
            ]
        )
        assert result.allowed
        assert storage.get("k") == 5

    def test_merged_cost_that_exceeds_limit_is_rejected_without_deduction(self):
        storage = MemoryStorage()
        result = storage.incr_many(
            [
                MultiLimitEntry("k", 4, 60, 2),
                MultiLimitEntry("k", 4, 60, 3),
            ]
        )
        assert not result.allowed
        assert result.rejected == ("k",)
        assert storage.get("k") == 0

    def test_different_windows_use_their_own_expiry(self):
        storage = MemoryStorage()
        result = storage.incr_many(
            [
                MultiLimitEntry("short", 5, 1),
                MultiLimitEntry("long", 5, 100),
            ]
        )
        assert result.allowed
        assert storage.get_expiry("short") < storage.get_expiry("long")

    def test_conflicting_windows_reject_without_any_deduction(self):
        storage = MemoryStorage()
        with pytest.raises(ValueError):
            storage.incr_many(
                [MultiLimitEntry("k", 10, 60), MultiLimitEntry("k", 10, 30)]
            )
        assert storage.get("k") == 0

    @pytest.mark.parametrize(
        "entries",
        [
            [],
            [MultiLimitEntry("k", 0, 60)],
            [MultiLimitEntry("k", 5, 0)],
            [MultiLimitEntry("k", 5, 60, 0)],
            [MultiLimitEntry("k", 5, 60, -2)],
            [MultiLimitEntry("", 5, 60)],
        ],
    )
    def test_invalid_parameters_change_nothing(self, entries):
        storage = MemoryStorage()
        storage.incr("untouched", 60)
        with pytest.raises(ValueError):
            storage.incr_many(entries)
        assert storage.get("untouched") == 1
        assert storage.reset() == 1

    def test_window_boundary_resets_counters_for_new_window(self):
        storage = MemoryStorage()
        limit = MultiLimitEntry("boundary", 2, 1)
        assert storage.incr_many([limit]).allowed
        assert storage.incr_many([limit]).allowed
        assert not storage.incr_many([limit]).allowed
        assert storage.get("boundary") == 2
        time.sleep(1.1)
        # Expired window behaves as a fresh counter again.
        result = storage.incr_many([limit])
        assert result.allowed
        assert result.counters == {"boundary": 1}

    def test_single_rule_incr_interface_is_kept(self):
        storage = MemoryStorage()
        assert storage.incr("single", 60) == 1
        assert storage.incr("single", 60, amount=2) == 3
        assert storage.get("single") == 3

    def test_unsupported_backend_raises_capability_error(self):
        class FixedOnlyStorage(Storage):
            STORAGE_SCHEME = ["fixed-only-test://"]

            @property
            def base_exceptions(self):
                return ValueError

            def incr(self, key, expiry, amount=1):
                return 1

            def get(self, key):
                return 0

            def get_expiry(self, key):
                return time.time()

            def check(self):
                return True

            def reset(self):
                return None

            def clear(self, key):
                return None

        storage = FixedOnlyStorage()
        with pytest.raises(CapabilityError):
            storage.incr_many([MultiLimitEntry("k", 5, 60)])


class TestIncrManyConcurrency:
    def test_two_requests_contend_for_last_shared_quota_with_barrier(self):
        storage = MemoryStorage()
        # One slot left on every dimension; the global rule is the bottleneck.
        storage.incr_many(login_entries(global_limit=2))
        barrier = threading.Barrier(2)
        outcomes: dict[str, MultiLimitResult] = {}

        def worker(name: str) -> None:
            barrier.wait()
            outcomes[name] = storage.incr_many(login_entries(global_limit=2))

        threads = [
            threading.Thread(target=worker, args=("first",)),
            threading.Thread(target=worker, args=("second",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        allowed = [name for name, result in outcomes.items() if result.allowed]
        rejected = [name for name, result in outcomes.items() if not result.allowed]
        assert len(allowed) == 1
        assert len(rejected) == 1

        winner, loser = allowed[0], rejected[0]
        assert outcomes[winner].rejected == ()
        assert outcomes[loser].rejected == ("login/global",)

        # Exactly one request was accounted for on every dimension.
        assert storage.get("login/global") == 2
        assert storage.get("login/ip/10.0.0.1") == 2
        assert storage.get("login/account/alice") == 2

    def test_locks_acquired_in_consistent_order_do_not_deadlock(self):
        storage = MemoryStorage()
        barrier = threading.Barrier(2)
        finished: list[bool] = []

        def worker(keys: list[str]) -> None:
            barrier.wait()
            for _ in range(20):
                result = storage.incr_many(
                    [MultiLimitEntry(key, 100, 60) for key in keys]
                )
                assert result.allowed
            finished.append(True)

        threads = [
            threading.Thread(target=worker, args=(["a", "b", "c"],)),
            threading.Thread(target=worker, args=(["c", "b", "a"],)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert len(finished) == 2
        assert storage.get("a") == 40
        assert storage.get("b") == 40
        assert storage.get("c") == 40
