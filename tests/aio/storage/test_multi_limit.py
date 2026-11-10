from __future__ import annotations

import asyncio
import time

import pytest

from limits.aio.storage import MemoryStorage, MultiLimitEntry, MultiLimitResult
from limits.aio.storage.base import Storage
from limits.errors import CapabilityError


def login_entries(global_limit: int = 1, cost: int = 1) -> list[MultiLimitEntry]:
    return [
        MultiLimitEntry("login/ip/10.0.0.1", 5, 60, cost),
        MultiLimitEntry("login/account/alice", 5, 60, cost),
        MultiLimitEntry("login/global", global_limit, 60, cost),
    ]


class TestIncrMany:
    async def test_all_rules_deducted_together(self):
        storage = MemoryStorage()
        result = await storage.incr_many(login_entries())
        assert result.allowed
        assert result.counters == {
            "login/ip/10.0.0.1": 1,
            "login/account/alice": 1,
            "login/global": 1,
        }

    async def test_one_rule_rejected_leaves_every_counter_untouched(self):
        storage = MemoryStorage()
        assert (await storage.incr_many(login_entries(global_limit=1))).allowed
        result = await storage.incr_many(login_entries(global_limit=1))
        assert not result.allowed
        assert result.rejected == ("login/global",)
        assert await storage.get("login/ip/10.0.0.1") == 1
        assert await storage.get("login/account/alice") == 1
        assert await storage.get("login/global") == 1

    async def test_several_rules_can_be_reported_as_rejected(self):
        storage = MemoryStorage()
        entries = [
            MultiLimitEntry("a", 1, 60),
            MultiLimitEntry("b", 1, 60),
            MultiLimitEntry("c", 5, 60),
        ]
        assert (await storage.incr_many(entries)).allowed
        result = await storage.incr_many(entries)
        assert not result.allowed
        assert result.rejected == ("a", "b")
        assert await storage.get("c") == 1

    async def test_duplicate_keys_merge_their_weighted_cost(self):
        storage = MemoryStorage()
        result = await storage.incr_many(
            [
                MultiLimitEntry("k", 10, 60, 2),
                MultiLimitEntry("k", 10, 60, 3),
            ]
        )
        assert result.allowed
        assert await storage.get("k") == 5

    async def test_merged_cost_that_exceeds_limit_is_rejected_without_deduction(self):
        storage = MemoryStorage()
        result = await storage.incr_many(
            [
                MultiLimitEntry("k", 4, 60, 2),
                MultiLimitEntry("k", 4, 60, 3),
            ]
        )
        assert not result.allowed
        assert result.rejected == ("k",)
        assert await storage.get("k") == 0

    async def test_different_windows_use_their_own_expiry(self):
        storage = MemoryStorage()
        result = await storage.incr_many(
            [MultiLimitEntry("short", 5, 1), MultiLimitEntry("long", 5, 100)]
        )
        assert result.allowed
        assert await storage.get_expiry("short") < await storage.get_expiry("long")

    async def test_conflicting_windows_reject_without_any_deduction(self):
        storage = MemoryStorage()
        with pytest.raises(ValueError):
            await storage.incr_many(
                [MultiLimitEntry("k", 10, 60), MultiLimitEntry("k", 10, 30)]
            )
        assert await storage.get("k") == 0

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
    async def test_invalid_parameters_change_nothing(self, entries):
        storage = MemoryStorage()
        await storage.incr("untouched", 60)
        with pytest.raises(ValueError):
            await storage.incr_many(entries)
        assert await storage.get("untouched") == 1
        assert await storage.reset() == 1

    async def test_window_boundary_resets_counters_for_new_window(self):
        storage = MemoryStorage()
        limit = MultiLimitEntry("boundary", 2, 1)
        assert (await storage.incr_many([limit])).allowed
        assert (await storage.incr_many([limit])).allowed
        assert not (await storage.incr_many([limit])).allowed
        assert await storage.get("boundary") == 2
        await asyncio.sleep(1.1)
        result = await storage.incr_many([limit])
        assert result.allowed
        assert result.counters == {"boundary": 1}

    async def test_single_rule_incr_interface_is_kept(self):
        storage = MemoryStorage()
        assert await storage.incr("single", 60) == 1
        assert await storage.incr("single", 60, amount=2) == 3
        assert await storage.get("single") == 3

    async def test_unsupported_backend_raises_capability_error(self):
        class FixedOnlyStorage(Storage):
            STORAGE_SCHEME = ["async+fixed-only-test://"]

            @property
            def base_exceptions(self):
                return ValueError

            async def incr(self, key, expiry, amount=1):
                return 1

            async def get(self, key):
                return 0

            async def get_expiry(self, key):
                return time.time()

            async def check(self):
                return True

            async def reset(self):
                return None

            async def clear(self, key):
                return None

        storage = FixedOnlyStorage()
        with pytest.raises(CapabilityError):
            await storage.incr_many([MultiLimitEntry("k", 5, 60)])


class TestIncrManyConcurrency:
    async def test_two_requests_contend_for_last_shared_quota_with_barrier(self):
        storage = MemoryStorage()
        assert (await storage.incr_many(login_entries(global_limit=2))).allowed
        barrier = asyncio.Barrier(2)
        outcomes: dict[str, MultiLimitResult] = {}

        async def worker(name: str) -> None:
            async with barrier:
                outcomes[name] = await storage.incr_many(
                    login_entries(global_limit=2)
                )

        await asyncio.gather(worker("first"), worker("second"))

        allowed = [name for name, result in outcomes.items() if result.allowed]
        rejected = [name for name, result in outcomes.items() if not result.allowed]
        assert len(allowed) == 1
        assert len(rejected) == 1
        assert outcomes[rejected[0]].rejected == ("login/global",)

        assert await storage.get("login/global") == 2
        assert await storage.get("login/ip/10.0.0.1") == 2
        assert await storage.get("login/account/alice") == 2

    async def test_locks_acquired_in_consistent_order_do_not_deadlock(self):
        storage = MemoryStorage()
        barrier = asyncio.Barrier(2)

        async def worker(keys: list[str]) -> None:
            async with barrier:
                for _ in range(20):
                    result = await storage.incr_many(
                        [MultiLimitEntry(key, 100, 60) for key in keys]
                    )
                    assert result.allowed

        await asyncio.wait_for(
            asyncio.gather(
                worker(["a", "b", "c"]),
                worker(["c", "b", "a"]),
            ),
            timeout=10,
        )
        assert await storage.get("a") == 40
        assert await storage.get("b") == 40
        assert await storage.get("c") == 40
