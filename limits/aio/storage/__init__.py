"""
Implementations of storage backends to be used with
:class:`limits.aio.strategies.RateLimiter` strategies
"""

from __future__ import annotations

from limits.storage.base import (
    MultiLimitEntry,
    MultiLimitResult,
    merge_multi_limit_entries,
)

from .base import MovingWindowSupport, SlidingWindowCounterSupport, Storage
from .memcached import MemcachedStorage
from .memory import MemoryStorage
from .mongodb import MongoDBStorage
from .redis import RedisClusterStorage, RedisSentinelStorage, RedisStorage

__all__ = [
    "MemcachedStorage",
    "MemoryStorage",
    "MongoDBStorage",
    "MovingWindowSupport",
    "MultiLimitEntry",
    "MultiLimitResult",
    "RedisClusterStorage",
    "RedisSentinelStorage",
    "RedisStorage",
    "SlidingWindowCounterSupport",
    "Storage",
    "merge_multi_limit_entries",
]
