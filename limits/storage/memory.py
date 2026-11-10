from __future__ import annotations

import bisect
import contextlib
import heapq
import threading
import time
from collections import Counter
from collections.abc import Iterator
from math import floor

import limits.typing
from limits.storage.base import (
    MovingWindowSupport,
    SlidingWindowCounterSupport,
    Storage,
    TimestampedSlidingWindow,
)
from limits.typing import ClassVar


class Entry:
    def __init__(self, expiry: float) -> None:
        self.atime = time.time()
        self.expiry = self.atime + expiry


class _LockEntry:
    """
    a lock together with the number of threads currently holding
    or waiting for it, so that idle locks can be reclaimed safely
    """

    __slots__ = ("holders", "lock")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.holders = 0


class MemoryStorage(
    Storage, MovingWindowSupport, SlidingWindowCounterSupport, TimestampedSlidingWindow
):
    """
    rate limit storage using :class:`collections.Counter`
    as an in memory storage for fixed and sliding window strategies,
    and a simple list to implement moving window strategy.

    Expired counters, events and idle locks are reclaimed by a single
    background timer that processes a bounded number of keys per pass,
    scheduled by expiry time instead of scanning every known key.

    """

    STORAGE_SCHEME: ClassVar[list[str]] = ["memory"]

    EXPIRY_INTERVAL: ClassVar[float] = 0.05
    """seconds between two background cleanup passes"""

    CLEANUP_BATCH_SIZE: ClassVar[int] = 500
    """maximum number of keys examined per category in one cleanup pass"""

    def __init__(self, uri: str | None = None, wrap_exceptions: bool = False, **_: str):
        self.storage: limits.typing.Counter[str] = Counter()
        self.locks: dict[str, _LockEntry] = {}
        self.expirations: dict[str, float] = {}
        self.events: dict[str, list[Entry]] = {}
        self._expiry_index: list[tuple[float, int, str]] = []
        self._event_expiry_index: list[tuple[float, int, str]] = []
        self._heap_sequence = 0
        self._locks_guard = threading.Lock()
        self._timer_lock = threading.Lock()
        self._closed = False
        self.timer = self.__new_timer(self.EXPIRY_INTERVAL)
        self.timer.start()
        super().__init__(uri, wrap_exceptions=wrap_exceptions, **_)

    def __getstate__(self) -> dict[str, limits.typing.Any]:  # type: ignore[explicit-any]
        state = self.__dict__.copy()
        del state["timer"]
        del state["locks"]
        del state["_timer_lock"]
        del state["_locks_guard"]
        return state

    def __setstate__(self, state: dict[str, limits.typing.Any]) -> None:  # type: ignore[explicit-any]
        self.__dict__.update(state)
        self.locks = {}
        self._locks_guard = threading.Lock()
        self._timer_lock = threading.Lock()
        self._closed = False
        if not hasattr(self, "_expiry_index"):
            # state pickled before expiry indexes existed
            self._expiry_index = []
            self._event_expiry_index = []
            self._heap_sequence = 0
            for key, expiry in self.expirations.items():
                self._push_expiry(key, expiry)
            for key, events in self.events.items():
                if events:
                    self._push_event_expiry(key, events[-1].expiry)
        self.timer = self.__new_timer(self.EXPIRY_INTERVAL)
        self.timer.start()

    def __new_timer(self, interval: float) -> threading.Timer:
        timer = threading.Timer(interval, self.__expire_and_reschedule)
        timer.daemon = True
        return timer

    @contextlib.contextmanager
    def _locked(self, key: str) -> Iterator[None]:
        """
        serialize access to the state of ``key`` under a per key lock.

        the lock entry is reference counted so that the cleanup can only
        reclaim it once no thread holds or waits for it any more.
        """
        with self._locks_guard:
            entry = self.locks.get(key)
            if entry is None:
                entry = self.locks[key] = _LockEntry()
            entry.holders += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._locks_guard:
                entry.holders -= 1

    def _reclaim_lock(self, key: str) -> None:
        with self._locks_guard:
            entry = self.locks.get(key)
            if entry is not None and not entry.holders:
                del self.locks[key]

    def _push_expiry(self, key: str, expiry: float) -> None:
        self._heap_sequence += 1
        heapq.heappush(self._expiry_index, (expiry, self._heap_sequence, key))

    def _push_event_expiry(self, key: str, expiry: float) -> None:
        self._heap_sequence += 1
        heapq.heappush(self._event_expiry_index, (expiry, self._heap_sequence, key))

    def _expire_key(self, key: str) -> None:
        # lazily evict an expired counter. caller must hold the lock of ``key``
        if self.expirations.get(key, 0) <= time.time():
            self.storage.pop(key, None)
            self.expirations.pop(key, None)

    def _expire_events(
        self, now: float | None = None, max_keys: int | None = None
    ) -> int:
        """
        reclaim expired counters, expired events and idle locks.

        at most ``max_keys`` keys are examined per category (fixed window
        counters and moving window events) so a single pass does bounded
        work no matter how many keys were created in the past.

        :param now: timestamp to expire against, defaults to the current time
        :param max_keys: per category budget, defaults to
         :attr:`CLEANUP_BATCH_SIZE`
        :return: the number of keys whose state was removed
        """
        now = time.time() if now is None else now
        max_keys = self.CLEANUP_BATCH_SIZE if max_keys is None else max_keys
        reclaimed = 0

        budget = max_keys
        while budget > 0 and self._expiry_index and self._expiry_index[0][0] <= now:
            expiry, _, key = heapq.heappop(self._expiry_index)
            budget -= 1
            with self._locked(key):
                if self.expirations.get(key) == expiry:
                    self.storage.pop(key, None)
                    self.expirations.pop(key, None)
                    reclaimed += 1
            if key not in self.expirations and key not in self.events:
                self._reclaim_lock(key)

        budget = max_keys
        while (
            budget > 0
            and self._event_expiry_index
            and self._event_expiry_index[0][0] <= now
        ):
            _, _, key = heapq.heappop(self._event_expiry_index)
            budget -= 1
            with self._locked(key):
                events = self.events.get(key)
                if events:
                    oldest = bisect.bisect_left(
                        events, -now, key=lambda event: -event.expiry
                    )
                    events = self.events[key] = events[:oldest]
                if events:
                    self._push_event_expiry(key, events[-1].expiry)
                else:
                    self.events.pop(key, None)
                    reclaimed += 1
            if key not in self.expirations and key not in self.events:
                self._reclaim_lock(key)

        return reclaimed

    def __expire_and_reschedule(self) -> None:
        self._expire_events()
        if not (self._expiry_index or self._event_expiry_index):
            # nothing left to expire: go dormant until the next request
            return
        now = time.time()
        backlog = (self._expiry_index and self._expiry_index[0][0] <= now) or (
            self._event_expiry_index and self._event_expiry_index[0][0] <= now
        )
        with self._timer_lock:
            if self._closed:
                return
            timer = self.__new_timer(0 if backlog else self.EXPIRY_INTERVAL)
            self.timer = timer
            timer.start()

    def __schedule_expiry(self) -> None:
        with self._timer_lock:
            if self._closed or self.timer.is_alive():
                return
            timer = self.__new_timer(self.EXPIRY_INTERVAL)
            self.timer = timer
            timer.start()

    def close(self) -> None:
        """
        stop the background cleanup thread.

        the storage remains usable afterwards: expired state is still
        evicted lazily on access, only the background reclamation stops.
        """
        with self._timer_lock:
            self._closed = True
            timer = self.timer
            timer.cancel()
        if timer.is_alive() and timer is not threading.current_thread():
            timer.join(timeout=1)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    @property
    def base_exceptions(
        self,
    ) -> type[Exception] | tuple[type[Exception], ...]:  # pragma: no cover
        return ValueError

    def incr(self, key: str, expiry: float, amount: int = 1) -> int:
        """
        increments the counter for a given rate limit key

        :param key: the key to increment
        :param expiry: amount in seconds for the key to expire in
        :param amount: the number to increment by
        """
        self.__schedule_expiry()
        with self._locked(key):
            self._expire_key(key)
            self.storage[key] += amount
            if self.storage[key] == amount:
                expiration = time.time() + expiry
                self.expirations[key] = expiration
                self._push_expiry(key, expiration)
            return self.storage[key]

    def decr(self, key: str, amount: int = 1) -> int:
        """
        decrements the counter for a given rate limit key

        :param key: the key to decrement
        :param amount: the number to decrement by
        """
        self.__schedule_expiry()
        with self._locked(key):
            self._expire_key(key)
            value = max(self.storage.get(key, 0) - amount, 0)
            if value:
                self.storage[key] = value
            else:
                self.storage.pop(key, None)
        if not value and key not in self.expirations and key not in self.events:
            self._reclaim_lock(key)
        return value

    def get(self, key: str) -> int:
        """
        :param key: the key to get the counter value for
        """
        with self._locked(key):
            self._expire_key(key)
            value = self.storage.get(key, 0)
        if not value and key not in self.expirations and key not in self.events:
            self._reclaim_lock(key)
        return value

    def clear(self, key: str) -> None:
        """
        :param key: the key to clear rate limits for
        """
        with self._locked(key):
            self.storage.pop(key, None)
            self.expirations.pop(key, None)
            self.events.pop(key, None)
        self._reclaim_lock(key)

    def acquire_entry(self, key: str, limit: int, expiry: int, amount: int = 1) -> bool:
        """
        :param key: rate limit key to acquire an entry in
        :param limit: amount of entries allowed
        :param expiry: expiry of the entry
        :param amount: the number of entries to acquire
        """
        if amount > limit:
            return False

        self.__schedule_expiry()
        with self._locked(key):
            events = self.events.setdefault(key, [])
            timestamp = time.time()
            try:
                entry = events[limit - amount]
            except IndexError:
                entry = None

            if entry and entry.atime >= timestamp - expiry:
                return False
            else:
                events[:0] = [Entry(expiry)] * amount
                if len(events) == amount:
                    self._push_event_expiry(key, events[-1].expiry)
                return True

    def get_expiry(self, key: str) -> float:
        """
        :param key: the key to get the expiry for
        """

        return self.expirations.get(key, time.time())

    def get_moving_window(self, key: str, limit: int, expiry: int) -> tuple[float, int]:
        """
        returns the starting point and the number of entries in the moving
        window

        :param key: rate limit key
        :param expiry: expiry of entry
        :return: (start of window, number of acquired entries)
        """
        timestamp = time.time()
        if events := self.events.get(key, []):
            oldest = bisect.bisect_left(
                events, -(timestamp - expiry), key=lambda entry: -entry.atime
            )
            return events[oldest - 1].atime, oldest
        return timestamp, 0

    def acquire_sliding_window_entry(
        self,
        key: str,
        limit: int,
        expiry: int,
        amount: int = 1,
    ) -> bool:
        if amount > limit:
            return False
        now = time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        (
            previous_count,
            previous_ttl,
            current_count,
            _,
        ) = self._get_sliding_window_info(previous_key, current_key, expiry, now)
        weighted_count = previous_count * previous_ttl / expiry + current_count
        if floor(weighted_count) + amount > limit:
            return False
        else:
            # Hit, increase the current counter.
            # If the counter doesn't exist yet, set twice the theorical expiry.
            current_count = self.incr(current_key, 2 * expiry, amount=amount)
            weighted_count = previous_count * previous_ttl / expiry + current_count
            if floor(weighted_count) > limit:
                # Another hit won the race condition: revert the incrementation and refuse this hit
                # Limitation: during high concurrency at the end of the window,
                # the counter is shifted and cannot be decremented, so less requests than expected are allowed.
                self.decr(current_key, amount)
                return False
            return True

    def _get_sliding_window_info(
        self,
        previous_key: str,
        current_key: str,
        expiry: int,
        now: float,
    ) -> tuple[int, float, int, float]:
        previous_count = self.get(previous_key)
        current_count = self.get(current_key)
        if previous_count == 0:
            previous_ttl = float(0)
        else:
            previous_ttl = (1 - (((now - expiry) / expiry) % 1)) * expiry
        current_ttl = (1 - ((now / expiry) % 1)) * expiry + expiry
        return previous_count, previous_ttl, current_count, current_ttl

    def get_sliding_window(
        self, key: str, expiry: int
    ) -> tuple[int, float, int, float]:
        now = time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        return self._get_sliding_window_info(previous_key, current_key, expiry, now)

    def clear_sliding_window(self, key: str, expiry: int) -> None:
        now = time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        self.clear(previous_key)
        self.clear(current_key)

    def check(self) -> bool:
        """
        check if storage is healthy
        """

        return True

    def reset(self) -> int | None:
        num_items = max(len(self.storage), len(self.events))
        self.storage.clear()
        self.expirations.clear()
        self.events.clear()
        self.locks.clear()
        self._expiry_index.clear()
        self._event_expiry_index.clear()
        return num_items
