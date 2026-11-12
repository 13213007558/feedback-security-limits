from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass

from deprecated.sphinx import versionadded

from limits import errors
from limits._storage_scheme import StorageRegistry
from limits.typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    P,
    R,
    cast,
)
from limits.util import LazyDependency


def _wrap_errors(
    fn: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    @functools.wraps(fn)
    async def inner(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[misc]
        instance = cast(Storage, args[0])
        try:
            return await fn(*args, **kwargs)
        except instance.base_exceptions as exc:
            if instance.wrap_exceptions:
                raise errors.StorageError(exc) from exc
            raise

    return inner


@versionadded(version="2.1")
class Storage(LazyDependency, metaclass=StorageRegistry):
    """
    Base class to extend when implementing an async storage backend.
    """

    STORAGE_SCHEME: ClassVar[list[str] | None] = None
    """The storage schemes to register against this implementation"""

    def __init_subclass__(cls, **kwargs: Any) -> None:  # type:ignore[explicit-any]
        super().__init_subclass__(**kwargs)
        for method in (
            "incr",
            "get",
            "get_expiry",
            "check",
            "reset",
            "clear",
        ):
            setattr(cls, method, _wrap_errors(getattr(cls, method)))
        super().__init_subclass__(**kwargs)

    def __init__(
        self,
        uri: str | None = None,
        wrap_exceptions: bool = False,
        **options: float | str | bool,
    ) -> None:
        """
        :param wrap_exceptions: Whether to wrap storage exceptions in
         :exc:`limits.errors.StorageError` before raising it.
        """
        super().__init__()
        self.wrap_exceptions = wrap_exceptions

    @property
    @abstractmethod
    def base_exceptions(self) -> type[Exception] | tuple[type[Exception], ...]:
        raise NotImplementedError

    @abstractmethod
    async def incr(self, key: str, expiry: int, amount: int = 1) -> int:
        """
        increments the counter for a given rate limit key

        :param key: the key to increment
        :param expiry: amount in seconds for the key to expire in
        :param amount: the number to increment by
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, key: str) -> int:
        """
        :param key: the key to get the counter value for
        """
        raise NotImplementedError

    @abstractmethod
    async def get_expiry(self, key: str) -> float:
        """
        :param key: the key to get the expiry for
        """
        raise NotImplementedError

    @abstractmethod
    async def check(self) -> bool:
        """
        check if storage is healthy
        """
        raise NotImplementedError

    @abstractmethod
    async def reset(self) -> int | None:
        """
        reset storage to clear limits
        """
        raise NotImplementedError

    @abstractmethod
    async def clear(self, key: str) -> None:
        """
        resets the rate limit key

        :param key: the key to clear rate limits for
        """
        raise NotImplementedError

    @versionadded(version="5.9")
    async def reserve(
        self, key: str, limit: int, expiry: float, amount: int = 1
    ) -> Reservation | None:
        """
        Reserve ``amount`` units of the rate limit identified by ``key``
        without committing the consumption. Implementations that support the
        two-phase reserve/commit/cancel protocol override this method and
        inherit from :class:`ReservationSupport`.

        :param key: the rate limit key to reserve capacity on
        :param limit: maximum capacity for ``key`` within ``expiry``
        :param expiry: amount in seconds until the reservation is no longer
         committable
        :param amount: the number of units to reserve
        :return: a unique :class:`Reservation` handle, or ``None`` if the
         remaining capacity is insufficient
        """
        raise NotImplementedError(
            f"Reservations are not supported by storage of type {type(self)}"
        )

    @versionadded(version="5.9")
    async def commit_reservation(self, reservation: Reservation) -> bool:
        """
        Commit a previously :meth:`reserve`d handle, turning the held
        capacity into an actual consumption.

        :param reservation: the handle returned by :meth:`reserve`
        :return: ``True`` only for the first, timely commit of an active
         reservation. Repeated commits, commits racing a cancel and commits
         arriving after the reservation expired deterministically return
         ``False`` and leave the ledgers untouched.
        """
        raise NotImplementedError(
            f"Reservations are not supported by storage of type {type(self)}"
        )

    @versionadded(version="5.9")
    async def cancel_reservation(self, reservation: Reservation) -> bool:
        """
        Release the capacity held by a reservation that will not be
        committed. Only the reservation's own held units are released; the
        committed counter is never decremented.

        :param reservation: the handle returned by :meth:`reserve`
        :return: ``True`` only for the first cancel of an active
         reservation. Repeated cancels and cancels racing a commit
         deterministically return ``False``.
        """
        raise NotImplementedError(
            f"Reservations are not supported by storage of type {type(self)}"
        )


@versionadded(version="5.9")
@dataclass(frozen=True)
class Reservation:
    """
    An opaque, unique handle representing capacity held by
    :meth:`ReservationSupport.reserve` until it is either committed via
    :meth:`ReservationSupport.commit_reservation` or released via
    :meth:`ReservationSupport.cancel_reservation`.
    """

    id: str
    key: str
    amount: int
    expiry: float
    created_at: float


@versionadded(version="5.9")
class ReservationSupport(ABC):
    """
    Abstract base class for async storages that support the optional
    two-phase reserve/commit/cancel protocol.

    A reservation first *holds* capacity so that concurrent work cannot
    consume it, then either commits the hold once the sensitive operation
    has actually started, or cancels the hold when the preparation failed
    before any business side effect took place.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:  # type: ignore[explicit-any]
        for method in (
            "reserve",
            "commit_reservation",
            "cancel_reservation",
        ):
            # Storage.__init_subclass__ already wraps these for storage
            # backends; only wrap standalone ReservationSupport subclasses.
            already_wrapped = hasattr(getattr(cls, method), "__wrapped__")
            if not already_wrapped:
                setattr(cls, method, _wrap_errors(getattr(cls, method)))
        super().__init_subclass__(**kwargs)

    @abstractmethod
    async def reserve(
        self, key: str, limit: int, expiry: float, amount: int = 1
    ) -> Reservation | None:
        raise NotImplementedError

    @abstractmethod
    async def commit_reservation(self, reservation: Reservation) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def cancel_reservation(self, reservation: Reservation) -> bool:
        raise NotImplementedError


class MovingWindowSupport(ABC):
    """
    Abstract base class for async storages that support
    the :ref:`strategies:moving window` strategy
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:  # type: ignore[explicit-any]
        for method in (
            "acquire_entry",
            "get_moving_window",
        ):
            setattr(
                cls,
                method,
                _wrap_errors(getattr(cls, method)),
            )
        super().__init_subclass__(**kwargs)

    @abstractmethod
    async def acquire_entry(
        self, key: str, limit: int, expiry: int, amount: int = 1
    ) -> bool:
        """
        :param key: rate limit key to acquire an entry in
        :param limit: amount of entries allowed
        :param expiry: expiry of the entry
        :param amount: the number of entries to acquire
        """
        raise NotImplementedError

    @abstractmethod
    async def get_moving_window(
        self, key: str, limit: int, expiry: int
    ) -> tuple[float, int]:
        """
        returns the starting point and the number of entries in the moving
        window

        :param key: rate limit key
        :param expiry: expiry of entry
        :return: (start of window, number of acquired entries)
        """
        raise NotImplementedError


class SlidingWindowCounterSupport(ABC):
    """
    Abstract base class for async storages that support
    the :ref:`strategies:sliding window counter` strategy
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:  # type: ignore[explicit-any]
        for method in (
            "acquire_sliding_window_entry",
            "get_sliding_window",
            "clear_sliding_window",
        ):
            setattr(
                cls,
                method,
                _wrap_errors(getattr(cls, method)),
            )
        super().__init_subclass__(**kwargs)

    @abstractmethod
    async def acquire_sliding_window_entry(
        self,
        key: str,
        limit: int,
        expiry: int,
        amount: int = 1,
    ) -> bool:
        """
        Acquire an entry if the weighted count of the current and previous
        windows is less than or equal to the limit

        :param key: rate limit key to acquire an entry in
        :param limit: amount of entries allowed
        :param expiry: expiry of the entry
        :param amount: the number of entries to acquire
        """
        raise NotImplementedError

    @abstractmethod
    async def get_sliding_window(
        self, key: str, expiry: int
    ) -> tuple[int, float, int, float]:
        """
        Return the previous and current window information.

        :param key: the rate limit key
        :param expiry: the rate limit expiry, needed to compute the key in some implementations
        :return: a tuple of (int, float, int, float) with the following information:
          - previous window counter
          - previous window TTL
          - current window counter
          - current window TTL
        """
        raise NotImplementedError

    @abstractmethod
    async def clear_sliding_window(self, key: str, expiry: int) -> None:
        """
        Resets the rate limit key(s) for the sliding window

        :param key: the key to clear rate limits for
        :param expiry: the rate limit expiry, needed to compute the key in some implemenations
        """
        ...
