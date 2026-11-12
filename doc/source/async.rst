=============
Async Support
=============

The namespace ``limits.aio``  mirrors ``limits.storage`` and ``limits.strategies``
with async variants.

The following async storage backends are implemented:

- In-Memory
- Redis (via `coredis <https://coredis.readthedocs.org>`_
  or `redis-py <https://redis-py.readthedocs.io>`_. Refer to
  :paramref:`limits.aio.storage.RedisStorage.implementation` for
  details on selecting the dependency)
- Memcached (via `memcachio <https://memcachio.readthedocs.org>`_)
- MongoDB (via `motor <https://motor.readthedocs.org>`_)

Quick start
===========

This example demonstrates the subtle differences in the ``limits.aio`` namespace:

.. code::

   from limits import parse
   from limits.storage import storage_from_string
   from limits.aio.strategies import MovingWindowRateLimiter

   redis = storage_from_string("async+redis://localhost:6379")

   moving_window = MovingWindowRateLimiter(redis)
   one_per_minute = parse("1/minute")

   async def hit():
      return await moving_window.hit(one_per_minute, "test_namespace", "foo")


Refer to :ref:`api:async storage` for more implementation details of the async
storage backends, and :ref:`api:async strategies` for the async rate limit strategies API.

Two-phase reservations
======================

Some operations need to claim quota *before* doing preparation work and only
consume it once the sensitive operation actually starts. Storages that
implement :class:`~limits.aio.storage.ReservationSupport` (currently the
async in-memory storage) expose an optional
reserve / commit / cancel protocol:

- :meth:`~limits.aio.storage.ReservationSupport.reserve` holds capacity and
  returns a unique :class:`~limits.aio.storage.Reservation` handle, or
  ``None`` when the limit is exhausted. A hold is counted against the limit
  just like a committed hit.
- :meth:`~limits.aio.storage.ReservationSupport.commit_reservation` turns the
  hold into an actual consumption.
- :meth:`~limits.aio.storage.ReservationSupport.cancel_reservation` releases
  the hold. It only ever removes this handle's own hold; it never decrements
  the committed counter, so a reservation cannot manufacture a refund for a
  later rate-limit window.

The protocol is deterministic under concurrency: a handle commits or cancels
exactly once. Repeated commits, repeated cancels, and a commit racing a cancel
all have a single winning call; the losing call returns ``False`` without
touching the ledgers. A reservation that is not committed before its expiry
can no longer be committed and its hold is released automatically.

Storages without reservation support reject the protocol explicitly by
raising :class:`NotImplementedError` from the default
:class:`~limits.aio.storage.Storage` methods, instead of silently accepting a
reservation they cannot honour.

.. code:: python

   from limits.aio.storage import MemoryStorage, ReservationSupport

   storage = MemoryStorage()
   assert isinstance(storage, ReservationSupport)

   async def charge_customer(key: str) -> None:
       reservation = await storage.reserve(key, limit=100, expiry=60)
       if reservation is None:
           raise RuntimeError("rate limit exhausted")
       try:
           # Preparation only: no business side effect has happened yet,
           # so a failure here is safe to cancel.
           await prepare_payment()
       except Exception:
           await storage.cancel_reservation(reservation)
           raise

       # From this point on the operation's side effects are in flight.
       await start_irreversible_side_effect()
       # Commit even if a later step raises: the request really happened,
       # so refunding the quota by cancelling would be incorrect.
       await storage.commit_reservation(reservation)

       await send_receipt()

Cancellation is for work that never started. Once an external side effect
(a charge, an email, a message on a queue) has been performed, the request has
consumed its quota regardless of any exception raised afterwards, and the
reservation must be committed rather than canceled. Treating arbitrary
exceptions as "the operation did not happen" leaks quota and lets callers
exceed the configured limit.

The regular ``hit`` flow is unchanged and stays atomic: it consumes
immediately and never creates a reservation.

