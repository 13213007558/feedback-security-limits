"""
errors and exceptions
"""

from __future__ import annotations


class ConfigurationError(Exception):
    """
    Error raised when a configuration problem is encountered
    """


class ConcurrentUpdateError(Exception):
    """
    Error raised when an update to limit fails due to concurrent
    updates
    """

    def __init__(self, key: str, attempts: int) -> None:
        super().__init__(f"Unable to update {key} after {attempts} retries")


class StorageError(Exception):
    """
    Error raised when an error is encountered in a storage
    """

    def __init__(self, storage_error: Exception) -> None:
        self.storage_error = storage_error


class CapabilityError(NotImplementedError):
    """
    Error raised when a storage backend does not support an operation
    (for example an atomic, all-or-nothing increment of several keys)
    and cannot emulate it without losing the required semantics.
    """

    def __init__(self, storage: object, operation: str) -> None:
        self.storage = storage
        self.operation = operation
        super().__init__(
            f"storage {storage.__class__.__name__} does not support "
            f"the operation '{operation}'"
        )
