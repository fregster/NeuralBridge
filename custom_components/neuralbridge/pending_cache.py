"""Generic TTL-bounded pending-result store."""

from __future__ import annotations

import time
from typing import Generic, TypeVar

V = TypeVar("V")


class PendingCache(Generic[V]):
    """Async-safe TTL-bounded cache for pending conversation state.

    Stores values by conversation_id with a configurable expiry.
    Entries are evicted lazily on access and proactively on :meth:`store`.

    Not thread-safe; designed for single-threaded asyncio use only.
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 300,
        key_prefix: str = "p",
    ) -> None:
        """Initialise the pending cache.

        Args:
            max_size: Maximum concurrent pending entries.
            ttl_seconds: Time-to-live in seconds for each entry.
            key_prefix: Prefix applied to all internal dict keys (avoids
                namespace collisions if multiple caches share storage).
        """
        self._cache: dict[str, tuple[V, float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def _key(self, conversation_id: str) -> str:
        """Build the internal cache key for *conversation_id*."""
        return f"{self._prefix}:{conversation_id}"

    async def store(self, conversation_id: str, value: V) -> None:
        """Store *value* keyed by *conversation_id*, replacing any prior entry.

        Args:
            conversation_id: Unique conversation identifier.
            value: The value to cache.
        """
        self._cache[self._key(conversation_id)] = (value, time.time() + self._ttl)
        await self._cleanup()

    async def get(self, conversation_id: str) -> V | None:
        """Return the stored value, or ``None`` if absent or expired.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            The stored value, or ``None`` if not found or expired.
        """
        key = self._key(conversation_id)
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.time() >= expiry:
            del self._cache[key]
            return None
        return value

    async def clear(self, conversation_id: str) -> None:
        """Remove any pending entry for *conversation_id*.

        Args:
            conversation_id: Unique conversation identifier.
        """
        self._cache.pop(self._key(conversation_id), None)

    async def _cleanup(self) -> None:
        """Evict expired entries; trim to max_size by oldest expiry if needed."""
        now = time.time()
        for key in [k for k, (_, exp) in self._cache.items() if now >= exp]:
            del self._cache[key]
        if len(self._cache) > self._max_size:
            oldest = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in oldest[: len(self._cache) - self._max_size]:
                del self._cache[key]
