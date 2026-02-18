"""Response cache for NeuralBridge.

Deduplicates identical conversation queries by caching successful agent
responses for a configurable TTL.  Enabled by default; can be toggled and
purged via the integration's Advanced Settings options flow.

Cache keys are SHA-256 hashes of the lower-cased, stripped input text so
capitalisation and leading/trailing whitespace differences are treated as
the same query.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass


@dataclass
class CachedResponse:
    """A single cached agent response."""

    response_text: str
    agent_name: str
    cached_at: float
    expires_at: float


class ResponseCache:
    """Short-TTL response cache for deduplicated conversation answers.

    Instances are stored in ``hass.data`` so the options flow handler can
    call :meth:`invalidate` and :meth:`configure` without restarting the
    integration.
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: int = 300,
        max_size: int = 200,
    ) -> None:
        """Initialise the response cache.

        Args:
            enabled: Whether caching is active (default True).
            ttl_seconds: How long a cached response is valid (default 300 s).
            max_size: Maximum number of cached entries before oldest are
                evicted (default 200).
        """
        self._enabled = enabled
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cache: dict[str, CachedResponse] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> str | None:
        """Return a cached response for *text*, or ``None`` on a miss.

        Expired entries are removed on access.

        Args:
            text: The raw user input text.

        Returns:
            Cached response string, or ``None`` if disabled/not found/expired.
        """
        if not self._enabled:
            return None

        key = self._make_key(text)
        entry = self._cache.get(key)
        if entry is None:
            return None

        if time.monotonic() > entry.expires_at:
            del self._cache[key]
            return None

        return entry.response_text

    def store(self, text: str, response: str, agent_name: str) -> None:
        """Cache a successful agent response.

        Triggers internal cleanup (expire old entries, enforce max_size)
        before inserting the new entry.

        Args:
            text: The raw user input text (used to derive the cache key).
            response: The agent response to cache.
            agent_name: Human-readable agent name (stored for diagnostics).
        """
        if not self._enabled:
            return

        self._cleanup()
        key = self._make_key(text)
        now = time.monotonic()
        self._cache[key] = CachedResponse(
            response_text=response,
            agent_name=agent_name,
            cached_at=now,
            expires_at=now + self._ttl,
        )

    def invalidate(self) -> int:
        """Purge all cached entries.

        Returns:
            The number of entries that were removed.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def configure(self, enabled: bool, ttl_seconds: int) -> None:
        """Update cache configuration without restarting.

        Disabling the cache also purges all existing entries.

        Args:
            enabled: Whether caching should be active.
            ttl_seconds: New TTL for subsequent cached entries.
        """
        self._enabled = enabled
        self._ttl = ttl_seconds
        if not enabled:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return whether the cache is currently enabled."""
        return self._enabled

    @property
    def ttl(self) -> int:
        """Return the current TTL in seconds."""
        return self._ttl

    @property
    def size(self) -> int:
        """Return the current number of cached entries (including expired)."""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, text: str) -> str:
        """Derive a cache key from the input text.

        Keys are SHA-256 hashes of the lower-cased, stripped text so minor
        differences in capitalisation or whitespace map to the same entry.

        Args:
            text: Raw input text.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        normalised = text.lower().strip()
        return hashlib.sha256(normalised.encode()).hexdigest()

    def _cleanup(self) -> None:
        """Remove expired entries and enforce max_size by evicting oldest."""
        now = time.monotonic()
        self._cache = {k: v for k, v in self._cache.items() if v.expires_at > now}

        if len(self._cache) >= self._max_size:
            # Evict oldest entries until we are one below max_size
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].cached_at)
            evict_count = len(self._cache) - self._max_size + 1
            for key, _ in sorted_entries[:evict_count]:
                del self._cache[key]
