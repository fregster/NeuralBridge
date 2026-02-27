"""Unit tests for the generic PendingCache[V] class."""

from __future__ import annotations

import asyncio
import time

import pytest

from custom_components.neuralbridge.pending_cache import PendingCache

pytestmark = pytest.mark.asyncio


class TestPendingCacheInit:
    """Test PendingCache initialisation."""

    def test_default_init(self):
        """Default constructor sets expected attributes."""
        cache: PendingCache[str] = PendingCache()
        assert cache._max_size == 100
        assert cache._ttl == 300
        assert cache._prefix == "p"
        assert len(cache._cache) == 0

    def test_custom_init(self):
        """Custom constructor values are stored correctly."""
        cache: PendingCache[int] = PendingCache(max_size=10, ttl_seconds=60, key_prefix="x")
        assert cache._max_size == 10
        assert cache._ttl == 60
        assert cache._prefix == "x"


class TestPendingCacheKey:
    """Test internal key generation."""

    def test_key_format(self):
        """_key returns prefix:conversation_id."""
        cache: PendingCache[str] = PendingCache(key_prefix="abc")
        assert cache._key("conv-1") == "abc:conv-1"

    def test_key_different_prefixes_differ(self):
        """Two caches with different prefixes produce different keys."""
        c1: PendingCache[str] = PendingCache(key_prefix="a")
        c2: PendingCache[str] = PendingCache(key_prefix="b")
        assert c1._key("x") != c2._key("x")


class TestPendingCacheStoreAndGet:
    """Test store / get pair."""

    async def test_store_then_get_returns_value(self):
        """store() followed by get() returns the stored value."""
        cache: PendingCache[str] = PendingCache(ttl_seconds=60)
        await cache.store("conv-1", "hello")
        result = await cache.get("conv-1")
        assert result == "hello"

    async def test_get_missing_returns_none(self):
        """get() for an absent key returns None."""
        cache: PendingCache[str] = PendingCache()
        assert await cache.get("missing") is None

    async def test_store_overwrites_prior_value(self):
        """A second store() for the same key replaces the first."""
        cache: PendingCache[str] = PendingCache(ttl_seconds=60)
        await cache.store("conv-1", "first")
        await cache.store("conv-1", "second")
        result = await cache.get("conv-1")
        assert result == "second"

    async def test_store_arbitrary_types(self):
        """store() works with complex value types."""
        cache: PendingCache[dict] = PendingCache(ttl_seconds=60)
        val = {"a": 1, "b": [1, 2, 3]}
        await cache.store("conv-1", val)
        result = await cache.get("conv-1")
        assert result == val


class TestPendingCacheTTL:
    """Test TTL expiry behaviour."""

    async def test_expired_entry_returns_none(self):
        """get() returns None and removes expired entries."""
        cache: PendingCache[str] = PendingCache(ttl_seconds=0)
        await cache.store("conv-1", "value")
        # ttl=0 means expired immediately; sleep to ensure time.time() >= expiry
        await _async_sleep(0.05)
        result = await cache.get("conv-1")
        assert result is None

    async def test_expired_entry_removed_from_cache(self):
        """An expired entry is deleted from _cache on access."""
        cache: PendingCache[str] = PendingCache(ttl_seconds=0)
        await cache.store("conv-1", "value")
        await _async_sleep(0.05)
        await cache.get("conv-1")
        assert cache._key("conv-1") not in cache._cache

    async def test_not_yet_expired_entry_remains(self):
        """An entry is still accessible before its TTL elapses."""
        cache: PendingCache[str] = PendingCache(ttl_seconds=600)
        await cache.store("conv-1", "valid")
        assert await cache.get("conv-1") == "valid"

    async def test_manual_expired_entry_via_direct_insertion(self):
        """Directly-inserted expired entries are treated as absent on get()."""
        cache: PendingCache[str] = PendingCache(ttl_seconds=60)
        key = cache._key("conv-x")
        cache._cache[key] = ("old", time.time() - 10)
        assert await cache.get("conv-x") is None
        assert key not in cache._cache


class TestPendingCacheClear:
    """Test clear() method."""

    async def test_clear_removes_stored_entry(self):
        """clear() makes the entry unavailable via get()."""
        cache: PendingCache[str] = PendingCache(ttl_seconds=60)
        await cache.store("conv-1", "value")
        await cache.clear("conv-1")
        assert await cache.get("conv-1") is None

    async def test_clear_missing_is_noop(self):
        """clear() on a non-existent key raises no error."""
        cache: PendingCache[str] = PendingCache()
        await cache.clear("does-not-exist")

    async def test_clear_only_removes_target_key(self):
        """clear() does not remove other entries."""
        cache: PendingCache[str] = PendingCache(ttl_seconds=60)
        await cache.store("conv-1", "a")
        await cache.store("conv-2", "b")
        await cache.clear("conv-1")
        assert await cache.get("conv-1") is None
        assert await cache.get("conv-2") == "b"


class TestPendingCacheMaxSize:
    """Test max_size eviction."""

    async def test_max_size_enforced_after_store(self):
        """Cache size never exceeds max_size after store() calls."""
        cache: PendingCache[int] = PendingCache(max_size=3, ttl_seconds=600)
        for i in range(5):
            await cache.store(f"conv-{i}", i)
        assert len(cache._cache) == 3

    async def test_oldest_entry_evicted_first(self):
        """The entry with the earliest expiry is removed when max_size is exceeded."""
        cache: PendingCache[int] = PendingCache(max_size=2, ttl_seconds=600)
        # Store entries with a tiny artificial gap so expiries are distinct
        await cache.store("conv-0", 0)
        await _async_sleep(0.01)
        await cache.store("conv-1", 1)
        await _async_sleep(0.01)
        await cache.store("conv-2", 2)  # triggers eviction of conv-0
        assert await cache.get("conv-0") is None
        assert await cache.get("conv-1") is not None
        assert await cache.get("conv-2") is not None


class TestPendingCacheCleanup:
    """Test _cleanup() method directly."""

    async def test_cleanup_removes_expired_entries(self):
        """_cleanup() evicts all expired entries."""
        cache: PendingCache[str] = PendingCache(max_size=10, ttl_seconds=60)
        key = cache._key("old")
        cache._cache[key] = ("stale", time.time() - 1)
        await cache.store("new", "fresh")  # triggers _cleanup
        assert key not in cache._cache

    async def test_cleanup_idempotent_on_empty_cache(self):
        """_cleanup() on an empty cache raises no error."""
        cache: PendingCache[str] = PendingCache()
        await cache._cleanup()
        assert len(cache._cache) == 0

    async def test_cleanup_trims_to_max_size_by_expiry_order(self):
        """_cleanup() removes the entry with the smallest expiry when over max_size."""
        cache: PendingCache[str] = PendingCache(max_size=2, ttl_seconds=60)
        now = time.time()
        cache._cache["p:oldest"] = ("a", now + 10)
        cache._cache["p:middle"] = ("b", now + 20)
        cache._cache["p:newest"] = ("c", now + 30)
        await cache._cleanup()
        assert len(cache._cache) == 2
        assert "p:oldest" not in cache._cache

    async def test_cleanup_called_twice_is_safe(self):
        """Calling _cleanup() twice in a row does not raise."""
        cache: PendingCache[str] = PendingCache(max_size=2, ttl_seconds=60)
        await cache.store("conv-1", "x")
        await cache._cleanup()
        await cache._cleanup()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _async_sleep(seconds: float) -> None:
    """Tiny async sleep so tests don't block the event loop."""
    await asyncio.sleep(seconds)
