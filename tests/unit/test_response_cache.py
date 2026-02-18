"""Unit tests for the response cache module."""

from __future__ import annotations

from unittest.mock import patch

from custom_components.neuralbridge.response_cache import CachedResponse, ResponseCache


class TestCachedResponse:
    """Tests for CachedResponse dataclass."""

    def test_fields(self) -> None:
        """Test CachedResponse stores all expected fields."""
        entry = CachedResponse(
            response_text="hello",
            agent_name="Llama3",
            cached_at=1000.0,
            expires_at=1300.0,
        )
        assert entry.response_text == "hello"
        assert entry.agent_name == "Llama3"
        assert entry.cached_at == 1000.0
        assert entry.expires_at == 1300.0


class TestResponseCache:
    """Tests for ResponseCache class."""

    def test_init_defaults(self) -> None:
        """Test ResponseCache initialises with correct defaults."""
        cache = ResponseCache()
        assert cache.enabled is True
        assert cache.ttl == 300
        assert cache.size == 0

    def test_init_custom(self) -> None:
        """Test ResponseCache accepts custom initialisation parameters."""
        cache = ResponseCache(enabled=False, ttl_seconds=60, max_size=50)
        assert cache.enabled is False
        assert cache.ttl == 60

    def test_get_disabled_returns_none(self) -> None:
        """Test get returns None when cache is disabled."""
        cache = ResponseCache(enabled=False)
        cache.store("hello", "world", "agent")  # Should not store
        assert cache.get("hello") is None

    def test_get_miss_returns_none(self) -> None:
        """Test get returns None for an unknown key."""
        cache = ResponseCache()
        assert cache.get("unknown query") is None

    def test_store_and_get(self) -> None:
        """Test storing a response and retrieving it."""
        cache = ResponseCache()
        cache.store("what is 2+2", "4", "Llama3")
        assert cache.get("what is 2+2") == "4"

    def test_store_disabled_does_not_cache(self) -> None:
        """Test store is a no-op when cache is disabled."""
        cache = ResponseCache(enabled=False)
        cache.store("hello", "world", "agent")
        assert cache.size == 0

    def test_key_normalisation_lowercase(self) -> None:
        """Test that uppercase and lowercase queries map to the same cache entry."""
        cache = ResponseCache()
        cache.store("Hello World", "response", "agent")
        assert cache.get("hello world") == "response"
        assert cache.get("HELLO WORLD") == "response"

    def test_key_normalisation_strips_whitespace(self) -> None:
        """Test that leading/trailing whitespace is ignored in cache keys."""
        cache = ResponseCache()
        cache.store("  hello  ", "response", "agent")
        assert cache.get("hello") == "response"
        assert cache.get("  hello  ") == "response"

    def test_ttl_expiry(self) -> None:
        """Test that expired entries are not returned."""
        cache = ResponseCache(ttl_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("hello", "world", "agent")

        with patch("time.monotonic", return_value=1061.0):
            assert cache.get("hello") is None

    def test_ttl_not_expired(self) -> None:
        """Test that entries within TTL are still returned."""
        cache = ResponseCache(ttl_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("hello", "world", "agent")

        with patch("time.monotonic", return_value=1059.0):
            assert cache.get("hello") == "world"

    def test_expired_entry_removed_on_get(self) -> None:
        """Test that accessing an expired entry removes it from the cache."""
        cache = ResponseCache(ttl_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("hello", "world", "agent")
            assert cache.size == 1

        with patch("time.monotonic", return_value=1061.0):
            cache.get("hello")  # Should remove the expired entry
            assert cache.size == 0

    def test_invalidate_clears_all(self) -> None:
        """Test invalidate removes all cached entries."""
        cache = ResponseCache()
        cache.store("q1", "a1", "agent")
        cache.store("q2", "a2", "agent")
        count = cache.invalidate()
        assert count == 2
        assert cache.size == 0
        assert cache.get("q1") is None

    def test_invalidate_empty_cache(self) -> None:
        """Test invalidate on an empty cache returns 0."""
        cache = ResponseCache()
        assert cache.invalidate() == 0

    def test_configure_enable_disable(self) -> None:
        """Test configure toggles the enabled state."""
        cache = ResponseCache(enabled=True)
        cache.store("q", "a", "agent")
        cache.configure(enabled=False, ttl_seconds=300)
        assert cache.enabled is False
        assert cache.get("q") is None

    def test_configure_disable_purges_cache(self) -> None:
        """Test disabling the cache via configure purges existing entries."""
        cache = ResponseCache()
        cache.store("q", "a", "agent")
        assert cache.size == 1
        cache.configure(enabled=False, ttl_seconds=300)
        assert cache.size == 0

    def test_configure_updates_ttl(self) -> None:
        """Test configure updates the TTL for subsequent stores."""
        cache = ResponseCache(ttl_seconds=300)
        cache.configure(enabled=True, ttl_seconds=60)
        assert cache.ttl == 60

    def test_max_size_evicts_oldest(self) -> None:
        """Test that oldest entries are evicted when max_size is reached."""
        cache = ResponseCache(max_size=2)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("q1", "a1", "agent")
        with patch("time.monotonic", return_value=1001.0):
            cache.store("q2", "a2", "agent")
        # Assertions are inside the patch block so that get() uses the same fake
        # clock; otherwise real time.monotonic() would exceed expires_at (patched
        # timestamp + TTL) and report all entries as expired.
        with patch("time.monotonic", return_value=1002.0):
            cache.store("q3", "a3", "agent")  # q1 should be evicted
            assert cache.get("q1") is None
            assert cache.get("q2") == "a2"
            assert cache.get("q3") == "a3"

    def test_size_property(self) -> None:
        """Test size reflects the current number of cached entries."""
        cache = ResponseCache()
        assert cache.size == 0
        cache.store("q1", "a1", "agent")
        assert cache.size == 1
        cache.store("q2", "a2", "agent")
        assert cache.size == 2

    def test_overwrite_same_key(self) -> None:
        """Test storing with the same key overwrites the previous entry."""
        cache = ResponseCache()
        cache.store("hello", "first response", "agent-A")
        cache.store("hello", "second response", "agent-B")
        assert cache.get("hello") == "second response"
        assert cache.size == 1

    def test_cleanup_removes_expired_before_insert(self) -> None:
        """Test _cleanup removes expired entries when a new entry is stored."""
        cache = ResponseCache(ttl_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("old-q", "old-a", "agent")

        with patch("time.monotonic", return_value=1061.0):
            cache.store("new-q", "new-a", "agent")
            # Expired entry should have been cleaned up
            assert cache.get("old-q") is None
            assert cache.get("new-q") == "new-a"
