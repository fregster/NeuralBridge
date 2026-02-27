"""Unit tests for the response cache module."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from unittest.mock import patch

from custom_components.neuralbridge.response_cache import (
    _MAX_KEY_CHARS,
    _MAX_VALUE_CHARS,
    CachedResponse,
    ResponseCache,
    _normalise_cache_key,
    _normalise_number_words,
)


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


class TestSemanticResponseCache:
    """Tests for semantic (normalised-key) mode of ResponseCache."""

    def test_semantic_store_and_get_paraphrase_hit(self) -> None:
        """Paraphrases of the same question share a semantic cache entry."""
        cache = ResponseCache(semantic=True)
        cache.store("what's the weather today", "sunny", "agent", normalise=True)
        result = cache.get("what is the weather", normalise=True)
        assert result == "sunny"

    def test_semantic_get_with_normalise_false_is_miss(self) -> None:
        """A semantic key is not found by a non-normalised get."""
        cache = ResponseCache(semantic=True)
        cache.store("what's the weather today", "sunny", "agent", normalise=True)
        result = cache.get("what's the weather today", normalise=False)
        assert result is None

    def test_semantic_get_normalised_text_mismatch_returns_none(self) -> None:
        """Returns None when stored normalised_text does not match the query."""
        cache = ResponseCache(semantic=True)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("set alarm for three", "done", "agent", normalise=True)
        # Directly corrupt the stored entry's normalised_text to force a mismatch
        entries = list(cache._cache.values())
        assert len(entries) == 1
        corrupted = dc_replace(entries[0], normalised_text="something completely different")
        key = next(iter(cache._cache.keys()))
        cache._cache[key] = corrupted

        with patch("time.monotonic", return_value=1001.0):
            result = cache.get("set alarm for three", normalise=True)
        assert result is None

    def test_semantic_store_uses_semantic_ttl(self) -> None:
        """Semantic entries expire according to the semantic TTL, not exact TTL."""
        cache = ResponseCache(ttl_seconds=300, semantic=True, semantic_ttl_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("what's the weather", "cloudy", "agent", normalise=True)
        # Within exact TTL (300s) but beyond semantic TTL (60s)
        with patch("time.monotonic", return_value=1065.0):
            assert cache.get("what is the weather", normalise=True) is None

    def test_configure_updates_semantic_flag(self) -> None:
        """configure() with semantic=True enables semantic mode."""
        cache = ResponseCache(semantic=False)
        cache.configure(enabled=True, ttl_seconds=300, semantic=True)
        assert cache.semantic is True

    def test_configure_updates_semantic_ttl(self) -> None:
        """configure() with semantic_ttl_seconds updates the semantic TTL."""
        cache = ResponseCache(semantic_ttl_seconds=60)
        cache.configure(enabled=True, ttl_seconds=300, semantic_ttl_seconds=30)
        assert cache.semantic_ttl == 30

    def test_configure_semantic_none_leaves_flag_unchanged(self) -> None:
        """configure() without semantic kwarg leaves the semantic flag unchanged."""
        cache = ResponseCache(semantic=True)
        cache.configure(enabled=True, ttl_seconds=300)
        assert cache.semantic is True

    def test_semantic_property_returns_current_mode(self) -> None:
        """semantic property reflects the initial setting."""
        cache = ResponseCache(semantic=True)
        assert cache.semantic is True
        cache2 = ResponseCache(semantic=False)
        assert cache2.semantic is False


# ---------------------------------------------------------------------------
# Feature 9 — Semantic (normalised) cache keying
# ---------------------------------------------------------------------------


class TestNormaliseNumberWords:
    """Tests for _normalise_number_words helper."""

    def test_empty_list(self) -> None:
        """Empty word list returns empty list."""
        assert _normalise_number_words([]) == []

    def test_single_unit(self) -> None:
        """Single unit number word is converted to digit."""
        assert _normalise_number_words(["three"]) == ["3"]

    def test_single_teen(self) -> None:
        """Teen number word is converted to digit."""
        assert _normalise_number_words(["seventeen"]) == ["17"]

    def test_single_ten(self) -> None:
        """Plain tens number word is converted to digit."""
        assert _normalise_number_words(["forty"]) == ["40"]

    def test_compound_tens_and_units(self) -> None:
        """Compound 'twenty two' is combined into '22'."""
        assert _normalise_number_words(["twenty", "two"]) == ["22"]

    def test_compound_ninety_nine(self) -> None:
        """Compound 'ninety nine' maps to '99'."""
        assert _normalise_number_words(["ninety", "nine"]) == ["99"]

    def test_tens_without_following_unit(self) -> None:
        """A plain tens word at end of list is not combined."""
        assert _normalise_number_words(["twenty"]) == ["20"]

    def test_tens_followed_by_non_unit(self) -> None:
        """Tens followed by a non-unit word are converted separately."""
        assert _normalise_number_words(["twenty", "lights"]) == ["20", "lights"]

    def test_non_number_words_pass_through(self) -> None:
        """Non-number words pass through unchanged."""
        assert _normalise_number_words(["hello", "world"]) == ["hello", "world"]

    def test_mixed_words_and_numbers(self) -> None:
        """Mix of normal words and numbers is handled correctly."""
        result = _normalise_number_words(["turn", "on", "two", "lights"])
        assert result == ["turn", "on", "2", "lights"]

    def test_zero(self) -> None:
        """Zero is converted to '0'."""
        assert _normalise_number_words(["zero"]) == ["0"]


class TestNormaliseCacheKey:
    """Tests for the _normalise_cache_key module-level function."""

    def test_lowercase(self) -> None:
        """Input is lowercased."""
        assert _normalise_cache_key("WEATHER") == "weather"

    def test_strips_punctuation(self) -> None:
        """Punctuation is replaced with spaces."""
        assert _normalise_cache_key("hello, world!") == "hello world"

    def test_strips_filler_what_s(self) -> None:
        """Filler phrase 'what's' is removed."""
        result = _normalise_cache_key("What's the weather?")
        assert "what" not in result
        assert "weather" in result

    def test_strips_filler_how_is(self) -> None:
        """Filler phrase 'how is' is removed."""
        result = _normalise_cache_key("How is the weather today")
        assert "how" not in result
        assert "weather" in result

    def test_strips_filler_can_you(self) -> None:
        """Filler phrase 'can you' is removed."""
        result = _normalise_cache_key("can you tell me the weather")
        assert "can" not in result

    def test_strips_filler_could_you(self) -> None:
        """Filler phrase 'could you' is removed."""
        result = _normalise_cache_key("could you check the temperature")
        assert "could" not in result
        assert "temperature" in result

    def test_strips_filler_would_you(self) -> None:
        """Filler phrase 'would you' is removed."""
        result = _normalise_cache_key("would you tell me the time")
        assert "would" not in result

    def test_strips_filler_tell_me(self) -> None:
        """Filler phrase 'tell me' is removed."""
        result = _normalise_cache_key("tell me the weather")
        assert "tell" not in result
        assert "weather" in result

    def test_strips_filler_please(self) -> None:
        """Filler word 'please' is removed."""
        result = _normalise_cache_key("please check the weather")
        assert "please" not in result
        assert "weather" in result

    def test_strips_filler_today(self) -> None:
        """Filler word 'today' is removed."""
        result = _normalise_cache_key("weather today")
        assert "today" not in result

    def test_strips_filler_right_now(self) -> None:
        """Filler phrase 'right now' is removed."""
        result = _normalise_cache_key("weather right now")
        assert "right" not in result

    def test_strips_filler_currently(self) -> None:
        """Filler word 'currently' is removed."""
        result = _normalise_cache_key("currently the weather is nice")
        assert "currently" not in result

    def test_strips_filler_now(self) -> None:
        """Standalone 'now' is removed as a filler."""
        result = _normalise_cache_key("weather now")
        assert "now" not in result

    def test_strips_filler_how_s(self) -> None:
        """Filler contraction 'how's' is removed."""
        result = _normalise_cache_key("How's the temperature?")
        assert "how" not in result
        assert "temperature" in result

    def test_strips_filler_what_is(self) -> None:
        """Filler phrase 'what is' is removed."""
        result = _normalise_cache_key("what is the temperature")
        assert "what" not in result

    def test_number_word_normalisation(self) -> None:
        """Number words are converted to digits."""
        # "lights" (6 chars, ends in 's') is also stemmed to "light"
        assert _normalise_cache_key("turn on two lights") == "turn on 2 light"

    def test_compound_number_normalisation(self) -> None:
        """Compound number words convert correctly."""
        assert _normalise_cache_key("set to twenty five percent") == "set to 25 percent"

    def test_suffix_ing_stripped(self) -> None:
        """'-ing' suffix stripped from words > 4 chars with sufficient stem."""
        result = _normalise_cache_key("running")
        assert result == "runn"

    def test_suffix_ed_stripped(self) -> None:
        """'-ed' suffix stripped from words > 4 chars with sufficient stem."""
        result = _normalise_cache_key("locked")
        assert result == "lock"

    def test_suffix_s_stripped(self) -> None:
        """'-s' suffix stripped from words > 4 chars with sufficient stem."""
        result = _normalise_cache_key("lights")
        assert result == "light"

    def test_short_word_not_stemmed(self) -> None:
        """Words ≤ 4 chars are not stemmed."""
        assert _normalise_cache_key("cats") == "cats"

    def test_collapses_whitespace(self) -> None:
        """Multiple spaces are collapsed to single space."""
        result = _normalise_cache_key("  hello   world  ")
        assert result == "hello world"

    def test_paraphrase_weather_variants(self) -> None:
        """Paraphrase variants produce the same normalised key."""
        key1 = _normalise_cache_key("What's the weather?")
        key2 = _normalise_cache_key("How is the weather today")
        assert key1 == key2

    def test_paraphrase_temperature_variants(self) -> None:
        """Different phrasings of temperature query normalise identically."""
        key1 = _normalise_cache_key("What is the temperature?")
        key2 = _normalise_cache_key("How's the temperature currently?")
        assert key1 == key2

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert _normalise_cache_key("") == ""

    def test_only_filler_words(self) -> None:
        """Text containing only filler words returns an empty string."""
        result = _normalise_cache_key("please tell me")
        assert result == ""


class TestCachedResponseNormalisedTextField:
    """Tests for the normalised_text field on CachedResponse."""

    def test_normalised_text_defaults_to_none(self) -> None:
        """normalised_text is None when not provided."""
        entry = CachedResponse(
            response_text="hello",
            agent_name="agent",
            cached_at=1000.0,
            expires_at=1300.0,
        )
        assert entry.normalised_text is None

    def test_normalised_text_can_be_set(self) -> None:
        """normalised_text can be populated at construction."""
        entry = CachedResponse(
            response_text="hello",
            agent_name="agent",
            cached_at=1000.0,
            expires_at=1300.0,
            normalised_text="weather",
        )
        assert entry.normalised_text == "weather"


class TestResponseCacheSemanticInit:
    """Tests for semantic parameters on ResponseCache.__init__."""

    def test_semantic_defaults_to_false(self) -> None:
        """semantic defaults to False."""
        cache = ResponseCache()
        assert cache.semantic is False

    def test_semantic_ttl_defaults_to_60(self) -> None:
        """semantic_ttl defaults to 60 seconds."""
        cache = ResponseCache()
        assert cache.semantic_ttl == 60

    def test_semantic_can_be_enabled(self) -> None:
        """semantic can be set to True at construction."""
        cache = ResponseCache(semantic=True)
        assert cache.semantic is True

    def test_semantic_ttl_custom(self) -> None:
        """semantic_ttl_seconds parameter is respected."""
        cache = ResponseCache(semantic_ttl_seconds=45)
        assert cache.semantic_ttl == 45


class TestResponseCacheSemanticGetStore:
    """Tests for normalise=True on get() and store()."""

    def test_store_normalise_then_get_normalise_hit(self) -> None:
        """Storing and retrieving with normalise=True produces a cache hit."""
        cache = ResponseCache()
        cache.store("What's the weather?", "Sunny", "agent", normalise=True)
        result = cache.get("What's the weather?", normalise=True)
        assert result == "Sunny"

    def test_store_normalise_get_paraphrase_hit(self) -> None:
        """A paraphrase variant retrieves the same entry via normalise=True."""
        cache = ResponseCache()
        cache.store("What's the weather?", "Sunny", "agent", normalise=True)
        result = cache.get("How is the weather today", normalise=True)
        assert result == "Sunny"

    def test_store_normalise_get_exact_miss(self) -> None:
        """A semantic-keyed entry is NOT found by an exact (normalise=False) lookup."""
        cache = ResponseCache()
        cache.store("What's the weather?", "Sunny", "agent", normalise=True)
        # Exact key is different from semantic key
        assert cache.get("What's the weather?", normalise=False) is None

    def test_store_exact_get_normalise_miss(self) -> None:
        """An exact-keyed entry is NOT found by a normalised (normalise=True) lookup."""
        cache = ResponseCache()
        cache.store("What's the weather?", "Sunny", "agent", normalise=False)
        assert cache.get("How is the weather today", normalise=True) is None

    def test_get_normalise_disabled_returns_none(self) -> None:
        """get(normalise=True) returns None when cache is disabled."""
        cache = ResponseCache(enabled=False)
        cache.store("hello", "world", "agent", normalise=True)
        assert cache.get("hello", normalise=True) is None

    def test_store_normalise_disabled_no_op(self) -> None:
        """store(normalise=True) is a no-op when cache is disabled."""
        cache = ResponseCache(enabled=False)
        cache.store("hello", "world", "agent", normalise=True)
        assert cache.size == 0

    def test_semantic_entry_uses_semantic_ttl(self) -> None:
        """Semantic entries expire after semantic_ttl, not the exact TTL."""
        cache = ResponseCache(ttl_seconds=300, semantic_ttl_seconds=30)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("What's the weather?", "Sunny", "agent", normalise=True)

        with patch("time.monotonic", return_value=1031.0):
            # Semantic TTL (30s) elapsed — entry should be expired
            assert cache.get("What's the weather?", normalise=True) is None

    def test_semantic_entry_within_semantic_ttl(self) -> None:
        """Semantic entries are still valid before the semantic_ttl expires."""
        cache = ResponseCache(ttl_seconds=300, semantic_ttl_seconds=30)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("What's the weather?", "Sunny", "agent", normalise=True)

        with patch("time.monotonic", return_value=1025.0):
            assert cache.get("What's the weather?", normalise=True) == "Sunny"

    def test_get_normalise_normalised_text_mismatch_returns_none(self) -> None:
        """get(normalise=True) returns None when normalised_text is corrupted."""
        cache = ResponseCache()
        cache.store("hello world", "response", "agent", normalise=True)
        # Corrupt the stored normalised_text to simulate a hash collision scenario
        for entry in cache._cache.values():
            entry.normalised_text = "completely different normalised form"
        assert cache.get("hello world", normalise=True) is None

    def test_semantic_entry_stores_normalised_text(self) -> None:
        """Entry stored with normalise=True has normalised_text set."""
        cache = ResponseCache()
        cache.store("What's the weather?", "Sunny", "agent", normalise=True)
        entry = next(iter(cache._cache.values()))
        assert entry.normalised_text is not None
        assert entry.normalised_text == _normalise_cache_key("What's the weather?")

    def test_exact_entry_normalised_text_is_none(self) -> None:
        """Entry stored with normalise=False has normalised_text as None."""
        cache = ResponseCache()
        cache.store("hello", "world", "agent", normalise=False)
        entry = next(iter(cache._cache.values()))
        assert entry.normalised_text is None

    def test_get_normalise_expired_entry_removed(self) -> None:
        """Expired normalised entry is removed from cache on access."""
        cache = ResponseCache(semantic_ttl_seconds=30)
        with patch("time.monotonic", return_value=1000.0):
            cache.store("What's the weather?", "Sunny", "agent", normalise=True)
            assert cache.size == 1

        with patch("time.monotonic", return_value=1031.0):
            cache.get("What's the weather?", normalise=True)
            assert cache.size == 0


class TestResponseCacheSemanticConfigure:
    """Tests for semantic params in configure()."""

    def test_configure_enables_semantic(self) -> None:
        """configure() with semantic=True enables semantic keying."""
        cache = ResponseCache(semantic=False)
        cache.configure(enabled=True, ttl_seconds=300, semantic=True)
        assert cache.semantic is True

    def test_configure_disables_semantic(self) -> None:
        """configure() with semantic=False disables semantic keying."""
        cache = ResponseCache(semantic=True)
        cache.configure(enabled=True, ttl_seconds=300, semantic=False)
        assert cache.semantic is False

    def test_configure_updates_semantic_ttl(self) -> None:
        """configure() with semantic_ttl_seconds updates the semantic TTL."""
        cache = ResponseCache(semantic_ttl_seconds=60)
        cache.configure(enabled=True, ttl_seconds=300, semantic_ttl_seconds=45)
        assert cache.semantic_ttl == 45

    def test_configure_without_semantic_does_not_change_it(self) -> None:
        """configure() without semantic param leaves it unchanged."""
        cache = ResponseCache(semantic=True)
        cache.configure(enabled=True, ttl_seconds=300)
        assert cache.semantic is True

    def test_configure_without_semantic_ttl_does_not_change_it(self) -> None:
        """configure() without semantic_ttl_seconds leaves it unchanged."""
        cache = ResponseCache(semantic_ttl_seconds=90)
        cache.configure(enabled=True, ttl_seconds=300)
        assert cache.semantic_ttl == 90

    def test_configure_semantic_none_explicitly_does_not_change(self) -> None:
        """configure() with semantic=None explicitly leaves semantic unchanged."""
        cache = ResponseCache(semantic=True)
        cache.configure(enabled=True, ttl_seconds=300, semantic=None)
        assert cache.semantic is True

    def test_configure_semantic_ttl_none_explicitly_does_not_change(self) -> None:
        """configure() with semantic_ttl_seconds=None explicitly leaves it unchanged."""
        cache = ResponseCache(semantic_ttl_seconds=90)
        cache.configure(enabled=True, ttl_seconds=300, semantic_ttl_seconds=None)
        assert cache.semantic_ttl == 90


# ---------------------------------------------------------------------------
# P6 — Input size caps
# ---------------------------------------------------------------------------


class TestResponseCacheSizeCaps:
    """Tests for _MAX_KEY_CHARS and _MAX_VALUE_CHARS size guards."""

    def test_store_oversized_key_is_silently_ignored(self) -> None:
        """store() with text longer than _MAX_KEY_CHARS does not add an entry."""
        cache = ResponseCache()
        oversized_text = "x" * (_MAX_KEY_CHARS + 1)
        cache.store(oversized_text, "response", "agent")
        assert len(cache._cache) == 0

    def test_store_oversized_value_is_silently_ignored(self) -> None:
        """store() with response longer than _MAX_VALUE_CHARS does not add an entry."""
        cache = ResponseCache()
        oversized_response = "y" * (_MAX_VALUE_CHARS + 1)
        cache.store("short query", oversized_response, "agent")
        assert len(cache._cache) == 0

    def test_get_oversized_key_returns_none(self) -> None:
        """get() with text over _MAX_KEY_CHARS returns None without error."""
        cache = ResponseCache()
        oversized_text = "z" * (_MAX_KEY_CHARS + 1)
        result = cache.get(oversized_text)
        assert result is None

    def test_normalise_cache_key_truncates_oversized_input(self) -> None:
        """_normalise_cache_key truncates input at _MAX_KEY_CHARS before processing."""
        long_text = "word " * (_MAX_KEY_CHARS // 5 + 10)
        assert len(long_text) > _MAX_KEY_CHARS
        result = _normalise_cache_key(long_text)
        # Result is derived from a truncated prefix, so it must be shorter
        # than the original input would produce. The key property is no error.
        assert isinstance(result, str)
        assert len(result) <= _MAX_KEY_CHARS
