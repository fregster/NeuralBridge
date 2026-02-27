"""Unit tests for preference_memory.py (Feature 15 — Adaptive Preference Learning)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.neuralbridge.preference_memory import (
    PREF_CATEGORY_FORMAT,
    PREF_CATEGORY_LOCATION,
    PREF_CATEGORY_SOURCE,
    PreferenceEntry,
    PreferenceMemory,
    _entry_from_dict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pref_memory(hass):
    """Return a PreferenceMemory wired to an AsyncMock store."""
    with patch("custom_components.neuralbridge.persistent_store.Store") as MockStore:
        mock_store = AsyncMock()
        MockStore.return_value = mock_store
        pm = PreferenceMemory(hass, "test-entry-id", max_entries=5)
    pm._store.async_load = AsyncMock(return_value=None)
    pm._store.async_save = AsyncMock()
    return pm


# ---------------------------------------------------------------------------
# _entry_from_dict
# ---------------------------------------------------------------------------


def test_entry_from_dict_missing_key_raises() -> None:
    """_entry_from_dict raises KeyError when a required key is absent."""
    with pytest.raises(KeyError):
        _entry_from_dict({"key": "k", "category": "source"})  # missing 'value', etc.


def test_entry_from_dict_bad_confidence_raises() -> None:
    """_entry_from_dict raises ValueError for a non-numeric confidence."""
    data = {
        "key": "news_source",
        "value": "BBC",
        "category": PREF_CATEGORY_SOURCE,
        "confirmed": False,
        "confidence": "not_a_float",
    }
    with pytest.raises((ValueError, TypeError)):
        _entry_from_dict(data)


def test_entry_from_dict_valid_minimal() -> None:
    """_entry_from_dict succeeds with the minimum required keys."""
    data = {
        "key": "news_source",
        "value": "BBC",
        "category": PREF_CATEGORY_SOURCE,
        "confirmed": False,
        "confidence": 0.8,
    }
    entry = _entry_from_dict(data)
    assert entry.key == "news_source"
    assert entry.value == "BBC"
    assert entry.confidence == 0.8
    assert entry.suggestion_count == 0
    assert entry.last_suggested == 0.0


def test_entry_from_dict_full_fields() -> None:
    """_entry_from_dict preserves optional fields when present."""
    now = time.time()
    data = {
        "key": "time_format",
        "value": "24h",
        "category": PREF_CATEGORY_FORMAT,
        "confirmed": True,
        "confidence": 1.0,
        "suggestion_count": 3,
        "last_suggested": now - 3600,
        "created_at": now - 7200,
        "updated_at": now,
    }
    entry = _entry_from_dict(data)
    assert entry.confirmed is True
    assert entry.suggestion_count == 3


# ---------------------------------------------------------------------------
# async_load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_load_empty_store_is_noop(pref_memory: PreferenceMemory) -> None:
    """async_load does nothing when the store returns None."""
    pref_memory._store.async_load = AsyncMock(return_value=None)
    await pref_memory.async_load()
    assert pref_memory._entries == {}


@pytest.mark.asyncio
async def test_async_load_non_dict_raw_is_noop(pref_memory: PreferenceMemory) -> None:
    """async_load ignores non-dict raw values (corrupt store)."""
    pref_memory._store.async_load = AsyncMock(return_value=[1, 2, 3])
    await pref_memory.async_load()
    assert pref_memory._entries == {}


@pytest.mark.asyncio
async def test_async_load_skips_corrupt_entry(pref_memory: PreferenceMemory) -> None:
    """async_load skips individual corrupt entries and continues loading."""
    good_entry = {
        "key": "news_source",
        "value": "BBC",
        "category": PREF_CATEGORY_SOURCE,
        "confirmed": False,
        "confidence": 0.8,
    }
    corrupt_entry = {"this_key_is_missing": True}
    pref_memory._store.async_load = AsyncMock(
        return_value={"entries": {"news_source": good_entry, "bad_key": corrupt_entry}}
    )
    await pref_memory.async_load()
    assert "news_source" in pref_memory._entries
    assert "bad_key" not in pref_memory._entries


@pytest.mark.asyncio
async def test_async_load_skips_non_dict_entry_value(pref_memory: PreferenceMemory) -> None:
    """async_load skips entries whose value is not a dict."""
    pref_memory._store.async_load = AsyncMock(
        return_value={"entries": {"news_source": "not_a_dict"}}
    )
    await pref_memory.async_load()
    assert pref_memory._entries == {}


@pytest.mark.asyncio
async def test_async_load_populates_entries(pref_memory: PreferenceMemory) -> None:
    """async_load correctly populates _entries from stored data."""
    entry_data = {
        "key": "time_format",
        "value": "24h",
        "category": PREF_CATEGORY_FORMAT,
        "confirmed": True,
        "confidence": 1.0,
    }
    pref_memory._store.async_load = AsyncMock(return_value={"entries": {"time_format": entry_data}})
    await pref_memory.async_load()
    assert "time_format" in pref_memory._entries
    assert pref_memory._entries["time_format"].value == "24h"


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def test_get_returns_none_for_missing_key(pref_memory: PreferenceMemory) -> None:
    """get() returns None when the key is not in the store."""
    assert pref_memory.get("nonexistent") is None


def test_get_returns_entry_when_present(pref_memory: PreferenceMemory) -> None:
    """get() returns the correct entry when present."""
    now = time.time()
    entry = PreferenceEntry(
        key="news_source",
        value="BBC",
        category=PREF_CATEGORY_SOURCE,
        confirmed=False,
        confidence=0.8,
        suggestion_count=0,
        last_suggested=0.0,
        created_at=now,
        updated_at=now,
    )
    pref_memory._entries["news_source"] = entry
    result = pref_memory.get("news_source")
    assert result is entry


# ---------------------------------------------------------------------------
# all_confirmed / all_entries
# ---------------------------------------------------------------------------


def test_all_confirmed_returns_only_confirmed(pref_memory: PreferenceMemory) -> None:
    """all_confirmed() returns only confirmed entries."""
    now = time.time()
    pref_memory._entries = {
        "a": PreferenceEntry("a", "va", PREF_CATEGORY_SOURCE, True, 1.0, 0, 0.0, now - 2, now),
        "b": PreferenceEntry("b", "vb", PREF_CATEGORY_LOCATION, False, 0.8, 0, 0.0, now - 1, now),
    }
    confirmed = pref_memory.all_confirmed()
    assert len(confirmed) == 1
    assert confirmed[0].key == "a"


def test_all_confirmed_sorted_by_created_at(pref_memory: PreferenceMemory) -> None:
    """all_confirmed() is sorted oldest-first."""
    now = time.time()
    pref_memory._entries = {
        "newer": PreferenceEntry("newer", "v", PREF_CATEGORY_SOURCE, True, 1.0, 0, 0.0, now, now),
        "older": PreferenceEntry(
            "older", "v", PREF_CATEGORY_SOURCE, True, 1.0, 0, 0.0, now - 100, now - 100
        ),
    }
    confirmed = pref_memory.all_confirmed()
    assert confirmed[0].key == "older"
    assert confirmed[1].key == "newer"


def test_all_entries_returns_all(pref_memory: PreferenceMemory) -> None:
    """all_entries() returns both confirmed and unconfirmed."""
    now = time.time()
    pref_memory._entries = {
        "a": PreferenceEntry("a", "va", PREF_CATEGORY_SOURCE, True, 1.0, 0, 0.0, now, now),
        "b": PreferenceEntry("b", "vb", PREF_CATEGORY_LOCATION, False, 0.5, 0, 0.0, now, now),
    }
    entries = pref_memory.all_entries()
    assert len(entries) == 2


def test_all_entries_sorted_oldest_first(pref_memory: PreferenceMemory) -> None:
    """all_entries() is sorted oldest-first."""
    now = time.time()
    pref_memory._entries = {
        "x": PreferenceEntry("x", "vx", PREF_CATEGORY_FORMAT, False, 0.7, 0, 0.0, now, now),
        "y": PreferenceEntry("y", "vy", PREF_CATEGORY_FORMAT, False, 0.7, 0, 0.0, now - 50, now),
    }
    entries = pref_memory.all_entries()
    assert entries[0].key == "y"


# ---------------------------------------------------------------------------
# is_suggestion_cooling_down
# ---------------------------------------------------------------------------


def test_is_suggestion_cooling_down_no_entry(pref_memory: PreferenceMemory) -> None:
    """Returns False for a key that does not exist."""
    assert pref_memory.is_suggestion_cooling_down("unknown") is False


def test_is_suggestion_cooling_down_never_suggested(pref_memory: PreferenceMemory) -> None:
    """Returns False when last_suggested is 0.0 (never suggested)."""
    now = time.time()
    pref_memory._entries["k"] = PreferenceEntry(
        "k", "v", PREF_CATEGORY_SOURCE, False, 0.5, 0, 0.0, now, now
    )
    assert pref_memory.is_suggestion_cooling_down("k") is False


def test_is_suggestion_cooling_down_within_window(pref_memory: PreferenceMemory) -> None:
    """Returns True when last_suggested was within the cooldown window."""
    now = time.time()
    pref_memory._entries["k"] = PreferenceEntry(
        "k", "v", PREF_CATEGORY_SOURCE, False, 0.5, 1, now - 3600, now, now
    )
    assert pref_memory.is_suggestion_cooling_down("k", cooldown_days=7) is True


def test_is_suggestion_cooling_down_outside_window(pref_memory: PreferenceMemory) -> None:
    """Returns False when the cooldown window has expired."""
    now = time.time()
    eight_days_ago = now - (8 * 86400)
    pref_memory._entries["k"] = PreferenceEntry(
        "k", "v", PREF_CATEGORY_SOURCE, False, 0.5, 1, eight_days_ago, now, now
    )
    assert pref_memory.is_suggestion_cooling_down("k", cooldown_days=7) is False


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_creates_new_entry(pref_memory: PreferenceMemory) -> None:
    """upsert() creates a new entry when the key does not exist."""
    entry = await pref_memory.upsert("news_source", "BBC", PREF_CATEGORY_SOURCE, 0.8)
    assert entry.key == "news_source"
    assert entry.value == "BBC"
    assert entry.confirmed is False
    pref_memory._store.async_save.assert_called()


@pytest.mark.asyncio
async def test_upsert_updates_existing_entry(pref_memory: PreferenceMemory) -> None:
    """upsert() updates an existing entry with new value."""
    await pref_memory.upsert("news_source", "BBC", PREF_CATEGORY_SOURCE, 0.8)
    entry = await pref_memory.upsert("news_source", "CNN", PREF_CATEGORY_SOURCE, 0.9)
    assert entry.value == "CNN"


@pytest.mark.asyncio
async def test_upsert_confidence_only_increases(pref_memory: PreferenceMemory) -> None:
    """upsert() never lowers confidence for an existing entry."""
    await pref_memory.upsert("k", "v", PREF_CATEGORY_FORMAT, 0.9)
    entry = await pref_memory.upsert("k", "v", PREF_CATEGORY_FORMAT, 0.5)
    assert entry.confidence == 0.9


@pytest.mark.asyncio
async def test_upsert_confirmed_flag_is_sticky(pref_memory: PreferenceMemory) -> None:
    """upsert() never un-confirms an already-confirmed entry."""
    await pref_memory.upsert("k", "v", PREF_CATEGORY_SOURCE, 1.0, confirmed=True)
    entry = await pref_memory.upsert("k", "v2", PREF_CATEGORY_SOURCE, 0.5, confirmed=False)
    assert entry.confirmed is True


@pytest.mark.asyncio
async def test_upsert_key_length_capped(pref_memory: PreferenceMemory) -> None:
    """upsert() silently truncates keys longer than 50 chars."""
    long_key = "k" * 100
    entry = await pref_memory.upsert(long_key, "v", PREF_CATEGORY_SOURCE, 0.8)
    assert len(entry.key) == 50


@pytest.mark.asyncio
async def test_upsert_value_length_capped(pref_memory: PreferenceMemory) -> None:
    """upsert() silently truncates values longer than 200 chars."""
    long_val = "v" * 300
    entry = await pref_memory.upsert("k", long_val, PREF_CATEGORY_SOURCE, 0.8)
    assert len(entry.value) == 200


@pytest.mark.asyncio
async def test_upsert_confidence_clamped(pref_memory: PreferenceMemory) -> None:
    """upsert() clamps confidence to [0.0, 1.0]."""
    entry_high = await pref_memory.upsert("k1", "v", PREF_CATEGORY_SOURCE, 5.0)
    entry_low = await pref_memory.upsert("k2", "v", PREF_CATEGORY_SOURCE, -1.0)
    assert entry_high.confidence == 1.0
    assert entry_low.confidence == 0.0


# ---------------------------------------------------------------------------
# confirm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_existing_entry(pref_memory: PreferenceMemory) -> None:
    """confirm() sets confirmed=True and confidence=1.0."""
    await pref_memory.upsert("news_source", "BBC", PREF_CATEGORY_SOURCE, 0.8)
    result = await pref_memory.confirm("news_source")
    assert result is True
    entry = pref_memory.get("news_source")
    assert entry is not None
    assert entry.confirmed is True
    assert entry.confidence == 1.0


@pytest.mark.asyncio
async def test_confirm_missing_key_returns_false(pref_memory: PreferenceMemory) -> None:
    """confirm() returns False when the key does not exist."""
    result = await pref_memory.confirm("nonexistent")
    assert result is False


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reject_existing_entry_removes_it(pref_memory: PreferenceMemory) -> None:
    """reject() removes the entry and returns True."""
    await pref_memory.upsert("news_source", "BBC", PREF_CATEGORY_SOURCE, 0.8)
    result = await pref_memory.reject("news_source")
    assert result is True
    assert pref_memory.get("news_source") is None


@pytest.mark.asyncio
async def test_reject_missing_key_returns_false(pref_memory: PreferenceMemory) -> None:
    """reject() returns False when the key does not exist."""
    result = await pref_memory.reject("nonexistent")
    assert result is False


# ---------------------------------------------------------------------------
# record_suggestion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_suggestion_increments_count(pref_memory: PreferenceMemory) -> None:
    """record_suggestion() increments suggestion_count and sets last_suggested."""
    await pref_memory.upsert("news_source", "BBC", PREF_CATEGORY_SOURCE, 0.8)
    before = time.time()
    await pref_memory.record_suggestion("news_source")
    entry = pref_memory.get("news_source")
    assert entry is not None
    assert entry.suggestion_count == 1
    assert entry.last_suggested >= before


@pytest.mark.asyncio
async def test_record_suggestion_increments_again(pref_memory: PreferenceMemory) -> None:
    """record_suggestion() accumulates across multiple calls."""
    await pref_memory.upsert("news_source", "BBC", PREF_CATEGORY_SOURCE, 0.8)
    await pref_memory.record_suggestion("news_source")
    await pref_memory.record_suggestion("news_source")
    entry = pref_memory.get("news_source")
    assert entry is not None
    assert entry.suggestion_count == 2


@pytest.mark.asyncio
async def test_record_suggestion_missing_key_is_noop(pref_memory: PreferenceMemory) -> None:
    """record_suggestion() is a no-op for a missing key."""
    await pref_memory.record_suggestion("nonexistent")
    # No exception; nothing changes


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_returns_count_and_empties(pref_memory: PreferenceMemory) -> None:
    """clear() removes all entries and returns the count."""
    await pref_memory.upsert("a", "v1", PREF_CATEGORY_SOURCE, 0.8)
    await pref_memory.upsert("b", "v2", PREF_CATEGORY_LOCATION, 0.7)
    count = await pref_memory.clear()
    assert count == 2
    assert pref_memory.all_entries() == []


@pytest.mark.asyncio
async def test_clear_empty_store_returns_zero(pref_memory: PreferenceMemory) -> None:
    """clear() returns 0 when the store is already empty."""
    count = await pref_memory.clear()
    assert count == 0


# ---------------------------------------------------------------------------
# max_entries property / set_max_entries
# ---------------------------------------------------------------------------


def test_max_entries_property(pref_memory: PreferenceMemory) -> None:
    """max_entries property returns the current cap."""
    assert pref_memory.max_entries == 5


def test_set_max_entries_updates_cap(pref_memory: PreferenceMemory) -> None:
    """set_max_entries() updates the cap."""
    pref_memory.set_max_entries(10)
    assert pref_memory.max_entries == 10


def test_set_max_entries_clamps_to_one(pref_memory: PreferenceMemory) -> None:
    """set_max_entries() clamps to minimum 1."""
    pref_memory.set_max_entries(0)
    assert pref_memory.max_entries == 1

    pref_memory.set_max_entries(-5)
    assert pref_memory.max_entries == 1


# ---------------------------------------------------------------------------
# _evict_if_needed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evict_under_capacity_no_eviction(pref_memory: PreferenceMemory) -> None:
    """_evict_if_needed does nothing when under capacity."""
    pref_memory.set_max_entries(10)
    await pref_memory.upsert("a", "v", PREF_CATEGORY_SOURCE, 0.8)
    assert len(pref_memory._entries) == 1


@pytest.mark.asyncio
async def test_evict_oldest_unconfirmed_first(pref_memory: PreferenceMemory) -> None:
    """_evict_if_needed evicts the oldest unconfirmed entry when at capacity."""
    now = time.time()
    pref_memory.set_max_entries(3)
    # Directly populate to control timestamps
    pref_memory._entries = {
        "old_unconfirmed": PreferenceEntry(
            "old_unconfirmed", "v", PREF_CATEGORY_SOURCE, False, 0.5, 0, 0.0, now - 100, now - 100
        ),
        "confirmed": PreferenceEntry(
            "confirmed", "v", PREF_CATEGORY_SOURCE, True, 1.0, 0, 0.0, now - 50, now - 50
        ),
        "recent_unconfirmed": PreferenceEntry(
            "recent_unconfirmed", "v", PREF_CATEGORY_SOURCE, False, 0.5, 0, 0.0, now - 10, now - 10
        ),
    }
    # Adding a 4th entry triggers eviction
    await pref_memory.upsert("new_entry", "v", PREF_CATEGORY_FORMAT, 0.7)
    assert "old_unconfirmed" not in pref_memory._entries
    assert "confirmed" in pref_memory._entries


@pytest.mark.asyncio
async def test_evict_oldest_confirmed_when_all_confirmed(pref_memory: PreferenceMemory) -> None:
    """_evict_if_needed evicts the least-recently-updated confirmed entry when all are confirmed."""
    now = time.time()
    pref_memory.set_max_entries(2)
    pref_memory._entries = {
        "oldest": PreferenceEntry(
            "oldest", "v", PREF_CATEGORY_SOURCE, True, 1.0, 0, 0.0, now - 200, now - 200
        ),
        "newest": PreferenceEntry(
            "newest", "v", PREF_CATEGORY_SOURCE, True, 1.0, 0, 0.0, now - 10, now - 10
        ),
    }
    await pref_memory.upsert("third", "v", PREF_CATEGORY_FORMAT, 0.7)
    assert "oldest" not in pref_memory._entries
    assert "newest" in pref_memory._entries
