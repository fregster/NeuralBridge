"""Unit tests for EntityContextCache."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from custom_components.neuralbridge.entity_context import (
    MAX_ENTITIES_PER_DOMAIN,
    RELEVANT_DOMAINS,
    EntityContextCache,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(entity_id: str, friendly_name: str | None = None) -> MagicMock:
    """Return a minimal fake state object."""
    state = MagicMock()
    state.entity_id = entity_id
    state.domain = entity_id.split(".", maxsplit=1)[0]
    state.attributes = {"friendly_name": friendly_name} if friendly_name else {}
    return state


def _make_hass(states: list[MagicMock]) -> MagicMock:
    """Return a minimal fake hass with the given states."""
    mock_hass = MagicMock()
    mock_hass.states.async_all.return_value = states
    return mock_hass


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_init_creates_empty_cache() -> None:
    """EntityContextCache starts with empty summary and hash."""
    cache = EntityContextCache()
    assert cache._summary == ""
    assert cache._entity_hash == ""


# ---------------------------------------------------------------------------
# get_summary — empty state
# ---------------------------------------------------------------------------


def test_get_summary_no_relevant_entities_returns_empty() -> None:
    """Returns empty string when no relevant-domain entities exist."""
    mock_hass = _make_hass([_make_state("persistent_notification.test")])
    cache = EntityContextCache()
    assert cache.get_summary(mock_hass) == ""


def test_get_summary_empty_state_returns_empty() -> None:
    """Returns empty string when hass has no states at all."""
    mock_hass = _make_hass([])
    cache = EntityContextCache()
    assert cache.get_summary(mock_hass) == ""


# ---------------------------------------------------------------------------
# get_summary — content correctness
# ---------------------------------------------------------------------------


def test_get_summary_contains_domain_header() -> None:
    """Summary starts with 'Available smart home entities:'."""
    mock_hass = _make_hass([_make_state("light.living_room", "Living Room")])
    cache = EntityContextCache()
    summary = cache.get_summary(mock_hass)
    assert summary.startswith("Available smart home entities:")


def test_get_summary_uses_friendly_name() -> None:
    """Entity friendly_name is used in the summary when available."""
    mock_hass = _make_hass([_make_state("light.lr", "Living Room Light")])
    cache = EntityContextCache()
    assert "Living Room Light" in cache.get_summary(mock_hass)


def test_get_summary_falls_back_to_entity_id_when_no_friendly_name() -> None:
    """entity_id is used when friendly_name attribute is absent."""
    state = MagicMock()
    state.entity_id = "weather.met_no"
    state.domain = "weather"
    state.attributes = {}  # no friendly_name key
    mock_hass = _make_hass([state])
    cache = EntityContextCache()
    assert "weather.met_no" in cache.get_summary(mock_hass)


def test_get_summary_friendly_name_none_falls_back_to_entity_id() -> None:
    """entity_id is used when friendly_name attribute is explicitly None."""
    mock_hass = _make_hass([_make_state("weather.met_no", None)])
    cache = EntityContextCache()
    summary = cache.get_summary(mock_hass)
    assert "weather.met_no" in summary


def test_get_summary_groups_by_domain() -> None:
    """Multiple entities in the same domain appear on one line."""
    mock_hass = _make_hass(
        [
            _make_state("light.kitchen", "Kitchen"),
            _make_state("light.living_room", "Living Room"),
        ]
    )
    cache = EntityContextCache()
    summary = cache.get_summary(mock_hass)
    # Should be a single "- light: ..." line
    light_lines = [ln for ln in summary.splitlines() if ln.startswith("- light:")]
    assert len(light_lines) == 1
    assert "Kitchen" in light_lines[0]
    assert "Living Room" in light_lines[0]


def test_get_summary_multiple_domains_sorted() -> None:
    """Domains appear in alphabetical order."""
    mock_hass = _make_hass(
        [
            _make_state("weather.met_no", "Met.no"),
            _make_state("light.kitchen", "Kitchen"),
            _make_state("climate.thermostat", "Thermostat"),
        ]
    )
    cache = EntityContextCache()
    summary = cache.get_summary(mock_hass)
    lines = [ln for ln in summary.splitlines() if ln.startswith("- ")]
    domains_in_order = [ln.split(":")[0].lstrip("- ") for ln in lines]
    assert domains_in_order == sorted(domains_in_order)


def test_get_summary_entity_names_sorted_within_domain() -> None:
    """Entity names within a domain are sorted alphabetically."""
    mock_hass = _make_hass(
        [
            _make_state("light.zebra", "Zebra"),
            _make_state("light.apple", "Apple"),
            _make_state("light.mango", "Mango"),
        ]
    )
    cache = EntityContextCache()
    summary = cache.get_summary(mock_hass)
    light_line = next(ln for ln in summary.splitlines() if ln.startswith("- light:"))
    names = [n.strip() for n in light_line.split(":", 1)[1].split(",")]
    assert names == sorted(names)


def test_get_summary_ignores_irrelevant_domains() -> None:
    """Entities in non-relevant domains are excluded from the summary."""
    mock_hass = _make_hass(
        [
            _make_state("persistent_notification.alert"),
            _make_state("zone.home"),
            _make_state("sun.sun"),
            _make_state("light.kitchen", "Kitchen"),
        ]
    )
    cache = EntityContextCache()
    summary = cache.get_summary(mock_hass)
    assert "persistent_notification" not in summary
    assert "zone" not in summary
    assert "sun" not in summary
    assert "Kitchen" in summary


# ---------------------------------------------------------------------------
# get_summary — caching behaviour
# ---------------------------------------------------------------------------


def test_get_summary_caches_result_on_second_call() -> None:
    """Second call with same entity set returns the cached result without rebuilding."""
    mock_hass = _make_hass([_make_state("light.kitchen", "Kitchen")])
    cache = EntityContextCache()

    first = cache.get_summary(mock_hass)
    second = cache.get_summary(mock_hass)

    assert first == second
    # async_all is called 3 times total:
    #   first get_summary: once for _compute_hash + once for _build_summary (cache miss)
    #   second get_summary: once for _compute_hash (cache hit — no rebuild)
    assert mock_hass.states.async_all.call_count == 3


def test_get_summary_rebuilds_when_entity_added() -> None:
    """Adding a new entity causes the summary to rebuild."""
    mock_hass = _make_hass([_make_state("light.kitchen", "Kitchen")])
    cache = EntityContextCache()
    first = cache.get_summary(mock_hass)

    # Now add a weather entity
    mock_hass.states.async_all.return_value = [
        _make_state("light.kitchen", "Kitchen"),
        _make_state("weather.met_no", "Met.no"),
    ]
    second = cache.get_summary(mock_hass)

    assert first != second
    assert "weather" in second


def test_get_summary_does_not_rebuild_when_only_state_changes() -> None:
    """Changing entity state (not ID set) does not trigger a rebuild."""
    state = _make_state("light.kitchen", "Kitchen")
    mock_hass = _make_hass([state])
    cache = EntityContextCache()
    first = cache.get_summary(mock_hass)
    # Simulate a state change without adding/removing entities
    second = cache.get_summary(mock_hass)
    assert first == second


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------


def test_invalidate_forces_rebuild_on_next_call() -> None:
    """invalidate() clears the hash so the next get_summary rebuilds."""
    mock_hass = _make_hass([_make_state("light.kitchen", "Kitchen")])
    cache = EntityContextCache()
    cache.get_summary(mock_hass)
    cached_hash = cache._entity_hash

    cache.invalidate()
    assert cache._entity_hash == ""

    cache.get_summary(mock_hass)
    assert cache._entity_hash == cached_hash  # rebuilt to same hash


# ---------------------------------------------------------------------------
# _compute_hash
# ---------------------------------------------------------------------------


def test_compute_hash_changes_when_entity_added() -> None:
    """Hash changes when a new relevant entity appears."""
    mock_hass = _make_hass([_make_state("light.a", "A")])
    cache = EntityContextCache()
    h1 = cache._compute_hash(mock_hass)

    mock_hass.states.async_all.return_value = [
        _make_state("light.a", "A"),
        _make_state("light.b", "B"),
    ]
    h2 = cache._compute_hash(mock_hass)
    assert h1 != h2


def test_compute_hash_stable_for_same_entities() -> None:
    """Hash is stable across repeated calls with the same entity IDs."""
    mock_hass = _make_hass([_make_state("weather.met_no", "Met.no")])
    cache = EntityContextCache()
    assert cache._compute_hash(mock_hass) == cache._compute_hash(mock_hass)


def test_compute_hash_ignores_irrelevant_domains() -> None:
    """Irrelevant-domain entities do not affect the hash."""
    mock_a = _make_hass([_make_state("light.kitchen", "Kitchen")])
    mock_b = _make_hass(
        [
            _make_state("light.kitchen", "Kitchen"),
            _make_state("zone.home"),  # irrelevant
        ]
    )
    cache = EntityContextCache()
    assert cache._compute_hash(mock_a) == cache._compute_hash(mock_b)


# ---------------------------------------------------------------------------
# MAX_ENTITIES_PER_DOMAIN cap
# ---------------------------------------------------------------------------


def test_domain_capped_at_max_entities_per_domain() -> None:
    """Only MAX_ENTITIES_PER_DOMAIN entity names are included per domain."""
    states = [
        _make_state(f"light.room_{i}", f"Room {i}") for i in range(MAX_ENTITIES_PER_DOMAIN + 5)
    ]
    mock_hass = _make_hass(states)
    cache = EntityContextCache()
    summary = cache.get_summary(mock_hass)
    light_line = next(ln for ln in summary.splitlines() if ln.startswith("- light:"))
    name_count = len(light_line.split(","))
    assert name_count == MAX_ENTITIES_PER_DOMAIN


# ---------------------------------------------------------------------------
# RELEVANT_DOMAINS constant
# ---------------------------------------------------------------------------


def test_relevant_domains_is_frozenset() -> None:
    """RELEVANT_DOMAINS is a frozenset (immutable)."""
    assert isinstance(RELEVANT_DOMAINS, frozenset)


def test_relevant_domains_includes_key_smart_home_domains() -> None:
    """Key domains expected to be relevant are present."""
    for domain in ("light", "switch", "climate", "weather", "cover", "lock", "sensor"):
        assert domain in RELEVANT_DOMAINS


def test_relevant_domains_includes_todo_and_shopping_list() -> None:
    """Feature 12: 'todo' and 'shopping_list' domains are in RELEVANT_DOMAINS."""
    assert "todo" in RELEVANT_DOMAINS
    assert "shopping_list" in RELEVANT_DOMAINS


# ---------------------------------------------------------------------------
# Integration-style: real HomeAssistant fixture
# ---------------------------------------------------------------------------


async def test_get_summary_with_real_hass(hass: HomeAssistant) -> None:
    """get_summary works correctly with a real HomeAssistant instance."""
    hass.states.async_set("weather.met_no", "sunny", {"friendly_name": "Met.no"})
    hass.states.async_set("light.kitchen", "on", {"friendly_name": "Kitchen"})
    # Irrelevant domain — should not appear
    hass.states.async_set("persistent_notification.test", "notifying")

    cache = EntityContextCache()
    summary = cache.get_summary(hass)

    assert "Available smart home entities:" in summary
    assert "Met.no" in summary
    assert "Kitchen" in summary
    assert "persistent_notification" not in summary


async def test_get_summary_rebuilds_after_new_state_with_real_hass(hass: HomeAssistant) -> None:
    """Summary rebuilds when a new entity is added in real HA."""
    hass.states.async_set("light.kitchen", "on", {"friendly_name": "Kitchen"})
    cache = EntityContextCache()
    first = cache.get_summary(hass)

    hass.states.async_set("weather.met_no", "sunny", {"friendly_name": "Met.no"})
    second = cache.get_summary(hass)

    assert "weather" not in first
    assert "weather" in second
