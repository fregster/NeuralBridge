"""Unit tests for EntityContextCache."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from custom_components.neuralbridge.entity_context import (
    _SENSOR_DOMAINS,
    MAX_ENTITIES_PER_DOMAIN,
    MAX_SENSOR_VALUES,
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


# ---------------------------------------------------------------------------
# Helpers for get_sensor_values tests
# ---------------------------------------------------------------------------


def _make_sensor_state(
    entity_id: str,
    state_value: str,
    friendly_name: str | None = None,
    unit: str | None = None,
) -> MagicMock:
    """Return a minimal fake state object with a state value and optional unit."""
    s = MagicMock()
    s.entity_id = entity_id
    s.domain = entity_id.split(".", maxsplit=1)[0]
    s.state = state_value
    attrs: dict[str, str] = {}
    if friendly_name is not None:
        attrs["friendly_name"] = friendly_name
    if unit is not None:
        attrs["unit_of_measurement"] = unit
    s.attributes = attrs
    return s


# ---------------------------------------------------------------------------
# _SENSOR_DOMAINS and MAX_SENSOR_VALUES constants
# ---------------------------------------------------------------------------


def test_sensor_domains_is_frozenset() -> None:
    """_SENSOR_DOMAINS is a frozenset (immutable)."""
    assert isinstance(_SENSOR_DOMAINS, frozenset)


def test_sensor_domains_contains_sensor_and_binary_sensor() -> None:
    """_SENSOR_DOMAINS contains 'sensor' and 'binary_sensor'."""
    assert "sensor" in _SENSOR_DOMAINS
    assert "binary_sensor" in _SENSOR_DOMAINS


def test_sensor_domains_does_not_contain_light_or_switch() -> None:
    """_SENSOR_DOMAINS excludes on/off domains that add noise for routing."""
    assert "light" not in _SENSOR_DOMAINS
    assert "switch" not in _SENSOR_DOMAINS


def test_max_sensor_values_is_positive_int() -> None:
    """MAX_SENSOR_VALUES is a positive integer."""
    assert isinstance(MAX_SENSOR_VALUES, int)
    assert MAX_SENSOR_VALUES > 0


# ---------------------------------------------------------------------------
# get_sensor_values — basic behaviour
# ---------------------------------------------------------------------------


def test_get_sensor_values_returns_empty_when_no_states() -> None:
    """Returns '' when hass has no states."""
    mock_hass = _make_hass([])
    cache = EntityContextCache()
    assert cache.get_sensor_values(mock_hass) == ""


def test_get_sensor_values_returns_empty_when_no_sensor_domains() -> None:
    """Returns '' when no sensor/binary_sensor entities exist."""
    mock_hass = _make_hass(
        [
            _make_sensor_state("light.kitchen", "on", "Kitchen"),
            _make_sensor_state("switch.fan", "off", "Fan"),
        ]
    )
    cache = EntityContextCache()
    assert cache.get_sensor_values(mock_hass) == ""


def test_get_sensor_values_contains_header() -> None:
    """Result starts with 'Current sensor values:'."""
    mock_hass = _make_hass([_make_sensor_state("sensor.temp", "12.6", "Air temperature", "°C")])
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    assert result.startswith("Current sensor values:")


def test_get_sensor_values_includes_value_and_unit() -> None:
    """Each entry includes the state value and unit_of_measurement."""
    mock_hass = _make_hass([_make_sensor_state("sensor.temp", "12.6", "Air temperature", "°C")])
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    assert "Air temperature: 12.6 °C" in result


def test_get_sensor_values_no_unit_omits_unit() -> None:
    """When no unit_of_measurement, the value is shown without a trailing space."""
    mock_hass = _make_hass([_make_sensor_state("sensor.precip_type", "none", "Precipitation Type")])
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    assert "Precipitation Type: none" in result
    # Should not have a trailing space after value
    assert "none " not in result


def test_get_sensor_values_uses_friendly_name() -> None:
    """Friendly name is used in the output when available."""
    mock_hass = _make_hass([_make_sensor_state("sensor.s1", "3.43", "River level downstream", "m")])
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    assert "River level downstream: 3.43 m" in result


def test_get_sensor_values_falls_back_to_entity_id() -> None:
    """Falls back to entity_id when no friendly_name attribute is set."""
    mock_hass = _make_hass([_make_sensor_state("sensor.river_level", "3.43", None, "m")])
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    assert "sensor.river_level: 3.43 m" in result


def test_get_sensor_values_includes_binary_sensor() -> None:
    """binary_sensor entities are included in the sensor values block."""
    mock_hass = _make_hass([_make_sensor_state("binary_sensor.rain", "on", "Rain detected")])
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    assert "Rain detected: on" in result


def test_get_sensor_values_excludes_unavailable() -> None:
    """States with value 'unavailable' are excluded."""
    mock_hass = _make_hass(
        [
            _make_sensor_state("sensor.broken", "unavailable", "Broken sensor"),
            _make_sensor_state("sensor.ok", "5.0", "Good sensor", "m"),
        ]
    )
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    assert "Broken sensor" not in result
    assert "Good sensor: 5.0 m" in result


def test_get_sensor_values_excludes_unknown() -> None:
    """States with value 'unknown' are excluded."""
    mock_hass = _make_hass([_make_sensor_state("sensor.mystery", "unknown", "Mystery sensor")])
    cache = EntityContextCache()
    assert cache.get_sensor_values(mock_hass) == ""


def test_get_sensor_values_sorted_alphabetically() -> None:
    """Output lines are sorted alphabetically by entry text."""
    mock_hass = _make_hass(
        [
            _make_sensor_state("sensor.z", "1", "Zebra sensor"),
            _make_sensor_state("sensor.a", "2", "Apple sensor"),
            _make_sensor_state("sensor.m", "3", "Mango sensor"),
        ]
    )
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    lines = [ln for ln in result.splitlines() if ln.startswith("- ")]
    assert lines == sorted(lines)


def test_get_sensor_values_capped_at_max_sensor_values() -> None:
    """Output is capped at MAX_SENSOR_VALUES entries (hard ceiling for extreme installs)."""
    states = [
        _make_sensor_state(f"sensor.s{i:04d}", str(i), f"Sensor {i:04d}")
        for i in range(MAX_SENSOR_VALUES + 10)
    ]
    mock_hass = _make_hass(states)
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    lines = [ln for ln in result.splitlines() if ln.startswith("- ")]
    assert len(lines) == MAX_SENSOR_VALUES


def test_get_sensor_values_excludes_non_sensor_domains() -> None:
    """Only sensor and binary_sensor domains appear in the values block."""
    mock_hass = _make_hass(
        [
            _make_sensor_state("light.bedroom", "on", "Bedroom Light"),
            _make_sensor_state("climate.hall", "heat", "Hall Thermostat"),
            _make_sensor_state("sensor.temp", "21.0", "Temperature", "°C"),
        ]
    )
    cache = EntityContextCache()
    result = cache.get_sensor_values(mock_hass)
    assert "Bedroom Light" not in result
    assert "Hall Thermostat" not in result
    assert "Temperature: 21.0 °C" in result


def test_get_sensor_values_always_fresh_not_cached() -> None:
    """get_sensor_values always reads live state (is not cached like get_summary)."""
    state = _make_sensor_state("sensor.temp", "10.0", "Temperature", "°C")
    mock_hass = _make_hass([state])
    cache = EntityContextCache()

    result1 = cache.get_sensor_values(mock_hass)
    assert "Temperature: 10.0 °C" in result1

    # Simulate a state change without adding entities (entity set unchanged)
    state.state = "20.0"
    mock_hass.states.async_all.return_value = [state]

    result2 = cache.get_sensor_values(mock_hass)
    assert "Temperature: 20.0 °C" in result2
    assert result1 != result2


# ---------------------------------------------------------------------------
# get_sensor_names — produces a compact list of available sensor entity IDs
# ---------------------------------------------------------------------------


def test_get_sensor_names_with_unit() -> None:
    """get_sensor_names includes entity_id, friendly name, and unit in brackets."""
    mock_hass = _make_hass(
        [_make_sensor_state("sensor.precipitation_type", "none", "Precipitation Type", "")]
    )
    # override friendly_name via the helper
    states = [_make_sensor_state("sensor.precip", "none", "Precip Type")]
    states[0].attributes = {"friendly_name": "Precip Type", "unit_of_measurement": "mm/h"}
    mock_hass = _make_hass(states)

    cache = EntityContextCache()
    result = cache.get_sensor_names(mock_hass)

    assert result.startswith("Available sensors:")
    assert "sensor.precip" in result
    assert "mm/h" in result


def test_get_sensor_names_without_unit() -> None:
    """get_sensor_names omits brackets when no unit_of_measurement is present."""
    states = [_make_sensor_state("sensor.motion", "on", "Motion Sensor")]
    mock_hass = _make_hass(states)

    cache = EntityContextCache()
    result = cache.get_sensor_names(mock_hass)

    assert "sensor.motion" in result
    assert "[" not in result  # no brackets when no unit


def test_get_sensor_names_excludes_unavailable() -> None:
    """get_sensor_names excludes sensors with state 'unavailable' or 'unknown'."""
    states = [
        _make_sensor_state("sensor.online", "23.5", "Online Sensor"),
        _make_sensor_state("sensor.offline", "unavailable", "Offline Sensor"),
        _make_sensor_state("sensor.unknown", "unknown", "Unknown Sensor"),
    ]
    mock_hass = _make_hass(states)

    cache = EntityContextCache()
    result = cache.get_sensor_names(mock_hass)

    assert "sensor.online" in result
    assert "sensor.offline" not in result
    assert "sensor.unknown" not in result


def test_get_sensor_names_excludes_non_sensor_domains() -> None:
    """get_sensor_names only includes sensor and binary_sensor domains."""
    states = [
        _make_sensor_state("sensor.temp", "20.0", "Temperature"),
        _make_sensor_state("binary_sensor.motion", "on", "Motion"),
        _make_sensor_state("light.ceiling", "on", "Ceiling Light"),
    ]
    mock_hass = _make_hass(states)

    cache = EntityContextCache()
    result = cache.get_sensor_names(mock_hass)

    assert "sensor.temp" in result
    assert "binary_sensor.motion" in result
    assert "light.ceiling" not in result


def test_get_sensor_names_empty_returns_empty_string() -> None:
    """get_sensor_names returns '' when no eligible sensors exist."""
    mock_hass = _make_hass([_make_sensor_state("sensor.unavail", "unavailable", "U")])

    cache = EntityContextCache()
    assert cache.get_sensor_names(mock_hass) == ""


# ---------------------------------------------------------------------------
# get_sensor_values_for — fetches live values for specific entity IDs
# ---------------------------------------------------------------------------


def _make_hass_with_get(states: list[MagicMock]) -> MagicMock:
    """Return a mock hass where states.get(entity_id) works correctly."""
    mock_hass = MagicMock()
    state_map = {s.entity_id: s for s in states}
    mock_hass.states.get.side_effect = state_map.get
    return mock_hass


def test_get_sensor_values_for_returns_values() -> None:
    """get_sensor_values_for returns a formatted block for the requested entities."""
    state = _make_sensor_state("sensor.rain", "0.5", "Rainfall", "mm/h")
    mock_hass = _make_hass_with_get([state])

    cache = EntityContextCache()
    result = cache.get_sensor_values_for(mock_hass, ["sensor.rain"])

    assert result.startswith("Current sensor values:")
    assert "Rainfall: 0.5 mm/h" in result


def test_get_sensor_values_for_without_unit() -> None:
    """get_sensor_values_for formats 'Name: value' when no unit is present."""
    state = _make_sensor_state("sensor.motion", "on", "Motion Sensor")
    mock_hass = _make_hass_with_get([state])

    cache = EntityContextCache()
    result = cache.get_sensor_values_for(mock_hass, ["sensor.motion"])

    assert "Motion Sensor: on" in result


def test_get_sensor_values_for_excludes_unavailable() -> None:
    """get_sensor_values_for excludes unavailable and unknown states."""
    states = [
        _make_sensor_state("sensor.ok", "21.0", "OK", "°C"),
        _make_sensor_state("sensor.bad", "unavailable", "Bad"),
        _make_sensor_state("sensor.unk", "unknown", "Unknown"),
    ]
    mock_hass = _make_hass_with_get(states)

    cache = EntityContextCache()
    result = cache.get_sensor_values_for(mock_hass, ["sensor.ok", "sensor.bad", "sensor.unk"])

    assert "OK" in result
    assert "Bad" not in result
    assert "Unknown" not in result


def test_get_sensor_values_for_entity_not_found() -> None:
    """get_sensor_values_for silently skips entity IDs not in hass.states."""
    mock_hass = _make_hass_with_get([])  # empty — all states.get() return None

    cache = EntityContextCache()
    result = cache.get_sensor_values_for(mock_hass, ["sensor.nonexistent"])

    assert result == ""


def test_get_sensor_values_for_empty_list_returns_empty() -> None:
    """get_sensor_values_for returns '' when given an empty entity list."""
    cache = EntityContextCache()
    result = cache.get_sensor_values_for(MagicMock(), [])
    assert result == ""
