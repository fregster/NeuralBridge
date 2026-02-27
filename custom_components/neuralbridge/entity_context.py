"""Entity context cache for NeuralBridge router classification.

Builds a compact summary of available smart home entities and caches it,
keyed by a hash of the current entity set.  The cache automatically
rebuilds whenever the entity set changes, ensuring the router classification
prompt always reflects the current device landscape.

The summary is appended to the router classification prompt so that small
local models can correctly identify whether a query targets a locally
available entity (e.g. a weather integration) rather than treating it as a
general knowledge question.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

# Domains relevant for smart home context.
# Intentionally excludes HA-internal domains (persistent_notification, etc.).
RELEVANT_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "alarm_control_panel",
        "automation",
        "binary_sensor",
        "button",
        "camera",
        "climate",
        "cover",
        "fan",
        "humidifier",
        "input_boolean",
        "input_number",
        "input_select",
        "light",
        "lock",
        "media_player",
        "number",
        "scene",
        "script",
        "select",
        "sensor",
        "shopping_list",
        "switch",
        "todo",
        "vacuum",
        "water_heater",
        "weather",
    }
)

# Maximum entities listed per domain in the summary prompt.
# Keeps the appended context manageable on large installations.
MAX_ENTITIES_PER_DOMAIN: Final = 20

# Domains where the current state VALUE carries semantic meaning for routing.
# Only these domains appear in the live sensor-values block appended to the
# router classification prompt.  Other domains (lights, switches, etc.) have
# states like "on" / "off" that add noise rather than useful routing signal.
_SENSOR_DOMAINS: Final[frozenset[str]] = frozenset({"sensor", "binary_sensor"})

# Maximum entries in the live sensor-values block.
# This is a hard ceiling against truly pathological installs (thousands of
# sensors).  In normal use every sensor is included — the cap should never
# be hit.  A sensor entry is ~30-50 chars; even 500 sensors is only ~6,000
# tokens, well within every LLM context window used with HA.
MAX_SENSOR_VALUES: Final = 500


class EntityContextCache:
    """Lazy-rebuilding cache of a compact smart home entity summary.

    The summary is injected into router classification prompts so small
    models can correctly identify whether a query targets a locally
    available entity (e.g. a weather integration) rather than treating
    it as a general knowledge question.

    The cache is keyed by a stable MD5 hash of the sorted entity IDs for
    all relevant domains.  :meth:`get_summary` is cheap on repeated calls
    (just a hash comparison) and only rebuilds when the entity set changes.
    """

    def __init__(self) -> None:
        """Initialise an empty EntityContextCache."""
        self._summary: str = ""
        self._entity_hash: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_summary(self, hass: HomeAssistant) -> str:
        """Return the entity summary, rebuilding if the entity set has changed.

        The returned string is empty when no relevant entities exist.

        Args:
            hass: The Home Assistant core instance.

        Returns:
            Compact multi-line summary of available entities, or ``""`` when
            no relevant entities are registered.
        """
        current_hash = self._compute_hash(hass)
        if current_hash != self._entity_hash:
            self._summary = self._build_summary(hass)
            self._entity_hash = current_hash
        return self._summary

    def get_sensor_names(self, hass: HomeAssistant) -> str:
        """Return a fresh (uncached) list of sensor names and units — no live values.

        Intended for the router classification prompt only.  Providing just the
        sensor name and unit lets the router identify *which* sensors exist and
        decide whether a query is answerable locally, without exposing the live
        readings that could vary between the routing decision and the final answer.

        Only ``sensor`` and ``binary_sensor`` domains are included.  Entities
        reporting ``unavailable`` or ``unknown`` are excluded.  Output is sorted
        alphabetically and capped at :data:`MAX_SENSOR_VALUES` entries.

        Args:
            hass: The Home Assistant core instance.

        Returns:
            Multi-line block starting with ``"Available sensors:\\n"`` and one
            ``"- Name [unit]"`` or ``"- Name"`` line per sensor, or ``""`` when
            no eligible sensor states exist.
        """
        entries: list[str] = []
        for state in hass.states.async_all():
            if state.domain not in _SENSOR_DOMAINS:
                continue
            if state.state in ("unavailable", "unknown"):
                continue
            name = str(state.attributes.get("friendly_name") or state.entity_id)
            unit: str = str(state.attributes.get("unit_of_measurement") or "")
            # Include entity_id so the router can populate relevant_sensors with
            # exact entity IDs rather than fuzzy friendly names.
            label = (
                f"- {state.entity_id}: {name} [{unit}]" if unit else f"- {state.entity_id}: {name}"
            )
            entries.append(label)
        if not entries:
            return ""
        return "Available sensors:\n" + "\n".join(sorted(entries)[:MAX_SENSOR_VALUES])

    def get_sensor_values_for(
        self, hass: HomeAssistant, entity_ids: list[str] | tuple[str, ...]
    ) -> str:
        """Return live values for a specific set of sensor entity IDs.

        Used by the processing pipeline after the router has identified which
        sensors are relevant to the current query via ``relevant_sensors`` in
        the router JSON response.  Only the nominated sensors are fetched,
        keeping the injection block compact and the LLM focus narrow.

        Entities reporting ``unavailable`` or ``unknown`` are excluded.  An
        empty string is returned when none of the requested entities produce
        usable state.

        Args:
            hass:       The Home Assistant core instance.
            entity_ids: Iterable of entity ID strings to look up.

        Returns:
            Multi-line block starting with ``"Current sensor values:\\n"`` and
            one ``"- Name: value unit"`` line per sensor, or ``""`` when no
            eligible states are found.
        """
        entries: list[str] = []
        for entity_id in entity_ids:
            state = hass.states.get(entity_id)
            if state is None:
                continue
            if state.state in ("unavailable", "unknown"):
                continue
            name = str(state.attributes.get("friendly_name") or state.entity_id)
            unit_raw: str = str(state.attributes.get("unit_of_measurement") or "")
            value_str = f"{state.state} {unit_raw}".strip() if unit_raw else state.state
            entries.append(f"- {name}: {value_str}")
        if not entries:
            return ""
        return "Current sensor values:\n" + "\n".join(entries)

    def get_sensor_values(self, hass: HomeAssistant) -> str:
        """Return a fresh (uncached) snapshot of sensor and binary_sensor values.

        This method is intentionally NOT cached — it always reads live state so
        that the router classification prompt reflects current readings.  It is
        appended to the router prompt alongside the cached entity-name summary,
        giving small local models enough semantic signal to route questions like
        "is it raining?" or "what is the river level?" to the correct agent even
        when the question phrasing differs from the sensor's friendly name.

        Only ``sensor`` and ``binary_sensor`` domains are included; other domains
        (lights, switches, etc.) have on/off states that add noise rather than
        useful routing signal.  Entities reporting ``unavailable`` or ``unknown``
        are excluded.  Output is capped at :data:`MAX_SENSOR_VALUES` entries.

        This method is also reachable from
        :meth:`~custom_components.neuralbridge.conversation.NeuralBridgeAgent.\
        _render_ha_context` via the ``{ha_sensor_states}`` prompt token, which
        allows users who connect Ollama **directly** (``AGENT_TYPE_OLLAMA``) to
        opt-in to receiving sensor data in the system prompt.  It must **never**
        be called automatically for ``LOCAL_HA`` or ``EXISTING`` agents because
        those agents receive sensor data natively through HA's own conversation
        infrastructure.

        Args:
            hass: The Home Assistant core instance.

        Returns:
            Multi-line block starting with ``"Current sensor values:\\n"`` and
            one ``"- Name: value unit"`` line per sensor, sorted alphabetically,
            or ``""`` when no eligible sensor states exist.
        """
        entries: list[str] = []
        for state in hass.states.async_all():
            if state.domain not in _SENSOR_DOMAINS:
                continue
            if state.state in ("unavailable", "unknown"):
                continue
            name = str(state.attributes.get("friendly_name") or state.entity_id)
            unit: str = str(state.attributes.get("unit_of_measurement") or "")
            value_str = f"{state.state} {unit}".strip() if unit else state.state
            entries.append(f"- {name}: {value_str}")
        if not entries:
            return ""
        # Sort alphabetically BEFORE capping so that the cap is deterministic
        # regardless of HA's internal entity registration order.
        return "Current sensor values:\n" + "\n".join(sorted(entries)[:MAX_SENSOR_VALUES])

    def invalidate(self) -> None:
        """Force the cache to rebuild on the next :meth:`get_summary` call.

        Useful when an integration is added or removed and the caller wants
        to guarantee a fresh summary without waiting for the hash check.
        """
        self._entity_hash = ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_hash(self, hass: HomeAssistant) -> str:
        """Return a stable hash of the current relevant entity ID set.

        Args:
            hass: The Home Assistant core instance.

        Returns:
            Hex-encoded MD5 digest of the sorted, comma-joined entity IDs.
        """
        entity_ids = sorted(
            state.entity_id for state in hass.states.async_all() if state.domain in RELEVANT_DOMAINS
        )
        return hashlib.md5(",".join(entity_ids).encode()).hexdigest()  # noqa: S324

    def _build_summary(self, hass: HomeAssistant) -> str:
        """Build a compact entity summary grouped by domain.

        Each domain lists up to :data:`MAX_ENTITIES_PER_DOMAIN` entity friendly
        names (falling back to entity_id when no friendly_name is set).  Domains
        are sorted alphabetically; entity names within each domain are also sorted.

        Args:
            hass: The Home Assistant core instance.

        Returns:
            Multi-line summary string, or ``""`` when no relevant entities exist.
        """
        by_domain: dict[str, list[str]] = {}
        for state in hass.states.async_all():
            if state.domain not in RELEVANT_DOMAINS:
                continue
            name = str(state.attributes.get("friendly_name") or state.entity_id)
            by_domain.setdefault(state.domain, []).append(name)

        if not by_domain:
            return ""

        lines = [
            f"- {domain}: {', '.join(sorted(by_domain[domain])[:MAX_ENTITIES_PER_DOMAIN])}"
            for domain in sorted(by_domain)
        ]
        return "Available smart home entities:\n" + "\n".join(lines)
