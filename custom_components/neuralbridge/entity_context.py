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
        "switch",
        "vacuum",
        "water_heater",
        "weather",
    }
)

# Maximum entities listed per domain in the summary prompt.
# Keeps the appended context manageable on large installations.
MAX_ENTITIES_PER_DOMAIN: Final = 20


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
