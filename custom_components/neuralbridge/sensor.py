"""NeuralBridge statistics sensor entity.

Exposes a single sensor entity per config entry that reports the total number
of routing requests as its state and per-agent breakdowns as attributes.
Updates are pushed from the conversation entity via the HA dispatcher rather
than polled.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DATA_STATISTICS, DOMAIN, SIGNAL_STATS_UPDATED
from .statistics import AgentStatistics


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the NeuralBridge statistics sensor.

    Args:
        hass: Home Assistant instance.
        config_entry: The integration's config entry.
        async_add_entities: Callback to register new entities.
    """
    statistics: AgentStatistics = hass.data[DOMAIN][config_entry.entry_id][DATA_STATISTICS]
    async_add_entities([NeuralBridgeStatsSensor(config_entry, statistics)])


class NeuralBridgeStatsSensor(SensorEntity):
    """Sensor entity reporting NeuralBridge routing statistics.

    State: total routing requests since last HA restart.
    Attributes: per-agent breakdown (requests, successes, failures, timeouts,
                avg_latency_ms, success_rate).
    """

    _attr_icon = "mdi:chart-bar"
    _attr_native_unit_of_measurement = "requests"
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_should_poll = False

    def __init__(self, config_entry: ConfigEntry, statistics: AgentStatistics) -> None:
        """Initialise the sensor.

        Args:
            config_entry: The integration's config entry.
            statistics: Shared statistics tracker written to by the conversation entity.
        """
        self._config_entry = config_entry
        self._statistics = statistics
        self._attr_name = "NeuralBridge Statistics"
        self._attr_unique_id = f"{config_entry.entry_id}_stats"

    async def async_added_to_hass(self) -> None:
        """Subscribe to dispatcher signal when added to Home Assistant."""
        signal = SIGNAL_STATS_UPDATED.format(entry_id=self._config_entry.entry_id)
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                signal,
                self._handle_stats_update,
            )
        )

    @callback
    def _handle_stats_update(self) -> None:
        """Handle a statistics update notification from the conversation entity."""
        self.async_write_ha_state()

    @property
    def native_value(self) -> int:
        """Return the total number of routing requests as the sensor state.

        Returns:
            Total request count across all agents.
        """
        return self._statistics.total_requests()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return per-agent statistics as sensor attributes.

        Returns:
            Dictionary keyed by agent name with stats dicts as values.
        """
        return self._statistics.get_all()
