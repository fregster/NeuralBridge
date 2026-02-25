"""NeuralBridge sensor entities.

Provides two entity types:
- NeuralBridgeStatsSensor: one per config entry; overall routing request count.
- NeuralBridgeAgentSensor: one per configured agent; per-agent metrics and
  capability attributes (agent type, priority, guard rails, circuit breaker, etc.).

Updates are pushed from the conversation entity via the HA dispatcher rather
than polled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import (
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    CONF_AGENT_CACHE_ENABLED,
    CONF_AGENT_ENABLED,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_AGENTS,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_OLLAMA_MODEL,
    CONF_PRIORITY,
    DATA_CIRCUIT_BREAKER,
    DATA_STATISTICS,
    DEFAULT_AGENT_CACHE_ENABLED,
    DEFAULT_AGENT_ENABLED,
    DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    DOMAIN,
    PRIORITY_ROUTER,
    SIGNAL_STATS_UPDATED,
)

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from .circuit_breaker import CircuitBreaker
    from .statistics import AgentStatistics, AgentStats


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the NeuralBridge sensor entities.

    Creates one global statistics sensor and one per-agent sensor for every
    agent configured in the integration.

    Args:
        hass: Home Assistant instance.
        config_entry: The integration's config entry.
        async_add_entities: Callback to register new entities.
    """
    entry_data = hass.data[DOMAIN][config_entry.entry_id]
    statistics: AgentStatistics = entry_data[DATA_STATISTICS]
    circuit_breaker: CircuitBreaker = entry_data[DATA_CIRCUIT_BREAKER]
    agents: list[dict[str, Any]] = config_entry.data.get(CONF_AGENTS, [])

    entities: list[SensorEntity] = [NeuralBridgeStatsSensor(config_entry, statistics)]
    entities.extend(
        NeuralBridgeAgentSensor(config_entry, agent, statistics, circuit_breaker)
        for agent in agents
    )
    async_add_entities(entities)


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


class NeuralBridgeAgentSensor(SensorEntity):
    """Per-agent sensor exposing operational metrics and capability attributes.

    State: total routing requests routed to this agent since last HA restart.
    Attributes: agent type, priority, guard rails status, circuit breaker state,
                response time, success rate, queries per hour, and (for routers)
                block count and block rate.
    """

    _attr_native_unit_of_measurement = "requests"
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_should_poll = False

    def __init__(
        self,
        config_entry: ConfigEntry,
        agent_config: dict[str, Any],
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Initialise the per-agent sensor.

        Args:
            config_entry: The integration's config entry.
            agent_config: Configuration dict for this specific agent.
            statistics: Shared statistics tracker.
            circuit_breaker: Shared circuit breaker for reading per-agent state.
        """
        self._config_entry = config_entry
        self._agent_config = agent_config
        self._agent_id: str = agent_config.get("id", "")
        self._statistics = statistics
        self._circuit_breaker = circuit_breaker
        agent_name: str = agent_config.get(CONF_AGENT_NAME, "Unknown")
        is_router = agent_config.get(CONF_PRIORITY, 1) == PRIORITY_ROUTER
        self._attr_name = f"NeuralBridge {agent_name}"
        self._attr_unique_id = f"{config_entry.entry_id}_{self._agent_id}_agent"
        self._attr_icon = "mdi:filter-outline" if is_router else "mdi:robot"

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
        """Return the total number of requests routed to this agent.

        Returns:
            Request count for this agent, or 0 if no requests recorded yet.
        """
        stats: AgentStats | None = self._statistics.get_agent_stats(self._agent_id)
        return stats.requests if stats is not None else 0

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return agent metrics and capabilities as sensor attributes.

        Returns:
            Dictionary with operational metrics and static capability info.
        """
        agent_type: str = self._agent_config.get(CONF_AGENT_TYPE, "")
        priority: int = self._agent_config.get(CONF_PRIORITY, 50)
        is_router = priority == PRIORITY_ROUTER
        stats: AgentStats | None = self._statistics.get_agent_stats(self._agent_id)

        attrs: dict[str, Any] = {
            "agent_type": agent_type,
            "priority": priority,
            "is_router": is_router,
            "enabled": self._agent_config.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED),
            "guard_rails_enabled": self._agent_config.get(
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT, DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
            ),
            "cache_enabled": self._agent_config.get(
                CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
            ),
            "can_control_local_devices": agent_type == AGENT_TYPE_LOCAL_HA,
            "circuit_breaker": (
                "open" if self._circuit_breaker.is_open(self._agent_id) else "closed"
            ),
        }

        if agent_type == AGENT_TYPE_OLLAMA:
            attrs["model"] = self._agent_config.get(CONF_OLLAMA_MODEL, "")

        if stats is not None:
            attrs.update(
                {
                    "successes": stats.successes,
                    "failures": stats.failures,
                    "timeouts": stats.timeouts,
                    "success_rate": stats.success_rate,
                    "avg_response_time_ms": stats.avg_latency_ms,
                    "queries_per_hour": stats.queries_per_hour,
                }
            )
            if is_router:
                attrs["blocks"] = stats.blocks
                attrs["block_rate"] = stats.block_rate

        return attrs
