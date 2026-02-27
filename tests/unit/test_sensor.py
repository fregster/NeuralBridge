"""Unit tests for the sensor module."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.components.sensor import SensorStateClass
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.neuralbridge.agent_benchmark import AgentBenchmarker, BenchmarkStatus
from custom_components.neuralbridge.circuit_breaker import CircuitBreaker
from custom_components.neuralbridge.const import (
    AGENT_TYPE_INTEGRATED,
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
    DATA_PREFERENCE_MEMORY,
    DATA_STATISTICS,
    DOMAIN,
    SIGNAL_PREFERENCES_UPDATED,
    SIGNAL_STATS_UPDATED,
)
from custom_components.neuralbridge.preference_memory import PreferenceEntry, PreferenceMemory
from custom_components.neuralbridge.sensor import (
    NeuralBridgeAgentSensor,
    NeuralBridgePreferencesSensor,
    NeuralBridgeStatsSensor,
    async_setup_entry,
)
from custom_components.neuralbridge.statistics import AgentStatistics

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Return a mock config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge",
        data={},
        unique_id="test_sensor_entry",
    )


@pytest.fixture
def statistics() -> AgentStatistics:
    """Return a fresh AgentStatistics instance."""
    return AgentStatistics()


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    """Return a fresh CircuitBreaker instance."""
    return CircuitBreaker()


@pytest.fixture
def sensor(
    mock_config_entry: MockConfigEntry, statistics: AgentStatistics
) -> NeuralBridgeStatsSensor:
    """Return a NeuralBridgeStatsSensor instance."""
    return NeuralBridgeStatsSensor(mock_config_entry, statistics)


def _make_ollama_agent(
    agent_id: str = "agent-1",
    name: str = "Test Agent",
    priority: int = 50,
    model: str = "llama3",
    guard_rails: bool = True,
    cache: bool = True,
    enabled: bool = True,
) -> dict[str, Any]:
    """Return a minimal Ollama agent config dict for sensor tests."""
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_NAME: name,
        CONF_PRIORITY: priority,
        CONF_OLLAMA_MODEL: model,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: guard_rails,
        CONF_AGENT_CACHE_ENABLED: cache,
        CONF_AGENT_ENABLED: enabled,
    }


# ---------------------------------------------------------------------------
# NeuralBridgeStatsSensor tests (pre-existing, updated for new hass.data shape)
# ---------------------------------------------------------------------------


class TestNeuralBridgeStatsSensor:
    """Tests for NeuralBridgeStatsSensor entity."""

    def test_attr_name(self, sensor: NeuralBridgeStatsSensor) -> None:
        """Test sensor has the expected friendly name."""
        assert sensor._attr_name == "NeuralBridge Statistics"

    def test_unique_id(
        self,
        sensor: NeuralBridgeStatsSensor,
        mock_config_entry: MockConfigEntry,
    ) -> None:
        """Test sensor unique_id is derived from the config entry id."""
        assert sensor._attr_unique_id == f"{mock_config_entry.entry_id}_stats"

    def test_native_value_empty(self, sensor: NeuralBridgeStatsSensor) -> None:
        """Test native_value returns 0 when no requests have been recorded."""
        assert sensor.native_value == 0

    def test_native_value_reflects_statistics(
        self,
        sensor: NeuralBridgeStatsSensor,
        statistics: AgentStatistics,
    ) -> None:
        """Test native_value reflects the total_requests from statistics."""
        statistics.record_request("id-1", "Llama3")
        statistics.record_request("id-1", "Llama3")
        assert sensor.native_value == 2

    def test_extra_state_attributes_empty(self, sensor: NeuralBridgeStatsSensor) -> None:
        """Test extra_state_attributes is empty when no agents have been used."""
        assert sensor.extra_state_attributes == {}

    def test_extra_state_attributes_reflect_statistics(
        self,
        sensor: NeuralBridgeStatsSensor,
        statistics: AgentStatistics,
    ) -> None:
        """Test extra_state_attributes returns per-agent stats."""
        statistics.record_request("id-1", "Llama3")
        statistics.record_success("id-1", 200.0)
        attrs = sensor.extra_state_attributes
        assert "Llama3" in attrs
        assert attrs["Llama3"]["requests"] == 1
        assert attrs["Llama3"]["successes"] == 1

    def test_should_poll_false(self, sensor: NeuralBridgeStatsSensor) -> None:
        """Test sensor does not poll — it uses push updates via dispatcher."""
        assert sensor._attr_should_poll is False

    def test_icon(self, sensor: NeuralBridgeStatsSensor) -> None:
        """Test sensor has the expected icon."""
        assert sensor._attr_icon == "mdi:chart-bar"

    def test_state_class(self, sensor: NeuralBridgeStatsSensor) -> None:
        """Test sensor has TOTAL_INCREASING state class."""
        assert sensor._attr_state_class == SensorStateClass.TOTAL_INCREASING

    async def test_async_added_to_hass_subscribes_dispatcher(
        self,
        hass: HomeAssistant,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
    ) -> None:
        """Test async_added_to_hass subscribes to the stats dispatcher signal."""
        sensor = NeuralBridgeStatsSensor(mock_config_entry, statistics)
        sensor.hass = hass

        subscribed_signals: list[str] = []

        def fake_connect(hass_: HomeAssistant, signal: str, target: object) -> object:
            subscribed_signals.append(signal)
            return lambda: None  # unsub callback

        with patch(
            "custom_components.neuralbridge.sensor.async_dispatcher_connect",
            side_effect=fake_connect,
        ):
            await sensor.async_added_to_hass()

        expected_signal = SIGNAL_STATS_UPDATED.format(entry_id=mock_config_entry.entry_id)
        assert expected_signal in subscribed_signals

    def test_handle_stats_update_writes_state(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
    ) -> None:
        """Test _handle_stats_update calls async_write_ha_state."""
        sensor = NeuralBridgeStatsSensor(mock_config_entry, statistics)
        sensor.async_write_ha_state = MagicMock()  # type: ignore[assignment]

        sensor._handle_stats_update()

        sensor.async_write_ha_state.assert_called_once()


# ---------------------------------------------------------------------------
# async_setup_entry — creates global stats sensor + per-agent sensors
# ---------------------------------------------------------------------------


async def test_async_setup_entry_registers_sensors(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    statistics: AgentStatistics,
    circuit_breaker: CircuitBreaker,
) -> None:
    """async_setup_entry creates one NeuralBridgeStatsSensor and no agent sensors when no agents."""
    mock_config_entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})[mock_config_entry.entry_id] = {
        DATA_STATISTICS: statistics,
        DATA_CIRCUIT_BREAKER: circuit_breaker,
    }

    added: list = []
    await async_setup_entry(hass, mock_config_entry, added.append)

    entities = added[0]
    assert len(entities) == 1
    assert isinstance(entities[0], NeuralBridgeStatsSensor)


async def test_async_setup_entry_creates_per_agent_sensors(
    hass: HomeAssistant,
    statistics: AgentStatistics,
    circuit_breaker: CircuitBreaker,
) -> None:
    """async_setup_entry creates one per-agent sensor for each configured agent."""
    agent1 = _make_ollama_agent(agent_id="a1", name="Llama3")
    agent2 = _make_ollama_agent(agent_id="a2", name="Qwen")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [agent1, agent2]},
    )
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        DATA_STATISTICS: statistics,
        DATA_CIRCUIT_BREAKER: circuit_breaker,
    }

    added: list = []
    await async_setup_entry(hass, entry, added.append)

    entities = added[0]
    # 1 global stats sensor + 2 agent sensors
    assert len(entities) == 3
    assert isinstance(entities[0], NeuralBridgeStatsSensor)
    assert isinstance(entities[1], NeuralBridgeAgentSensor)
    assert isinstance(entities[2], NeuralBridgeAgentSensor)


# ---------------------------------------------------------------------------
# NeuralBridgeAgentSensor tests
# ---------------------------------------------------------------------------


class TestNeuralBridgeAgentSensor:
    """Tests for NeuralBridgeAgentSensor entity."""

    def test_attr_name(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test sensor name includes the agent name."""
        agent = _make_ollama_agent(name="Llama3")
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor._attr_name == "NeuralBridge Llama3"

    def test_unique_id(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test unique_id combines entry_id and agent_id."""
        agent = _make_ollama_agent(agent_id="abc-123", name="Llama3")
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor._attr_unique_id == f"{mock_config_entry.entry_id}_abc-123_agent"

    def test_icon_router(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test icon is filter-outline for router agents (priority=0)."""
        agent = _make_ollama_agent(priority=0)
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor._attr_icon == "mdi:filter-outline"

    def test_icon_processor(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test icon is robot for processing agents (priority>0)."""
        agent = _make_ollama_agent(priority=50)
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor._attr_icon == "mdi:robot"

    def test_state_class(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test sensor has TOTAL_INCREASING state class."""
        agent = _make_ollama_agent()
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor._attr_state_class == SensorStateClass.TOTAL_INCREASING

    def test_should_poll_false(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test sensor does not poll."""
        agent = _make_ollama_agent()
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor._attr_should_poll is False

    def test_native_value_no_stats(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test native_value returns 0 when no requests have been recorded."""
        agent = _make_ollama_agent(agent_id="agent-1")
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor.native_value == 0

    def test_native_value_reflects_agent_requests(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test native_value returns this agent's request count only."""
        statistics.record_request("agent-1", "Llama3")
        statistics.record_request("agent-1", "Llama3")
        statistics.record_request("agent-2", "Other")  # different agent — should not count
        agent = _make_ollama_agent(agent_id="agent-1")
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor.native_value == 2

    def test_extra_state_attributes_capabilities(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test extra_state_attributes includes static capability fields."""
        agent = _make_ollama_agent(
            agent_id="agent-1",
            name="Llama3",
            priority=10,
            model="llama3:8b",
            guard_rails=True,
            cache=False,
            enabled=True,
        )
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        attrs = sensor.extra_state_attributes

        assert attrs["agent_type"] == AGENT_TYPE_OLLAMA
        assert attrs["priority"] == 10
        assert attrs["is_router"] is False
        assert attrs["enabled"] is True
        assert attrs["guard_rails_enabled"] is True
        assert attrs["cache_enabled"] is False
        assert attrs["can_control_local_devices"] is False
        assert attrs["model"] == "llama3:8b"

    def test_extra_state_attributes_local_ha_can_control_devices(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test can_control_local_devices is True only for home_assistant agents."""
        agent = {
            "id": "ha-1",
            CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
            CONF_AGENT_NAME: "Home Assistant",
            CONF_PRIORITY: 5,
        }
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor.extra_state_attributes["can_control_local_devices"] is True

    def test_extra_state_attributes_existing_cannot_control_devices(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test can_control_local_devices is False for existing_integration agents."""
        agent = {
            "id": "ext-1",
            CONF_AGENT_TYPE: AGENT_TYPE_INTEGRATED,
            CONF_AGENT_NAME: "Gemini",
            CONF_PRIORITY: 20,
        }
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor.extra_state_attributes["can_control_local_devices"] is False

    def test_extra_state_attributes_no_model_for_non_ollama(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test model attribute is absent for non-Ollama agents."""
        agent = {
            "id": "ext-1",
            CONF_AGENT_TYPE: AGENT_TYPE_INTEGRATED,
            CONF_AGENT_NAME: "Gemini",
            CONF_PRIORITY: 20,
        }
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert "model" not in sensor.extra_state_attributes

    def test_extra_state_attributes_circuit_breaker_closed(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test circuit_breaker attribute is 'closed' when circuit is not tripped."""
        agent = _make_ollama_agent(agent_id="agent-1")
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor.extra_state_attributes["circuit_breaker"] == "closed"

    def test_extra_state_attributes_circuit_breaker_open(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test circuit_breaker attribute is 'open' when the circuit is tripped."""
        circuit_breaker._failure_threshold = 1
        circuit_breaker.record_failure("agent-1")
        agent = _make_ollama_agent(agent_id="agent-1")
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        assert sensor.extra_state_attributes["circuit_breaker"] == "open"

    def test_extra_state_attributes_includes_runtime_stats(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test extra_state_attributes includes runtime metrics when stats exist."""
        statistics.record_request("agent-1", "Llama3")
        statistics.record_success("agent-1", 300.0)
        agent = _make_ollama_agent(agent_id="agent-1")
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        attrs = sensor.extra_state_attributes

        assert attrs["successes"] == 1
        assert attrs["failures"] == 0
        assert attrs["timeouts"] == 0
        assert "success_rate" in attrs
        assert "avg_response_time_ms" in attrs
        assert "queries_per_hour" in attrs

    def test_extra_state_attributes_no_runtime_stats_when_no_requests(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test runtime stats keys are absent when no requests recorded for this agent."""
        agent = _make_ollama_agent(agent_id="never-used")
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        attrs = sensor.extra_state_attributes
        assert "successes" not in attrs
        assert "failures" not in attrs
        assert "avg_response_time_ms" not in attrs

    def test_extra_state_attributes_router_includes_block_stats(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test blocks and block_rate are included for router agents (priority=0)."""
        statistics.record_request("router-1", "Router")
        statistics.record_block("router-1")
        agent = _make_ollama_agent(agent_id="router-1", priority=0)
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        attrs = sensor.extra_state_attributes

        assert attrs["is_router"] is True
        assert attrs["blocks"] == 1
        assert "block_rate" in attrs

    def test_extra_state_attributes_processor_excludes_block_stats(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test blocks and block_rate are absent for non-router agents."""
        statistics.record_request("proc-1", "Processor")
        agent = _make_ollama_agent(agent_id="proc-1", priority=10)
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        attrs = sensor.extra_state_attributes

        assert attrs["is_router"] is False
        assert "blocks" not in attrs
        assert "block_rate" not in attrs

    async def test_async_added_to_hass_subscribes_dispatcher(
        self,
        hass: HomeAssistant,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test async_added_to_hass subscribes to the stats dispatcher signal."""
        agent = _make_ollama_agent()
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        sensor.hass = hass

        subscribed_signals: list[str] = []

        def fake_connect(hass_: HomeAssistant, signal: str, target: object) -> object:
            subscribed_signals.append(signal)
            return lambda: None

        with patch(
            "custom_components.neuralbridge.sensor.async_dispatcher_connect",
            side_effect=fake_connect,
        ):
            await sensor.async_added_to_hass()

        expected_signal = SIGNAL_STATS_UPDATED.format(entry_id=mock_config_entry.entry_id)
        assert expected_signal in subscribed_signals

    def test_handle_stats_update_writes_state(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Test _handle_stats_update calls async_write_ha_state."""
        agent = _make_ollama_agent()
        sensor = NeuralBridgeAgentSensor(mock_config_entry, agent, statistics, circuit_breaker)
        sensor.async_write_ha_state = MagicMock()  # type: ignore[assignment]

        sensor._handle_stats_update()

        sensor.async_write_ha_state.assert_called_once()

    def test_extra_state_attributes_benchmark_sub_dict(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """extra_state_attributes includes 'benchmark' dict when benchmarker has a profile."""
        agent = _make_ollama_agent(agent_id="bench-agent-1")
        hass_mock = MagicMock()
        benchmarker = AgentBenchmarker(hass_mock)
        config = {
            "id": "bench-agent-1",
            CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
            CONF_AGENT_NAME: "Bench Agent",
            CONF_PRIORITY: 10,
        }
        profile = benchmarker.ensure_profile(config)
        profile.status = BenchmarkStatus.COMPLETE  # type: ignore[attr-defined]
        profile.benchmark_timestamp = 1700000000.0

        sensor = NeuralBridgeAgentSensor(
            mock_config_entry, agent, statistics, circuit_breaker, benchmarker
        )
        attrs = sensor.extra_state_attributes

        assert "benchmark" in attrs
        assert attrs["benchmark"]["status"] == BenchmarkStatus.COMPLETE.value

    def test_extra_state_attributes_benchmark_timestamp_overflow(
        self,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """benchmark_timestamp that raises OSError falls back to str()."""
        agent = _make_ollama_agent(agent_id="bench-agent-2")
        hass_mock = MagicMock()
        benchmarker = AgentBenchmarker(hass_mock)
        config = {
            "id": "bench-agent-2",
            CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
            CONF_AGENT_NAME: "Bench Agent 2",
            CONF_PRIORITY: 10,
        }
        profile = benchmarker.ensure_profile(config)
        profile.status = BenchmarkStatus.COMPLETE  # type: ignore[attr-defined]
        # Very large timestamp — will overflow datetime.datetime.fromtimestamp on some platforms
        profile.benchmark_timestamp = 99999999999999.9

        sensor = NeuralBridgeAgentSensor(
            mock_config_entry, agent, statistics, circuit_breaker, benchmarker
        )
        attrs = sensor.extra_state_attributes

        assert "benchmark" in attrs
        # The fallback str() path should at least produce a string last_ts
        assert attrs["benchmark"]["status"] == BenchmarkStatus.COMPLETE.value


# ---------------------------------------------------------------------------
# NeuralBridgePreferencesSensor tests
# ---------------------------------------------------------------------------


@pytest.fixture
def pref_memory_mock() -> MagicMock:
    """Return a mock PreferenceMemory with two entries (one confirmed, one pending)."""
    now = time.time()
    confirmed = PreferenceEntry(
        key="news_source",
        value="BBC",
        category="source",
        confirmed=True,
        confidence=1.0,
        suggestion_count=1,
        last_suggested=now - 3600,
        created_at=now - 7200,
        updated_at=now - 3600,
    )
    pending = PreferenceEntry(
        key="time_format",
        value="24h",
        category="format",
        confirmed=False,
        confidence=0.7,
        suggestion_count=0,
        last_suggested=0.0,
        created_at=now - 1800,
        updated_at=now - 1800,
    )
    mock = MagicMock(spec=PreferenceMemory)
    mock.all_confirmed.return_value = [confirmed]
    mock.all_entries.return_value = [confirmed, pending]
    mock.max_entries = 25
    return mock


@pytest.fixture
def prefs_sensor(
    mock_config_entry: MockConfigEntry, pref_memory_mock: MagicMock
) -> NeuralBridgePreferencesSensor:
    """Return a NeuralBridgePreferencesSensor with a mock PreferenceMemory."""
    return NeuralBridgePreferencesSensor(mock_config_entry, pref_memory_mock)


async def test_async_setup_entry_creates_preferences_sensor_when_memory_present(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    statistics: AgentStatistics,
    circuit_breaker: CircuitBreaker,
    pref_memory_mock: MagicMock,
) -> None:
    """async_setup_entry creates a NeuralBridgePreferencesSensor when memory is present."""
    mock_config_entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})[mock_config_entry.entry_id] = {
        DATA_STATISTICS: statistics,
        DATA_CIRCUIT_BREAKER: circuit_breaker,
        DATA_PREFERENCE_MEMORY: pref_memory_mock,
    }

    added: list = []
    await async_setup_entry(hass, mock_config_entry, added.append)

    entities = added[0]
    pref_sensors = [e for e in entities if isinstance(e, NeuralBridgePreferencesSensor)]
    assert len(pref_sensors) == 1


def test_preferences_sensor_native_value(
    prefs_sensor: NeuralBridgePreferencesSensor,
    pref_memory_mock: MagicMock,
) -> None:
    """native_value returns the count of confirmed preferences."""
    assert prefs_sensor.native_value == 1


def test_preferences_sensor_extra_state_attributes_counts(
    prefs_sensor: NeuralBridgePreferencesSensor,
) -> None:
    """extra_state_attributes exposes total_stored, pending_count, and max_entries."""
    attrs = prefs_sensor.extra_state_attributes
    assert attrs["total_stored"] == 2
    assert attrs["pending_count"] == 1
    assert attrs["max_entries"] == 25


def test_preferences_sensor_extra_state_attributes_preferences_list(
    prefs_sensor: NeuralBridgePreferencesSensor,
) -> None:
    """extra_state_attributes 'preferences' is a list with one entry per stored pref."""
    attrs = prefs_sensor.extra_state_attributes
    prefs = attrs["preferences"]
    assert len(prefs) == 2
    # All expected keys are present in each entry
    required_keys = {
        "key",
        "value",
        "category",
        "confirmed",
        "confidence",
        "suggestion_count",
        "created_at",
        "updated_at",
    }
    for pref in prefs:
        assert required_keys.issubset(pref.keys())


async def test_preferences_sensor_async_added_to_hass_subscribes(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    prefs_sensor: NeuralBridgePreferencesSensor,
) -> None:
    """async_added_to_hass subscribes to the SIGNAL_PREFERENCES_UPDATED dispatcher."""
    from homeassistant.helpers.dispatcher import async_dispatcher_send  # noqa: PLC0415

    mock_config_entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})[mock_config_entry.entry_id] = {}

    prefs_sensor.hass = hass  # type: ignore[assignment]

    with patch.object(prefs_sensor, "async_write_ha_state") as mock_write:
        await prefs_sensor.async_added_to_hass()
        signal = SIGNAL_PREFERENCES_UPDATED.format(entry_id=mock_config_entry.entry_id)
        async_dispatcher_send(hass, signal)
        await hass.async_block_till_done()
        mock_write.assert_called()
