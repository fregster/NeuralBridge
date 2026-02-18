"""Unit tests for the sensor module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.neuralbridge.const import DATA_STATISTICS, DOMAIN, SIGNAL_STATS_UPDATED
from custom_components.neuralbridge.sensor import NeuralBridgeStatsSensor
from custom_components.neuralbridge.statistics import AgentStatistics


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
def sensor(
    mock_config_entry: MockConfigEntry, statistics: AgentStatistics
) -> NeuralBridgeStatsSensor:
    """Return a NeuralBridgeStatsSensor instance."""
    return NeuralBridgeStatsSensor(mock_config_entry, statistics)


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
        from homeassistant.components.sensor import SensorStateClass

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

    async def test_handle_stats_update_writes_state(
        self,
        hass: HomeAssistant,
        mock_config_entry: MockConfigEntry,
        statistics: AgentStatistics,
    ) -> None:
        """Test _handle_stats_update calls async_write_ha_state."""
        sensor = NeuralBridgeStatsSensor(mock_config_entry, statistics)
        sensor.hass = hass
        sensor.async_write_ha_state = MagicMock()  # type: ignore[assignment]

        sensor._handle_stats_update()

        sensor.async_write_ha_state.assert_called_once()


async def test_async_setup_entry_registers_sensor(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    statistics: AgentStatistics,
) -> None:
    """async_setup_entry creates a NeuralBridgeStatsSensor and calls async_add_entities."""
    from custom_components.neuralbridge.sensor import async_setup_entry

    mock_config_entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})[mock_config_entry.entry_id] = {
        DATA_STATISTICS: statistics,
    }

    added: list = []
    await async_setup_entry(hass, mock_config_entry, added.append)

    # async_add_entities is called as async_add_entities([sensor]), so
    # added[0] is the list [sensor] — the entity is at added[0][0].
    assert len(added) == 1
    entities = added[0]
    assert len(entities) == 1
    assert isinstance(entities[0], NeuralBridgeStatsSensor)
