"""Unit tests for NeuralBridge integration initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from custom_components.neuralbridge import async_reload_entry, async_setup_entry, async_unload_entry
from custom_components.neuralbridge.circuit_breaker import CircuitBreaker
from custom_components.neuralbridge.const import (
    DATA_CIRCUIT_BREAKER,
    DATA_ENTITY_CONTEXT,
    DATA_SESSION_MEMORY,
    DATA_STATISTICS,
    DOMAIN,
    SERVICE_CLEAR_CONVERSATION,
)
from custom_components.neuralbridge.entity_context import EntityContextCache
from custom_components.neuralbridge.statistics import AgentStatistics

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    from custom_components.neuralbridge.session_memory import SessionMemory


async def test_setup_entry(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> None:
    """Test setting up the integration."""
    # Use _setup (defined below) which patches async_forward_entry_setups so the
    # test doesn't need the full HA integration loader to be active.
    result = await _setup(hass, mock_config_entry)
    assert result
    assert DOMAIN in hass.data


async def test_unload_entry(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> None:
    """Test unloading the integration."""
    await _setup(hass, mock_config_entry)
    result = await _unload(hass, mock_config_entry)
    assert result
    assert mock_config_entry.entry_id not in hass.data[DOMAIN]


# ---------------------------------------------------------------------------
# Helpers — mock platform loading so unit tests don't need the full HA loader
# ---------------------------------------------------------------------------

_SETUP_PATCH = "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"
_UNLOAD_PATCH = "homeassistant.config_entries.ConfigEntries.async_unload_platforms"


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> bool:
    """Set up a NeuralBridge entry with platform loading mocked out."""
    entry.add_to_hass(hass)
    with patch(_SETUP_PATCH, return_value=None):
        return await async_setup_entry(hass, entry)


async def _unload(hass: HomeAssistant, entry: MockConfigEntry) -> bool:
    """Unload a NeuralBridge entry with platform unloading mocked out."""
    with patch(_UNLOAD_PATCH, return_value=True):
        return await async_unload_entry(hass, entry)


# ---------------------------------------------------------------------------
# clear_conversation service — registration
# ---------------------------------------------------------------------------


async def test_clear_conversation_service_registered(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """clear_conversation service is registered after async_setup_entry."""
    await _setup(hass, mock_config_entry)

    assert hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION)


async def test_clear_conversation_service_not_registered_twice(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Calling async_setup_entry twice does not double-register the service."""
    await _setup(hass, mock_config_entry)
    # A second setup attempt (e.g. reload) must not raise
    await _setup(hass, mock_config_entry)

    assert hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION)


# ---------------------------------------------------------------------------
# clear_conversation service — handler clears session memory
# ---------------------------------------------------------------------------


async def test_clear_conversation_service_clears_session(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Calling clear_conversation removes the given conversation_id from SessionMemory."""
    await _setup(hass, mock_config_entry)

    # Seed session memory directly via hass.data
    memory: SessionMemory = hass.data[DOMAIN][mock_config_entry.entry_id][DATA_SESSION_MEMORY]
    memory.add_turn("conv-to-clear", "hello", "world")
    assert memory.session_count() == 1

    # Act: call the service
    await hass.services.async_call(
        DOMAIN,
        SERVICE_CLEAR_CONVERSATION,
        {"conversation_id": "conv-to-clear"},
        blocking=True,
    )

    assert memory.session_count() == 0


async def test_clear_conversation_service_noop_for_unknown_id(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Calling clear_conversation with an unknown ID does not raise."""
    await _setup(hass, mock_config_entry)

    # Should not raise even with a conversation_id that has no session
    await hass.services.async_call(
        DOMAIN,
        SERVICE_CLEAR_CONVERSATION,
        {"conversation_id": "nonexistent-session"},
        blocking=True,
    )


# ---------------------------------------------------------------------------
# clear_conversation service — unregistered on last entry unload
# ---------------------------------------------------------------------------


async def test_clear_conversation_service_unregistered_on_unload(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Service is removed from hass when the last entry is unloaded."""
    await _setup(hass, mock_config_entry)
    assert hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION)

    await _unload(hass, mock_config_entry)

    assert not hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION)


# ---------------------------------------------------------------------------
# async_reload_entry
# ---------------------------------------------------------------------------


async def test_async_reload_entry_calls_async_reload(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_reload_entry delegates to hass.config_entries.async_reload."""
    with patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(hass, mock_config_entry)

    mock_reload.assert_called_once_with(mock_config_entry.entry_id)


# ---------------------------------------------------------------------------
# DATA_CIRCUIT_BREAKER stored in hass.data
# ---------------------------------------------------------------------------


async def test_setup_entry_stores_circuit_breaker(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_setup_entry stores a CircuitBreaker instance in hass.data."""
    await _setup(hass, mock_config_entry)

    entry_data = hass.data[DOMAIN][mock_config_entry.entry_id]
    assert DATA_CIRCUIT_BREAKER in entry_data
    assert isinstance(entry_data[DATA_CIRCUIT_BREAKER], CircuitBreaker)


async def test_setup_entry_stores_statistics(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_setup_entry stores an AgentStatistics instance in hass.data."""
    await _setup(hass, mock_config_entry)

    entry_data = hass.data[DOMAIN][mock_config_entry.entry_id]
    assert DATA_STATISTICS in entry_data
    assert isinstance(entry_data[DATA_STATISTICS], AgentStatistics)


async def test_setup_entry_stores_entity_context_cache(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_setup_entry stores an EntityContextCache instance in hass.data."""
    await _setup(hass, mock_config_entry)

    entry_data = hass.data[DOMAIN][mock_config_entry.entry_id]
    assert DATA_ENTITY_CONTEXT in entry_data
    assert isinstance(entry_data[DATA_ENTITY_CONTEXT], EntityContextCache)
