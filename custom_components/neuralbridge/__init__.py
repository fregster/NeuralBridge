"""The NeuralBridge integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import voluptuous as vol
from homeassistant.const import Platform
from homeassistant.helpers import config_validation as cv

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant, ServiceCall

from .circuit_breaker import CircuitBreaker
from .const import (
    CONF_RESPONSE_CACHE_ENABLED,
    CONF_RESPONSE_CACHE_TTL,
    DATA_CIRCUIT_BREAKER,
    DATA_RESPONSE_CACHE,
    DATA_SESSION_MEMORY,
    DATA_STATISTICS,
    DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_RESPONSE_CACHE_ENABLED,
    DEFAULT_RESPONSE_CACHE_TTL,
    DOMAIN,
    SERVICE_CLEAR_CONVERSATION,
)
from .languages_loader import async_preload_all_languages
from .response_cache import ResponseCache
from .session_memory import SessionMemory
from .statistics import AgentStatistics

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION, Platform.SENSOR]

CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)

_CLEAR_CONVERSATION_SCHEMA = vol.Schema({vol.Required("conversation_id"): cv.string})


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up NeuralBridge from a config entry."""
    _LOGGER.debug("Setting up NeuralBridge integration")

    # Create shared objects and store alongside config data
    statistics = AgentStatistics()
    merged_config = {**entry.data, **entry.options}
    cache_enabled = merged_config.get(CONF_RESPONSE_CACHE_ENABLED, DEFAULT_RESPONSE_CACHE_ENABLED)
    cache_ttl = merged_config.get(CONF_RESPONSE_CACHE_TTL, DEFAULT_RESPONSE_CACHE_TTL)
    response_cache = ResponseCache(enabled=cache_enabled, ttl_seconds=cache_ttl)
    session_memory = SessionMemory()
    circuit_breaker = CircuitBreaker(
        failure_threshold=DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        cooldown_seconds=DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
    )

    # Pre-load language files in an executor so later synchronous get_string /
    # list_available_languages calls never block the event loop.
    await async_preload_all_languages(hass)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        **entry.data,
        **entry.options,
        DATA_STATISTICS: statistics,
        DATA_RESPONSE_CACHE: response_cache,
        DATA_SESSION_MEMORY: session_memory,
        DATA_CIRCUIT_BREAKER: circuit_breaker,
    }

    # Set up platforms (conversation + sensor)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register the clear_conversation service (once, at domain level)
    if not hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION):
        _register_clear_conversation_service(hass)

    # Register update listener for when config entry is updated
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True


def _register_clear_conversation_service(hass: HomeAssistant) -> None:
    """Register the clear_conversation service on the NeuralBridge domain.

    The service iterates all active NeuralBridge config entries and clears
    the given conversation_id from each entry's SessionMemory.
    """

    def _handle_clear_conversation(call: ServiceCall) -> None:
        """Clear a conversation session from NeuralBridge session memory.

        Args:
            call: The service call, containing ``conversation_id`` in its data.
        """
        conversation_id: str = call.data["conversation_id"]
        _LOGGER.debug("Clearing conversation session: %s", conversation_id)
        for entry_data in hass.data.get(DOMAIN, {}).values():
            memory: SessionMemory | None = entry_data.get(DATA_SESSION_MEMORY)
            if memory is not None:
                memory.clear_session(conversation_id)

    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_CONVERSATION,
        _handle_clear_conversation,
        schema=_CLEAR_CONVERSATION_SCHEMA,
    )


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.debug("Unloading NeuralBridge integration")

    # Unload the conversation platform
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
        # Remove service when the last entry is unloaded
        if not hass.data[DOMAIN]:
            hass.services.async_remove(DOMAIN, SERVICE_CLEAR_CONVERSATION)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the config entry when it changes."""
    await hass.config_entries.async_reload(entry.entry_id)
