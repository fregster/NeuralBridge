"""The NeuralBridge integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.const import Platform
from homeassistant.helpers import config_validation as cv

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant, ServiceCall

from .agent_benchmark import AgentBenchmarker
from .circuit_breaker import CircuitBreaker
from .const import (
    CONF_AGENTS,
    CONF_BENCHMARK_WARM_UP_DELAY,
    CONF_PREFERENCE_MAX_ENTRIES,
    CONF_RESPONSE_CACHE_ENABLED,
    CONF_RESPONSE_CACHE_SEMANTIC,
    CONF_RESPONSE_CACHE_TTL,
    CONF_SEMANTIC_CACHE_TTL,
    DATA_BENCHMARKER,
    DATA_CIRCUIT_BREAKER,
    DATA_ENTITY_CONTEXT,
    DATA_PREFERENCE_MEMORY,
    DATA_RESPONSE_CACHE,
    DATA_SESSION_MEMORY,
    DATA_STATISTICS,
    DEFAULT_BENCHMARK_WARM_UP_DELAY,
    DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_PREFERENCE_MAX_ENTRIES,
    DEFAULT_RESPONSE_CACHE_ENABLED,
    DEFAULT_RESPONSE_CACHE_SEMANTIC,
    DEFAULT_RESPONSE_CACHE_TTL,
    DEFAULT_SEMANTIC_CACHE_TTL,
    DOMAIN,
    EVENT_PREFERENCE_LEARNED,
    SERVICE_CLEAR_CONVERSATION,
    SERVICE_CLEAR_PREFERENCES,
    SERVICE_MANAGE_PREFERENCE,
    SERVICE_RUN_BENCHMARK,
)
from .entity_context import EntityContextCache
from .languages_loader import async_preload_all_languages
from .preference_memory import (
    PREF_CATEGORY_FORMAT,
    PREF_CATEGORY_LOCATION,
    PREF_CATEGORY_SOURCE,
    PreferenceMemory,
)
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
    cache_semantic = merged_config.get(
        CONF_RESPONSE_CACHE_SEMANTIC, DEFAULT_RESPONSE_CACHE_SEMANTIC
    )
    cache_semantic_ttl = merged_config.get(CONF_SEMANTIC_CACHE_TTL, DEFAULT_SEMANTIC_CACHE_TTL)
    response_cache = ResponseCache(
        enabled=cache_enabled,
        ttl_seconds=cache_ttl,
        semantic=cache_semantic,
        semantic_ttl_seconds=cache_semantic_ttl,
    )
    session_memory = SessionMemory()
    circuit_breaker = CircuitBreaker(
        failure_threshold=DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        cooldown_seconds=DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
    )

    # Pre-load language files in an executor so later synchronous get_string /
    # list_available_languages calls never block the event loop.
    await async_preload_all_languages(hass)

    entity_context = EntityContextCache()

    # Feature 14 — Agent Benchmark Profiling
    warm_up_delay = merged_config.get(CONF_BENCHMARK_WARM_UP_DELAY, DEFAULT_BENCHMARK_WARM_UP_DELAY)
    benchmarker = AgentBenchmarker(hass, warm_up_delay_seconds=int(warm_up_delay))
    await benchmarker.async_load()

    # Feature 15 — Adaptive Preference Learning
    max_entries = int(
        merged_config.get(CONF_PREFERENCE_MAX_ENTRIES, DEFAULT_PREFERENCE_MAX_ENTRIES)
    )
    preference_memory = PreferenceMemory(hass, entry.entry_id, max_entries=max_entries)
    await preference_memory.async_load()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        **entry.data,
        **entry.options,
        DATA_STATISTICS: statistics,
        DATA_RESPONSE_CACHE: response_cache,
        DATA_SESSION_MEMORY: session_memory,
        DATA_CIRCUIT_BREAKER: circuit_breaker,
        DATA_ENTITY_CONTEXT: entity_context,
        DATA_BENCHMARKER: benchmarker,
        DATA_PREFERENCE_MEMORY: preference_memory,
    }

    # Schedule background benchmark runs for all configured agents
    agents: list[dict[str, Any]] = merged_config.get(CONF_AGENTS, [])
    for agent_config in agents:
        agent_id: str = agent_config.get("id", "")
        if not agent_id:
            continue
        profile = benchmarker.ensure_profile(agent_config)
        if profile.re_benchmark_on_save:
            hass.async_create_task(benchmarker.async_schedule_benchmark(agent_id, agent_config))

    # Set up platforms (conversation + sensor)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services (once, at domain level)
    if not hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION):
        _register_clear_conversation_service(hass)
    if not hass.services.has_service(DOMAIN, SERVICE_RUN_BENCHMARK):
        _register_run_benchmark_service(hass)
    if not hass.services.has_service(DOMAIN, SERVICE_CLEAR_PREFERENCES):
        _register_clear_preferences_service(hass)
    if not hass.services.has_service(DOMAIN, SERVICE_MANAGE_PREFERENCE):
        _register_manage_preference_service(hass)

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


def _register_run_benchmark_service(hass: HomeAssistant) -> None:
    """Register the run_benchmark service on the NeuralBridge domain.

    The service triggers an immediate benchmark run for the specified agent
    across all active NeuralBridge config entries.
    """
    _run_benchmark_schema = vol.Schema({vol.Required("agent_id"): cv.string})

    async def _handle_run_benchmark(call: ServiceCall) -> None:
        """Trigger an immediate benchmark run for an agent.

        Args:
            call: The service call, containing ``agent_id`` in its data.
        """
        agent_id: str = call.data["agent_id"]
        _LOGGER.debug("run_benchmark service called for agent_id=%s", agent_id)
        for entry_data in hass.data.get(DOMAIN, {}).values():
            nb: AgentBenchmarker | None = entry_data.get(DATA_BENCHMARKER)
            if nb is None:
                continue
            profile = nb.get_profile(agent_id)
            if profile is None:
                continue
            # Find the agent config matching the id
            agents: list[dict[str, Any]] = []
            for key, value in entry_data.items():
                if key == CONF_AGENTS:
                    agents = value
                    break
            agent_config = next((a for a in agents if a.get("id") == agent_id), None)
            if agent_config is not None:
                await nb.async_schedule_benchmark(agent_id, agent_config)

    hass.services.async_register(
        DOMAIN,
        SERVICE_RUN_BENCHMARK,
        _handle_run_benchmark,
        schema=_run_benchmark_schema,
    )


def _register_clear_preferences_service(hass: HomeAssistant) -> None:
    """Register the clear_preferences service on the NeuralBridge domain.

    Wipes all stored preferences (confirmed and unconfirmed) for every active
    NeuralBridge config entry.
    """

    async def _handle_clear_preferences(call: ServiceCall) -> None:  # noqa: ARG001
        """Clear all stored preferences.

        Args:
            call: The service call (no parameters required).
        """
        _LOGGER.debug("clear_preferences service called")
        for entry_data in hass.data.get(DOMAIN, {}).values():
            pm: PreferenceMemory | None = entry_data.get(DATA_PREFERENCE_MEMORY)
            if pm is None:
                continue
            removed = await pm.clear()
            _LOGGER.info("Cleared %d preference(s)", removed)

    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_PREFERENCES,
        _handle_clear_preferences,
        schema=vol.Schema({}),
    )


def _register_manage_preference_service(hass: HomeAssistant) -> None:
    """Register the manage_preference service on the NeuralBridge domain.

    Allows add, edit, or remove of a single preference entry across all active
    NeuralBridge config entries.
    """
    _valid_categories = vol.In([PREF_CATEGORY_SOURCE, PREF_CATEGORY_LOCATION, PREF_CATEGORY_FORMAT])
    _manage_schema = vol.Schema(
        {
            vol.Required("action"): vol.In(["add", "edit", "remove"]),
            vol.Required("key"): cv.string,
            vol.Optional("value"): cv.string,
            vol.Optional("category"): _valid_categories,
        }
    )

    async def _handle_manage_preference(call: ServiceCall) -> None:
        """Add, edit, or remove a preference entry.

        Args:
            call: The service call with ``action``, ``key``, and optionally
                ``value`` / ``category``.
        """
        action: str = call.data["action"]
        key: str = call.data["key"]
        value: str = call.data.get("value", "")
        category: str = call.data.get("category", PREF_CATEGORY_SOURCE)

        _LOGGER.debug("manage_preference service: action=%s key=%s", action, key)
        for entry_data in hass.data.get(DOMAIN, {}).values():
            pm: PreferenceMemory | None = entry_data.get(DATA_PREFERENCE_MEMORY)
            if pm is None:
                continue
            if action == "remove":
                await pm.reject(key)
            else:
                await pm.upsert(key, value, category, confidence=1.0, confirmed=True)
                hass.bus.async_fire(
                    EVENT_PREFERENCE_LEARNED,
                    {"key": key, "value": value, "category": category, "action": action},
                )

    hass.services.async_register(
        DOMAIN,
        SERVICE_MANAGE_PREFERENCE,
        _handle_manage_preference,
        schema=_manage_schema,
    )


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.debug("Unloading NeuralBridge integration")

    # Cancel any pending benchmark tasks before unloading
    entry_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
    benchmarker: AgentBenchmarker | None = entry_data.get(DATA_BENCHMARKER)
    if benchmarker is not None:
        benchmarker.cancel_all_pending()

    # Unload the conversation platform
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
        # Remove services when the last entry is unloaded
        if not hass.data[DOMAIN]:
            hass.services.async_remove(DOMAIN, SERVICE_CLEAR_CONVERSATION)
            for svc in (
                SERVICE_RUN_BENCHMARK,
                SERVICE_CLEAR_PREFERENCES,
                SERVICE_MANAGE_PREFERENCE,
            ):
                if hass.services.has_service(DOMAIN, svc):
                    hass.services.async_remove(DOMAIN, svc)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the config entry when it changes."""
    await hass.config_entries.async_reload(entry.entry_id)
