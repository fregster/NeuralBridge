"""Unit tests for NeuralBridge conversation agent."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.conversation import (
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.dispatcher import async_dispatcher_send
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.neuralbridge.circuit_breaker import CircuitBreaker
from custom_components.neuralbridge.const import (
    AGENT_TYPE_EXISTING,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    AGENT_TYPE_WEB_SEARCH,
    COMPOUND_COMMAND_SEPARATOR,
    CONF_AGENT_ASSIST_MODE,
    CONF_AGENT_CACHE_ENABLED,
    CONF_AGENT_ENABLED,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_AGENTS,
    CONF_ANNOUNCE_MEDIA_PLAYERS,
    CONF_ENABLE_HOME_CONTROL,
    CONF_ENTITY_ID,
    CONF_FORCE_RESPONSE_LANGUAGE,
    CONF_GUARD_RAIL_ACTION,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_GUARD_RAIL_RULES,
    CONF_HIGH_STAKES_DOMAINS,
    CONF_HIGH_STAKES_ENABLED,
    CONF_HIGH_STAKES_SECRET,
    CONF_HIGH_STAKES_SECRET_ENABLED,
    CONF_IS_ROUTER,
    CONF_MAX_RETRIES,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_PRIORITY,
    CONF_RESPONSE_VERBOSITY,
    CONF_RETRY_BASE_DELAY,
    CONF_ROUTER_CUSTOM_PROMPT,
    CONF_ROUTER_FALLBACK,
    CONF_ROUTER_LOG_LEVEL,
    CONF_SEARCH_API_KEY,
    CONF_SEARCH_PROVIDER,
    CONF_SEARCH_RESULT_COUNT,
    CONF_SPLIT_COMPOUND_COMMANDS,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DATA_CIRCUIT_BREAKER,
    DEFAULT_ROUTER_COMPLEXITY,
    DOMAIN,
    EVENT_ANNOUNCE_SENT,
    EVENT_GUARD_RAIL_TRIGGERED,
    EVENT_HIGH_STAKES_TRIGGERED,
    FALLBACK_RESPONSE,
    GUARD_RAIL_ACTION_BLOCK,
    GUARD_RAIL_ACTION_NOTIFY_ASK,
    GUARD_RAIL_ACTION_WARN,
    GUARD_RAIL_BLOCKED_RESPONSE,
    MAX_COMPOUND_FRAGMENTS,
    NO_AGENTS_RESPONSE,
    ROUTER_CONFIDENCE_LOW,
    ROUTER_FALLBACK_BLOCK,
    ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
    ROUTER_FALLBACK_SKIP_ROUTING,
    ROUTER_LOG_LEVEL_COMPLEXITY,
    ROUTER_LOG_LEVEL_DEBUG,
    ROUTER_LOG_LEVEL_DEBUG_QUERY,
    ROUTER_LOG_LEVEL_NONE,
    ROUTER_SKIP_ROUTING_COMPLEXITY,
    SEARCH_PROVIDER_BRAVE,
    SIGNAL_STATS_UPDATED,
    VERBOSITY_BRIEF,
    VERBOSITY_NORMAL,
    VERBOSITY_VERBOSE,
)
from custom_components.neuralbridge.conversation import (
    NeuralBridgeAgent,
    RouterDecision,
    _apply_router_decision,
    _extract_announce_text,
    _get_language_name,
    _parse_router_response,
    _split_compound_input,
    _truncate_to_first_sentence,
    async_setup_entry,
)
from custom_components.neuralbridge.entity_context import EntityContextCache
from custom_components.neuralbridge.guard_rail import GuardRailResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(
    text: str = "Hello",
    conversation_id: str | None = None,
) -> ConversationInput:
    """Return a minimal ConversationInput for routing tests."""
    return ConversationInput(
        text=text,
        context=Context(),
        conversation_id=conversation_id,
        device_id=None,
        language="en",
        satellite_id=None,
        agent_id=None,
    )


def _make_ollama_agent(
    priority: int = 50,
    agent_id: str = "agent-1",
    name: str = "Test Ollama",
    enabled: bool = True,
    cache_enabled: bool = True,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Return a minimal Ollama agent config dict."""
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_ENABLED: enabled,
        CONF_AGENT_NAME: name,
        CONF_PRIORITY: priority,
        CONF_OLLAMA_URL: "http://localhost:11434",
        CONF_OLLAMA_MODEL: "llama3",
        CONF_TIMEOUT: 30,
        CONF_AGENT_CACHE_ENABLED: cache_enabled,
        CONF_SYSTEM_PROMPT: system_prompt,
    }


def _entry_with_agents(*agents: dict[str, Any]) -> MockConfigEntry:
    """Return a MockConfigEntry whose data contains the given agent list."""
    return MockConfigEntry(domain=DOMAIN, data={CONF_AGENTS: list(agents)})


def _entry_with_guard_rails(*agents: dict[str, Any]) -> MockConfigEntry:
    """Return a MockConfigEntry with guard rails enabled and block action."""
    return MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: list(agents),
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_ACTION: GUARD_RAIL_ACTION_BLOCK,
        },
    )


# ---------------------------------------------------------------------------
# Test 1 — agent initialisation
# ---------------------------------------------------------------------------


def test_agent_initialization(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> None:
    """NeuralBridgeAgent stores name and unique_id from the config entry."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)

    assert agent._attr_name == "NeuralBridge"
    assert agent._attr_unique_id == mock_config_entry.entry_id


# ---------------------------------------------------------------------------
# Test 2 — async_process: no agents configured → NO_AGENTS_RESPONSE
# ---------------------------------------------------------------------------


async def test_async_process_no_agents_returns_no_agents_response(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_process() returns NO_AGENTS_RESPONSE when agents list is empty."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)  # fixture has CONF_AGENTS: []
    result = await agent.async_process(_make_input())

    assert result.response.speech["plain"]["speech"] == NO_AGENTS_RESPONSE


# ---------------------------------------------------------------------------
# Test 3 — async_process: single agent succeeds → its result is returned
# ---------------------------------------------------------------------------


async def test_async_process_single_agent_success(hass: HomeAssistant) -> None:
    """async_process() returns the agent result when one agent succeeds."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    expected = conv_agent._create_result("Agent said hello")

    with patch.object(
        conv_agent, "_try_agent_with_tracking", new_callable=AsyncMock, return_value=expected
    ):
        result = await conv_agent.async_process(_make_input())

    assert result.response.speech["plain"]["speech"] == "Agent said hello"


# ---------------------------------------------------------------------------
# Test 4 — async_process: priority ordering — lower number tried first
# ---------------------------------------------------------------------------


async def test_async_process_priority_order_low_number_first(
    hass: HomeAssistant,
) -> None:
    """Agents are tried in ascending priority-number order (1 before 99)."""
    agent_p99 = _make_ollama_agent(priority=99, agent_id="p99", name="Low Priority")
    agent_p1 = _make_ollama_agent(priority=1, agent_id="p1", name="High Priority")

    # Deliberately add p99 first so the list order differs from priority order.
    entry = _entry_with_agents(agent_p99, agent_p1)
    conv_agent = NeuralBridgeAgent(hass, entry)

    call_order: list[str] = []

    def tracking_try(agent_config: dict[str, Any], user_input: ConversationInput):
        call_order.append(agent_config["id"])
        return conv_agent._create_result(f"from {agent_config['id']}")

    with patch.object(conv_agent, "_try_agent_with_tracking", side_effect=tracking_try):
        await conv_agent.async_process(_make_input())

    assert call_order[0] == "p1"  # priority 1 must be tried before priority 99


# ---------------------------------------------------------------------------
# Test 5 — async_process: first agent fails → falls back to second
# ---------------------------------------------------------------------------


async def test_async_process_first_agent_fails_tries_second(
    hass: HomeAssistant,
) -> None:
    """When the first agent returns None, the next agent is tried."""
    entry = _entry_with_agents(
        _make_ollama_agent(priority=10, agent_id="first"),
        _make_ollama_agent(priority=20, agent_id="second"),
    )
    conv_agent = NeuralBridgeAgent(hass, entry)

    def mock_try(agent_config: dict[str, Any], user_input: ConversationInput):
        if agent_config["id"] == "first":
            return None  # first agent fails
        return conv_agent._create_result("Second agent answered")

    with patch.object(conv_agent, "_try_agent_with_tracking", side_effect=mock_try):
        result = await conv_agent.async_process(_make_input())

    assert result.response.speech["plain"]["speech"] == "Second agent answered"


# ---------------------------------------------------------------------------
# Test 6 — async_process: all agents fail → FALLBACK_RESPONSE
# ---------------------------------------------------------------------------


async def test_async_process_all_agents_fail_returns_fallback(
    hass: HomeAssistant,
) -> None:
    """async_process() returns FALLBACK_RESPONSE when every agent returns None."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    with patch.object(
        conv_agent, "_try_agent_with_tracking", new_callable=AsyncMock, return_value=None
    ):
        result = await conv_agent.async_process(_make_input())

    assert result.response.speech["plain"]["speech"] == FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# Test 7 — _create_result: speech text is set correctly
# ---------------------------------------------------------------------------


def test_create_result_speech_text(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> None:
    """_create_result() stores the given text in the plain speech slot."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    result = agent._create_result("Test response text")

    assert result.response.speech["plain"]["speech"] == "Test response text"
    assert result.conversation_id is None


# ---------------------------------------------------------------------------
# Test 8 — _create_error_result: error message is set correctly
# ---------------------------------------------------------------------------


def test_create_error_result_speech_text(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_create_error_result() stores the error message in the plain speech slot."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    result = agent._create_error_result("Something went wrong")

    assert result.response.speech["plain"]["speech"] == "Something went wrong"
    assert result.conversation_id is None


# ---------------------------------------------------------------------------
# Test 9 — async_process: disabled agent is filtered out (#11)
# ---------------------------------------------------------------------------


async def test_async_process_disabled_agent_is_skipped(hass: HomeAssistant) -> None:
    """A disabled agent is never passed to _try_agent_with_tracking."""
    disabled = _make_ollama_agent(priority=10, agent_id="disabled", enabled=False)
    entry = _entry_with_agents(disabled)
    conv_agent = NeuralBridgeAgent(hass, entry)

    with patch.object(conv_agent, "_try_agent_with_tracking", new_callable=AsyncMock) as mock_try:
        result = await conv_agent.async_process(_make_input())

    # disabled agent list → no agents → NO_AGENTS_RESPONSE; _try_agent_with_tracking never called
    mock_try.assert_not_called()
    assert result.response.speech["plain"]["speech"] == NO_AGENTS_RESPONSE


async def test_async_process_enabled_agent_tried_disabled_skipped(hass: HomeAssistant) -> None:
    """Disabled agents are skipped; the enabled agent is tried."""
    disabled = _make_ollama_agent(priority=10, agent_id="disabled", enabled=False)
    enabled = _make_ollama_agent(priority=20, agent_id="enabled", enabled=True)
    entry = _entry_with_agents(disabled, enabled)
    conv_agent = NeuralBridgeAgent(hass, entry)

    tried_ids: list[str] = []

    def mock_try(agent_config: dict[str, Any], user_input: ConversationInput):
        tried_ids.append(agent_config["id"])
        return conv_agent._create_result("ok")

    with patch.object(conv_agent, "_try_agent_with_tracking", side_effect=mock_try):
        await conv_agent.async_process(_make_input())

    assert "disabled" not in tried_ids
    assert "enabled" in tried_ids


# ---------------------------------------------------------------------------
# Test 10 — async_process: response cache hit returns early (#14)
# ---------------------------------------------------------------------------


async def test_async_process_cache_hit_returns_immediately(hass: HomeAssistant) -> None:
    """A cache hit bypasses all agents and returns the cached response."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    # Pre-fill the cache
    conv_agent._response_cache.store("Hello", "Cached response", "TestAgent")

    with patch.object(conv_agent, "_try_agent_with_tracking", new_callable=AsyncMock) as mock_try:
        result = await conv_agent.async_process(_make_input("Hello"))

    mock_try.assert_not_called()
    assert result.response.speech["plain"]["speech"] == "Cached response"


# ---------------------------------------------------------------------------
# Test 11 — async_process: response stored in cache on success (#14)
# ---------------------------------------------------------------------------


async def test_async_process_response_stored_in_cache_on_success(
    hass: HomeAssistant,
) -> None:
    """Successful agent responses are stored in the response cache."""
    agent_cfg = _make_ollama_agent(cache_enabled=True)
    entry = _entry_with_agents(agent_cfg)
    conv_agent = NeuralBridgeAgent(hass, entry)

    success_result = conv_agent._create_result("Fresh response")

    # Patch _try_agent (one level below _try_agent_with_tracking) so that
    # _handle_successful_result → _maybe_cache_response still runs.
    with (
        patch.object(
            conv_agent,
            "_try_agent",
            new_callable=AsyncMock,
            return_value=(success_result, False),
        ),
        patch.object(conv_agent, "_check_guardrails", new_callable=AsyncMock, return_value=None),
    ):
        await conv_agent.async_process(_make_input("unique query"))

    cached = conv_agent._response_cache.get("unique query")
    assert cached == "Fresh response"


# ---------------------------------------------------------------------------
# Test 12 — async_process: agent cache disabled → not stored (#14)
# ---------------------------------------------------------------------------


async def test_async_process_agent_cache_disabled_not_stored(
    hass: HomeAssistant,
) -> None:
    """When agent cache is disabled the response is not written to the cache."""
    agent_cfg = _make_ollama_agent(cache_enabled=False)
    entry = _entry_with_agents(agent_cfg)
    conv_agent = NeuralBridgeAgent(hass, entry)

    with patch.object(
        conv_agent,
        "_try_agent_with_tracking",
        new_callable=AsyncMock,
        return_value=conv_agent._create_result("Uncacheable response"),
    ):
        await conv_agent.async_process(_make_input("no-cache query"))

    cached = conv_agent._response_cache.get("no-cache query")
    assert cached is None


# ---------------------------------------------------------------------------
# Test 13 — _try_agent_with_tracking: circuit-tripped agent skipped (#7)
# ---------------------------------------------------------------------------


async def test_try_agent_with_tracking_circuit_open_returns_none(
    hass: HomeAssistant,
) -> None:
    """_try_agent_with_tracking returns None when the agent's circuit is open."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    # Trip the circuit breaker
    conv_agent._circuit_breaker._states["agent-1"] = MagicMock(tripped=True, trip_time=float("inf"))
    # Make is_open return True unconditionally
    with patch.object(conv_agent._circuit_breaker, "is_open", return_value=True):
        result = await conv_agent._try_agent_with_tracking(_make_ollama_agent(), _make_input())

    assert result is None


# ---------------------------------------------------------------------------
# Test 14 — _try_agent_with_tracking: records timeout (#7, #8)
# ---------------------------------------------------------------------------


async def test_try_agent_with_tracking_records_timeout(hass: HomeAssistant) -> None:
    """A timed-out agent call records timeout in circuit breaker and statistics."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [_make_ollama_agent()], CONF_MAX_RETRIES: 0},
    )
    conv_agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = _make_ollama_agent()

    with patch.object(conv_agent, "_try_agent", new_callable=AsyncMock, return_value=(None, True)):
        result = await conv_agent._try_agent_with_tracking(agent_cfg, _make_input())

    assert result is None
    stats = conv_agent._statistics.get_all()
    assert stats["Test Ollama"]["timeouts"] == 1


# ---------------------------------------------------------------------------
# Test 15 — _try_agent_with_tracking: records failure (#7, #8)
# ---------------------------------------------------------------------------


async def test_try_agent_with_tracking_records_failure(hass: HomeAssistant) -> None:
    """A failed (non-timeout) agent call records failure in stats and circuit breaker."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [_make_ollama_agent()], CONF_MAX_RETRIES: 0},
    )
    conv_agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = _make_ollama_agent()

    with patch.object(conv_agent, "_try_agent", new_callable=AsyncMock, return_value=(None, False)):
        result = await conv_agent._try_agent_with_tracking(agent_cfg, _make_input())

    assert result is None
    stats = conv_agent._statistics.get_all()
    assert stats["Test Ollama"]["failures"] == 1


# ---------------------------------------------------------------------------
# Test 16 — _handle_confirmation_check: yes/no paths (#guard rails)
# ---------------------------------------------------------------------------


async def test_handle_confirmation_check_non_yes_no_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Non-yes/no text returns None (no pending check)."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    result = await conv_agent._handle_confirmation_check(_make_input("tell me a joke"))
    assert result is None


async def test_handle_confirmation_check_yes_no_no_pending_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """'yes' with no pending guard rail response returns None."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    result = await conv_agent._handle_confirmation_check(_make_input("yes"))
    assert result is None


async def test_handle_confirmation_check_yes_returns_pending_response(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """'yes' with a pending guard rail response returns the stored response."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    mock_gr = MagicMock()
    await conv_agent._guard_rail_cache.store_pending_response("conv-1", "Safe content", mock_gr)

    result = await conv_agent._handle_confirmation_check(
        _make_input("yes", conversation_id="conv-1")
    )

    assert result is not None
    assert result.response.speech["plain"]["speech"] == "Safe content"


async def test_handle_confirmation_check_no_blocks_response(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """'no' with a pending guard rail response returns the blocked message."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    mock_gr = MagicMock()
    await conv_agent._guard_rail_cache.store_pending_response("conv-2", "Blocked text", mock_gr)

    result = await conv_agent._handle_confirmation_check(
        _make_input("no", conversation_id="conv-2")
    )

    assert result is not None
    assert result.response.speech["plain"]["speech"] == GUARD_RAIL_BLOCKED_RESPONSE


# ---------------------------------------------------------------------------
# Test 17 — _extract_response_text helper
# ---------------------------------------------------------------------------


def test_extract_response_text_with_speech(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_extract_response_text returns the plain speech string."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    result = conv_agent._create_result("Hello world")
    assert conv_agent._extract_response_text(result) == "Hello world"


def test_extract_response_text_no_speech(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_extract_response_text returns empty string when speech is absent."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    empty_response = intent.IntentResponse(language="en")
    cr = ConversationResult(response=empty_response, conversation_id=None)
    # Remove speech dict so result is empty
    cr.response.speech = {}
    assert conv_agent._extract_response_text(cr) == ""


# ---------------------------------------------------------------------------
# Test 18 — _maybe_cache_response: store / skip logic (#14)
# ---------------------------------------------------------------------------


def test_maybe_cache_response_stores_when_enabled(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_maybe_cache_response stores the response when cache is enabled."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_AGENT_CACHE_ENABLED: True}
    conv_agent._maybe_cache_response(agent_cfg, "query text", "answer text", "AgentA")
    assert conv_agent._response_cache.get("query text") == "answer text"


def test_maybe_cache_response_skips_empty_response(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_maybe_cache_response does not store an empty response string."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_AGENT_CACHE_ENABLED: True}
    conv_agent._maybe_cache_response(agent_cfg, "query text", "", "AgentA")
    assert conv_agent._response_cache.get("query text") is None


def test_maybe_cache_response_skips_when_disabled(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_maybe_cache_response does not store when agent cache is disabled."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_AGENT_CACHE_ENABLED: False}
    conv_agent._maybe_cache_response(agent_cfg, "query text", "answer text", "AgentA")
    assert conv_agent._response_cache.get("query text") is None


# ---------------------------------------------------------------------------
# Test 19 — _record_agent_failure: timeout vs error paths (#7, #8)
# ---------------------------------------------------------------------------


def test_record_agent_failure_timeout_path(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_record_agent_failure with timed_out=True increments timeout counters."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    conv_agent._statistics.record_request("a1", "AgentX")
    conv_agent._record_agent_failure("a1", timed_out=True)
    stats = conv_agent._statistics.get_all()
    assert stats["AgentX"]["timeouts"] == 1
    assert stats["AgentX"]["failures"] == 0


def test_record_agent_failure_error_path(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_record_agent_failure with timed_out=False increments failure counters."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    conv_agent._statistics.record_request("a1", "AgentY")
    conv_agent._record_agent_failure("a1", timed_out=False)
    stats = conv_agent._statistics.get_all()
    assert stats["AgentY"]["failures"] == 1
    assert stats["AgentY"]["timeouts"] == 0


# ---------------------------------------------------------------------------
# Test 20 — _check_guardrails fires EVENT_GUARD_RAIL_TRIGGERED (#13)
# ---------------------------------------------------------------------------


async def test_check_guardrails_fires_event_on_unsafe_content(
    hass: HomeAssistant,
) -> None:
    """_check_guardrails fires EVENT_GUARD_RAIL_TRIGGERED when content is flagged."""
    entry = _entry_with_guard_rails()
    conv_agent = NeuralBridgeAgent(hass, entry)

    # Listen via the real event bus (EventBus.async_fire is a read-only slot;
    # cannot be patched with patch.object).
    fired_events: list = []
    hass.bus.async_listen(
        EVENT_GUARD_RAIL_TRIGGERED,
        fired_events.append,
    )

    unsafe_result = GuardRailResult(
        is_safe=False,
        confidence=0.95,
        category="harmful_content",
        reason="Test flag",
    )
    mock_checker = MagicMock()
    mock_checker.check_output = AsyncMock(return_value=unsafe_result)
    conv_agent._guard_rail_checker = mock_checker

    agent_cfg = {
        "id": "agent-1",
        CONF_AGENT_NAME: "TestAgent",
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: True,
    }
    conversation_result = conv_agent._create_result("some unsafe response")

    await conv_agent._check_guardrails(agent_cfg, conversation_result, None)
    await hass.async_block_till_done()

    assert len(fired_events) == 1
    assert fired_events[0].data["agent_name"] == "TestAgent"
    assert fired_events[0].data["action"] == GUARD_RAIL_ACTION_BLOCK


# ---------------------------------------------------------------------------
# Test 21 — _handle_successful_result: guard rail ConversationResult returned (#13)
# ---------------------------------------------------------------------------


async def test_handle_successful_result_returns_guard_rail_result(
    hass: HomeAssistant,
) -> None:
    """When guard rails return a ConversationResult, it replaces the original."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    blocked = conv_agent._create_error_result("Blocked by guard rail")

    with patch.object(
        conv_agent, "_check_guardrails", new_callable=AsyncMock, return_value=blocked
    ):
        final = await conv_agent._handle_successful_result(
            agent_config=_make_ollama_agent(),
            result=conv_agent._create_result("original response"),
            user_input=_make_input(),
            agent_id="agent-1",
            elapsed_ms=100.0,
        )

    assert final.response.speech["plain"]["speech"] == "Blocked by guard rail"


# ---------------------------------------------------------------------------
# Test 22 — router agents: blocked request returns error (#5)
# ---------------------------------------------------------------------------


async def test_async_process_router_blocks_request(hass: HomeAssistant) -> None:
    """When a router agent rejects the input, an error result is returned."""
    router = _make_ollama_agent(priority=0, agent_id="router", name="Router")
    entry = _entry_with_agents(router)
    conv_agent = NeuralBridgeAgent(hass, entry)

    with patch.object(conv_agent, "_check_with_routers", new_callable=AsyncMock, return_value=None):
        result = await conv_agent.async_process(_make_input())

    assert "cannot process" in result.response.speech["plain"]["speech"].lower()


# ---------------------------------------------------------------------------
# Test 23 — async_will_remove_from_hass: cleans up Ollama clients
# ---------------------------------------------------------------------------


async def test_async_will_remove_from_hass_closes_clients(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_will_remove_from_hass closes all Ollama clients."""
    conv_agent = NeuralBridgeAgent(hass, mock_config_entry)
    mock_client = AsyncMock()
    conv_agent._ollama_clients["agent-1"] = mock_client

    await conv_agent.async_will_remove_from_hass()

    mock_client.close.assert_called_once()
    assert len(conv_agent._ollama_clients) == 0


# ---------------------------------------------------------------------------
# Test 24 — conversation_id is echoed through _create_result / _create_error_result
# ---------------------------------------------------------------------------


def test_create_result_echoes_conversation_id(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_create_result() echoes back the supplied conversation_id."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    result = agent._create_result("Hello", "conv-abc")

    assert result.conversation_id == "conv-abc"


def test_create_error_result_echoes_conversation_id(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_create_error_result() echoes back the supplied conversation_id."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    result = agent._create_error_result("Error", "conv-xyz")

    assert result.conversation_id == "conv-xyz"


# ---------------------------------------------------------------------------
# Test 25 — conversation_id is propagated through async_process paths
# ---------------------------------------------------------------------------


async def test_async_process_result_carries_conversation_id(hass: HomeAssistant) -> None:
    """A successful async_process() result echoes the input conversation_id."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    user_input = _make_input("Hello", conversation_id="session-123")
    expected = conv_agent._create_result("Agent response", "session-123")

    with patch.object(
        conv_agent, "_try_agent_with_tracking", new_callable=AsyncMock, return_value=expected
    ):
        result = await conv_agent.async_process(user_input)

    assert result.conversation_id == "session-123"


async def test_async_process_fallback_carries_conversation_id(hass: HomeAssistant) -> None:
    """FALLBACK_RESPONSE result echoes the input conversation_id."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    user_input = _make_input("Hello", conversation_id="session-fallback")

    with patch.object(
        conv_agent, "_try_agent_with_tracking", new_callable=AsyncMock, return_value=None
    ):
        result = await conv_agent.async_process(user_input)

    assert result.response.speech["plain"]["speech"] == FALLBACK_RESPONSE
    assert result.conversation_id == "session-fallback"


async def test_async_process_no_agents_carries_conversation_id(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """NO_AGENTS_RESPONSE result echoes the input conversation_id."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    user_input = _make_input("Hello", conversation_id="session-no-agents")

    result = await agent.async_process(user_input)

    assert result.response.speech["plain"]["speech"] == NO_AGENTS_RESPONSE
    assert result.conversation_id == "session-no-agents"


async def test_async_process_cache_hit_carries_conversation_id(hass: HomeAssistant) -> None:
    """A cache-hit result echoes the input conversation_id."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    conv_agent._response_cache.store("Hi", "Cached", "AgentA")
    user_input = _make_input("Hi", conversation_id="session-cache")

    result = await conv_agent.async_process(user_input)

    assert result.response.speech["plain"]["speech"] == "Cached"
    assert result.conversation_id == "session-cache"


# ---------------------------------------------------------------------------
# Test 26 — _try_agent: AGENT_TYPE_OLLAMA dispatches to _process_with_ollama
# ---------------------------------------------------------------------------


async def test_try_agent_dispatches_to_ollama(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """AGENT_TYPE_OLLAMA routes _try_agent to _process_with_ollama."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = _make_ollama_agent()
    expected = agent._create_result("Ollama response")

    with patch.object(agent, "_process_with_ollama", new_callable=AsyncMock, return_value=expected):
        result, timed_out = await agent._try_agent(agent_cfg, _make_input())

    assert result is expected
    assert timed_out is False


# ---------------------------------------------------------------------------
# Test 27 — _try_agent: AGENT_TYPE_EXISTING dispatches to _process_with_existing
# ---------------------------------------------------------------------------


async def test_try_agent_dispatches_to_existing(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """AGENT_TYPE_EXISTING routes _try_agent to _process_with_existing."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {
        "id": "ext-1",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "ChatGPT",
        CONF_PRIORITY: 10,
        CONF_ENTITY_ID: "conversation.openai",
        CONF_TIMEOUT: 30,
    }
    expected = agent._create_result("External response")

    with patch.object(
        agent, "_process_with_existing", new_callable=AsyncMock, return_value=expected
    ):
        result, timed_out = await agent._try_agent(agent_cfg, _make_input())

    assert result is expected
    assert timed_out is False


# ---------------------------------------------------------------------------
# Test 28 — _try_agent: AGENT_TYPE_LOCAL_HA also dispatches to _process_with_existing
# ---------------------------------------------------------------------------


async def test_try_agent_local_ha_dispatches_to_existing(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """AGENT_TYPE_LOCAL_HA routes _try_agent to _process_with_existing."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {
        "id": "local-1",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_NAME: "HA Local",
        CONF_PRIORITY: 50,
        CONF_ENTITY_ID: "conversation.home_assistant",
        CONF_TIMEOUT: 30,
    }
    expected = agent._create_result("Local HA response")

    with patch.object(
        agent, "_process_with_existing", new_callable=AsyncMock, return_value=expected
    ):
        result, timed_out = await agent._try_agent(agent_cfg, _make_input())

    assert result is expected
    assert timed_out is False


# ---------------------------------------------------------------------------
# Test 29 — _try_agent: unknown agent type returns (None, False)
# ---------------------------------------------------------------------------


async def test_try_agent_unknown_type_returns_none_false(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """An unrecognised agent type returns (None, False) without raising."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {
        "id": "mystery",
        CONF_AGENT_TYPE: "super_futuristic_agent",
        CONF_AGENT_NAME: "Mystery",
        CONF_PRIORITY: 10,
        CONF_TIMEOUT: 30,
    }

    result, timed_out = await agent._try_agent(agent_cfg, _make_input())

    assert result is None
    assert timed_out is False


# ---------------------------------------------------------------------------
# Test 30 — _try_agent: asyncio.TimeoutError → (None, True)
# ---------------------------------------------------------------------------


async def test_try_agent_timeout_returns_timed_out_true(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """asyncio.TimeoutError inside _try_agent returns (None, timed_out=True)."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)

    with patch.object(
        agent,
        "_process_with_ollama",
        new_callable=AsyncMock,
        side_effect=asyncio.TimeoutError,
    ):
        result, timed_out = await agent._try_agent(_make_ollama_agent(), _make_input())

    assert result is None
    assert timed_out is True


# ---------------------------------------------------------------------------
# Test 31 — _process_with_ollama: success — result returned and memory updated
# ---------------------------------------------------------------------------


async def test_process_with_ollama_success(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_process_with_ollama returns a ConversationResult and stores the turn."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = _make_ollama_agent(agent_id="ollama-x", system_prompt="Be concise.")

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value="Ollama answered!")

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        result = await agent._process_with_ollama(
            agent_cfg, _make_input("Hello", conversation_id="sess-42")
        )

    assert result is not None
    assert result.response.speech["plain"]["speech"] == "Ollama answered!"
    # Turn must be stored so follow-up messages include conversation history
    messages = agent._session_memory.get_messages("sess-42")
    assert any(m["role"] == "assistant" for m in messages)


# ---------------------------------------------------------------------------
# Test 32 — _process_with_ollama: chat() returns None → result is None
# ---------------------------------------------------------------------------


async def test_process_with_ollama_client_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """When OllamaClient.chat() returns None, _process_with_ollama returns None."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = _make_ollama_agent(agent_id="ollama-y")

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value=None)

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        result = await agent._process_with_ollama(agent_cfg, _make_input())

    assert result is None


# ---------------------------------------------------------------------------
# Test 32b — _process_with_ollama: missing/non-string url or model → None
# ---------------------------------------------------------------------------


async def test_process_with_ollama_missing_url_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_process_with_ollama returns None when ollama_url is None (not a string)."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = _make_ollama_agent(agent_id="ollama-no-url")
    agent_cfg[CONF_OLLAMA_URL] = None  # override with non-string

    result = await agent._process_with_ollama(agent_cfg, _make_input())

    assert result is None


# ---------------------------------------------------------------------------
# Test 33 — _process_with_existing: entity state absent → None
# ---------------------------------------------------------------------------


async def test_process_with_existing_entity_not_found(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_process_with_existing returns None when the entity state does not exist."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_ENTITY_ID: "conversation.ghost", CONF_AGENT_NAME: "Ghost"}

    # hass.states.get returns None for unknown entities by default
    result = await agent._process_with_existing(agent_cfg, _make_input())

    assert result is None


# ---------------------------------------------------------------------------
# Test 33b — _process_with_existing: no entity_id configured → None
# ---------------------------------------------------------------------------


async def test_process_with_existing_no_entity_id_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_process_with_existing returns None when CONF_ENTITY_ID is absent."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_AGENT_NAME: "Missing ID"}  # no CONF_ENTITY_ID

    result = await agent._process_with_existing(agent_cfg, _make_input())

    assert result is None


# ---------------------------------------------------------------------------
# Test 34 — _process_with_existing: service call returns speech → ConversationResult
# ---------------------------------------------------------------------------


async def test_process_with_existing_success(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_process_with_existing returns a result containing the downstream agent's speech."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_ENTITY_ID: "conversation.openai", CONF_AGENT_NAME: "OpenAI"}

    service_response = {"response": {"speech": {"plain": {"speech": "GPT says hello"}}}}

    # ServiceRegistry.async_call and StateMachine.get are read-only slot attributes in
    # this version of HA — replace agent.hass with a lightweight mock that lets us
    # control both responses without touching the real HA internals.
    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language  # preserve real language
    mock_hass.states.get.return_value = MagicMock()  # entity exists (non-None)
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    result = await agent._process_with_existing(agent_cfg, _make_input("Hi"))

    assert result is not None
    assert result.response.speech["plain"]["speech"] == "GPT says hello"


# ---------------------------------------------------------------------------
# Helpers for router tests
# ---------------------------------------------------------------------------


def _make_router_config(
    agent_id: str = "router-1",
    name: str = "Router",
    url: str = "http://localhost:11434",
    model: str = "tinyllama",
    agent_type: str = AGENT_TYPE_OLLAMA,
) -> dict[str, Any]:
    """Return a minimal router agent config (priority 0, Ollama type by default)."""
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: agent_type,
        CONF_AGENT_NAME: name,
        CONF_PRIORITY: 0,
        CONF_OLLAMA_URL: url,
        CONF_OLLAMA_MODEL: model,
        CONF_TIMEOUT: 10,
    }


# ---------------------------------------------------------------------------
# Test 35 — _check_with_routers: all routers return RouterDecision → last decision returned
# ---------------------------------------------------------------------------


async def test_check_with_routers_all_pass_returns_decision(hass: HomeAssistant) -> None:
    """_check_with_routers returns the last RouterDecision when every router classifies."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    routers = [_make_router_config("r1"), _make_router_config("r2")]
    expected_decision = RouterDecision(local_ha=True, complexity=30)

    with patch.object(
        conv_agent, "_classify_with_router", new_callable=AsyncMock, return_value=expected_decision
    ):
        result = await conv_agent._check_with_routers(_make_input("turn on lights"), routers)

    assert result is not None
    assert result.local_ha is True
    assert result.complexity == 30


# ---------------------------------------------------------------------------
# Test 36 — _check_with_routers: first None (block) stops immediately
# ---------------------------------------------------------------------------


async def test_check_with_routers_block_stops_at_first_router(hass: HomeAssistant) -> None:
    """_check_with_routers returns None on the first block without calling later routers."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    routers = [_make_router_config("r1"), _make_router_config("r2")]

    call_count = 0

    def always_block(
        router_config: dict,
        user_text: str,
        area_context: str | None = None,
        language: str | None = None,
        agent_types: list[str] | None = None,
    ) -> None:
        nonlocal call_count
        call_count += 1

    with patch.object(conv_agent, "_classify_with_router", side_effect=always_block):
        result = await conv_agent._check_with_routers(_make_input("bad"), routers)

    assert result is None
    assert call_count == 1  # stopped after first block


# ---------------------------------------------------------------------------
# Test 37 — _check_with_routers: single router block → None
# ---------------------------------------------------------------------------


async def test_check_with_routers_single_block_returns_none(hass: HomeAssistant) -> None:
    """A single router returning None causes _check_with_routers to return None."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    with patch.object(
        conv_agent, "_classify_with_router", new_callable=AsyncMock, return_value=None
    ):
        result = await conv_agent._check_with_routers(
            _make_input("harmful query"), [_make_router_config()]
        )

    assert result is None


# ---------------------------------------------------------------------------
# Test 38 — _classify_with_router: AGENT_TYPE_EXISTING missing entity_id → fail-open
# ---------------------------------------------------------------------------


async def test_classify_with_router_non_ollama_returns_fail_open(hass: HomeAssistant) -> None:
    """AGENT_TYPE_EXISTING router with no entity_id returns fail-open RouterDecision."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_type=AGENT_TYPE_EXISTING)
    # _make_router_config does not set entity_id, triggering the missing-entity path

    result = await conv_agent._classify_with_router(router, "some text")

    assert result is not None
    assert isinstance(result, RouterDecision)
    assert result.complexity == DEFAULT_ROUTER_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 39 — _classify_with_router: missing URL → fail-open RouterDecision
# ---------------------------------------------------------------------------


async def test_classify_with_router_missing_url_returns_fail_open(hass: HomeAssistant) -> None:
    """Router with no URL configured returns fail-open RouterDecision."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(url="")

    result = await conv_agent._classify_with_router(router, "some text")

    assert result is not None
    assert isinstance(result, RouterDecision)


# ---------------------------------------------------------------------------
# Test 40 — _classify_with_router: missing model → fail-open RouterDecision
# ---------------------------------------------------------------------------


async def test_classify_with_router_missing_model_returns_fail_open(hass: HomeAssistant) -> None:
    """Router with no model configured returns fail-open RouterDecision."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(model="")

    result = await conv_agent._classify_with_router(router, "some text")

    assert result is not None
    assert isinstance(result, RouterDecision)


# ---------------------------------------------------------------------------
# Test 41 — _classify_with_router: valid JSON PASS response → RouterDecision
# ---------------------------------------------------------------------------


async def test_classify_with_router_json_pass_returns_decision(hass: HomeAssistant) -> None:
    """Router returning valid JSON with complexity>0 yields a RouterDecision."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config()

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value='{"local_ha": true, "complexity": 12}',
    ):
        result = await conv_agent._classify_with_router(router, "turn on the lights")

    assert result is not None
    assert result.local_ha is True
    assert result.complexity == 12


# ---------------------------------------------------------------------------
# Test 42 — _classify_with_router: JSON complexity=0 → None (block)
# ---------------------------------------------------------------------------


async def test_classify_with_router_json_block_returns_none(hass: HomeAssistant) -> None:
    """Router returning complexity=0 JSON yields None (block signal)."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config()

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value='{"local_ha": false, "complexity": 0}',
    ):
        result = await conv_agent._classify_with_router(router, "harmful request")

    assert result is None


# ---------------------------------------------------------------------------
# Test 43 — _classify_with_router: Ollama returns None → fail-open RouterDecision
# ---------------------------------------------------------------------------


async def test_classify_with_router_none_response_returns_fail_open(hass: HomeAssistant) -> None:
    """Router returning None (network error / timeout) returns fail-open RouterDecision."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config()

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await conv_agent._classify_with_router(router, "some text")

    assert result is not None
    assert isinstance(result, RouterDecision)
    assert result.complexity == DEFAULT_ROUTER_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 44 — _classify_with_router: unparseable response → fail-open RouterDecision
# ---------------------------------------------------------------------------


async def test_classify_with_router_unparseable_response_returns_fail_open(
    hass: HomeAssistant,
) -> None:
    """Router returning non-JSON text returns fail-open RouterDecision."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config()

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value="I am not sure what to say here",
    ):
        result = await conv_agent._classify_with_router(router, "some text")

    assert result is not None
    assert isinstance(result, RouterDecision)
    assert result.complexity == DEFAULT_ROUTER_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 45 — _classify_with_router: JSON local_ha=false, general question
# ---------------------------------------------------------------------------


async def test_classify_with_router_general_question_not_local_ha(
    hass: HomeAssistant,
) -> None:
    """Router classifying a general question returns local_ha=False."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="r-general")

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value='{"local_ha": false, "complexity": 70}',
    ):
        result = await conv_agent._classify_with_router(router, "What is the capital of France?")

    assert result is not None
    assert result.local_ha is False
    assert result.complexity == 70


# ---------------------------------------------------------------------------
# Test 46 — _classify_with_router: creates OllamaClient on first call
# ---------------------------------------------------------------------------


async def test_classify_with_router_creates_client(hass: HomeAssistant) -> None:
    """_classify_with_router creates and caches an OllamaClient on the first call."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="new-router")

    assert "new-router" not in conv_agent._ollama_clients

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value='{"local_ha": false, "complexity": 30}',
    ):
        await conv_agent._classify_with_router(router, "hello")

    assert "new-router" in conv_agent._ollama_clients


# ---------------------------------------------------------------------------
# Test 47 — _classify_with_router: reuses existing OllamaClient
# ---------------------------------------------------------------------------


async def test_classify_with_router_reuses_existing_client(hass: HomeAssistant) -> None:
    """_classify_with_router does not create a second client if one already exists."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="cached-router")

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(return_value='{"local_ha": false, "complexity": 20}')
    conv_agent._ollama_clients["cached-router"] = mock_client

    await conv_agent._classify_with_router(router, "hello")

    mock_client.generate.assert_called_once()
    assert conv_agent._ollama_clients["cached-router"] is mock_client


# ---------------------------------------------------------------------------
# Test 47b — _classify_with_router: entity context appended to prompt
# ---------------------------------------------------------------------------


async def test_classify_with_router_appends_entity_context(hass: HomeAssistant) -> None:
    """Entity context summary is appended to the classification prompt when non-empty."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="ctx-router")

    mock_cache = MagicMock(spec=EntityContextCache)
    mock_cache.get_summary.return_value = "Available smart home entities:\n- weather: Met.no"
    conv_agent._entity_context_cache = mock_cache

    captured_prompt: list[str] = []

    async def _capture_generate(prompt: str, **_kwargs: object) -> str:
        captured_prompt.append(prompt)
        return '{"local_ha": true, "complexity": 8}'

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        side_effect=_capture_generate,
    ):
        result = await conv_agent._classify_with_router(router, "What is the weather?")

    assert result is not None
    assert result.local_ha is True
    assert len(captured_prompt) == 1
    assert "Available smart home entities:" in captured_prompt[0]
    assert "Met.no" in captured_prompt[0]


async def test_classify_with_router_skips_empty_entity_context(hass: HomeAssistant) -> None:
    """No entity context is appended to the prompt when the summary is empty."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="no-ctx-router")

    mock_cache = MagicMock(spec=EntityContextCache)
    mock_cache.get_summary.return_value = ""
    conv_agent._entity_context_cache = mock_cache

    captured_prompt: list[str] = []

    async def _capture_generate(prompt: str, **_kwargs: object) -> str:
        captured_prompt.append(prompt)
        return '{"local_ha": false, "complexity": 30}'

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        side_effect=_capture_generate,
    ):
        await conv_agent._classify_with_router(router, "What is 2+2?")

    assert "Available smart home entities:" not in captured_prompt[0]


# ---------------------------------------------------------------------------
# Test 48 — async_setup_entry creates and registers agent entity
# ---------------------------------------------------------------------------


async def test_async_setup_entry_creates_agent(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_setup_entry instantiates NeuralBridgeAgent and calls async_add_entities."""
    added: list = []

    await async_setup_entry(hass, mock_config_entry, added.append)

    # async_add_entities is called as async_add_entities([agent]), so
    # added[0] is the list [agent] — the entity is at added[0][0].
    assert len(added) == 1
    entities = added[0]
    assert len(entities) == 1
    assert isinstance(entities[0], NeuralBridgeAgent)


# ---------------------------------------------------------------------------
# Test 49 — supported_languages returns "*"
# ---------------------------------------------------------------------------


def test_supported_languages_returns_wildcard(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """supported_languages property returns '*' to accept all languages."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    assert agent.supported_languages == "*"


# ---------------------------------------------------------------------------
# Test 50 — _try_agent: non-timeout exception → (None, False)
# ---------------------------------------------------------------------------


async def test_try_agent_generic_exception_returns_none_false(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """A non-TimeoutError exception in _try_agent returns (None, timed_out=False)."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)

    with patch.object(
        agent,
        "_process_with_ollama",
        new_callable=AsyncMock,
        side_effect=RuntimeError("unexpected"),
    ):
        result, timed_out = await agent._try_agent(_make_ollama_agent(), _make_input())

    assert result is None
    assert timed_out is False


# ---------------------------------------------------------------------------
# Test 51 — _process_with_existing: service returns unexpected shape → None
# ---------------------------------------------------------------------------


async def test_process_with_existing_unexpected_response_shape_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_process_with_existing returns None when service response lacks 'response' key."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_ENTITY_ID: "conversation.test", CONF_AGENT_NAME: "Test"}

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value={"unexpected": "shape"})
    agent.hass = mock_hass

    result = await agent._process_with_existing(agent_cfg, _make_input())

    assert result is None


# ---------------------------------------------------------------------------
# Test 52 — _process_with_existing: service raises → None
# ---------------------------------------------------------------------------


async def test_process_with_existing_service_raises_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_process_with_existing returns None when the service call raises an exception."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_ENTITY_ID: "conversation.test", CONF_AGENT_NAME: "Test"}

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(side_effect=RuntimeError("service error"))
    agent.hass = mock_hass

    result = await agent._process_with_existing(agent_cfg, _make_input())

    assert result is None


async def test_process_with_existing_empty_speech_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_process_with_existing returns None when the response speech text is empty."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_ENTITY_ID: "conversation.test", CONF_AGENT_NAME: "Test"}

    # Response has valid structure but empty speech text.
    service_response = {"response": {"speech": {"plain": {"speech": ""}}}}

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    result = await agent._process_with_existing(agent_cfg, _make_input())

    assert result is None


# ---------------------------------------------------------------------------
# Test 52b — _process_with_existing: ChatLog is isolated during sub-call
# ---------------------------------------------------------------------------
# When HA 2025.2+ is in use, the built-in HA conversation agent writes the
# response to the shared ChatLog as part of its own async_process.  Because
# _process_with_existing shares the same asyncio context, that write would land
# in the *outer* ChatLog and cause the response text to appear twice in the Assist
# UI.  The fix temporarily sets current_chat_log to None before the service call
# and restores the original value afterwards.


async def test_process_with_existing_chat_log_isolated_during_sub_call(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """current_chat_log is set to None for the sub-agent service call and restored.

    In HA 2025.2+ the built-in HA conversation agent can write its response to the
    shared ChatLog as part of its own async_process.  Because _process_with_existing
    runs in the same asyncio context the write would land in the *outer* ChatLog,
    causing the response text to be added twice and appearing doubled in Assist UI.

    The fix temporarily sets current_chat_log to None for the duration of the service
    call (isolating the sub-agent) then restores the original value so
    _maybe_add_to_chat_log can add the response exactly once.
    """
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {CONF_ENTITY_ID: "conversation.builtin", CONF_AGENT_NAME: "BuiltIn"}

    # A sentinel object that represents the "outer" ChatLog instance.
    outer_chat_log = MagicMock(name="outer_chat_log")

    # Use a real ContextVar to simulate current_chat_log.
    fake_current_chat_log: ContextVar[Any] = ContextVar("current_chat_log", default=None)
    outer_token = fake_current_chat_log.set(outer_chat_log)

    # Track what current_chat_log.get() returns at the moment the service is called.
    chat_log_value_during_call: list[Any] = []

    service_response = {"response": {"speech": {"plain": {"speech": "hello"}}}}

    async def mock_service_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        chat_log_value_during_call.append(fake_current_chat_log.get())
        return service_response

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = mock_service_call
    agent.hass = mock_hass

    # Patch the chat_log module inside conversation.py to use our fake ContextVar.
    fake_chat_log_module = MagicMock()
    fake_chat_log_module.current_chat_log = fake_current_chat_log

    with patch.dict(
        "sys.modules",
        {"homeassistant.components.conversation.chat_log": fake_chat_log_module},
    ):
        result = await agent._process_with_existing(agent_cfg, _make_input("Hi"))

    # The sub-agent service call must have seen None (isolated from outer ChatLog).
    assert len(chat_log_value_during_call) == 1
    assert chat_log_value_during_call[0] is None, (
        "ChatLog was not isolated: sub-agent saw a non-None ChatLog and "
        "could produce duplicate response text in Assist UI"
    )
    # The result is still correctly returned.
    assert result is not None
    assert result.response.speech["plain"]["speech"] == "hello"
    # The outer ChatLog is restored — _maybe_add_to_chat_log can re-add correctly.
    assert fake_current_chat_log.get() == outer_chat_log

    # Clean up the ContextVar (restore its value to the pre-test state).
    fake_current_chat_log.reset(outer_token)


# ---------------------------------------------------------------------------
# Test 53 — _check_guardrails: empty response_text → None (no event fired)
# ---------------------------------------------------------------------------


async def test_check_guardrails_empty_response_text_returns_none(
    hass: HomeAssistant,
) -> None:
    """_check_guardrails returns None immediately when response text is empty."""
    entry = _entry_with_guard_rails()
    conv_agent = NeuralBridgeAgent(hass, entry)

    agent_cfg = {
        "id": "agent-empty",
        CONF_AGENT_NAME: "Empty",
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: True,
    }
    # Create a result with empty speech so _extract_response_text returns ""
    empty_result = conv_agent._create_result("")

    result = await conv_agent._check_guardrails(agent_cfg, empty_result, None)

    assert result is None


# ---------------------------------------------------------------------------
# Test 54 — _check_guardrails: unknown action → None (fallthrough)
# ---------------------------------------------------------------------------


async def test_check_guardrails_unknown_action_returns_none(
    hass: HomeAssistant,
) -> None:
    """_check_guardrails returns None when action is not BLOCK/WARN/NOTIFY_ASK."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_ACTION: "unknown_action",
        },
    )
    conv_agent = NeuralBridgeAgent(hass, entry)

    unsafe_result = GuardRailResult(is_safe=False, confidence=0.95, category="harmful")
    mock_checker = MagicMock()
    mock_checker.check_output = AsyncMock(return_value=unsafe_result)
    conv_agent._guard_rail_checker = mock_checker

    agent_cfg = {
        "id": "agent-x",
        CONF_AGENT_NAME: "TestAgent",
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: True,
    }
    conversation_result = conv_agent._create_result("some response")

    result = await conv_agent._check_guardrails(agent_cfg, conversation_result, None)

    assert result is None


# ---------------------------------------------------------------------------
# Test 55 — _get_guard_rail_agent_config: agent id not found → None
# ---------------------------------------------------------------------------


async def test_get_guard_rail_agent_config_id_not_found_returns_none(
    hass: HomeAssistant,
) -> None:
    """_get_guard_rail_agent_config returns None when guard_rail_agent_id doesn't match."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [_make_ollama_agent(agent_id="agent-1")],
            "guard_rail_agent_id": "nonexistent-id",
        },
    )
    conv_agent = NeuralBridgeAgent(hass, entry)

    result = await conv_agent._get_guard_rail_agent_config()

    assert result is None


# ---------------------------------------------------------------------------
# Test 56 — async_process: confirmation result returned early (line 172)
# ---------------------------------------------------------------------------


async def test_async_process_returns_confirmation_result(hass: HomeAssistant) -> None:
    """async_process returns early when _handle_confirmation_check returns non-None."""
    entry = _entry_with_agents()
    conv_agent = NeuralBridgeAgent(hass, entry)
    expected = conv_agent._create_result("confirmed response")

    with patch.object(
        conv_agent,
        "_handle_confirmation_check",
        new_callable=AsyncMock,
        return_value=expected,
    ):
        result = await conv_agent.async_process(_make_input("yes"))

    assert result.response.speech["plain"]["speech"] == "confirmed response"


# ---------------------------------------------------------------------------
# Test 57 — _try_agent_with_retries: retries on failure (lines 306-314)
# ---------------------------------------------------------------------------


async def test_retry_succeeds_on_second_attempt(hass: HomeAssistant) -> None:
    """_try_agent_with_retries retries after a failure and succeeds on second attempt."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [_make_ollama_agent()], CONF_MAX_RETRIES: 1, CONF_RETRY_BASE_DELAY: 0.1},
    )
    conv_agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = _make_ollama_agent()
    success_result = conv_agent._create_result("success on retry")

    call_count = 0

    def try_agent_side_effect(_cfg, _inp):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None, False
        return success_result, False

    with (
        patch.object(conv_agent, "_try_agent", side_effect=try_agent_side_effect),
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        result = await conv_agent._try_agent_with_retries(
            agent_cfg, _make_input(), "agent-1", "Test Ollama"
        )

    assert result is not None
    assert result.response.speech["plain"]["speech"] == "success on retry"
    mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# Test 58 — _try_agent_with_retries: all retries exhausted (line 335)
# ---------------------------------------------------------------------------


async def test_retry_exhausted_returns_none(hass: HomeAssistant) -> None:
    """_try_agent_with_retries returns None when all attempts fail."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [_make_ollama_agent()], CONF_MAX_RETRIES: 2, CONF_RETRY_BASE_DELAY: 0.1},
    )
    conv_agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = _make_ollama_agent()

    with (
        patch.object(conv_agent, "_try_agent", new_callable=AsyncMock, return_value=(None, False)),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await conv_agent._try_agent_with_retries(
            agent_cfg, _make_input(), "agent-1", "Test Ollama"
        )

    assert result is None


# ---------------------------------------------------------------------------
# Test 59 — _try_agent_with_retries: circuit trips → stops retrying (lines 328-333)
# ---------------------------------------------------------------------------


async def test_retry_stops_when_circuit_trips(hass: HomeAssistant) -> None:
    """_try_agent_with_retries stops after the circuit breaker trips mid-retry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [_make_ollama_agent()], CONF_MAX_RETRIES: 3, CONF_RETRY_BASE_DELAY: 0.1},
    )
    conv_agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = _make_ollama_agent()

    with (
        patch.object(conv_agent, "_try_agent", new_callable=AsyncMock, return_value=(None, False)),
        patch.object(conv_agent._circuit_breaker, "is_open", return_value=True),
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        result = await conv_agent._try_agent_with_retries(
            agent_cfg, _make_input(), "agent-1", "Test Ollama"
        )

    assert result is None
    # No sleep because circuit tripped immediately after first attempt
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Test 60 — _check_guardrails: guard rail globally disabled → None (line 712)
# ---------------------------------------------------------------------------


async def test_check_guardrails_disabled_globally_returns_none(hass: HomeAssistant) -> None:
    """_check_guardrails returns None immediately when guard rails are globally disabled."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [], CONF_GUARD_RAIL_ENABLED: False},
    )
    conv_agent = NeuralBridgeAgent(hass, entry)

    agent_cfg = {CONF_AGENT_NAME: "Agent", CONF_GUARD_RAIL_ENABLED_FOR_AGENT: True}
    conv_result = conv_agent._create_result("some text")

    result = await conv_agent._check_guardrails(agent_cfg, conv_result, None)

    assert result is None


# ---------------------------------------------------------------------------
# Test 61 — _check_guardrails: per-agent guard rail disabled → None (line 712)
# ---------------------------------------------------------------------------


async def test_check_guardrails_disabled_for_agent_returns_none(hass: HomeAssistant) -> None:
    """_check_guardrails returns None when guard rails are disabled for the agent."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [], CONF_GUARD_RAIL_ENABLED: True},
    )
    conv_agent = NeuralBridgeAgent(hass, entry)

    agent_cfg = {CONF_AGENT_NAME: "Agent", CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False}
    conv_result = conv_agent._create_result("some text")

    result = await conv_agent._check_guardrails(agent_cfg, conv_result, None)

    assert result is None


# ---------------------------------------------------------------------------
# Test 62 — _check_guardrails: checker initialized with custom rules (lines 724-729)
# ---------------------------------------------------------------------------


async def test_guard_rail_checker_uses_custom_rules(hass: HomeAssistant) -> None:
    """When CONF_GUARD_RAIL_RULES is set, GuardRailChecker is constructed with those rules."""
    custom_rules = {"harmful": ["\\bbomb\\b"]}
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_RULES: custom_rules,
        },
    )
    conv_agent = NeuralBridgeAgent(hass, entry)

    agent_cfg = {CONF_AGENT_NAME: "Agent", CONF_GUARD_RAIL_ENABLED_FOR_AGENT: True}
    safe_result_gr = GuardRailResult(is_safe=True, confidence=0.1, category="")

    captured_kwargs: dict = {}

    def capture_checker(**kwargs):
        captured_kwargs.update(kwargs)
        mock_instance = MagicMock()
        mock_instance.check_output = AsyncMock(return_value=safe_result_gr)
        return mock_instance

    with patch(
        "custom_components.neuralbridge.conversation.GuardRailChecker",
        side_effect=capture_checker,
    ):
        await conv_agent._check_guardrails(
            agent_cfg, conv_agent._create_result("some response"), None
        )

    assert captured_kwargs.get("rules") == custom_rules


# ---------------------------------------------------------------------------
# Test 63 — _check_guardrails: guard rail result is safe → None (line 744)
# ---------------------------------------------------------------------------


async def test_check_guardrails_safe_result_returns_none(hass: HomeAssistant) -> None:
    """_check_guardrails returns None when the checker reports the content as safe."""
    entry = _entry_with_guard_rails()
    conv_agent = NeuralBridgeAgent(hass, entry)

    safe_result_gr = GuardRailResult(is_safe=True, confidence=0.05, category="")
    mock_checker = MagicMock()
    mock_checker.check_output = AsyncMock(return_value=safe_result_gr)
    conv_agent._guard_rail_checker = mock_checker

    agent_cfg = {CONF_AGENT_NAME: "Agent", CONF_GUARD_RAIL_ENABLED_FOR_AGENT: True}
    result = await conv_agent._check_guardrails(
        agent_cfg, conv_agent._create_result("safe content"), None
    )

    assert result is None


# ---------------------------------------------------------------------------
# Test 64 — _apply_guard_rail_action: WARN modifies result (lines 677-679)
# ---------------------------------------------------------------------------


async def test_apply_guard_rail_action_warn_modifies_response(hass: HomeAssistant) -> None:
    """GUARD_RAIL_ACTION_WARN prefixes the response and returns None."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: []},
    )
    conv_agent = NeuralBridgeAgent(hass, entry)
    conv_result = conv_agent._create_result("original text")
    guard_result = GuardRailResult(is_safe=False, confidence=0.9, category="harmful")

    returned = await conv_agent._apply_guard_rail_action(
        GUARD_RAIL_ACTION_WARN, conv_result, "original text", None, guard_result
    )

    assert returned is None
    speech = conv_result.response.speech["plain"]["speech"]
    assert "original text" in speech


# ---------------------------------------------------------------------------
# Test 65 — _apply_guard_rail_action: NOTIFY_ASK caches response (lines 681-684)
# ---------------------------------------------------------------------------


async def test_apply_guard_rail_action_notify_ask_caches_and_returns(
    hass: HomeAssistant,
) -> None:
    """GUARD_RAIL_ACTION_NOTIFY_ASK stores the response and returns a prompt result."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: []},
    )
    conv_agent = NeuralBridgeAgent(hass, entry)
    conv_result = conv_agent._create_result("sensitive text")
    guard_result = GuardRailResult(is_safe=False, confidence=0.9, category="harmful")

    returned = await conv_agent._apply_guard_rail_action(
        GUARD_RAIL_ACTION_NOTIFY_ASK, conv_result, "sensitive text", "conv-99", guard_result
    )

    assert returned is not None
    pending = await conv_agent._guard_rail_cache.get_pending_response("conv-99")
    assert pending is not None
    assert pending[0] == "sensitive text"


# ---------------------------------------------------------------------------
# Test 66 — _check_guardrails: rules change invalidates checker (lines 720-721)
# ---------------------------------------------------------------------------


async def test_guard_rail_checker_reinitialised_when_rules_change(
    hass: HomeAssistant,
) -> None:
    """_check_guardrails recreates the checker when CONF_GUARD_RAIL_RULES changes."""
    new_rules = {"harmful": ["\\bnewrule\\b"]}
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_RULES: new_rules,
        },
    )
    conv_agent = NeuralBridgeAgent(hass, entry)

    # Simulate an old checker and a different (old) snapshot
    old_checker = MagicMock()
    conv_agent._guard_rail_checker = old_checker
    conv_agent._guard_rail_rules_snapshot = {"harmful": ["\\boldRule\\b"]}

    agent_cfg = {CONF_AGENT_NAME: "Agent", CONF_GUARD_RAIL_ENABLED_FOR_AGENT: True}
    safe_gr = GuardRailResult(is_safe=True, confidence=0.1, category="")

    new_checker = MagicMock()
    new_checker.check_output = AsyncMock(return_value=safe_gr)

    with patch(
        "custom_components.neuralbridge.conversation.GuardRailChecker",
        return_value=new_checker,
    ):
        await conv_agent._check_guardrails(agent_cfg, conv_agent._create_result("some text"), None)

    assert conv_agent._guard_rail_checker is new_checker
    assert conv_agent._guard_rail_rules_snapshot == new_rules


# ---------------------------------------------------------------------------
# Test 67 — _get_guard_rail_agent_config: agent found → returns config (line 787)
# ---------------------------------------------------------------------------


async def test_get_guard_rail_agent_config_found_returns_config(
    hass: HomeAssistant,
) -> None:
    """_get_guard_rail_agent_config returns the agent config when the ID matches."""
    agent = _make_ollama_agent(agent_id="gr-agent")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [agent],
            "guard_rail_agent_id": "gr-agent",
        },
    )
    conv_agent = NeuralBridgeAgent(hass, entry)

    result = await conv_agent._get_guard_rail_agent_config()

    assert result is not None


# ---------------------------------------------------------------------------
# Test 68 — _classify_with_router: AGENT_TYPE_EXISTING missing entity_id records failure
# ---------------------------------------------------------------------------


async def test_classify_with_router_existing_missing_entity_records_failure(
    hass: HomeAssistant,
) -> None:
    """AGENT_TYPE_EXISTING router with no entity_id records a request + failure."""

    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="non-ollama-router", agent_type=AGENT_TYPE_EXISTING)
    # _make_router_config does not set entity_id, so it will be missing

    result = await conv_agent._classify_with_router(router, "some text")

    # Fallback=default_complexity returns a RouterDecision, not None
    assert result is not None
    stats = conv_agent._statistics.get_agent_stats("non-ollama-router")
    assert stats is not None
    assert stats.requests == 1
    assert stats.successes == 0
    assert stats.failures == 1


# ---------------------------------------------------------------------------
# Test 69 — _classify_with_router: records request + failure when URL missing
# ---------------------------------------------------------------------------


async def test_classify_with_router_missing_url_records_failure(
    hass: HomeAssistant,
) -> None:
    """Router with missing URL records a request + failure in statistics."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="no-url-router", url="")

    result = await conv_agent._classify_with_router(router, "some text")

    assert result is not None
    stats = conv_agent._statistics.get_agent_stats("no-url-router")
    assert stats is not None
    assert stats.requests == 1
    assert stats.failures == 1
    assert stats.successes == 0


# ---------------------------------------------------------------------------
# Test 70 — _classify_with_router: records request + failure when no response
# ---------------------------------------------------------------------------


async def test_classify_with_router_no_response_records_failure(
    hass: HomeAssistant,
) -> None:
    """Router returning None records a request + failure in statistics."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="silent-router")

    with patch(
        "custom_components.neuralbridge.ollama_client.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await conv_agent._classify_with_router(router, "some text")

    assert result is not None
    stats = conv_agent._statistics.get_agent_stats("silent-router")
    assert stats is not None
    assert stats.requests == 1
    assert stats.failures == 1
    assert stats.successes == 0


# ---------------------------------------------------------------------------
# Test 71 — _classify_with_router: PASS response records success, no block
# ---------------------------------------------------------------------------


async def test_classify_with_router_pass_records_success_not_block(
    hass: HomeAssistant,
) -> None:
    """JSON PASS classification records a success but does NOT increment blocks."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="pass-router")

    with patch(
        "custom_components.neuralbridge.ollama_client.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value='{"local_ha": true, "complexity": 8}',
    ):
        result = await conv_agent._classify_with_router(router, "turn on lights")

    assert result is not None
    stats = conv_agent._statistics.get_agent_stats("pass-router")
    assert stats is not None
    assert stats.requests == 1
    assert stats.successes == 1
    assert stats.blocks == 0


# ---------------------------------------------------------------------------
# Test 72 — _classify_with_router: BLOCK response records success + block
# ---------------------------------------------------------------------------


async def test_classify_with_router_block_records_success_and_block(
    hass: HomeAssistant,
) -> None:
    """JSON complexity=0 block records both a success and a block in statistics."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="block-router")

    with patch(
        "custom_components.neuralbridge.ollama_client.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value='{"local_ha": false, "complexity": 0}',
    ):
        result = await conv_agent._classify_with_router(router, "harmful request")

    assert result is None
    stats = conv_agent._statistics.get_agent_stats("block-router")
    assert stats is not None
    assert stats.requests == 1
    assert stats.successes == 1
    assert stats.blocks == 1


# ---------------------------------------------------------------------------
# Test 73 — _classify_with_router: PASS dispatches stats update signal
# ---------------------------------------------------------------------------


async def test_classify_with_router_pass_dispatches_signal(
    hass: HomeAssistant,
) -> None:
    """_classify_with_router sends SIGNAL_STATS_UPDATED after a JSON PASS classification."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config(agent_id="signal-router")
    dispatched: list[str] = []

    original_send = async_dispatcher_send

    def capturing_send(h: HomeAssistant, signal: str, *args: object) -> None:
        dispatched.append(signal)
        original_send(h, signal, *args)

    with (
        patch(
            "custom_components.neuralbridge.ollama_client.OllamaClient.generate",
            new_callable=AsyncMock,
            return_value='{"local_ha": false, "complexity": 40}',
        ),
        patch(
            "custom_components.neuralbridge.conversation.async_dispatcher_send",
            side_effect=capturing_send,
        ),
    ):
        await conv_agent._classify_with_router(router, "hello")

    expected = SIGNAL_STATS_UPDATED.format(entry_id=entry.entry_id)
    assert expected in dispatched


# ---------------------------------------------------------------------------
# Test 74 — NeuralBridgeAgent reads CircuitBreaker from hass.data
# ---------------------------------------------------------------------------


def test_agent_reads_circuit_breaker_from_hass_data(hass: HomeAssistant) -> None:
    """NeuralBridgeAgent uses the CircuitBreaker instance stored in hass.data."""
    cb = CircuitBreaker(failure_threshold=99)
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {DATA_CIRCUIT_BREAKER: cb}

    conv_agent = NeuralBridgeAgent(hass, entry)

    assert conv_agent._circuit_breaker is cb


# ---------------------------------------------------------------------------
# Test 75 — supported_features: CONTROL returned when enable_home_control is True
# ---------------------------------------------------------------------------


def test_supported_features_returns_control_when_enabled(hass: HomeAssistant) -> None:
    """supported_features includes CONTROL when CONF_ENABLE_HOME_CONTROL is True."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_ENABLE_HOME_CONTROL: True})
    conv_agent = NeuralBridgeAgent(hass, entry)

    assert conv_agent.supported_features == ConversationEntityFeature.CONTROL


# ---------------------------------------------------------------------------
# Test 76 — supported_features: 0 returned when enable_home_control is False
# ---------------------------------------------------------------------------


def test_supported_features_returns_zero_when_disabled(hass: HomeAssistant) -> None:
    """supported_features is 0 (no CONTROL) when CONF_ENABLE_HOME_CONTROL is False."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_ENABLE_HOME_CONTROL: False})
    conv_agent = NeuralBridgeAgent(hass, entry)

    assert conv_agent.supported_features == ConversationEntityFeature(0)


# ---------------------------------------------------------------------------
# Test 77 — supported_features: defaults to CONTROL when key is absent
# ---------------------------------------------------------------------------


def test_supported_features_defaults_to_control(hass: HomeAssistant) -> None:
    """supported_features defaults to CONTROL when CONF_ENABLE_HOME_CONTROL is not set."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    conv_agent = NeuralBridgeAgent(hass, entry)

    assert conv_agent.supported_features == ConversationEntityFeature.CONTROL


# ---------------------------------------------------------------------------
# Test 78 — _process_with_existing: assist_mode=True + action_done → result returned
# ---------------------------------------------------------------------------


async def test_process_with_existing_assist_mode_action_done_returns_result(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """In assist mode, an action_done response is returned as a ConversationResult."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.home_assistant",
        CONF_AGENT_NAME: "HA Local",
        CONF_AGENT_ASSIST_MODE: True,
    }

    service_response = {
        "response": {
            "response_type": "action_done",
            "speech": {"plain": {"speech": "OK, turning on the lights"}},
        }
    }

    mock_hass = MagicMock()
    mock_hass.config.language = "en"
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    result = await agent._process_with_existing(agent_cfg, _make_input("Turn on the lights"))

    assert result is not None
    assert result.response.speech["plain"]["speech"] == "OK, turning on the lights"


# ---------------------------------------------------------------------------
# Test 79 — _process_with_existing: assist_mode=True + response_type "error" → None
# ---------------------------------------------------------------------------


async def test_process_with_existing_assist_mode_error_response_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """In assist mode, an error response falls through (returns None) to the next agent."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.home_assistant",
        CONF_AGENT_NAME: "HA Local",
        CONF_AGENT_ASSIST_MODE: True,
    }

    service_response = {
        "response": {
            "response_type": "error",
            "speech": {"plain": {"speech": "I don't understand that command"}},
        }
    }

    mock_hass = MagicMock()
    mock_hass.config.language = "en"
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    result = await agent._process_with_existing(
        agent_cfg, _make_input("What is the capital of France?")
    )

    assert result is None


# ---------------------------------------------------------------------------
# Test 80 — _process_with_existing: assist_mode=True + response_type "plain" → None
# ---------------------------------------------------------------------------


async def test_process_with_existing_assist_mode_plain_response_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """In assist mode, a plain (non-intent) response falls through to the next agent."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.home_assistant",
        CONF_AGENT_NAME: "HA Local",
        CONF_AGENT_ASSIST_MODE: True,
    }

    service_response = {
        "response": {
            "response_type": "plain",
            "speech": {"plain": {"speech": "I'm not sure how to help with that"}},
        }
    }

    mock_hass = MagicMock()
    mock_hass.config.language = "en"
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    result = await agent._process_with_existing(agent_cfg, _make_input("Tell me a joke"))

    assert result is None


# ---------------------------------------------------------------------------
# Test 81 — _process_with_existing: assist_mode=False (default) → returns any speech
# ---------------------------------------------------------------------------


async def test_process_with_existing_assist_mode_off_returns_any_speech(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """With assist_mode=False, any speech response is returned (original behaviour)."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.home_assistant",
        CONF_AGENT_NAME: "HA Local",
        CONF_AGENT_ASSIST_MODE: False,
    }

    service_response = {
        "response": {
            "response_type": "plain",
            "speech": {"plain": {"speech": "I'm not sure how to help with that"}},
        }
    }

    mock_hass = MagicMock()
    mock_hass.config.language = "en"
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    result = await agent._process_with_existing(agent_cfg, _make_input("Tell me a joke"))

    assert result is not None
    assert result.response.speech["plain"]["speech"] == "I'm not sure how to help with that"


# ---------------------------------------------------------------------------
# Test 82 — _process_with_existing: assist_mode absent defaults to False (no change)
# ---------------------------------------------------------------------------


async def test_process_with_existing_assist_mode_absent_defaults_to_false(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """When assist_mode is absent from config, it defaults to False (returns any speech)."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.home_assistant",
        CONF_AGENT_NAME: "HA Local",
        # No CONF_AGENT_ASSIST_MODE key
    }

    service_response = {
        "response": {
            "response_type": "error",
            "speech": {"plain": {"speech": "Sorry, I can't do that"}},
        }
    }

    mock_hass = MagicMock()
    mock_hass.config.language = "en"
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    result = await agent._process_with_existing(agent_cfg, _make_input("Something"))

    assert result is not None
    assert result.response.speech["plain"]["speech"] == "Sorry, I can't do that"


# ---------------------------------------------------------------------------
# Test 83 — async_process: debug log uses character count, not user text (privacy)
# ---------------------------------------------------------------------------


async def test_async_process_logs_char_count_not_user_text(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, caplog: Any
) -> None:
    """async_process debug log records character count, not the user text (PII protection)."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    user_text = "This is a private message"

    with caplog.at_level(logging.DEBUG, logger="custom_components.neuralbridge.conversation"):
        await agent.async_process(_make_input(user_text))

    assert user_text not in caplog.text
    assert f"{len(user_text)} chars" in caplog.text


# ---------------------------------------------------------------------------
# Test 84 — _maybe_add_to_chat_log: no-op when no active ChatLog in context
# ---------------------------------------------------------------------------


def test_maybe_add_to_chat_log_noop_when_no_active_chat_log(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_maybe_add_to_chat_log is a no-op when current_chat_log.get() returns None."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    result = agent._create_result("Hello")

    mock_current = MagicMock()
    mock_current.get.return_value = None
    mock_module = MagicMock()
    mock_module.current_chat_log = mock_current

    with patch.dict("sys.modules", {"homeassistant.components.conversation.chat_log": mock_module}):
        agent._maybe_add_to_chat_log(result)

    mock_module.AssistantContent.assert_not_called()


# ---------------------------------------------------------------------------
# Test 85 — _maybe_add_to_chat_log: adds AssistantContent to the active ChatLog
# ---------------------------------------------------------------------------


def test_maybe_add_to_chat_log_adds_assistant_content(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_maybe_add_to_chat_log adds AssistantContent to the active ChatLog."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    result = agent._create_result("Hi there")

    mock_chat_log_obj = MagicMock()
    mock_current = MagicMock()
    mock_current.get.return_value = mock_chat_log_obj
    mock_assistant_content_cls = MagicMock()
    mock_module = MagicMock()
    mock_module.current_chat_log = mock_current
    mock_module.AssistantContent = mock_assistant_content_cls

    with patch.dict("sys.modules", {"homeassistant.components.conversation.chat_log": mock_module}):
        agent._maybe_add_to_chat_log(result)

    mock_chat_log_obj.async_add_assistant_content_without_tools.assert_called_once()
    mock_assistant_content_cls.assert_called_once_with(agent_id=agent.entity_id, content="Hi there")


# ---------------------------------------------------------------------------
# Test 86 — async_process: delegates to _compute_result and _maybe_add_to_chat_log
# ---------------------------------------------------------------------------


async def test_async_process_calls_maybe_add_to_chat_log(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_process passes its computed result to _maybe_add_to_chat_log."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)

    with patch.object(agent, "_maybe_add_to_chat_log") as mock_add:
        result = await agent.async_process(_make_input("test"))

    mock_add.assert_called_once_with(result)


# ===========================================================================
# Helper factory for LOCAL_HA agents (used in router-decision tests below)
# ===========================================================================


def _make_local_ha_agent(
    priority: int = 30,
    agent_id: str = "local-ha-1",
    name: str = "Local HA",
) -> dict[str, Any]:
    """Return a minimal LOCAL_HA agent config dict."""
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: name,
        CONF_PRIORITY: priority,
    }


# ===========================================================================
# RouterDecision dataclass behaviour (Tests 87-89)
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 87 — RouterDecision: immutable (setattr raises AttributeError)
# ---------------------------------------------------------------------------


def test_router_decision_is_immutable() -> None:
    """Setting any attribute on a RouterDecision raises AttributeError."""
    decision = RouterDecision(local_ha=True, complexity=50)

    with pytest.raises(AttributeError):
        decision.local_ha = False  # type: ignore[misc]

    with pytest.raises(AttributeError):
        decision.complexity = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 88 — RouterDecision: equality
# ---------------------------------------------------------------------------


def test_router_decision_equality() -> None:
    """Two RouterDecision instances with the same fields are equal."""
    assert RouterDecision(local_ha=True, complexity=5) == RouterDecision(
        local_ha=True, complexity=5
    )
    assert RouterDecision(local_ha=False, complexity=50) != RouterDecision(
        local_ha=True, complexity=50
    )
    assert RouterDecision(local_ha=True, complexity=1) != RouterDecision(
        local_ha=True, complexity=2
    )
    # Call __eq__ directly to verify NotImplemented is returned for non-RouterDecision
    sentinel = RouterDecision(local_ha=True, complexity=10).__eq__("not a decision")
    assert sentinel is NotImplemented


# ---------------------------------------------------------------------------
# Test 88b — RouterDecision: __hash__ works (usable in sets/dicts)
# ---------------------------------------------------------------------------


def test_router_decision_hashable() -> None:
    """RouterDecision can be used in sets and as dict keys."""
    d1 = RouterDecision(local_ha=True, complexity=5)
    d2 = RouterDecision(local_ha=True, complexity=5)
    d3 = RouterDecision(local_ha=False, complexity=10)

    decision_set = {d1, d2, d3}
    assert len(decision_set) == 2  # d1 and d2 are equal, so only 2 unique
    assert d1 in decision_set


# ---------------------------------------------------------------------------
# Test 89 — RouterDecision: repr contains both fields
# ---------------------------------------------------------------------------


def test_router_decision_repr() -> None:
    """RouterDecision repr includes local_ha and complexity values."""
    decision = RouterDecision(local_ha=True, complexity=42)
    text = repr(decision)

    assert "local_ha=True" in text
    assert "complexity=42" in text


# ===========================================================================
# _parse_router_response (Tests 90-101)
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 90 — _parse_router_response: valid JSON PASS → RouterDecision
# ---------------------------------------------------------------------------


def test_parse_router_response_valid_json_pass() -> None:
    """Valid JSON with complexity>0 returns a RouterDecision."""
    result = _parse_router_response('{"local_ha": false, "complexity": 30}')

    assert result is not None
    assert result.local_ha is False
    assert result.complexity == 30


# ---------------------------------------------------------------------------
# Test 91 — _parse_router_response: local_ha=true → promoted flag set
# ---------------------------------------------------------------------------


def test_parse_router_response_local_ha_true() -> None:
    """JSON with local_ha=true returns RouterDecision with local_ha=True."""
    result = _parse_router_response('{"local_ha": true, "complexity": 5}')

    assert result is not None
    assert result.local_ha is True
    assert result.complexity == 5


# ---------------------------------------------------------------------------
# Test 92 — _parse_router_response: complexity=0 → RouterDecision(complexity=0) block marker
# ---------------------------------------------------------------------------


def test_parse_router_response_complexity_zero_returns_block_decision() -> None:
    """complexity=0 in the JSON is the block signal; _parse_router_response returns RouterDecision(complexity=0)."""
    result = _parse_router_response('{"local_ha": false, "complexity": 0}')

    assert result is not None
    assert isinstance(result, RouterDecision)
    assert result.complexity == 0
    assert result.local_ha is False


# ---------------------------------------------------------------------------
# Test 93 — _parse_router_response: markdown fences stripped
# ---------------------------------------------------------------------------


def test_parse_router_response_markdown_fences_stripped() -> None:
    """JSON wrapped in ```json ... ``` markdown fences is parsed correctly."""
    raw = '```json\n{"local_ha": false, "complexity": 20}\n```'
    result = _parse_router_response(raw)

    assert result is not None
    assert result.complexity == 20


# ---------------------------------------------------------------------------
# Test 94 — _parse_router_response: malformed JSON → None
# ---------------------------------------------------------------------------


def test_parse_router_response_malformed_json_returns_none() -> None:
    """Invalid JSON returns None without raising."""
    assert _parse_router_response("not json at all") is None
    assert _parse_router_response("{broken") is None
    assert _parse_router_response("") is None
    # Has valid {..} delimiters but invalid JSON content inside
    assert _parse_router_response("{broken json}") is None


# ---------------------------------------------------------------------------
# Test 95 — _parse_router_response: missing complexity key → None
# ---------------------------------------------------------------------------


def test_parse_router_response_missing_complexity_returns_none() -> None:
    """JSON missing the 'complexity' key returns None."""
    assert _parse_router_response('{"local_ha": true}') is None


# ---------------------------------------------------------------------------
# Test 96 — _parse_router_response: missing local_ha key → None
# ---------------------------------------------------------------------------


def test_parse_router_response_missing_local_ha_returns_none() -> None:
    """JSON missing the 'local_ha' key returns None."""
    assert _parse_router_response('{"complexity": 40}') is None


# ---------------------------------------------------------------------------
# Test 97 — _parse_router_response: complexity clamped above 100 → 100
# ---------------------------------------------------------------------------


def test_parse_router_response_complexity_clamped_high() -> None:
    """complexity > 100 is clamped to 100."""
    result = _parse_router_response('{"local_ha": false, "complexity": 200}')

    assert result is not None
    assert result.complexity == 100


# ---------------------------------------------------------------------------
# Test 98 — _parse_router_response: negative non-zero complexity → clamped to 1
# ---------------------------------------------------------------------------


def test_parse_router_response_complexity_clamped_low_nonzero() -> None:
    """Negative complexity (not zero) is clamped to 1, not treated as block."""
    result = _parse_router_response('{"local_ha": false, "complexity": -10}')

    assert result is not None
    assert result.complexity == 1


# ---------------------------------------------------------------------------
# Test 99 — _parse_router_response: response with no JSON object → None
# ---------------------------------------------------------------------------


def test_parse_router_response_no_json_object_returns_none() -> None:
    """Plain text with no '{' returns None."""
    assert _parse_router_response("PASS") is None
    assert _parse_router_response("BLOCK this request") is None


# ---------------------------------------------------------------------------
# Test 100 — _parse_router_response: non-dict JSON (list) → None
# ---------------------------------------------------------------------------


def test_parse_router_response_non_dict_json_returns_none() -> None:
    """JSON array of primitives (no inner object) returns None."""
    assert _parse_router_response("[1, 2, 3]") is None
    assert _parse_router_response("[]") is None


# ---------------------------------------------------------------------------
# Test 101 — _parse_router_response: extra JSON fields are ignored
# ---------------------------------------------------------------------------


def test_parse_router_response_extra_fields_ignored() -> None:
    """Extra unknown fields in the JSON do not cause a failure."""
    result = _parse_router_response(
        '{"local_ha": true, "complexity": 15, "explanation": "Home automation task"}'
    )

    assert result is not None
    assert result.local_ha is True
    assert result.complexity == 15


# ===========================================================================
# _apply_router_decision (Tests 102-104)
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 102 — _apply_router_decision: local_ha=True promotes LOCAL_HA agents to front
# ---------------------------------------------------------------------------


def test_apply_router_decision_local_ha_promotes_local_agents() -> None:
    """With local_ha=True, LOCAL_HA agents are moved to the front of the list."""
    local = _make_local_ha_agent(priority=30, agent_id="ha-local")
    ollama = _make_ollama_agent(priority=50, agent_id="ollama-cloud")
    processing_agents = [ollama, local]

    decision = RouterDecision(local_ha=True, complexity=10)
    result = _apply_router_decision(decision, processing_agents)

    assert result[0]["id"] == "ha-local", "LOCAL_HA agent should be promoted to first position"


# ---------------------------------------------------------------------------
# Test 103 — _apply_router_decision: local_ha=False excludes LOCAL_HA agents
# ---------------------------------------------------------------------------


def test_apply_router_decision_local_ha_false_excludes_local_agents() -> None:
    """With local_ha=False, LOCAL_HA agents are removed from the list."""
    local = _make_local_ha_agent(agent_id="ha-local")
    ollama = _make_ollama_agent(agent_id="ollama-cloud")
    processing_agents = [local, ollama]

    decision = RouterDecision(local_ha=False, complexity=70)
    result = _apply_router_decision(decision, processing_agents)

    ids = [a["id"] for a in result]
    assert "ha-local" not in ids
    assert "ollama-cloud" in ids


# ---------------------------------------------------------------------------
# Test 104 — _apply_router_decision: local_ha=False with no LOCAL_HA agents
# ---------------------------------------------------------------------------


def test_apply_router_decision_local_ha_false_no_local_agents() -> None:
    """With local_ha=False and no LOCAL_HA agents, the list is returned unchanged."""
    ollama1 = _make_ollama_agent(priority=50, agent_id="a1")
    ollama2 = _make_ollama_agent(priority=40, agent_id="a2")
    processing_agents = [ollama1, ollama2]

    decision = RouterDecision(local_ha=False, complexity=50)
    result = _apply_router_decision(decision, processing_agents)

    assert len(result) == 2
    assert result[0]["id"] == "a1"
    assert result[1]["id"] == "a2"


# ===========================================================================
# CONF_IS_ROUTER agent-split in _compute_result (Tests 105-106)
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 105 — _compute_result: agent with is_router=True is treated as router
# ---------------------------------------------------------------------------


async def test_compute_result_uses_is_router_flag_for_split(hass: HomeAssistant) -> None:
    """Agent with CONF_IS_ROUTER=True is routed to router_agents, not processing_agents."""
    router_agent = {
        **_make_ollama_agent(priority=5, agent_id="my-router"),
        CONF_IS_ROUTER: True,
    }
    processing_agent = _make_ollama_agent(priority=10, agent_id="processor")
    entry = _entry_with_agents(router_agent, processing_agent)
    conv_agent = NeuralBridgeAgent(hass, entry)

    decision = RouterDecision(local_ha=False, complexity=50)

    with (
        patch.object(
            conv_agent, "_check_with_routers", new_callable=AsyncMock, return_value=decision
        ) as mock_router,
        patch.object(
            conv_agent,
            "_try_agent_with_tracking",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        await conv_agent.async_process(_make_input("hello"))

    # The router was called — verify the routers list contained our is_router agent
    call_args = mock_router.call_args
    routers_passed = call_args[0][1]  # second positional arg is router_agents
    router_ids = [a["id"] for a in routers_passed]
    assert "my-router" in router_ids
    assert "processor" not in router_ids


# ---------------------------------------------------------------------------
# Test 106 — _compute_result: router block (None) returns blocked response
# ---------------------------------------------------------------------------


async def test_compute_result_router_none_returns_blocked_response(
    hass: HomeAssistant,
) -> None:
    """When _check_with_routers returns None, async_process returns the block response."""
    router_agent = {
        **_make_ollama_agent(priority=0, agent_id="blocking-router"),
        CONF_IS_ROUTER: True,
    }
    entry = _entry_with_agents(router_agent, _make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    with patch.object(conv_agent, "_check_with_routers", new_callable=AsyncMock, return_value=None):
        result = await conv_agent.async_process(_make_input("bad content"))

    speech = result.response.speech["plain"]["speech"].lower()
    assert "cannot process" in speech


# ===========================================================================
# _check_with_routers edge cases (Tests 107-108)
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 107 — _check_with_routers: empty router list returns fail-open RouterDecision
# ---------------------------------------------------------------------------


async def test_check_with_routers_empty_list_returns_fail_open(
    hass: HomeAssistant,
) -> None:
    """With no routers, _check_with_routers returns the fail-open default RouterDecision."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    result = await conv_agent._check_with_routers(_make_input("hello"), [])

    assert result is not None
    assert isinstance(result, RouterDecision)
    assert result.complexity == DEFAULT_ROUTER_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 108 — _check_with_routers: all routers return fail-open on error → RouterDecision
# ---------------------------------------------------------------------------


async def test_check_with_routers_all_fail_open_returns_decision(
    hass: HomeAssistant,
) -> None:
    """When every router returns a fail-open RouterDecision, the last one is returned."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    fail_open = RouterDecision(local_ha=False, complexity=DEFAULT_ROUTER_COMPLEXITY)

    with patch.object(
        conv_agent,
        "_classify_with_router",
        new_callable=AsyncMock,
        return_value=fail_open,
    ):
        result = await conv_agent._check_with_routers(
            _make_input("some text"), [_make_router_config("r1"), _make_router_config("r2")]
        )

    assert result is not None
    assert result.complexity == DEFAULT_ROUTER_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 109 — _apply_router_decision: skip-routing sentinel passes all agents through
# ---------------------------------------------------------------------------


def test_apply_router_decision_skip_routing_passes_all_agents() -> None:
    """ROUTER_SKIP_ROUTING_COMPLEXITY sentinel returns all agents unchanged."""
    local = {CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA, "id": "local"}
    ollama = {CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA, "id": "ollama"}
    skip_decision = RouterDecision(local_ha=False, complexity=ROUTER_SKIP_ROUTING_COMPLEXITY)

    result = _apply_router_decision(skip_decision, [local, ollama])

    assert result == [local, ollama]


# ---------------------------------------------------------------------------
# Test 110 — _router_fallback: default_complexity returns mid-range RouterDecision
# ---------------------------------------------------------------------------


def test_router_fallback_default_complexity(hass: HomeAssistant) -> None:
    """_router_fallback with default_complexity returns RouterDecision(50)."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    result = conv_agent._router_fallback(ROUTER_FALLBACK_DEFAULT_COMPLEXITY)

    assert result is not None
    assert isinstance(result, RouterDecision)
    assert result.complexity == DEFAULT_ROUTER_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 111 — _router_fallback: skip_routing returns ROUTER_SKIP_ROUTING_COMPLEXITY
# ---------------------------------------------------------------------------


def test_router_fallback_skip_routing(hass: HomeAssistant) -> None:
    """_router_fallback with skip_routing returns RouterDecision with sentinel complexity."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    result = conv_agent._router_fallback(ROUTER_FALLBACK_SKIP_ROUTING)

    assert result is not None
    assert isinstance(result, RouterDecision)
    assert result.complexity == ROUTER_SKIP_ROUTING_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 112 — _router_fallback: block returns None
# ---------------------------------------------------------------------------


def test_router_fallback_block_returns_none(hass: HomeAssistant) -> None:
    """_router_fallback with block returns None (block signal)."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    result = conv_agent._router_fallback(ROUTER_FALLBACK_BLOCK)

    assert result is None


# ---------------------------------------------------------------------------
# Test 113 — _classify_with_router: AGENT_TYPE_EXISTING with valid entity + good response
# ---------------------------------------------------------------------------


async def test_classify_with_router_existing_entity_success(
    hass: HomeAssistant,
) -> None:
    """AGENT_TYPE_EXISTING router with a valid entity + JSON response returns RouterDecision."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = {
        "id": "existing-router",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "Cloud Router",
        CONF_PRIORITY: 0,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }
    json_response = '{"local_ha": false, "complexity": 42}'

    with patch.object(
        conv_agent,
        "_call_existing_agent_for_routing",
        new_callable=AsyncMock,
        return_value=json_response,
    ):
        result = await conv_agent._classify_with_router(router, "what is 2+2?")

    assert result is not None
    assert isinstance(result, RouterDecision)
    assert result.local_ha is False
    assert result.complexity == 42


# ---------------------------------------------------------------------------
# Test 114 — _classify_with_router: AGENT_TYPE_EXISTING returns empty → fallback
# ---------------------------------------------------------------------------


async def test_classify_with_router_existing_empty_response_fallback(
    hass: HomeAssistant,
) -> None:
    """AGENT_TYPE_EXISTING router returning empty string falls back to default complexity."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = {
        "id": "existing-router-empty",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "Cloud Router",
        CONF_PRIORITY: 0,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }

    with patch.object(
        conv_agent,
        "_call_existing_agent_for_routing",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await conv_agent._classify_with_router(router, "hello")

    assert result is not None
    assert result.complexity == DEFAULT_ROUTER_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 115 — _classify_with_router: fallback=block on error returns None
# ---------------------------------------------------------------------------


async def test_classify_with_router_fallback_block_returns_none(
    hass: HomeAssistant,
) -> None:
    """Router failure with fallback=block returns None (blocks the request)."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = {
        "id": "block-fallback-router",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "Strict Router",
        CONF_PRIORITY: 0,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_BLOCK,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }

    with patch.object(
        conv_agent,
        "_call_existing_agent_for_routing",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await conv_agent._classify_with_router(router, "some query")

    assert result is None


# ---------------------------------------------------------------------------
# Test 116 — _classify_with_router: fallback=skip_routing on error returns sentinel
# ---------------------------------------------------------------------------


async def test_classify_with_router_fallback_skip_routing_returns_sentinel(
    hass: HomeAssistant,
) -> None:
    """Router failure with fallback=skip_routing returns ROUTER_SKIP_ROUTING_COMPLEXITY."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = {
        "id": "skip-fallback-router",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "Lenient Router",
        CONF_PRIORITY: 0,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_SKIP_ROUTING,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }

    with patch.object(
        conv_agent,
        "_call_existing_agent_for_routing",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await conv_agent._classify_with_router(router, "some query")

    assert result is not None
    assert result.complexity == ROUTER_SKIP_ROUTING_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 117 — _classify_with_router: log_level=complexity_only logs the score
# ---------------------------------------------------------------------------


async def test_classify_with_router_log_complexity_only(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_level=complexity_only emits an INFO log with the complexity score."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = {
        "id": "log-complexity-router",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "Verbose Router",
        CONF_PRIORITY: 0,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_COMPLEXITY,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }

    with (
        patch.object(
            conv_agent,
            "_call_existing_agent_for_routing",
            new_callable=AsyncMock,
            return_value='{"local_ha": false, "complexity": 30}',
        ),
        caplog.at_level(logging.INFO, logger="custom_components.neuralbridge.conversation"),
    ):
        await conv_agent._classify_with_router(router, "some query")

    assert "complexity score" in caplog.text.lower() or "30" in caplog.text


# ---------------------------------------------------------------------------
# Test 118 — _classify_with_router: log_level=debug_info logs the decision
# ---------------------------------------------------------------------------


async def test_classify_with_router_log_debug_info(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_level=debug_info emits a DEBUG log with the routing decision."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = {
        "id": "log-debug-router",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "Debug Router",
        CONF_PRIORITY: 0,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_DEBUG,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }

    with (
        patch.object(
            conv_agent,
            "_call_existing_agent_for_routing",
            new_callable=AsyncMock,
            return_value='{"local_ha": true, "complexity": 15}',
        ),
        caplog.at_level(logging.DEBUG, logger="custom_components.neuralbridge.conversation"),
    ):
        await conv_agent._classify_with_router(router, "turn on lights")

    assert "decision" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Test 119 — _classify_with_router: log_level=debug_with_query logs the query
# ---------------------------------------------------------------------------


async def test_classify_with_router_log_debug_with_query(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_level=debug_with_query emits a DEBUG log containing the query text."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = {
        "id": "log-query-router",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "Trace Router",
        CONF_PRIORITY: 0,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_DEBUG_QUERY,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }

    with (
        patch.object(
            conv_agent,
            "_call_existing_agent_for_routing",
            new_callable=AsyncMock,
            return_value='{"local_ha": false, "complexity": 20}',
        ),
        caplog.at_level(logging.DEBUG, logger="custom_components.neuralbridge.conversation"),
    ):
        await conv_agent._classify_with_router(router, "unique-query-marker")

    assert "unique-query-marker" in caplog.text


# ---------------------------------------------------------------------------
# Test 120 — _classify_with_router: custom prompt is sent to the back-end
# ---------------------------------------------------------------------------


async def test_classify_with_router_uses_custom_prompt(
    hass: HomeAssistant,
) -> None:
    """When CONF_ROUTER_CUSTOM_PROMPT is set, it is used instead of the default prompt."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    custom = 'Reply ONLY: {"local_ha": false, "complexity": 1}. User: {user_text}'
    router = {
        "id": "custom-prompt-router",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "Custom Router",
        CONF_PRIORITY: 0,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: custom,
    }

    captured_prompt: list[str] = []

    async def capture_call(entity_id: str, prompt: str, timeout: int) -> str | None:
        captured_prompt.append(prompt)
        return '{"local_ha": false, "complexity": 1}'

    with patch.object(conv_agent, "_call_existing_agent_for_routing", side_effect=capture_call):
        await conv_agent._classify_with_router(router, "hello world")

    assert len(captured_prompt) == 1
    assert "{user_text}" not in captured_prompt[0]  # placeholder was expanded
    assert "hello world" in captured_prompt[0]
    assert "Reply ONLY" in captured_prompt[0]


# ---------------------------------------------------------------------------
# Test 121 — _classify_with_router: unknown agent type records failure + fallback
# ---------------------------------------------------------------------------


async def test_classify_with_router_unknown_type_records_failure(
    hass: HomeAssistant,
) -> None:
    """Unknown router agent type records a failure and applies the fallback."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = {
        "id": "unknown-type-router",
        CONF_AGENT_TYPE: "totally_unknown",
        CONF_AGENT_NAME: "Mystery Router",
        CONF_PRIORITY: 0,
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }

    result = await conv_agent._classify_with_router(router, "some text")

    assert result is not None
    assert result.complexity == DEFAULT_ROUTER_COMPLEXITY
    stats = conv_agent._statistics.get_agent_stats("unknown-type-router")
    assert stats is not None
    assert stats.failures == 1


# ---------------------------------------------------------------------------
# Test 122 — _call_existing_agent_for_routing: entity not in states → None
# ---------------------------------------------------------------------------


async def test_call_existing_agent_entity_not_found(hass: HomeAssistant) -> None:
    """_call_existing_agent_for_routing returns None if the entity state is absent."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    # hass.states won't have this entity
    result = await conv_agent._call_existing_agent_for_routing(
        "conversation.nonexistent", "classify this", 5
    )
    assert result is None


# ---------------------------------------------------------------------------
# Test 123 — _call_existing_agent_for_routing: service call times out → None
# ---------------------------------------------------------------------------


async def test_call_existing_agent_timeout(hass: HomeAssistant) -> None:
    """_call_existing_agent_for_routing returns None when the service call times out."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    mock_hass = MagicMock()
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(side_effect=asyncio.TimeoutError)
    conv_agent.hass = mock_hass

    result = await conv_agent._call_existing_agent_for_routing("conversation.gemini", "classify", 1)
    assert result is None


# ---------------------------------------------------------------------------
# Test 124 — _call_existing_agent_for_routing: service call raises exception → None
# ---------------------------------------------------------------------------


async def test_call_existing_agent_exception(hass: HomeAssistant) -> None:
    """_call_existing_agent_for_routing returns None when the service call raises."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    mock_hass = MagicMock()
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(side_effect=RuntimeError("service error"))
    conv_agent.hass = mock_hass

    result = await conv_agent._call_existing_agent_for_routing("conversation.gemini", "classify", 5)
    assert result is None


# ---------------------------------------------------------------------------
# Test 125 — _call_existing_agent_for_routing: service returns speech → text
# ---------------------------------------------------------------------------


async def test_call_existing_agent_success(hass: HomeAssistant) -> None:
    """_call_existing_agent_for_routing extracts speech from the service response."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    service_response = {
        "response": {"speech": {"plain": {"speech": '{"local_ha": false, "complexity": 10}'}}}
    }
    mock_hass = MagicMock()
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    conv_agent.hass = mock_hass

    result = await conv_agent._call_existing_agent_for_routing(
        "conversation.gemini", "some prompt", 5
    )
    assert result == '{"local_ha": false, "complexity": 10}'


# ---------------------------------------------------------------------------
# Test 126 — _call_existing_agent_for_routing: empty speech → None
# ---------------------------------------------------------------------------


async def test_call_existing_agent_empty_speech(hass: HomeAssistant) -> None:
    """_call_existing_agent_for_routing returns None when speech text is empty."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    service_response = {"response": {"speech": {"plain": {"speech": ""}}}}
    mock_hass = MagicMock()
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    conv_agent.hass = mock_hass

    result = await conv_agent._call_existing_agent_for_routing(
        "conversation.gemini", "some prompt", 5
    )
    assert result is None


# ---------------------------------------------------------------------------
# Test 127 — _call_existing_agent_for_routing: no response key → None
# ---------------------------------------------------------------------------


async def test_call_existing_agent_no_response_key(hass: HomeAssistant) -> None:
    """_call_existing_agent_for_routing returns None when service returns no response key."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)

    mock_hass = MagicMock()
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value={})
    conv_agent.hass = mock_hass

    result = await conv_agent._call_existing_agent_for_routing("conversation.gemini", "prompt", 5)
    assert result is None


# ---------------------------------------------------------------------------
# Test 128 — _classify_with_router: AGENT_TYPE_EXISTING unparseable JSON → fallback
# ---------------------------------------------------------------------------


async def test_classify_with_router_existing_unparseable_json_fallback(
    hass: HomeAssistant,
) -> None:
    """AGENT_TYPE_EXISTING router returning unparseable JSON applies configured fallback."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = {
        "id": "parse-fail-router",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_NAME: "Bad JSON Router",
        CONF_PRIORITY: 0,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }

    with patch.object(
        conv_agent,
        "_call_existing_agent_for_routing",
        new_callable=AsyncMock,
        return_value="not valid json at all",
    ):
        result = await conv_agent._classify_with_router(router, "any query")

    assert result is not None
    assert result.complexity == DEFAULT_ROUTER_COMPLEXITY


# ---------------------------------------------------------------------------
# Test 129 — async_process: skip_routing sentinel lets all agents run in order
# ---------------------------------------------------------------------------


async def test_async_process_skip_routing_uses_all_agents(hass: HomeAssistant) -> None:
    """When routing fallback=skip_routing fires, all processing agents are tried."""
    local = _make_local_ha_agent(priority=5, agent_id="local-skip")
    ollama = _make_ollama_agent(priority=50, agent_id="ollama-skip")
    router = {
        "id": "skip-router",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_IS_ROUTER: True,
        CONF_AGENT_NAME: "Skip Router",
        CONF_PRIORITY: 0,
        CONF_AGENT_ENABLED: True,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: 5,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_SKIP_ROUTING,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }
    entry = _entry_with_agents(local, ollama, router)
    conv_agent = NeuralBridgeAgent(hass, entry)

    # Router returns skip sentinel
    skip_decision = RouterDecision(local_ha=False, complexity=ROUTER_SKIP_ROUTING_COMPLEXITY)

    with (
        patch.object(
            conv_agent, "_check_with_routers", new_callable=AsyncMock, return_value=skip_decision
        ),
        patch.object(
            conv_agent,
            "_try_processing_agents",
            new_callable=AsyncMock,
        ) as mock_try,
    ):
        ok_response = intent.IntentResponse(language="en")
        ok_response.async_set_speech("ok")
        mock_try.return_value = ConversationResult(response=ok_response, conversation_id=None)
        await conv_agent.async_process(_make_input("some query"))

    # Assert both agents were passed (skip means all pass through)
    call_agents = mock_try.call_args[0][0]
    agent_ids = [a["id"] for a in call_agents]
    assert "local-skip" in agent_ids
    assert "ollama-skip" in agent_ids


# ---------------------------------------------------------------------------
# Helpers for web search tests
# ---------------------------------------------------------------------------


def _make_web_search_agent(
    priority: int = 40,
    agent_id: str = "ws-agent-1",
    name: str = "Web Search",
    enabled: bool = True,
) -> dict[str, Any]:
    """Return a minimal web search agent config dict."""
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_WEB_SEARCH,
        CONF_AGENT_ENABLED: enabled,
        CONF_AGENT_NAME: name,
        CONF_PRIORITY: priority,
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
        CONF_SEARCH_API_KEY: "brave-test-key",
        CONF_SEARCH_RESULT_COUNT: 3,
        CONF_TIMEOUT: 10,
    }


# ---------------------------------------------------------------------------
# Test 130 — RouterDecision with web_search field
# ---------------------------------------------------------------------------


def test_router_decision_web_search_defaults_false() -> None:
    """RouterDecision.web_search defaults to False when not supplied."""
    decision = RouterDecision(local_ha=False, complexity=50)
    assert decision.web_search is False


def test_router_decision_web_search_can_be_true() -> None:
    """RouterDecision stores web_search=True correctly."""
    decision = RouterDecision(local_ha=False, complexity=40, web_search=True)
    assert decision.web_search is True


def test_router_decision_equality_includes_web_search() -> None:
    """Two RouterDecisions with different web_search values are not equal."""
    d1 = RouterDecision(local_ha=False, complexity=40, web_search=True)
    d2 = RouterDecision(local_ha=False, complexity=40, web_search=False)
    assert d1 != d2


def test_router_decision_hash_includes_web_search() -> None:
    """RouterDecision hash differs when web_search differs."""
    d1 = RouterDecision(local_ha=False, complexity=40, web_search=True)
    d2 = RouterDecision(local_ha=False, complexity=40, web_search=False)
    assert hash(d1) != hash(d2)


def test_router_decision_repr_includes_web_search() -> None:
    """RouterDecision repr includes all three fields."""
    d = RouterDecision(local_ha=True, complexity=30, web_search=True)
    r = repr(d)
    assert "web_search=True" in r
    assert "local_ha=True" in r
    assert "complexity=30" in r


def test_router_decision_immutability_web_search() -> None:
    """RouterDecision with web_search cannot be mutated."""
    d = RouterDecision(local_ha=False, complexity=50, web_search=True)
    with pytest.raises(AttributeError):
        d.web_search = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 131 — _parse_router_response with web_search field
# ---------------------------------------------------------------------------


def test_parse_router_response_with_web_search_true() -> None:
    """web_search: true in JSON is parsed into RouterDecision.web_search=True."""
    raw = '{"local_ha": false, "web_search": true, "complexity": 40}'
    result = _parse_router_response(raw)
    assert result is not None
    assert result.web_search is True
    assert result.local_ha is False
    assert result.complexity == 40


def test_parse_router_response_without_web_search_defaults_false() -> None:
    """Absent web_search key defaults RouterDecision.web_search to False."""
    raw = '{"local_ha": false, "complexity": 30}'
    result = _parse_router_response(raw)
    assert result is not None
    assert result.web_search is False


def test_parse_router_response_web_search_false_explicit() -> None:
    """Explicit web_search: false is stored as False."""
    raw = '{"local_ha": true, "web_search": false, "complexity": 5}'
    result = _parse_router_response(raw)
    assert result is not None
    assert result.web_search is False
    assert result.local_ha is True


# ---------------------------------------------------------------------------
# Test 132 — _apply_router_decision web_search routing
# ---------------------------------------------------------------------------


def test_apply_router_decision_web_search_promotes_web_agents() -> None:
    """web_search=True moves WEB_SEARCH agents to the front."""
    local = {**_make_local_ha_agent(priority=10, agent_id="local"), CONF_AGENT_ENABLED: True}
    ollama = {**_make_ollama_agent(priority=30, agent_id="ollama"), CONF_AGENT_ENABLED: True}
    web = {**_make_web_search_agent(priority=50, agent_id="ws"), CONF_AGENT_ENABLED: True}

    decision = RouterDecision(local_ha=False, complexity=40, web_search=True)
    ordered = _apply_router_decision(decision, [local, ollama, web])

    assert ordered[0]["id"] == "ws"


def test_apply_router_decision_web_search_also_includes_other_agents() -> None:
    """After promotion, non-web-search agents are still included."""
    local = {**_make_local_ha_agent(priority=10, agent_id="local"), CONF_AGENT_ENABLED: True}
    ollama = {**_make_ollama_agent(priority=30, agent_id="ollama"), CONF_AGENT_ENABLED: True}
    web = {**_make_web_search_agent(priority=50, agent_id="ws"), CONF_AGENT_ENABLED: True}

    decision = RouterDecision(local_ha=False, complexity=40, web_search=True)
    ordered = _apply_router_decision(decision, [local, ollama, web])

    ids = [a["id"] for a in ordered]
    assert "ollama" in ids
    assert "local" in ids


def test_apply_router_decision_web_search_multiple_web_agents_ordered() -> None:
    """Multiple WEB_SEARCH agents are promoted and keep their relative order."""
    ws1 = {**_make_web_search_agent(priority=20, agent_id="ws1"), CONF_AGENT_ENABLED: True}
    ws2 = {**_make_web_search_agent(priority=60, agent_id="ws2"), CONF_AGENT_ENABLED: True}
    ollama = {**_make_ollama_agent(priority=40, agent_id="ol"), CONF_AGENT_ENABLED: True}

    decision = RouterDecision(local_ha=False, complexity=40, web_search=True)
    ordered = _apply_router_decision(decision, [ws1, ollama, ws2])

    assert ordered[0]["id"] == "ws1"
    assert ordered[1]["id"] == "ws2"


def test_apply_router_decision_no_web_search_flag_does_not_promote() -> None:
    """web_search=False keeps normal local_ha routing logic."""
    local = {**_make_local_ha_agent(priority=10, agent_id="local"), CONF_AGENT_ENABLED: True}
    web = {**_make_web_search_agent(priority=50, agent_id="ws"), CONF_AGENT_ENABLED: True}

    decision = RouterDecision(local_ha=True, complexity=5, web_search=False)
    ordered = _apply_router_decision(decision, [local, web])

    # local_ha=True promotes LOCAL_HA agents; web search not at front
    assert ordered[0]["id"] == "local"


# ---------------------------------------------------------------------------
# Test 133 — _try_agent dispatches to _process_with_web_search
# ---------------------------------------------------------------------------


async def test_try_agent_dispatches_to_web_search(hass: HomeAssistant) -> None:
    """_try_agent calls _process_with_web_search for AGENT_TYPE_WEB_SEARCH."""
    entry = _entry_with_agents(_make_web_search_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    agent_config = _make_web_search_agent()

    ok_response = intent.IntentResponse(language="en")
    ok_response.async_set_speech("search result")
    expected = ConversationResult(response=ok_response, conversation_id=None)

    with patch.object(
        conv_agent,
        "_process_with_web_search",
        new_callable=AsyncMock,
        return_value=expected,
    ) as mock_ws:
        result, timed_out = await conv_agent._try_agent(agent_config, _make_input("latest news"))

    mock_ws.assert_called_once()
    assert result is expected
    assert timed_out is False


# ---------------------------------------------------------------------------
# Test 134 — _process_with_web_search
# ---------------------------------------------------------------------------


async def test_process_with_web_search_success_returns_result(hass: HomeAssistant) -> None:
    """_process_with_web_search returns a ConversationResult on success."""
    entry = _entry_with_agents(_make_web_search_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    agent_config = _make_web_search_agent()

    with patch("custom_components.neuralbridge.conversation.WebSearchClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.search_and_summarise = AsyncMock(return_value="Here is what I found: ...")
        mock_cls.return_value = mock_client

        result = await conv_agent._process_with_web_search(agent_config, _make_input("news"))

    assert result is not None
    speech = result.response.speech.get("plain", {}).get("speech", "")
    assert "Here is what I found" in speech


async def test_process_with_web_search_no_results_returns_none(hass: HomeAssistant) -> None:
    """_process_with_web_search returns None when search returns no summary."""
    entry = _entry_with_agents(_make_web_search_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    agent_config = _make_web_search_agent()

    with patch("custom_components.neuralbridge.conversation.WebSearchClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.search_and_summarise = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        result = await conv_agent._process_with_web_search(agent_config, _make_input("q"))

    assert result is None


async def test_process_with_web_search_caches_client_by_agent_id(hass: HomeAssistant) -> None:
    """A WebSearchClient is created once per agent_id and reused."""
    entry = _entry_with_agents(_make_web_search_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    agent_config = _make_web_search_agent(agent_id="ws-cache-test")

    with patch("custom_components.neuralbridge.conversation.WebSearchClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.search_and_summarise = AsyncMock(return_value="result")
        mock_cls.return_value = mock_client

        await conv_agent._process_with_web_search(agent_config, _make_input("q1"))
        await conv_agent._process_with_web_search(agent_config, _make_input("q2"))

    # Constructor called only once despite two calls
    assert mock_cls.call_count == 1


# ---------------------------------------------------------------------------
# Test 135 — async_process routes to web search agent via router
# ---------------------------------------------------------------------------


async def test_async_process_web_search_agent_handles_timely_query(
    hass: HomeAssistant,
) -> None:
    """When router sets web_search=True the web search agent is promoted and called."""
    ws_agent = _make_web_search_agent(priority=50, agent_id="ws-integration")
    entry = _entry_with_agents(ws_agent)
    conv_agent = NeuralBridgeAgent(hass, entry)

    web_decision = RouterDecision(local_ha=False, complexity=40, web_search=True)
    ok_response = intent.IntentResponse(language="en")
    ok_response.async_set_speech("The PM is Jane Smith.")
    ws_result = ConversationResult(response=ok_response, conversation_id=None)

    with (
        patch.object(
            conv_agent,
            "_check_with_routers",
            new_callable=AsyncMock,
            return_value=web_decision,
        ),
        patch.object(
            conv_agent,
            "_try_agent_with_tracking",
            new_callable=AsyncMock,
            return_value=ws_result,
        ) as mock_try,
    ):
        result = await conv_agent.async_process(
            _make_input("Who is the current UK prime minister?")
        )

    mock_try.assert_called_once()
    first_agent = mock_try.call_args[0][0]
    assert first_agent[CONF_AGENT_TYPE] == AGENT_TYPE_WEB_SEARCH
    speech = result.response.speech.get("plain", {}).get("speech", "")
    assert "Jane Smith" in speech


# ===========================================================================
# Feature: Area-aware routing (_get_device_area)
# ===========================================================================


def test_get_device_area_returns_none_for_none_device_id(hass: HomeAssistant) -> None:
    """_get_device_area returns None when device_id is None."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    assert agent._get_device_area(None) is None


def test_get_device_area_returns_none_for_unknown_device(hass: HomeAssistant) -> None:
    """_get_device_area returns None when the device_id is not in the registry."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    with patch("custom_components.neuralbridge.conversation.dr.async_get") as mock_dr:
        mock_dr.return_value.async_get.return_value = None
        result = agent._get_device_area("unknown-device-id")

    assert result is None


def test_get_device_area_returns_none_for_device_without_area(hass: HomeAssistant) -> None:
    """_get_device_area returns None when the device has no area_id."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    mock_device = MagicMock()
    mock_device.area_id = None

    with patch("custom_components.neuralbridge.conversation.dr.async_get") as mock_dr:
        mock_dr.return_value.async_get.return_value = mock_device
        result = agent._get_device_area("device-no-area")

    assert result is None


def test_get_device_area_returns_none_when_area_registry_returns_none(
    hass: HomeAssistant,
) -> None:
    """_get_device_area returns None when the area_id is not in the area registry."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    mock_device = MagicMock()
    mock_device.area_id = "area-ghost"

    with (
        patch("custom_components.neuralbridge.conversation.dr.async_get") as mock_dr,
        patch("custom_components.neuralbridge.conversation.ar.async_get") as mock_ar,
    ):
        mock_dr.return_value.async_get.return_value = mock_device
        mock_ar.return_value.async_get_area.return_value = None
        result = agent._get_device_area("device-with-ghost-area")

    assert result is None


def test_get_device_area_returns_area_name(hass: HomeAssistant) -> None:
    """_get_device_area returns the area name when device and area are known."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    mock_device = MagicMock()
    mock_device.area_id = "area-kitchen"
    mock_area = MagicMock()
    mock_area.name = "Kitchen"

    with (
        patch("custom_components.neuralbridge.conversation.dr.async_get") as mock_dr,
        patch("custom_components.neuralbridge.conversation.ar.async_get") as mock_ar,
    ):
        mock_dr.return_value.async_get.return_value = mock_device
        mock_ar.return_value.async_get_area.return_value = mock_area
        result = agent._get_device_area("device-kitchen-echo")

    assert result == "Kitchen"


async def test_classify_with_router_injects_area_context_into_prompt(
    hass: HomeAssistant,
) -> None:
    """_classify_with_router appends 'Device area: ...' when area_context is provided."""
    router = {
        "id": "router-1",
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_NAME: "Router",
        CONF_OLLAMA_URL: "http://localhost:11434",
        CONF_OLLAMA_MODEL: "qwen2.5:0.5b",
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
        CONF_TIMEOUT: 5,
    }
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    captured: list[str] = []

    async def _mock_backend(_cfg: dict, prompt: str) -> str:
        captured.append(prompt)
        return '{"local_ha": true, "web_search": false, "complexity": 10}'

    with patch.object(conv_agent, "_call_router_backend", side_effect=_mock_backend):
        decision = await conv_agent._classify_with_router(
            router, "Turn on the lights", area_context="Kitchen"
        )

    assert decision is not None
    assert "Device area: Kitchen" in captured[0]


async def test_classify_with_router_no_area_context_omits_area_line(
    hass: HomeAssistant,
) -> None:
    """_classify_with_router does not add a 'Device area' line when area_context is None."""
    router = {
        "id": "router-1",
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_NAME: "Router",
        CONF_OLLAMA_URL: "http://localhost:11434",
        CONF_OLLAMA_MODEL: "qwen2.5:0.5b",
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
        CONF_TIMEOUT: 5,
    }
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    captured: list[str] = []

    async def _mock_backend(_cfg: dict, prompt: str) -> str:
        captured.append(prompt)
        return '{"local_ha": false, "web_search": false, "complexity": 30}'

    with patch.object(conv_agent, "_call_router_backend", side_effect=_mock_backend):
        await conv_agent._classify_with_router(router, "What is the boiling point of water?")

    assert "Device area" not in captured[0]


async def test_check_with_routers_passes_area_context_to_classify(
    hass: HomeAssistant,
) -> None:
    """_check_with_routers resolves device area and forwards it to each router."""
    router = {
        "id": "router-1",
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_NAME: "Router",
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
        CONF_TIMEOUT: 5,
    }
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    user_input = ConversationInput(
        text="Turn on the lights",
        context=Context(),
        conversation_id=None,
        device_id="device-kitchen",
        language="en",
        satellite_id=None,
        agent_id=None,
    )

    calls: list[tuple] = []

    async def _mock_classify(
        router_cfg: dict,
        user_text: str,
        area_context: str | None = None,
        language: str | None = None,
        agent_types: list[str] | None = None,
    ) -> RouterDecision:
        calls.append((user_text, area_context, language))
        return RouterDecision(local_ha=True, complexity=10)

    with (
        patch.object(conv_agent, "_get_device_area", return_value="Living Room"),
        patch.object(conv_agent, "_classify_with_router", side_effect=_mock_classify),
    ):
        await conv_agent._check_with_routers(user_input, [router])

    assert len(calls) == 1
    assert calls[0] == ("Turn on the lights", "Living Room", "en")


async def test_process_with_ollama_injects_area_context(hass: HomeAssistant) -> None:
    """_process_with_ollama prepends '[Area: X]' to the user message when area is known."""
    ollama_agent = _make_ollama_agent()
    entry = _entry_with_agents(ollama_agent)
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    user_input = ConversationInput(
        text="turn on the lights",
        context=Context(),
        conversation_id=None,
        device_id="device-bedroom",
        language="en",
        satellite_id=None,
        agent_id=None,
    )

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value="OK, bedroom lights on.")

    with (
        patch.object(conv_agent, "_get_device_area", return_value="Bedroom"),
        patch(
            "custom_components.neuralbridge.conversation.OllamaClient",
            return_value=mock_client,
        ),
    ):
        result = await conv_agent._process_with_ollama(ollama_agent, user_input)

    assert result is not None
    messages = mock_client.chat.call_args[0][0]
    user_msg = messages[-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "[Area: Bedroom] turn on the lights"


async def test_process_with_ollama_no_area_no_prefix(hass: HomeAssistant) -> None:
    """_process_with_ollama sends the original text unmodified when there is no area."""
    ollama_agent = _make_ollama_agent()
    entry = _entry_with_agents(ollama_agent)
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value="It is 14:30.")

    with (
        patch.object(conv_agent, "_get_device_area", return_value=None),
        patch(
            "custom_components.neuralbridge.conversation.OllamaClient",
            return_value=mock_client,
        ),
    ):
        result = await conv_agent._process_with_ollama(
            ollama_agent, _make_input("What time is it?")
        )

    assert result is not None
    messages = mock_client.chat.call_args[0][0]
    user_msg = messages[-1]
    assert user_msg["content"] == "What time is it?"


# ===========================================================================
# Feature: Explicit agent override (_check_explicit_agent_override)
# ===========================================================================


def test_check_explicit_agent_override_returns_none_when_no_match(
    hass: HomeAssistant,
) -> None:
    """No override prefix → returns None (follow normal routing)."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    agents = [_make_ollama_agent(name="Gemini")]
    result = conv_agent._check_explicit_agent_override(
        _make_input("What is the weather today?"), agents
    )

    assert result is None


def test_check_explicit_agent_override_ask_prefix(hass: HomeAssistant) -> None:
    """'Ask {name}: {query}' detects the override and strips the prefix."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    agent = _make_ollama_agent(name="Gemini")
    user_input = _make_input("Ask Gemini: What is the weather today?")

    result = conv_agent._check_explicit_agent_override(user_input, [agent])

    assert result is not None
    stripped_input, matched = result
    assert stripped_input.text == "What is the weather today?"
    assert matched == [agent]


def test_check_explicit_agent_override_use_prefix(hass: HomeAssistant) -> None:
    """'Use {name}: {query}' detects the override and strips the prefix."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    agent = _make_ollama_agent(name="Ollama")
    user_input = _make_input("Use Ollama: Summarise the news")

    result = conv_agent._check_explicit_agent_override(user_input, [agent])

    assert result is not None
    stripped_input, matched = result
    assert stripped_input.text == "Summarise the news"
    assert matched == [agent]


def test_check_explicit_agent_override_name_only_prefix(hass: HomeAssistant) -> None:
    """'{name}: {query}' plain prefix also triggers the override."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    agent = _make_ollama_agent(name="ChatGPT")
    user_input = _make_input("ChatGPT: write me a poem about robots")

    result = conv_agent._check_explicit_agent_override(user_input, [agent])

    assert result is not None
    stripped_input, _ = result
    assert stripped_input.text == "write me a poem about robots"


def test_check_explicit_agent_override_case_insensitive(hass: HomeAssistant) -> None:
    """Override prefix matching is case-insensitive."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    agent = _make_ollama_agent(name="Gemini")

    for text in ("ask gemini: hello", "ASK GEMINI: hello", "Ask GEMINI: hello"):
        result = conv_agent._check_explicit_agent_override(_make_input(text), [agent])
        assert result is not None, f"Expected match for '{text}'"
        assert result[0].text == "hello"


def test_check_explicit_agent_override_empty_query_not_matched(
    hass: HomeAssistant,
) -> None:
    """An override prefix with an empty body (only whitespace) is not matched."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    agent = _make_ollama_agent(name="Gemini")

    for text in ("Ask Gemini:", "Ask Gemini:   ", "Gemini:"):
        result = conv_agent._check_explicit_agent_override(_make_input(text), [agent])
        assert result is None, f"Expected no match for '{text}'"


def test_check_explicit_agent_override_no_agents_returns_none(
    hass: HomeAssistant,
) -> None:
    """With an empty agent list, override always returns None."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    result = conv_agent._check_explicit_agent_override(_make_input("Ask Gemini: anything"), [])

    assert result is None


def test_check_explicit_agent_override_agent_without_name_skipped(
    hass: HomeAssistant,
) -> None:
    """An agent with no name is skipped and does not cause a match."""
    entry = _entry_with_agents()
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    nameless = {
        "id": "agent-nameless",
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_NAME: "",
        CONF_PRIORITY: 50,
    }

    result = conv_agent._check_explicit_agent_override(_make_input(": hello"), [nameless])
    assert result is None


async def test_compute_result_uses_explicit_agent_override(hass: HomeAssistant) -> None:
    """When an override prefix is detected, routing is bypassed and the named agent is used."""
    gemini = _make_ollama_agent(name="Gemini", agent_id="gemini-id", priority=60)
    ollama_local = _make_ollama_agent(name="Ollama Local", agent_id="ollama-id", priority=10)
    entry = _entry_with_agents(gemini, ollama_local)
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    ok_response = intent.IntentResponse(language="en")
    ok_response.async_set_speech("Gemini says hello.")
    gemini_result = ConversationResult(response=ok_response, conversation_id=None)

    called_agents: list[str] = []

    async def _mock_try(agent_cfg: dict, _input: ConversationInput) -> ConversationResult | None:
        called_agents.append(agent_cfg.get(CONF_AGENT_NAME, ""))
        if agent_cfg.get(CONF_AGENT_NAME) == "Gemini":
            return gemini_result
        return None

    with patch.object(conv_agent, "_try_agent_with_tracking", side_effect=_mock_try):
        result = await conv_agent._compute_result(_make_input("Ask Gemini: Who invented Python?"))

    # Only Gemini should have been tried; Ollama Local must not be called
    assert called_agents == ["Gemini"]
    speech = result.response.speech.get("plain", {}).get("speech", "")
    assert "Gemini says hello." in speech


async def test_compute_result_override_stripped_text_forwarded(
    hass: HomeAssistant,
) -> None:
    """The stripped query (without the override prefix) is forwarded to the agent."""
    agent = _make_ollama_agent(name="Claude", agent_id="claude-id")
    entry = _entry_with_agents(agent)
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    received_texts: list[str] = []

    async def _mock_try(_cfg: dict, user_input: ConversationInput) -> ConversationResult | None:
        received_texts.append(user_input.text)
        resp = intent.IntentResponse(language="en")
        resp.async_set_speech("Done.")
        return ConversationResult(response=resp, conversation_id=None)

    with patch.object(conv_agent, "_try_agent_with_tracking", side_effect=_mock_try):
        await conv_agent._compute_result(_make_input("Use Claude: Tell me about the Turing test"))

    assert received_texts == ["Tell me about the Turing test"]


# ===========================================================================
# Feature: Fallback response includes recovery suggestion
# ===========================================================================


async def test_fallback_response_contains_recovery_suggestion(
    hass: HomeAssistant,
) -> None:
    """When all agents fail, the response includes a helpful rephrasing suggestion."""
    agent = _make_ollama_agent()
    entry = _entry_with_agents(agent)
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    with patch.object(
        conv_agent, "_try_agent_with_tracking", new_callable=AsyncMock, return_value=None
    ):
        result = await conv_agent.async_process(_make_input("Hello"))

    speech = result.response.speech.get("plain", {}).get("speech", "")
    assert speech == FALLBACK_RESPONSE
    # Verify the updated message includes actionable guidance
    assert (
        "rephrasing" in speech.lower()
        or "rephrase" in speech.lower()
        or "settings" in speech.lower()
    )


# ===========================================================================
# Feature 8 — RouterDecision intent_hint field
# ===========================================================================


def test_router_decision_intent_hint_default_is_none() -> None:
    """RouterDecision.intent_hint defaults to None when not supplied."""
    decision = RouterDecision(local_ha=True, complexity=10)
    assert decision.intent_hint is None


def test_router_decision_intent_hint_stored_correctly() -> None:
    """RouterDecision stores a provided intent_hint."""
    decision = RouterDecision(local_ha=True, complexity=5, intent_hint="timer")
    assert decision.intent_hint == "timer"


def test_router_decision_equality_includes_intent_hint() -> None:
    """Two RouterDecisions with the same fields but different intent_hint are unequal."""
    d1 = RouterDecision(local_ha=True, complexity=5, intent_hint="timer")
    d2 = RouterDecision(local_ha=True, complexity=5, intent_hint=None)
    assert d1 != d2


def test_router_decision_equality_same_intent_hint() -> None:
    """Two RouterDecisions with identical fields including intent_hint are equal."""
    d1 = RouterDecision(local_ha=True, complexity=5, intent_hint="todo")
    d2 = RouterDecision(local_ha=True, complexity=5, intent_hint="todo")
    assert d1 == d2


def test_router_decision_hash_includes_intent_hint() -> None:
    """RouterDecision hash changes when intent_hint differs."""
    d1 = RouterDecision(local_ha=True, complexity=5, intent_hint="timer")
    d2 = RouterDecision(local_ha=True, complexity=5, intent_hint=None)
    assert hash(d1) != hash(d2)


def test_router_decision_repr_includes_intent_hint() -> None:
    """RouterDecision repr includes the intent_hint field."""
    d = RouterDecision(local_ha=True, complexity=5, intent_hint="reminder")
    assert "intent_hint='reminder'" in repr(d)


def test_router_decision_is_immutable_intent_hint() -> None:
    """Setting intent_hint on a RouterDecision raises AttributeError."""
    d = RouterDecision(local_ha=False, complexity=10)
    with pytest.raises(AttributeError):
        d.intent_hint = "timer"  # type: ignore[misc]


# ===========================================================================
# Feature 8 — _parse_router_response intent_hint parsing
# ===========================================================================


def test_parse_router_response_valid_intent_hint_timer() -> None:
    """JSON with a valid intent_hint='timer' is parsed and stored."""
    result = _parse_router_response('{"local_ha": true, "complexity": 5, "intent_hint": "timer"}')
    assert result is not None
    assert result.intent_hint == "timer"


def test_parse_router_response_valid_intent_hint_todo() -> None:
    """JSON with intent_hint='todo' is parsed correctly."""
    result = _parse_router_response('{"local_ha": true, "complexity": 5, "intent_hint": "todo"}')
    assert result is not None
    assert result.intent_hint == "todo"


def test_parse_router_response_valid_intent_hint_shopping_list() -> None:
    """JSON with intent_hint='shopping_list' is parsed correctly."""
    result = _parse_router_response(
        '{"local_ha": true, "complexity": 5, "intent_hint": "shopping_list"}'
    )
    assert result is not None
    assert result.intent_hint == "shopping_list"


def test_parse_router_response_valid_intent_hint_reminder() -> None:
    """JSON with intent_hint='reminder' is parsed correctly."""
    result = _parse_router_response(
        '{"local_ha": true, "complexity": 8, "intent_hint": "reminder"}'
    )
    assert result is not None
    assert result.intent_hint == "reminder"


def test_parse_router_response_valid_intent_hint_announce() -> None:
    """JSON with intent_hint='announce' is parsed correctly."""
    result = _parse_router_response(
        '{"local_ha": true, "complexity": 8, "intent_hint": "announce"}'
    )
    assert result is not None
    assert result.intent_hint == "announce"


def test_parse_router_response_intent_hint_null_yields_none() -> None:
    """JSON with intent_hint=null results in intent_hint=None."""
    result = _parse_router_response('{"local_ha": false, "complexity": 30, "intent_hint": null}')
    assert result is not None
    assert result.intent_hint is None


def test_parse_router_response_missing_intent_hint_yields_none() -> None:
    """JSON without intent_hint field results in intent_hint=None (backward-compatible)."""
    result = _parse_router_response('{"local_ha": true, "complexity": 5}')
    assert result is not None
    assert result.intent_hint is None


def test_parse_router_response_unknown_intent_hint_discarded() -> None:
    """An unrecognised intent_hint value is ignored and stored as None."""
    result = _parse_router_response(
        '{"local_ha": true, "complexity": 5, "intent_hint": "unknown_future_type"}'
    )
    assert result is not None
    assert result.intent_hint is None


# ===========================================================================
# Feature 8 — _apply_router_decision intent_hint forces LOCAL_HA
# ===========================================================================


def _local_ha_agent(agent_id: str = "local-1") -> dict:
    return {"id": agent_id, CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA, CONF_AGENT_NAME: "LocalHA"}


def _ollama_agent_cfg(agent_id: str = "ollama-1") -> dict:
    return {"id": agent_id, CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA, CONF_AGENT_NAME: "Ollama"}


def test_apply_router_decision_timer_hint_forces_local_ha() -> None:
    """intent_hint='timer' promotes LOCAL_HA agents to the front regardless of local_ha flag."""
    decision = RouterDecision(local_ha=False, complexity=5, intent_hint="timer")
    agents = [_ollama_agent_cfg(), _local_ha_agent()]
    result = _apply_router_decision(decision, agents)
    assert result[0][CONF_AGENT_TYPE] == AGENT_TYPE_LOCAL_HA


def test_apply_router_decision_reminder_hint_forces_local_ha() -> None:
    """intent_hint='reminder' promotes LOCAL_HA agents to the front."""
    decision = RouterDecision(local_ha=False, complexity=8, intent_hint="reminder")
    agents = [_ollama_agent_cfg(), _local_ha_agent()]
    result = _apply_router_decision(decision, agents)
    assert result[0][CONF_AGENT_TYPE] == AGENT_TYPE_LOCAL_HA


def test_apply_router_decision_todo_hint_forces_local_ha() -> None:
    """intent_hint='todo' promotes LOCAL_HA agents to the front."""
    decision = RouterDecision(local_ha=False, complexity=5, intent_hint="todo")
    agents = [_ollama_agent_cfg(), _local_ha_agent()]
    result = _apply_router_decision(decision, agents)
    assert result[0][CONF_AGENT_TYPE] == AGENT_TYPE_LOCAL_HA


def test_apply_router_decision_shopping_list_hint_forces_local_ha() -> None:
    """intent_hint='shopping_list' promotes LOCAL_HA agents to the front."""
    decision = RouterDecision(local_ha=False, complexity=5, intent_hint="shopping_list")
    agents = [_ollama_agent_cfg(), _local_ha_agent()]
    result = _apply_router_decision(decision, agents)
    assert result[0][CONF_AGENT_TYPE] == AGENT_TYPE_LOCAL_HA


def test_apply_router_decision_announce_hint_does_not_force_local_ha() -> None:
    """intent_hint='announce' is not a LOCAL_HA-forcing hint; follows normal local_ha flag."""
    # local_ha=False → LOCAL_HA agents excluded for general routing
    decision = RouterDecision(local_ha=False, complexity=8, intent_hint="announce")
    agents = [_ollama_agent_cfg(), _local_ha_agent()]
    result = _apply_router_decision(decision, agents)
    # LOCAL_HA agent should be excluded since local_ha=False and hint is 'announce'
    types = [a[CONF_AGENT_TYPE] for a in result]
    assert AGENT_TYPE_LOCAL_HA not in types


def test_apply_router_decision_none_hint_preserves_existing_logic() -> None:
    """When intent_hint is None, existing local_ha/web_search logic is unchanged."""
    decision = RouterDecision(local_ha=True, complexity=5, intent_hint=None)
    agents = [_ollama_agent_cfg(), _local_ha_agent()]
    result = _apply_router_decision(decision, agents)
    assert result[0][CONF_AGENT_TYPE] == AGENT_TYPE_LOCAL_HA


# ===========================================================================
# Feature 8 — _classify_with_router records intent_hint in statistics
# ===========================================================================


async def test_classify_with_router_records_intent_hint(hass: HomeAssistant) -> None:
    """When the router returns an intent_hint, record_intent_hint is called on statistics."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config()

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value='{"local_ha": true, "complexity": 5, "intent_hint": "timer"}',
    ):
        result = await conv_agent._classify_with_router(router, "set a timer for 5 minutes")

    assert result is not None
    assert result.intent_hint == "timer"
    router_id = router["id"]
    stats = conv_agent._statistics.get_agent_stats(router_id)
    assert stats is not None
    assert stats.intent_hints.get("timer", 0) == 1


async def test_classify_with_router_no_intent_hint_no_stats_recorded(
    hass: HomeAssistant,
) -> None:
    """When intent_hint is null, record_intent_hint is NOT called."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config()

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value='{"local_ha": false, "complexity": 30, "intent_hint": null}',
    ):
        await conv_agent._classify_with_router(router, "what is the capital of France?")

    router_id = router["id"]
    stats = conv_agent._statistics.get_agent_stats(router_id)
    assert stats is not None
    assert stats.intent_hints == {}


# ===========================================================================
# Feature 10 — _get_language_name helper
# ===========================================================================


def test_get_language_name_known_codes() -> None:
    """_get_language_name returns the human-readable name for known codes."""
    assert _get_language_name("de") == "German"
    assert _get_language_name("fr") == "French"
    assert _get_language_name("es") == "Spanish"
    assert _get_language_name("en") == "English"


def test_get_language_name_regional_variant_normalised() -> None:
    """BCP-47 regional variants are normalised to the base code."""
    assert _get_language_name("fr-FR") == "French"
    assert _get_language_name("pt-BR") == "Portuguese"
    assert _get_language_name("zh-CN") == "Chinese"


def test_get_language_name_unknown_code_returns_input() -> None:
    """Unknown language codes are returned as-is."""
    assert _get_language_name("xx") == "xx"
    assert _get_language_name("tlh") == "tlh"


# ===========================================================================
# Feature 10 — _classify_with_router language injection
# ===========================================================================


async def test_classify_with_router_appends_language_to_prompt(hass: HomeAssistant) -> None:
    """When language is provided, 'Language: {lang}' is appended to the classification prompt."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config()

    captured_prompts: list[str] = []

    async def _capture_generate(prompt: str) -> str:
        captured_prompts.append(prompt)
        return '{"local_ha": false, "complexity": 20, "intent_hint": null}'

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        side_effect=_capture_generate,
    ):
        await conv_agent._classify_with_router(router, "some query", language="de")

    assert len(captured_prompts) == 1
    assert "Language: de" in captured_prompts[0]


async def test_classify_with_router_no_language_no_language_line(hass: HomeAssistant) -> None:
    """When language is None, no 'Language:' line is added to the prompt."""
    entry = _entry_with_agents(_make_ollama_agent())
    conv_agent = NeuralBridgeAgent(hass, entry)
    router = _make_router_config()

    captured_prompts: list[str] = []

    async def _capture_generate(prompt: str) -> str:
        captured_prompts.append(prompt)
        return '{"local_ha": false, "complexity": 20, "intent_hint": null}'

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient.generate",
        side_effect=_capture_generate,
    ):
        await conv_agent._classify_with_router(router, "some query", language=None)

    assert len(captured_prompts) == 1
    assert "Language:" not in captured_prompts[0]


# ===========================================================================
# Feature 10 — _process_with_ollama language passthrough
# ===========================================================================


async def test_process_with_ollama_injects_language_instruction_when_different(
    hass: HomeAssistant,
) -> None:
    """When user language differs from config, 'Respond in {Lang}.' is appended to system prompt."""
    ollama_agent = _make_ollama_agent(system_prompt="You are a helpful assistant.")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [ollama_agent],
            CONF_FORCE_RESPONSE_LANGUAGE: True,
            "language": "en_gb",
        },
    )
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    user_input = ConversationInput(
        text="Hallo",
        context=Context(),
        conversation_id=None,
        device_id=None,
        language="de",
        satellite_id=None,
        agent_id=None,
    )

    captured_messages: list[list[dict]] = []
    mock_client = MagicMock()

    async def _capture_chat(messages: list[dict]) -> str:
        captured_messages.append(messages)
        return "Hallo!"

    mock_client.chat = _capture_chat

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        result = await conv_agent._process_with_ollama(ollama_agent, user_input)

    assert result is not None
    system_msgs = [m for m in captured_messages[0] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "Respond in German." in system_msgs[0]["content"]


async def test_process_with_ollama_no_language_injection_when_same_language(
    hass: HomeAssistant,
) -> None:
    """No language instruction is injected when user language matches the config language."""
    ollama_agent = _make_ollama_agent(system_prompt="You are helpful.")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [ollama_agent],
            CONF_FORCE_RESPONSE_LANGUAGE: True,
            "language": "en_gb",
        },
    )
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    user_input = ConversationInput(
        text="Hello",
        context=Context(),
        conversation_id=None,
        device_id=None,
        language="en",
        satellite_id=None,
        agent_id=None,
    )

    captured_messages: list[list[dict]] = []
    mock_client = MagicMock()

    async def _capture_chat(messages: list[dict]) -> str:
        captured_messages.append(messages)
        return "Hello!"

    mock_client.chat = _capture_chat

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        await conv_agent._process_with_ollama(ollama_agent, user_input)

    system_msgs = [m for m in captured_messages[0] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "Respond in" not in system_msgs[0]["content"]


async def test_process_with_ollama_no_language_injection_when_force_disabled(
    hass: HomeAssistant,
) -> None:
    """When CONF_FORCE_RESPONSE_LANGUAGE is False, no language instruction is injected."""
    ollama_agent = _make_ollama_agent(system_prompt="You are helpful.")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [ollama_agent],
            CONF_FORCE_RESPONSE_LANGUAGE: False,
            "language": "en_gb",
        },
    )
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    user_input = ConversationInput(
        text="Bonjour",
        context=Context(),
        conversation_id=None,
        device_id=None,
        language="fr",
        satellite_id=None,
        agent_id=None,
    )

    captured_messages: list[list[dict]] = []
    mock_client = MagicMock()

    async def _capture_chat(messages: list[dict]) -> str:
        captured_messages.append(messages)
        return "Bonjour!"

    mock_client.chat = _capture_chat

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        await conv_agent._process_with_ollama(ollama_agent, user_input)

    system_msgs = [m for m in captured_messages[0] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "Respond in" not in system_msgs[0]["content"]


async def test_process_with_ollama_no_language_injection_when_language_none(
    hass: HomeAssistant,
) -> None:
    """When user_input.language is an empty string, no language injection occurs."""
    ollama_agent = _make_ollama_agent(system_prompt="You are helpful.")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [ollama_agent],
            CONF_FORCE_RESPONSE_LANGUAGE: True,
            "language": "en_gb",
        },
    )
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    # language="" behaves the same as no language provided
    user_input = ConversationInput(
        text="Hello",
        context=Context(),
        conversation_id=None,
        device_id=None,
        language="",
        satellite_id=None,
        agent_id=None,
    )

    captured_messages: list[list[dict]] = []
    mock_client = MagicMock()

    async def _capture_chat(messages: list[dict]) -> str:
        captured_messages.append(messages)
        return "Hello!"

    mock_client.chat = _capture_chat

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        await conv_agent._process_with_ollama(ollama_agent, user_input)

    system_msgs = [m for m in captured_messages[0] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "Respond in" not in system_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Tests — _render_ha_context
# ---------------------------------------------------------------------------


def test_render_ha_context_substitutes_all_tokens(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_render_ha_context replaces all supported {ha_*} tokens."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)

    hass.config.location_name = "My Home"
    hass.config.time_zone = "Europe/London"
    # hass.config.units defaults to METRIC_SYSTEM in the test fixture,
    # so temperature_unit.value == "°C" without any extra configuration.

    prompt = "Located at {ha_location_name}. TZ={ha_timezone}. Unit={ha_unit_temperature}."
    result = agent._render_ha_context(prompt)

    assert result == "Located at My Home. TZ=Europe/London. Unit=°C."


def test_render_ha_context_no_tokens_returns_unchanged(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_render_ha_context is a no-op when the prompt contains no {ha_*} tokens."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    prompt = "You are a helpful assistant."

    result = agent._render_ha_context(prompt)

    assert result == prompt


def test_render_ha_context_missing_units_attribute_falls_back_to_empty(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_render_ha_context replaces {ha_unit_temperature} with '' when attribute absent."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    # Replace units with an object that has no 'temperature_unit' attribute
    hass.config.units = object()  # type: ignore[assignment]

    prompt = "Unit: {ha_unit_temperature}"
    result = agent._render_ha_context(prompt)

    assert result == "Unit: "


def test_render_ha_context_empty_location_name(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_render_ha_context replaces {ha_location_name} with '' when location_name is empty."""
    agent = NeuralBridgeAgent(hass, mock_config_entry)
    hass.config.location_name = ""

    prompt = "Located at {ha_location_name}."
    result = agent._render_ha_context(prompt)

    assert result == "Located at ."


async def test_process_with_ollama_renders_ha_context_in_system_prompt(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """_process_with_ollama renders {ha_*} tokens in the resolved system prompt."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            "default_prompt": "TZ={ha_timezone} LOC={ha_location_name}",
        },
    )
    entry.add_to_hass(hass)
    hass.config.location_name = "Test Home"
    hass.config.time_zone = "Australia/Sydney"

    agent = NeuralBridgeAgent(hass, entry)
    # Agent has no per-agent system_prompt, so the global default_prompt is used
    agent_cfg = _make_ollama_agent(agent_id="render-test")
    agent_cfg[CONF_SYSTEM_PROMPT] = ""

    captured_messages: list[list[dict]] = []
    mock_client = MagicMock()

    async def _capture(messages: list[dict]) -> str:
        captured_messages.append(messages)
        return "ok"

    mock_client.chat = _capture

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        await agent._process_with_ollama(agent_cfg, _make_input())

    system_content = next(m["content"] for m in captured_messages[0] if m["role"] == "system")
    assert "TZ=Australia/Sydney" in system_content
    assert "LOC=Test Home" in system_content


# ===========================================================================
# Feature 6 — _truncate_to_first_sentence helper
# ===========================================================================


def test_truncate_to_first_sentence_at_period() -> None:
    """Returns text up to and including the first period."""
    result = _truncate_to_first_sentence("Done. Here is some extra context.")
    assert result == "Done."


def test_truncate_to_first_sentence_at_question_mark() -> None:
    """Returns text up to and including the first question mark."""
    result = _truncate_to_first_sentence("Are you sure? Please confirm.")
    assert result == "Are you sure?"


def test_truncate_to_first_sentence_at_exclamation() -> None:
    """Returns text up to and including the first exclamation mark."""
    result = _truncate_to_first_sentence("OK! And here is more.")
    assert result == "OK!"


def test_truncate_to_first_sentence_no_boundary_returns_unchanged() -> None:
    """Returns the full text when no sentence-ending punctuation is found."""
    result = _truncate_to_first_sentence("No punctuation here")
    assert result == "No punctuation here"


def test_truncate_to_first_sentence_already_single_sentence() -> None:
    """Returns single-sentence text unchanged."""
    result = _truncate_to_first_sentence("The light is on.")
    assert result == "The light is on."


def test_truncate_to_first_sentence_strips_trailing_whitespace() -> None:
    """Trailing whitespace after the boundary is stripped."""
    result = _truncate_to_first_sentence("Done.    More text follows.")
    assert result == "Done."


# ===========================================================================
# Feature 6 — _process_with_ollama verbosity injection
# ===========================================================================


async def test_process_with_ollama_brief_appends_brief_instruction(
    hass: HomeAssistant,
) -> None:
    """Brief verbosity appends the brief instruction to the Ollama system prompt."""
    ollama_agent = _make_ollama_agent(system_prompt="You are a helpful assistant.")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [ollama_agent],
            CONF_RESPONSE_VERBOSITY: VERBOSITY_BRIEF,
        },
    )
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    captured_messages: list[list[dict]] = []
    mock_client = MagicMock()

    async def _capture(messages: list[dict]) -> str:
        captured_messages.append(messages)
        return "OK"

    mock_client.chat = _capture

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        result = await conv_agent._process_with_ollama(
            ollama_agent,
            ConversationInput(
                text="Hello",
                context=Context(),
                conversation_id=None,
                device_id=None,
                language="en",
                satellite_id=None,
                agent_id=None,
            ),
        )

    assert result is not None
    system_msgs = [m for m in captured_messages[0] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "one to five words" in system_msgs[0]["content"]


async def test_process_with_ollama_verbose_appends_verbose_instruction(
    hass: HomeAssistant,
) -> None:
    """Verbose verbosity appends the verbose instruction to the Ollama system prompt."""
    ollama_agent = _make_ollama_agent(system_prompt="You are a helpful assistant.")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [ollama_agent],
            CONF_RESPONSE_VERBOSITY: VERBOSITY_VERBOSE,
        },
    )
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    captured_messages: list[list[dict]] = []
    mock_client = MagicMock()

    async def _capture(messages: list[dict]) -> str:
        captured_messages.append(messages)
        return "Here is a detailed answer."

    mock_client.chat = _capture

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        result = await conv_agent._process_with_ollama(
            ollama_agent,
            ConversationInput(
                text="Hello",
                context=Context(),
                conversation_id=None,
                device_id=None,
                language="en",
                satellite_id=None,
                agent_id=None,
            ),
        )

    assert result is not None
    system_msgs = [m for m in captured_messages[0] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "detailed" in system_msgs[0]["content"]


async def test_process_with_ollama_normal_verbosity_no_instruction(
    hass: HomeAssistant,
) -> None:
    """Normal (default) verbosity does not inject any verbosity instruction."""
    ollama_agent = _make_ollama_agent(system_prompt="You are a helpful assistant.")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [ollama_agent],
            CONF_RESPONSE_VERBOSITY: VERBOSITY_NORMAL,
        },
    )
    entry.add_to_hass(hass)
    conv_agent = NeuralBridgeAgent(hass, entry)

    captured_messages: list[list[dict]] = []
    mock_client = MagicMock()

    async def _capture(messages: list[dict]) -> str:
        captured_messages.append(messages)
        return "A balanced answer."

    mock_client.chat = _capture

    with patch(
        "custom_components.neuralbridge.conversation.OllamaClient",
        return_value=mock_client,
    ):
        await conv_agent._process_with_ollama(
            ollama_agent,
            ConversationInput(
                text="Hello",
                context=Context(),
                conversation_id=None,
                device_id=None,
                language="en",
                satellite_id=None,
                agent_id=None,
            ),
        )

    system_msgs = [m for m in captured_messages[0] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "one to five words" not in system_msgs[0]["content"]
    assert "detailed" not in system_msgs[0]["content"]


# ===========================================================================
# Feature 6 — _process_with_existing verbosity handling
# ===========================================================================

# This test module targets Python 3.12's HA, which does not ship
# homeassistant.components.conversation.chat_log.  Patch in a lightweight mock
# so that the inline import inside _process_with_existing does not raise
# ModuleNotFoundError.  The mock ContextVar honours .set()/.reset() to satisfy
# the chat-log isolation pattern used by NeuralBridge.

_CHAT_LOG_PATCH = "homeassistant.components.conversation.chat_log"


def _make_chat_log_mock() -> MagicMock:
    """Return a mock that satisfies the current_chat_log ContextVar API."""
    mock_module = MagicMock()
    mock_var = MagicMock()
    mock_var.set.return_value = MagicMock()  # opaque token
    mock_module.current_chat_log = mock_var
    return mock_module


async def test_process_with_existing_brief_prepends_prefix(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Brief verbosity prepends '[Brief response]' to text for EXISTING (cloud) agents."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_RESPONSE_VERBOSITY: VERBOSITY_BRIEF,
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.openai",
        CONF_AGENT_NAME: "OpenAI",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
    }

    service_response = {"response": {"speech": {"plain": {"speech": "OK"}}}}
    captured_texts: list[str] = []

    async def _capture_call(domain: str, service: str, data: dict, **kwargs: object) -> dict:
        captured_texts.append(str(data.get("text", "")))
        return service_response

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = _capture_call
    agent.hass = mock_hass

    with patch.dict("sys.modules", {_CHAT_LOG_PATCH: _make_chat_log_mock()}):
        result = await agent._process_with_existing(
            agent_cfg,
            ConversationInput(
                text="Turn off the lights",
                context=Context(),
                conversation_id=None,
                device_id=None,
                language="en",
                satellite_id=None,
                agent_id=None,
            ),
        )

    assert result is not None
    assert captured_texts[0].startswith("[Brief response]")
    assert "Turn off the lights" in captured_texts[0]


async def test_process_with_existing_verbose_prepends_prefix(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Verbose verbosity prepends '[Verbose response]' to text for EXISTING (cloud) agents."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_RESPONSE_VERBOSITY: VERBOSITY_VERBOSE,
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.openai",
        CONF_AGENT_NAME: "OpenAI",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
    }

    service_response = {"response": {"speech": {"plain": {"speech": "Here is a detailed answer."}}}}
    captured_texts: list[str] = []

    async def _capture_call(domain: str, service: str, data: dict, **kwargs: object) -> dict:
        captured_texts.append(str(data.get("text", "")))
        return service_response

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = _capture_call
    agent.hass = mock_hass

    with patch.dict("sys.modules", {_CHAT_LOG_PATCH: _make_chat_log_mock()}):
        result = await agent._process_with_existing(
            agent_cfg,
            ConversationInput(
                text="Tell me about the weather",
                context=Context(),
                conversation_id=None,
                device_id=None,
                language="en",
                satellite_id=None,
                agent_id=None,
            ),
        )

    assert result is not None
    assert captured_texts[0].startswith("[Verbose response]")
    assert "Tell me about the weather" in captured_texts[0]


async def test_process_with_existing_normal_does_not_modify_text(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Normal verbosity does not modify the user text for EXISTING agents."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_RESPONSE_VERBOSITY: VERBOSITY_NORMAL,
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.openai",
        CONF_AGENT_NAME: "OpenAI",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
    }

    service_response = {"response": {"speech": {"plain": {"speech": "Normal reply"}}}}
    captured_texts: list[str] = []

    async def _capture_call(domain: str, service: str, data: dict, **kwargs: object) -> dict:
        captured_texts.append(str(data.get("text", "")))
        return service_response

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = _capture_call
    agent.hass = mock_hass

    with patch.dict("sys.modules", {_CHAT_LOG_PATCH: _make_chat_log_mock()}):
        await agent._process_with_existing(
            agent_cfg,
            ConversationInput(
                text="What time is it?",
                context=Context(),
                conversation_id=None,
                device_id=None,
                language="en",
                satellite_id=None,
                agent_id=None,
            ),
        )

    assert captured_texts[0] == "What time is it?"


async def test_process_with_existing_local_ha_brief_truncates_speech(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """LOCAL_HA + brief mode truncates a multi-sentence speech_text at the first sentence."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_RESPONSE_VERBOSITY: VERBOSITY_BRIEF,
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.homeassistant",
        CONF_AGENT_NAME: "Home Assistant",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
    }

    multi_sentence = "I have turned on the kitchen lights. The brightness is now at 100%."
    service_response = {"response": {"speech": {"plain": {"speech": multi_sentence}}}}

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    with patch.dict("sys.modules", {_CHAT_LOG_PATCH: _make_chat_log_mock()}):
        result = await agent._process_with_existing(
            agent_cfg,
            ConversationInput(
                text="Turn on kitchen lights",
                context=Context(),
                conversation_id=None,
                device_id=None,
                language="en",
                satellite_id=None,
                agent_id=None,
            ),
        )

    assert result is not None
    speech = result.response.speech["plain"]["speech"]
    assert speech == "I have turned on the kitchen lights."


async def test_process_with_existing_local_ha_brief_leaves_short_response(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """LOCAL_HA + brief mode leaves a single-sentence response unchanged."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_RESPONSE_VERBOSITY: VERBOSITY_BRIEF,
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.homeassistant",
        CONF_AGENT_NAME: "Home Assistant",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
    }

    single_sentence = "Done."
    service_response = {"response": {"speech": {"plain": {"speech": single_sentence}}}}

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    with patch.dict("sys.modules", {_CHAT_LOG_PATCH: _make_chat_log_mock()}):
        result = await agent._process_with_existing(
            agent_cfg,
            ConversationInput(
                text="Turn on lights",
                context=Context(),
                conversation_id=None,
                device_id=None,
                language="en",
                satellite_id=None,
                agent_id=None,
            ),
        )

    assert result is not None
    assert result.response.speech["plain"]["speech"] == "Done."


async def test_process_with_existing_local_ha_normal_does_not_truncate(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """LOCAL_HA + normal verbosity leaves multi-sentence speech_text unchanged."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_RESPONSE_VERBOSITY: VERBOSITY_NORMAL,
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)
    agent_cfg = {
        CONF_ENTITY_ID: "conversation.homeassistant",
        CONF_AGENT_NAME: "Home Assistant",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
    }

    multi_sentence = "I have turned on the kitchen lights. The brightness is now at 100%."
    service_response = {"response": {"speech": {"plain": {"speech": multi_sentence}}}}

    mock_hass = MagicMock()
    mock_hass.config.language = hass.config.language
    mock_hass.states.get.return_value = MagicMock()
    mock_hass.services.async_call = AsyncMock(return_value=service_response)
    agent.hass = mock_hass

    with patch.dict("sys.modules", {_CHAT_LOG_PATCH: _make_chat_log_mock()}):
        result = await agent._process_with_existing(
            agent_cfg,
            ConversationInput(
                text="Turn on kitchen lights",
                context=Context(),
                conversation_id=None,
                device_id=None,
                language="en",
                satellite_id=None,
                agent_id=None,
            ),
        )

    assert result is not None
    speech = result.response.speech["plain"]["speech"]
    assert speech == multi_sentence


# ---------------------------------------------------------------------------
# Feature 2 — Compound Command Splitting: _split_compound_input
# ---------------------------------------------------------------------------


def test_split_compound_input_splits_on_and() -> None:
    """_split_compound_input splits 'A and B' into two fragments."""
    result = _split_compound_input("turn off the lights and set the thermostat to 22")
    assert result == ["turn off the lights", "set the thermostat to 22"]


def test_split_compound_input_splits_on_then() -> None:
    """_split_compound_input splits on 'then'."""
    result = _split_compound_input("turn off the lights then lock the door")
    assert result == ["turn off the lights", "lock the door"]


def test_split_compound_input_splits_on_also() -> None:
    """_split_compound_input splits on 'also'."""
    result = _split_compound_input("pause the music also dim the lights")
    assert result == ["pause the music", "dim the lights"]


def test_split_compound_input_splits_on_after_that() -> None:
    """_split_compound_input splits on 'after that'."""
    result = _split_compound_input("turn on the fan after that close the blinds")
    assert result == ["turn on the fan", "close the blinds"]


def test_split_compound_input_splits_on_and_then_as_single_point() -> None:
    """'and then' counts as one split point, producing exactly two fragments."""
    result = _split_compound_input("turn off the lights and then lock the door")
    assert result == ["turn off the lights", "lock the door"]


def test_split_compound_input_no_conjunction_returns_original() -> None:
    """_split_compound_input returns [text] unchanged when no conjunction is present."""
    text = "turn off the lights"
    result = _split_compound_input(text)
    assert result == [text]


def test_split_compound_input_case_insensitive() -> None:
    """_split_compound_input is case-insensitive for conjunctions."""
    result = _split_compound_input("turn off the lights AND set the thermostat to 22")
    assert result == ["turn off the lights", "set the thermostat to 22"]


def test_split_compound_input_caps_at_max_fragments() -> None:
    """_split_compound_input returns at most MAX_COMPOUND_FRAGMENTS fragments."""
    # Build a command with more conjunctions than the limit allows
    parts = [f"command {i}" for i in range(MAX_COMPOUND_FRAGMENTS + 2)]
    text = " and ".join(parts)
    result = _split_compound_input(text)
    assert len(result) == MAX_COMPOUND_FRAGMENTS
    assert result == parts[:MAX_COMPOUND_FRAGMENTS]


def test_split_compound_input_three_fragments() -> None:
    """_split_compound_input handles exactly three fragments (at the limit)."""
    result = _split_compound_input("close the blinds and lock the door and turn off the lights")
    assert len(result) == MAX_COMPOUND_FRAGMENTS
    assert result == ["close the blinds", "lock the door", "turn off the lights"]


def test_split_compound_input_strips_whitespace() -> None:
    """_split_compound_input strips leading/trailing whitespace from fragments."""
    result = _split_compound_input("  turn off the lights  and  set the thermostat  ")
    assert result == ["turn off the lights", "set the thermostat"]


# ---------------------------------------------------------------------------
# Feature 2 — Compound Command Splitting: _process_compound_fragments
# ---------------------------------------------------------------------------


async def test_process_compound_fragments_combines_successful_results(
    hass: HomeAssistant,
) -> None:
    """_process_compound_fragments joins fragment responses with the separator."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [], CONF_SPLIT_COMPOUND_COMMANDS: True},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    call_count = 0
    responses = ["Lights turned off.", "Thermostat set to 22."]

    async def fake_try_processing(_agents: list, _input: ConversationInput) -> "ConversationResult":
        nonlocal call_count
        resp = agent._create_result(responses[call_count])
        call_count += 1
        return resp

    with patch.object(agent, "_try_processing_agents", side_effect=fake_try_processing):
        user_input = _make_input("turn off the lights and set the thermostat to 22")
        result = await agent._process_compound_fragments(
            ["turn off the lights", "set the thermostat to 22"],
            [],
            user_input,
        )

    combined = result.response.speech["plain"]["speech"]
    assert combined == f"Lights turned off.{COMPOUND_COMMAND_SEPARATOR}Thermostat set to 22."


async def test_process_compound_fragments_includes_failure_text(
    hass: HomeAssistant,
) -> None:
    """_process_compound_fragments includes the fallback text for failed fragments."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [], CONF_SPLIT_COMPOUND_COMMANDS: True},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    # Second fragment fails — _try_processing_agents returns a result with error speech
    call_count = 0
    fallback = agent._localized("responses", "fallback")

    async def fake_try_processing(_agents: list, _input: ConversationInput) -> "ConversationResult":
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return agent._create_result("Lights turned off.")
        call_count += 1
        return agent._create_error_result(fallback)

    with patch.object(agent, "_try_processing_agents", side_effect=fake_try_processing):
        user_input = _make_input("turn off the lights and set the thermostat to 22")
        result = await agent._process_compound_fragments(
            ["turn off the lights", "set the thermostat to 22"],
            [],
            user_input,
        )

    combined = result.response.speech["plain"]["speech"]
    assert combined == f"Lights turned off.{COMPOUND_COMMAND_SEPARATOR}{fallback}"


async def test_process_compound_fragments_uses_conversation_id(
    hass: HomeAssistant,
) -> None:
    """_process_compound_fragments preserves the original conversation_id in the combined result."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [], CONF_SPLIT_COMPOUND_COMMANDS: True},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    async def fake_try_processing(_agents: list, _input: ConversationInput) -> "ConversationResult":
        return agent._create_result("OK")

    with patch.object(agent, "_try_processing_agents", side_effect=fake_try_processing):
        user_input = _make_input("A and B", conversation_id="conv-123")
        result = await agent._process_compound_fragments(["A", "B"], [], user_input)

    assert result.conversation_id == "conv-123"


# ---------------------------------------------------------------------------
# Feature 2 — Compound Command Splitting: _compute_result integration
# ---------------------------------------------------------------------------


async def test_compute_result_compound_disabled_by_default(hass: HomeAssistant) -> None:
    """Compound splitting is disabled by default; full text is passed as-is."""
    local_ha_cfg = {
        "id": "ha-1",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "HA",
        CONF_PRIORITY: 10,
        CONF_TIMEOUT: 30,
        CONF_AGENT_CACHE_ENABLED: False,
    }
    # CONF_SPLIT_COMPOUND_COMMANDS not set → defaults to False
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_AGENTS: [local_ha_cfg]})
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    captured_texts: list[str] = []

    async def spy_try_processing(
        agents: list, user_input: ConversationInput
    ) -> "ConversationResult":
        captured_texts.append(user_input.text)
        return agent._create_result("Done.")

    compound_input = _make_input("turn off the lights and set the thermostat to 22")
    with patch.object(agent, "_try_processing_agents", side_effect=spy_try_processing):
        await agent._compute_result(compound_input)

    # Should be called exactly once with the full text (no splitting)
    assert len(captured_texts) == 1
    assert captured_texts[0] == "turn off the lights and set the thermostat to 22"


async def test_compute_result_compound_splits_when_enabled_with_local_ha(
    hass: HomeAssistant,
) -> None:
    """When split_compound_commands=True with a LOCAL_HA agent, compound input is split."""
    local_ha_cfg = {
        "id": "ha-1",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "HA",
        CONF_PRIORITY: 10,
        CONF_TIMEOUT: 30,
        CONF_AGENT_CACHE_ENABLED: False,
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [local_ha_cfg], CONF_SPLIT_COMPOUND_COMMANDS: True},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    captured_texts: list[str] = []
    call_count = 0
    responses = ["Lights turned off.", "Thermostat set."]

    async def spy_try_processing(
        agents: list, user_input: ConversationInput
    ) -> "ConversationResult":
        nonlocal call_count
        captured_texts.append(user_input.text)
        resp = agent._create_result(responses[call_count % len(responses)])
        call_count += 1
        return resp

    compound_input = _make_input("turn off the lights and set the thermostat to 22")
    with patch.object(agent, "_try_processing_agents", side_effect=spy_try_processing):
        result = await agent._compute_result(compound_input)

    # _try_processing_agents should have been called once per fragment
    assert len(captured_texts) == 2
    assert captured_texts[0] == "turn off the lights"
    assert captured_texts[1] == "set the thermostat to 22"

    combined_speech = result.response.speech["plain"]["speech"]
    expected = f"Lights turned off.{COMPOUND_COMMAND_SEPARATOR}Thermostat set."
    assert combined_speech == expected


async def test_compute_result_compound_no_split_without_local_ha(
    hass: HomeAssistant,
) -> None:
    """Even with split_compound_commands=True, no split occurs if no LOCAL_HA agent."""
    ollama_cfg = {
        "id": "ollama-1",
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "Ollama",
        CONF_PRIORITY: 10,
        CONF_TIMEOUT: 30,
        CONF_AGENT_CACHE_ENABLED: False,
        CONF_OLLAMA_URL: "http://localhost:11434",
        CONF_OLLAMA_MODEL: "llama3",
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [ollama_cfg], CONF_SPLIT_COMPOUND_COMMANDS: True},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    captured_texts: list[str] = []

    async def spy_try_processing(
        agents: list, user_input: ConversationInput
    ) -> "ConversationResult":
        captured_texts.append(user_input.text)
        return agent._create_result("Done.")

    compound_input = _make_input("turn off the lights and set the thermostat to 22")
    with patch.object(agent, "_try_processing_agents", side_effect=spy_try_processing):
        await agent._compute_result(compound_input)

    # Full text should pass through unchanged
    assert len(captured_texts) == 1
    assert captured_texts[0] == "turn off the lights and set the thermostat to 22"


async def test_compute_result_compound_no_split_single_fragment(
    hass: HomeAssistant,
) -> None:
    """When split_compound_commands=True but input has no conjunction, no split occurs."""
    local_ha_cfg = {
        "id": "ha-1",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "HA",
        CONF_PRIORITY: 10,
        CONF_TIMEOUT: 30,
        CONF_AGENT_CACHE_ENABLED: False,
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [local_ha_cfg], CONF_SPLIT_COMPOUND_COMMANDS: True},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    captured_texts: list[str] = []

    async def spy_try_processing(
        agents: list, user_input: ConversationInput
    ) -> "ConversationResult":
        captured_texts.append(user_input.text)
        return agent._create_result("Done.")

    simple_input = _make_input("turn off the lights")
    with patch.object(agent, "_try_processing_agents", side_effect=spy_try_processing):
        await agent._compute_result(simple_input)

    # No split — single call with the original text
    assert len(captured_texts) == 1
    assert captured_texts[0] == "turn off the lights"


async def test_compute_result_compound_caps_at_max_fragments(
    hass: HomeAssistant,
) -> None:
    """Compound splitting caps at MAX_COMPOUND_FRAGMENTS even if more conjunctions exist."""
    local_ha_cfg = {
        "id": "ha-1",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "HA",
        CONF_PRIORITY: 10,
        CONF_TIMEOUT: 30,
        CONF_AGENT_CACHE_ENABLED: False,
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [local_ha_cfg], CONF_SPLIT_COMPOUND_COMMANDS: True},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    captured_texts: list[str] = []
    call_count = 0

    async def spy_try_processing(
        agents: list, user_input: ConversationInput
    ) -> "ConversationResult":
        nonlocal call_count
        captured_texts.append(user_input.text)
        call_count += 1
        return agent._create_result(f"done {call_count}")

    # Four fragments joined by "and" — only the first MAX_COMPOUND_FRAGMENTS are processed
    long_input = _make_input("cmd1 and cmd2 and cmd3 and cmd4")
    with patch.object(agent, "_try_processing_agents", side_effect=spy_try_processing):
        await agent._compute_result(long_input)

    assert len(captured_texts) == MAX_COMPOUND_FRAGMENTS


# ---------------------------------------------------------------------------
# Feature 11 — _extract_announce_text module helper
# ---------------------------------------------------------------------------


def test_extract_announce_text_empty_returns_none() -> None:
    """Empty text returns None."""
    assert _extract_announce_text("") is None
    assert _extract_announce_text("   ") is None


def test_extract_announce_text_regex_match_with_content() -> None:
    """Trigger phrase followed by content returns the content."""
    result = _extract_announce_text("announce: dinner is ready")
    assert result == "dinner is ready"


def test_extract_announce_text_regex_match_empty_content_returns_none() -> None:
    """Trigger phrase with no following content returns None."""
    result = _extract_announce_text("announce: ")
    assert result is None


def test_extract_announce_text_no_match_no_hint_returns_none() -> None:
    """Non-trigger phrase without intent_hint returns None."""
    result = _extract_announce_text("turn on the lights")
    assert result is None


def test_extract_announce_text_intent_hint_announce_returns_full_text() -> None:
    """When intent_hint='announce' but no trigger phrase, returns the full text."""
    result = _extract_announce_text("dinner is ready", intent_hint="announce")
    assert result == "dinner is ready"


def test_extract_announce_text_broadcast_trigger() -> None:
    """'broadcast' trigger phrase is recognised."""
    result = _extract_announce_text("broadcast the meeting is starting")
    assert result == "the meeting is starting"


def test_extract_announce_text_tell_everyone_trigger() -> None:
    """'tell everyone' trigger phrase is recognised."""
    result = _extract_announce_text("tell everyone that pizza has arrived")
    assert result == "pizza has arrived"


# ---------------------------------------------------------------------------
# Feature 14b — _apply_router_decision with ROUTER_CONFIDENCE_LOW
# ---------------------------------------------------------------------------


def test_apply_router_decision_confidence_low_returns_all_agents_unchanged() -> None:
    """With confidence=low, Phase 3 flag-based promotion is skipped."""
    local = {CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA, "id": "local"}
    ollama = {CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA, "id": "ollama"}
    # local_ha=False would normally exclude LOCAL_HA, but confidence=low skips
    decision = RouterDecision(local_ha=False, complexity=50, confidence=ROUTER_CONFIDENCE_LOW)
    result = _apply_router_decision(decision, [local, ollama])
    assert result == [local, ollama]


# ---------------------------------------------------------------------------
# Feature 11 — broadcast announcement paths inside _compute_result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_result_regex_fast_path_triggers_broadcast(
    hass: HomeAssistant,
) -> None:
    """When announce_media_players configured, regex fast path sends broadcast."""
    local_ha_cfg = {
        "id": "ha-1",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "HA",
        CONF_PRIORITY: 10,
        CONF_TIMEOUT: 30,
        CONF_AGENT_CACHE_ENABLED: False,
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [local_ha_cfg],
            CONF_ANNOUNCE_MEDIA_PLAYERS: ["media_player.living_room"],
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)
    expected = agent._create_result("Message sent to 1 speaker.")

    with patch.object(
        agent, "_send_broadcast_announcement", new_callable=AsyncMock, return_value=expected
    ) as mock_bcast:
        result = await agent._compute_result(_make_input("announce: lights off"))

    mock_bcast.assert_awaited_once()
    assert result is expected


@pytest.mark.asyncio
async def test_compute_result_router_intent_hint_announce_triggers_broadcast(
    hass: HomeAssistant,
) -> None:
    """When router returns intent_hint='announce', broadcast is triggered."""
    router_cfg = {
        "id": "router-1",
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "Router",
        CONF_PRIORITY: 99,
        CONF_TIMEOUT: 5,
        CONF_AGENT_CACHE_ENABLED: False,
        CONF_IS_ROUTER: True,
        CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
        CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        CONF_ROUTER_CUSTOM_PROMPT: "",
    }
    local_ha_cfg = {
        "id": "ha-1",
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "HA",
        CONF_PRIORITY: 10,
        CONF_TIMEOUT: 30,
        CONF_AGENT_CACHE_ENABLED: False,
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [router_cfg, local_ha_cfg],
            CONF_ANNOUNCE_MEDIA_PLAYERS: ["media_player.kitchen"],
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)
    expected = agent._create_result("Message sent to 1 speaker.")

    announce_decision = RouterDecision(local_ha=False, complexity=10, intent_hint="announce")

    with (
        patch.object(
            agent, "_check_with_routers", new_callable=AsyncMock, return_value=announce_decision
        ),
        patch.object(
            agent, "_send_broadcast_announcement", new_callable=AsyncMock, return_value=expected
        ) as mock_bcast,
    ):
        # Use an input without a trigger phrase so the regex fast-path is skipped;
        # the router intent_hint="announce" path (lines 761-764) must fire instead.
        result = await agent._compute_result(_make_input("please say hello to the family"))

    mock_bcast.assert_awaited_once()
    assert result is expected


@pytest.mark.asyncio
async def test_find_tts_entity_returns_first_tts(hass: HomeAssistant) -> None:
    """_find_tts_entity returns the first entity_id starting with tts."""
    entry = _entry_with_agents(_make_ollama_agent())
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    with patch(
        "homeassistant.core.StateMachine.async_entity_ids",
        return_value=["light.living", "tts.cloud_say", "media_player.kitchen"],
    ):
        assert agent._find_tts_entity() == "tts.cloud_say"


@pytest.mark.asyncio
async def test_find_tts_entity_returns_none_when_missing(hass: HomeAssistant) -> None:
    """_find_tts_entity returns None when no tts entity exists."""
    entry = _entry_with_agents(_make_ollama_agent())
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    with patch("homeassistant.core.StateMachine.async_entity_ids", return_value=["light.living"]):
        assert agent._find_tts_entity() is None


@pytest.mark.asyncio
async def test_send_broadcast_announcement_no_tts_entity(hass: HomeAssistant) -> None:
    """_send_broadcast_announcement returns error when no TTS entity found."""
    entry = _entry_with_agents(_make_ollama_agent())
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    with patch.object(agent, "_find_tts_entity", return_value=None):
        result = await agent._send_broadcast_announcement(
            "hello", ["media_player.kitchen"], _make_input("hello")
        )
    assert "No text-to-speech" in result.response.speech["plain"]["speech"]


@pytest.mark.asyncio
async def test_send_broadcast_announcement_fires_event_and_calls_tts(
    hass: HomeAssistant,
) -> None:
    """_send_broadcast_announcement calls tts.speak for each player and fires event."""
    entry = _entry_with_agents(_make_ollama_agent())
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)
    players = ["media_player.living_room", "media_player.kitchen"]

    fired_events: list[tuple] = []

    with (
        patch.object(agent, "_find_tts_entity", return_value="tts.cloud_say"),
        patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock) as mock_call,
        patch(
            "homeassistant.core.EventBus.async_fire",
            side_effect=lambda et, data=None: fired_events.append((et, data)),
        ),
    ):
        result = await agent._send_broadcast_announcement(
            "pizza is here", players, _make_input("announce: pizza is here")
        )

    assert mock_call.call_count == 2
    assert any(et == EVENT_ANNOUNCE_SENT for et, _ in fired_events)
    assert "2 speaker" in result.response.speech["plain"]["speech"]


@pytest.mark.asyncio
async def test_send_broadcast_single_player_grammar(hass: HomeAssistant) -> None:
    """Single player uses singular 'speaker.' text."""
    entry = _entry_with_agents(_make_ollama_agent())
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    with (
        patch.object(agent, "_find_tts_entity", return_value="tts.cloud_say"),
        patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock),
        patch("homeassistant.core.EventBus.async_fire"),
    ):
        result = await agent._send_broadcast_announcement(
            "dinner time", ["media_player.kitchen"], _make_input("announce: dinner time")
        )
    speech = result.response.speech["plain"]["speech"]
    assert "1 speaker." in speech


@pytest.mark.asyncio
async def test_send_broadcast_long_text_truncated_in_event(hass: HomeAssistant) -> None:
    """Text longer than 50 chars is truncated with ellipsis in the event."""
    entry = _entry_with_agents(_make_ollama_agent())
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    event_data: list[dict] = []
    long_text = "a" * 60

    with (
        patch.object(agent, "_find_tts_entity", return_value="tts.cloud_say"),
        patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock),
        patch(
            "homeassistant.core.EventBus.async_fire",
            side_effect=lambda et, data=None: event_data.append(data or {}),
        ),
    ):
        await agent._send_broadcast_announcement(
            long_text, ["media_player.kitchen"], _make_input("announce something")
        )
    assert event_data[0]["text_preview"].endswith("\u2026")


# ---------------------------------------------------------------------------
# Feature 4 — _check_pending_response: high-stakes pending path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_confirmation_check_hs_pending_resolved(
    hass: HomeAssistant,
) -> None:
    """When high-stakes pending exists and user says 'yes', returns original result."""
    entry = _entry_with_agents(_make_ollama_agent())
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    original = agent._create_result("lights locked")
    await agent._high_stakes_cache.store_pending("conv-hs", original, [])

    result = await agent._handle_confirmation_check(_make_input("yes", conversation_id="conv-hs"))
    assert result is not None
    assert result.response.speech["plain"]["speech"] == "lights locked"


@pytest.mark.asyncio
async def test_handle_confirmation_check_hs_pending_fallthrough_on_unrecognised(
    hass: HomeAssistant,
) -> None:
    """Unrecognised response with hs pending returns None (fall-through)."""
    entry = _entry_with_agents(_make_ollama_agent())
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    original = agent._create_result("door unlocked")
    await agent._high_stakes_cache.store_pending("conv-hs2", original, [])

    result = await agent._handle_confirmation_check(
        _make_input("maybe", conversation_id="conv-hs2")
    )
    # "maybe" is not yes/no, so high-stakes returns None to fall through
    assert result is None


# ---------------------------------------------------------------------------
# Feature 4 — _resolve_high_stakes_confirmation: all branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_high_stakes_passphrase_correct(hass: HomeAssistant) -> None:
    """Correct passphrase returns the original result."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_HIGH_STAKES_SECRET_ENABLED: True,
            CONF_HIGH_STAKES_SECRET: "opensesame",
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    original = agent._create_result("door unlocked")
    hs_pending = (original, [])

    result = await agent._resolve_high_stakes_confirmation(
        _make_input("opensesame", conversation_id="c1"), hs_pending
    )
    assert result is not None
    assert result.response.speech["plain"]["speech"] == "door unlocked"


@pytest.mark.asyncio
async def test_resolve_high_stakes_passphrase_wrong(hass: HomeAssistant) -> None:
    """Wrong passphrase returns cancelled error result."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [],
            CONF_HIGH_STAKES_SECRET_ENABLED: True,
            CONF_HIGH_STAKES_SECRET: "opensesame",
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    original = agent._create_result("door unlocked")
    hs_pending = (original, [])

    result = await agent._resolve_high_stakes_confirmation(
        _make_input("wrongphrase", conversation_id="c2"), hs_pending
    )
    assert result is not None
    speech = result.response.speech["plain"]["speech"]
    assert "cancel" in speech.lower() or "action" in speech.lower()


@pytest.mark.asyncio
async def test_resolve_high_stakes_yes_confirms(hass: HomeAssistant) -> None:
    """'yes' in simple mode returns the original result."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [], CONF_HIGH_STAKES_SECRET_ENABLED: False},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    original = agent._create_result("door locked")
    hs_pending = (original, [])

    result = await agent._resolve_high_stakes_confirmation(
        _make_input("yes", conversation_id="c3"), hs_pending
    )
    assert result is not None
    assert result.response.speech["plain"]["speech"] == "door locked"


@pytest.mark.asyncio
async def test_resolve_high_stakes_no_cancels(hass: HomeAssistant) -> None:
    """'no' in simple mode returns a cancellation error."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [], CONF_HIGH_STAKES_SECRET_ENABLED: False},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    original = agent._create_result("door locked")
    hs_pending = (original, [])

    result = await agent._resolve_high_stakes_confirmation(
        _make_input("no", conversation_id="c4"), hs_pending
    )
    assert result is not None
    speech = result.response.speech["plain"]["speech"]
    assert "cancel" in speech.lower() or "action" in speech.lower()


@pytest.mark.asyncio
async def test_resolve_high_stakes_unrecognised_falls_through(
    hass: HomeAssistant,
) -> None:
    """Unrecognised input in simple mode returns None to fall through."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [], CONF_HIGH_STAKES_SECRET_ENABLED: False},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    original = agent._create_result("door locked")
    hs_pending = (original, [])

    result = await agent._resolve_high_stakes_confirmation(
        _make_input("maybe", conversation_id="c5"), hs_pending
    )
    assert result is None


# ---------------------------------------------------------------------------
# Feature 4 — _check_high_stakes: enabled paths
# ---------------------------------------------------------------------------


def _make_local_ha_cfg(agent_id: str = "ha-1") -> dict:
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "HA",
        CONF_PRIORITY: 10,
        CONF_TIMEOUT: 30,
        CONF_AGENT_CACHE_ENABLED: False,
    }


@pytest.mark.asyncio
async def test_check_high_stakes_disabled_returns_none(hass: HomeAssistant) -> None:
    """When high_stakes_enabled=False (default), returns None."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [_make_local_ha_cfg()], CONF_HIGH_STAKES_ENABLED: False},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    result_obj = agent._create_result("I've unlocked the front door")
    ret = await agent._check_high_stakes(
        _make_local_ha_cfg(), result_obj, _make_input("unlock the front door")
    )
    assert ret is None


@pytest.mark.asyncio
async def test_check_high_stakes_non_local_ha_returns_none(hass: HomeAssistant) -> None:
    """When agent is not LOCAL_HA, returns None even if enabled."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_AGENTS: [], CONF_HIGH_STAKES_ENABLED: True},
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    ollama_cfg = _make_ollama_agent()
    result_obj = agent._create_result("door unlocked")
    ret = await agent._check_high_stakes(ollama_cfg, result_obj, _make_input("unlock the door"))
    assert ret is None


@pytest.mark.asyncio
async def test_check_high_stakes_domain_match_returns_confirm_prompt(
    hass: HomeAssistant,
) -> None:
    """Matching a high-stakes domain returns the confirmation prompt and fires event."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [_make_local_ha_cfg()],
            CONF_HIGH_STAKES_ENABLED: True,
            CONF_HIGH_STAKES_DOMAINS: ["lock"],
        },
    )
    entry.add_to_hass(hass)

    mock_hass = MagicMock()
    fired: list[str] = []
    mock_hass.bus.async_fire.side_effect = lambda et, data=None: fired.append(et)
    agent = NeuralBridgeAgent(mock_hass, entry)

    result_obj = agent._create_result("I've unlocked the front lock")
    ret = await agent._check_high_stakes(
        _make_local_ha_cfg(),
        result_obj,
        _make_input("unlock the front lock", conversation_id="hs-test"),
    )
    assert ret is not None
    speech = ret.response.speech["plain"]["speech"]
    assert "confirm" in speech.lower() or "sure" in speech.lower() or "yes" in speech.lower()
    assert EVENT_HIGH_STAKES_TRIGGERED in fired


@pytest.mark.asyncio
async def test_check_high_stakes_secret_enabled_returns_passphrase_prompt(
    hass: HomeAssistant,
) -> None:
    """When secret is enabled, _check_high_stakes returns the passphrase prompt instead."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [_make_local_ha_cfg()],
            CONF_HIGH_STAKES_ENABLED: True,
            CONF_HIGH_STAKES_DOMAINS: ["lock"],
            CONF_HIGH_STAKES_SECRET_ENABLED: True,
            CONF_HIGH_STAKES_SECRET: "sesame",
        },
    )
    entry.add_to_hass(hass)

    mock_hass = MagicMock()
    fired: list[str] = []
    mock_hass.bus.async_fire.side_effect = lambda et, data=None: fired.append(et)
    agent = NeuralBridgeAgent(mock_hass, entry)

    result_obj = agent._create_result("I've unlocked the front lock")
    ret = await agent._check_high_stakes(
        _make_local_ha_cfg(),
        result_obj,
        _make_input("unlock the front lock", conversation_id="hs-secret"),
    )
    assert ret is not None
    speech = ret.response.speech["plain"]["speech"]
    # The passphrase prompt should differ from a plain yes/no confirmation
    assert speech != ""
    assert EVENT_HIGH_STAKES_TRIGGERED in fired


@pytest.mark.asyncio
async def test_check_high_stakes_no_domain_match_returns_none(
    hass: HomeAssistant,
) -> None:
    """When speech does not match any high-stakes domain, returns None."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [_make_local_ha_cfg()],
            CONF_HIGH_STAKES_ENABLED: True,
            CONF_HIGH_STAKES_DOMAINS: ["lock", "alarm"],
        },
    )
    entry.add_to_hass(hass)
    agent = NeuralBridgeAgent(hass, entry)

    result_obj = agent._create_result("I've turned on the light")
    ret = await agent._check_high_stakes(
        _make_local_ha_cfg(), result_obj, _make_input("turn on the light")
    )
    assert ret is None


@pytest.mark.asyncio
async def test_handle_successful_result_with_high_stakes_match(
    hass: HomeAssistant,
) -> None:
    """_handle_successful_result returns confirmation prompt when high-stakes matches."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_AGENTS: [_make_local_ha_cfg()],
            CONF_HIGH_STAKES_ENABLED: True,
            CONF_HIGH_STAKES_DOMAINS: ["lock"],
        },
    )
    entry.add_to_hass(hass)

    mock_hass = MagicMock()
    agent = NeuralBridgeAgent(mock_hass, entry)

    local_cfg = _make_local_ha_cfg()
    result_obj = agent._create_result("I've locked the front lock")

    with patch.object(agent, "_check_guardrails", new_callable=AsyncMock, return_value=None):
        ret = await agent._handle_successful_result(
            local_cfg,
            result_obj,
            _make_input("lock the door", conversation_id="hs-flow"),
            agent_id="local_ha",
            elapsed_ms=42.0,
        )

    assert ret is not None
    speech = ret.response.speech["plain"]["speech"]
    assert "confirm" in speech.lower() or "sure" in speech.lower() or "yes" in speech.lower()
