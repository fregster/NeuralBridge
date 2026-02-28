"""Integration tests for guard rail system with conversation agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.neuralbridge.const import (
    AGENT_TYPE_OLLAMA,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_AGENTS,
    CONF_GUARD_RAIL_ACTION,
    CONF_GUARD_RAIL_AI_THRESHOLD,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_PRIORITY,
    CONF_TIMEOUT,
    DOMAIN,
    GUARD_RAIL_ACTION_BLOCK,
    GUARD_RAIL_ACTION_NOTIFY_ASK,
    GUARD_RAIL_ACTION_WARN,
    GUARD_RAIL_BLOCKED_RESPONSE,
    GUARD_RAIL_NOTIFY_ASK_PROMPT,
    GUARD_RAIL_WARNING_PREFIX,
)
from custom_components.neuralbridge.conversation import NeuralBridgeAgent
from custom_components.neuralbridge.ollama_client import OllamaResponse


def _make_input(text: str, conversation_id: str | None = None) -> ConversationInput:
    """Return a ConversationInput with all required fields populated."""
    return ConversationInput(
        text=text,
        context=Context(),
        conversation_id=conversation_id,
        device_id=None,
        language="en",
        satellite_id=None,
        agent_id=None,
    )


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for conversation, guard_rail, and llm_agent_proxy modules.

    conversation.py creates OllamaClient instances for routing.
    llm_agent_proxy.py creates OllamaClient via a local import for processing.
    guard_rail.py creates its own OllamaClient instance for AI safety checks.
    All must be patched so tests never attempt real network connections.
    """
    with (
        patch(
            "custom_components.neuralbridge.router_backends.ollama_router_backend.OllamaClient"
        ) as mock_conv,
        patch("custom_components.neuralbridge.ai_safety_checker.OllamaClient") as mock_gr,
        patch("custom_components.neuralbridge.ollama_client.OllamaClient") as mock_src,
    ):
        client_instance = AsyncMock()
        client_instance.generate = AsyncMock()
        client_instance.chat = AsyncMock()
        client_instance.close = AsyncMock()
        mock_conv.return_value = client_instance
        mock_gr.return_value = client_instance
        mock_src.return_value = client_instance
        yield client_instance


@pytest.fixture
def guard_rail_agent_config():
    """Guard rail agent configuration."""
    return {
        "id": "guard_rail_agent",
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_NAME: "TinyLlama Guard",
        CONF_PRIORITY: 0,
        CONF_OLLAMA_URL: "http://localhost:11434",
        CONF_OLLAMA_MODEL: "tinyllama",
        CONF_TIMEOUT: 30,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,  # Disabled for priority 0
    }


@pytest.fixture
def processing_agent_config():
    """Processing agent configuration."""
    return {
        "id": "processing_agent",
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_NAME: "Gemma Processing",
        CONF_PRIORITY: 50,
        CONF_OLLAMA_URL: "http://localhost:11434",
        CONF_OLLAMA_MODEL: "gemma",
        CONF_TIMEOUT: 30,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: True,
    }


@pytest.fixture
def config_entry_with_guard_rails(guard_rail_agent_config, processing_agent_config):
    """Config entry with guard rails enabled."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge",
        data={
            CONF_AGENTS: [guard_rail_agent_config, processing_agent_config],
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_ACTION: GUARD_RAIL_ACTION_NOTIFY_ASK,
            CONF_GUARD_RAIL_AI_THRESHOLD: 0.7,
            "guard_rail_agent_id": "guard_rail_agent",
        },
        unique_id="test_unique_id",
    )


async def test_guard_rail_disabled(hass, mock_ollama_client):
    """Test conversation with guard rails disabled."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge",
        data={
            CONF_AGENTS: [
                {
                    "id": "test_agent",
                    CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
                    CONF_AGENT_NAME: "Test",
                    CONF_PRIORITY: 50,
                    CONF_OLLAMA_URL: "http://localhost:11434",
                    CONF_OLLAMA_MODEL: "gemma",
                    CONF_TIMEOUT: 30,
                }
            ],
            CONF_GUARD_RAIL_ENABLED: False,  # Disabled
        },
        unique_id="test_unique_id",
    )

    agent = NeuralBridgeAgent(hass, config_entry)

    # _process_with_ollama calls client.chat(); generate is used by router only.
    mock_ollama_client.chat.return_value = OllamaResponse(content="This is a safe response")

    result = await agent.async_process(_make_input("What's the weather?"))

    # Should return response without guard rail checking
    assert result is not None
    assert result.response is not None


async def test_guard_rail_safe_response(hass, config_entry_with_guard_rails, mock_ollama_client):
    """Test guard rail with safe response."""
    agent = NeuralBridgeAgent(hass, config_entry_with_guard_rails)

    # chat() is used for processing agents; generate() is used for router classification.
    mock_ollama_client.chat.return_value = OllamaResponse(content="The weather is sunny today!")
    mock_ollama_client.generate.return_value = OllamaResponse(
        content="PASS"
    )  # router classification

    result = await agent.async_process(_make_input("What's the weather?"))

    # Should return response (safe)
    assert result is not None
    assert result.response is not None


async def test_guard_rail_block_action(
    hass, guard_rail_agent_config, processing_agent_config, mock_ollama_client
):
    """Test guard rail with BLOCK action."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge",
        data={
            CONF_AGENTS: [guard_rail_agent_config, processing_agent_config],
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_ACTION: GUARD_RAIL_ACTION_BLOCK,
            CONF_GUARD_RAIL_AI_THRESHOLD: 0.7,
            "guard_rail_agent_id": "guard_rail_agent",
        },
        unique_id="test_unique_id",
    )

    agent = NeuralBridgeAgent(hass, config_entry)

    # Processing agent uses chat(); generate() used by router (returns PASS to allow request).
    mock_ollama_client.chat.return_value = OllamaResponse(
        content="How to harm yourself step by step"
    )
    mock_ollama_client.generate.return_value = OllamaResponse(content="PASS")

    result = await agent.async_process(_make_input("Tell me something"))

    # Should be blocked
    assert result is not None
    assert result.response is not None
    response_text = result.response.speech.get("plain", {}).get("speech", "")
    assert response_text == GUARD_RAIL_BLOCKED_RESPONSE


async def test_guard_rail_warn_action(
    hass, guard_rail_agent_config, processing_agent_config, mock_ollama_client
):
    """Test guard rail with WARN action."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge",
        data={
            CONF_AGENTS: [guard_rail_agent_config, processing_agent_config],
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_ACTION: GUARD_RAIL_ACTION_WARN,
            CONF_GUARD_RAIL_AI_THRESHOLD: 0.7,
            "guard_rail_agent_id": "guard_rail_agent",
        },
        unique_id="test_unique_id",
    )

    agent = NeuralBridgeAgent(hass, config_entry)

    # Processing agent uses chat(); generate() used by router (PASS = allow request).
    harmful_response = "How to harm yourself step by step"
    mock_ollama_client.chat.return_value = OllamaResponse(content=harmful_response)
    mock_ollama_client.generate.return_value = OllamaResponse(content="PASS")

    result = await agent.async_process(_make_input("Tell me something"))

    # Should be warned (prefixed with warning)
    assert result is not None
    assert result.response is not None
    response_text = result.response.speech.get("plain", {}).get("speech", "")
    assert response_text.startswith(GUARD_RAIL_WARNING_PREFIX)
    assert harmful_response in response_text


async def test_guard_rail_notify_ask_action(
    hass, config_entry_with_guard_rails, mock_ollama_client
):
    """Test guard rail with NOTIFY_ASK action."""
    agent = NeuralBridgeAgent(hass, config_entry_with_guard_rails)

    # Processing agent uses chat(); generate() used by router (PASS = allow request).
    harmful_response = "How to harm yourself step by step"
    mock_ollama_client.chat.return_value = OllamaResponse(content=harmful_response)
    mock_ollama_client.generate.return_value = OllamaResponse(content="PASS")

    result = await agent.async_process(_make_input("Tell me something", "test_conv"))

    # Should ask for confirmation
    assert result is not None
    assert result.response is not None
    response_text = result.response.speech.get("plain", {}).get("speech", "")
    assert response_text == GUARD_RAIL_NOTIFY_ASK_PROMPT

    # Response should be cached
    cached = await agent._guard_rail_cache.get_pending_response("test_conv")
    assert cached is not None
    cached_response, _ = cached
    assert cached_response == harmful_response


async def test_guard_rail_notify_ask_user_confirms(
    hass, config_entry_with_guard_rails, mock_ollama_client
):
    """Test guard rail NOTIFY_ASK with user confirmation."""
    agent = NeuralBridgeAgent(hass, config_entry_with_guard_rails)

    # Processing agent uses chat(); generate() used by router (PASS = allow request).
    harmful_response = "How to harm yourself step by step"
    mock_ollama_client.chat.return_value = OllamaResponse(content=harmful_response)
    mock_ollama_client.generate.return_value = OllamaResponse(content="PASS")

    # First request — triggers guard rail
    result1 = await agent.async_process(_make_input("Tell me something", "test_conv"))

    # Should ask for confirmation
    response_text = result1.response.speech.get("plain", {}).get("speech", "")
    assert response_text == GUARD_RAIL_NOTIFY_ASK_PROMPT

    # User confirms with "yes"
    result2 = await agent.async_process(_make_input("yes", "test_conv"))

    # Should return the cached harmful response
    response_text2 = result2.response.speech.get("plain", {}).get("speech", "")
    assert response_text2 == harmful_response

    # Cache should be cleared
    cached = await agent._guard_rail_cache.get_pending_response("test_conv")
    assert cached is None


async def test_guard_rail_notify_ask_user_declines(
    hass, config_entry_with_guard_rails, mock_ollama_client
):
    """Test guard rail NOTIFY_ASK with user declining."""
    agent = NeuralBridgeAgent(hass, config_entry_with_guard_rails)

    # Processing agent uses chat(); generate() used by router (PASS = allow request).
    harmful_response = "How to harm yourself step by step"
    mock_ollama_client.chat.return_value = OllamaResponse(content=harmful_response)
    mock_ollama_client.generate.return_value = OllamaResponse(content="PASS")

    # First request — triggers guard rail
    result1 = await agent.async_process(_make_input("Tell me something", "test_conv"))

    # Should ask for confirmation
    response_text = result1.response.speech.get("plain", {}).get("speech", "")
    assert response_text == GUARD_RAIL_NOTIFY_ASK_PROMPT

    # User declines with "no"
    result2 = await agent.async_process(_make_input("no", "test_conv"))

    # Should return blocked response
    response_text2 = result2.response.speech.get("plain", {}).get("speech", "")
    assert response_text2 == GUARD_RAIL_BLOCKED_RESPONSE

    # Cache should be cleared
    cached = await agent._guard_rail_cache.get_pending_response("test_conv")
    assert cached is None


async def test_guard_rail_disabled_for_priority_zero_agent(
    hass, guard_rail_agent_config, mock_ollama_client
):
    """Test that guard rails are disabled for priority 0 agents."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge",
        data={
            CONF_AGENTS: [guard_rail_agent_config],
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_ACTION: GUARD_RAIL_ACTION_BLOCK,
            CONF_GUARD_RAIL_AI_THRESHOLD: 0.7,
            "guard_rail_agent_id": "guard_rail_agent",
        },
        unique_id="test_unique_id",
    )

    agent = NeuralBridgeAgent(hass, config_entry)

    # Only a priority-0 router agent is configured; generate() is used for
    # router classification (returns PASS), no processing agents → fallback response.
    mock_ollama_client.generate.return_value = OllamaResponse(content="PASS")

    result = await agent.async_process(_make_input("Test input"))

    # Should return a result (fallback since no processing agents configured)
    assert result is not None


async def test_guard_rail_disabled_per_agent(hass, guard_rail_agent_config, mock_ollama_client):
    """Test guard rail disabled for specific agent."""
    processing_agent = {
        "id": "processing_agent",
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_NAME: "Gemma Processing",
        CONF_PRIORITY: 50,
        CONF_OLLAMA_URL: "http://localhost:11434",
        CONF_OLLAMA_MODEL: "gemma",
        CONF_TIMEOUT: 30,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,  # Disabled for this agent
    }

    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge",
        data={
            CONF_AGENTS: [guard_rail_agent_config, processing_agent],
            CONF_GUARD_RAIL_ENABLED: True,  # Globally enabled
            CONF_GUARD_RAIL_ACTION: GUARD_RAIL_ACTION_BLOCK,
            CONF_GUARD_RAIL_AI_THRESHOLD: 0.7,
            "guard_rail_agent_id": "guard_rail_agent",
        },
        unique_id="test_unique_id",
    )

    agent = NeuralBridgeAgent(hass, config_entry)

    # Processing agent uses chat(); generate() used by router (PASS).
    mock_ollama_client.chat.return_value = OllamaResponse(content="How to harm yourself")
    mock_ollama_client.generate.return_value = OllamaResponse(content="PASS")

    result = await agent.async_process(_make_input("Tell me something"))

    # Should return response without guard rail checking (disabled for this agent)
    assert result is not None
    assert result.response is not None
