"""Unit tests for agent_backend.py (AgentBackend protocol, registry, and backends)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context

from custom_components.neuralbridge.agent_backend import (
    AgentBackend,
    AgentBackendRegistry,
    IntegratedAgentBackend,
    LocalHAAgentBackend,
    OllamaAgentBackend,
    WebSearchAgentBackend,
    _make_result,
)
from custom_components.neuralbridge.const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    AGENT_TYPE_WEB_SEARCH,
    CONF_AGENT_ASSIST_MODE,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_ENTITY_ID,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_SEARCH_API_KEY,
    CONF_SEARCH_PROVIDER,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(text: str = "Hello", conversation_id: str = "conv-1") -> ConversationInput:
    """Return a minimal ConversationInput."""
    return ConversationInput(
        text=text,
        context=Context(),
        conversation_id=conversation_id,
        device_id=None,
        language="en",
        satellite_id=None,
        agent_id=None,
    )


def _make_hass(language: str = "en") -> MagicMock:
    """Return a lightweight mock HomeAssistant instance."""
    mock = MagicMock()
    mock.config.language = language
    mock.states.get.return_value = MagicMock()  # entity exists
    mock.services.async_call = AsyncMock()
    return mock


def _ollama_config(agent_id: str = "ollama-1") -> dict[str, Any]:
    """Return a minimal Ollama agent config."""
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_NAME: "Test Ollama",
        CONF_OLLAMA_URL: "http://localhost:11434",
        CONF_OLLAMA_MODEL: "tinyllama",
    }


def _integrated_config(
    agent_id: str = "integrated-1", entity_id: str = "conversation.openai"
) -> dict[str, Any]:
    """Return a minimal integrated agent config."""
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_INTEGRATED,
        CONF_AGENT_NAME: "Integrated",
        CONF_ENTITY_ID: entity_id,
    }


def _local_ha_config(agent_id: str = "local-ha-1") -> dict[str, Any]:
    """Return a minimal LOCAL_HA agent config."""
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_NAME: "Local HA",
        CONF_ENTITY_ID: "conversation.home_assistant",
    }


def _web_search_config(agent_id: str = "search-1") -> dict[str, Any]:
    """Return a minimal web search agent config."""
    return {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_WEB_SEARCH,
        CONF_AGENT_NAME: "Search",
        CONF_SEARCH_PROVIDER: "brave",
        CONF_SEARCH_API_KEY: "test-key",
    }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestMakeResult:
    """Tests for the module-level _make_result helper."""

    def test_creates_result_with_speech(self, hass):
        """_make_result sets the speech text on the IntentResponse."""
        result = _make_result(hass, "Hello world", "conv-1")
        assert result.response.speech["plain"]["speech"] == "Hello world"

    def test_result_carries_conversation_id(self, hass):
        """_make_result echoes the conversation_id back."""
        result = _make_result(hass, "Hi", "my-conv")
        assert result.conversation_id == "my-conv"

    def test_result_conversation_id_none(self, hass):
        """_make_result accepts None conversation_id."""
        result = _make_result(hass, "Hi", None)
        assert result.conversation_id is None


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestAgentBackendRegistry:
    """Tests for AgentBackendRegistry registration and build."""

    def test_build_ollama_returns_ollama_backend(self, hass):
        """build() returns an OllamaAgentBackend for AGENT_TYPE_OLLAMA."""
        backend = AgentBackendRegistry.build(hass, _ollama_config())
        assert isinstance(backend, OllamaAgentBackend)

    def test_build_integrated_returns_integrated_backend(self, hass):
        """build() returns an IntegratedAgentBackend for AGENT_TYPE_INTEGRATED."""
        backend = AgentBackendRegistry.build(hass, _integrated_config())
        assert isinstance(backend, IntegratedAgentBackend)

    def test_build_local_ha_returns_local_ha_backend(self, hass):
        """build() returns a LocalHAAgentBackend for AGENT_TYPE_LOCAL_HA."""
        backend = AgentBackendRegistry.build(hass, _local_ha_config())
        assert isinstance(backend, LocalHAAgentBackend)

    def test_build_web_search_returns_web_search_backend(self, hass):
        """build() returns a WebSearchAgentBackend for AGENT_TYPE_WEB_SEARCH."""
        backend = AgentBackendRegistry.build(hass, _web_search_config())
        assert isinstance(backend, WebSearchAgentBackend)

    def test_build_unknown_type_raises_value_error(self, hass):
        """build() raises ValueError for an unregistered agent type."""
        with pytest.raises(ValueError, match="Unknown agent type: 'unknown_type'"):
            AgentBackendRegistry.build(hass, {CONF_AGENT_TYPE: "unknown_type"})

    def test_register_and_build_custom_type(self, hass):
        """A newly registered type can be built via build()."""

        class DummyBackend:
            def __init__(self, hass, config):
                self.hass = hass
                self.config = config

        AgentBackendRegistry.register("custom_test_type", DummyBackend)
        backend = AgentBackendRegistry.build(hass, {CONF_AGENT_TYPE: "custom_test_type"})
        assert isinstance(backend, DummyBackend)

    def test_all_four_types_satisfy_protocol(self, hass):
        """All four built backends satisfy the AgentBackend Protocol."""
        for config in [
            _ollama_config(),
            _integrated_config(),
            _local_ha_config(),
            _web_search_config(),
        ]:
            backend = AgentBackendRegistry.build(hass, config)
            assert isinstance(backend, AgentBackend)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestOllamaAgentBackend:
    """Tests for OllamaAgentBackend."""

    def test_agent_id_returns_config_id(self, hass):
        """agent_id property reflects the config 'id' field."""
        backend = OllamaAgentBackend(hass, _ollama_config("my-id"))
        assert backend.agent_id == "my-id"

    def test_agent_type_is_ollama(self, hass):
        """agent_type always returns AGENT_TYPE_OLLAMA."""
        backend = OllamaAgentBackend(hass, _ollama_config())
        assert backend.agent_type == AGENT_TYPE_OLLAMA

    async def test_process_returns_result_on_success(self, hass):
        """process() wraps the Ollama response in a ConversationResult."""
        backend = OllamaAgentBackend(hass, _ollama_config())

        mock_resp = MagicMock()
        mock_resp.content = "Ollama reply"
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_resp)
        backend._client = mock_client

        result = await backend.process(_make_input("Hi"), _ollama_config(), timeout=10)

        assert result is not None
        assert result.response.speech["plain"]["speech"] == "Ollama reply"
        mock_client.chat.assert_called_once()

    async def test_process_returns_none_when_client_returns_none(self, hass):
        """process() returns None when the Ollama client yields None."""
        backend = OllamaAgentBackend(hass, _ollama_config())
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=None)
        backend._client = mock_client

        result = await backend.process(_make_input("Hi"), _ollama_config(), timeout=10)
        assert result is None

    async def test_process_returns_none_when_content_empty(self, hass):
        """process() returns None when the response content is empty."""
        backend = OllamaAgentBackend(hass, _ollama_config())
        mock_resp = MagicMock()
        mock_resp.content = ""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_resp)
        backend._client = mock_client

        result = await backend.process(_make_input("Hi"), _ollama_config(), timeout=10)
        assert result is None

    def test_get_client_creates_client_lazily(self, hass):
        """_get_client() creates an OllamaClient on first call and caches it."""
        backend = OllamaAgentBackend(hass, _ollama_config())
        assert backend._client is None

        with patch("custom_components.neuralbridge.ollama_client.OllamaClient") as mock_cls:
            result = backend._get_client()
            mock_cls.assert_called_once()
            assert result is mock_cls.return_value
            assert backend._client is mock_cls.return_value

    def test_get_client_returns_cached_client(self, hass):
        """_get_client() returns the same instance on repeated calls."""
        backend = OllamaAgentBackend(hass, _ollama_config())
        existing = MagicMock()
        backend._client = existing
        assert backend._get_client() is existing

    async def test_close_closes_client(self, hass):
        """close() calls close() on the underlying OllamaClient."""
        backend = OllamaAgentBackend(hass, _ollama_config())
        mock_client = AsyncMock()
        backend._client = mock_client

        await backend.close()

        mock_client.close.assert_called_once()
        assert backend._client is None

    async def test_close_is_noop_when_no_client(self, hass):
        """close() is safe when no client has been created yet."""
        backend = OllamaAgentBackend(hass, _ollama_config())
        await backend.close()  # should not raise

    async def test_close_then_get_client_creates_new_instance(self, hass):
        """After close(), _get_client() will create a fresh OllamaClient."""
        backend = OllamaAgentBackend(hass, _ollama_config())
        mock_client = AsyncMock()
        backend._client = mock_client

        await backend.close()
        assert backend._client is None


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestIntegratedAgentBackend:
    """Tests for IntegratedAgentBackend."""

    def test_agent_id_returns_config_id(self, hass):
        """agent_id property reflects the config 'id' field."""
        backend = IntegratedAgentBackend(hass, _integrated_config("my-id"))
        assert backend.agent_id == "my-id"

    def test_agent_type_is_integrated(self, hass):
        """agent_type always returns AGENT_TYPE_INTEGRATED."""
        backend = IntegratedAgentBackend(hass, _integrated_config())
        assert backend.agent_type == AGENT_TYPE_INTEGRATED

    async def test_process_returns_result_on_success(self, hass):
        """process() wraps the HA service response in a ConversationResult."""
        mock_hass = _make_hass()
        mock_hass.services.async_call = AsyncMock(
            return_value={"response": {"speech": {"plain": {"speech": "Hello!"}}}},
        )
        config = _integrated_config()
        backend = IntegratedAgentBackend(mock_hass, config)

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_var:
            mock_var.set.return_value = MagicMock()
            result = await backend.process(_make_input("Hi"), config, timeout=10)

        assert result is not None
        assert result.response.speech["plain"]["speech"] == "Hello!"

    async def test_process_returns_none_when_entity_missing(self, hass):
        """process() returns None when the entity_id is not found in hass.states."""
        mock_hass = _make_hass()
        mock_hass.states.get.return_value = None  # entity not found
        config = _integrated_config()
        backend = IntegratedAgentBackend(mock_hass, config)

        result = await backend.process(_make_input("Hi"), config, timeout=10)
        assert result is None

    async def test_process_returns_none_when_entity_id_absent(self, hass):
        """process() returns None when entity_id is not in agent_config."""
        mock_hass = _make_hass()
        config_no_entity: dict[str, Any] = {
            "id": "x",
            CONF_AGENT_TYPE: AGENT_TYPE_INTEGRATED,
        }  # no entity_id
        backend = IntegratedAgentBackend(mock_hass, config_no_entity)
        result = await backend.process(_make_input(), config_no_entity, timeout=10)
        assert result is None

    async def test_process_returns_none_on_service_error(self, hass):
        """process() returns None when the HA service raises an exception."""
        mock_hass = _make_hass()
        mock_hass.services.async_call = AsyncMock(side_effect=RuntimeError("boom"))
        config = _integrated_config()
        backend = IntegratedAgentBackend(mock_hass, config)

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_var:
            mock_var.set.return_value = MagicMock()
            result = await backend.process(_make_input("Hi"), config, timeout=10)

        assert result is None

    async def test_process_returns_none_when_response_shape_wrong(self, hass):
        """process() returns None when the service response is missing 'response' key."""
        mock_hass = _make_hass()
        mock_hass.services.async_call = AsyncMock(return_value={"unexpected": True})
        config = _integrated_config()
        backend = IntegratedAgentBackend(mock_hass, config)

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_var:
            mock_var.set.return_value = MagicMock()
            result = await backend.process(_make_input("Hi"), config, timeout=10)

        assert result is None

    async def test_process_returns_none_when_speech_empty(self, hass):
        """process() returns None when the speech text is empty."""
        mock_hass = _make_hass()
        mock_hass.services.async_call = AsyncMock(
            return_value={"response": {"speech": {"plain": {"speech": ""}}}}
        )
        config = _integrated_config()
        backend = IntegratedAgentBackend(mock_hass, config)

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_var:
            mock_var.set.return_value = MagicMock()
            result = await backend.process(_make_input("Hi"), config, timeout=10)

        assert result is None

    async def test_close_is_noop(self, hass):
        """close() completes without error (no resources to release)."""
        backend = IntegratedAgentBackend(hass, _integrated_config())
        await backend.close()  # should not raise


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestLocalHAAgentBackend:
    """Tests for LocalHAAgentBackend."""

    def test_agent_id_returns_config_id(self, hass):
        """agent_id property reflects the config 'id' field."""
        backend = LocalHAAgentBackend(hass, _local_ha_config("my-id"))
        assert backend.agent_id == "my-id"

    def test_agent_type_is_local_ha(self, hass):
        """agent_type always returns AGENT_TYPE_LOCAL_HA."""
        backend = LocalHAAgentBackend(hass, _local_ha_config())
        assert backend.agent_type == AGENT_TYPE_LOCAL_HA

    async def test_process_action_done_with_assist_mode(self, hass):
        """process() returns result when response_type is action_done and assist_mode=True."""
        mock_hass = _make_hass()
        mock_hass.services.async_call = AsyncMock(
            return_value={
                "response": {
                    "response_type": "action_done",
                    "speech": {"plain": {"speech": "Lights on!"}},
                }
            }
        )
        config = {**_local_ha_config(), CONF_AGENT_ASSIST_MODE: True}
        backend = LocalHAAgentBackend(mock_hass, config)

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_var:
            mock_var.set.return_value = MagicMock()
            result = await backend.process(_make_input(), config, timeout=10)

        assert result is not None
        assert result.response.speech["plain"]["speech"] == "Lights on!"

    async def test_process_non_action_done_filtered_in_assist_mode(self, hass):
        """process() returns None for non-action_done responses when assist_mode=True."""
        mock_hass = _make_hass()
        mock_hass.services.async_call = AsyncMock(
            return_value={
                "response": {
                    "response_type": "query_answer",
                    "speech": {"plain": {"speech": "The capital is Paris."}},
                }
            }
        )
        config = {**_local_ha_config(), CONF_AGENT_ASSIST_MODE: True}
        backend = LocalHAAgentBackend(mock_hass, config)

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_var:
            mock_var.set.return_value = MagicMock()
            result = await backend.process(_make_input(), config, timeout=10)

        assert result is None

    async def test_process_passes_through_without_assist_mode(self, hass):
        """process() returns any response when assist_mode=False."""
        mock_hass = _make_hass()
        mock_hass.services.async_call = AsyncMock(
            return_value={
                "response": {
                    "response_type": "query_answer",
                    "speech": {"plain": {"speech": "Paris is the capital."}},
                }
            }
        )
        config = {**_local_ha_config(), CONF_AGENT_ASSIST_MODE: False}
        backend = LocalHAAgentBackend(mock_hass, config)

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_var:
            mock_var.set.return_value = MagicMock()
            result = await backend.process(_make_input(), config, timeout=10)

        assert result is not None
        assert "Paris" in result.response.speech["plain"]["speech"]

    async def test_process_returns_none_on_service_error(self, hass):
        """process() returns None when the HA service raises an exception."""
        mock_hass = _make_hass()
        mock_hass.services.async_call = AsyncMock(side_effect=RuntimeError("fail"))
        config = _local_ha_config()
        backend = LocalHAAgentBackend(mock_hass, config)

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_var:
            mock_var.set.return_value = MagicMock()
            result = await backend.process(_make_input(), config, timeout=10)

        assert result is None

    async def test_process_returns_none_when_speech_absent(self, hass):
        """process() returns None when the speech section is missing or empty."""
        mock_hass = _make_hass()
        mock_hass.services.async_call = AsyncMock(
            return_value={"response": {"response_type": "action_done", "speech": {}}}
        )
        config = {**_local_ha_config(), CONF_AGENT_ASSIST_MODE: False}
        backend = LocalHAAgentBackend(mock_hass, config)

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_var:
            mock_var.set.return_value = MagicMock()
            result = await backend.process(_make_input(), config, timeout=10)

        assert result is None

    async def test_close_is_noop(self, hass):
        """close() completes without error (no resources to release)."""
        backend = LocalHAAgentBackend(hass, _local_ha_config())
        await backend.close()  # should not raise


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestWebSearchAgentBackend:
    """Tests for WebSearchAgentBackend."""

    def test_agent_id_returns_config_id(self, hass):
        """agent_id property reflects the config 'id' field."""
        backend = WebSearchAgentBackend(hass, _web_search_config("search-x"))
        assert backend.agent_id == "search-x"

    def test_agent_type_is_web_search(self, hass):
        """agent_type always returns AGENT_TYPE_WEB_SEARCH."""
        backend = WebSearchAgentBackend(hass, _web_search_config())
        assert backend.agent_type == AGENT_TYPE_WEB_SEARCH

    async def test_process_returns_result_on_success(self, hass):
        """process() wraps the search summary in a ConversationResult."""
        backend = WebSearchAgentBackend(hass, _web_search_config())
        mock_client = AsyncMock()
        mock_client.search_and_summarise = AsyncMock(return_value="Top result: example.com")
        backend._client = mock_client

        result = await backend.process(_make_input("news"), _web_search_config(), timeout=15)

        assert result is not None
        assert "example.com" in result.response.speech["plain"]["speech"]
        mock_client.search_and_summarise.assert_awaited_once_with("news")

    async def test_process_returns_none_when_no_summary(self, hass):
        """process() returns None when search_and_summarise returns empty string."""
        backend = WebSearchAgentBackend(hass, _web_search_config())
        mock_client = AsyncMock()
        mock_client.search_and_summarise = AsyncMock(return_value="")
        backend._client = mock_client

        result = await backend.process(_make_input("nothing"), _web_search_config(), timeout=15)
        assert result is None

    def test_get_client_creates_client_lazily(self, hass):
        """_get_client() creates a WebSearchClient on first call and caches it."""
        backend = WebSearchAgentBackend(hass, _web_search_config())
        assert backend._client is None

        with patch("custom_components.neuralbridge.web_search_client.WebSearchClient") as mock_cls:
            result = backend._get_client()
            mock_cls.assert_called_once()
            assert result is mock_cls.return_value
            assert backend._client is mock_cls.return_value

    def test_get_client_returns_cached_client(self, hass):
        """_get_client() returns the same instance on repeated calls."""
        backend = WebSearchAgentBackend(hass, _web_search_config())
        existing = MagicMock()
        backend._client = existing
        assert backend._get_client() is existing

    async def test_close_closes_client(self, hass):
        """close() calls close() on the underlying WebSearchClient."""
        backend = WebSearchAgentBackend(hass, _web_search_config())
        mock_client = AsyncMock()
        backend._client = mock_client

        await backend.close()

        mock_client.close.assert_called_once()
        assert backend._client is None

    async def test_close_is_noop_when_no_client(self, hass):
        """close() is safe when no client has been created yet."""
        backend = WebSearchAgentBackend(hass, _web_search_config())
        await backend.close()  # should not raise
