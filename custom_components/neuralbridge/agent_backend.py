"""AgentBackend protocol, registry, and concrete transport implementations.

This module defines the agent-dispatch abstraction for NeuralBridge:

* :class:`AgentBackend` — runtime-checkable ``Protocol`` that every transport
  must satisfy.
* :class:`AgentBackendRegistry` — maps ``agent_type`` strings to backend
  factory callables.
* Four concrete implementations: :class:`OllamaAgentBackend`,
  :class:`IntegratedAgentBackend`, :class:`LocalHAAgentBackend`, and
  :class:`WebSearchAgentBackend`.

**Phase notes**

Phase 3a (this step) — creates the file.  ``conversation.py`` continues to
use its existing ``if agent_type == …`` dispatch chains unchanged.

Phase 3b (part of R5) — ``NeuralBridgeAgent._try_agent`` will be refactored
to delegate to ``AgentBackendRegistry.build(hass, agent_config)``.

See plan-02-code-restructure.md § R3 for details.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

from homeassistant.components.conversation import ConversationInput, ConversationResult
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.helpers import intent

from .const import (
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
    CONF_SEARCH_MAX_SNIPPET_LEN,
    CONF_SEARCH_PROVIDER,
    CONF_SEARCH_RESULT_COUNT,
    CONF_TIMEOUT,
    DEFAULT_AGENT_ASSIST_MODE,
    DEFAULT_SEARCH_MAX_SNIPPET_LEN,
    DEFAULT_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_TIMEOUT,
    DEFAULT_TIMEOUT,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentBackend(Protocol):
    """Transport interface every NeuralBridge agent backend must satisfy.

    Concrete implementations handle a single ``agent_type``.  The pipeline
    in ``conversation.py`` (after Phase 3b) calls only this protocol — it
    never inspects ``agent_type`` directly.
    """

    @property
    def agent_id(self) -> str:
        """Unique identifier of this agent instance."""
        ...  # pragma: no cover

    @property
    def agent_type(self) -> str:
        """The agent type constant (e.g. ``AGENT_TYPE_OLLAMA``)."""
        ...  # pragma: no cover

    async def process(
        self,
        user_input: ConversationInput,
        agent_config: dict[str, Any],
        *,
        timeout: int,
    ) -> ConversationResult | None:
        """Process a user utterance and return a result, or None on failure."""
        ...  # pragma: no cover

    async def close(self) -> None:
        """Release any held resources (HTTP sessions, model handles, etc.)."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AgentBackendRegistry:
    """Maps ``agent_type`` strings to backend factory callables.

    Usage::

        AgentBackendRegistry.register(AGENT_TYPE_OLLAMA, OllamaAgentBackend)
        backend = AgentBackendRegistry.build(hass, agent_config)
    """

    _registry: ClassVar[dict[str, type[Any]]] = {}

    @classmethod
    def register(cls, agent_type: str, backend_cls: type[Any]) -> None:
        """Register *backend_cls* as the factory for *agent_type*.

        Args:
            agent_type:  The string constant (e.g. ``AGENT_TYPE_OLLAMA``).
            backend_cls: The concrete class whose constructor accepts
                         ``(hass, agent_config)``.
        """
        cls._registry[agent_type] = backend_cls

    @classmethod
    def build(cls, hass: "HomeAssistant", agent_config: dict[str, Any]) -> "AgentBackend":
        """Construct and return a backend for the agent type in *agent_config*.

        Args:
            hass:         Home Assistant instance.
            agent_config: Agent configuration dict (must contain ``CONF_AGENT_TYPE``).

        Returns:
            A new :class:`AgentBackend` instance.

        Raises:
            ValueError: If the agent type is not registered.
        """
        agent_type: str = agent_config.get(CONF_AGENT_TYPE, "")
        backend_cls = cls._registry.get(agent_type)
        if backend_cls is None:
            raise ValueError(f"Unknown agent type: {agent_type!r}")
        return cast("AgentBackend", backend_cls(hass, agent_config))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    hass: "HomeAssistant",
    text: str,
    conversation_id: str | None,
) -> ConversationResult:
    """Build a :class:`ConversationResult` with *text* as the speech response."""
    response = intent.IntentResponse(language=hass.config.language)
    response.async_set_speech(text)
    return ConversationResult(response=response, conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------


class OllamaAgentBackend:
    """Direct HTTP backend for an Ollama-served model.

    Creates an :class:`~.ollama_client.OllamaClient` on first use.  Prompt
    enrichment (session memory, preference injection, etc.) is added in
    Phase 3b when this backend is wired into the NeuralBridge pipeline.
    """

    def __init__(self, hass: "HomeAssistant", agent_config: dict[str, Any]) -> None:
        """Initialise with HA instance and agent configuration.

        Args:
            hass:         Home Assistant instance.
            agent_config: Agent configuration dict.
        """
        self._hass = hass
        self._agent_config = agent_config
        self._client: Any = None

    @property
    def agent_id(self) -> str:
        """Return the agent identifier from its configuration."""
        return str(self._agent_config.get("id", ""))

    @property
    def agent_type(self) -> str:
        """Return the agent type constant."""
        return AGENT_TYPE_OLLAMA

    def _get_client(self) -> Any:
        """Lazily create and return the :class:`OllamaClient`."""
        from .ollama_client import OllamaClient  # noqa: PLC0415

        if self._client is None:
            url: str = self._agent_config.get(CONF_OLLAMA_URL, "")
            model: str = self._agent_config.get(CONF_OLLAMA_MODEL, "")
            timeout: int = self._agent_config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
            self._client = OllamaClient(url, model, timeout)
        return self._client

    async def process(
        self,
        user_input: ConversationInput,
        agent_config: dict[str, Any],  # noqa: ARG002
        *,
        timeout: int,  # noqa: ARG002
    ) -> ConversationResult | None:
        """Call the Ollama HTTP API with the user's text.

        Args:
            user_input:   The user's conversation input.
            agent_config: Not used — this backend uses the config supplied at
                          construction time.
            timeout:      Not used — the client uses its own configured timeout.

        Returns:
            :class:`ConversationResult` on success, ``None`` on any failure.
        """
        client = self._get_client()
        messages = [{"role": "user", "content": user_input.text}]
        ollama_resp = await client.chat(messages)
        if ollama_resp is None or not ollama_resp.content:
            return None
        return _make_result(self._hass, ollama_resp.content, user_input.conversation_id)

    async def close(self) -> None:
        """Close the underlying :class:`OllamaClient` HTTP session."""
        if self._client is not None:
            await self._client.close()
            self._client = None


class IntegratedAgentBackend:
    """Backend for any HA-registered conversation entity (Ollama, Gemini, etc.)."""

    def __init__(self, hass: "HomeAssistant", agent_config: dict[str, Any]) -> None:
        """Initialise with HA instance and agent configuration.

        Args:
            hass:         Home Assistant instance.
            agent_config: Agent configuration dict (must contain ``CONF_ENTITY_ID``).
        """
        self._hass = hass
        self._agent_config = agent_config

    @property
    def agent_id(self) -> str:
        """Return the agent identifier from its configuration."""
        return str(self._agent_config.get("id", ""))

    @property
    def agent_type(self) -> str:
        """Return the agent type constant."""
        return AGENT_TYPE_INTEGRATED

    async def process(
        self,
        user_input: ConversationInput,
        agent_config: dict[str, Any],
        *,
        timeout: int,  # noqa: ARG002
    ) -> ConversationResult | None:
        """Forward the user text to the configured HA conversation entity.

        Args:
            user_input:   The user's conversation input.
            agent_config: Agent configuration.
            timeout:      Not used — applied by the caller's asyncio.timeout context.

        Returns:
            :class:`ConversationResult` on success, ``None`` on failure.
        """
        from homeassistant.components.conversation.chat_log import (  # noqa: PLC0415
            current_chat_log,
        )

        entity_id: str | None = agent_config.get(CONF_ENTITY_ID)
        if not entity_id or not self._hass.states.get(entity_id):
            _LOGGER.error(
                "Integrated agent '%s' has no entity_id or entity not found",
                agent_config.get(CONF_AGENT_NAME, "?"),
            )
            return None

        token = current_chat_log.set(None)
        response: Any = None
        try:
            response = await self._hass.services.async_call(
                CONVERSATION_DOMAIN,
                "process",
                {
                    "text": user_input.text,
                    "agent_id": entity_id,
                    "conversation_id": user_input.conversation_id,
                    "language": user_input.language,
                },
                blocking=True,
                context=user_input.context,
                return_response=True,
            )
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error calling conversation agent '%s': %s", entity_id, err)
        finally:
            current_chat_log.reset(token)

        if not response or "response" not in response:
            return None

        speech = response["response"].get("speech", {}).get("plain", {}).get("speech", "")
        return _make_result(self._hass, speech, user_input.conversation_id) if speech else None

    async def close(self) -> None:
        """No persistent resources to release."""


class LocalHAAgentBackend:
    """Backend for ``conversation.home_assistant`` (the HA built-in intent engine).

    Applies the ``assist_mode`` filter: when ``True``, only ``action_done``
    responses (device-control successes) are returned; all other response types
    cause fall-through to the next agent.
    """

    def __init__(self, hass: "HomeAssistant", agent_config: dict[str, Any]) -> None:
        """Initialise with HA instance and agent configuration.

        Args:
            hass:         Home Assistant instance.
            agent_config: Agent configuration dict.
        """
        self._hass = hass
        self._agent_config = agent_config

    @property
    def agent_id(self) -> str:
        """Return the agent identifier from its configuration."""
        return str(self._agent_config.get("id", ""))

    @property
    def agent_type(self) -> str:
        """Return the agent type constant."""
        return AGENT_TYPE_LOCAL_HA

    async def process(
        self,
        user_input: ConversationInput,
        agent_config: dict[str, Any],
        *,
        timeout: int,  # noqa: ARG002
    ) -> ConversationResult | None:
        """Dispatch to the built-in HA intent processor with assist_mode filtering.

        Args:
            user_input:   The user's conversation input.
            agent_config: Agent configuration (used for assist_mode and entity_id).
            timeout:      Not used — applied by the caller's asyncio.timeout context.

        Returns:
            :class:`ConversationResult` on success, ``None`` if the response was
            filtered out by assist_mode or on failure.
        """
        from homeassistant.components.conversation.chat_log import (  # noqa: PLC0415
            current_chat_log,
        )

        entity_id: str | None = agent_config.get(CONF_ENTITY_ID, "conversation.home_assistant")
        token = current_chat_log.set(None)
        response: Any = None
        try:
            response = await self._hass.services.async_call(
                CONVERSATION_DOMAIN,
                "process",
                {
                    "text": user_input.text,
                    "agent_id": entity_id,
                    "conversation_id": user_input.conversation_id,
                    "language": user_input.language,
                },
                blocking=True,
                context=user_input.context,
                return_response=True,
            )
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error calling LOCAL_HA agent '%s': %s", entity_id, err)
        finally:
            current_chat_log.reset(token)

        if not response or "response" not in response:
            return None

        response_section = response["response"]
        assist_mode: bool = agent_config.get(CONF_AGENT_ASSIST_MODE, DEFAULT_AGENT_ASSIST_MODE)
        if assist_mode:
            response_type = str(response_section.get("response_type", ""))
            if response_type != "action_done":
                return None

        speech = response_section.get("speech", {}).get("plain", {}).get("speech", "")
        return _make_result(self._hass, speech, user_input.conversation_id) if speech else None

    async def close(self) -> None:
        """No persistent resources to release."""


class WebSearchAgentBackend:
    """Backend for Brave Search (or compatible) web-search agents."""

    def __init__(self, hass: "HomeAssistant", agent_config: dict[str, Any]) -> None:
        """Initialise with HA instance and agent configuration.

        Args:
            hass:         Home Assistant instance.
            agent_config: Agent configuration dict (must contain search API key).
        """
        self._hass = hass
        self._agent_config = agent_config
        self._client: Any = None

    @property
    def agent_id(self) -> str:
        """Return the agent identifier from its configuration."""
        return str(self._agent_config.get("id", ""))

    @property
    def agent_type(self) -> str:
        """Return the agent type constant."""
        return AGENT_TYPE_WEB_SEARCH

    def _get_client(self) -> Any:
        """Lazily create and return the :class:`WebSearchClient`."""
        from .web_search_client import WebSearchClient  # noqa: PLC0415

        if self._client is None:
            self._client = WebSearchClient(
                {
                    CONF_SEARCH_PROVIDER: self._agent_config.get(CONF_SEARCH_PROVIDER),
                    CONF_SEARCH_API_KEY: self._agent_config.get(CONF_SEARCH_API_KEY, ""),
                    CONF_SEARCH_RESULT_COUNT: self._agent_config.get(
                        CONF_SEARCH_RESULT_COUNT, DEFAULT_SEARCH_RESULT_COUNT
                    ),
                    CONF_SEARCH_MAX_SNIPPET_LEN: self._agent_config.get(
                        CONF_SEARCH_MAX_SNIPPET_LEN, DEFAULT_SEARCH_MAX_SNIPPET_LEN
                    ),
                    "timeout": self._agent_config.get("timeout", DEFAULT_SEARCH_TIMEOUT),
                }
            )
        return self._client

    async def process(
        self,
        user_input: ConversationInput,
        agent_config: dict[str, Any],  # noqa: ARG002
        *,
        timeout: int,  # noqa: ARG002
    ) -> ConversationResult | None:
        """Execute a web search and return the summarised result.

        Args:
            user_input:   The user's conversation input (uses ``text`` as the query).
            agent_config: Not used — this backend uses the config supplied at
                          construction time.
            timeout:      Not used — applied by the caller's asyncio.timeout context.

        Returns:
            :class:`ConversationResult` containing the search summary on success,
            ``None`` if the search returned no useful results.
        """
        client = self._get_client()
        summary = await client.search_and_summarise(user_input.text)
        if not summary:
            return None
        return _make_result(self._hass, summary, user_input.conversation_id)

    async def close(self) -> None:
        """Close the underlying :class:`WebSearchClient` HTTP session."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# Auto-register all four concrete backends
# ---------------------------------------------------------------------------

AgentBackendRegistry.register(AGENT_TYPE_OLLAMA, OllamaAgentBackend)
AgentBackendRegistry.register(AGENT_TYPE_INTEGRATED, IntegratedAgentBackend)
AgentBackendRegistry.register(AGENT_TYPE_LOCAL_HA, LocalHAAgentBackend)
AgentBackendRegistry.register(AGENT_TYPE_WEB_SEARCH, WebSearchAgentBackend)
