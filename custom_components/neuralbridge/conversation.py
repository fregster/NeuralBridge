"""Conversation agent for NeuralBridge."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .circuit_breaker import CircuitBreaker
from .const import (
    AGENT_TYPE_EXISTING,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    CONF_AGENT_CACHE_ENABLED,
    CONF_AGENT_ENABLED,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_AGENTS,
    CONF_ENTITY_ID,
    CONF_GUARD_RAIL_ACTION,
    CONF_GUARD_RAIL_AI_THRESHOLD,
    CONF_GUARD_RAIL_DETOXIFY_THRESHOLD,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_GUARD_RAIL_USE_DETOXIFY,
    CONF_LANGUAGE,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_PRIORITY,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DATA_RESPONSE_CACHE,
    DATA_SESSION_MEMORY,
    DATA_STATISTICS,
    DEFAULT_AGENT_CACHE_ENABLED,
    DEFAULT_AGENT_ENABLED,
    DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_GUARD_RAIL_ACTION,
    DEFAULT_GUARD_RAIL_AI_THRESHOLD,
    DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD,
    DEFAULT_GUARD_RAIL_ENABLED,
    DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    DEFAULT_GUARD_RAIL_USE_DETOXIFY,
    DEFAULT_LANGUAGE,
    DEFAULT_RESPONSE_CACHE_ENABLED,
    DEFAULT_RESPONSE_CACHE_TTL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    DOMAIN,
    EVENT_GUARD_RAIL_TRIGGERED,
    GUARD_RAIL_ACTION_BLOCK,
    GUARD_RAIL_ACTION_NOTIFY_ASK,
    GUARD_RAIL_ACTION_WARN,
    MSG_AGENT_FAILED,
    MSG_AGENT_SUCCESS,
    MSG_ALL_AGENTS_FAILED,
    MSG_NO_AGENTS_CONFIGURED,
    PRIORITY_ROUTER,
    ROUTER_CLASSIFICATION_PROMPT,
    SIGNAL_STATS_UPDATED,
)
from .guard_rail import GuardRailCache, GuardRailChecker
from .languages_loader import get_string
from .ollama_client import OllamaClient
from .response_cache import ResponseCache
from .session_memory import SessionMemory
from .statistics import AgentStatistics

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up NeuralBridge conversation agent."""
    agent = NeuralBridgeAgent(hass, config_entry)
    async_add_entities([agent])


class NeuralBridgeAgent(ConversationEntity):
    """NeuralBridge conversation agent with priority-based routing."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self._config_entry = config_entry
        self._attr_name = "NeuralBridge"
        self._attr_unique_id = config_entry.entry_id
        self._ollama_clients: dict[str, OllamaClient] = {}
        self._guard_rail_checker: GuardRailChecker | None = None
        self._guard_rail_cache = GuardRailCache()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            cooldown_seconds=DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
        )
        # Shared objects created by __init__.py and stored in hass.data
        entry_data = hass.data.get(DOMAIN, {}).get(config_entry.entry_id, {})
        self._statistics: AgentStatistics = entry_data.get(DATA_STATISTICS, AgentStatistics())
        self._response_cache: ResponseCache = entry_data.get(
            DATA_RESPONSE_CACHE,
            ResponseCache(
                enabled=DEFAULT_RESPONSE_CACHE_ENABLED,
                ttl_seconds=DEFAULT_RESPONSE_CACHE_TTL,
            ),
        )
        self._session_memory: SessionMemory = entry_data.get(DATA_SESSION_MEMORY, SessionMemory())

    def _localized(self, *keys: str) -> str:
        """Return a localised string for the currently configured language.

        Args:
            *keys: Path within the language YAML (e.g. ``"responses"``, ``"fallback"``).

        Returns:
            Localised string, or empty string if the key path is not found.
        """
        lang = self._get_config().get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
        return get_string(str(lang), *keys)

    @property
    def supported_languages(self) -> Literal["*"]:
        """Return supported languages.

        NeuralBridge routes to underlying agents that handle language themselves,
        so all languages are accepted.
        """
        return "*"

    def _get_config(self) -> dict[str, Any]:
        """Return a merged view of data + options from the config entry.

        Options override data so live changes (options flow) take effect
        without a restart.

        Returns:
            Merged configuration dictionary.
        """
        merged: dict[str, Any] = {**self._config_entry.data, **self._config_entry.options}
        return merged

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a user input through the priority-based routing system."""
        _LOGGER.debug("Processing input: %s", user_input.text)

        # Step 1: Handle confirmation response for a pending guard rail check
        confirmation_result = await self._handle_confirmation_check(user_input)
        if confirmation_result is not None:
            return confirmation_result

        # Step 2: Check response cache
        cached = self._response_cache.get(user_input.text)
        if cached is not None:
            _LOGGER.debug("Cache hit — returning cached response")
            return self._create_result(cached, user_input.conversation_id)

        # Steps 3–5: Load enabled agents, split by role, run router check
        config = self._get_config()
        agents = [
            a
            for a in config.get(CONF_AGENTS, [])
            if a.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED)
        ]
        if not agents:
            _LOGGER.warning(MSG_NO_AGENTS_CONFIGURED)
            return self._create_error_result(
                self._localized("responses", "no_agents"), user_input.conversation_id
            )

        sorted_agents = sorted(agents, key=lambda x: x.get(CONF_PRIORITY, 50))
        router_agents = [a for a in sorted_agents if a.get(CONF_PRIORITY) == PRIORITY_ROUTER]
        processing_agents = [a for a in sorted_agents if a.get(CONF_PRIORITY, 0) > PRIORITY_ROUTER]

        if router_agents:
            should_process = await self._check_with_routers(user_input, router_agents)
            if not should_process:
                _LOGGER.debug("Router agents determined input should not be processed")
                return self._create_error_result(
                    self._localized("responses", "router_blocked"),
                    user_input.conversation_id,
                )

        # Step 6: Try processing agents in priority order
        return await self._try_processing_agents(processing_agents, user_input)

    async def _handle_confirmation_check(
        self, user_input: ConversationInput
    ) -> ConversationResult | None:
        """Handle a yes/no confirmation for a pending guard rail response.

        Args:
            user_input: The user's conversation input.

        Returns:
            A ConversationResult if a pending response was resolved, else None.
        """
        if user_input.text.lower() not in ("yes", "no"):
            return None

        pending = await self._guard_rail_cache.get_pending_response(
            user_input.conversation_id or "default"
        )
        if not pending:
            return None

        response_text, _ = pending
        await self._guard_rail_cache.clear_pending_response(user_input.conversation_id or "default")
        if user_input.text.lower() == "yes":
            return self._create_result(response_text, user_input.conversation_id)
        return self._create_error_result(
            self._localized("responses", "guard_rail_blocked"), user_input.conversation_id
        )

    async def _try_processing_agents(
        self, processing_agents: list[dict[str, Any]], user_input: ConversationInput
    ) -> ConversationResult:
        """Try each processing agent in priority order until one succeeds.

        Args:
            processing_agents: Priority-sorted list of non-router agent configs.
            user_input: The user's conversation input.

        Returns:
            The first successful ConversationResult, or a fallback error result.
        """
        for agent_config in processing_agents:
            result = await self._try_agent_with_tracking(agent_config, user_input)
            if result is not None:
                return result

        _LOGGER.warning(MSG_ALL_AGENTS_FAILED)
        return self._create_error_result(
            self._localized("responses", "fallback"), user_input.conversation_id
        )

    async def _try_agent_with_tracking(
        self, agent_config: dict[str, Any], user_input: ConversationInput
    ) -> ConversationResult | None:
        """Try a single agent with circuit breaker, stats, and cache tracking.

        Args:
            agent_config: Configuration dict for the agent to try.
            user_input: The user's conversation input.

        Returns:
            ConversationResult on success, None if the agent failed or was skipped.
        """
        agent_id: str = agent_config.get("id", "")
        agent_name: str = agent_config.get(CONF_AGENT_NAME, "Unknown")

        if self._circuit_breaker.is_open(agent_id):
            _LOGGER.warning(
                "Agent %s circuit is tripped — skipping until cooldown expires", agent_name
            )
            return None

        self._statistics.record_request(agent_id, agent_name)
        start_time = time.monotonic()
        result, timed_out = await self._try_agent(agent_config, user_input)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        if result is not None:
            return await self._handle_successful_result(
                agent_config, result, user_input, agent_id, elapsed_ms
            )

        self._record_agent_failure(agent_id, timed_out)
        return None

    async def _handle_successful_result(
        self,
        agent_config: dict[str, Any],
        result: ConversationResult,
        user_input: ConversationInput,
        agent_id: str,
        elapsed_ms: float,
    ) -> ConversationResult:
        """Update stats/cache and run guard rails on a successful agent response.

        Args:
            agent_config: Configuration of the agent that succeeded.
            result: The successful ConversationResult.
            user_input: The user's original conversation input.
            agent_id: Unique identifier for the agent.
            elapsed_ms: Time taken by the agent in milliseconds.

        Returns:
            The final ConversationResult to return (may be modified by guard rails).
        """
        agent_name: str = agent_config.get(CONF_AGENT_NAME, "Unknown")

        self._circuit_breaker.record_success(agent_id)
        self._statistics.record_success(agent_id, elapsed_ms)
        async_dispatcher_send(
            self.hass,
            SIGNAL_STATS_UPDATED.format(entry_id=self._config_entry.entry_id),
        )

        response_text = self._extract_response_text(result)
        self._maybe_cache_response(agent_config, user_input.text, response_text, agent_name)

        guard_action = await self._check_guardrails(
            agent_config, result, user_input.conversation_id
        )
        if isinstance(guard_action, ConversationResult):
            return guard_action

        _LOGGER.info(MSG_AGENT_SUCCESS, agent_name, agent_config.get(CONF_PRIORITY))
        return result

    def _extract_response_text(self, result: ConversationResult) -> str:
        """Extract the plain speech text from a ConversationResult.

        Args:
            result: The conversation result to extract text from.

        Returns:
            Speech text string, or empty string if unavailable.
        """
        if result.response and result.response.speech:
            return result.response.speech.get("plain", {}).get("speech", "")
        return ""

    def _maybe_cache_response(
        self,
        agent_config: dict[str, Any],
        input_text: str,
        response_text: str,
        agent_name: str,
    ) -> None:
        """Store a response in the cache if the agent and global settings allow it.

        Args:
            agent_config: Configuration of the agent that produced the response.
            input_text: The original user input (used as cache key).
            response_text: The response text to cache.
            agent_name: Name of the agent (for cache metadata).
        """
        if response_text and agent_config.get(
            CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
        ):
            self._response_cache.store(input_text, response_text, agent_name)

    def _record_agent_failure(self, agent_id: str, timed_out: bool) -> None:
        """Record a failure or timeout for circuit breaker, stats, and dispatcher.

        Args:
            agent_id: Unique identifier for the agent that failed.
            timed_out: True if the failure was a timeout; False for other errors.
        """
        if timed_out:
            self._circuit_breaker.record_timeout(agent_id)
            self._statistics.record_timeout(agent_id)
        else:
            self._circuit_breaker.record_failure(agent_id)
            self._statistics.record_failure(agent_id)
        async_dispatcher_send(
            self.hass,
            SIGNAL_STATS_UPDATED.format(entry_id=self._config_entry.entry_id),
        )

    async def _check_with_routers(
        self, user_input: ConversationInput, router_agents: list[dict[str, Any]]
    ) -> bool:
        """Check with router/filter agents whether the input should be processed.

        Each router agent is asked to classify the input as PASS or BLOCK.
        The first BLOCK response causes the method to return False immediately.
        Falls back to True (PASS) if no router is able to classify.

        Args:
            user_input: The user's conversation input.
            router_agents: List of priority-0 agent configurations.

        Returns:
            True if input should proceed; False to block it.
        """
        for router_config in router_agents:
            _LOGGER.debug("Checking with router: %s", router_config.get(CONF_AGENT_NAME))
            should_pass = await self._classify_with_router(router_config, user_input.text)
            if not should_pass:
                _LOGGER.info(
                    "Router agent '%s' blocked the request",
                    router_config.get(CONF_AGENT_NAME),
                )
                return False
        return True

    async def _classify_with_router(self, router_config: dict[str, Any], user_text: str) -> bool:
        """Ask a router (priority-0) Ollama agent to classify user text.

        Sends a one-shot classification prompt via ``OllamaClient.generate()``
        and looks for ``BLOCK`` in the response.  Any failure — wrong agent type,
        missing configuration, network error, or empty response — defaults to
        ``True`` (fail-open) so a broken router never silences the assistant.

        Args:
            router_config: Configuration dict for the router agent.
            user_text: The raw user input text to classify.

        Returns:
            True if the input should be processed (PASS), False to block it.
        """
        if router_config.get(CONF_AGENT_TYPE) != AGENT_TYPE_OLLAMA:
            _LOGGER.debug(
                "Non-Ollama router agent '%s' — defaulting to PASS",
                router_config.get(CONF_AGENT_NAME),
            )
            return True

        agent_id: str = router_config.get("id", "")
        ollama_url: str = router_config.get(CONF_OLLAMA_URL, "")
        ollama_model: str = router_config.get(CONF_OLLAMA_MODEL, "")
        timeout: int = router_config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)

        if not ollama_url or not ollama_model:
            _LOGGER.warning(
                "Router agent '%s' is missing URL or model — defaulting to PASS",
                router_config.get(CONF_AGENT_NAME),
            )
            return True

        if agent_id not in self._ollama_clients:
            self._ollama_clients[agent_id] = OllamaClient(ollama_url, ollama_model, timeout)

        client = self._ollama_clients[agent_id]
        prompt = ROUTER_CLASSIFICATION_PROMPT.format(user_text=user_text)
        response = await client.generate(prompt)

        if not response:
            _LOGGER.warning(
                "Router agent '%s' returned no response — defaulting to PASS",
                router_config.get(CONF_AGENT_NAME),
            )
            return True

        decision = response.strip().upper()
        _LOGGER.debug(
            "Router agent '%s' decision: %s",
            router_config.get(CONF_AGENT_NAME),
            decision[:10],  # Truncate; avoids echoing full user text in verbose model output
        )
        return "BLOCK" not in decision

    async def _try_agent(
        self, agent_config: dict[str, Any], user_input: ConversationInput
    ) -> tuple[ConversationResult | None, bool]:
        """Attempt to process input with a specific agent.

        Args:
            agent_config: Configuration dict for the agent to try.
            user_input: The user's conversation input.

        Returns:
            Tuple of (result, timed_out).  result is None on failure or timeout.
            timed_out is True only when the failure was a timeout.
        """
        agent_type = agent_config.get(CONF_AGENT_TYPE)
        timeout = agent_config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)

        _LOGGER.debug(
            "Trying agent: %s (type: %s, priority: %d)",
            agent_config.get(CONF_AGENT_NAME),
            agent_type,
            agent_config.get(CONF_PRIORITY, 0),
        )

        try:
            async with asyncio.timeout(timeout):
                if agent_type == AGENT_TYPE_OLLAMA:
                    result = await self._process_with_ollama(agent_config, user_input)
                elif agent_type in (AGENT_TYPE_EXISTING, AGENT_TYPE_LOCAL_HA):
                    result = await self._process_with_existing(agent_config, user_input)
                else:
                    _LOGGER.error("Unknown agent type: %s", agent_type)
                    return None, False
                return result, False

        except asyncio.TimeoutError:
            _LOGGER.warning(
                "Agent %s timed out after %d seconds",
                agent_config.get(CONF_AGENT_NAME),
                timeout,
            )
            return None, True

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error(
                MSG_AGENT_FAILED,
                agent_config.get(CONF_AGENT_NAME),
                agent_config.get(CONF_PRIORITY, 0),
                err,
            )
            return None, False

    async def _process_with_ollama(
        self, agent_config: dict[str, Any], user_input: ConversationInput
    ) -> ConversationResult | None:
        """Process input with an Ollama agent using session context (#9, #10).

        Args:
            agent_config: Configuration dict for the Ollama agent.
            user_input: The user's conversation input.

        Returns:
            ConversationResult on success, None on failure.
        """
        agent_id: str = agent_config.get("id", "")
        ollama_url = agent_config.get(CONF_OLLAMA_URL)
        ollama_model = agent_config.get(CONF_OLLAMA_MODEL)
        timeout = agent_config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
        system_prompt: str = agent_config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)

        if agent_id not in self._ollama_clients:
            self._ollama_clients[agent_id] = OllamaClient(ollama_url, ollama_model, timeout)

        client = self._ollama_clients[agent_id]

        # Build message list: optional system prompt + history + current turn (#9)
        history = self._session_memory.get_messages(user_input.conversation_id)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_input.text})

        response_text = await client.chat(messages)

        if response_text:
            # Record turn in session memory (#9)
            self._session_memory.add_turn(
                user_input.conversation_id, user_input.text, response_text
            )
            return self._create_result(response_text, user_input.conversation_id)

        return None

    async def _process_with_existing(
        self, agent_config: dict[str, Any], user_input: ConversationInput
    ) -> ConversationResult | None:
        """Process input with an existing Home Assistant conversation agent.

        Args:
            agent_config: Configuration dict for the existing agent.
            user_input: The user's conversation input.

        Returns:
            ConversationResult on success, None on failure.
        """
        entity_id = agent_config.get(CONF_ENTITY_ID)

        agent_state = self.hass.states.get(entity_id)
        if not agent_state:
            _LOGGER.error("Conversation agent %s not found", entity_id)
            return None

        try:
            response = await self.hass.services.async_call(
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

            if not response or "response" not in response:
                _LOGGER.error("Agent %s returned unexpected response shape", entity_id)
                return None

            speech_text = response["response"].get("speech", {}).get("plain", {}).get("speech", "")
            if speech_text:
                return self._create_result(speech_text, user_input.conversation_id)

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error calling conversation agent %s: %s", entity_id, err)

        return None

    async def _check_guardrails(
        self,
        agent_config: dict[str, Any],
        result: ConversationResult,
        conversation_id: str | None,
    ) -> ConversationResult | None:
        """Check guard rails on agent output and fire an HA event if triggered (#13).

        Args:
            agent_config: Configuration of the agent that produced the result.
            result: The conversation result to check.
            conversation_id: Conversation ID for pending-response caching.

        Returns:
            ConversationResult if a guard rail action produced a response, None otherwise.
        """
        config = self._get_config()

        guard_rail_enabled = config.get(CONF_GUARD_RAIL_ENABLED, DEFAULT_GUARD_RAIL_ENABLED)
        if not guard_rail_enabled:
            return None

        agent_guard_rail_enabled = agent_config.get(
            CONF_GUARD_RAIL_ENABLED_FOR_AGENT, DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
        )
        if not agent_guard_rail_enabled:
            return None

        response_text = self._extract_response_text(result)
        if not response_text:
            return None

        if self._guard_rail_checker is None:
            ai_threshold = config.get(CONF_GUARD_RAIL_AI_THRESHOLD, DEFAULT_GUARD_RAIL_AI_THRESHOLD)
            use_detoxify = config.get(CONF_GUARD_RAIL_USE_DETOXIFY, DEFAULT_GUARD_RAIL_USE_DETOXIFY)
            detoxify_threshold = config.get(
                CONF_GUARD_RAIL_DETOXIFY_THRESHOLD, DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD
            )
            self._guard_rail_checker = GuardRailChecker(
                rules=None,
                ai_threshold=ai_threshold,
                use_detoxify=use_detoxify,
                detoxify_threshold=detoxify_threshold,
            )

        guard_rail_agent_config = await self._get_guard_rail_agent_config()
        guard_rail_result = await self._guard_rail_checker.check_output(
            response_text,
            use_ai=guard_rail_agent_config is not None,
            ai_agent_config=guard_rail_agent_config,
        )

        if guard_rail_result.is_safe:
            return None

        action = config.get(CONF_GUARD_RAIL_ACTION, DEFAULT_GUARD_RAIL_ACTION)
        agent_name = agent_config.get(CONF_AGENT_NAME, "Unknown")

        _LOGGER.warning(
            "Guard rail flagged output from agent %s: category=%s, confidence=%.2f, action=%s",
            agent_name,
            guard_rail_result.category,
            guard_rail_result.confidence,
            action,
        )

        # Fire HA event so users can build automations on guard rail triggers (#13)
        self.hass.bus.async_fire(
            EVENT_GUARD_RAIL_TRIGGERED,
            {
                "agent_name": agent_name,
                "agent_id": agent_config.get("id", ""),
                "category": guard_rail_result.category,
                "confidence": round(guard_rail_result.confidence, 3),
                "action": action,
                "reason": guard_rail_result.reason,
            },
        )

        if action == GUARD_RAIL_ACTION_BLOCK:
            return self._create_error_result(
                self._localized("responses", "guard_rail_blocked"), conversation_id
            )

        if action == GUARD_RAIL_ACTION_WARN:
            prefix = self._localized("responses", "guard_rail_warning_prefix")
            result.response.async_set_speech(f"{prefix}{response_text}")
            return None

        if action == GUARD_RAIL_ACTION_NOTIFY_ASK:
            await self._guard_rail_cache.store_pending_response(
                conversation_id or "default", response_text, guard_rail_result
            )
            return self._create_result(
                self._localized("responses", "guard_rail_notify_ask"), conversation_id
            )

        return None

    async def _get_guard_rail_agent_config(self) -> dict[str, Any] | None:
        """Return the configuration for the designated guard rail agent.

        Returns:
            Agent configuration dict, or None if not configured.
        """
        config = self._get_config()
        guard_rail_agent_id = config.get("guard_rail_agent_id")
        if not guard_rail_agent_id:
            return None

        for agent in config.get(CONF_AGENTS, []):
            if agent.get("id") == guard_rail_agent_id:
                return agent

        return None

    def _create_result(
        self, response_text: str, conversation_id: str | None = None
    ) -> ConversationResult:
        """Create a successful conversation result.

        Args:
            response_text: The text to return to the user.
            conversation_id: The conversation session ID to echo back to the caller.

        Returns:
            A ConversationResult with the given speech text.
        """
        response = intent.IntentResponse(language=self.hass.config.language)
        response.async_set_speech(response_text)
        return ConversationResult(response=response, conversation_id=conversation_id)

    def _create_error_result(
        self, error_message: str, conversation_id: str | None = None
    ) -> ConversationResult:
        """Create an error conversation result.

        Args:
            error_message: The error text to return to the user.
            conversation_id: The conversation session ID to echo back to the caller.

        Returns:
            A ConversationResult with the given error speech text.
        """
        response = intent.IntentResponse(language=self.hass.config.language)
        response.async_set_speech(error_message)
        return ConversationResult(response=response, conversation_id=conversation_id)

    async def async_will_remove_from_hass(self) -> None:
        """Clean up resources when entity is removed from Home Assistant."""
        for client in self._ollama_clients.values():
            await client.close()
        self._ollama_clients.clear()
