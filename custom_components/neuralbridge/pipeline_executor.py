"""Pipeline executor for NeuralBridge — processes agents and manages the routing pipeline."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

from homeassistant.components.conversation import (
    ConversationInput,
    ConversationResult,
)
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    AGENT_TYPE_WEB_SEARCH,
    COMPOUND_COMMAND_SEPARATOR,
    CONF_AGENT_CACHE_ENABLED,
    CONF_AGENT_ENABLED,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_AGENTS,
    CONF_ANNOUNCE_MEDIA_PLAYERS,
    CONF_IS_ROUTER,
    CONF_MAX_RETRIES,
    CONF_PRIORITY,
    CONF_RETRY_BASE_DELAY,
    CONF_SEARCH_API_KEY,
    CONF_SEARCH_MAX_SNIPPET_LEN,
    CONF_SEARCH_PROVIDER,
    CONF_SEARCH_RESULT_COUNT,
    CONF_SPLIT_COMPOUND_COMMANDS,
    CONF_TIMEOUT,
    DEFAULT_AGENT_CACHE_ENABLED,
    DEFAULT_AGENT_ENABLED,
    DEFAULT_ANNOUNCE_MEDIA_PLAYERS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_SEARCH_MAX_SNIPPET_LEN,
    DEFAULT_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_TIMEOUT,
    DEFAULT_SPLIT_COMPOUND_COMMANDS,
    DEFAULT_TIMEOUT,
    MSG_AGENT_FAILED,
    MSG_AGENT_SUCCESS,
    MSG_ALL_AGENTS_FAILED,
    MSG_NO_AGENTS_CONFIGURED,
    PRIORITY_ROUTER,
    SIGNAL_STATS_UPDATED,
)
from .router_engine import (
    RouterDecision,
    _apply_router_decision,
    _deterministic_classify,
    _extract_announce_text,
    _is_unhelpful_response,
    _split_compound_input,
)
from .web_search_client import WebSearchClient

if TYPE_CHECKING:
    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


class PipelineExecutor:
    """Encapsulates pipeline execution logic for NeuralBridgeAgent.

    Access to agent state is via ``self._agent``; within-class method calls
    stay as ``self.<method>()``.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this executor.
        """
        self._agent = agent

    async def _compute_result(self, user_input: ConversationInput) -> ConversationResult:
        """Compute the conversation result for the given user input.

        Implements priority-based routing: confirmation check → cache →
        router agents → processing agents.

        Args:
            user_input: The user's conversation input.

        Returns:
            The ConversationResult to return to the caller.
        """
        # Step 1: Handle confirmation response for a pending guard rail check
        confirmation_result = await self._agent._handle_confirmation_check(user_input)
        if confirmation_result is not None:
            return confirmation_result

        # Step 2: Check response cache
        cached = self._agent._response_cache.get(
            user_input.text, normalise=self._agent._response_cache.semantic
        )
        if cached is not None:
            _LOGGER.debug("Cache hit — returning cached response")
            return self._agent._create_result(cached, user_input.conversation_id)

        # Steps 3-5: Load enabled agents, split by role, run router check
        config = self._agent._get_config()
        agents = [
            a
            for a in config.get(CONF_AGENTS, [])
            if a.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED)
        ]
        if not agents:
            _LOGGER.warning(MSG_NO_AGENTS_CONFIGURED)
            return self._agent._create_error_result(
                self._agent._localized("responses", "no_agents"), user_input.conversation_id
            )

        sorted_agents = sorted(agents, key=lambda x: x.get(CONF_PRIORITY, 50))
        # Identify router agents: first-class is_router flag; falls back to priority==0
        router_agents = [
            a
            for a in sorted_agents
            if a.get(CONF_IS_ROUTER, False) or a.get(CONF_PRIORITY) == PRIORITY_ROUTER
        ]
        processing_agents = [
            a
            for a in sorted_agents
            if not (a.get(CONF_IS_ROUTER, False) or a.get(CONF_PRIORITY) == PRIORITY_ROUTER)
        ]

        return await self._agent._run_pipeline(config, user_input, router_agents, processing_agents)

    async def _run_pipeline(
        self,
        config: dict[str, Any],
        user_input: ConversationInput,
        router_agents: list[dict[str, Any]],
        processing_agents: list[dict[str, Any]],
    ) -> ConversationResult:
        """Run the routing and processing pipeline for a single conversation turn.

        Handles: named-agent override → Feature 11 broadcast announce (regex fast
        path) → router classification → Feature 11 broadcast announce (intent_hint
        path) → Feature 2 compound splitting → processing agents.

        Args:
            config:            The merged configuration dict (data + options).
            user_input:        The user's conversation input.
            router_agents:     Priority-sorted list of router agent configs.
            processing_agents: Priority-sorted list of processing agent configs.

        Returns:
            The ConversationResult produced by the first successful agent.
        """
        # Check for explicit named-agent override ("Ask Gemini: ...", "Use Ollama: ...", etc.)
        override = self._agent._check_explicit_agent_override(user_input, processing_agents)
        if override is not None:
            override_input, override_agents = override
            return await self._agent._try_processing_agents(override_agents, override_input)

        # Feature 11 — Broadcast Announcements: regex fast path (no router needed)
        announce_players: list[str] = list(
            config.get(CONF_ANNOUNCE_MEDIA_PLAYERS, DEFAULT_ANNOUNCE_MEDIA_PLAYERS) or []
        )
        if announce_players:
            announce_text = _extract_announce_text(user_input.text)
            if announce_text is not None:
                return await self._agent._send_broadcast_announcement(
                    announce_text, announce_players, user_input
                )

        # Deterministic pre-router: classify obvious web-search queries without
        # calling the LLM router.  Runs before _check_with_routers so that small
        # or restricted LLMs cannot misclassify news / external weather / travel
        # queries.  Only fires when no override or announce path has already
        # returned above.
        decision: RouterDecision | None = None
        det_decision = _deterministic_classify(user_input.text)
        if det_decision is not None:
            _LOGGER.debug(
                "Deterministic pre-router: web_search=True for %r",
                user_input.text[:80],
            )
            processing_agents = _apply_router_decision(det_decision, processing_agents)
        elif router_agents:
            decision = await self._agent._check_with_routers(
                user_input, router_agents, processing_agents
            )
            if decision is None:
                _LOGGER.debug("Router agents blocked the request")
                return self._agent._create_error_result(
                    self._agent._localized("responses", "router_blocked"),
                    user_input.conversation_id,
                )
            # Feature 11 — Broadcast Announcements: intent_hint path (router detected announce)
            if announce_players and decision.intent_hint == "announce":
                announce_text = _extract_announce_text(user_input.text, intent_hint="announce")
                if announce_text is not None:
                    return await self._agent._send_broadcast_announcement(
                        announce_text, announce_players, user_input
                    )
            processing_agents = _apply_router_decision(decision, processing_agents)

        # Feature 2 — Compound Command Splitting
        if config.get(CONF_SPLIT_COMPOUND_COMMANDS, DEFAULT_SPLIT_COMPOUND_COMMANDS):
            has_local_ha = any(
                a.get(CONF_AGENT_TYPE) == AGENT_TYPE_LOCAL_HA for a in processing_agents
            )
            if has_local_ha:
                fragments = _split_compound_input(user_input.text)
                if len(fragments) > 1:
                    return await self._agent._process_compound_fragments(
                        fragments, processing_agents, user_input
                    )

        # Step 6: Try processing agents in priority order
        return await self._agent._try_processing_agents(processing_agents, user_input, decision)

    def _maybe_add_to_chat_log(self, result: ConversationResult) -> None:
        """Add the assistant response to the active ChatLog if one exists.

        The conversation framework maintains a ChatLog to track the conversation
        history.  Our response must be recorded as an AssistantContent entry so
        HA does not generate a WARNING that includes the user's message text (PII).

        This is a no-op when no ChatLog is active for the current context.

        Args:
            result: The ConversationResult containing the response to record.
        """
        from homeassistant.components.conversation.chat_log import (  # noqa: PLC0415
            AssistantContent,
            current_chat_log,
        )

        if (chat_log := current_chat_log.get()) is None:
            return

        speech = self._agent._extract_response_text(result)
        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(agent_id=self._agent.entity_id, content=speech or None)
        )

    async def _process_compound_fragments(
        self,
        fragments: list[str],
        processing_agents: list[dict[str, Any]],
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Process each compound command fragment independently and combine results.

        Each fragment is sent to the processing agents in turn.  If a fragment
        fails (no result), a localised fallback string is used so no fragment is
        silently dropped.  All per-fragment responses are joined with the
        ``COMPOUND_COMMAND_SEPARATOR`` (`` · ``).

        Args:
            fragments: Individual command strings from _split_compound_input.
            processing_agents: Priority-sorted processing agent configs.
            user_input: Original ConversationInput (text will be replaced per fragment).

        Returns:
            A ConversationResult whose speech text is the joined responses.
        """
        speech_parts: list[str] = []
        fallback_text = self._agent._localized("responses", "fallback")
        for fragment in fragments:
            fragment_input = dataclasses.replace(user_input, text=fragment)
            result = await self._agent._try_processing_agents(processing_agents, fragment_input)
            fragment_speech = self._agent._extract_response_text(result)
            speech_parts.append(fragment_speech if fragment_speech else fallback_text)
        combined = COMPOUND_COMMAND_SEPARATOR.join(speech_parts)
        return self._agent._create_result(combined, user_input.conversation_id)

    async def _try_processing_agents(
        self,
        processing_agents: list[dict[str, Any]],
        user_input: ConversationInput,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult:
        """Try each processing agent in priority order until one gives a helpful answer.

        When an agent returns the CANNOT_ANSWER sentinel the pipeline continues
        to the next agent.  If a subsequent agent succeeds, a localized preamble
        is prepended to its response so the user knows why the answer is coming
        from a different source (e.g. "Let me check the web for that.").

        Args:
            processing_agents: Priority-sorted list of non-router agent configs.
            user_input: The user's conversation input.
            router_decision: Optional routing decision carrying ``relevant_sensors``
                             for targeted sensor injection.

        Returns:
            The first genuinely helpful ConversationResult, or a fallback error result.
        """
        rerouted: bool = False
        for agent_config in processing_agents:
            result = await self._agent._try_agent_with_tracking(
                agent_config, user_input, router_decision
            )
            if result is None:
                continue
            response_text = self._agent._extract_response_text(result)
            if _is_unhelpful_response(response_text):
                _LOGGER.info(
                    "Agent %s returned CANNOT_ANSWER — continuing to next agent",
                    agent_config.get(CONF_AGENT_NAME, "Unknown"),
                )
                rerouted = True
                continue
            # Got a real answer — prefix with a processing preamble if we re-routed
            if rerouted:
                agent_type: str = agent_config.get(CONF_AGENT_TYPE, "")
                if agent_type == AGENT_TYPE_WEB_SEARCH:
                    preamble = self._agent._localized("responses", "reroute_web_search")
                else:
                    preamble = self._agent._localized("responses", "reroute_agent")
                if preamble:
                    result = self._agent._create_result(
                        preamble + response_text, user_input.conversation_id
                    )
            return result

        _LOGGER.warning(MSG_ALL_AGENTS_FAILED)
        return self._agent._create_error_result(
            self._agent._localized("responses", "fallback"), user_input.conversation_id
        )

    async def _try_agent_with_tracking(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult | None:
        """Try a single agent with circuit breaker guard, then retry logic.

        Args:
            agent_config:    Configuration dict for the agent to try.
            user_input:      The user's conversation input.
            router_decision: Optional routing decision carrying ``relevant_sensors``
                             for targeted sensor injection.

        Returns:
            ConversationResult on success, None if the agent failed or was skipped.
        """
        agent_id: str = agent_config.get("id", "")
        agent_name: str = agent_config.get(CONF_AGENT_NAME, "Unknown")

        if self._agent._circuit_breaker.is_open(agent_id):
            _LOGGER.warning(
                "Agent %s circuit is tripped — skipping until cooldown expires", agent_name
            )
            return None

        return await self._agent._try_agent_with_retries(
            agent_config, user_input, agent_id, agent_name, router_decision
        )

    async def _try_agent_with_retries(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        agent_id: str,
        agent_name: str,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult | None:
        """Attempt an agent call with exponential back-off on failure.

        Args:
            agent_config:    Configuration dict for the agent to try.
            user_input:      The user's conversation input.
            agent_id:        Unique identifier for the agent.
            agent_name:      Display name of the agent (for logging).
            router_decision: Optional routing decision carrying ``relevant_sensors``
                             for targeted sensor injection.

        Returns:
            ConversationResult on success, None if all attempts are exhausted.
        """
        config = self._agent._get_config()
        max_retries = int(config.get(CONF_MAX_RETRIES, DEFAULT_MAX_RETRIES))
        retry_base_delay = float(config.get(CONF_RETRY_BASE_DELAY, DEFAULT_RETRY_BASE_DELAY))

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = retry_base_delay * (2 ** (attempt - 1))
                _LOGGER.debug(
                    "Retry %d/%d for agent %s — waiting %.1fs before next attempt",
                    attempt,
                    max_retries,
                    agent_name,
                    delay,
                )
                await asyncio.sleep(delay)

            self._agent._statistics.record_request(agent_id, agent_name)
            start_time = time.monotonic()
            result, timed_out = await self._agent._try_agent(
                agent_config, user_input, router_decision
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000

            if result is not None:
                response_text = self._agent._extract_response_text(result)
                if _is_unhelpful_response(response_text):
                    _LOGGER.info(
                        "Agent %s cannot answer — sentinel detected, will re-route",
                        agent_name,
                    )
                    # Return the canonical sentinel result so _try_processing_agents
                    # can distinguish re-route from real failure and add a preamble.
                    from .const import CANNOT_ANSWER_SENTINEL  # noqa: PLC0415

                    return self._agent._create_result(
                        CANNOT_ANSWER_SENTINEL, user_input.conversation_id
                    )
                return await self._agent._handle_successful_result(
                    agent_config, result, user_input, agent_id, elapsed_ms
                )

            self._agent._record_agent_failure(agent_id, timed_out)
            if self._agent._circuit_breaker.is_open(agent_id):
                _LOGGER.warning(
                    "Circuit tripped for agent %s after attempt %d — stopping retries",
                    agent_name,
                    attempt + 1,
                )
                break

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

        self._agent._circuit_breaker.record_success(agent_id)
        self._agent._statistics.record_success(agent_id, elapsed_ms)
        async_dispatcher_send(
            self._agent.hass,
            SIGNAL_STATS_UPDATED.format(entry_id=self._agent._config_entry.entry_id),
        )

        response_text = self._agent._extract_response_text(result)
        self._agent._maybe_cache_response(agent_config, user_input.text, response_text, agent_name)

        guard_action = await self._agent._check_guardrails(
            agent_config, result, user_input.conversation_id
        )
        if isinstance(guard_action, ConversationResult):
            return guard_action

        # Feature 4 — high-stakes confirmation (LOCAL_HA only)
        high_stakes_action = await self._agent._check_high_stakes(agent_config, result, user_input)
        if isinstance(high_stakes_action, ConversationResult):
            return high_stakes_action

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
            return str(result.response.speech.get("plain", {}).get("speech", ""))
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
            self._agent._response_cache.store(
                input_text,
                response_text,
                agent_name,
                normalise=self._agent._response_cache.semantic,
            )

    def _record_agent_failure(self, agent_id: str, timed_out: bool) -> None:
        """Record a failure or timeout for circuit breaker, stats, and dispatcher.

        Args:
            agent_id: Unique identifier for the agent that failed.
            timed_out: True if the failure was a timeout; False for other errors.
        """
        if timed_out:
            self._agent._circuit_breaker.record_timeout(agent_id)
            self._agent._statistics.record_timeout(agent_id)
        else:
            self._agent._circuit_breaker.record_failure(agent_id)
            self._agent._statistics.record_failure(agent_id)
        async_dispatcher_send(
            self._agent.hass,
            SIGNAL_STATS_UPDATED.format(entry_id=self._agent._config_entry.entry_id),
        )

    async def _try_agent(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: RouterDecision | None = None,
    ) -> tuple[ConversationResult | None, bool]:
        """Attempt to process input with a specific agent.

        Args:
            agent_config:    Configuration dict for the agent to try.
            user_input:      The user's conversation input.
            router_decision: Optional routing decision carrying ``relevant_sensors``
                             for targeted sensor injection into LOCAL_HA / EXISTING
                             agents.

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
                if agent_type in (AGENT_TYPE_OLLAMA, AGENT_TYPE_INTEGRATED, AGENT_TYPE_LOCAL_HA):
                    result = await self._agent._llm_proxy.process(
                        agent_config, user_input, router_decision
                    )
                elif agent_type == AGENT_TYPE_WEB_SEARCH:
                    result = await self._agent._process_with_web_search(agent_config, user_input)
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

    async def _process_with_web_search(
        self, agent_config: dict[str, Any], user_input: ConversationInput
    ) -> ConversationResult | None:
        """Process input by executing a web search and returning a summary.

        A ``WebSearchClient`` is instantiated on first use and cached per agent
        ID (keyed by ``agent_config["id"]``).  The summary returned by the
        client is turned directly into a ``ConversationResult``.

        Args:
            agent_config: Configuration dict for the web-search agent.
            user_input: The user's conversation input.

        Returns:
            ConversationResult containing the search summary on success,
            None if the search returned no results.
        """
        agent_id: str = agent_config.get("id", "")

        config_for_client: dict[str, Any] = {
            CONF_SEARCH_PROVIDER: agent_config.get(CONF_SEARCH_PROVIDER),
            CONF_SEARCH_API_KEY: agent_config.get(CONF_SEARCH_API_KEY, ""),
            CONF_SEARCH_RESULT_COUNT: agent_config.get(
                CONF_SEARCH_RESULT_COUNT, DEFAULT_SEARCH_RESULT_COUNT
            ),
            CONF_SEARCH_MAX_SNIPPET_LEN: agent_config.get(
                CONF_SEARCH_MAX_SNIPPET_LEN, DEFAULT_SEARCH_MAX_SNIPPET_LEN
            ),
            "timeout": agent_config.get("timeout", DEFAULT_SEARCH_TIMEOUT),
        }

        if agent_id not in self._agent._web_search_clients:
            self._agent._web_search_clients[agent_id] = WebSearchClient(config_for_client)

        client = self._agent._web_search_clients[agent_id]
        summary = await client.search_and_summarise(user_input.text)

        if summary:
            return self._agent._create_result(summary, user_input.conversation_id)

        return None

    def _check_explicit_agent_override(
        self,
        user_input: ConversationInput,
        agents: list[dict[str, Any]],
    ) -> tuple[ConversationInput, list[dict[str, Any]]] | None:
        """Detect a named-agent override prefix and route directly to that agent.

        Recognises three pattern variants (case-insensitive):

        * ``"Ask {name}: {query}"``
        * ``"Use {name}: {query}"``
        * ``"{name}: {query}"``

        When matched, the query text is stripped of the prefix and a modified
        :class:`ConversationInput` is returned alongside the matched agent,
        bypassing router classification entirely.  This mirrors the Alexa
        ``"Ask {skill}"`` pattern, letting power users target a specific agent
        by name without touching the routing pipeline.

        Args:
            user_input: The user's conversation input.
            agents:     The list of enabled processing agent configs.

        Returns:
            ``(stripped_input, [matched_agent])`` when an override prefix is
            detected, or ``None`` when the input should follow normal routing.
        """
        text_lower = user_input.text.lower()
        for agent in agents:
            name: str = agent.get(CONF_AGENT_NAME, "")
            if not name:
                continue
            name_lower = name.lower()
            for prefix in (
                f"ask {name_lower}:",
                f"use {name_lower}:",
                f"{name_lower}:",
            ):
                if text_lower.startswith(prefix):
                    stripped = user_input.text[len(prefix) :].strip()
                    if not stripped:
                        continue
                    new_input = dataclasses.replace(user_input, text=stripped)
                    return new_input, [agent]
        return None
