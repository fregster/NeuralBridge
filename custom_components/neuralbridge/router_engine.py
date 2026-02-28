"""Router engine for NeuralBridge — orchestration, fallback, stats dispatch, logging.

Stage 3 refactor: this module is now the thin orchestrator.  Heavy concerns
have been split into:

* :mod:`.router_decision` — :class:`RouterDecision` value object and JSON parsers.
* :mod:`.router_utils` — pure text-processing helpers (refusal detection,
  deterministic classify, announce extraction, compound split, prompt-block builders).
* :mod:`.router_backend` — :class:`RouterBackendProtocol` structural interface.
* :mod:`.router_backends` — concrete backend implementations.

All previously public symbols are re-exported here for backward compatibility.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_ENTITY_ID,
    CONF_ROUTER_CUSTOM_PROMPT,
    CONF_ROUTER_FAILOVER_MODE,
    CONF_ROUTER_FALLBACK,
    CONF_ROUTER_LOG_LEVEL,
    CONF_STRATEGY_MAP,
    CONF_TIMEOUT,
    DEFAULT_ROUTER_COMPLEXITY,
    DEFAULT_ROUTER_CUSTOM_PROMPT,
    DEFAULT_ROUTER_FAILOVER_MODE,
    DEFAULT_ROUTER_FALLBACK,
    DEFAULT_ROUTER_LOG_LEVEL,
    DEFAULT_ROUTER_TIMEOUT,
    ROUTER_CLASSIFICATION_PROMPT,
    ROUTER_FAILOVER_BACKUP,
    ROUTER_FAILOVER_PRIMARY_ONLY,
    ROUTER_FALLBACK_BLOCK,
    ROUTER_FALLBACK_SKIP_ROUTING,
    ROUTER_LOG_LEVEL_COMPLEXITY,
    ROUTER_LOG_LEVEL_DEBUG,
    ROUTER_LOG_LEVEL_DEBUG_QUERY,
    ROUTER_SKIP_ROUTING_COMPLEXITY,
    ROUTING_STRATEGY_DEFAULT,
    SIGNAL_STATS_UPDATED,
)
from .router_backend import RouterBackendProtocol as RouterBackendProtocol  # noqa: PLC0414
from .router_backends import IntegratedRouterBackend, OllamaRouterBackend

# Re-export RouterDecision and JSON helpers from their canonical module
from .router_decision import (
    RouterDecision as RouterDecision,  # noqa: PLC0414
)
from .router_decision import (
    _apply_router_decision as _apply_router_decision,  # noqa: PLC0414
)
from .router_decision import (
    _parse_router_response as _parse_router_response,  # noqa: PLC0414
)

# Re-export pure utility functions from their canonical module
from .router_utils import (
    _ANNOUNCE_PREVIEW_MAX_LEN as _ANNOUNCE_PREVIEW_MAX_LEN,  # noqa: PLC0414
)
from .router_utils import (
    _ANNOUNCE_RE as _ANNOUNCE_RE,  # noqa: PLC0414
)
from .router_utils import (
    _COMPOUND_RE as _COMPOUND_RE,  # noqa: PLC0414
)
from .router_utils import (
    _REFUSAL_MAX_LEN as _REFUSAL_MAX_LEN,  # noqa: PLC0414
)
from .router_utils import (
    _REFUSAL_PATTERNS as _REFUSAL_PATTERNS,  # noqa: PLC0414
)
from .router_utils import (
    _WEB_SEARCH_DETERMINISTIC_RE as _WEB_SEARCH_DETERMINISTIC_RE,  # noqa: PLC0414
)
from .router_utils import (
    _build_capability_block as _build_capability_block,  # noqa: PLC0414
)
from .router_utils import (
    _build_strategy_block as _build_strategy_block,  # noqa: PLC0414
)
from .router_utils import (
    _deterministic_classify as _deterministic_classify,  # noqa: PLC0414
)
from .router_utils import (
    _extract_announce_text as _extract_announce_text,  # noqa: PLC0414
)
from .router_utils import (
    _is_unhelpful_response as _is_unhelpful_response,  # noqa: PLC0414
)
from .router_utils import (
    _split_compound_input as _split_compound_input,  # noqa: PLC0414
)

if TYPE_CHECKING:
    from .agent_benchmark import AgentBenchmarker
    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class RouterEngineProtocol(Protocol):
    """Structural interface for the router engine subsystem.

    Defines the surface used by :class:`~.conversation.NeuralBridgeAgent`
    to perform request classification.  Concrete implementation:
    :class:`RouterEngine`.
    """

    async def _check_with_routers(
        self,
        user_input: Any,
        router_agents: list[dict[str, Any]],
        processing_agents: list[dict[str, Any]] | None = None,
    ) -> "RouterDecision | None":
        """Run all router agents and return the last classification."""
        ...

    async def _classify_with_router(
        self,
        router_config: dict[str, Any],
        user_text: str,
        area_context: str | None = None,
        language: str | None = None,
        agent_types: list[str] | None = None,
        processing_agents: list[dict[str, Any]] | None = None,
        benchmarker: "AgentBenchmarker | None" = None,
    ) -> "RouterDecision | None":
        """Ask a single router agent to classify user text."""
        ...


# ---------------------------------------------------------------------------
# Concrete engine
# ---------------------------------------------------------------------------


class RouterEngine:
    """Orchestrates request classification for NeuralBridgeAgent.

    Responsibility: coordinate router agents, apply fallback/stats dispatch,
    and delegate transport to :class:`OllamaRouterBackend` or
    :class:`IntegratedRouterBackend`.

    Access to agent state is via ``self._agent``; within-class method calls
    stay as ``self.<method>()``.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this engine.
        """
        self._agent = agent
        # Share the agent's client cache so tests can inspect / seed it directly.
        self._ollama_backend = OllamaRouterBackend(agent._ollama_clients)
        self._integrated_backend = IntegratedRouterBackend(agent.hass)

    async def _check_with_routers(
        self,
        user_input: Any,
        router_agents: list[dict[str, Any]],
        processing_agents: list[dict[str, Any]] | None = None,
    ) -> RouterDecision | None:
        """Run configured router agents using the configured failover strategy.

        The failover strategy is read from ``CONF_ROUTER_FAILOVER_MODE`` on the
        primary (first) routing agent:

        * ``primary_only`` (default) — only the primary router runs.  On error
          its own ``CONF_ROUTER_FALLBACK`` setting applies.
        * ``backup`` — primary runs first; if it gives a fallback decision (error
          / timeout) the second-priority router is tried automatically.  Explicit
          blocks from the primary are never overridden.
        * ``priority_order`` — all routers run in priority order; stops at the
          first authentic (non-fallback) decision.

        If every router errors the pipeline fails open so a broken router never
        silences the assistant.

        Args:
            user_input:        The user's conversation input.
            router_agents:     List of router agent configurations.
            processing_agents: Optional list of processing agent configs used
                               to construct the agent-type manifest (14c).

        Returns:
            A RouterDecision if the request should proceed (with optional hints),
            or None to block the request.
        """
        default_decision: RouterDecision = RouterDecision(
            local_ha=False, complexity=DEFAULT_ROUTER_COMPLEXITY, is_fallback=True
        )
        if not router_agents:
            return default_decision

        area_context = self._agent._llm_proxy.get_device_area(user_input.device_id)
        agent_types = self._build_agent_types(processing_agents)
        benchmarker = self._agent._get_benchmarker()

        async def _classify(router_config: dict[str, Any]) -> RouterDecision | None:
            _LOGGER.debug("Checking with router: %s", router_config.get(CONF_AGENT_NAME))
            return await self._agent._classify_with_router(
                router_config,
                user_input.text,
                area_context,
                user_input.language,
                agent_types,
                processing_agents=processing_agents,
                benchmarker=benchmarker,
            )

        failover_mode: str = router_agents[0].get(
            CONF_ROUTER_FAILOVER_MODE, DEFAULT_ROUTER_FAILOVER_MODE
        )
        if failover_mode == ROUTER_FAILOVER_PRIMARY_ONLY:
            return await self._run_router_primary_only(_classify, router_agents[0])
        if failover_mode == ROUTER_FAILOVER_BACKUP:
            return await self._run_router_with_backup(_classify, router_agents, default_decision)
        return await self._run_routers_priority_order(_classify, router_agents, default_decision)

    @staticmethod
    def _build_agent_types(
        processing_agents: list[dict[str, Any]] | None,
    ) -> list[str] | None:
        """Build deduplicated ordered agent-type list from processing agents (Feature 14c).

        Args:
            processing_agents: List of processing agent config dicts.

        Returns:
            Ordered list of unique agent type strings, or None if empty.
        """
        if not processing_agents:
            return None
        seen: set[str] = set()
        types_list: list[str] = []
        for agent in processing_agents:
            agent_type = agent.get(CONF_AGENT_TYPE, "")
            if agent_type and agent_type not in seen:
                seen.add(agent_type)
                types_list.append(agent_type)
        return types_list or None

    @staticmethod
    async def _run_router_primary_only(
        classify: Callable[[dict[str, Any]], Awaitable[RouterDecision | None]],
        primary: dict[str, Any],
    ) -> RouterDecision | None:
        """Run only the primary router and return its decision.

        Args:
            classify: Async callable that runs a single router config.
            primary:  The primary router agent configuration dict.

        Returns:
            RouterDecision from the primary router, or None to block.
        """
        decision = await classify(primary)
        if decision is None:
            _LOGGER.info("Router agent '%s' blocked the request", primary.get(CONF_AGENT_NAME))
        return decision

    @staticmethod
    async def _run_router_with_backup(
        classify: Callable[[dict[str, Any]], Awaitable[RouterDecision | None]],
        router_agents: list[dict[str, Any]],
        default_decision: RouterDecision,  # noqa: ARG004
    ) -> RouterDecision | None:
        """Run the primary router; fall through to the backup on error fallback.

        The second entry in *router_agents* (by priority order) is used as the
        backup.  Explicit blocks (``None``) from either router are never
        overridden — only technical errors (``is_fallback=True``) trigger
        the backup path.

        Args:
            classify:        Async callable that runs a single router config.
            router_agents:   All router agent configuration dicts (primary first).
            default_decision: Unused sentinel kept for a consistent signature.

        Returns:
            RouterDecision from the primary or backup, or None to block.
        """
        primary = router_agents[0]
        decision = await classify(primary)
        if decision is None:
            _LOGGER.info("Router agent '%s' blocked the request", primary.get(CONF_AGENT_NAME))
            return None
        if decision.is_fallback and len(router_agents) > 1:
            backup = router_agents[1]
            _LOGGER.info(
                "Primary router '%s' failed — trying backup '%s'",
                primary.get(CONF_AGENT_NAME),
                backup.get(CONF_AGENT_NAME),
            )
            backup_decision = await classify(backup)
            if backup_decision is None:
                _LOGGER.info("Backup router '%s' blocked the request", backup.get(CONF_AGENT_NAME))
                return None
            return backup_decision
        return decision

    @staticmethod
    async def _run_routers_priority_order(
        classify: Callable[[dict[str, Any]], Awaitable[RouterDecision | None]],
        router_agents: list[dict[str, Any]],
        default_decision: RouterDecision,
    ) -> RouterDecision | None:
        """Run all routers in order; stop at the first authentic decision.

        An authentic decision is one where ``is_fallback=False`` — i.e. the router
        successfully classified the input rather than returning an error fallback.
        If every router errors the *default_decision* (fail-open) is returned.

        Args:
            classify:         Async callable that runs a single router config.
            router_agents:    All router agent configuration dicts in priority order.
            default_decision: Returned when every router fails (fail-open).

        Returns:
            RouterDecision from the first successful router, or None to block.
        """
        last_decision: RouterDecision = default_decision
        for router_config in router_agents:
            decision = await classify(router_config)
            if decision is None:
                _LOGGER.info(
                    "Router agent '%s' blocked the request",
                    router_config.get(CONF_AGENT_NAME),
                )
                return None
            last_decision = decision
            if not decision.is_fallback:
                break  # authentic classification — skip remaining routers
        return last_decision

    async def _classify_with_router(
        self,
        router_config: dict[str, Any],
        user_text: str,
        area_context: str | None = None,
        language: str | None = None,
        agent_types: list[str] | None = None,
        processing_agents: list[dict[str, Any]] | None = None,
        benchmarker: "AgentBenchmarker | None" = None,
    ) -> RouterDecision | None:
        """Ask a router agent to classify user text via a JSON prompt.

        Supports both Ollama-based and Home Assistant conversation agent routers.
        The classification prompt is sent to the agent and the JSON response is
        parsed into a :class:`RouterDecision`.

        Logging verbosity is controlled by ``CONF_ROUTER_LOG_LEVEL``:

        * ``none`` — only warnings and errors.
        * ``complexity_only`` — also log the complexity score.
        * ``debug_info`` — log the full routing decision.
        * ``debug_with_query`` — additionally log the query text (logs PII).

        Failure modes (wrong agent type, missing config, network error, empty or
        unparseable response) are handled according to ``CONF_ROUTER_FALLBACK``.

        Only a ``complexity == 0`` JSON response is treated as an explicit block
        signal, regardless of the fallback setting.

        Args:
            router_config:     Configuration dict for the router agent.
            user_text:         The raw user input text to classify.
            area_context:      Friendly area name of the originating device.
            language:          BCP-47 language tag from the HA pipeline.
            agent_types:       Optional ordered list of agent-type identifiers
                               for the compact manifest hint (Feature 14c).
            processing_agents: Optional list of processing agent configs for
                               capability block (Feature 14d).
            benchmarker:       Optional :class:`AgentBenchmarker` for
                               capability block data (Feature 14d).

        Returns:
            A RouterDecision on success or fallback, None to block the request.
        """
        agent_id: str = router_config.get("id", "")
        agent_name: str = router_config.get(CONF_AGENT_NAME, "Unknown")
        log_level: str = router_config.get(CONF_ROUTER_LOG_LEVEL, DEFAULT_ROUTER_LOG_LEVEL)
        fallback: str = router_config.get(CONF_ROUTER_FALLBACK, DEFAULT_ROUTER_FALLBACK)

        self._agent._statistics.record_request(agent_id, agent_name)

        custom_prompt: str = (
            router_config.get(CONF_ROUTER_CUSTOM_PROMPT, DEFAULT_ROUTER_CUSTOM_PROMPT) or ""
        ).strip()
        prompt_template = custom_prompt if custom_prompt else ROUTER_CLASSIFICATION_PROMPT
        prompt = prompt_template.replace("{user_text}", user_text)

        entity_context = self._agent._entity_context_cache.get_summary(self._agent.hass)
        if entity_context:
            prompt = f"{prompt}\n\n{entity_context}"
        sensor_names = self._agent._entity_context_cache.get_sensor_names(self._agent.hass)
        if sensor_names:
            prompt = f"{prompt}\n\n{sensor_names}"
        if area_context:
            prompt = f"{prompt}\n\nDevice area: {area_context}"
        if language:
            prompt = f"{prompt}\nLanguage: {language}"
        # Feature 14c: append compact agent-type manifest
        if agent_types:
            manifest = ", ".join(agent_types)
            prompt = f"{prompt}\n[Available: {manifest}]"

        # Feature 14d: append agent capability block when benchmark data available
        capability_block = _build_capability_block(processing_agents or [], benchmarker)
        if capability_block:
            prompt = f"{prompt}\n\n{capability_block}"

        # Feature 14d: append dimension routing strategies when any are non-default
        dimension_strategies = {
            d: router_config.get(CONF_STRATEGY_MAP[d], ROUTING_STRATEGY_DEFAULT)
            for d in sorted(CONF_STRATEGY_MAP)
        }
        strategy_block = _build_strategy_block(dimension_strategies)
        if strategy_block:
            prompt = f"{prompt}\n\n{strategy_block}"

        # Feature 15: append routing-relevant preference hints
        pref_hint = self._agent._llm_proxy.build_pref_hint(user_text, for_router=True)
        if pref_hint:
            prompt = f"{prompt}\n\n{pref_hint}"

        if log_level == ROUTER_LOG_LEVEL_DEBUG_QUERY:
            _LOGGER.debug("Router '%s' classifying query: %s", agent_name, user_text)

        start_time = time.monotonic()
        response = await self._agent._call_router_backend(router_config, prompt)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        if not response:
            _LOGGER.warning("Router '%s' returned no response — applying fallback", agent_name)
            return self._record_router_failure_and_fallback(agent_id, fallback)

        parsed = _parse_router_response(response)
        if parsed is None:
            _LOGGER.warning(
                "Router '%s' returned unparseable response — applying fallback", agent_name
            )
            return self._record_router_failure_and_fallback(agent_id, fallback)

        self._agent._statistics.record_success(agent_id, elapsed_ms)
        if parsed.complexity == 0:
            self._agent._statistics.record_block(agent_id)
            self._dispatch_stats_updated()
            return None

        # Record intent_hint in stats if the router provided one (Feature 8)
        if parsed.intent_hint:
            self._agent._statistics.record_intent_hint(agent_id, parsed.intent_hint)

        self._log_router_decision(log_level, agent_name, parsed)
        self._dispatch_stats_updated()
        return parsed

    def _dispatch_stats_updated(self) -> None:
        """Fire the stats-updated dispatcher signal for this config entry."""
        async_dispatcher_send(
            self._agent.hass,
            SIGNAL_STATS_UPDATED.format(entry_id=self._agent._config_entry.entry_id),
        )

    def _record_router_failure_and_fallback(
        self, agent_id: str, fallback: str
    ) -> RouterDecision | None:
        """Record a routing failure, fire the stats signal, and return the fallback.

        Args:
            agent_id: The unique ID of the router agent that failed.
            fallback: One of the ``ROUTER_FALLBACK_*`` constants.

        Returns:
            A :class:`RouterDecision` for fail-open / skip-routing fallbacks, or
            ``None`` when ``fallback == ROUTER_FALLBACK_BLOCK``.
        """
        self._agent._statistics.record_failure(agent_id)
        self._dispatch_stats_updated()
        return self._router_fallback(fallback)

    def _log_router_decision(self, log_level: str, agent_name: str, parsed: RouterDecision) -> None:
        """Emit a routing-decision log at the configured verbosity level.

        Args:
            log_level:  One of the ``ROUTER_LOG_LEVEL_*`` constants.
            agent_name: Display name of the router agent.
            parsed:     The :class:`RouterDecision` returned by the router.
        """
        if log_level == ROUTER_LOG_LEVEL_COMPLEXITY:
            _LOGGER.info("Router '%s' complexity score: %d", agent_name, parsed.complexity)
        elif log_level in (ROUTER_LOG_LEVEL_DEBUG, ROUTER_LOG_LEVEL_DEBUG_QUERY):
            _LOGGER.debug(
                "Router '%s' decision: local_ha=%s complexity=%d",
                agent_name,
                parsed.local_ha,
                parsed.complexity,
            )

    async def _call_router_backend(self, router_config: dict[str, Any], prompt: str) -> str | None:
        """Invoke the appropriate router back-end and return the raw response text.

        Delegates to :class:`OllamaRouterBackend` for ``AGENT_TYPE_OLLAMA``
        configs, or to :class:`IntegratedRouterBackend` for
        ``AGENT_TYPE_INTEGRATED`` / ``AGENT_TYPE_LOCAL_HA`` configs.

        Args:
            router_config: Configuration dict for the router agent.
            prompt:        The fully-formatted classification prompt.

        Returns:
            The raw text response from the router back-end, or None on error.
        """
        agent_type: str = router_config.get(CONF_AGENT_TYPE, "")
        agent_name: str = router_config.get(CONF_AGENT_NAME, "Unknown")

        if agent_type == AGENT_TYPE_OLLAMA:
            return await self._ollama_backend.call(router_config, prompt)

        if agent_type in (AGENT_TYPE_INTEGRATED, AGENT_TYPE_LOCAL_HA):
            # Route through the agent's _call_existing_agent_for_routing so tests
            # can patch that method on NeuralBridgeAgent as a clean seam.
            entity_id: str = router_config.get(CONF_ENTITY_ID, "")
            timeout: int = router_config.get(CONF_TIMEOUT, DEFAULT_ROUTER_TIMEOUT)
            return await self._agent._call_existing_agent_for_routing(entity_id, prompt, timeout)

        _LOGGER.debug(
            "Router '%s' has unknown type '%s' — applying fallback", agent_name, agent_type
        )
        return None

    def _router_fallback(self, fallback: str) -> RouterDecision | None:
        """Return the fallback RouterDecision (or None to block) based on the config.

        Decisions returned here are tagged with ``is_fallback=True`` so that
        the failover engine in :meth:`_check_with_routers` can distinguish
        error-fallback decisions from authentic classifications and attempt
        a backup or next-priority router when configured.

        Args:
            fallback: One of the ``ROUTER_FALLBACK_*`` constants.

        Returns:
            A RouterDecision (``is_fallback=True``) for fail-open or
            skip-routing fallbacks, or ``None`` when
            ``fallback == ROUTER_FALLBACK_BLOCK``.
        """
        if fallback == ROUTER_FALLBACK_SKIP_ROUTING:
            return RouterDecision(
                local_ha=False, complexity=ROUTER_SKIP_ROUTING_COMPLEXITY, is_fallback=True
            )
        if fallback == ROUTER_FALLBACK_BLOCK:
            return None
        return RouterDecision(
            local_ha=False, complexity=DEFAULT_ROUTER_COMPLEXITY, is_fallback=True
        )

    async def _call_existing_agent_for_routing(
        self, entity_id: str, prompt: str, timeout: int
    ) -> str | None:
        """Backward-compat wrapper — delegate to :class:`IntegratedRouterBackend`.

        Kept for :class:`~.conversation.NeuralBridgeAgent` delegation and tests
        that call this method directly on the agent.

        Args:
            entity_id: The conversation entity to call.
            prompt:    The fully-formatted classification prompt text.
            timeout:   Maximum seconds to wait for a response.

        Returns:
            The response speech text, or None on error/timeout.
        """
        # Create a fresh backend bound to the *current* hass so that tests
        # replacing conv_agent.hass after construction are handled correctly.
        config = {
            CONF_ENTITY_ID: entity_id,
            CONF_TIMEOUT: timeout,
            CONF_AGENT_NAME: entity_id,
        }
        return await IntegratedRouterBackend(self._agent.hass).call(config, prompt)
