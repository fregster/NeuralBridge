"""Conversation agent for NeuralBridge."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.helpers import intent

from .circuit_breaker import CircuitBreaker
from .confirmation_flows import ConfirmationFlows
from .const import (
    CONF_ENABLE_HOME_CONTROL,
    CONF_HIGH_STAKES_SECRET,
    CONF_LANGUAGE,
    CONF_OLLAMA_URL,
    CONF_ROUTER_CUSTOM_PROMPT,
    CONF_SEARCH_API_KEY,
    DATA_BENCHMARKER,
    DATA_CIRCUIT_BREAKER,
    DATA_ENTITY_CONTEXT,
    DATA_PREFERENCE_MEMORY,
    DATA_RESPONSE_CACHE,
    DATA_SESSION_MEMORY,
    DATA_STATISTICS,
    DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_ENABLE_HOME_CONTROL,
    DEFAULT_LANGUAGE,
    DEFAULT_RESPONSE_CACHE_ENABLED,
    DEFAULT_RESPONSE_CACHE_SEMANTIC,
    DEFAULT_RESPONSE_CACHE_TTL,
    DEFAULT_SEMANTIC_CACHE_TTL,
    DOMAIN,
)
from .entity_context import EntityContextCache
from .guard_rail import GuardRailCache, GuardRailChecker, HighStakesCache
from .languages_loader import get_string
from .llm_agent_proxy import LLMAgentProxy
from .pipeline_executor import PipelineExecutor
from .response_cache import ResponseCache
from .router_engine import (
    RouterDecision as RouterDecision,  # noqa: PLC0414
)
from .router_engine import (
    RouterEngine,
)
from .router_engine import (
    _apply_router_decision as _apply_router_decision,  # noqa: PLC0414
)
from .router_engine import (
    _build_capability_block as _build_capability_block,  # noqa: PLC0414
)
from .router_engine import (
    _build_strategy_block as _build_strategy_block,  # noqa: PLC0414
)
from .router_engine import (
    _deterministic_classify as _deterministic_classify,  # noqa: PLC0414
)
from .router_engine import (
    _extract_announce_text as _extract_announce_text,  # noqa: PLC0414
)
from .router_engine import (
    _is_unhelpful_response as _is_unhelpful_response,  # noqa: PLC0414
)
from .router_engine import (
    _parse_router_response as _parse_router_response,  # noqa: PLC0414
)
from .router_engine import (
    _split_compound_input as _split_compound_input,  # noqa: PLC0414
)
from .session_memory import SessionMemory
from .statistics import AgentStatistics

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from .agent_benchmark import AgentBenchmarker
    from .ollama_client import OllamaClient
    from .preference_analyser import PreferenceSuggestion
    from .preference_memory import PreferenceMemory
    from .web_search_client import WebSearchClient

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging safety helper
# ---------------------------------------------------------------------------

# Fields whose values must never appear in log output.
_REDACT_FIELDS: frozenset[str] = frozenset(
    {
        CONF_SEARCH_API_KEY,
        CONF_HIGH_STAKES_SECRET,
        CONF_OLLAMA_URL,  # may embed credentials as URL auth tokens
        CONF_ROUTER_CUSTOM_PROMPT,  # may contain user-defined PII
    }
)


def _safe_log_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *config* with sensitive fields replaced by '<redacted>'.

    Use this helper instead of logging a raw ``agent_config`` or
    ``router_config`` dict directly.  The helper is safe to call with any
    dict — unexpected keys are passed through unchanged.

    Args:
        config: The configuration mapping to sanitise.

    Returns:
        A new dict identical to *config* except that any key in
        :data:`_REDACT_FIELDS` has its value replaced by the string
        ``'<redacted>'``.  The original *config* is not modified.
    """
    return {k: "<redacted>" if k in _REDACT_FIELDS else v for k, v in config.items()}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ConversationAgentProtocol(Protocol):
    """Public interface for the NeuralBridge conversation agent.

    Defines the minimal surface that Home Assistant requires of any
    ConversationEntity.  Sub-engines accept ``NeuralBridgeAgent`` directly;
    this Protocol exists for documentation and static-analysis purposes.
    """

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process user input and return a conversation result."""
        ...

    @property
    def supported_languages(self) -> Literal["*"]:
        """Return supported languages."""
        ...

    @property
    def supported_features(self) -> ConversationEntityFeature:
        """Return supported features."""
        ...


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
        self._web_search_clients: dict[str, WebSearchClient] = {}
        self._guard_rail_checker: GuardRailChecker | None = None
        self._guard_rail_rules_snapshot: dict[str, list[str]] | None = None
        self._guard_rail_cache = GuardRailCache()
        self._high_stakes_cache = HighStakesCache()
        # Shared objects created by __init__.py and stored in hass.data
        entry_data = hass.data.get(DOMAIN, {}).get(config_entry.entry_id, {})
        self._circuit_breaker: CircuitBreaker = entry_data.get(
            DATA_CIRCUIT_BREAKER,
            CircuitBreaker(
                failure_threshold=DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
                cooldown_seconds=DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
            ),
        )
        self._statistics: AgentStatistics = entry_data.get(DATA_STATISTICS, AgentStatistics())
        self._response_cache: ResponseCache = entry_data.get(
            DATA_RESPONSE_CACHE,
            ResponseCache(
                enabled=DEFAULT_RESPONSE_CACHE_ENABLED,
                ttl_seconds=DEFAULT_RESPONSE_CACHE_TTL,
                semantic=DEFAULT_RESPONSE_CACHE_SEMANTIC,
                semantic_ttl_seconds=DEFAULT_SEMANTIC_CACHE_TTL,
            ),
        )
        self._session_memory: SessionMemory = entry_data.get(DATA_SESSION_MEMORY, SessionMemory())
        self._entity_context_cache: EntityContextCache = entry_data.get(
            DATA_ENTITY_CONTEXT, EntityContextCache()
        )
        # Feature 15 — Adaptive Preference Learning
        self._preference_memory: PreferenceMemory | None = entry_data.get(DATA_PREFERENCE_MEMORY)
        self._pending_preference_suggestions: dict[str, PreferenceSuggestion] = {}
        # Unified LLM execution proxy — handles INTEGRATED and OLLAMA backends with
        # shared prompt-enrichment (verbosity, preferences, sensor context, language).
        self._llm_proxy = LLMAgentProxy(
            hass=hass,
            config_getter=self._get_config,
            entity_context_cache=self._entity_context_cache,
            session_memory=self._session_memory,
            preference_memory=self._preference_memory,
            ollama_clients=self._ollama_clients,
            benchmarker_getter=self._get_benchmarker,
        )
        # Sub-engines: instantiated after all shared state is ready
        self._router = RouterEngine(self)
        self._pipeline = PipelineExecutor(self)
        self._confirmation = ConfirmationFlows(self)

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

    @property
    def supported_features(self) -> ConversationEntityFeature:
        """Return supported features.

        Declares home-control capability when the global 'Enable Home Control'
        toggle is on, causing HA to show 'This assistant can control your home'.
        """
        if self._get_config().get(CONF_ENABLE_HOME_CONTROL, DEFAULT_ENABLE_HOME_CONTROL):
            return ConversationEntityFeature.CONTROL
        return ConversationEntityFeature(0)

    def _get_config(self) -> dict[str, Any]:
        """Return a merged view of data + options from the config entry.

        Options override data so live changes (options flow) take effect
        without a restart.

        Returns:
            Merged configuration dictionary.
        """
        merged: dict[str, Any] = {**self._config_entry.data, **self._config_entry.options}
        return merged

    def _get_benchmarker(self) -> "AgentBenchmarker | None":
        """Return the shared AgentBenchmarker, or None when not yet set up.

        Returns:
            The :class:`AgentBenchmarker` stored in ``hass.data`` for this
            config entry, or ``None`` when the benchmarker has not been
            initialised (e.g. first boot before setup completes).
        """
        entry_data = self.hass.data.get(DOMAIN, {}).get(self._config_entry.entry_id, {})
        return entry_data.get(DATA_BENCHMARKER)  # type: ignore[no-any-return]

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a user input through the priority-based routing system."""
        _LOGGER.debug("Processing input: %d chars", len(user_input.text))
        result = await self._pipeline._compute_result(user_input)
        result = await self._confirmation._analyse_and_suggest(user_input, result)
        self._maybe_add_to_chat_log(result)
        return result

    # ------------------------------------------------------------------
    # Delegation methods — retained because tests call or patch these
    # directly on the NeuralBridgeAgent instance.  Each simply forwards
    # to the responsible sub-engine.
    # ------------------------------------------------------------------

    async def _handle_confirmation_check(
        self, user_input: ConversationInput
    ) -> ConversationResult | None:
        """Delegate to ConfirmationFlows."""
        return await self._confirmation._handle_confirmation_check(user_input)

    async def _resolve_high_stakes_confirmation(
        self, user_input: ConversationInput, hs_pending: tuple[ConversationResult, list[str]]
    ) -> ConversationResult | None:
        """Delegate to ConfirmationFlows."""
        return await self._confirmation._resolve_high_stakes_confirmation(user_input, hs_pending)

    async def _resolve_preference_confirmation(
        self, user_input: ConversationInput
    ) -> ConversationResult | None:
        """Delegate to ConfirmationFlows."""
        return await self._confirmation._resolve_preference_confirmation(user_input)

    def _fire_preferences_updated(self) -> None:
        """Delegate to ConfirmationFlows."""
        self._confirmation._fire_preferences_updated()

    async def _analyse_and_suggest(
        self, user_input: ConversationInput, result: ConversationResult
    ) -> ConversationResult:
        """Delegate to ConfirmationFlows."""
        return await self._confirmation._analyse_and_suggest(user_input, result)

    async def _apply_guard_rail_action(
        self,
        action: str,
        result: ConversationResult,
        response_text: str,
        conversation_id: str | None,
        guard_rail_result: Any,
    ) -> ConversationResult | None:
        """Delegate to ConfirmationFlows."""
        return await self._confirmation._apply_guard_rail_action(
            action, result, response_text, conversation_id, guard_rail_result
        )

    async def _check_high_stakes(
        self,
        agent_config: dict[str, Any],
        result: ConversationResult,
        user_input: ConversationInput,
    ) -> ConversationResult | None:
        """Delegate to ConfirmationFlows."""
        return await self._confirmation._check_high_stakes(agent_config, result, user_input)

    async def _get_guard_rail_agent_config(self) -> dict[str, Any] | None:
        """Delegate to ConfirmationFlows."""
        return await self._confirmation._get_guard_rail_agent_config()

    async def _compute_result(self, user_input: ConversationInput) -> ConversationResult:
        """Delegate to PipelineExecutor."""
        return await self._pipeline._compute_result(user_input)

    def _maybe_add_to_chat_log(self, result: ConversationResult) -> None:
        """Delegate to PipelineExecutor."""
        self._pipeline._maybe_add_to_chat_log(result)

    async def _process_compound_fragments(
        self,
        fragments: list[str],
        processing_agents: list[dict[str, Any]],
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Delegate to PipelineExecutor."""
        return await self._pipeline._process_compound_fragments(
            fragments, processing_agents, user_input
        )

    async def _try_agent_with_retries(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        agent_id: str,
        agent_name: str,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult | None:
        """Delegate to PipelineExecutor."""
        return await self._pipeline._try_agent_with_retries(
            agent_config, user_input, agent_id, agent_name, router_decision
        )

    async def _handle_successful_result(
        self,
        agent_config: dict[str, Any],
        result: ConversationResult,
        user_input: ConversationInput,
        agent_id: str,
        elapsed_ms: float,
    ) -> ConversationResult:
        """Delegate to PipelineExecutor."""
        return await self._pipeline._handle_successful_result(
            agent_config, result, user_input, agent_id, elapsed_ms
        )

    def _extract_response_text(self, result: ConversationResult) -> str:
        """Delegate to PipelineExecutor."""
        return self._pipeline._extract_response_text(result)

    def _maybe_cache_response(
        self, agent_config: dict[str, Any], input_text: str, response_text: str, agent_name: str
    ) -> None:
        """Delegate to PipelineExecutor."""
        self._pipeline._maybe_cache_response(agent_config, input_text, response_text, agent_name)

    def _record_agent_failure(self, agent_id: str, timed_out: bool) -> None:
        """Delegate to PipelineExecutor."""
        self._pipeline._record_agent_failure(agent_id, timed_out)

    def _check_explicit_agent_override(
        self, user_input: ConversationInput, agents: list[dict[str, Any]]
    ) -> tuple[ConversationInput, list[dict[str, Any]]] | None:
        """Delegate to PipelineExecutor."""
        return self._pipeline._check_explicit_agent_override(user_input, agents)

    async def _try_processing_agents(
        self,
        processing_agents: list[dict[str, Any]],
        user_input: ConversationInput,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult:
        """Delegate to PipelineExecutor."""
        return await self._pipeline._try_processing_agents(
            processing_agents, user_input, router_decision
        )

    async def _try_agent_with_tracking(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult | None:
        """Delegate to PipelineExecutor."""
        return await self._pipeline._try_agent_with_tracking(
            agent_config, user_input, router_decision
        )

    async def _check_with_routers(
        self,
        user_input: ConversationInput,
        router_agents: list[dict[str, Any]],
        processing_agents: list[dict[str, Any]] | None = None,
    ) -> RouterDecision | None:
        """Delegate to RouterEngine."""
        return await self._router._check_with_routers(user_input, router_agents, processing_agents)

    async def _classify_with_router(
        self,
        router_config: dict[str, Any],
        user_text: str,
        area_context: str | None = None,
        language: str | None = None,
        agent_types: list[str] | None = None,
        processing_agents: list[dict[str, Any]] | None = None,
        benchmarker: Any = None,
    ) -> RouterDecision | None:
        """Delegate to RouterEngine."""
        return await self._router._classify_with_router(
            router_config,
            user_text,
            area_context,
            language,
            agent_types,
            processing_agents,
            benchmarker,
        )

    async def _call_router_backend(self, router_config: dict[str, Any], prompt: str) -> str | None:
        """Delegate to RouterEngine."""
        return await self._router._call_router_backend(router_config, prompt)

    async def _call_existing_agent_for_routing(
        self, entity_id: str, prompt: str, timeout: int
    ) -> str | None:
        """Delegate to RouterEngine."""
        return await self._router._call_existing_agent_for_routing(entity_id, prompt, timeout)

    def _router_fallback(self, fallback: str) -> RouterDecision | None:
        """Delegate to RouterEngine."""
        return self._router._router_fallback(fallback)

    async def _try_agent(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: RouterDecision | None = None,
    ) -> tuple[ConversationResult | None, bool]:
        """Delegate to PipelineExecutor."""
        return await self._pipeline._try_agent(agent_config, user_input, router_decision)

    async def _process_with_web_search(
        self, agent_config: dict[str, Any], user_input: ConversationInput
    ) -> ConversationResult | None:
        """Delegate to PipelineExecutor."""
        return await self._pipeline._process_with_web_search(agent_config, user_input)

    async def _check_guardrails(
        self, agent_config: dict[str, Any], result: ConversationResult, conversation_id: str | None
    ) -> ConversationResult | None:
        """Delegate to ConfirmationFlows."""
        return await self._confirmation._check_guardrails(agent_config, result, conversation_id)

    def _find_tts_entity(self) -> str | None:
        """Delegate to ConfirmationFlows."""
        return self._confirmation._find_tts_entity()

    async def _send_broadcast_announcement(
        self, text: str, media_players: list[str], user_input: ConversationInput
    ) -> ConversationResult:
        """Delegate to ConfirmationFlows."""
        return await self._confirmation._send_broadcast_announcement(
            text, media_players, user_input
        )

    # ------------------------------------------------------------------
    # Utility helpers — used by sub-engines and tests.
    # ------------------------------------------------------------------

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
        for web_client in self._web_search_clients.values():
            await web_client.close()
        self._web_search_clients.clear()
