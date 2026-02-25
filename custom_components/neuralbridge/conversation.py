"""Conversation agent for NeuralBridge."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .circuit_breaker import CircuitBreaker
from .const import (
    AGENT_TYPE_EXISTING,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    AGENT_TYPE_WEB_SEARCH,
    CONF_AGENT_ASSIST_MODE,
    CONF_AGENT_CACHE_ENABLED,
    CONF_AGENT_ENABLED,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_AGENTS,
    CONF_DEFAULT_PROMPT,
    CONF_ENABLE_HOME_CONTROL,
    CONF_ENTITY_ID,
    CONF_FORCE_RESPONSE_LANGUAGE,
    CONF_GUARD_RAIL_ACTION,
    CONF_GUARD_RAIL_AI_THRESHOLD,
    CONF_GUARD_RAIL_DETOXIFY_THRESHOLD,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_GUARD_RAIL_RULES,
    CONF_GUARD_RAIL_USE_DETOXIFY,
    CONF_IS_ROUTER,
    CONF_LANGUAGE,
    CONF_MAX_RETRIES,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_PRIORITY,
    CONF_RETRY_BASE_DELAY,
    CONF_ROUTER_CUSTOM_PROMPT,
    CONF_ROUTER_FALLBACK,
    CONF_ROUTER_LOG_LEVEL,
    CONF_SEARCH_API_KEY,
    CONF_SEARCH_MAX_SNIPPET_LEN,
    CONF_SEARCH_PROVIDER,
    CONF_SEARCH_RESULT_COUNT,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DATA_CIRCUIT_BREAKER,
    DATA_ENTITY_CONTEXT,
    DATA_RESPONSE_CACHE,
    DATA_SESSION_MEMORY,
    DATA_STATISTICS,
    DEFAULT_AGENT_ASSIST_MODE,
    DEFAULT_AGENT_CACHE_ENABLED,
    DEFAULT_AGENT_ENABLED,
    DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_DEFAULT_PROMPT,
    DEFAULT_ENABLE_HOME_CONTROL,
    DEFAULT_FORCE_RESPONSE_LANGUAGE,
    DEFAULT_GUARD_RAIL_ACTION,
    DEFAULT_GUARD_RAIL_AI_THRESHOLD,
    DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD,
    DEFAULT_GUARD_RAIL_ENABLED,
    DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    DEFAULT_GUARD_RAIL_USE_DETOXIFY,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RESPONSE_CACHE_ENABLED,
    DEFAULT_RESPONSE_CACHE_TTL,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_ROUTER_COMPLEXITY,
    DEFAULT_ROUTER_CUSTOM_PROMPT,
    DEFAULT_ROUTER_FALLBACK,
    DEFAULT_ROUTER_LOG_LEVEL,
    DEFAULT_ROUTER_TIMEOUT,
    DEFAULT_SEARCH_MAX_SNIPPET_LEN,
    DEFAULT_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_TIMEOUT,
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
    ROUTER_FALLBACK_BLOCK,
    ROUTER_FALLBACK_SKIP_ROUTING,
    ROUTER_LOG_LEVEL_COMPLEXITY,
    ROUTER_LOG_LEVEL_DEBUG,
    ROUTER_LOG_LEVEL_DEBUG_QUERY,
    ROUTER_RESPONSE_KEY_COMPLEXITY,
    ROUTER_RESPONSE_KEY_INTENT_HINT,
    ROUTER_RESPONSE_KEY_LOCAL_HA,
    ROUTER_RESPONSE_KEY_WEB_SEARCH,
    ROUTER_SKIP_ROUTING_COMPLEXITY,
    SIGNAL_STATS_UPDATED,
    VALID_INTENT_HINTS,
)
from .entity_context import EntityContextCache
from .guard_rail import GuardRailCache, GuardRailChecker
from .languages_loader import get_string
from .ollama_client import OllamaClient
from .response_cache import ResponseCache
from .session_memory import SessionMemory
from .statistics import AgentStatistics
from .web_search_client import WebSearchClient

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router decision dataclass
# ---------------------------------------------------------------------------


class RouterDecision:
    """Immutable classification result produced by a router agent.

    Attributes:
        local_ha:    True when the request is a home-automation / device-control
                     command that should be handled by a LOCAL_HA agent.
        web_search:  True when the request requires real-time or current data
                     (news, live scores, current leaders, financial data, etc.).
        complexity:  Integer 1-100 indicating estimated request complexity.
                     Higher values suggest cloud-capable agents are preferred.
        intent_hint: Optional routing hint (``"timer"``, ``"reminder"``,
                     ``"todo"``, ``"shopping_list"``, ``"announce"``, or
                     ``None``).  Forces LOCAL_HA routing for certain intents.
    """

    __slots__ = ("complexity", "intent_hint", "local_ha", "web_search")

    # Class-level annotations required for mypy __slots__ attribute resolution
    complexity: int
    intent_hint: str | None
    local_ha: bool
    web_search: bool

    def __init__(
        self,
        local_ha: bool,
        complexity: int,
        web_search: bool = False,
        intent_hint: str | None = None,
    ) -> None:
        """Initialise a RouterDecision.

        Args:
            local_ha:    Whether the request targets local home-automation.
            complexity:  Estimated complexity score (1-100).
            web_search:  Whether the request needs real-time web data.
            intent_hint: Optional intent classification hint (Feature 8).
        """
        object.__setattr__(self, "local_ha", local_ha)
        object.__setattr__(self, "complexity", complexity)
        object.__setattr__(self, "web_search", web_search)
        object.__setattr__(self, "intent_hint", intent_hint)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("RouterDecision is immutable")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RouterDecision):
            return NotImplemented
        return (
            self.local_ha == other.local_ha
            and self.complexity == other.complexity
            and self.web_search == other.web_search
            and self.intent_hint == other.intent_hint
        )

    def __hash__(self) -> int:
        return hash((self.local_ha, self.complexity, self.web_search, self.intent_hint))

    def __repr__(self) -> str:
        return (
            f"RouterDecision(local_ha={self.local_ha!r}, "
            f"complexity={self.complexity!r}, "
            f"web_search={self.web_search!r}, "
            f"intent_hint={self.intent_hint!r})"
        )


# ---------------------------------------------------------------------------
# Module-level routing helpers
# ---------------------------------------------------------------------------

# Mapping of BCP-47 base language codes to human-readable names.
# Used by Feature 10 to inject "Respond in {language}." into Ollama prompts.
_LANG_CODE_TO_NAME: dict[str, str] = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "zh": "Chinese",
}


def _get_language_name(language_code: str) -> str:
    """Return a human-readable language name for the given BCP-47 language code.

    Normalises the code to a 2-letter base (e.g. ``"fr-FR"`` → ``"fr"``) before
    looking up in ``_LANG_CODE_TO_NAME``.  Falls back to the original code when
    the language is not in the map.

    Args:
        language_code: BCP-47 language tag (e.g. ``"en"``, ``"fr-FR"``).

    Returns:
        Human-readable name (e.g. ``"French"``) or the original code if unknown.
    """
    base = language_code.lower().split("-")[0].split("_")[0]
    return _LANG_CODE_TO_NAME.get(base, language_code)


def _parse_router_response(raw: str) -> RouterDecision | None:
    """Parse a JSON router-agent response into a RouterDecision.

    Strips optional Markdown code fences, extracts the first ``{...}`` block,
    then parses it.  Returns ``None`` when:

    * The response cannot be parsed as JSON.
    * ``complexity`` is 0 — the router signal to block the request.

    A ``complexity`` outside 1-100 is clamped to that range (after ruling out 0).

    Args:
        raw: The raw string returned by the Ollama router model.

    Returns:
        A RouterDecision if the response is valid and not a block signal, else None.
    """

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence
        lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        _LOGGER.debug("Router response contains no JSON object")
        return None

    json_text = text[start : end + 1]
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as err:
        _LOGGER.debug("Router response JSON parse error: %s", err)
        return None

    if not isinstance(data, dict):  # pragma: no cover
        _LOGGER.debug("Router response JSON is not an object")  # pragma: no cover
        return None  # pragma: no cover

    raw_complexity = data.get(ROUTER_RESPONSE_KEY_COMPLEXITY)
    if not isinstance(raw_complexity, (int, float)) or ROUTER_RESPONSE_KEY_LOCAL_HA not in data:
        _LOGGER.debug("Router response missing required 'complexity' or 'local_ha' field")
        return None

    complexity = int(raw_complexity)

    # complexity == 0 is the router BLOCK signal; preserved for caller to handle
    if complexity == 0:
        return RouterDecision(local_ha=False, complexity=0, web_search=False)

    # Clamp to 1-100
    complexity = max(1, min(100, complexity))

    local_ha = bool(data[ROUTER_RESPONSE_KEY_LOCAL_HA])
    web_search = bool(data.get(ROUTER_RESPONSE_KEY_WEB_SEARCH, False))

    # Extract optional intent_hint (Feature 8); accept only known values
    raw_hint = data.get(ROUTER_RESPONSE_KEY_INTENT_HINT)
    intent_hint: str | None = str(raw_hint) if raw_hint in VALID_INTENT_HINTS else None

    return RouterDecision(
        local_ha=local_ha, complexity=complexity, web_search=web_search, intent_hint=intent_hint
    )


def _apply_router_decision(
    decision: RouterDecision,
    processing_agents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter and reorder processing agents based on the router decision.

    When ``decision.web_search`` is ``True``, web-search agents are promoted to
    the front of the list.  When ``decision.local_ha`` is ``True``, LOCAL_HA
    agents are promoted.  When neither flag is set, LOCAL_HA agents are excluded
    entirely (request is a general knowledge query; leave web-search agents in
    their normal priority position).

    A ``decision.complexity`` of ``ROUTER_SKIP_ROUTING_COMPLEXITY`` (-1) is a
    special sentinel meaning "skip routing" — all processing agents are returned
    unchanged in their original priority order.

    Args:
        decision:          The RouterDecision produced by the router agent.
        processing_agents: Priority-sorted list of processing agent configs.

    Returns:
        Filtered and/or reordered list of processing agent configs.
    """
    if decision.complexity == ROUTER_SKIP_ROUTING_COMPLEXITY:
        # Fallback=skip_routing — bypass agent filtering, try everything
        return list(processing_agents)
    # Feature 8: intent_hint forces LOCAL_HA for timer/reminder/todo/shopping_list commands
    # regardless of what the router returned for local_ha (extra safety net).
    if decision.intent_hint in ("timer", "reminder", "todo", "shopping_list"):
        local = [a for a in processing_agents if a.get(CONF_AGENT_TYPE) == AGENT_TYPE_LOCAL_HA]
        others = [a for a in processing_agents if a.get(CONF_AGENT_TYPE) != AGENT_TYPE_LOCAL_HA]
        return local + others
    if decision.web_search:
        # Promote WEB_SEARCH agents to front; keep original order within each group
        web = [a for a in processing_agents if a.get(CONF_AGENT_TYPE) == AGENT_TYPE_WEB_SEARCH]
        others = [a for a in processing_agents if a.get(CONF_AGENT_TYPE) != AGENT_TYPE_WEB_SEARCH]
        return web + others
    if decision.local_ha:
        # Promote LOCAL_HA agents to front; keep original order within each group
        local = [a for a in processing_agents if a.get(CONF_AGENT_TYPE) == AGENT_TYPE_LOCAL_HA]
        others = [a for a in processing_agents if a.get(CONF_AGENT_TYPE) != AGENT_TYPE_LOCAL_HA]
        return local + others
    # Exclude LOCAL_HA agents — request is not a home-automation command
    return [a for a in processing_agents if a.get(CONF_AGENT_TYPE) != AGENT_TYPE_LOCAL_HA]


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
            ),
        )
        self._session_memory: SessionMemory = entry_data.get(DATA_SESSION_MEMORY, SessionMemory())
        self._entity_context_cache: EntityContextCache = entry_data.get(
            DATA_ENTITY_CONTEXT, EntityContextCache()
        )

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

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a user input through the priority-based routing system."""
        _LOGGER.debug("Processing input: %d chars", len(user_input.text))
        result = await self._compute_result(user_input)
        self._maybe_add_to_chat_log(result)
        return result

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
        confirmation_result = await self._handle_confirmation_check(user_input)
        if confirmation_result is not None:
            return confirmation_result

        # Step 2: Check response cache
        cached = self._response_cache.get(user_input.text)
        if cached is not None:
            _LOGGER.debug("Cache hit — returning cached response")
            return self._create_result(cached, user_input.conversation_id)

        # Steps 3-5: Load enabled agents, split by role, run router check
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

        # Check for explicit named-agent override ("Ask Gemini: ...", "Use Ollama: ...", etc.)
        override = self._check_explicit_agent_override(user_input, processing_agents)
        if override is not None:
            override_input, override_agents = override
            return await self._try_processing_agents(override_agents, override_input)

        if router_agents:
            decision = await self._check_with_routers(user_input, router_agents)
            if decision is None:
                _LOGGER.debug("Router agents blocked the request")
                return self._create_error_result(
                    self._localized("responses", "router_blocked"),
                    user_input.conversation_id,
                )
            processing_agents = _apply_router_decision(decision, processing_agents)

        # Step 6: Try processing agents in priority order
        return await self._try_processing_agents(processing_agents, user_input)

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

        speech = self._extract_response_text(result)
        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(agent_id=self.entity_id, content=speech or None)
        )

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
        """Try a single agent with circuit breaker guard, then retry logic.

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

        return await self._try_agent_with_retries(agent_config, user_input, agent_id, agent_name)

    async def _try_agent_with_retries(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        agent_id: str,
        agent_name: str,
    ) -> ConversationResult | None:
        """Attempt an agent call with exponential back-off on failure.

        Args:
            agent_config: Configuration dict for the agent to try.
            user_input: The user's conversation input.
            agent_id: Unique identifier for the agent.
            agent_name: Display name of the agent (for logging).

        Returns:
            ConversationResult on success, None if all attempts are exhausted.
        """
        config = self._get_config()
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

            self._statistics.record_request(agent_id, agent_name)
            start_time = time.monotonic()
            result, timed_out = await self._try_agent(agent_config, user_input)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            if result is not None:
                return await self._handle_successful_result(
                    agent_config, result, user_input, agent_id, elapsed_ms
                )

            self._record_agent_failure(agent_id, timed_out)
            if self._circuit_breaker.is_open(agent_id):
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
    ) -> RouterDecision | None:
        """Query router agents for a routing decision.

        Each router agent is asked to classify the input via the JSON prompt.
        The first agent that returns ``None`` (block signal) causes the method
        to return ``None`` immediately, short-circuiting remaining routers.
        Falls back to a default :class:`RouterDecision` if every router errors
        out, so that a broken router never silences the assistant (fail-open).

        Args:
            user_input: The user's conversation input.
            router_agents: List of router agent configurations.

        Returns:
            A RouterDecision if the request should proceed (with optional hints),
            or None to block the request.
        """
        last_decision: RouterDecision | None = RouterDecision(
            local_ha=False, complexity=DEFAULT_ROUTER_COMPLEXITY, web_search=False
        )
        area_context = self._get_device_area(user_input.device_id)
        all_errors = True
        for router_config in router_agents:
            _LOGGER.debug("Checking with router: %s", router_config.get(CONF_AGENT_NAME))
            decision = await self._classify_with_router(
                router_config, user_input.text, area_context, user_input.language
            )
            if decision is None:
                _LOGGER.info(
                    "Router agent '%s' blocked the request",
                    router_config.get(CONF_AGENT_NAME),
                )
                return None
            # A non-None decision means the router produced a valid classification
            all_errors = False
            last_decision = decision

        # If every router errored (all_errors still True), last_decision is the
        # fail-open default; otherwise it is the last successful classification.
        _ = all_errors  # variable consumed implicitly via last_decision logic above
        return last_decision

    async def _classify_with_router(
        self,
        router_config: dict[str, Any],
        user_text: str,
        area_context: str | None = None,
        language: str | None = None,
    ) -> RouterDecision | None:
        """Ask a router agent to classify user text via a JSON prompt.

        Supports both Ollama-based and Home Assistant conversation agent routers.
        The classification prompt is sent to the agent and the JSON response is
        parsed into a :class:`RouterDecision`.

        Logging verbosity is controlled by ``CONF_ROUTER_LOG_LEVEL``:

        * ``none`` — only warnings and errors.
        * ``complexity_only`` — also log the complexity score.
        * ``debug_info`` — log the full routing decision.
        * ``debug_with_query`` — additionally log the query text (⚠ logs PII).

        Failure modes (wrong agent type, missing config, network error, empty or
        unparseable response) are handled according to ``CONF_ROUTER_FALLBACK``.

        Only a ``complexity == 0`` JSON response is treated as an explicit block
        signal, regardless of the fallback setting.

        Args:
            router_config: Configuration dict for the router agent.
            user_text:     The raw user input text to classify.
            area_context:  Friendly area name of the originating device, or None.
            language:      BCP-47 language tag from the HA pipeline, or None.

        Returns:
            A RouterDecision on success or fallback, None to block the request.
        """
        agent_id: str = router_config.get("id", "")
        agent_name: str = router_config.get(CONF_AGENT_NAME, "Unknown")
        log_level: str = router_config.get(CONF_ROUTER_LOG_LEVEL, DEFAULT_ROUTER_LOG_LEVEL)
        fallback: str = router_config.get(CONF_ROUTER_FALLBACK, DEFAULT_ROUTER_FALLBACK)

        self._statistics.record_request(agent_id, agent_name)

        custom_prompt: str = (
            router_config.get(CONF_ROUTER_CUSTOM_PROMPT, DEFAULT_ROUTER_CUSTOM_PROMPT) or ""
        ).strip()
        prompt_template = custom_prompt if custom_prompt else ROUTER_CLASSIFICATION_PROMPT
        prompt = prompt_template.replace("{user_text}", user_text)

        entity_context = self._entity_context_cache.get_summary(self.hass)
        if entity_context:
            prompt = f"{prompt}\n\n{entity_context}"
        if area_context:
            prompt = f"{prompt}\n\nDevice area: {area_context}"
        if language:
            prompt = f"{prompt}\nLanguage: {language}"

        if log_level == ROUTER_LOG_LEVEL_DEBUG_QUERY:
            _LOGGER.debug("Router '%s' classifying query: %s", agent_name, user_text)

        start_time = time.monotonic()
        response = await self._call_router_backend(router_config, prompt)
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

        self._statistics.record_success(agent_id, elapsed_ms)
        if parsed.complexity == 0:
            self._statistics.record_block(agent_id)
            self._dispatch_stats_updated()
            return None

        # Record intent_hint in stats if the router provided one (Feature 8)
        if parsed.intent_hint:
            self._statistics.record_intent_hint(agent_id, parsed.intent_hint)

        self._log_router_decision(log_level, agent_name, parsed)
        self._dispatch_stats_updated()
        return parsed

    def _dispatch_stats_updated(self) -> None:
        """Fire the stats-updated dispatcher signal for this config entry."""
        async_dispatcher_send(
            self.hass,
            SIGNAL_STATS_UPDATED.format(entry_id=self._config_entry.entry_id),
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
        self._statistics.record_failure(agent_id)
        self._dispatch_stats_updated()
        return self._router_fallback(fallback)

    def _log_router_decision(self, log_level: str, agent_name: str, parsed: RouterDecision) -> None:
        """Emit a routing-decision log at the configured verbosity level.

        Args:
            log_level: One of the ``ROUTER_LOG_LEVEL_*`` constants.
            agent_name: Display name of the router agent (for log messages).
            parsed: The :class:`RouterDecision` returned by the router.
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

        Routes to the Ollama client for ``AGENT_TYPE_OLLAMA`` configs, or
        to the HA conversation service for ``AGENT_TYPE_EXISTING`` /
        ``AGENT_TYPE_LOCAL_HA`` configs.  Returns ``None`` for unknown types
        or missing required configuration.

        Args:
            router_config: Configuration dict for the router agent.
            prompt: The fully-formatted classification prompt.

        Returns:
            The raw text response from the router back-end, or None on error.
        """
        agent_type: str = router_config.get(CONF_AGENT_TYPE, "")
        agent_name: str = router_config.get(CONF_AGENT_NAME, "Unknown")
        agent_id: str = router_config.get("id", "")
        timeout: int = router_config.get(CONF_TIMEOUT, DEFAULT_ROUTER_TIMEOUT)

        if agent_type == AGENT_TYPE_OLLAMA:
            return await self._call_ollama_router(
                router_config, prompt, agent_id, agent_name, timeout
            )

        if agent_type in (AGENT_TYPE_EXISTING, AGENT_TYPE_LOCAL_HA):
            entity_id: str = router_config.get(CONF_ENTITY_ID, "")
            if not entity_id:
                _LOGGER.warning(
                    "Router '%s' has no entity_id configured — applying fallback", agent_name
                )
                return None
            return await self._call_existing_agent_for_routing(entity_id, prompt, timeout)

        _LOGGER.debug(
            "Router '%s' has unknown type '%s' — applying fallback", agent_name, agent_type
        )
        return None

    async def _call_ollama_router(
        self,
        router_config: dict[str, Any],
        prompt: str,
        agent_id: str,
        agent_name: str,
        timeout: int,
    ) -> str | None:
        """Call the Ollama back-end for a routing classification.

        Args:
            router_config: Configuration dict for the Ollama router agent.
            prompt: The fully-formatted classification prompt.
            agent_id: Unique ID for the Ollama client cache key.
            agent_name: Display name used in warning messages.
            timeout: Request timeout in seconds.

        Returns:
            The raw text response, or None if URL/model are not configured.
        """
        ollama_url: str = router_config.get(CONF_OLLAMA_URL, "")
        ollama_model: str = router_config.get(CONF_OLLAMA_MODEL, "")
        if not ollama_url or not ollama_model:
            _LOGGER.warning(
                "Router '%s' (Ollama) is missing URL or model — applying fallback", agent_name
            )
            return None
        if agent_id not in self._ollama_clients:
            self._ollama_clients[agent_id] = OllamaClient(ollama_url, ollama_model, timeout)
        return await self._ollama_clients[agent_id].generate(prompt)

    def _router_fallback(self, fallback: str) -> RouterDecision | None:
        """Return the fallback RouterDecision (or None to block) based on the config.

        Called when a router agent errors, times out, or returns an unparseable
        response.  Statistics and dispatcher signals should already have been
        sent by the caller before invoking this method.

        Args:
            fallback: One of the ``ROUTER_FALLBACK_*`` constants.

        Returns:
            A RouterDecision for fail-open or skip-routing fallbacks, or
            ``None`` when ``fallback == ROUTER_FALLBACK_BLOCK``.
        """
        if fallback == ROUTER_FALLBACK_SKIP_ROUTING:
            return RouterDecision(local_ha=False, complexity=ROUTER_SKIP_ROUTING_COMPLEXITY)
        if fallback == ROUTER_FALLBACK_BLOCK:
            return None
        return RouterDecision(local_ha=False, complexity=DEFAULT_ROUTER_COMPLEXITY)

    async def _call_existing_agent_for_routing(
        self, entity_id: str, prompt: str, timeout: int
    ) -> str | None:
        """Send the classification prompt to an installed HA conversation agent.

        The agent is called via the ``conversation.process`` service with the
        formatted classification prompt as the user message.  The text of the
        first speech response is returned for JSON parsing.

        Args:
            entity_id: The conversation entity to call (e.g.
                       ``conversation.google_generative_ai_conversation_1``).
            prompt: The fully-formatted classification prompt text.
            timeout: Maximum seconds to wait for a response.

        Returns:
            The response speech text, or None on error/timeout.
        """
        if not self.hass.states.get(entity_id):
            _LOGGER.warning("Routing agent entity %s not found", entity_id)
            return None

        try:
            async with asyncio.timeout(timeout):
                response = await self.hass.services.async_call(
                    CONVERSATION_DOMAIN,
                    "process",
                    {
                        "text": prompt,
                        "agent_id": entity_id,
                        # Fresh conversation each time — routing must be stateless
                        "conversation_id": None,
                    },
                    blocking=True,
                    return_response=True,
                )
        except asyncio.TimeoutError:
            _LOGGER.warning("Routing agent %s timed out after %d seconds", entity_id, timeout)
            return None
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error calling routing agent %s: %s", entity_id, err)
            return None

        return self._extract_speech_from_response(response)

    def _extract_speech_from_response(self, response: Any) -> str | None:
        """Extract the speech text from a ``conversation.process`` service call response.

        Traverses ``response["response"]["speech"]["plain"]["speech"]``.
        Returns the speech text if present and non-empty, otherwise ``None``.

        Args:
            response: The raw return value from ``hass.services.async_call``.

        Returns:
            The speech text string, or None if absent, empty, or malformed.
        """
        try:
            speech_text = response["response"]["speech"]["plain"]["speech"]
            return str(speech_text) if speech_text else None
        except (KeyError, TypeError, AttributeError):
            return None

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
                elif agent_type == AGENT_TYPE_WEB_SEARCH:
                    result = await self._process_with_web_search(agent_config, user_input)
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

    def _render_ha_context(self, prompt: str) -> str:
        """Substitute ``{ha_*}`` template tokens with live HA configuration values.

        Supported tokens:

        * ``{ha_location_name}`` — friendly name of the HA installation
          (``hass.config.location_name``).
        * ``{ha_timezone}`` — IANA timezone string, e.g. ``"Europe/London"``
          (``hass.config.time_zone``).
        * ``{ha_unit_temperature}`` — temperature unit symbol, e.g. ``"°C"``
          (``hass.config.units.temperature_unit.value``).

        Tokens not present in *prompt* are silently ignored, making this safe
        to call on any user-supplied prompt text.

        Args:
            prompt: Raw prompt text, possibly containing ``{ha_*}`` tokens.

        Returns:
            Prompt with all recognised tokens replaced by their runtime values.
        """
        cfg = self.hass.config
        unit_temp = getattr(cfg.units, "temperature_unit", None)
        temperature_unit: str = (
            unit_temp.value if unit_temp is not None and hasattr(unit_temp, "value") else ""
        )
        return (
            prompt.replace("{ha_location_name}", cfg.location_name or "")
            .replace("{ha_timezone}", cfg.time_zone or "")
            .replace("{ha_unit_temperature}", temperature_unit)
        )

    async def _process_with_ollama(
        self, agent_config: dict[str, Any], user_input: ConversationInput
    ) -> ConversationResult | None:
        """Process input with an Ollama agent using session context (#9, #10).

        When the request originates from a device that has an area assigned in
        HA, the area name is prepended to the user message (e.g.
        ``"[Area: Kitchen] turn on the lights"``).  This gives the local model
        enough context to correctly scope device control commands without
        needing to enumerate all entities.  The area prefix is **not** stored
        in session memory — only the original user text is persisted.

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
        agent_prompt: str = agent_config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
        global_default: str = self._config_entry.data.get(
            CONF_DEFAULT_PROMPT, DEFAULT_DEFAULT_PROMPT
        )
        system_prompt: str = agent_prompt if agent_prompt else global_default
        system_prompt = self._render_ha_context(system_prompt)

        if not isinstance(ollama_url, str) or not isinstance(ollama_model, str):
            return None
        if agent_id not in self._ollama_clients:
            self._ollama_clients[agent_id] = OllamaClient(ollama_url, ollama_model, timeout)

        client = self._ollama_clients[agent_id]

        # Feature 10 — Language passthrough: append "Respond in {lang}." when
        # the pipeline language differs from the configured integration language
        # and CONF_FORCE_RESPONSE_LANGUAGE is enabled (default: True).
        config = self._get_config()
        force_lang: bool = config.get(CONF_FORCE_RESPONSE_LANGUAGE, DEFAULT_FORCE_RESPONSE_LANGUAGE)
        if force_lang and user_input.language:
            input_base = user_input.language.lower().split("-")[0].split("_")[0]
            conf_base = (
                str(config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)).lower().split("-")[0].split("_")[0]
            )
            if input_base != conf_base:
                lang_name = _get_language_name(user_input.language)
                system_prompt = f"{system_prompt}\nRespond in {lang_name}.".strip()

        # Build message list: optional system prompt + history + current turn (#9)
        history = self._session_memory.get_messages(user_input.conversation_id)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        area_ctx = self._get_device_area(user_input.device_id)
        user_text = f"[Area: {area_ctx}] {user_input.text}" if area_ctx else user_input.text
        messages.append({"role": "user", "content": user_text})

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

        The sub-agent is intentionally isolated from the outer ``current_chat_log``
        context variable.  In HA 2025.2+ the built-in HA conversation agent (and
        some other agents) call ``async_add_assistant_content_without_tools``
        internally when they process a request.  Because the outer
        ``conversation.process`` call shares the same asyncio context, that write
        would land in the *same* ChatLog that NeuralBridge's own
        ``_maybe_add_to_chat_log`` also writes to — resulting in the response text
        appearing twice in the Assist UI.

        Temporarily setting ``current_chat_log`` to ``None`` for the duration of
        the sub-call prevents the sub-agent from polluting the outer ChatLog.
        NeuralBridge re-adds the response exactly once via ``_maybe_add_to_chat_log``
        after ``async_process`` returns.

        Args:
            agent_config: Configuration dict for the existing agent.
            user_input: The user's conversation input.

        Returns:
            ConversationResult on success, None on failure.
        """
        entity_id: str | None = agent_config.get(CONF_ENTITY_ID)
        if entity_id is None:
            _LOGGER.error("Existing agent has no entity_id configured")
            return None

        agent_state = self.hass.states.get(entity_id)
        if not agent_state:
            _LOGGER.error("Conversation agent %s not found", entity_id)
            return None

        # Isolate the sub-agent from the outer ChatLog so the sub-agent cannot
        # write a duplicate entry into it.  Restore after the call regardless of
        # outcome.  See docstring for the full explanation.
        from homeassistant.components.conversation.chat_log import (  # noqa: PLC0415
            current_chat_log,
        )

        _chat_log_var = current_chat_log
        _chat_log_token = current_chat_log.set(None)

        response: Any = None
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
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error calling conversation agent %s: %s", entity_id, err)
        finally:
            _chat_log_var.reset(_chat_log_token)

        if response is None:
            return None

        if not response or "response" not in response:
            _LOGGER.error("Agent %s returned unexpected response shape", entity_id)
            return None

        response_section: Any = response["response"]

        # In assist mode, only accept responses where HA successfully executed an
        # intent (response_type == "action_done").  Any other response type means
        # HA didn't understand the request, so we return None and let the router
        # continue to the next lower-priority agent (e.g. Ollama / Gemini).
        assist_mode = agent_config.get(CONF_AGENT_ASSIST_MODE, DEFAULT_AGENT_ASSIST_MODE)
        if assist_mode:
            response_type = str(response_section.get("response_type", ""))
            if response_type != "action_done":
                _LOGGER.debug(
                    "Agent %s assist mode: response_type '%s' is not action_done, "
                    "falling through to next agent",
                    entity_id,
                    response_type,
                )
                return None

        speech_text = str(response_section.get("speech", {}).get("plain", {}).get("speech", ""))
        return self._create_result(speech_text, user_input.conversation_id) if speech_text else None

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

        if agent_id not in self._web_search_clients:
            self._web_search_clients[agent_id] = WebSearchClient(config_for_client)

        client = self._web_search_clients[agent_id]
        summary = await client.search_and_summarise(user_input.text)

        if summary:
            return self._create_result(summary, user_input.conversation_id)

        return None

    async def _apply_guard_rail_action(
        self,
        action: str,
        result: ConversationResult,
        response_text: str,
        conversation_id: str | None,
        guard_rail_result: Any,
    ) -> ConversationResult | None:
        """Apply the configured guard rail action after a triggered rule.

        Args:
            action: The configured guard rail action key.
            result: The conversation result to modify or replace.
            response_text: The extracted response text.
            conversation_id: Conversation ID for pending-response caching.
            guard_rail_result: The guard rail check result.

        Returns:
            ConversationResult if the action produces a response, None otherwise.
        """
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
        agent_guard_rail_enabled = agent_config.get(
            CONF_GUARD_RAIL_ENABLED_FOR_AGENT, DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
        )
        if not guard_rail_enabled or not agent_guard_rail_enabled:
            return None

        response_text = self._extract_response_text(result)
        if not response_text:
            return None

        current_rules: dict[str, list[str]] | None = config.get(CONF_GUARD_RAIL_RULES)
        if current_rules != self._guard_rail_rules_snapshot:
            self._guard_rail_checker = None
            self._guard_rail_rules_snapshot = current_rules

        if self._guard_rail_checker is None:
            ai_threshold = config.get(CONF_GUARD_RAIL_AI_THRESHOLD, DEFAULT_GUARD_RAIL_AI_THRESHOLD)
            use_detoxify = config.get(CONF_GUARD_RAIL_USE_DETOXIFY, DEFAULT_GUARD_RAIL_USE_DETOXIFY)
            detoxify_threshold = config.get(
                CONF_GUARD_RAIL_DETOXIFY_THRESHOLD, DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD
            )
            self._guard_rail_checker = GuardRailChecker(
                rules=current_rules,
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

        return await self._apply_guard_rail_action(
            action, result, response_text, conversation_id, guard_rail_result
        )

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
            agent_config: dict[str, Any] = agent
            if agent_config.get("id") == guard_rail_agent_id:
                return agent_config

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

    def _get_device_area(self, device_id: str | None) -> str | None:
        """Return the friendly area name for a device, or None if unavailable.

        Looks up the device in the HA device registry, then resolves its area
        via the area registry.  Used to inject location context into router
        classification prompts and Ollama user messages so that phrases like
        "turn on the lights" are automatically scoped to the caller's room.

        Args:
            device_id: The HA device ID from :attr:`ConversationInput.device_id`.

        Returns:
            The area name string, or None when the device has no area or is
            unknown.
        """
        if not device_id:
            return None
        dev_registry = dr.async_get(self.hass)
        device = dev_registry.async_get(device_id)
        if device is None or not device.area_id:
            return None
        area_reg = ar.async_get(self.hass)
        area = area_reg.async_get_area(device.area_id)
        if area is None:
            return None
        return str(area.name)

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

    async def async_will_remove_from_hass(self) -> None:
        """Clean up resources when entity is removed from Home Assistant."""
        for client in self._ollama_clients.values():
            await client.close()
        self._ollama_clients.clear()
