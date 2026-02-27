"""Router engine for NeuralBridge — ClassifIcation logic and RouterDecision type."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    AGENT_TYPE_WEB_SEARCH,
    CANNOT_ANSWER_SENTINEL,
    CONF_AGENT_MAX_COMPLEXITY,
    CONF_AGENT_MIN_COMPLEXITY,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_ENTITY_ID,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_ROUTER_CUSTOM_PROMPT,
    CONF_ROUTER_FALLBACK,
    CONF_ROUTER_LOG_LEVEL,
    CONF_STRATEGY_MAP,
    CONF_TIMEOUT,
    DEFAULT_AGENT_MAX_COMPLEXITY,
    DEFAULT_AGENT_MIN_COMPLEXITY,
    DEFAULT_ROUTER_COMPLEXITY,
    DEFAULT_ROUTER_CUSTOM_PROMPT,
    DEFAULT_ROUTER_FALLBACK,
    DEFAULT_ROUTER_LOG_LEVEL,
    DEFAULT_ROUTER_TIMEOUT,
    MAX_COMPOUND_FRAGMENTS,
    ROUTER_CLASSIFICATION_PROMPT,
    ROUTER_CONFIDENCE_HIGH,
    ROUTER_CONFIDENCE_LOW,
    ROUTER_FALLBACK_BLOCK,
    ROUTER_FALLBACK_SKIP_ROUTING,
    ROUTER_LOG_LEVEL_COMPLEXITY,
    ROUTER_LOG_LEVEL_DEBUG,
    ROUTER_LOG_LEVEL_DEBUG_QUERY,
    ROUTER_RESPONSE_KEY_COMPLEXITY,
    ROUTER_RESPONSE_KEY_CONFIDENCE,
    ROUTER_RESPONSE_KEY_DIMENSION,
    ROUTER_RESPONSE_KEY_INTENT_HINT,
    ROUTER_RESPONSE_KEY_LOCAL_HA,
    ROUTER_RESPONSE_KEY_RELEVANT_SENSORS,
    ROUTER_RESPONSE_KEY_SUGGESTED_ORDER,
    ROUTER_RESPONSE_KEY_WEB_SEARCH,
    ROUTER_SKIP_ROUTING_COMPLEXITY,
    ROUTING_STRATEGY_DEFAULT,
    SIGNAL_STATS_UPDATED,
    SUGGESTED_ORDER_NAME_MAX_LEN,
    VALID_INTENT_HINTS,
    VALID_ROUTER_CONFIDENCE_VALUES,
    VALID_ROUTER_DIMENSIONS,
)
from .ollama_client import OllamaClient

if TYPE_CHECKING:
    from .agent_benchmark import AgentBenchmarker
    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


class RouterDecision:
    """Immutable classification result produced by a router agent.

    Attributes:
        local_ha:              True when the request is a home-automation /
                               device-control command that should be handled by
                               a LOCAL_HA agent.
        web_search:            True when the request requires real-time or
                               current data (news, live scores, etc.).
        complexity:            Integer 1-100 indicating estimated request
                               complexity.  Higher values suggest cloud-capable
                               agents are preferred.
        intent_hint:           Optional routing hint (``"timer"``,
                               ``"reminder"``, ``"todo"``,
                               ``"shopping_list"``, ``"announce"``, or
                               ``None``).  Forces LOCAL_HA routing for certain
                               intents.
        confidence:            How confident the router is in its
                               classification.  ``"high"`` (default) or
                               ``"low"`` (ambiguous input).  When ``"low"``,
                               flag-based promotion (local_ha/web_search) is
                               skipped and all eligible agents are returned in
                               priority order (Feature 14b).
        dimension:             The dominant capability dimension this query
                               tests: ``"reasoning"``,
                               ``"instruction_following"``,
                               ``"smart_home_intent"``, ``"factual"``,
                               ``"memory"``, or ``None``.  Used by Feature 14d
                               to record the dimension even when
                               ``suggested_agent_order`` is null.
        suggested_agent_order: Ordered tuple of agent *names* in the order the
                               router recommends trying them.  Empty tuple when
                               the router returned null or no order was given.
    """

    __slots__ = (
        "complexity",
        "confidence",
        "dimension",
        "intent_hint",
        "local_ha",
        "relevant_sensors",
        "suggested_agent_order",
        "web_search",
    )

    # Class-level annotations required for mypy __slots__ attribute resolution
    complexity: int
    confidence: str
    dimension: str | None
    intent_hint: str | None
    local_ha: bool
    relevant_sensors: tuple[str, ...]
    suggested_agent_order: tuple[str, ...]
    web_search: bool

    def __init__(
        self,
        local_ha: bool,
        complexity: int,
        web_search: bool = False,
        intent_hint: str | None = None,
        confidence: str = ROUTER_CONFIDENCE_HIGH,
        relevant_sensors: tuple[str, ...] = (),
        dimension: str | None = None,
        suggested_agent_order: tuple[str, ...] = (),
    ) -> None:
        """Initialise a RouterDecision.

        Args:
            local_ha:              Whether the request targets local
                                   home-automation.
            complexity:            Estimated complexity score (1-100).
            web_search:            Whether the request needs real-time web data.
            intent_hint:           Optional intent classification hint
                                   (Feature 8).
            confidence:            Router's confidence: "high" (default) or
                                   "low" (Feature 14b).
            relevant_sensors:      Tuple of entity IDs the router identified
                                   as relevant to the query.
            dimension:             Dominant capability dimension (Feature 14d).
            suggested_agent_order: Router-recommended agent name order
                                   (Feature 14d).
        """
        object.__setattr__(self, "local_ha", local_ha)
        object.__setattr__(self, "complexity", complexity)
        object.__setattr__(self, "web_search", web_search)
        object.__setattr__(self, "intent_hint", intent_hint)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "relevant_sensors", relevant_sensors)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "suggested_agent_order", suggested_agent_order)

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
            and self.confidence == other.confidence
            and self.relevant_sensors == other.relevant_sensors
            and self.dimension == other.dimension
            and self.suggested_agent_order == other.suggested_agent_order
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.local_ha,
                self.complexity,
                self.web_search,
                self.intent_hint,
                self.confidence,
                self.relevant_sensors,
                self.dimension,
                self.suggested_agent_order,
            )
        )

    def __repr__(self) -> str:
        return (
            f"RouterDecision(local_ha={self.local_ha!r}, "
            f"complexity={self.complexity!r}, "
            f"web_search={self.web_search!r}, "
            f"intent_hint={self.intent_hint!r}, "
            f"confidence={self.confidence!r}, "
            f"relevant_sensors={self.relevant_sensors!r}, "
            f"dimension={self.dimension!r}, "
            f"suggested_agent_order={self.suggested_agent_order!r})"
        )


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

    # Extract optional confidence (Feature 14b); default to "high" when absent or invalid
    raw_confidence = data.get(ROUTER_RESPONSE_KEY_CONFIDENCE, ROUTER_CONFIDENCE_HIGH)
    confidence: str = (
        str(raw_confidence)
        if raw_confidence in VALID_ROUTER_CONFIDENCE_VALUES
        else ROUTER_CONFIDENCE_HIGH
    )

    # Extract optional relevant_sensors (router-guided injection); accept only strings
    raw_relevant = data.get(ROUTER_RESPONSE_KEY_RELEVANT_SENSORS, [])
    relevant_sensors: tuple[str, ...] = (
        tuple(str(e) for e in raw_relevant if isinstance(e, str))
        if isinstance(raw_relevant, list)
        else ()
    )

    # Feature 14d: extract dimension; validate against known values
    raw_dimension = data.get(ROUTER_RESPONSE_KEY_DIMENSION)
    dimension: str | None = (
        str(raw_dimension)
        if isinstance(raw_dimension, str) and raw_dimension in VALID_ROUTER_DIMENSIONS
        else None
    )

    # Feature 14d: extract suggested_agent_order; cap each name to prevent injection
    raw_order = data.get(ROUTER_RESPONSE_KEY_SUGGESTED_ORDER)
    suggested_agent_order: tuple[str, ...] = ()
    if isinstance(raw_order, list):
        seen: set[str] = set()
        valid_names: list[str] = []
        for item in raw_order:
            if not isinstance(item, str) or not item.strip():
                continue
            name = item.strip()[:SUGGESTED_ORDER_NAME_MAX_LEN]
            if name not in seen:
                seen.add(name)
                valid_names.append(name)
        suggested_agent_order = tuple(valid_names)

    return RouterDecision(
        local_ha=local_ha,
        complexity=complexity,
        web_search=web_search,
        intent_hint=intent_hint,
        confidence=confidence,
        relevant_sensors=relevant_sensors,
        dimension=dimension,
        suggested_agent_order=suggested_agent_order,
    )


def _apply_router_decision(
    decision: RouterDecision,
    processing_agents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter and reorder processing agents based on the router decision.

    **Phase 1 — Complexity Thresholds (Feature 14a)**:
    Agents whose ``agent_min_complexity``/``agent_max_complexity`` range does
    not contain ``decision.complexity`` are excluded.  If every agent is
    excluded, all agents fall through unchanged (safety fallback).

    **Phase 2 — Confidence Gate (Feature 14b)**:
    When ``decision.confidence == "low"`` the router was uncertain, so
    flag-based promotion (local_ha, web_search, intent_hint) is skipped and
    the eligible list is returned as-is in priority order.

    **Phase 3 — Flag-based Promotion**:
    Standard logic: intent_hint forces LOCAL_HA; web_search promotes
    WEB_SEARCH agents; local_ha promotes LOCAL_HA agents; otherwise both
    LOCAL_HA and WEB_SEARCH agents are excluded so that general knowledge
    queries are handled only by general-purpose agents (e.g. Ollama).  A
    safety fallback re-admits WEB_SEARCH agents if no general-purpose agents
    remain after exclusion.

    A ``decision.complexity`` of ``ROUTER_SKIP_ROUTING_COMPLEXITY`` (-1) is a
    special sentinel meaning "skip routing" — all processing agents are returned
    unchanged in their original priority order (bypasses all phases).

    Args:
        decision:          The RouterDecision produced by the router agent.
        processing_agents: Priority-sorted list of processing agent configs.

    Returns:
        Filtered and/or reordered list of processing agent configs.
    """
    if decision.complexity == ROUTER_SKIP_ROUTING_COMPLEXITY:
        # Fallback=skip_routing — bypass agent filtering, try everything
        return list(processing_agents)

    # ---- Phase 1: complexity threshold filtering (Feature 14a) ----
    score = decision.complexity
    eligible = [
        a
        for a in processing_agents
        if (
            a.get(CONF_AGENT_MIN_COMPLEXITY, DEFAULT_AGENT_MIN_COMPLEXITY) <= score
            and a.get(CONF_AGENT_MAX_COMPLEXITY, DEFAULT_AGENT_MAX_COMPLEXITY) >= score
        )
    ]
    if not eligible:
        # Safety fallback: never leave the pipeline empty due to threshold config
        _LOGGER.debug(
            "All agents filtered by complexity thresholds (score=%d); using full agent list",
            score,
        )
        eligible = list(processing_agents)

    # ---- Phase 2: confidence gate — skip flag-based promotion if uncertain (Feature 14b) ----
    if decision.confidence == ROUTER_CONFIDENCE_LOW:
        return eligible

    # ---- Phase 3: flag-based promotion (existing logic) ----

    # Feature 8: intent_hint forces LOCAL_HA for timer/reminder/todo/shopping_list commands
    # regardless of what the router returned for local_ha (extra safety net).
    if decision.intent_hint in ("timer", "reminder", "todo", "shopping_list"):
        local = [a for a in eligible if a.get(CONF_AGENT_TYPE) == AGENT_TYPE_LOCAL_HA]
        others = [a for a in eligible if a.get(CONF_AGENT_TYPE) != AGENT_TYPE_LOCAL_HA]
        return local + others
    if decision.web_search:
        # Promote WEB_SEARCH agents to front; keep original order within each group
        web = [a for a in eligible if a.get(CONF_AGENT_TYPE) == AGENT_TYPE_WEB_SEARCH]
        others = [a for a in eligible if a.get(CONF_AGENT_TYPE) != AGENT_TYPE_WEB_SEARCH]
        return web + others
    if decision.local_ha:
        # Promote LOCAL_HA agents to front; keep original order within each group
        local = [a for a in eligible if a.get(CONF_AGENT_TYPE) == AGENT_TYPE_LOCAL_HA]
        others = [a for a in eligible if a.get(CONF_AGENT_TYPE) != AGENT_TYPE_LOCAL_HA]
        return local + others
    # Exclude both LOCAL_HA and WEB_SEARCH agents — request needs neither HA control nor
    # live internet data; only general-purpose (e.g. Ollama) agents should handle it.
    non_routing = [
        a
        for a in eligible
        if a.get(CONF_AGENT_TYPE) not in (AGENT_TYPE_LOCAL_HA, AGENT_TYPE_WEB_SEARCH)
    ]
    if not non_routing:
        # Safety fallback: no general-purpose agents available; exclude LOCAL_HA only so the
        # pipeline is never left empty (WEB_SEARCH is still better than no response at all).
        _LOGGER.debug(
            "No general-purpose agents available after web-search exclusion; excluding LOCAL_HA only"
        )
    result_agents = non_routing or [
        a for a in eligible if a.get(CONF_AGENT_TYPE) != AGENT_TYPE_LOCAL_HA
    ]

    # ---- Phase 4: honour router's suggested agent order (Feature 14d) ----
    if decision.suggested_agent_order:
        name_to_agent = {a.get(CONF_AGENT_NAME): a for a in result_agents}
        ordered: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for name in decision.suggested_agent_order:
            if name in name_to_agent and name not in seen_names:
                ordered.append(name_to_agent[name])
                seen_names.add(name)
        # Agents not mentioned by the router are appended at the end (safety net —
        # never drop an agent just because the router didn't list it)
        remainder = [a for a in result_agents if a.get(CONF_AGENT_NAME) not in seen_names]
        if ordered:
            result_agents = ordered + remainder

    return result_agents


def _build_capability_block(
    processing_agents: list[dict[str, Any]],
    benchmarker: "AgentBenchmarker | None",
) -> str:
    """Build the ``[Agent capabilities:]`` prompt block for the router LLM.

    Returns an empty string when no benchmark data is available or
    *benchmarker* is ``None``.  Agents with no COMPLETE profile show ``n/a``
    for all score columns — they are never omitted from the table so the
    router knows every agent exists.

    Args:
        processing_agents: Priority-sorted list of processing agent configs.
        benchmarker:        The shared :class:`AgentBenchmarker` instance.

    Returns:
        Formatted prompt block string, or ``""`` when unavailable.
    """
    if benchmarker is None or not processing_agents:
        return ""

    rows: list[str] = []
    for agent in processing_agents:
        name: str = agent.get(CONF_AGENT_NAME, "?")
        agent_type: str = agent.get(CONF_AGENT_TYPE, "?")
        agent_id: str = agent.get("id", "")
        profile = benchmarker.get_profile(agent_id)

        from .agent_benchmark import BenchmarkStatus  # noqa: PLC0415

        if profile is not None and profile.status == BenchmarkStatus.COMPLETE:
            cap = f"{profile.capability_score}/8"
            reasoning = (
                f"{profile.score_reasoning}/2" if profile.score_reasoning is not None else "n/a"
            )
            inst = (
                f"{profile.score_instruction_following}/3"
                if profile.score_instruction_following is not None
                else "n/a"
            )
            factual = f"{profile.score_factual}/1" if profile.score_factual is not None else "n/a"
            memory = f"{profile.score_memory}/1" if profile.score_memory is not None else "n/a"
            latency = (
                str(int(profile.median_latency_ms))
                if profile.median_latency_ms is not None
                else "n/a"
            )
        else:
            cap = reasoning = inst = factual = memory = latency = "n/a"

        rows.append(
            f"{name:<20} | {agent_type:<22} | {cap:<8} | {reasoning:<9} | "
            f"{inst:<10} | {factual:<7} | {memory:<6} | {latency}"
        )

    if not rows:  # pragma: no cover
        return ""

    header = (
        "Name                 | Type                   | CapScore | Reasoning | "
        "InstFollow | Factual | Memory | Latency(ms)"
    )
    return "[Agent capabilities:]\n" + header + "\n" + "\n".join(rows)


def _build_strategy_block(dimension_strategies: dict[str, str]) -> str:
    """Build the ``[Routing preferences:]`` prompt block.

    Returns an empty string when all strategies are ``"default"`` (avoids
    adding an empty section that wastes tokens).

    Args:
        dimension_strategies: Mapping of dimension name to strategy value.

    Returns:
        Formatted prompt block string, or ``""`` when all strategies are default.
    """
    non_default = {d: s for d, s in dimension_strategies.items() if s != ROUTING_STRATEGY_DEFAULT}
    if not non_default:
        return ""
    lines = [f"{dimension}: {strategy}" for dimension, strategy in dimension_strategies.items()]
    return "[Routing preferences:]\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure text-processing utility functions
# ---------------------------------------------------------------------------

# Maximum response length (characters) to apply refusal-pattern matching against.
# Longer responses are substantive answers even if they mention limitations in
# passing, so we do not re-route them.
_REFUSAL_MAX_LEN: int = 400

# Refusal phrases emitted by the built-in HA agent and cloud agents when a
# question falls outside their scope.  Matched against the lowercased response.
_REFUSAL_PATTERNS: re.Pattern[str] = re.compile(
    # "I cannot answer / I can't answer"
    r"i (?:am |'m )?(?:sorry[,.]?\s+)?(?:i )?(?:cannot|can't)\s+answer"
    # "I am sorry, I cannot provide\u2026" / "I cannot provide\u2026"
    r"|i (?:am |'m )?(?:sorry[,.]?\s+i |really )?(?:cannot|can't|am unable to|'m unable to)\s+provide"
    # "I'm unable to provide/access/browse/search/look up/find"
    r"|i(?:'m| am) unable to (?:provide|access|browse|search|look up|find|answer)"
    # "I don't have access to / real-time / internet"
    r"|i (?:don't|do not) have (?:access to|the ability to|real.?time|internet)"
    # "My capabilities are limited/restricted"
    r"|my capabilities (?:are )?(?:limited|restricted)"
    # "My purpose is to control\u2026" (HA built-in boilerplate)
    r"|my purpose is to (?:control|manage|help with)"
    # "beyond/outside my capabilities"
    r"|(?:beyond|outside) my (?:current )?capabilities"
    # "I cannot/can't browse/search the internet/web"
    r"|i (?:cannot|can't) (?:browse|search|look up|access) the (?:internet|web)"
    # "not able/designed/trained to browse/search/provide/access/answer"
    r"|not (?:able|designed|trained) to (?:browse|search|provide|access|answer)",
    re.IGNORECASE,
)

# Patterns that unambiguously require a web-search agent.  Checked BEFORE the
# LLM router so that small/restricted models cannot misclassify them.
_WEB_SEARCH_DETERMINISTIC_RE: re.Pattern[str] = re.compile(
    # Headlines and news (with or without a named outlet)
    r"\bheadlines?\b"
    r"|\b(?:latest|breaking|top|today's?|current|recent)\s+news\b"
    r"|\bnews\s+(?:today|headlines?|update|brief)\b"
    r"|\b(?:BBC|CNN|Sky\s+News|ITV\s+News|Reuters|AP\s+News|The\s+Guardian|"
    r"Daily\s+Mail|The\s+Times|Fox\s+News|NBC\s+News|ABC\s+News)\b"
    # Weather for a NAMED external place (bare "weather" without location stays for LOCAL_HA)
    r"|\bweather\s+(?:in|at|for|near)\b" r"|\b(?:temperature|forecast)\s+(?:in|at|for)\b"
    # Travel: distance, time, directions
    r"|\bhow\s+(?:far|long)\s+(?:is\s+(?:it\s+)?)?(?:to|from|between)\b"
    r"|\bdistance\s+(?:from|to|between)\b"
    r"|\btravel\s+(?:time|distance)\s+(?:to|from)\b"
    r"|\bdirections?\s+(?:to|from)\b"
    r"|\bhow\s+do\s+I\s+get\s+to\b"
    # Live scores and match results
    r"|\b(?:latest|live|current|today's?|last\s+night's?)\s+(?:score|scores|result|results)\b"
    r"|\bwho\s+won\b"
    r"|\b(?:match|game)\s+result\b"
    # Financial prices
    r"|\bprice\s+of\s+\w"
    r"|\b(?:bitcoin|btc|ethereum|eth|crypto)\s+price\b"
    r"|\bstock\s+price\b"
    r"|\bexchange\s+rate\b",
    re.IGNORECASE,
)

# Feature 11 \u2014 Broadcast announce trigger phrases
_ANNOUNCE_RE = re.compile(
    r"^(?:announce:?\s+|broadcast\s+|tell\s+everyone(?:\s+that)?\s+"
    r"|say\s+on\s+all\s+(?:the\s+)?speakers\s+)",
    re.IGNORECASE,
)
# Maximum number of characters kept in the announce event preview field
_ANNOUNCE_PREVIEW_MAX_LEN: int = 50

_COMPOUND_RE: re.Pattern[str] = re.compile(
    r"\s+(?:and\s+then|and|then|also|after\s+that)\s+",
    re.IGNORECASE,
)


def _is_unhelpful_response(text: str) -> bool:
    """Return True when *text* indicates the agent could not answer the question.

    Two detection strategies are applied:

    1. **Sentinel check** \u2014 Ollama agents are instructed to reply with the exact
       ``CANNOT_ANSWER`` token when they cannot answer.  Matched
       case-insensitively with optional surrounding whitespace/punctuation.

    2. **Refusal pattern check** \u2014 HA built-in and cloud agents emit natural-
       language refusals ("I am sorry, I cannot provide\u2026", "My capabilities are
       limited to\u2026", etc.).  These are matched via ``_REFUSAL_PATTERNS`` but
       only for responses shorter than ``_REFUSAL_MAX_LEN`` characters to avoid
       false positives on long, substantive answers that mention limitations in
       passing.

    Args:
        text: Raw response text from an agent.

    Returns:
        True if the response signals that the agent could not answer.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Strategy 1: exact sentinel (Ollama agents)
    if stripped.rstrip(".").upper() == CANNOT_ANSWER_SENTINEL:
        return True
    # Strategy 2: natural-language refusal (HA built-in / cloud agents)
    return bool(len(stripped) <= _REFUSAL_MAX_LEN and _REFUSAL_PATTERNS.search(stripped))


def _deterministic_classify(text: str) -> RouterDecision | None:
    """Return a high-confidence RouterDecision without calling an LLM.

    Applies ``_WEB_SEARCH_DETERMINISTIC_RE`` against *text*.  When matched,
    returns a ``RouterDecision`` with ``web_search=True`` and
    ``confidence="high"`` so the LLM router is bypassed entirely for
    well-known web-search patterns (news, external weather, travel, scores,
    prices).

    Returns ``None`` when the query does not match any known pattern, allowing
    the LLM-based router (or priority-order fallback) to handle it.

    Args:
        text: Raw user query text.

    Returns:
        A ``RouterDecision`` with ``web_search=True``, or ``None``.
    """
    if _WEB_SEARCH_DETERMINISTIC_RE.search(text):
        return RouterDecision(
            local_ha=False,
            web_search=True,
            complexity=20,
            confidence=ROUTER_CONFIDENCE_HIGH,
        )
    return None


def _extract_announce_text(text: str, intent_hint: str | None = None) -> str | None:
    """Extract the announcement message from *text*.

    Checks whether *text* begins with a known trigger phrase (announce,
    broadcast, tell everyone, say on all speakers, \u2026).  If so, returns
    the content that follows the trigger.  Returns the full *text* unchanged
    when *intent_hint* is ``"announce"`` and no trigger phrase is found
    (router already identified this as an announce command).  Otherwise
    returns ``None``.

    Args:
        text: Raw user input text.
        intent_hint: Optional hint from the router (e.g. ``"announce"``).

    Returns:
        Extracted message string, or ``None`` if not an announce command.
    """
    stripped = text.strip()
    if not stripped:
        return None

    match = _ANNOUNCE_RE.match(stripped)
    if match:
        content = stripped[match.end() :].strip()
        return content if content else None

    if intent_hint == "announce":
        return stripped

    return None


def _split_compound_input(text: str) -> list[str]:
    """Split a compound voice command into individual fragment strings.

    Splits on natural conjunctions used in multi-command voice requests
    ("and then", "and", "then", "also", "after that").  Returns a list of
    up to MAX_COMPOUND_FRAGMENTS fragments.  If the text does not contain
    any of those conjunctions, returns a single-element list with the
    original text.

    Args:
        text: Raw user input text.

    Returns:
        List of one or more command fragment strings.
    """
    parts = _COMPOUND_RE.split(text.strip())
    fragments = [p.strip() for p in parts if p.strip()]
    if len(fragments) <= 1:
        return [text.strip()]
    return fragments[:MAX_COMPOUND_FRAGMENTS]


class RouterEngine:
    """Encapsulates all router-related classification methods for NeuralBridgeAgent.

    Access to agent state is via ``self._agent``; within-class method calls
    stay as ``self.<method>()``.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this engine.
        """
        self._agent = agent

    async def _check_with_routers(
        self,
        user_input: Any,
        router_agents: list[dict[str, Any]],
        processing_agents: list[dict[str, Any]] | None = None,
    ) -> RouterDecision | None:
        """Run all configured router agents and return the last classification.

        If every router agent errors the pipeline fails open — returning the
        default :class:`RouterDecision` — so that a broken router never silences
        the assistant (fail-open).

        Builds a compact agent-type manifest (e.g. ``[Available: home_assistant,
        ollama]``) from *processing_agents* and passes it to each router call
        so the model can tailor its classification to the available agents
        (Feature 14c).

        Args:
            user_input:        The user's conversation input.
            router_agents:     List of router agent configurations.
            processing_agents: Optional list of processing agent configs used
                               to construct the agent-type manifest (14c).

        Returns:
            A RouterDecision if the request should proceed (with optional hints),
            or None to block the request.
        """
        last_decision: RouterDecision | None = RouterDecision(
            local_ha=False, complexity=DEFAULT_ROUTER_COMPLEXITY, web_search=False
        )
        area_context = self._agent._llm_proxy.get_device_area(user_input.device_id)
        all_errors = True

        # Feature 14c: build deduplicated ordered agent-type list from processing agents
        agent_types: list[str] | None = None
        if processing_agents:
            seen: set[str] = set()
            types_list: list[str] = []
            for agent in processing_agents:
                agent_type = agent.get(CONF_AGENT_TYPE, "")
                if agent_type and agent_type not in seen:
                    seen.add(agent_type)
                    types_list.append(agent_type)
            if types_list:
                agent_types = types_list

        benchmarker = self._agent._get_benchmarker()
        for router_config in router_agents:
            _LOGGER.debug("Checking with router: %s", router_config.get(CONF_AGENT_NAME))
            decision = await self._agent._classify_with_router(
                router_config,
                user_input.text,
                area_context,
                user_input.language,
                agent_types,
                processing_agents=processing_agents,
                benchmarker=benchmarker,
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
        * ``debug_with_query`` — additionally log the query text (⚠ logs PII).

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
            processing_agents: Optional list of processing agent configs — used
                               to build the capability block (Feature 14d).
            benchmarker:       Optional :class:`AgentBenchmarker` — required
                               for capability block data (Feature 14d).

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
        # Append a fresh (uncached) compact sensor-name-and-unit list so the router
        # can identify which sensors exist locally (for routing decisions) without
        # receiving live values — those are only injected into the answering agent.
        # The router only needs to know *what sensors exist* to correctly route a
        # query like "is it raining?"; the live value is irrelevant at routing time.
        sensor_names = self._agent._entity_context_cache.get_sensor_names(self._agent.hass)
        if sensor_names:
            prompt = f"{prompt}\n\n{sensor_names}"
        if area_context:
            prompt = f"{prompt}\n\nDevice area: {area_context}"
        if language:
            prompt = f"{prompt}\nLanguage: {language}"
        # Feature 14c: append compact agent-type manifest so the model knows
        # which agent types are available (suppresses web_search when absent).
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

        # Feature 15: append routing-relevant preference hints (source + location only)
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
            return self._agent._record_router_failure_and_fallback(agent_id, fallback)

        parsed = _parse_router_response(response)
        if parsed is None:
            _LOGGER.warning(
                "Router '%s' returned unparseable response — applying fallback", agent_name
            )
            return self._agent._record_router_failure_and_fallback(agent_id, fallback)

        self._agent._statistics.record_success(agent_id, elapsed_ms)
        if parsed.complexity == 0:
            self._agent._statistics.record_block(agent_id)
            self._agent._dispatch_stats_updated()
            return None

        # Record intent_hint in stats if the router provided one (Feature 8)
        if parsed.intent_hint:
            self._agent._statistics.record_intent_hint(agent_id, parsed.intent_hint)

        self._agent._log_router_decision(log_level, agent_name, parsed)
        self._agent._dispatch_stats_updated()
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
        self._agent._dispatch_stats_updated()
        return self._agent._router_fallback(fallback)

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
        to the HA conversation service for ``AGENT_TYPE_INTEGRATED`` /
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
            return await self._agent._call_ollama_router(
                router_config, prompt, agent_id, agent_name, timeout
            )

        if agent_type in (AGENT_TYPE_INTEGRATED, AGENT_TYPE_LOCAL_HA):
            entity_id: str = router_config.get(CONF_ENTITY_ID, "")
            if not entity_id:
                _LOGGER.warning(
                    "Router '%s' has no entity_id configured — applying fallback", agent_name
                )
                return None
            return await self._agent._call_existing_agent_for_routing(entity_id, prompt, timeout)

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
        if agent_id not in self._agent._ollama_clients:
            self._agent._ollama_clients[agent_id] = OllamaClient(ollama_url, ollama_model, timeout)
        ollama_resp = await self._agent._ollama_clients[agent_id].generate(prompt)
        return ollama_resp.content if ollama_resp is not None else None

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
        if not self._agent.hass.states.get(entity_id):
            _LOGGER.warning("Routing agent entity %s not found", entity_id)
            return None

        try:
            async with asyncio.timeout(timeout):
                response = await self._agent.hass.services.async_call(
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

        return self._agent._extract_speech_from_response(response)

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
