"""RouterDecision value object and JSON parsing helpers.

Extracted from ``router_engine.py`` as part of Stage 3 god-class refactor.

This module contains:
* :class:`RouterDecision` — the immutable classification result produced by a
  router agent.
* :func:`_parse_router_response` — parses a raw JSON string from the router
  LLM into a :class:`RouterDecision`.
* :func:`_apply_router_decision` — filters and reorders processing agents
  based on a :class:`RouterDecision`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .const import (
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_WEB_SEARCH,
    CONF_AGENT_MAX_COMPLEXITY,
    CONF_AGENT_MIN_COMPLEXITY,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    DEFAULT_AGENT_MAX_COMPLEXITY,
    DEFAULT_AGENT_MIN_COMPLEXITY,
    ROUTER_CONFIDENCE_HIGH,
    ROUTER_CONFIDENCE_LOW,
    ROUTER_RESPONSE_KEY_COMPLEXITY,
    ROUTER_RESPONSE_KEY_CONFIDENCE,
    ROUTER_RESPONSE_KEY_DIMENSION,
    ROUTER_RESPONSE_KEY_INTENT_HINT,
    ROUTER_RESPONSE_KEY_LOCAL_HA,
    ROUTER_RESPONSE_KEY_RELEVANT_SENSORS,
    ROUTER_RESPONSE_KEY_SUGGESTED_ORDER,
    ROUTER_RESPONSE_KEY_WEB_SEARCH,
    ROUTER_SKIP_ROUTING_COMPLEXITY,
    SUGGESTED_ORDER_NAME_MAX_LEN,
    VALID_INTENT_HINTS,
    VALID_ROUTER_CONFIDENCE_VALUES,
    VALID_ROUTER_DIMENSIONS,
)

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
        "is_fallback",
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
    is_fallback: bool
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
        is_fallback: bool = False,
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
            is_fallback:           True when this decision was produced by error
                                   fallback (router timed-out / unparseable)
                                   rather than an authentic classification.
                                   Used by the failover engine in
                                   :func:`_check_with_routers`.
        """
        object.__setattr__(self, "local_ha", local_ha)
        object.__setattr__(self, "complexity", complexity)
        object.__setattr__(self, "web_search", web_search)
        object.__setattr__(self, "intent_hint", intent_hint)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "relevant_sensors", relevant_sensors)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "suggested_agent_order", suggested_agent_order)
        object.__setattr__(self, "is_fallback", is_fallback)

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
            f"suggested_agent_order={self.suggested_agent_order!r}, "
            f"is_fallback={self.is_fallback!r})"
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
