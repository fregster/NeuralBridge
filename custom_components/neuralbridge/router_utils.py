"""Pure utility functions for the NeuralBridge router pipeline.

Extracted from ``router_engine.py`` as part of Stage 3 god-class refactor.

This module contains stateless helper functions used by :class:`.RouterEngine`
and re-exported through ``router_engine`` for backward compatibility:

* :func:`_is_unhelpful_response` — detect agent refusals / sentinel responses.
* :func:`_deterministic_classify` — pattern-based web-search classification
  that bypasses the LLM router.
* :func:`_extract_announce_text` — extract announcement content from user input.
* :func:`_split_compound_input` — split compound voice commands into fragments.
* :func:`_build_capability_block` — build the ``[Agent capabilities:]`` prompt
  block from benchmark data.
* :func:`_build_strategy_block` — build the ``[Routing preferences:]`` prompt
  block from dimension strategies.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .const import (
    CANNOT_ANSWER_SENTINEL,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    MAX_COMPOUND_FRAGMENTS,
    ROUTER_CONFIDENCE_HIGH,
    ROUTING_STRATEGY_DEFAULT,
)
from .router_decision import RouterDecision

if TYPE_CHECKING:
    from .agent_benchmark import AgentBenchmarker


# ---------------------------------------------------------------------------
# Agent refusal detection
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
    # "I am sorry, I cannot provide…" / "I cannot provide…"
    r"|i (?:am |'m )?(?:sorry[,.]?\s+i |really )?(?:cannot|can't|am unable to|'m unable to)\s+provide"
    # "I'm unable to provide/access/browse/search/look up/find"
    r"|i(?:'m| am) unable to (?:provide|access|browse|search|look up|find|answer)"
    # "I don't have access to / real-time / internet"
    r"|i (?:don't|do not) have (?:access to|the ability to|real.?time|internet)"
    # "My capabilities are limited/restricted"
    r"|my capabilities (?:are )?(?:limited|restricted)"
    # "My purpose is to control…" (HA built-in boilerplate)
    r"|my purpose is to (?:control|manage|help with)"
    # "beyond/outside my capabilities"
    r"|(?:beyond|outside) my (?:current )?capabilities"
    # "I cannot/can't browse/search the internet/web"
    r"|i (?:cannot|can't) (?:browse|search|look up|access) the (?:internet|web)"
    # "not able/designed/trained to browse/search/provide/access/answer"
    r"|not (?:able|designed|trained) to (?:browse|search|provide|access|answer)",
    re.IGNORECASE,
)


def _is_unhelpful_response(text: str) -> bool:
    """Return True when *text* indicates the agent could not answer the question.

    Two detection strategies are applied:

    1. **Sentinel check** — Ollama agents are instructed to reply with the exact
       ``CANNOT_ANSWER`` token when they cannot answer.  Matched
       case-insensitively with optional surrounding whitespace/punctuation.

    2. **Refusal pattern check** — HA built-in and cloud agents emit natural-
       language refusals ("I am sorry, I cannot provide…", "My capabilities are
       limited to…", etc.).  These are matched via ``_REFUSAL_PATTERNS`` but
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


# ---------------------------------------------------------------------------
# Deterministic classification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Announce and compound-command helpers
# ---------------------------------------------------------------------------

# Feature 11 — Broadcast announce trigger phrases
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


def _extract_announce_text(text: str, intent_hint: str | None = None) -> str | None:
    """Extract the announcement message from *text*.

    Checks whether *text* begins with a known trigger phrase (announce,
    broadcast, tell everyone, say on all speakers, …).  If so, returns
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


# ---------------------------------------------------------------------------
# Prompt-block builders
# ---------------------------------------------------------------------------


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
