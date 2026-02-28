"""Router / classification constants for NeuralBridge.

Covers: classification prompt, response JSON keys, intent hints,
log levels, fallback & failover modes, routing strategies,
dimension-aware routing, confidence values, and related sentinels.
"""

from __future__ import annotations

from typing import Final

from ..prompts_loader import load_prompt

# ── Router classification prompt ──────────────────────────────────────────────
# Sent to router agents (is_router=True, or priority == 0) to classify and route
# incoming requests.  Supports Ollama backends and HA conversation agents.
# Returns a JSON object with local_ha, web_search, complexity, intent_hint, and
# confidence fields.
# Use .replace("{user_text}", user_text) when building the final prompt.
# Edit custom_components/neuralbridge/prompts/router_classification.txt to customise.
ROUTER_CLASSIFICATION_PROMPT: Final = load_prompt(
    "router_classification.txt",
    fallback=(
        "You are a smart home request classifier.\n"
        "Analyse the user message and respond with ONLY a valid JSON object — no other text.\n"
        "The JSON must contain exactly these five fields:\n"
        '  "local_ha": boolean — true if this is a home automation or device control request, '
        "false for general questions or knowledge queries.\n"
        '  "web_search": boolean — true if the answer requires real-time or current information '
        "(e.g. news, current leaders, live sport scores, today's weather, financial data, "
        "recent events). False for static knowledge or home control.\n"
        '  "complexity": integer — 1 (simple/factual) to 100 (complex reasoning). '
        "Use 0 to signal that the request should be BLOCKED.\n"
        '  "intent_hint": string or null — "timer", "reminder", "todo", '
        '"shopping_list", "announce", or null.\n'
        '  "confidence": "high" or "low" — your confidence in the above classification. '
        'Use "low" when the request is ambiguous or unclear.\n'
        "Respond with complexity 0 ONLY for harmful, illegal, abusive, or clearly "
        "inappropriate requests.\n"
        "When web_search is true, local_ha should be false.\n"
        "Examples:\n"
        '  Home control: {"local_ha": true, "web_search": false, "complexity": 5, '
        '"intent_hint": null, "confidence": "high"}\n'
        '  Timer: {"local_ha": true, "web_search": false, "complexity": 5, '
        '"intent_hint": "timer", "confidence": "high"}\n'
        '  To-do: {"local_ha": true, "web_search": false, "complexity": 5, '
        '"intent_hint": "todo", "confidence": "high"}\n'
        '  General knowledge: {"local_ha": false, "web_search": false, "complexity": 30, '
        '"intent_hint": null, "confidence": "high"}\n'
        '  Current news/data: {"local_ha": false, "web_search": true, "complexity": 40, '
        '"intent_hint": null, "confidence": "high"}\n'
        '  Ambiguous: {"local_ha": false, "web_search": false, "complexity": 50, '
        '"intent_hint": null, "confidence": "low"}\n'
        '  Block: {"local_ha": false, "web_search": false, "complexity": 0, '
        '"intent_hint": null, "confidence": "high"}\n\n'
        "User message: {user_text}"
    ),
)

# ── Router response JSON key names ────────────────────────────────────────────
ROUTER_RESPONSE_KEY_LOCAL_HA: Final = "local_ha"
ROUTER_RESPONSE_KEY_COMPLEXITY: Final = "complexity"
ROUTER_RESPONSE_KEY_INTENT_HINT: Final = "intent_hint"
ROUTER_RESPONSE_KEY_WEB_SEARCH: Final = "web_search"
ROUTER_RESPONSE_KEY_CONFIDENCE: Final = "confidence"
ROUTER_RESPONSE_KEY_RELEVANT_SENSORS: Final = "relevant_sensors"
ROUTER_RESPONSE_KEY_DIMENSION: Final = "dimension"
ROUTER_RESPONSE_KEY_SUGGESTED_ORDER: Final = "suggested_agent_order"

# ── Intent hints ──────────────────────────────────────────────────────────────
# Valid intent_hint values the router may return (Feature 8)
VALID_INTENT_HINTS: Final = frozenset({"timer", "reminder", "shopping_list", "announce", "todo"})

# ── Router defaults ───────────────────────────────────────────────────────────
# Default complexity assumed when a router agent errors or returns unparseable output.
# Mid-range so neither purely simple nor purely complex agents are excluded.
DEFAULT_ROUTER_COMPLEXITY: Final = 50

# ── Router timeout ────────────────────────────────────────────────────────────
# Dedicated timeout for routing decisions (separate from processing-agent timeout).
# Routing decisions should be fast; a shorter cap avoids stalling the pipeline.
CONF_ROUTER_TIMEOUT: Final = "router_timeout"
DEFAULT_ROUTER_TIMEOUT: Final = 5  # seconds

# ── Router log levels ─────────────────────────────────────────────────────────
# Controls how much routing detail appears in the logs.
CONF_ROUTER_LOG_LEVEL: Final = "router_log_level"
ROUTER_LOG_LEVEL_NONE: Final = "none"
ROUTER_LOG_LEVEL_COMPLEXITY: Final = "complexity_only"
ROUTER_LOG_LEVEL_DEBUG: Final = "debug_info"
ROUTER_LOG_LEVEL_DEBUG_QUERY: Final = "debug_with_query"  # ⚠ logs PII (query text)
DEFAULT_ROUTER_LOG_LEVEL: Final = ROUTER_LOG_LEVEL_NONE

# ── Router fallback behaviour ─────────────────────────────────────────────────
# What to do when the router errors or times out.
CONF_ROUTER_FALLBACK: Final = "router_fallback"
ROUTER_FALLBACK_DEFAULT_COMPLEXITY: Final = "default_complexity"  # fail-open with score 50
ROUTER_FALLBACK_SKIP_ROUTING: Final = "skip_routing"  # skip routing, try all agents
ROUTER_FALLBACK_BLOCK: Final = "block"  # block the request
DEFAULT_ROUTER_FALLBACK: Final = ROUTER_FALLBACK_DEFAULT_COMPLEXITY

# ── Router failover mode ──────────────────────────────────────────────────────
# Governs how multiple routing agents work together.
# Stored on the PRIMARY (highest-priority) routing agent.
#
# primary_only   — Only the primary router runs; on error its own CONF_ROUTER_FALLBACK applies.
# backup         — Primary runs first.  If it gives a fallback decision (due to error/timeout),
#                  the second-priority router is tried.  Primary blocks are never overridden.
# priority_order — All routers run in priority order; stops at the first authentic (non-fallback)
#                  decision.  If every router errors, the last fallback decision is returned.
CONF_ROUTER_FAILOVER_MODE: Final = "router_failover_mode"
ROUTER_FAILOVER_PRIMARY_ONLY: Final = "primary_only"
ROUTER_FAILOVER_BACKUP: Final = "backup"
ROUTER_FAILOVER_PRIORITY_ORDER: Final = "priority_order"
DEFAULT_ROUTER_FAILOVER_MODE: Final = ROUTER_FAILOVER_PRIMARY_ONLY
VALID_ROUTER_FAILOVER_MODES: Final = frozenset({"primary_only", "backup", "priority_order"})

# ── Custom classification prompt override ─────────────────────────────────────
# Stored per routing agent.  Empty string means use the built-in
# ROUTER_CLASSIFICATION_PROMPT.
CONF_ROUTER_CUSTOM_PROMPT: Final = "router_custom_prompt"
DEFAULT_ROUTER_CUSTOM_PROMPT: Final = ""

# ── Internal routing sentinel ─────────────────────────────────────────────────
# Used by _apply_router_decision to signal "skip routing entirely — pass all
# processing agents through unchanged".
ROUTER_SKIP_ROUTING_COMPLEXITY: Final = -1

# ── Router confidence ─────────────────────────────────────────────────────────
ROUTER_CONFIDENCE_HIGH: Final = "high"
ROUTER_CONFIDENCE_LOW: Final = "low"
VALID_ROUTER_CONFIDENCE_VALUES: Final = frozenset({"high", "low"})

# ── Routing strategies (dimension-aware, Feature 14d) ─────────────────────────
CONF_STRATEGY_REASONING: Final = "strategy_reasoning"
CONF_STRATEGY_INSTRUCTION_FOLLOWING: Final = "strategy_instruction_following"
CONF_STRATEGY_SMART_HOME_INTENT: Final = "strategy_smart_home_intent"
CONF_STRATEGY_FACTUAL: Final = "strategy_factual"
CONF_STRATEGY_MEMORY: Final = "strategy_memory"

# Convenience mapping: dimension name → its CONF_STRATEGY_* constant.
# Used by _classify_with_router to build the dimension_strategies dict at runtime.
# NOT stored directly in the config entry.
CONF_STRATEGY_MAP: Final[dict[str, str]] = {
    "reasoning": CONF_STRATEGY_REASONING,
    "instruction_following": CONF_STRATEGY_INSTRUCTION_FOLLOWING,
    "smart_home_intent": CONF_STRATEGY_SMART_HOME_INTENT,
    "factual": CONF_STRATEGY_FACTUAL,
    "memory": CONF_STRATEGY_MEMORY,
}

# Routing strategy values communicated to the router LLM
ROUTING_STRATEGY_DEFAULT: Final = "default"
ROUTING_STRATEGY_FASTEST: Final = "fastest"
ROUTING_STRATEGY_MOST_CAPABLE: Final = "most_capable"
ROUTING_STRATEGY_LOCAL_ONLY: Final = "local_only"
VALID_ROUTING_STRATEGIES: Final = frozenset(
    {
        "default",
        "fastest",
        "most_capable",
        "local_only",
    }
)

# Valid router dimensions
VALID_ROUTER_DIMENSIONS: Final = frozenset(
    {
        "reasoning",
        "instruction_following",
        "smart_home_intent",
        "factual",
        "memory",
    }
)

# Cap on each agent-name string accepted from suggested_agent_order
# (prevents prompt injection bleed-through)
SUGGESTED_ORDER_NAME_MAX_LEN: Final = 100
