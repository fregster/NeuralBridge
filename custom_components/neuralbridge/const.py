"""Constants for the NeuralBridge integration."""

from __future__ import annotations

from typing import Final

from .prompts_loader import load_prompt

DOMAIN: Final = "neuralbridge"

# Language / i18n
CONF_LANGUAGE: Final = "language"
DEFAULT_LANGUAGE: Final = "en_gb"

# Storage keys
STORAGE_VERSION: Final = 1
STORAGE_KEY: Final = f"{DOMAIN}.agents"

# Configuration keys
CONF_AGENTS: Final = "agents"
CONF_AGENT_TYPE: Final = "agent_type"
CONF_AGENT_NAME: Final = "agent_name"
CONF_PRIORITY: Final = "priority"
CONF_OLLAMA_URL: Final = "ollama_url"
CONF_OLLAMA_MODEL: Final = "ollama_model"
CONF_ENTITY_ID: Final = "entity_id"
CONF_TIMEOUT: Final = "timeout"

# ─────────────────────────────────────────────────────────────────────────────
# Agent types
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CRITICAL: "Ollama" appears in two completely different contexts here.  │
# │                                                                         │
# │  The HA Ollama integration (Settings → Devices & Services → Ollama)     │
# │  registers a NATIVE HA conversation entity.  NeuralBridge routes to it  │
# │  via HA's conversation service exactly like any other HA-registered LLM. │
# │  This is an INTEGRATED ("existing_integration") agent — the user selects  │
# │  the entity_id in the config UI.  This is the path taken by the vast     │
# │  majority (~99%) of Home Assistant users who run Ollama.                 │
# │                                                                         │
# │  AGENT_TYPE_OLLAMA ("ollama") is a completely separate, direct HTTP     │
# │  connection that bypasses HA entirely.  It is a niche, low-usage path  │
# │  for advanced setups where NeuralBridge manages the Ollama connection   │
# │  itself.  Do NOT default to this type when a user says "Ollama" —      │
# │  assume they are using the HA Ollama integration (INTEGRATED type)      │
# │  unless they explicitly state otherwise.                                │
# └─────────────────────────────────────────────────────────────────────────┘
#
# LOCAL_HA ("home_assistant"):
#   Routes to the HA built-in intent processor, hardcoded to the entity
#   "conversation.home_assistant".  This is the DEVICE CONTROL path — it handles
#   voice commands such as "turn the lights off" and returns
#   response_type="action_done" when a device intent is matched.
#   The `assist_mode` flag controls the response acceptance policy:
#     True  — only accepts action_done; all other responses fall through to the
#             next lower-priority agent (device control only).
#     False — accepts any non-empty response (device control + free-text LLM).
#   Note: this type does NOT allow selecting a different HA entity.  If you want
#   to route to a different HA conversation entity (Gemini, HA Ollama integration,
#   ChatGPT, etc.) use INTEGRATED instead.
#
# INTEGRATED ("existing_integration"):
#   Routes to any HA conversation entity that the user selects — the HA Ollama
#   integration, Gemini, OpenAI Conversation, Claude, or any other HA-registered
#   LLM.  Same HA conversation service transport as LOCAL_HA but the entity_id
#   is user-configurable, not hardcoded.  This is the path for cloud/LLM agents
#   used for Q&A and general knowledge.  No assist_mode filter (any non-empty
#   response accepted).  Receives verbosity hints prepended to the query.
#   NOTE: the stored config value remains "existing_integration" for backwards
#   compatibility with existing users' config entries.
#   used for Q&A and general knowledge — there is no assist_mode filter (any
#   non-empty response is accepted).  Verbosity hints are prepended to the query.
#
# OLLAMA ("ollama"):
#   ⚠ DIRECT HTTP connection to an Ollama server — NOT the HA Ollama integration.
#   NeuralBridge calls the Ollama REST API at the configured URL independently of
#   HA.  NeuralBridge injects its own system prompt and controls model parameters.
#   LOW-USAGE PATH — only use when Ollama is not configured as a HA integration.
#   Most deployments should use LOCAL_HA or EXISTING instead.
#
# WEB_SEARCH ("web_search"):
#   DIRECT API query to an external search/AI service (e.g. Brave Search, Brave AI).
#   Zero HA involvement — raw HTTP calls made by NeuralBridge's WebSearchClient.
# ─────────────────────────────────────────────────────────────────────────────
AGENT_TYPE_OLLAMA: Final = "ollama"
AGENT_TYPE_INTEGRATED: Final = "existing_integration"
AGENT_TYPE_LOCAL_HA: Final = "home_assistant"
AGENT_TYPE_WEB_SEARCH: Final = "web_search"

# Default values
DEFAULT_PRIORITY: Final = 50
DEFAULT_TIMEOUT: Final = 5
DEFAULT_OLLAMA_URL: Final = "http://localhost:11434"

# Priority ranges
PRIORITY_MIN: Final = 0
PRIORITY_MAX: Final = 100
PRIORITY_ROUTER: Final = 0  # Reserved for routing/filter agents (TinyLlama, Qwen)
PRIORITY_MIN_PROCESSING: Final = 1  # Minimum allowed priority for processing agents (UI slider)

# Routing agent — first-class designation
CONF_IS_ROUTER: Final = "is_router"
DEFAULT_IS_ROUTER: Final = False

# Logging messages
MSG_AGENT_SUCCESS: Final = "Agent %s (priority %d) handled the request"
MSG_AGENT_FAILED: Final = "Agent %s (priority %d) failed: %s"
MSG_ALL_AGENTS_FAILED: Final = "All agents failed to process the request"
MSG_NO_AGENTS_CONFIGURED: Final = "No agents configured"

# Response messages
FALLBACK_RESPONSE: Final = (
    "I'm having trouble connecting to my AI agents right now. "
    "Try rephrasing your request or check your agent settings."
)
NO_AGENTS_RESPONSE: Final = (
    "No AI agents are configured. Please add agents in the integration settings."
)

# Guard rail configuration
CONF_GUARD_RAIL_ENABLED: Final = "guard_rail_enabled"
CONF_GUARD_RAIL_AGENT_ID: Final = "guard_rail_agent_id"
CONF_GUARD_RAIL_ACTION: Final = "guard_rail_action"
CONF_GUARD_RAIL_RULES: Final = "guard_rail_rules"
CONF_GUARD_RAIL_AI_THRESHOLD: Final = "guard_rail_ai_threshold"
CONF_GUARD_RAIL_ENABLED_FOR_AGENT: Final = "guard_rail_enabled_for_agent"

# Guard rail actions
GUARD_RAIL_ACTION_BLOCK: Final = "block"
GUARD_RAIL_ACTION_WARN: Final = "warn"
GUARD_RAIL_ACTION_NOTIFY_ASK: Final = "notify_ask"

# Guard rail defaults
DEFAULT_GUARD_RAIL_ENABLED: Final = False
DEFAULT_GUARD_RAIL_ACTION: Final = GUARD_RAIL_ACTION_NOTIFY_ASK
DEFAULT_GUARD_RAIL_AI_THRESHOLD: Final = 0.7
DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT: Final = True

# Guard rail — detoxify ML model (optional, opt-in)
CONF_GUARD_RAIL_USE_DETOXIFY: Final = "guard_rail_use_detoxify"
CONF_GUARD_RAIL_DETOXIFY_THRESHOLD: Final = "guard_rail_detoxify_threshold"
DEFAULT_GUARD_RAIL_USE_DETOXIFY: Final = False
DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD: Final = 0.7

# Guard rail messages
GUARD_RAIL_BLOCKED_RESPONSE: Final = (
    "I cannot process this request as it may contain harmful content."
)
GUARD_RAIL_WARNING_PREFIX: Final = "⚠️ Warning: This content may be sensitive. "
GUARD_RAIL_NOTIFY_ASK_PROMPT: Final = (
    "This response may contain harmful or sensitive information. Would you like to continue?"
)

# Guard rail rule categories
GUARD_RAIL_CATEGORY_HARMFUL: Final = "harmful"
GUARD_RAIL_CATEGORY_PRIVACY: Final = "privacy"
GUARD_RAIL_CATEGORY_SECURITY: Final = "security"
GUARD_RAIL_CATEGORY_INAPPROPRIATE: Final = "inappropriate"

# Guard rail event
EVENT_GUARD_RAIL_TRIGGERED: Final = f"{DOMAIN}_guard_rail_triggered"

# Broadcast announcement (Feature 11 — Announce Media Players)
CONF_ANNOUNCE_MEDIA_PLAYERS: Final = "announce_media_players"
DEFAULT_ANNOUNCE_MEDIA_PLAYERS: Final[list[str]] = []
EVENT_ANNOUNCE_SENT: Final = f"{DOMAIN}_announce_sent"

# High-stakes action confirmation (Feature 4)
CONF_HIGH_STAKES_ENABLED: Final = "high_stakes_enabled"
CONF_HIGH_STAKES_DOMAINS: Final = "high_stakes_domains"
CONF_HIGH_STAKES_SECRET_ENABLED: Final = "high_stakes_secret_enabled"  # noqa: S105
CONF_HIGH_STAKES_SECRET: Final = "high_stakes_secret"  # noqa: S105
DEFAULT_HIGH_STAKES_ENABLED: Final = False
DEFAULT_HIGH_STAKES_DOMAINS: Final[list[str]] = [
    "lock",
    "alarm_control_panel",
    "cover",
    "garage_door",
]
DEFAULT_HIGH_STAKES_SECRET_ENABLED: Final = False
DEFAULT_HIGH_STAKES_SECRET: Final = ""
EVENT_HIGH_STAKES_TRIGGERED: Final = f"{DOMAIN}_high_stakes_triggered"

# Circuit breaker
DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final = 3
DEFAULT_CIRCUIT_BREAKER_COOLDOWN: Final = 60  # seconds

# Agent enable/disable
CONF_AGENT_ENABLED: Final = "enabled"
DEFAULT_AGENT_ENABLED: Final = True

# Per-agent system prompt (Ollama only)
CONF_SYSTEM_PROMPT: Final = "system_prompt"
DEFAULT_SYSTEM_PROMPT: Final = ""

# Global default prompt — used as fallback when an Ollama agent has no per-agent system prompt.
# Edit custom_components/neuralbridge/prompts/default_agent.txt to customise.
CONF_DEFAULT_PROMPT: Final = "default_prompt"
DEFAULT_DEFAULT_PROMPT: Final = load_prompt(
    "default_agent.txt",
    fallback=(
        "You are a voice assistant for Home Assistant.\n"
        "Answer questions about the world truthfully.\n"
        "Answer in the style of a witty British butler, answer only in plain text; "
        "keep it simple, to the point, and avoid swearing.\n\n"
        "Answer with time in 24-hour format and state the current timezone "
        "(For example British Summer Time or GMT).\n\n"
        "When saying a date use the format day month year eg 5th of January 2025."
    ),
)

# Response cache (global)
CONF_RESPONSE_CACHE_ENABLED: Final = "response_cache_enabled"
CONF_RESPONSE_CACHE_TTL: Final = "response_cache_ttl"
DEFAULT_RESPONSE_CACHE_ENABLED: Final = True
DEFAULT_RESPONSE_CACHE_TTL: Final = 300  # seconds

# Semantic/normalised cache keying (Feature 9)
CONF_RESPONSE_CACHE_SEMANTIC: Final = "response_cache_semantic"
DEFAULT_RESPONSE_CACHE_SEMANTIC: Final = False
CONF_SEMANTIC_CACHE_TTL: Final = "semantic_cache_ttl"
DEFAULT_SEMANTIC_CACHE_TTL: Final = 60  # seconds (shorter due to fuzzier key)

# Response cache (per-agent override)
CONF_AGENT_CACHE_ENABLED: Final = "agent_cache_enabled"
DEFAULT_AGENT_CACHE_ENABLED: Final = True

# Statistics dispatcher signal (format with entry_id)
SIGNAL_STATS_UPDATED: Final = f"{DOMAIN}_stats_updated_{{entry_id}}"

# hass.data sub-keys
DATA_STATISTICS: Final = "statistics"
DATA_RESPONSE_CACHE: Final = "response_cache"
DATA_SESSION_MEMORY: Final = "session_memory"
DATA_CIRCUIT_BREAKER: Final = "circuit_breaker"
DATA_ENTITY_CONTEXT: Final = "entity_context_cache"

# Service names
SERVICE_CLEAR_CONVERSATION: Final = "clear_conversation"

# Home control / Assist
CONF_ENABLE_HOME_CONTROL: Final = "enable_home_control"
DEFAULT_ENABLE_HOME_CONTROL: Final = True

# Per-agent assist mode (LOCAL_HA only): fall through on non-intent responses
CONF_AGENT_ASSIST_MODE: Final = "assist_mode"
DEFAULT_AGENT_ASSIST_MODE: Final = False  # Defaulted to True for LOCAL_HA at config time

# Per-agent complexity filter (Feature 7 — Multi-User / Speaker Profile Routing)
# Router assigns a complexity score 1-100; agents can declare a range they handle.
CONF_AGENT_MIN_COMPLEXITY: Final = "agent_min_complexity"
CONF_AGENT_MAX_COMPLEXITY: Final = "agent_max_complexity"
DEFAULT_AGENT_MIN_COMPLEXITY: Final = 1
DEFAULT_AGENT_MAX_COMPLEXITY: Final = 100

# Retry / exponential back-off
CONF_MAX_RETRIES: Final = "max_retries"
DEFAULT_MAX_RETRIES: Final = 2
CONF_RETRY_BASE_DELAY: Final = "retry_base_delay"
DEFAULT_RETRY_BASE_DELAY: Final = 1.0  # seconds

# Router agent JSON classification prompt
# Sent to router agents (is_router=True, or priority == 0) to classify and route
# incoming requests. Supports Ollama backends and HA conversation agents.
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
        '  Home control: {"local_ha": true, "web_search": false, "complexity": 5, "intent_hint": null, "confidence": "high"}\n'
        '  Timer: {"local_ha": true, "web_search": false, "complexity": 5, "intent_hint": "timer", "confidence": "high"}\n'
        '  To-do: {"local_ha": true, "web_search": false, "complexity": 5, "intent_hint": "todo", "confidence": "high"}\n'
        '  General knowledge: {"local_ha": false, "web_search": false, "complexity": 30, "intent_hint": null, "confidence": "high"}\n'
        '  Current news/data: {"local_ha": false, "web_search": true, "complexity": 40, "intent_hint": null, "confidence": "high"}\n'
        '  Ambiguous: {"local_ha": false, "web_search": false, "complexity": 50, "intent_hint": null, "confidence": "low"}\n'
        '  Block: {"local_ha": false, "web_search": false, "complexity": 0, "intent_hint": null, "confidence": "high"}\n\n'
        "User message: {user_text}"
    ),
)

# JSON response key names expected in a router agent's classification response
ROUTER_RESPONSE_KEY_LOCAL_HA: Final = "local_ha"
ROUTER_RESPONSE_KEY_COMPLEXITY: Final = "complexity"
ROUTER_RESPONSE_KEY_INTENT_HINT: Final = "intent_hint"

# Valid intent_hint values the router may return (Feature 8)
VALID_INTENT_HINTS: Final = frozenset({"timer", "reminder", "shopping_list", "announce", "todo"})

# Default complexity assumed when a router agent errors or returns unparseable output.
# Mid-range so neither purely simple nor purely complex agents are excluded.
DEFAULT_ROUTER_COMPLEXITY: Final = 50

# Dedicated timeout for routing decisions (separate from processing-agent timeout).
# Routing decisions should be fast; a shorter cap avoids stalling the pipeline.
CONF_ROUTER_TIMEOUT: Final = "router_timeout"
DEFAULT_ROUTER_TIMEOUT: Final = 5  # seconds

# Router agent log levels — controls how much routing detail appears in the logs.
CONF_ROUTER_LOG_LEVEL: Final = "router_log_level"
ROUTER_LOG_LEVEL_NONE: Final = "none"
ROUTER_LOG_LEVEL_COMPLEXITY: Final = "complexity_only"
ROUTER_LOG_LEVEL_DEBUG: Final = "debug_info"
ROUTER_LOG_LEVEL_DEBUG_QUERY: Final = "debug_with_query"  # ⚠ logs PII (query text)
DEFAULT_ROUTER_LOG_LEVEL: Final = ROUTER_LOG_LEVEL_NONE

# Router agent fallback behaviour — what to do when the router errors or times out.
CONF_ROUTER_FALLBACK: Final = "router_fallback"
ROUTER_FALLBACK_DEFAULT_COMPLEXITY: Final = "default_complexity"  # fail-open with score 50
ROUTER_FALLBACK_SKIP_ROUTING: Final = "skip_routing"  # skip routing, try all agents
ROUTER_FALLBACK_BLOCK: Final = "block"  # block the request
DEFAULT_ROUTER_FALLBACK: Final = ROUTER_FALLBACK_DEFAULT_COMPLEXITY

# Custom classification prompt override — stored per routing agent.
# Empty string means use the built-in ROUTER_CLASSIFICATION_PROMPT.
CONF_ROUTER_CUSTOM_PROMPT: Final = "router_custom_prompt"
DEFAULT_ROUTER_CUSTOM_PROMPT: Final = ""

# Internal sentinel complexity value used by _apply_router_decision to signal
# "skip routing entirely — pass all processing agents through unchanged".
ROUTER_SKIP_ROUTING_COMPLEXITY: Final = -1

# Web search provider keys
SEARCH_PROVIDER_BRAVE: Final = "brave"
SEARCH_PROVIDER_BRAVE_ANSWERS: Final = "brave_answers"
SEARCH_PROVIDER_BRAVE_COMBINED: Final = "brave_combined"
SEARCH_PROVIDERS: Final[list[str]] = [
    SEARCH_PROVIDER_BRAVE,
    SEARCH_PROVIDER_BRAVE_ANSWERS,
    SEARCH_PROVIDER_BRAVE_COMBINED,
]

# Web search agent configuration keys
CONF_SEARCH_PROVIDER: Final = "search_provider"
CONF_SEARCH_API_KEY: Final = "search_api_key"
CONF_SEARCH_ANSWERS_API_KEY: Final = "search_answers_api_key"
CONF_SEARCH_RESULT_COUNT: Final = "search_result_count"
CONF_SEARCH_MAX_SNIPPET_LEN: Final = "search_max_snippet_len"

# Web search defaults
DEFAULT_SEARCH_RESULT_COUNT: Final = 5
DEFAULT_SEARCH_MAX_SNIPPET_LEN: Final = 200
DEFAULT_SEARCH_TIMEOUT: Final = 15  # seconds — network round-trip is slower than local LLM

# Router response key for web search routing
ROUTER_RESPONSE_KEY_WEB_SEARCH: Final = "web_search"

# Feature 14b — Router confidence field
ROUTER_RESPONSE_KEY_CONFIDENCE: Final = "confidence"
ROUTER_CONFIDENCE_HIGH: Final = "high"
ROUTER_CONFIDENCE_LOW: Final = "low"
VALID_ROUTER_CONFIDENCE_VALUES: Final = frozenset({"high", "low"})

# Router response key for relevant sensor entity IDs
# The router returns a list of entity IDs it used to answer the query so that
# NeuralBridge can inject only those live values into the answering agent.
ROUTER_RESPONSE_KEY_RELEVANT_SENSORS: Final = "relevant_sensors"

# Feature 10 — Language passthrough: force Ollama to respond in the user's language
CONF_FORCE_RESPONSE_LANGUAGE: Final = "force_response_language"
DEFAULT_FORCE_RESPONSE_LANGUAGE: Final = True

# Feature 2 — Compound Command Splitting
CONF_SPLIT_COMPOUND_COMMANDS: Final = "split_compound_commands"
DEFAULT_SPLIT_COMPOUND_COMMANDS: Final = False
MAX_COMPOUND_FRAGMENTS: Final = 3
COMPOUND_COMMAND_SEPARATOR: Final = " · "

# Feature 6 — Response Verbosity
CONF_RESPONSE_VERBOSITY: Final = "response_verbosity"
VERBOSITY_BRIEF: Final = "brief"
VERBOSITY_NORMAL: Final = "normal"
VERBOSITY_VERBOSE: Final = "verbose"
DEFAULT_RESPONSE_VERBOSITY: Final = VERBOSITY_NORMAL
VERBOSITY_INSTRUCTION_BRIEF: Final = (
    "Respond with the shortest possible acknowledgement — one to five words."
)
VERBOSITY_INSTRUCTION_VERBOSE: Final = "Give detailed, explanatory responses."

# Cannot-answer sentinel — injected into Ollama system prompts so the model
# returns a predictable token when it cannot answer, triggering re-routing.
# Keep this short so even a 0.6B model follows the instruction reliably.
CANNOT_ANSWER_SENTINEL: Final = "CANNOT_ANSWER"
CANNOT_ANSWER_INSTRUCTION: Final = (
    f"If you cannot answer this question, respond only with: {CANNOT_ANSWER_SENTINEL}"
)

# ---------------------------------------------------------------------------
# Feature 14 — Agent Benchmark Profiling
# ---------------------------------------------------------------------------

# NOTE: CONF_AGENT_RE_BENCHMARK is intentionally absent — re_benchmark_on_save
# is owned exclusively by BenchmarkProfile in HA storage, never by the agent
# config dict.  See config flow section for rationale.

# Probe suite versioning — increment this integer whenever any probe in
# benchmark_probes.py is added, modified, or removed.  All stored profiles
# whose probe_suite_version does not match will be automatically invalidated
# and re-queued on next HA start.  Never decrement.
BENCHMARK_PROBE_SUITE_VERSION: Final = 2

CONF_BENCHMARK_WARM_UP_DELAY: Final = "benchmark_warm_up_delay"
DEFAULT_BENCHMARK_WARM_UP_DELAY: Final = 60  # seconds
BENCHMARK_STORAGE_KEY: Final = f"{DOMAIN}.benchmark"
BENCHMARK_STORAGE_VERSION: Final = 1
EVENT_BENCHMARK_STARTED: Final = f"{DOMAIN}_benchmark_started"
EVENT_BENCHMARK_COMPLETE: Final = f"{DOMAIN}_benchmark_complete"
EVENT_BENCHMARK_FAILED: Final = f"{DOMAIN}_benchmark_failed"
SERVICE_RUN_BENCHMARK: Final = "run_benchmark"
DATA_BENCHMARKER: Final = "benchmarker"

# Per-dimension strategy config keys (flat — one per dimension, stored in router agent config)
CONF_STRATEGY_REASONING: Final = "strategy_reasoning"
CONF_STRATEGY_INSTRUCTION_FOLLOWING: Final = "strategy_instruction_following"
CONF_STRATEGY_SMART_HOME_INTENT: Final = "strategy_smart_home_intent"
CONF_STRATEGY_FACTUAL: Final = "strategy_factual"
CONF_STRATEGY_MEMORY: Final = "strategy_memory"

# Convenience mapping: dimension name → its CONF_STRATEGY_* constant
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

# Feature 14d — Router JSON response keys for dimension-aware routing
ROUTER_RESPONSE_KEY_DIMENSION: Final = "dimension"
ROUTER_RESPONSE_KEY_SUGGESTED_ORDER: Final = "suggested_agent_order"
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

# ---------------------------------------------------------------------------
# Feature 15 — Adaptive Preference Learning (APL)
# ---------------------------------------------------------------------------

# Toggle — master on/off switch for preference learning
CONF_ADAPTIVE_LEARNING_ENABLED: Final = "adaptive_learning_enabled"
DEFAULT_ADAPTIVE_LEARNING_ENABLED: Final = True

# Max stored preference entries (admin-configurable, range 1-500)
CONF_PREFERENCE_MAX_ENTRIES: Final = "preference_max_entries"
DEFAULT_PREFERENCE_MAX_ENTRIES: Final = 25
PREFERENCE_MAX_ENTRIES_MIN: Final = 1
PREFERENCE_MAX_ENTRIES_HARD_LIMIT: Final = 500
# Performance warning threshold — shown in UI when max > this value
PREFERENCE_MAX_ENTRIES_WARNING_THRESHOLD: Final = 50

# Cooldown between repeated suggestions for the same preference key
CONF_PREFERENCE_SUGGESTION_COOLDOWN_DAYS: Final = "preference_suggestion_cooldown_days"
DEFAULT_PREFERENCE_SUGGESTION_COOLDOWN_DAYS: Final = 7

# Whether emphatic corrections (all-caps, "I meant") are auto-stored
CONF_PREFERENCE_AUTO_CONFIRM_CORRECTIONS: Final = "preference_auto_confirm_corrections"
DEFAULT_PREFERENCE_AUTO_CONFIRM_CORRECTIONS: Final = True

# hass.data sub-key
DATA_PREFERENCE_MEMORY: Final = "preference_memory"

# HA service names
SERVICE_CLEAR_PREFERENCES: Final = "clear_preferences"
SERVICE_MANAGE_PREFERENCE: Final = "manage_preference"

# HA event fired after a preference is confirmed/auto-stored
EVENT_PREFERENCE_LEARNED: Final = f"{DOMAIN}_preference_learned"

# Dispatcher signal fired after any preference write (used by the sensor)
SIGNAL_PREFERENCES_UPDATED: Final = f"{DOMAIN}_preferences_updated_{{entry_id}}"

# Confirmation keywords that resolve a pending preference suggestion
PREFERENCE_CONFIRM_WORDS: Final = frozenset({"yes", "yep", "yeah", "sure", "always"})
PREFERENCE_REJECT_WORDS: Final = frozenset({"no", "nope", "never", "don't"})
