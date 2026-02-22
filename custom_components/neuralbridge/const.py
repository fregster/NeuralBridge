"""Constants for the NeuralBridge integration."""

from __future__ import annotations

from typing import Final

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

# Agent types
AGENT_TYPE_OLLAMA: Final = "ollama"
AGENT_TYPE_EXISTING: Final = "existing_integration"
AGENT_TYPE_LOCAL_HA: Final = "home_assistant"

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
FALLBACK_RESPONSE: Final = "I'm having trouble connecting to my AI agents right now."
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

# Circuit breaker
DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final = 3
DEFAULT_CIRCUIT_BREAKER_COOLDOWN: Final = 60  # seconds

# Agent enable/disable
CONF_AGENT_ENABLED: Final = "enabled"
DEFAULT_AGENT_ENABLED: Final = True

# Per-agent system prompt (Ollama only)
CONF_SYSTEM_PROMPT: Final = "system_prompt"
DEFAULT_SYSTEM_PROMPT: Final = ""

# Global default prompt — used as fallback when an Ollama agent has no per-agent system prompt
CONF_DEFAULT_PROMPT: Final = "default_prompt"
DEFAULT_DEFAULT_PROMPT: Final = (
    "You are a voice assistant for Home Assistant.\n"
    "Answer questions about the world truthfully.\n"
    "Answer in the style of a witty British butler, answer only in plain text; "
    "keep it simple, to the point, and avoid swearing.\n\n"
    "Answer with time in 24-hour format and state the current timezone "
    "(For example British Summer Time or GMT).\n\n"
    "When saying a date use the format day month year eg 5th of January 2025."
)

# Response cache (global)
CONF_RESPONSE_CACHE_ENABLED: Final = "response_cache_enabled"
CONF_RESPONSE_CACHE_TTL: Final = "response_cache_ttl"
DEFAULT_RESPONSE_CACHE_ENABLED: Final = True
DEFAULT_RESPONSE_CACHE_TTL: Final = 300  # seconds

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

# Service names
SERVICE_CLEAR_CONVERSATION: Final = "clear_conversation"

# Home control / Assist
CONF_ENABLE_HOME_CONTROL: Final = "enable_home_control"
DEFAULT_ENABLE_HOME_CONTROL: Final = True

# Per-agent assist mode (LOCAL_HA only): fall through on non-intent responses
CONF_AGENT_ASSIST_MODE: Final = "assist_mode"
DEFAULT_AGENT_ASSIST_MODE: Final = False  # Defaulted to True for LOCAL_HA at config time

# Retry / exponential back-off
CONF_MAX_RETRIES: Final = "max_retries"
DEFAULT_MAX_RETRIES: Final = 2
CONF_RETRY_BASE_DELAY: Final = "retry_base_delay"
DEFAULT_RETRY_BASE_DELAY: Final = 1.0  # seconds

# Router agent JSON classification prompt
# Sent to is_router Ollama agents to classify and route incoming requests.
# Returns a JSON object with local_ha and complexity fields.
# Use .format(user_text=...) when building the final prompt.
ROUTER_CLASSIFICATION_PROMPT: Final = (
    "You are a smart home request classifier.\n"
    "Analyse the user message and respond with ONLY a valid JSON object — no other text.\n"
    "The JSON must contain exactly these two fields:\n"
    '  "local_ha": boolean — true if this is a home automation or device control request, '
    "false for general questions or knowledge queries.\n"
    '  "complexity": integer — 1 (simple/factual) to 100 (complex reasoning). '
    "Use 0 to signal that the request should be BLOCKED.\n"
    "Respond with complexity 0 ONLY for harmful, illegal, abusive, or clearly "
    "inappropriate requests.\n"
    "Examples:\n"
    '  Home control: {"local_ha": true, "complexity": 5}\n'
    '  General question: {"local_ha": false, "complexity": 30}\n'
    '  Block: {"local_ha": false, "complexity": 0}\n\n'
    "User message: {user_text}"
)

# JSON response key names expected in a router agent's classification response
ROUTER_RESPONSE_KEY_LOCAL_HA: Final = "local_ha"
ROUTER_RESPONSE_KEY_COMPLEXITY: Final = "complexity"

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
