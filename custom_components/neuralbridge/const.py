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
DEFAULT_TIMEOUT: Final = 30
DEFAULT_OLLAMA_URL: Final = "http://localhost:11434"

# Priority ranges
PRIORITY_MIN: Final = 0
PRIORITY_MAX: Final = 100
PRIORITY_ROUTER: Final = 0  # Reserved for routing/filter agents (TinyLlama, Qwen)

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
    "This response may contain harmful or sensitive information. " "Would you like to continue?"
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

# Service names
SERVICE_CLEAR_CONVERSATION: Final = "clear_conversation"

# Router agent classification prompt
# Sent to priority-0 Ollama agents to classify whether a request should be processed.
# Use .format(user_text=...) when building the final prompt.
ROUTER_CLASSIFICATION_PROMPT: Final = (
    "You are a smart home request classifier.\n"
    "Decide if the following user message should be processed by the AI assistant.\n"
    "Respond with exactly one word — PASS or BLOCK.\n"
    "Respond PASS for normal smart home requests, general questions, and safe queries.\n"
    "Respond BLOCK for harmful, illegal, abusive, or clearly inappropriate requests.\n\n"
    "User message: {user_text}"
)
