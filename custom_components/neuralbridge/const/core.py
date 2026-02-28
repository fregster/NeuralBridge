"""Core / foundational constants for NeuralBridge.

Covers: domain identity, language, storage, basic config keys,
agent types, priority ranges, hass.data sub-keys, dispatcher
signals, service names, and generic log/response messages.
"""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "neuralbridge"

# ── Language / i18n ─────────────────────────────────────────────────────────
CONF_LANGUAGE: Final = "language"
DEFAULT_LANGUAGE: Final = "en_gb"

# ── Storage keys ─────────────────────────────────────────────────────────────
STORAGE_VERSION: Final = 1
STORAGE_KEY: Final = f"{DOMAIN}.agents"

# ── Basic configuration keys ──────────────────────────────────────────────────
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

# ── Default values ────────────────────────────────────────────────────────────
DEFAULT_PRIORITY: Final = 50
DEFAULT_TIMEOUT: Final = 5
DEFAULT_OLLAMA_URL: Final = "http://localhost:11434"

# ── Priority ranges ───────────────────────────────────────────────────────────
PRIORITY_MIN: Final = 0
PRIORITY_MAX: Final = 100
PRIORITY_ROUTER: Final = 0  # Reserved for routing/filter agents (TinyLlama, Qwen)
PRIORITY_MIN_PROCESSING: Final = 1  # Minimum allowed priority for processing agents (UI slider)

# ── hass.data sub-keys ────────────────────────────────────────────────────────
DATA_STATISTICS: Final = "statistics"
DATA_RESPONSE_CACHE: Final = "response_cache"
DATA_SESSION_MEMORY: Final = "session_memory"
DATA_CIRCUIT_BREAKER: Final = "circuit_breaker"
DATA_ENTITY_CONTEXT: Final = "entity_context_cache"

# ── Dispatcher signals ────────────────────────────────────────────────────────
# Format with entry_id: SIGNAL_STATS_UPDATED.format(entry_id=entry_id)
SIGNAL_STATS_UPDATED: Final = f"{DOMAIN}_stats_updated_{{entry_id}}"

# ── Service names ─────────────────────────────────────────────────────────────
SERVICE_CLEAR_CONVERSATION: Final = "clear_conversation"

# ── Logging messages ──────────────────────────────────────────────────────────
MSG_AGENT_SUCCESS: Final = "Agent %s (priority %d) handled the request"
MSG_AGENT_FAILED: Final = "Agent %s (priority %d) failed: %s"
MSG_ALL_AGENTS_FAILED: Final = "All agents failed to process the request"
MSG_NO_AGENTS_CONFIGURED: Final = "No agents configured"

# ── Response messages ─────────────────────────────────────────────────────────
FALLBACK_RESPONSE: Final = (
    "I'm having trouble connecting to my AI agents right now. "
    "Try rephrasing your request or check your agent settings."
)
NO_AGENTS_RESPONSE: Final = (
    "No AI agents are configured. Please add agents in the integration settings."
)
