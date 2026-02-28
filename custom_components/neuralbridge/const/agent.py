"""Per-agent configuration constants for NeuralBridge.

Covers: agent enable/disable, system prompts, assist mode,
complexity filters, retry/back-off, circuit breaker, response
cache, and the cannot-answer sentinel token.
"""

from __future__ import annotations

from typing import Final

from ..prompts_loader import load_prompt

# ── Routing agent designation ─────────────────────────────────────────────────
CONF_IS_ROUTER: Final = "is_router"
DEFAULT_IS_ROUTER: Final = False

# ── Agent enable / disable ────────────────────────────────────────────────────
CONF_AGENT_ENABLED: Final = "enabled"
DEFAULT_AGENT_ENABLED: Final = True

# ── Per-agent system prompt (Ollama direct only) ──────────────────────────────
CONF_SYSTEM_PROMPT: Final = "system_prompt"
DEFAULT_SYSTEM_PROMPT: Final = ""

# ── Global default prompt ─────────────────────────────────────────────────────
# Used as fallback when an Ollama agent has no per-agent system prompt.
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

# ── Assist mode (LOCAL_HA agents only) ────────────────────────────────────────
# True  — only accept action_done responses (device control only).
# False — accept any non-empty HA conversation response.
# Defaulted to True for LOCAL_HA at config time; stored per-agent.
CONF_AGENT_ASSIST_MODE: Final = "assist_mode"
DEFAULT_AGENT_ASSIST_MODE: Final = False

# ── Per-agent complexity filter ───────────────────────────────────────────────
# Router assigns a complexity score 1-100; agents can declare a range they handle.
CONF_AGENT_MIN_COMPLEXITY: Final = "agent_min_complexity"
CONF_AGENT_MAX_COMPLEXITY: Final = "agent_max_complexity"
DEFAULT_AGENT_MIN_COMPLEXITY: Final = 1
DEFAULT_AGENT_MAX_COMPLEXITY: Final = 100

# ── Retry / exponential back-off ──────────────────────────────────────────────
CONF_MAX_RETRIES: Final = "max_retries"
DEFAULT_MAX_RETRIES: Final = 2
CONF_RETRY_BASE_DELAY: Final = "retry_base_delay"
DEFAULT_RETRY_BASE_DELAY: Final = 1.0  # seconds

# ── Circuit breaker ───────────────────────────────────────────────────────────
DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final = 3
DEFAULT_CIRCUIT_BREAKER_COOLDOWN: Final = 60  # seconds

# ── Per-agent response cache override ────────────────────────────────────────
CONF_AGENT_CACHE_ENABLED: Final = "agent_cache_enabled"
DEFAULT_AGENT_CACHE_ENABLED: Final = True

# ── Cannot-answer sentinel ────────────────────────────────────────────────────
# Injected into Ollama system prompts so the model returns a predictable token
# when it cannot answer, triggering re-routing.  Keep this short so even a 0.6B
# model follows the instruction reliably.
CANNOT_ANSWER_SENTINEL: Final = "CANNOT_ANSWER"
CANNOT_ANSWER_INSTRUCTION: Final = (
    f"If you cannot answer this question, respond only with: {CANNOT_ANSWER_SENTINEL}"
)
