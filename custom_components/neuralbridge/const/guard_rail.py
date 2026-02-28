"""Guard rail constants for NeuralBridge.

Covers: guard rail enable/disable, action types, detection
thresholds (including optional detoxify ML model), response
messages, rule categories, and the HA event name.
"""

from __future__ import annotations

from typing import Final

from .core import DOMAIN

# ── Guard rail config keys ────────────────────────────────────────────────────
CONF_GUARD_RAIL_ENABLED: Final = "guard_rail_enabled"
CONF_GUARD_RAIL_AGENT_ID: Final = "guard_rail_agent_id"
CONF_GUARD_RAIL_ACTION: Final = "guard_rail_action"
CONF_GUARD_RAIL_RULES: Final = "guard_rail_rules"
CONF_GUARD_RAIL_AI_THRESHOLD: Final = "guard_rail_ai_threshold"
CONF_GUARD_RAIL_ENABLED_FOR_AGENT: Final = "guard_rail_enabled_for_agent"

# ── Guard rail actions ────────────────────────────────────────────────────────
GUARD_RAIL_ACTION_BLOCK: Final = "block"
GUARD_RAIL_ACTION_WARN: Final = "warn"
GUARD_RAIL_ACTION_NOTIFY_ASK: Final = "notify_ask"

# ── Guard rail defaults ───────────────────────────────────────────────────────
DEFAULT_GUARD_RAIL_ENABLED: Final = False
DEFAULT_GUARD_RAIL_ACTION: Final = GUARD_RAIL_ACTION_NOTIFY_ASK
DEFAULT_GUARD_RAIL_AI_THRESHOLD: Final = 0.7
DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT: Final = True

# ── Detoxify ML model (optional, opt-in) ──────────────────────────────────────
CONF_GUARD_RAIL_USE_DETOXIFY: Final = "guard_rail_use_detoxify"
CONF_GUARD_RAIL_DETOXIFY_THRESHOLD: Final = "guard_rail_detoxify_threshold"
DEFAULT_GUARD_RAIL_USE_DETOXIFY: Final = False
DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD: Final = 0.7

# ── Guard rail response messages ──────────────────────────────────────────────
GUARD_RAIL_BLOCKED_RESPONSE: Final = (
    "I cannot process this request as it may contain harmful content."
)
GUARD_RAIL_WARNING_PREFIX: Final = "⚠️ Warning: This content may be sensitive. "
GUARD_RAIL_NOTIFY_ASK_PROMPT: Final = (
    "This response may contain harmful or sensitive information. Would you like to continue?"
)

# ── Guard rail rule categories ────────────────────────────────────────────────
GUARD_RAIL_CATEGORY_HARMFUL: Final = "harmful"
GUARD_RAIL_CATEGORY_PRIVACY: Final = "privacy"
GUARD_RAIL_CATEGORY_SECURITY: Final = "security"
GUARD_RAIL_CATEGORY_INAPPROPRIATE: Final = "inappropriate"

# ── HA event ─────────────────────────────────────────────────────────────────
EVENT_GUARD_RAIL_TRIGGERED: Final = f"{DOMAIN}_guard_rail_triggered"
