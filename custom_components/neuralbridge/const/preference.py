"""Adaptive Preference Learning (APL) constants for NeuralBridge (Feature 15).

Covers: master toggle, storage limits, suggestion cooldown, auto-confirm
corrections, hass.data key, HA services, events, dispatcher signal,
and confirmation/rejection keyword sets.
"""

from __future__ import annotations

from typing import Final

from .core import DOMAIN

# ── Master toggle ─────────────────────────────────────────────────────────────
CONF_ADAPTIVE_LEARNING_ENABLED: Final = "adaptive_learning_enabled"
DEFAULT_ADAPTIVE_LEARNING_ENABLED: Final = True

# ── Storage limits ────────────────────────────────────────────────────────────
CONF_PREFERENCE_MAX_ENTRIES: Final = "preference_max_entries"
DEFAULT_PREFERENCE_MAX_ENTRIES: Final = 25
PREFERENCE_MAX_ENTRIES_MIN: Final = 1
PREFERENCE_MAX_ENTRIES_HARD_LIMIT: Final = 500
# Performance warning threshold — shown in UI when max > this value
PREFERENCE_MAX_ENTRIES_WARNING_THRESHOLD: Final = 50

# ── Suggestion cooldown ───────────────────────────────────────────────────────
CONF_PREFERENCE_SUGGESTION_COOLDOWN_DAYS: Final = "preference_suggestion_cooldown_days"
DEFAULT_PREFERENCE_SUGGESTION_COOLDOWN_DAYS: Final = 7

# ── Auto-confirm emphatic corrections ────────────────────────────────────────
CONF_PREFERENCE_AUTO_CONFIRM_CORRECTIONS: Final = "preference_auto_confirm_corrections"
DEFAULT_PREFERENCE_AUTO_CONFIRM_CORRECTIONS: Final = True

# ── hass.data sub-key ─────────────────────────────────────────────────────────
DATA_PREFERENCE_MEMORY: Final = "preference_memory"

# ── HA services ───────────────────────────────────────────────────────────────
SERVICE_CLEAR_PREFERENCES: Final = "clear_preferences"
SERVICE_MANAGE_PREFERENCE: Final = "manage_preference"

# ── HA event ─────────────────────────────────────────────────────────────────
EVENT_PREFERENCE_LEARNED: Final = f"{DOMAIN}_preference_learned"

# ── Dispatcher signal ────────────────────────────────────────────────────────
# Format with entry_id: SIGNAL_PREFERENCES_UPDATED.format(entry_id=entry_id)
SIGNAL_PREFERENCES_UPDATED: Final = f"{DOMAIN}_preferences_updated_{{entry_id}}"

# ── Confirmation / rejection keywords ────────────────────────────────────────
# Keywords that resolve a pending preference suggestion
PREFERENCE_CONFIRM_WORDS: Final = frozenset({"yes", "yep", "yeah", "sure", "always"})
PREFERENCE_REJECT_WORDS: Final = frozenset({"no", "nope", "never", "don't"})
