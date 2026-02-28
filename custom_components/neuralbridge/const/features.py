"""Feature-specific constants for NeuralBridge.

Covers: broadcast/announce (Feature 11), high-stakes action
confirmation (Feature 4), home control toggle, global response cache
(Feature 9 — semantic keying), response verbosity (Feature 6),
language passthrough (Feature 10), and compound command splitting
(Feature 2).
"""

from __future__ import annotations

from typing import Final

from .core import DOMAIN

# ── Broadcast / announce media players (Feature 11) ──────────────────────────
CONF_ANNOUNCE_MEDIA_PLAYERS: Final = "announce_media_players"
DEFAULT_ANNOUNCE_MEDIA_PLAYERS: Final[list[str]] = []
EVENT_ANNOUNCE_SENT: Final = f"{DOMAIN}_announce_sent"

# ── High-stakes action confirmation (Feature 4) ───────────────────────────────
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

# ── Home control / Assist toggle ──────────────────────────────────────────────
CONF_ENABLE_HOME_CONTROL: Final = "enable_home_control"
DEFAULT_ENABLE_HOME_CONTROL: Final = True

# ── Global response cache ─────────────────────────────────────────────────────
CONF_RESPONSE_CACHE_ENABLED: Final = "response_cache_enabled"
CONF_RESPONSE_CACHE_TTL: Final = "response_cache_ttl"
DEFAULT_RESPONSE_CACHE_ENABLED: Final = True
DEFAULT_RESPONSE_CACHE_TTL: Final = 300  # seconds

# ── Semantic / normalised cache keying (Feature 9) ───────────────────────────
CONF_RESPONSE_CACHE_SEMANTIC: Final = "response_cache_semantic"
DEFAULT_RESPONSE_CACHE_SEMANTIC: Final = False
CONF_SEMANTIC_CACHE_TTL: Final = "semantic_cache_ttl"
DEFAULT_SEMANTIC_CACHE_TTL: Final = 60  # seconds (shorter due to fuzzier key)

# ── Response verbosity (Feature 6) ───────────────────────────────────────────
CONF_RESPONSE_VERBOSITY: Final = "response_verbosity"
VERBOSITY_BRIEF: Final = "brief"
VERBOSITY_NORMAL: Final = "normal"
VERBOSITY_VERBOSE: Final = "verbose"
DEFAULT_RESPONSE_VERBOSITY: Final = VERBOSITY_NORMAL
VERBOSITY_INSTRUCTION_BRIEF: Final = (
    "Respond with the shortest possible acknowledgement — one to five words."
)
VERBOSITY_INSTRUCTION_VERBOSE: Final = "Give detailed, explanatory responses."

# ── Language passthrough (Feature 10) ────────────────────────────────────────
# Force Ollama to respond in the user's language
CONF_FORCE_RESPONSE_LANGUAGE: Final = "force_response_language"
DEFAULT_FORCE_RESPONSE_LANGUAGE: Final = True

# ── Compound command splitting (Feature 2) ────────────────────────────────────
CONF_SPLIT_COMPOUND_COMMANDS: Final = "split_compound_commands"
DEFAULT_SPLIT_COMPOUND_COMMANDS: Final = False
MAX_COMPOUND_FRAGMENTS: Final = 3
COMPOUND_COMMAND_SEPARATOR: Final = " · "
