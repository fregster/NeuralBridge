"""Web search agent constants for NeuralBridge.

Covers: search provider identifiers, configuration keys,
and default values for web search agents.
"""

from __future__ import annotations

from typing import Final

# ── Search provider identifiers ───────────────────────────────────────────────
SEARCH_PROVIDER_BRAVE: Final = "brave"
SEARCH_PROVIDER_BRAVE_ANSWERS: Final = "brave_answers"
SEARCH_PROVIDER_BRAVE_COMBINED: Final = "brave_combined"
SEARCH_PROVIDERS: Final[list[str]] = [
    SEARCH_PROVIDER_BRAVE,
    SEARCH_PROVIDER_BRAVE_ANSWERS,
    SEARCH_PROVIDER_BRAVE_COMBINED,
]

# ── Web search agent configuration keys ──────────────────────────────────────
CONF_SEARCH_PROVIDER: Final = "search_provider"
CONF_SEARCH_API_KEY: Final = "search_api_key"
CONF_SEARCH_ANSWERS_API_KEY: Final = "search_answers_api_key"
CONF_SEARCH_RESULT_COUNT: Final = "search_result_count"
CONF_SEARCH_MAX_SNIPPET_LEN: Final = "search_max_snippet_len"

# ── Web search defaults ───────────────────────────────────────────────────────
DEFAULT_SEARCH_RESULT_COUNT: Final = 5
DEFAULT_SEARCH_MAX_SNIPPET_LEN: Final = 200
DEFAULT_SEARCH_TIMEOUT: Final = 15  # seconds — network round-trip is slower than local LLM
