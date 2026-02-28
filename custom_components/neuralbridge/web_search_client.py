"""Web search client for NeuralBridge.

Provides a generic provider interface for web search, with Brave Search as the
first concrete implementation.  Additional providers can be added by implementing
the :class:`WebSearchProvider` protocol and registering the provider key in
:data:`SEARCH_PROVIDER_MAP`.

Provider implementations live in the :mod:`providers` sub-package:

* :mod:`providers.brave_search` — Brave Web Search API
* :mod:`providers.brave_answers` — Brave Answers API (direct Q&A)
* :mod:`providers.brave_combined` — Combined Brave provider (Answers + Search fallback)

Security note: API keys must NEVER be logged at any level.
"""

from __future__ import annotations

import logging
from typing import Any

from .const import (
    CONF_SEARCH_ANSWERS_API_KEY,
    CONF_SEARCH_API_KEY,
    CONF_SEARCH_MAX_SNIPPET_LEN,
    CONF_SEARCH_PROVIDER,
    CONF_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_MAX_SNIPPET_LEN,
    DEFAULT_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_TIMEOUT,
    SEARCH_PROVIDER_BRAVE,
    SEARCH_PROVIDER_BRAVE_ANSWERS,
    SEARCH_PROVIDER_BRAVE_COMBINED,
)
from .providers import SearchResult as SearchResult  # noqa: PLC0414, TC001
from .providers import WebSearchProvider as WebSearchProvider  # noqa: PLC0414, TC001
from .providers.brave_answers import BraveAnswersProvider as BraveAnswersProvider  # noqa: PLC0414
from .providers.brave_answers import (
    _parse_brave_answers_response as _parse_brave_answers_response,  # noqa: PLC0414
)
from .providers.brave_combined import (
    BraveAnswersCombinedProvider as BraveAnswersCombinedProvider,  # noqa: PLC0414
)
from .providers.brave_search import BraveSearchProvider as BraveSearchProvider  # noqa: PLC0414
from .providers.brave_search import _parse_brave_response as _parse_brave_response  # noqa: PLC0414

_LOGGER = logging.getLogger(__name__)

# Maximum characters in the formatted summary returned to the user.
_MAX_SUMMARY_LEN = 2000

# ---------------------------------------------------------------------------
# Provider registry — extend this dict to add new providers.
# ---------------------------------------------------------------------------
# Note: BraveAnswersCombinedProvider is NOT in this map because it requires two
# keys and a different constructor.  WebSearchClient.__init__ handles it explicitly.
SEARCH_PROVIDER_MAP: dict[str, type[BraveSearchProvider] | type[BraveAnswersProvider]] = {
    SEARCH_PROVIDER_BRAVE: BraveSearchProvider,
    SEARCH_PROVIDER_BRAVE_ANSWERS: BraveAnswersProvider,
}


class WebSearchClient:
    """Provider-agnostic web search client used by the conversation agent.

    Reads ``CONF_SEARCH_PROVIDER``, ``CONF_SEARCH_API_KEY``,
    ``CONF_SEARCH_RESULT_COUNT``, ``CONF_SEARCH_MAX_SNIPPET_LEN``, and
    ``CONF_SEARCH_TIMEOUT`` from ``agent_config``.

    Usage::

        client = WebSearchClient(agent_config)
        summary = await client.search_and_summarise("Who is the UK prime minister?")
    """

    def __init__(self, agent_config: dict[str, Any]) -> None:
        """Initialise the client from an agent configuration dict.

        Args:
            agent_config: NeuralBridge agent config dict containing provider,
                          API key, result count, and snippet length settings.
                          When ``CONF_SEARCH_PROVIDER`` is ``brave_combined``,
                          both ``CONF_SEARCH_API_KEY`` (Search) and
                          ``CONF_SEARCH_ANSWERS_API_KEY`` (Answers) are required.
        """
        provider_key: str = agent_config.get(CONF_SEARCH_PROVIDER, SEARCH_PROVIDER_BRAVE)
        api_key: str = agent_config.get(CONF_SEARCH_API_KEY, "")
        answers_api_key: str = agent_config.get(CONF_SEARCH_ANSWERS_API_KEY, "")
        timeout: int = int(agent_config.get("timeout", DEFAULT_SEARCH_TIMEOUT))
        self._result_count: int = int(
            agent_config.get(CONF_SEARCH_RESULT_COUNT, DEFAULT_SEARCH_RESULT_COUNT)
        )
        self._max_snippet_len: int = int(
            agent_config.get(CONF_SEARCH_MAX_SNIPPET_LEN, DEFAULT_SEARCH_MAX_SNIPPET_LEN)
        )

        if provider_key == SEARCH_PROVIDER_BRAVE_COMBINED:
            self._provider: WebSearchProvider = BraveAnswersCombinedProvider(
                answers_api_key=answers_api_key,
                search_api_key=api_key,
                timeout=timeout,
            )
            return

        if provider_key == SEARCH_PROVIDER_BRAVE_ANSWERS:
            self._provider = BraveAnswersProvider(api_key=answers_api_key, timeout=timeout)
            return

        provider_cls = SEARCH_PROVIDER_MAP.get(provider_key)
        if provider_cls is None:
            _LOGGER.warning("Unknown search provider '%s' — falling back to Brave", provider_key)
            provider_cls = BraveSearchProvider

        self._provider = provider_cls(api_key=api_key, timeout=timeout)

    async def search_and_summarise(self, query: str) -> str | None:
        """Search for ``query`` and return a formatted plain-text summary.

        Args:
            query: The user's question or search string.

        Returns:
            A human-readable summary string ready for speech output, or None
            if no results were returned by the provider.
        """
        results = await self._provider.search(query, self._result_count)
        if not results:
            _LOGGER.warning("Web search returned no results for query (not logged for privacy)")
            return None

        return _format_results(results, self._max_snippet_len)

    async def close(self) -> None:
        """Close the underlying provider session(s).

        Safe to call even when the provider does not expose ``close``.
        """
        if hasattr(self._provider, "close"):
            await self._provider.close()


def _format_results(results: list[SearchResult], max_snippet_len: int) -> str:
    """Render a list of SearchResults as a concise plain-text summary.

    **Direct-answer shortcut:** when the list contains exactly one result whose
    URL is empty (the signature of a ``BraveAnswersProvider`` result), the
    snippet is returned as-is without the preamble, for more natural TTS output.

    Args:
        results: Non-empty list of SearchResult objects.
        max_snippet_len: Maximum characters per snippet before truncation.

    Returns:
        Multi-line plain-text string, truncated to ``_MAX_SUMMARY_LEN``.
    """
    if len(results) == 1 and not results[0].url:
        snippet = results[0].snippet[:max_snippet_len]
        if len(results[0].snippet) > max_snippet_len:
            snippet += "…"
        return snippet[:_MAX_SUMMARY_LEN]

    lines: list[str] = ["Here is what I found:"]
    for i, result in enumerate(results, start=1):
        snippet = result.snippet[:max_snippet_len]
        if len(result.snippet) > max_snippet_len:
            snippet += "…"
        if result.title and snippet:
            lines.append(f"{i}. {result.title}: {snippet}")
        elif result.title:
            lines.append(f"{i}. {result.title}")
        elif snippet:
            lines.append(f"{i}. {snippet}")

    summary = "\n".join(lines)
    if len(summary) > _MAX_SUMMARY_LEN:
        summary = summary[:_MAX_SUMMARY_LEN] + "…"
    return summary
