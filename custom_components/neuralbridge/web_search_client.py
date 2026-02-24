"""Web search client for NeuralBridge.

Provides a generic provider interface for web search, with Brave Search as the
first concrete implementation.  Additional providers (SearXNG, Bing, Google
Custom Search, etc.) can be added by implementing the ``WebSearchProvider``
protocol and registering the provider key in ``SEARCH_PROVIDER_MAP``.

Security note: API keys must NEVER be logged at any level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import aiohttp

from .const import (
    CONF_SEARCH_API_KEY,
    CONF_SEARCH_MAX_SNIPPET_LEN,
    CONF_SEARCH_PROVIDER,
    CONF_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_MAX_SNIPPET_LEN,
    DEFAULT_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_TIMEOUT,
    SEARCH_PROVIDER_BRAVE,
)

_LOGGER = logging.getLogger(__name__)
_HTTP_OK = 200

# Maximum characters in the formatted summary returned to the user.
_MAX_SUMMARY_LEN = 2000


@dataclass(frozen=True)
class SearchResult:
    """A single web search result.

    Attributes:
        title:   Page title.
        url:     Landing page URL.
        snippet: Short description snippet.
    """

    title: str
    url: str
    snippet: str


@runtime_checkable
class WebSearchProvider(Protocol):
    """Protocol all search providers must satisfy."""

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        """Execute a web search and return structured results.

        Args:
            query: The search query string.
            count: Maximum number of results to return.

        Returns:
            List of SearchResult objects (may be empty on failure).
        """
        ...  # pragma: no cover


class BraveSearchProvider:
    """Brave Search API provider.

    Uses the Brave Web Search API endpoint::

        GET https://api.search.brave.com/res/v1/web/search

    The API key is passed via the ``X-Subscription-Token`` header and is
    never written to any log.

    Docs: https://api.search.brave.com/app/documentation/web-search/get-started
    """

    _BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, timeout: int = 10) -> None:
        """Initialise the Brave provider.

        Args:
            api_key: Brave subscription token (never logged).
            timeout: HTTP request timeout in seconds.
        """
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        """Query the Brave Web Search API.

        Args:
            query: The search query string.
            count: Number of results to request (clamped to 1-20).

        Returns:
            List of SearchResult objects.  Returns an empty list on any error
            so callers can handle gracefully without exception propagation.
        """
        count = max(1, min(20, count))
        params: dict[str, Any] = {
            "q": query,
            "count": count,
            "text_decorations": False,
            "search_lang": "en",
        }
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    self._BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response,
            ):
                if response.status != _HTTP_OK:
                    _LOGGER.warning(
                        "Brave Search returned HTTP %d — check API key and quota",
                        response.status,
                    )
                    return []
                data = await response.json()
        except aiohttp.ClientError as err:
            _LOGGER.error("Brave Search request failed: %s", err)
            return []
        except Exception as err:
            _LOGGER.error("Unexpected error during Brave Search: %s", err)
            return []

        return _parse_brave_response(data)


def _parse_brave_response(data: dict[str, Any]) -> list[SearchResult]:
    """Extract SearchResult objects from a Brave API JSON response.

    Args:
        data: The parsed JSON dict from the Brave API.

    Returns:
        List of SearchResult dataclasses.
    """
    results: list[SearchResult] = []
    web_block = data.get("web", {}) if isinstance(data.get("web"), dict) else {}
    raw_results: Any = web_block.get("results", [])
    web_results: list[Any] = raw_results if isinstance(raw_results, list) else []
    for item in web_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        # Brave may supply the snippet under "description" or "extra_snippets"
        snippet = str(item.get("description", "")).strip()
        if not snippet:
            extras = item.get("extra_snippets", [])
            snippet = str(extras[0]).strip() if isinstance(extras, list) and extras else ""
        if title or url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


# ---------------------------------------------------------------------------
# Provider registry — extend this dict to add new providers.
# ---------------------------------------------------------------------------

SEARCH_PROVIDER_MAP: dict[str, type[BraveSearchProvider]] = {
    SEARCH_PROVIDER_BRAVE: BraveSearchProvider,
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
        """
        provider_key: str = agent_config.get(CONF_SEARCH_PROVIDER, SEARCH_PROVIDER_BRAVE)
        api_key: str = agent_config.get(CONF_SEARCH_API_KEY, "")
        timeout: int = int(agent_config.get("timeout", DEFAULT_SEARCH_TIMEOUT))
        self._result_count: int = int(
            agent_config.get(CONF_SEARCH_RESULT_COUNT, DEFAULT_SEARCH_RESULT_COUNT)
        )
        self._max_snippet_len: int = int(
            agent_config.get(CONF_SEARCH_MAX_SNIPPET_LEN, DEFAULT_SEARCH_MAX_SNIPPET_LEN)
        )

        provider_cls = SEARCH_PROVIDER_MAP.get(provider_key)
        if provider_cls is None:
            _LOGGER.warning("Unknown search provider '%s' — falling back to Brave", provider_key)
            provider_cls = BraveSearchProvider

        self._provider: WebSearchProvider = provider_cls(api_key=api_key, timeout=timeout)

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


def _format_results(results: list[SearchResult], max_snippet_len: int) -> str:
    """Render a list of SearchResults as a concise plain-text summary.

    The output is designed to be read aloud by a voice assistant, so URLs are
    omitted from the speech text (they are visual-only noise in voice contexts).

    Args:
        results: Non-empty list of SearchResult objects.
        max_snippet_len: Maximum characters per snippet before truncation.

    Returns:
        Multi-line plain-text string, truncated to ``_MAX_SUMMARY_LEN``.
    """
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
