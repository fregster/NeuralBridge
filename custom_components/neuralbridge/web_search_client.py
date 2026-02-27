"""Web search client for NeuralBridge.

Provides a generic provider interface for web search, with Brave Search as the
first concrete implementation.  Additional providers (SearXNG, Bing, Google
Custom Search, etc.) can be added by implementing the ``WebSearchProvider``
protocol and registering the provider key in ``SEARCH_PROVIDER_MAP``.

Security note: API keys must NEVER be logged at any level.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import aiohttp

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

    def __init__(self, api_key: str, timeout: int = 10, max_concurrent: int = 2) -> None:
        """Initialise the Brave provider.

        Args:
            api_key: Brave subscription token (never logged).
            timeout: HTTP request timeout in seconds.
            max_concurrent: Maximum number of parallel HTTP requests.
        """
        self._api_key = api_key
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared session, creating it lazily if needed.

        The session carries a ``User-Agent`` header so NeuralBridge traffic
        is identifiable in Brave's access logs.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "NeuralBridge/1.0"},
            )
        return self._session

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
            "text_decorations": "false",
            "search_lang": "en",
        }
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        async with self._semaphore:
            try:
                session = self._get_session()
                async with session.get(
                    self._BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response:
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

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


def _parse_brave_response(data: dict[str, Any]) -> list[SearchResult]:
    """Extract SearchResult objects from a Brave API JSON response.

    Parses both ``web.results`` and ``news.results`` blocks because the Brave
    Web Search API populates different blocks depending on the query type.
    News-heavy queries (e.g. "top headlines today", "latest news") typically
    return results in the ``news`` block while leaving ``web`` sparse or empty.
    Omitting the ``news`` block causes those queries to silently return no
    results, producing the fallback error message instead of an answer.

    Args:
        data: The parsed JSON dict from the Brave API.

    Returns:
        List of SearchResult dataclasses (web results first, then news results).
    """
    results: list[SearchResult] = []

    for block_key in ("web", "news"):
        block = data.get(block_key, {})
        if not isinstance(block, dict):
            continue
        raw: Any = block.get("results", [])
        items: list[Any] = raw if isinstance(raw, list) else []
        for item in items:
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


class BraveAnswersProvider:
    """Brave Answers API provider.

    Uses the Brave Answers API endpoint::

        POST https://api.search.brave.com/res/v1/answer

    Requires a **separate** Brave Answers subscription key (distinct from the
    Brave Search subscription).  The key is passed via ``X-Subscription-Token``
    and is never written to any log.

    The API returns a direct natural-language answer suitable for speech output.
    An empty list is returned when the API cannot produce a direct answer, so
    callers can fall back to a conventional search without raising exceptions.

    Docs: https://api-dashboard.search.brave.com/app/documentation/answer/get-started
    """

    _BASE_URL = "https://api.search.brave.com/res/v1/answer"

    def __init__(self, api_key: str, timeout: int = 10, max_concurrent: int = 2) -> None:
        """Initialise the Brave Answers provider.

        Args:
            api_key: Brave Answers subscription token (never logged).
            timeout: HTTP request timeout in seconds.
            max_concurrent: Maximum number of parallel HTTP requests.
        """
        self._api_key = api_key
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared session, creating it lazily if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "NeuralBridge/1.0"},
            )
        return self._session

    async def search(self, query: str, _count: int = 1) -> list[SearchResult]:
        """Query the Brave Answers API for a direct answer.

        ``_count`` is accepted for protocol compatibility but is ignored — the
        Answers API always returns a single answer text.

        Args:
            query: The question to answer.
            _count: Ignored; present for ``WebSearchProvider`` protocol compliance.

        Returns:
            A list containing one ``SearchResult`` whose ``snippet`` is the
            direct answer, or an empty list if no answer was returned.
        """
        params: dict[str, Any] = {"q": query}
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        async with self._semaphore:
            try:
                session = self._get_session()
                async with session.get(
                    self._BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response:
                    if response.status != _HTTP_OK:
                        _LOGGER.warning(
                            "Brave Answers returned HTTP %d — check API key and quota",
                            response.status,
                        )
                        return []
                    data = await response.json()
            except aiohttp.ClientError as err:
                _LOGGER.error("Brave Answers request failed: %s", err)
                return []
            except Exception as err:
                _LOGGER.error("Unexpected error during Brave Answers: %s", err)
                return []

        return _parse_brave_answers_response(data)

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


class BraveAnswersCombinedProvider:
    """Combined Brave provider: tries Answers first, falls back to Search.

    Requires two separate Brave subscription keys:
    - ``answers_api_key``: Brave Answers subscription token.
    - ``search_api_key``: Brave Search subscription token.

    Workflow::

        1. Query BraveAnswersProvider with the Answers key.
        2. If a direct answer is returned → return it (fast path).
        3. Otherwise query BraveSearchProvider with the Search key → return snippets.

    This delivers concise, speakable answers for factual Q&A while gracefully
    handling queries where no direct answer is available.
    """

    def __init__(
        self,
        answers_api_key: str,
        search_api_key: str,
        timeout: int = 10,
    ) -> None:
        """Initialise the combined Brave provider.

        Args:
            answers_api_key: Brave Answers subscription token (never logged).
            search_api_key: Brave Search subscription token (never logged).
            timeout: HTTP request timeout in seconds (shared by both sub-providers).
        """
        self._answers = BraveAnswersProvider(api_key=answers_api_key, timeout=timeout)
        self._search = BraveSearchProvider(api_key=search_api_key, timeout=timeout)

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        """Return a direct answer if available, otherwise fall back to search results.

        Args:
            query: The user's question or search string.
            count: Number of search results to request if the Answers fast-path misses.

        Returns:
            A single-item list with the direct answer, or a full search-result
            list from the Search API, or an empty list if both fail.
        """
        answer_results = await self._answers.search(query)
        if answer_results:
            return answer_results
        return await self._search.search(query, count)

    async def close(self) -> None:
        """Close both sub-provider sessions."""
        await self._answers.close()
        await self._search.close()


def _parse_brave_answers_response(data: dict[str, Any]) -> list[SearchResult]:
    """Extract a direct-answer SearchResult from a Brave Answers API JSON response.

    Handles two possible response shapes:

    * OpenAI-style chat completion: ``choices[0].message.content``
    * Brave-native: ``answer.text``

    Returns an empty list when neither shape yields usable text, so callers can
    fall back to conventional search without raising exceptions.

    Args:
        data: The parsed JSON dict from the Brave Answers API.

    Returns:
        A list containing one ``SearchResult`` (title and url are empty strings;
        the answer text is in ``snippet``), or an empty list.
    """
    # OpenAI-compatible shape
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            if isinstance(message, dict):
                content = str(message.get("content", "")).strip()
                if content:
                    return [SearchResult(title="", url="", snippet=content)]

    # Brave-native shape
    answer_block = data.get("answer")
    if isinstance(answer_block, dict):
        text = str(answer_block.get("text", "")).strip()
        if text:
            return [SearchResult(title="", url="", snippet=text)]

    # Top-level "answer" string (some API versions)
    top_answer = data.get("answer")
    if isinstance(top_answer, str):
        text = top_answer.strip()
        if text:
            return [SearchResult(title="", url="", snippet=text)]

    return []


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

        Safe to call even when the provider does not expose ``close``
        (e.g. third-party providers added via the ``WebSearchProvider`` protocol).
        """
        if hasattr(self._provider, "close"):
            await self._provider.close()


def _format_results(results: list[SearchResult], max_snippet_len: int) -> str:
    """Render a list of SearchResults as a concise plain-text summary.

    The output is designed to be read aloud by a voice assistant, so URLs are
    omitted from the speech text (they are visual-only noise in voice contexts).

    **Direct-answer shortcut:** when the list contains exactly one result whose
    URL is empty (the signature of a ``BraveAnswersProvider`` result), the
    snippet is returned as-is without the "Here is what I found:" preamble,
    for more natural TTS output.

    Args:
        results: Non-empty list of SearchResult objects.
        max_snippet_len: Maximum characters per snippet before truncation.

    Returns:
        Multi-line plain-text string, truncated to ``_MAX_SUMMARY_LEN``.
    """
    # Direct-answer path — single result with no URL means it came from the
    # Answers API.  Return the text directly without a numbered-list preamble.
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
