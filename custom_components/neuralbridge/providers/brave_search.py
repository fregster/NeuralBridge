"""Brave Web Search API provider for NeuralBridge.

Uses the Brave Web Search API endpoint::

    GET https://api.search.brave.com/res/v1/web/search

Docs: https://api.search.brave.com/app/documentation/web-search/get-started

Security note: API keys must NEVER be logged at any level.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from . import SearchResult

_LOGGER = logging.getLogger(__name__)
_HTTP_OK = 200


class BraveSearchProvider:
    """Brave Search API provider.

    The API key is passed via the ``X-Subscription-Token`` header and is
    never written to any log.
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
        """Return the shared session, creating it lazily if needed."""
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
            List of SearchResult objects.  Returns an empty list on any error.
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
            snippet = str(item.get("description", "")).strip()
            if not snippet:
                extras = item.get("extra_snippets", [])
                snippet = str(extras[0]).strip() if isinstance(extras, list) and extras else ""
            if title or url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results
