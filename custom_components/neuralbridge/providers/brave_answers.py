"""Brave Answers API provider for NeuralBridge.

Uses the Brave Answers API endpoint::

    GET https://api.search.brave.com/res/v1/answer

Requires a **separate** Brave Answers subscription key (distinct from the
Brave Search subscription).

Docs: https://api-dashboard.search.brave.com/app/documentation/answer/get-started

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


class BraveAnswersProvider:
    """Brave Answers API provider.

    The key is passed via ``X-Subscription-Token`` and is never written to any log.
    Returns a direct natural-language answer suitable for speech output.
    An empty list is returned when the API cannot produce a direct answer.
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


def _parse_brave_answers_response(data: dict[str, Any]) -> list[SearchResult]:
    """Extract a direct-answer SearchResult from a Brave Answers API JSON response.

    Handles two possible response shapes:

    * OpenAI-style chat completion: ``choices[0].message.content``
    * Brave-native: ``answer.text``

    Returns an empty list when neither shape yields usable text.

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
