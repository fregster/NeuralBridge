"""Combined Brave provider: Answers-first with Search fallback for NeuralBridge.

Requires two separate Brave subscription keys:
- ``answers_api_key``: Brave Answers subscription token.
- ``search_api_key``: Brave Search subscription token.

Workflow::

    1. Query BraveAnswersProvider with the Answers key.
    2. If a direct answer is returned → return it (fast path).
    3. Otherwise query BraveSearchProvider with the Search key → return snippets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import SearchResult

from .brave_answers import BraveAnswersProvider
from .brave_search import BraveSearchProvider


class BraveAnswersCombinedProvider:
    """Combined Brave provider: tries Answers first, falls back to Search.

    Delivers concise, speakable answers for factual Q&A while gracefully
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
