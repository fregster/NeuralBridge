"""Web search provider sub-package for NeuralBridge.

Provides the :class:`WebSearchProvider` protocol and :class:`SearchResult`
shared data type used by all concrete provider implementations.

Concrete providers:

* :mod:`providers.brave_search` — Brave Web Search API
* :mod:`providers.brave_answers` — Brave Answers API (direct Q&A)
* :mod:`providers.brave_combined` — Combined Brave provider (Answers + Search fallback)

All providers satisfy the :class:`WebSearchProvider` protocol, so they can be
swapped into :class:`WebSearchClient` transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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


__all__ = [
    "SearchResult",
    "WebSearchProvider",
]
