"""Unit tests for WebSearchClient and web search provider infrastructure."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.neuralbridge.const import (
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
from custom_components.neuralbridge.web_search_client import (
    SEARCH_PROVIDER_MAP,
    BraveAnswersCombinedProvider,
    BraveAnswersProvider,
    BraveSearchProvider,
    SearchResult,
    WebSearchClient,
    _format_results,
    _parse_brave_answers_response,
    _parse_brave_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_brave_response(
    status: int = 200,
    json_data: dict[str, Any] | None = None,
) -> AsyncMock:
    """Return a mock aiohttp response yielded by an async context manager."""
    mock = AsyncMock()
    mock.status = status
    mock.json = AsyncMock(return_value=json_data if json_data is not None else {})
    return mock


def _make_session_cm(mock_response: AsyncMock) -> MagicMock:
    """Return a mock aiohttp session whose ``.get()`` yields *mock_response*.

    After P3, providers hold a persistent ``_session`` rather than wrapping the
    whole call in ``async with aiohttp.ClientSession()``.  The helper now returns
    the **session object** directly (not an outer async CM), and ``session.closed``
    is set to ``False`` so ``_get_session`` re-uses the existing instance.

    Usage in tests::

        session_cm = _make_session_cm(mock_response)
        with patch(
            "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
            return_value=session_cm,
        ):
            ...
    """
    get_cm = AsyncMock()
    get_cm.__aenter__ = AsyncMock(return_value=mock_response)
    get_cm.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.closed = False
    session.get = MagicMock(return_value=get_cm)
    session.close = AsyncMock()
    return session


def _web_response(count: int = 2) -> dict[str, Any]:
    """Return a minimal Brave API-shaped response dict."""
    results = [
        {"title": f"Title {i}", "url": f"https://example.com/{i}", "description": f"Snippet {i}"}
        for i in range(count)
    ]
    return {"web": {"results": results}}


# ---------------------------------------------------------------------------
# SEARCH_PROVIDER_MAP
# ---------------------------------------------------------------------------


def test_search_provider_map_contains_brave() -> None:
    """SEARCH_PROVIDER_MAP must include the brave key."""
    assert SEARCH_PROVIDER_BRAVE in SEARCH_PROVIDER_MAP
    assert SEARCH_PROVIDER_MAP[SEARCH_PROVIDER_BRAVE] is BraveSearchProvider


# ---------------------------------------------------------------------------
# BraveSearchProvider — __init__
# ---------------------------------------------------------------------------


def test_brave_provider_stores_api_key_and_timeout() -> None:
    """BraveSearchProvider stores api_key and timeout (api_key is not exposed via repr)."""
    provider = BraveSearchProvider(api_key="secret-key", timeout=20)
    assert provider._api_key == "secret-key"
    assert provider._timeout == 20


def test_brave_provider_api_key_not_in_repr() -> None:
    """API key must not appear in repr output (security)."""
    provider = BraveSearchProvider(api_key="super-secret")
    assert "super-secret" not in repr(provider)


# ---------------------------------------------------------------------------
# BraveSearchProvider.search — success
# ---------------------------------------------------------------------------


async def test_brave_search_success_returns_results() -> None:
    """search() returns a non-empty list on a 200 response."""
    data = _web_response(2)
    mock_response = _make_brave_response(200, data)
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveSearchProvider(api_key="tok", timeout=5)
        results = await provider.search("test query", count=2)

    assert len(results) == 2
    assert results[0].title == "Title 0"
    assert results[0].url == "https://example.com/0"
    assert results[0].snippet == "Snippet 0"


async def test_brave_search_with_count_clamp_min() -> None:
    """Count below 1 is clamped to 1."""
    data = _web_response(1)
    mock_response = _make_brave_response(200, data)
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveSearchProvider(api_key="tok", timeout=5)
        results = await provider.search("q", count=0)

    assert len(results) == 1


async def test_brave_search_with_count_clamp_max() -> None:
    """Count above 20 is clamped to 20."""
    data = _web_response(3)
    mock_response = _make_brave_response(200, data)
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveSearchProvider(api_key="tok", timeout=5)
        # count=99 clamped to 20 before the API call
        results = await provider.search("q", count=99)

    # We still get 3 back (the mock only has 3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# BraveSearchProvider.search — non-200 HTTP status
# ---------------------------------------------------------------------------


async def test_brave_search_non_200_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    """Non-200 status returns empty list and logs a warning."""
    mock_response = _make_brave_response(status=429, json_data={})
    session_cm = _make_session_cm(mock_response)

    with (
        patch(
            "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
            return_value=session_cm,
        ),
        caplog.at_level(logging.WARNING),
    ):
        provider = BraveSearchProvider(api_key="tok", timeout=5)
        results = await provider.search("q")

    assert results == []
    assert "429" in caplog.text


# ---------------------------------------------------------------------------
# BraveSearchProvider.search — network errors
# ---------------------------------------------------------------------------


async def test_brave_search_client_error_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """aiohttp.ClientError is caught and returns empty list."""
    get_cm = AsyncMock()
    get_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("connection refused"))
    get_cm.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.get = MagicMock(return_value=get_cm)

    with (
        patch(
            "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
            return_value=session,
        ),
        caplog.at_level(logging.ERROR),
    ):
        provider = BraveSearchProvider(api_key="tok", timeout=5)
        results = await provider.search("q")

    assert results == []
    assert "Brave Search" in caplog.text


async def test_brave_search_unexpected_exception_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected exceptions are caught and return empty list."""
    get_cm = AsyncMock()
    get_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
    get_cm.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.get = MagicMock(return_value=get_cm)

    with (
        patch(
            "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
            return_value=session,
        ),
        caplog.at_level(logging.ERROR),
    ):
        provider = BraveSearchProvider(api_key="tok", timeout=5)
        results = await provider.search("q")

    assert results == []


# ---------------------------------------------------------------------------
# BraveSearchProvider.search — API key never logged
# ---------------------------------------------------------------------------


async def test_brave_search_api_key_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The API key must never appear in any log output, even on error."""
    secret = "my-very-secret-api-key-12345"  # noqa: S105
    mock_response = _make_brave_response(status=401, json_data={})
    session_cm = _make_session_cm(mock_response)

    with (
        patch(
            "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
            return_value=session_cm,
        ),
        caplog.at_level(logging.DEBUG),
    ):
        provider = BraveSearchProvider(api_key=secret, timeout=5)
        await provider.search("q")

    for record in caplog.records:
        assert secret not in record.getMessage()


# ---------------------------------------------------------------------------
# BraveSearchProvider — session lifecycle (P3)
# ---------------------------------------------------------------------------


async def test_brave_search_session_reused_across_calls() -> None:
    """A single ClientSession is created and reused for two consecutive searches."""
    data = _web_response(1)
    mock_response = _make_brave_response(200, data)
    session_mock = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
        return_value=session_mock,
    ) as session_cls:
        provider = BraveSearchProvider(api_key="tok")
        await provider.search("first query")
        await provider.search("second query")

    # ClientSession constructor called only once
    assert session_cls.call_count == 1


async def test_brave_search_close_closes_session_and_clears_ref() -> None:
    """close() closes the session and sets _session to None."""
    data = _web_response(1)
    mock_response = _make_brave_response(200, data)
    session_mock = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
        return_value=session_mock,
    ):
        provider = BraveSearchProvider(api_key="tok")
        await provider.search("query")  # creates session
        assert provider._session is not None

        await provider.close()

    assert provider._session is None
    session_mock.close.assert_awaited_once()


async def test_brave_search_session_recreated_after_close() -> None:
    """After close(), the next search call recreates the session."""
    data = _web_response(1)
    mock_response = _make_brave_response(200, data)
    session_mock1 = _make_session_cm(mock_response)
    session_mock2 = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
        side_effect=[session_mock1, session_mock2],
    ) as session_cls:
        provider = BraveSearchProvider(api_key="tok")
        await provider.search("first")  # uses session_mock1
        await provider.close()
        await provider.search("second")  # recreates → session_mock2

    assert session_cls.call_count == 2


async def test_brave_search_close_is_noop_when_no_session() -> None:
    """close() on a fresh provider (no session yet) does not raise."""
    provider = BraveSearchProvider(api_key="tok")
    assert provider._session is None
    await provider.close()  # should not raise


# ---------------------------------------------------------------------------
# _parse_brave_response
# ---------------------------------------------------------------------------


def test_parse_brave_response_empty_web_block() -> None:
    """Missing 'web' key returns empty list."""
    assert _parse_brave_response({}) == []


def test_parse_brave_response_non_dict_web_value() -> None:
    """Non-dict 'web' value returns empty list."""
    assert _parse_brave_response({"web": "invalid"}) == []


def test_parse_brave_response_empty_results_list() -> None:
    """Present but empty results list returns empty list."""
    assert _parse_brave_response({"web": {"results": []}}) == []


def test_parse_brave_response_non_dict_item_is_skipped() -> None:
    """Non-dict items in results are silently skipped."""
    results = _parse_brave_response({"web": {"results": ["not-a-dict", None]}})
    assert results == []


def test_parse_brave_response_fallback_to_extra_snippets() -> None:
    """When 'description' is absent, first extra_snippet is used."""
    data: dict[str, Any] = {
        "web": {
            "results": [
                {
                    "title": "Page",
                    "url": "https://example.com",
                    "description": "",
                    "extra_snippets": ["fallback snippet"],
                }
            ]
        }
    }
    results = _parse_brave_response(data)
    assert len(results) == 1
    assert results[0].snippet == "fallback snippet"


def test_parse_brave_response_item_with_only_url() -> None:
    """Item with only a URL (no title) is kept."""
    data: dict[str, Any] = {
        "web": {"results": [{"url": "https://example.com", "title": "", "description": ""}]}
    }
    results = _parse_brave_response(data)
    assert len(results) == 1
    assert results[0].url == "https://example.com"


def test_parse_brave_response_no_title_no_url_is_skipped() -> None:
    """Item with neither title nor URL is skipped."""
    data: dict[str, Any] = {"web": {"results": [{"description": "just a snippet"}]}}
    results = _parse_brave_response(data)
    assert results == []


def test_parse_brave_response_news_block_only_returns_results() -> None:
    """News results are returned when the web block is absent or empty."""
    data: dict[str, Any] = {
        "news": {
            "results": [
                {
                    "title": "Top Headline",
                    "url": "https://news.example.com/1",
                    "description": "Breaking news snippet.",
                }
            ]
        }
    }
    results = _parse_brave_response(data)
    assert len(results) == 1
    assert results[0].title == "Top Headline"
    assert results[0].url == "https://news.example.com/1"
    assert results[0].snippet == "Breaking news snippet."


def test_parse_brave_response_news_block_combined_with_web() -> None:
    """When both blocks have results they are combined: web first, then news."""
    data: dict[str, Any] = {
        "web": {
            "results": [
                {"title": "Web Result", "url": "https://web.example.com", "description": "web"}
            ]
        },
        "news": {
            "results": [
                {"title": "News Result", "url": "https://news.example.com", "description": "news"}
            ]
        },
    }
    results = _parse_brave_response(data)
    assert len(results) == 2
    assert results[0].title == "Web Result"
    assert results[1].title == "News Result"


def test_parse_brave_response_news_block_non_dict_is_skipped() -> None:
    """Non-dict 'news' value is skipped; web results still returned."""
    data: dict[str, Any] = {
        "web": {
            "results": [
                {"title": "Web Only", "url": "https://web.example.com", "description": "ok"}
            ]
        },
        "news": "invalid",
    }
    results = _parse_brave_response(data)
    assert len(results) == 1
    assert results[0].title == "Web Only"


def test_parse_brave_response_news_block_empty_results_returns_web_only() -> None:
    """Empty news results list does not affect web results."""
    data: dict[str, Any] = {
        "web": {
            "results": [
                {"title": "Web Item", "url": "https://web.example.com", "description": "desc"}
            ]
        },
        "news": {"results": []},
    }
    results = _parse_brave_response(data)
    assert len(results) == 1
    assert results[0].title == "Web Item"


def test_parse_brave_response_news_non_list_results_is_skipped() -> None:
    """Non-list news.results is treated as empty."""
    data: dict[str, Any] = {"news": {"results": "not-a-list"}}
    results = _parse_brave_response(data)
    assert results == []


def test_parse_brave_response_news_item_no_title_no_url_is_skipped() -> None:
    """News item with neither title nor URL is skipped."""
    data: dict[str, Any] = {"news": {"results": [{"description": "snippet only"}]}}
    results = _parse_brave_response(data)
    assert results == []


def test_parse_brave_response_news_item_non_dict_is_skipped() -> None:
    """Non-dict items in news.results are silently skipped."""
    data: dict[str, Any] = {"news": {"results": ["not-a-dict", None, 42]}}
    results = _parse_brave_response(data)
    assert results == []


def test_parse_brave_response_news_fallback_to_extra_snippets() -> None:
    """News items also use extra_snippets when description is absent."""
    data: dict[str, Any] = {
        "news": {
            "results": [
                {
                    "title": "News Page",
                    "url": "https://news.example.com",
                    "description": "",
                    "extra_snippets": ["news fallback snippet"],
                }
            ]
        }
    }
    results = _parse_brave_response(data)
    assert len(results) == 1
    assert results[0].snippet == "news fallback snippet"


# ---------------------------------------------------------------------------
# _format_results
# ---------------------------------------------------------------------------


def test_format_results_basic() -> None:
    """Formatted output begins with 'Here is what I found:' header."""
    results = [SearchResult(title="Alpha", url="https://a.com", snippet="Alpha snippet")]
    output = _format_results(results, max_snippet_len=200)
    assert output.startswith("Here is what I found:")
    assert "Alpha" in output
    assert "Alpha snippet" in output


def test_format_results_snippet_truncation() -> None:
    """Snippets longer than max_snippet_len are truncated with ellipsis."""
    long_snippet = "x" * 300
    results = [SearchResult(title="T", url="https://t.com", snippet=long_snippet)]
    output = _format_results(results, max_snippet_len=200)
    assert "…" in output
    # Output snippet should be 200 chars + "…"
    lines = output.splitlines()
    body = lines[1]  # "1. T: <snippet>"
    snippet_part = body.split(": ", 1)[1]
    assert len(snippet_part) == 201  # 200 + "…"


def test_format_results_title_only_item() -> None:
    """Items with title but no snippet render as '1. Title'."""
    results = [SearchResult(title="Title Only", url="https://t.com", snippet="")]
    output = _format_results(results, max_snippet_len=200)
    assert "1. Title Only" in output


def test_format_results_snippet_only_item() -> None:
    """Items with snippet but no title render as '1. snippet'."""
    results = [SearchResult(title="", url="https://t.com", snippet="Just a snippet")]
    output = _format_results(results, max_snippet_len=200)
    assert "1. Just a snippet" in output


def test_format_results_summary_truncation() -> None:
    """Total summary is hard-capped at 2000 characters."""
    # Create enough results to exceed 2000 chars
    results = [
        SearchResult(title=f"Title{i}", url=f"https://example.com/{i}", snippet="x" * 200)
        for i in range(20)
    ]
    output = _format_results(results, max_snippet_len=200)
    assert len(output) <= 2001  # 2000 + "…"


def test_format_results_url_not_in_output() -> None:
    """URLs are omitted from the formatted output (voice-assistant optimised)."""
    results = [SearchResult(title="Page", url="https://should-not-appear.com", snippet="details")]
    output = _format_results(results, max_snippet_len=200)
    assert "should-not-appear.com" not in output


# ---------------------------------------------------------------------------
# WebSearchClient — __init__
# ---------------------------------------------------------------------------


def test_web_search_client_init_brave() -> None:
    """WebSearchClient creates a BraveSearchProvider for the 'brave' key."""
    config: dict[str, Any] = {
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
        CONF_SEARCH_API_KEY: "key123",
        CONF_SEARCH_RESULT_COUNT: 3,
        CONF_SEARCH_MAX_SNIPPET_LEN: 150,
        "timeout": 10,
    }
    client = WebSearchClient(config)
    assert isinstance(client._provider, BraveSearchProvider)
    assert client._result_count == 3
    assert client._max_snippet_len == 150


def test_web_search_client_init_unknown_provider_falls_back_to_brave(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown provider key falls back to BraveSearchProvider with a warning."""
    config: dict[str, Any] = {
        CONF_SEARCH_PROVIDER: "nonexistent_provider",
        CONF_SEARCH_API_KEY: "key",
        CONF_SEARCH_RESULT_COUNT: DEFAULT_SEARCH_RESULT_COUNT,
        CONF_SEARCH_MAX_SNIPPET_LEN: DEFAULT_SEARCH_MAX_SNIPPET_LEN,
        "timeout": DEFAULT_SEARCH_TIMEOUT,
    }
    with caplog.at_level(logging.WARNING):
        client = WebSearchClient(config)

    assert isinstance(client._provider, BraveSearchProvider)
    assert "nonexistent_provider" in caplog.text


def test_web_search_client_defaults_applied_when_keys_absent() -> None:
    """Missing config keys fall back to module-level defaults."""
    client = WebSearchClient({CONF_SEARCH_API_KEY: "key"})
    assert client._result_count == DEFAULT_SEARCH_RESULT_COUNT
    assert client._max_snippet_len == DEFAULT_SEARCH_MAX_SNIPPET_LEN


# ---------------------------------------------------------------------------
# WebSearchClient.search_and_summarise
# ---------------------------------------------------------------------------


async def test_web_search_client_search_and_summarise_success() -> None:
    """search_and_summarise returns a non-empty string when results are found."""
    config: dict[str, Any] = {
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
        CONF_SEARCH_API_KEY: "key",
        CONF_SEARCH_RESULT_COUNT: 2,
        CONF_SEARCH_MAX_SNIPPET_LEN: 200,
        "timeout": 5,
    }
    mock_results = [
        SearchResult(title="Prime Minister", url="https://gov.uk", snippet="The PM is ..."),
        SearchResult(title="Wiki", url="https://en.wikipedia.org", snippet="Background info"),
    ]
    client = WebSearchClient(config)
    with patch.object(
        client._provider, "search", new_callable=AsyncMock, return_value=mock_results
    ):
        summary = await client.search_and_summarise("Who is the UK prime minister?")

    assert summary is not None
    assert "Prime Minister" in summary
    assert "Here is what I found:" in summary


async def test_web_search_client_no_results_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """search_and_summarise returns None when the provider returns no results."""
    config: dict[str, Any] = {
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
        CONF_SEARCH_API_KEY: "key",
        CONF_SEARCH_RESULT_COUNT: 5,
        CONF_SEARCH_MAX_SNIPPET_LEN: 200,
        "timeout": 5,
    }
    client = WebSearchClient(config)
    with (
        patch.object(client._provider, "search", new_callable=AsyncMock, return_value=[]),
        caplog.at_level(logging.WARNING),
    ):
        summary = await client.search_and_summarise("query with no results")

    assert summary is None
    # No PII (query text) should appear in the warning
    assert "query with no results" not in caplog.text


# ---------------------------------------------------------------------------
# WebSearchClient — close() session lifecycle (P3)
# ---------------------------------------------------------------------------


async def test_web_search_client_close_delegates_to_provider() -> None:
    """WebSearchClient.close() calls close() on the underlying provider."""
    config: dict[str, Any] = {
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
        CONF_SEARCH_API_KEY: "key",
    }
    client = WebSearchClient(config)
    client._provider.close = AsyncMock()  # type: ignore[attr-defined]
    await client.close()
    client._provider.close.assert_awaited_once()  # type: ignore[attr-defined]


async def test_web_search_client_close_safe_when_provider_has_no_close() -> None:
    """WebSearchClient.close() is a no-op when the provider has no close method."""
    config: dict[str, Any] = {
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
        CONF_SEARCH_API_KEY: "key",
    }
    client = WebSearchClient(config)
    # Replace the provider with a protocol-only object that has no close method
    bare_provider = MagicMock(spec=[])  # spec=[] means no attributes
    client._provider = bare_provider
    await client.close()  # should not raise


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


def test_search_result_is_frozen() -> None:
    """SearchResult is a frozen dataclass — mutation raises FrozenInstanceError."""
    result = SearchResult(title="T", url="https://u.com", snippet="S")
    with pytest.raises((AttributeError, TypeError)):
        result.title = "modified"  # type: ignore[misc]


def test_search_result_equality() -> None:
    """Two SearchResult objects with identical fields compare equal."""
    r1 = SearchResult(title="T", url="https://u.com", snippet="S")
    r2 = SearchResult(title="T", url="https://u.com", snippet="S")
    assert r1 == r2


# ---------------------------------------------------------------------------
# SEARCH_PROVIDER_MAP — new provider keys
# ---------------------------------------------------------------------------


def test_search_provider_map_contains_brave_answers() -> None:
    """SEARCH_PROVIDER_MAP must include the brave_answers key."""
    assert SEARCH_PROVIDER_BRAVE_ANSWERS in SEARCH_PROVIDER_MAP
    assert SEARCH_PROVIDER_MAP[SEARCH_PROVIDER_BRAVE_ANSWERS] is BraveAnswersProvider


def test_search_provider_map_does_not_contain_brave_combined() -> None:
    """brave_combined is handled by WebSearchClient directly (two-key constructor)."""
    assert SEARCH_PROVIDER_BRAVE_COMBINED not in SEARCH_PROVIDER_MAP


# ---------------------------------------------------------------------------
# BraveAnswersProvider — __init__
# ---------------------------------------------------------------------------


def test_brave_answers_provider_stores_api_key_and_timeout() -> None:
    """BraveAnswersProvider stores api_key and timeout."""
    provider = BraveAnswersProvider(api_key="answers-key", timeout=12)
    assert provider._api_key == "answers-key"
    assert provider._timeout == 12


def test_brave_answers_provider_api_key_not_in_repr() -> None:
    """Answers API key must not appear in repr output (security)."""
    provider = BraveAnswersProvider(api_key="super-secret-answers")
    assert "super-secret-answers" not in repr(provider)


# ---------------------------------------------------------------------------
# BraveAnswersProvider.search — success paths
# ---------------------------------------------------------------------------


def _make_answers_response(
    status: int = 200,
    json_data: dict[str, Any] | None = None,
) -> AsyncMock:
    """Return a mock aiohttp response for the Answers endpoint."""
    mock = AsyncMock()
    mock.status = status
    mock.json = AsyncMock(return_value=json_data if json_data is not None else {})
    return mock


async def test_brave_answers_search_openai_style_response() -> None:
    """search() parses OpenAI-style choices[0].message.content response."""
    data: dict[str, Any] = {"choices": [{"message": {"content": "The answer is 42."}}]}
    mock_response = _make_answers_response(200, data)
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveAnswersProvider(api_key="tok", timeout=5)
        results = await provider.search("What is the answer?")

    assert len(results) == 1
    assert results[0].snippet == "The answer is 42."
    assert results[0].title == ""
    assert results[0].url == ""


async def test_brave_answers_search_brave_native_dict_response() -> None:
    """search() parses Brave-native answer.text response shape."""
    data: dict[str, Any] = {"answer": {"text": "It is raining in Tokyo."}}
    mock_response = _make_answers_response(200, data)
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveAnswersProvider(api_key="tok", timeout=5)
        results = await provider.search("What is the weather in Tokyo?")

    assert len(results) == 1
    assert results[0].snippet == "It is raining in Tokyo."


async def test_brave_answers_search_top_level_string_response() -> None:
    """search() parses a top-level 'answer' string field."""
    data: dict[str, Any] = {"answer": "The capital is Paris."}
    mock_response = _make_answers_response(200, data)
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveAnswersProvider(api_key="tok", timeout=5)
        results = await provider.search("What is the capital of France?")

    assert len(results) == 1
    assert results[0].snippet == "The capital is Paris."


async def test_brave_answers_search_no_answer_returns_empty() -> None:
    """search() returns empty list when no recognisable answer field is present."""
    mock_response = _make_answers_response(200, {"type": "answer", "web": {}})
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveAnswersProvider(api_key="tok", timeout=5)
        results = await provider.search("unknowable query")

    assert results == []


async def test_brave_answers_search_count_param_is_accepted() -> None:
    """count parameter is accepted for protocol compliance without raising."""
    data: dict[str, Any] = {"choices": [{"message": {"content": "OK"}}]}
    mock_response = _make_answers_response(200, data)
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveAnswersProvider(api_key="tok", timeout=5)
        results = await provider.search("q", 10)  # positional; _count is intentionally unused

    assert len(results) == 1


# ---------------------------------------------------------------------------
# BraveAnswersProvider.search — non-200 / errors
# ---------------------------------------------------------------------------


async def test_brave_answers_search_non_200_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-200 HTTP status returns empty list and logs a warning."""
    mock_response = _make_answers_response(status=429)
    session_cm = _make_session_cm(mock_response)

    with (
        patch(
            "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
            return_value=session_cm,
        ),
        caplog.at_level(logging.WARNING),
    ):
        provider = BraveAnswersProvider(api_key="tok", timeout=5)
        results = await provider.search("q")

    assert results == []
    assert "429" in caplog.text


async def test_brave_answers_search_client_error_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """aiohttp.ClientError is caught and returns empty list."""
    get_cm = AsyncMock()
    get_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("no route"))
    get_cm.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.get = MagicMock(return_value=get_cm)

    with (
        patch(
            "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
            return_value=session,
        ),
        caplog.at_level(logging.ERROR),
    ):
        provider = BraveAnswersProvider(api_key="tok", timeout=5)
        results = await provider.search("q")

    assert results == []
    assert "Brave Answers" in caplog.text


async def test_brave_answers_search_unexpected_exception_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected exceptions are caught and return empty list."""
    get_cm = AsyncMock()
    get_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
    get_cm.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.get = MagicMock(return_value=get_cm)

    with (
        patch(
            "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
            return_value=session,
        ),
        caplog.at_level(logging.ERROR),
    ):
        provider = BraveAnswersProvider(api_key="tok", timeout=5)
        results = await provider.search("q")

    assert results == []


async def test_brave_answers_search_api_key_never_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Answers API key must never appear in any log output."""
    secret = "my-very-secret-answers-key-99"  # noqa: S105
    mock_response = _make_answers_response(status=401)
    session_cm = _make_session_cm(mock_response)

    with (
        patch(
            "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
            return_value=session_cm,
        ),
        caplog.at_level(logging.DEBUG),
    ):
        provider = BraveAnswersProvider(api_key=secret, timeout=5)
        await provider.search("q")

    for record in caplog.records:
        assert secret not in record.getMessage()


# ---------------------------------------------------------------------------
# BraveAnswersProvider — session lifecycle (P3)
# ---------------------------------------------------------------------------


async def test_brave_answers_session_reused_across_calls() -> None:
    """A single ClientSession is created and reused for two consecutive searches."""
    mock_response = _make_answers_response(200, {"answer": {"text": "42"}})
    session_mock = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
        return_value=session_mock,
    ) as session_cls:
        provider = BraveAnswersProvider(api_key="tok")
        await provider.search("first question")
        await provider.search("second question")

    assert session_cls.call_count == 1


async def test_brave_answers_close_closes_session_and_clears_ref() -> None:
    """close() closes the session and sets _session to None."""
    mock_response = _make_answers_response(200, {"answer": {"text": "yes"}})
    session_mock = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
        return_value=session_mock,
    ):
        provider = BraveAnswersProvider(api_key="tok")
        await provider.search("question")
        assert provider._session is not None

        await provider.close()

    assert provider._session is None
    session_mock.close.assert_awaited_once()


async def test_brave_answers_session_recreated_after_close() -> None:
    """After close(), the next search recreates the session."""
    mock_response = _make_answers_response(200, {"answer": {"text": "yes"}})
    session_mock1 = _make_session_cm(mock_response)
    session_mock2 = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
        side_effect=[session_mock1, session_mock2],
    ) as session_cls:
        provider = BraveAnswersProvider(api_key="tok")
        await provider.search("first")
        await provider.close()
        await provider.search("second")

    assert session_cls.call_count == 2


async def test_brave_answers_close_is_noop_when_no_session() -> None:
    """close() on a fresh provider does not raise."""
    provider = BraveAnswersProvider(api_key="tok")
    assert provider._session is None
    await provider.close()


# ---------------------------------------------------------------------------
# _parse_brave_answers_response
# ---------------------------------------------------------------------------


def test_parse_brave_answers_response_empty_dict() -> None:
    """Empty dict returns empty list."""
    assert _parse_brave_answers_response({}) == []


def test_parse_brave_answers_response_openai_choices_shape() -> None:
    """OpenAI-style choices[0].message.content is parsed correctly."""
    data: dict[str, Any] = {"choices": [{"message": {"content": "Direct answer."}}]}
    results = _parse_brave_answers_response(data)
    assert len(results) == 1
    assert results[0].snippet == "Direct answer."
    assert results[0].title == ""
    assert results[0].url == ""


def test_parse_brave_answers_response_openai_empty_content() -> None:
    """Empty content string in choices returns empty list."""
    data: dict[str, Any] = {"choices": [{"message": {"content": ""}}]}
    assert _parse_brave_answers_response(data) == []


def test_parse_brave_answers_response_openai_empty_choices_list() -> None:
    """Empty choices list falls through to other shapes."""
    data: dict[str, Any] = {"choices": [], "answer": {"text": "fallback"}}
    results = _parse_brave_answers_response(data)
    assert len(results) == 1
    assert results[0].snippet == "fallback"


def test_parse_brave_answers_response_openai_non_dict_message() -> None:
    """Non-dict message inside choices is skipped; fallback applies."""
    data: dict[str, Any] = {
        "choices": [{"message": "not-a-dict"}],
        "answer": {"text": "native fallback"},
    }
    results = _parse_brave_answers_response(data)
    assert len(results) == 1
    assert results[0].snippet == "native fallback"


def test_parse_brave_answers_response_brave_native_dict_shape() -> None:
    """answer.text dict shape is parsed correctly."""
    data: dict[str, Any] = {"answer": {"text": "The result."}}
    results = _parse_brave_answers_response(data)
    assert len(results) == 1
    assert results[0].snippet == "The result."


def test_parse_brave_answers_response_brave_native_empty_text() -> None:
    """Empty answer.text falls through to top-level string check."""
    data: dict[str, Any] = {"answer": {"text": ""}}
    assert _parse_brave_answers_response(data) == []


def test_parse_brave_answers_response_top_level_string() -> None:
    """Top-level answer string is parsed as snippet."""
    data: dict[str, Any] = {"answer": "Simple answer string."}
    results = _parse_brave_answers_response(data)
    assert len(results) == 1
    assert results[0].snippet == "Simple answer string."


def test_parse_brave_answers_response_top_level_empty_string() -> None:
    """Empty top-level answer string returns empty list."""
    data: dict[str, Any] = {"answer": "   "}
    assert _parse_brave_answers_response(data) == []


# ---------------------------------------------------------------------------
# BraveAnswersCombinedProvider
# ---------------------------------------------------------------------------


async def test_brave_combined_provider_fast_path_returns_answer() -> None:
    """Combined provider returns direct answer when Answers API succeeds."""
    answers_result = [SearchResult(title="", url="", snippet="Direct facts.")]

    answers_provider = AsyncMock(spec=BraveAnswersProvider)
    answers_provider.search = AsyncMock(return_value=answers_result)
    search_provider = AsyncMock(spec=BraveSearchProvider)
    search_provider.search = AsyncMock(return_value=[])

    provider = BraveAnswersCombinedProvider(
        answers_api_key="a-key", search_api_key="s-key", timeout=5
    )
    provider._answers = answers_provider
    provider._search = search_provider

    results = await provider.search("What is gravity?")

    assert results == answers_result
    answers_provider.search.assert_awaited_once_with("What is gravity?")
    search_provider.search.assert_not_awaited()


async def test_brave_combined_provider_fallback_to_search() -> None:
    """Combined provider falls back to Search when Answers returns empty."""
    search_results = [SearchResult(title="Page", url="https://example.com", snippet="Some snippet")]

    answers_provider = AsyncMock(spec=BraveAnswersProvider)
    answers_provider.search = AsyncMock(return_value=[])
    search_provider = AsyncMock(spec=BraveSearchProvider)
    search_provider.search = AsyncMock(return_value=search_results)

    provider = BraveAnswersCombinedProvider(
        answers_api_key="a-key", search_api_key="s-key", timeout=5
    )
    provider._answers = answers_provider
    provider._search = search_provider

    results = await provider.search("Latest news", count=3)

    assert results == search_results
    answers_provider.search.assert_awaited_once_with("Latest news")
    search_provider.search.assert_awaited_once_with("Latest news", 3)


async def test_brave_combined_provider_both_fail_returns_empty() -> None:
    """Combined provider returns empty list when both sub-providers fail."""
    answers_provider = AsyncMock(spec=BraveAnswersProvider)
    answers_provider.search = AsyncMock(return_value=[])
    search_provider = AsyncMock(spec=BraveSearchProvider)
    search_provider.search = AsyncMock(return_value=[])

    provider = BraveAnswersCombinedProvider(
        answers_api_key="a-key", search_api_key="s-key", timeout=5
    )
    provider._answers = answers_provider
    provider._search = search_provider

    results = await provider.search("impossible query")
    assert results == []


def test_brave_combined_provider_creates_sub_providers() -> None:
    """BraveAnswersCombinedProvider creates both sub-providers on init."""
    provider = BraveAnswersCombinedProvider(
        answers_api_key="ans-key", search_api_key="srch-key", timeout=8
    )
    assert isinstance(provider._answers, BraveAnswersProvider)
    assert isinstance(provider._search, BraveSearchProvider)
    assert provider._answers._api_key == "ans-key"
    assert provider._search._api_key == "srch-key"
    assert provider._answers._timeout == 8
    assert provider._search._timeout == 8


async def test_brave_combined_provider_close_delegates_to_sub_providers() -> None:
    """BraveAnswersCombinedProvider.close() delegates to both sub-providers."""
    provider = BraveAnswersCombinedProvider(answers_api_key="ans-key", search_api_key="srch-key")
    provider._answers.close = AsyncMock()  # type: ignore[method-assign]
    provider._search.close = AsyncMock()  # type: ignore[method-assign]

    await provider.close()

    provider._answers.close.assert_awaited_once()
    provider._search.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# WebSearchClient.__init__ — new providers
# ---------------------------------------------------------------------------


def test_web_search_client_init_brave_answers() -> None:
    """WebSearchClient creates BraveAnswersProvider for 'brave_answers' key."""
    config: dict[str, Any] = {
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE_ANSWERS,
        CONF_SEARCH_API_KEY: "not-used",
        CONF_SEARCH_ANSWERS_API_KEY: "answers-key",
        CONF_SEARCH_RESULT_COUNT: 1,
        CONF_SEARCH_MAX_SNIPPET_LEN: 200,
        "timeout": 10,
    }
    client = WebSearchClient(config)
    assert isinstance(client._provider, BraveAnswersProvider)
    assert client._provider._api_key == "answers-key"


def test_web_search_client_init_brave_combined() -> None:
    """WebSearchClient creates BraveAnswersCombinedProvider for 'brave_combined' key."""
    config: dict[str, Any] = {
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE_COMBINED,
        CONF_SEARCH_API_KEY: "search-key",
        CONF_SEARCH_ANSWERS_API_KEY: "answers-key",
        CONF_SEARCH_RESULT_COUNT: 5,
        CONF_SEARCH_MAX_SNIPPET_LEN: 200,
        "timeout": 10,
    }
    client = WebSearchClient(config)
    assert isinstance(client._provider, BraveAnswersCombinedProvider)


def test_web_search_client_init_brave_combined_passes_both_keys() -> None:
    """BraveAnswersCombinedProvider receives both keys from agent config."""
    config: dict[str, Any] = {
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE_COMBINED,
        CONF_SEARCH_API_KEY: "my-search-key",
        CONF_SEARCH_ANSWERS_API_KEY: "my-answers-key",
        "timeout": 5,
    }
    client = WebSearchClient(config)
    combined = client._provider
    assert isinstance(combined, BraveAnswersCombinedProvider)
    assert combined._answers._api_key == "my-answers-key"
    assert combined._search._api_key == "my-search-key"


# ---------------------------------------------------------------------------
# _format_results — direct-answer (no-URL) shortcut
# ---------------------------------------------------------------------------


def test_format_results_direct_answer_single_no_url_skips_preamble() -> None:
    """Single result with empty URL returns snippet directly, no preamble."""
    result = SearchResult(title="", url="", snippet="The capital is Paris.")
    output = _format_results([result], max_snippet_len=200)
    assert output == "The capital is Paris."
    assert "Here is what I found:" not in output


def test_format_results_direct_answer_truncates_at_max_snippet_len() -> None:
    """Direct-answer snippet is still truncated at max_snippet_len."""
    long_answer = "x" * 300
    result = SearchResult(title="", url="", snippet=long_answer)
    output = _format_results([result], max_snippet_len=200)
    assert output == "x" * 200 + "\u2026"


def test_format_results_direct_answer_truncates_at_max_summary_len() -> None:
    """Direct-answer snippet is hard-capped at _MAX_SUMMARY_LEN (2000 chars)."""
    very_long = "y" * 2500
    result = SearchResult(title="", url="", snippet=very_long)
    output = _format_results([result], max_snippet_len=3000)
    assert len(output) <= 2001  # 2000 + "…"


def test_format_results_single_result_with_url_still_uses_preamble() -> None:
    """Single result with a populated URL still uses the standard numbered list."""
    result = SearchResult(title="A page", url="https://example.com", snippet="Some info")
    output = _format_results([result], max_snippet_len=200)
    assert output.startswith("Here is what I found:")
    assert "1. A page" in output


# ---------------------------------------------------------------------------
# P5 — Rate-limiting semaphore
# ---------------------------------------------------------------------------


def test_brave_search_default_max_concurrent_is_two() -> None:
    """BraveSearchProvider defaults to 2 concurrent requests."""
    provider = BraveSearchProvider("test-key")
    assert provider._semaphore._value == 2


def test_brave_answers_default_max_concurrent_is_two() -> None:
    """BraveAnswersProvider defaults to 2 concurrent requests."""
    provider = BraveAnswersProvider("test-key")
    assert provider._semaphore._value == 2


async def test_brave_search_respects_max_concurrent_semaphore() -> None:
    """BraveSearchProvider.search() blocks when all semaphore slots are taken."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=_web_response(1))
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_search.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveSearchProvider("test-key", max_concurrent=1)
        assert provider._semaphore._value == 1

        await provider._semaphore.acquire()
        assert provider._semaphore._value == 0

        task = asyncio.create_task(provider.search("blocked"))
        await asyncio.sleep(0)
        assert not task.done()

        provider._semaphore.release()
        results = await task

    assert isinstance(results, list)


async def test_brave_answers_respects_max_concurrent_semaphore() -> None:
    """BraveAnswersProvider.search() blocks when all semaphore slots are taken."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"results": [{"title": "A", "text": "ans"}]})
    session_cm = _make_session_cm(mock_response)

    with patch(
        "custom_components.neuralbridge.providers.brave_answers.aiohttp.ClientSession",
        return_value=session_cm,
    ):
        provider = BraveAnswersProvider("test-key", max_concurrent=1)
        assert provider._semaphore._value == 1

        await provider._semaphore.acquire()
        assert provider._semaphore._value == 0

        task = asyncio.create_task(provider.search("blocked"))
        await asyncio.sleep(0)
        assert not task.done()

        provider._semaphore.release()
        results = await task

    assert isinstance(results, list)
