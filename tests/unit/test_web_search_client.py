"""Unit tests for WebSearchClient and web search provider infrastructure."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.neuralbridge.const import (
    CONF_SEARCH_API_KEY,
    CONF_SEARCH_MAX_SNIPPET_LEN,
    CONF_SEARCH_PROVIDER,
    CONF_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_MAX_SNIPPET_LEN,
    DEFAULT_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_TIMEOUT,
    SEARCH_PROVIDER_BRAVE,
)
from custom_components.neuralbridge.web_search_client import (
    SEARCH_PROVIDER_MAP,
    BraveSearchProvider,
    SearchResult,
    WebSearchClient,
    _format_results,
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
    """Wrap a response mock inside the two-level context manager aiohttp uses."""
    get_cm = AsyncMock()
    get_cm.__aenter__ = AsyncMock(return_value=mock_response)
    get_cm.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.get = MagicMock(return_value=get_cm)

    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    return session_cm


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
        "custom_components.neuralbridge.web_search_client.aiohttp.ClientSession",
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
        "custom_components.neuralbridge.web_search_client.aiohttp.ClientSession",
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
        "custom_components.neuralbridge.web_search_client.aiohttp.ClientSession",
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
            "custom_components.neuralbridge.web_search_client.aiohttp.ClientSession",
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
    session.get = MagicMock(return_value=get_cm)
    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "custom_components.neuralbridge.web_search_client.aiohttp.ClientSession",
            return_value=session_cm,
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
    session.get = MagicMock(return_value=get_cm)
    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "custom_components.neuralbridge.web_search_client.aiohttp.ClientSession",
            return_value=session_cm,
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
            "custom_components.neuralbridge.web_search_client.aiohttp.ClientSession",
            return_value=session_cm,
        ),
        caplog.at_level(logging.DEBUG),
    ):
        provider = BraveSearchProvider(api_key=secret, timeout=5)
        await provider.search("q")

    for record in caplog.records:
        assert secret not in record.getMessage()


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
