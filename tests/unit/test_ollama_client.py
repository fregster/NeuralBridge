"""Unit tests for OllamaClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from custom_components.neuralbridge.ollama_client import OllamaClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    status: int = 200,
    json_data: dict | None = None,
    text_data: str = "",
) -> AsyncMock:
    """Return a mock aiohttp response that works as an async context manager."""
    mock = AsyncMock()
    mock.status = status
    mock.json = AsyncMock(return_value=json_data or {})
    mock.text = AsyncMock(return_value=text_data)
    return mock


def _make_mock_session(mock_response: AsyncMock | None = None) -> MagicMock:
    """Return a mock aiohttp ClientSession.

    If mock_response is provided, session.post() returns a context manager
    that yields it.
    """
    session = MagicMock()
    session.close = AsyncMock()
    if mock_response is not None:
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_response)
        cm.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=cm)
    return session


# ---------------------------------------------------------------------------
# Test 1 — __init__: trailing slash stripped from base_url
# ---------------------------------------------------------------------------


def test_init_strips_trailing_slash() -> None:
    """Trailing slash is stripped from base_url in __init__."""
    client = OllamaClient("http://localhost:11434/", "llama3")
    assert client.base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Test 2 — __init__: model, timeout, and _session defaults
# ---------------------------------------------------------------------------


def test_init_stores_model_and_timeout() -> None:
    """model and timeout are stored; _session starts as None."""
    client = OllamaClient("http://localhost:11434", "llama3:8b", timeout=60)
    assert client.model == "llama3:8b"
    assert client.timeout == 60
    assert client._session is None


# ---------------------------------------------------------------------------
# Test 3 — generate(): 200 response returns response text
# ---------------------------------------------------------------------------


async def test_generate_success() -> None:
    """generate() returns the response text on a 200 reply."""
    mock_response = _make_mock_response(200, {"response": "Hello from Ollama!"})
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.generate("Say hello")
        await client.close()

    assert result == "Hello from Ollama!"


# ---------------------------------------------------------------------------
# Test 4 — generate(): non-200 status returns None
# ---------------------------------------------------------------------------


async def test_generate_non_200_returns_none() -> None:
    """generate() returns None when the server responds with a non-200 status."""
    mock_response = _make_mock_response(503, text_data="Service Unavailable")
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.generate("Hello")
        await client.close()

    assert result is None


# ---------------------------------------------------------------------------
# Test 5 — generate(): aiohttp.ClientError returns None
# ---------------------------------------------------------------------------


async def test_generate_client_error_returns_none() -> None:
    """generate() returns None when aiohttp raises ClientError."""
    mock_session = _make_mock_session()
    mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.generate("Hello")

    assert result is None


# ---------------------------------------------------------------------------
# Test 6 — generate(): unexpected exception returns None
# ---------------------------------------------------------------------------


async def test_generate_unexpected_exception_returns_none() -> None:
    """generate() returns None on any unexpected exception."""
    mock_session = _make_mock_session()
    mock_session.post = MagicMock(side_effect=RuntimeError("unexpected boom"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.generate("Hello")

    assert result is None


# ---------------------------------------------------------------------------
# Test 7 — generate(): session created lazily on first call
# ---------------------------------------------------------------------------


async def test_generate_creates_session_lazily() -> None:
    """_session is None before the first generate() call and set afterwards."""
    mock_response = _make_mock_response(200, {"response": "ok"})
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ) as mock_cls:
        client = OllamaClient("http://localhost:11434", "llama3")
        assert client._session is None  # not yet created

        await client.generate("Hello")

        mock_cls.assert_called_once()  # created exactly once
        assert client._session is mock_session  # now holds the session
        await client.close()


# ---------------------------------------------------------------------------
# Test 8 — close(): calls session.close() and resets _session to None
# ---------------------------------------------------------------------------


async def test_close_resets_session_to_none() -> None:
    """close() calls session.close() and resets _session to None."""
    mock_session = MagicMock()
    mock_session.close = AsyncMock()

    client = OllamaClient("http://localhost:11434", "llama3")
    client._session = mock_session  # inject directly, bypass lazy creation

    await client.close()

    mock_session.close.assert_called_once()
    assert client._session is None


# ---------------------------------------------------------------------------
# Test 9 — close(): no-op when session was never created
# ---------------------------------------------------------------------------


async def test_close_when_no_session_is_noop() -> None:
    """close() on a fresh client that has never made a request does not raise."""
    client = OllamaClient("http://localhost:11434", "llama3")
    assert client._session is None
    await client.close()  # must not raise
    assert client._session is None


# ---------------------------------------------------------------------------
# Test 10 — generate(): optional context dict included in POST body
# ---------------------------------------------------------------------------


async def test_generate_includes_context_in_payload() -> None:
    """generate() adds the context dict to the POST body when provided."""
    mock_response = _make_mock_response(200, {"response": "ok"})
    mock_session = _make_mock_session(mock_response)

    ctx = {"session": [1, 2, 3]}
    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        await client.generate("prompt", context=ctx)
        await client.close()

    _, call_kwargs = mock_session.post.call_args
    assert call_kwargs["json"]["context"] == ctx


# ---------------------------------------------------------------------------
# Test 11 — chat(): 200 response returns message content
# ---------------------------------------------------------------------------


async def test_chat_success() -> None:
    """chat() returns the message content field on a 200 reply."""
    mock_response = _make_mock_response(200, {"message": {"content": "Chat response!"}})
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.chat([{"role": "user", "content": "Hello"}])
        await client.close()

    assert result == "Chat response!"


# ---------------------------------------------------------------------------
# Test 12 — chat(): optional context dict included in POST body
# ---------------------------------------------------------------------------


async def test_chat_includes_context_in_payload() -> None:
    """chat() adds the context dict to the POST body when provided."""
    mock_response = _make_mock_response(200, {"message": {"content": "ok"}})
    mock_session = _make_mock_session(mock_response)

    ctx = {"session": [4, 5, 6]}
    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        await client.chat([{"role": "user", "content": "Hi"}], context=ctx)
        await client.close()

    _, call_kwargs = mock_session.post.call_args
    assert call_kwargs["json"]["context"] == ctx


# ---------------------------------------------------------------------------
# Test 13 — chat(): non-200 status returns None
# ---------------------------------------------------------------------------


async def test_chat_non_200_returns_none() -> None:
    """chat() returns None when the server responds with a non-200 status."""
    mock_response = _make_mock_response(503, text_data="Service Unavailable")
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.chat([{"role": "user", "content": "Hello"}])
        await client.close()

    assert result is None


# ---------------------------------------------------------------------------
# Test 14 — chat(): aiohttp.ClientError returns None
# ---------------------------------------------------------------------------


async def test_chat_client_error_returns_none() -> None:
    """chat() returns None when aiohttp raises ClientError."""
    mock_session = _make_mock_session()
    mock_session.post = MagicMock(side_effect=aiohttp.ClientError("refused"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.chat([{"role": "user", "content": "Hello"}])

    assert result is None


# ---------------------------------------------------------------------------
# Test 15 — chat(): unexpected exception returns None
# ---------------------------------------------------------------------------


async def test_chat_unexpected_exception_returns_none() -> None:
    """chat() returns None on any unexpected exception."""
    mock_session = _make_mock_session()
    mock_session.post = MagicMock(side_effect=RuntimeError("unexpected boom"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.chat([{"role": "user", "content": "Hello"}])

    assert result is None
