"""Unit tests for OllamaClient."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from custom_components.neuralbridge.ollama_client import OllamaClient, OllamaResponse

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
    session.closed = False
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

    assert result is not None
    assert result.content == "Hello from Ollama!"


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

    assert result is not None
    assert result.content == "Chat response!"


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


# ---------------------------------------------------------------------------
# Test 16 — OllamaResponse: tokens_per_second computes correctly
# ---------------------------------------------------------------------------


def test_ollama_response_tokens_per_second() -> None:
    """tokens_per_second computes eval_count / (eval_duration_ns / 1e9)."""
    resp = OllamaResponse(
        content="hello",
        eval_count=100,
        eval_duration_ns=1_000_000_000,  # 1 second
    )
    assert resp.tokens_per_second == 100.0


def test_ollama_response_tokens_per_second_none_when_missing() -> None:
    """tokens_per_second is None when eval_count or eval_duration_ns is absent."""
    assert OllamaResponse(content="x").tokens_per_second is None
    assert OllamaResponse(content="x", eval_count=50).tokens_per_second is None
    assert OllamaResponse(content="x", eval_duration_ns=1_000_000_000).tokens_per_second is None


def test_ollama_response_tokens_per_second_none_on_zero_duration() -> None:
    """tokens_per_second is None when eval_duration_ns is zero."""
    resp = OllamaResponse(content="x", eval_count=10, eval_duration_ns=0)
    assert resp.tokens_per_second is None


# ---------------------------------------------------------------------------
# Test 17 — OllamaResponse: prompt_tokens_per_second computes correctly
# ---------------------------------------------------------------------------


def test_ollama_response_prompt_tokens_per_second() -> None:
    """prompt_tokens_per_second computes from prompt_eval_count / duration."""
    resp = OllamaResponse(
        content="hello",
        prompt_eval_count=200,
        prompt_eval_duration_ns=2_000_000_000,  # 2 seconds
    )
    assert resp.prompt_tokens_per_second == 100.0


def test_ollama_response_prompt_tokens_per_second_none_when_missing() -> None:
    """prompt_tokens_per_second is None when required fields are absent."""
    assert OllamaResponse(content="x").prompt_tokens_per_second is None


# ---------------------------------------------------------------------------
# Test 18 — generate(): OllamaResponse populated with eval fields
# ---------------------------------------------------------------------------


async def test_generate_returns_eval_fields() -> None:
    """generate() populates OllamaResponse with eval counters from the API."""
    json_resp = {
        "response": "The answer",
        "eval_count": 42,
        "eval_duration": 500_000_000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 100_000_000,
    }
    mock_response = _make_mock_response(200, json_resp)
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.generate("Prompt")
        await client.close()

    assert result is not None
    assert result.content == "The answer"
    assert result.eval_count == 42
    assert result.eval_duration_ns == 500_000_000
    assert result.prompt_eval_count == 10
    assert result.prompt_eval_duration_ns == 100_000_000


# ---------------------------------------------------------------------------
# Test 19 — chat(): OllamaResponse populated with eval fields
# ---------------------------------------------------------------------------


async def test_chat_returns_eval_fields() -> None:
    """chat() populates OllamaResponse with eval counters from the API."""
    json_resp = {
        "message": {"content": "Chat answer"},
        "eval_count": 30,
        "eval_duration": 300_000_000,
        "prompt_eval_count": 8,
        "prompt_eval_duration": 80_000_000,
    }
    mock_response = _make_mock_response(200, json_resp)
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.chat([{"role": "user", "content": "Hi"}])
        await client.close()

    assert result is not None
    assert result.content == "Chat answer"
    assert result.eval_count == 30
    assert result.eval_duration_ns == 300_000_000


# ---------------------------------------------------------------------------
# Test 20 — async_show_model(): 200 response returns parsed dict
# ---------------------------------------------------------------------------


async def test_async_show_model_success() -> None:
    """async_show_model() returns the parsed JSON dict on a 200 response."""
    json_resp = {
        "details": {"family": "llama", "parameter_size": "7B"},
        "size": 4_000_000_000,
    }
    mock_response = _make_mock_response(200, json_resp)
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_show_model()
        await client.close()

    assert result is not None
    assert result["details"]["family"] == "llama"
    assert result["size"] == 4_000_000_000


async def test_async_show_model_non_200_returns_none() -> None:
    """async_show_model() returns None on a non-200 status."""
    mock_response = _make_mock_response(404, text_data="Not Found")
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "nonexistent-model")
        result = await client.async_show_model()

    assert result is None


async def test_async_show_model_client_error_returns_none() -> None:
    """async_show_model() returns None when aiohttp raises ClientError."""
    mock_session = _make_mock_session()
    mock_session.post = MagicMock(side_effect=aiohttp.ClientError("refused"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_show_model()

    assert result is None


async def test_async_show_model_unexpected_exception_returns_none() -> None:
    """async_show_model() returns None when an unexpected exception is raised."""
    mock_session = _make_mock_session()
    mock_session.post = MagicMock(side_effect=RuntimeError("unexpected!"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_show_model()

    assert result is None


async def test_generate_missing_response_key_returns_none() -> None:
    """generate() returns None when the JSON response has no 'response' key."""
    json_resp = {"eval_count": 5}  # no "response" key
    mock_response = _make_mock_response(200, json_resp)
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.generate("Prompt")
        await client.close()

    assert result is None


async def test_chat_missing_content_returns_none() -> None:
    """chat() returns None when the JSON response message has no 'content' key."""
    json_resp = {"message": {"role": "assistant"}}  # no "content" key
    mock_response = _make_mock_response(200, json_resp)
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.chat([{"role": "user", "content": "Hi"}])
        await client.close()

    assert result is None


def test_ollama_response_prompt_tokens_per_second_none_on_zero_duration() -> None:
    """prompt_tokens_per_second is None when prompt_eval_duration_ns is zero."""
    resp = OllamaResponse(content="x", prompt_eval_count=10, prompt_eval_duration_ns=0)
    assert resp.prompt_tokens_per_second is None


# ---------------------------------------------------------------------------
# P4 — _get_session() helper
# ---------------------------------------------------------------------------


def test_get_session_creates_session_with_user_agent_header() -> None:
    """_get_session() passes User-Agent: NeuralBridge/1.0 to ClientSession."""
    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
    ) as mock_cls:
        mock_session = MagicMock()
        mock_session.closed = False
        mock_cls.return_value = mock_session

        client = OllamaClient("http://localhost:11434", "llama3")
        returned = client._get_session()

    mock_cls.assert_called_once_with(headers={"User-Agent": "NeuralBridge/1.0"})
    assert returned is mock_session


def test_get_session_reuses_existing_open_session() -> None:
    """_get_session() returns the existing session when it is not closed."""
    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
    ) as mock_cls:
        mock_session = MagicMock()
        mock_session.closed = False
        mock_cls.return_value = mock_session

        client = OllamaClient("http://localhost:11434", "llama3")
        first = client._get_session()
        second = client._get_session()

    mock_cls.assert_called_once()  # only one session created
    assert first is second


def test_get_session_recreates_closed_session() -> None:
    """_get_session() creates a new session when the existing one is closed."""
    old_session = MagicMock()
    old_session.closed = True
    new_session = MagicMock()
    new_session.closed = False

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=new_session,
    ) as mock_cls:
        client = OllamaClient("http://localhost:11434", "llama3")
        client._session = old_session  # inject pre-closed session
        result = client._get_session()

    mock_cls.assert_called_once_with(headers={"User-Agent": "NeuralBridge/1.0"})
    assert result is new_session
    assert client._session is new_session


async def test_generate_reuses_session_across_calls() -> None:
    """generate() reuses the same ClientSession across multiple calls."""
    mock_response = _make_mock_response(200, {"response": "ok"})
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ) as mock_cls:
        client = OllamaClient("http://localhost:11434", "llama3")
        await client.generate("First call")
        await client.generate("Second call")
        await client.close()

    mock_cls.assert_called_once()  # session created only once


# ---------------------------------------------------------------------------
# P5 — Rate-limiting semaphore
# ---------------------------------------------------------------------------


def test_default_max_concurrent_is_four() -> None:
    """OllamaClient defaults to _DEFAULT_MAX_CONCURRENT=4 parallel requests."""
    client = OllamaClient("http://localhost:11434", "llama3")
    assert client._semaphore._value == 4


async def test_generate_respects_max_concurrent_semaphore() -> None:
    """generate() blocks when all semaphore slots are occupied."""
    mock_response = _make_mock_response(200, {"response": "ok"})
    mock_session = _make_mock_session(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3", max_concurrent=1)
        assert client._semaphore._value == 1

        # Occupy the sole semaphore slot manually.
        await client._semaphore.acquire()
        assert client._semaphore._value == 0

        # Dispatch a generate() call; it must block waiting for the slot.
        task = asyncio.create_task(client.generate("blocked"))
        await asyncio.sleep(0)  # yield to event loop
        assert not task.done()  # still waiting

        # Release the slot and let the task complete.
        client._semaphore.release()
        result = await task

    assert result is not None


# ---------------------------------------------------------------------------
# P6 — _get_json() GET helper
# ---------------------------------------------------------------------------


def _make_mock_session_with_get(mock_get_response: AsyncMock) -> MagicMock:
    """Return a mock session where session.get() returns a context manager."""
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_get_response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.get = MagicMock(return_value=cm)
    return session


async def test_get_json_200_returns_parsed_dict() -> None:
    """_get_json() returns parsed JSON dict on a 200 response."""
    mock_response = _make_mock_response(200, {"key": "value"})
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client._get_json("/api/ps")
        await client.close()

    assert result == {"key": "value"}
    mock_session.get.assert_called_once_with("http://localhost:11434/api/ps")


async def test_get_json_non_200_returns_none() -> None:
    """_get_json() returns None on a non-200 response."""
    mock_response = _make_mock_response(404, text_data="Not Found")
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client._get_json("/api/version")

    assert result is None


async def test_get_json_client_error_returns_none() -> None:
    """_get_json() returns None when aiohttp raises ClientError."""
    mock_session = _make_mock_session_with_get(_make_mock_response())
    mock_session.get = MagicMock(side_effect=aiohttp.ClientError("refused"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client._get_json("/api/ps")

    assert result is None


async def test_get_json_unexpected_exception_returns_none() -> None:
    """_get_json() returns None when an unexpected exception is raised."""
    mock_session = _make_mock_session_with_get(_make_mock_response())
    mock_session.get = MagicMock(side_effect=RuntimeError("boom"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client._get_json("/api/ps")

    assert result is None


# ---------------------------------------------------------------------------
# P7 — async_get_running_models()
# ---------------------------------------------------------------------------


async def test_async_get_running_models_returns_model_list() -> None:
    """async_get_running_models() returns the list of model dicts on 200."""
    models = [{"name": "llama3:8b", "size_vram": 4_900_000_000}]
    mock_response = _make_mock_response(200, {"models": models})
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_running_models()
        await client.close()

    assert result == models


async def test_async_get_running_models_empty_list() -> None:
    """async_get_running_models() returns empty list when no models are loaded."""
    mock_response = _make_mock_response(200, {"models": []})
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_running_models()

    assert result == []


async def test_async_get_running_models_missing_models_key_returns_empty() -> None:
    """async_get_running_models() returns empty list when 'models' key is absent."""
    mock_response = _make_mock_response(200, {"other": "data"})
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_running_models()

    assert result == []


async def test_async_get_running_models_non_list_models_returns_empty() -> None:
    """async_get_running_models() returns empty list when 'models' is not a list."""
    mock_response = _make_mock_response(200, {"models": "not-a-list"})
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_running_models()

    assert result == []


async def test_async_get_running_models_api_error_returns_empty() -> None:
    """async_get_running_models() returns empty list when the API call fails."""
    mock_session = _make_mock_session_with_get(_make_mock_response())
    mock_session.get = MagicMock(side_effect=aiohttp.ClientError("down"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_running_models()

    assert result == []


# ---------------------------------------------------------------------------
# P8 — async_get_version()
# ---------------------------------------------------------------------------


async def test_async_get_version_returns_version_string() -> None:
    """async_get_version() returns the version string on 200."""
    mock_response = _make_mock_response(200, {"version": "0.5.1"})
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_version()
        await client.close()

    assert result == "0.5.1"


async def test_async_get_version_missing_key_returns_none() -> None:
    """async_get_version() returns None when 'version' key is absent."""
    mock_response = _make_mock_response(200, {"other": "data"})
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_version()

    assert result is None


async def test_async_get_version_non_200_returns_none() -> None:
    """async_get_version() returns None on a non-200 response."""
    mock_response = _make_mock_response(503, text_data="Unavailable")
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_version()

    assert result is None


async def test_async_get_version_integer_version_coerced_to_string() -> None:
    """async_get_version() converts non-string version values to str."""
    mock_response = _make_mock_response(200, {"version": 5})
    mock_session = _make_mock_session_with_get(mock_response)

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_version()

    assert result == "5"


async def test_async_get_version_api_error_returns_none() -> None:
    """async_get_version() returns None when the API call fails."""
    mock_session = _make_mock_session_with_get(_make_mock_response())
    mock_session.get = MagicMock(side_effect=aiohttp.ClientError("down"))

    with patch(
        "custom_components.neuralbridge.ollama_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = OllamaClient("http://localhost:11434", "llama3")
        result = await client.async_get_version()

    assert result is None
