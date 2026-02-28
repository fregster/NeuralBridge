"""Ollama client for NeuralBridge."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)
_HTTP_OK = 200
_DEFAULT_MAX_CONCURRENT = 4


@dataclass
class OllamaResponse:
    """Structured response from an Ollama API call.

    Wraps the raw text content together with performance counters returned
    by the Ollama ``/api/generate`` and ``/api/chat`` endpoints.  All
    performance fields are ``None`` when the model omits them (e.g. tiny
    models or Ollama installations that do not report token counters).

    Attributes:
        content:                    The generated text (assistant turn content).
        eval_count:                 Number of output tokens generated.
        eval_duration_ns:           Wall-clock nanoseconds spent generating
                                    output tokens.
        prompt_eval_count:          Number of input prompt tokens processed.
        prompt_eval_duration_ns:    Wall-clock nanoseconds spent evaluating
                                    the input prompt.
    """

    content: str
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None

    @property
    def tokens_per_second(self) -> float | None:
        """Compute output token generation speed in tokens/second.

        Returns:
            Tokens per second, or ``None`` when the required counters are absent
            or ``eval_duration_ns`` is zero.
        """
        if self.eval_count is None or self.eval_duration_ns is None:
            return None
        if self.eval_duration_ns == 0:
            return None
        return self.eval_count / (self.eval_duration_ns / 1e9)

    @property
    def prompt_tokens_per_second(self) -> float | None:
        """Compute prompt ingestion speed in tokens/second.

        Returns:
            Tokens per second, or ``None`` when the required counters are absent
            or ``prompt_eval_duration_ns`` is zero.
        """
        if self.prompt_eval_count is None or self.prompt_eval_duration_ns is None:
            return None
        if self.prompt_eval_duration_ns == 0:
            return None
        return self.prompt_eval_count / (self.prompt_eval_duration_ns / 1e9)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 30,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    ) -> None:
        """Initialize the Ollama client."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared HTTP session, creating it lazily if needed.

        A single session is reused across all API calls to benefit from
        HTTP keep-alive.  The ``User-Agent`` header identifies NeuralBridge
        traffic in Ollama server logs.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "NeuralBridge/1.0"},
            )
        return self._session

    async def _get_json(self, path: str) -> dict[str, Any] | None:
        """Perform a GET request to *path* and return the parsed JSON body.

        Shares the session and timeout with other API methods.

        Args:
            path: URL path relative to :attr:`base_url`, e.g. ``"/api/ps"``.

        Returns:
            Parsed JSON dict on HTTP 200, otherwise ``None``.
        """
        try:
            session = self._get_session()
            async with asyncio.timeout(self.timeout):
                async with session.get(f"{self.base_url}{path}") as response:
                    if response.status == _HTTP_OK:
                        return await response.json()  # type: ignore[no-any-return]
                    _LOGGER.error("Ollama %s returned status %d", path, response.status)
                    return None
        except aiohttp.ClientError as err:
            _LOGGER.error("Error calling Ollama %s: %s", path, err)
            return None
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected error calling Ollama %s: %s", path, err)
            return None

    async def generate(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> OllamaResponse | None:
        """Generate a response from Ollama.

        Args:
            prompt:  The text prompt to send to the model.
            context: Optional Ollama context dict for conversation continuity.

        Returns:
            An :class:`OllamaResponse` on success, ``None`` on error.
        """
        async with self._semaphore:
            try:
                payload: dict[str, Any] = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                }

                if context:
                    payload["context"] = context

                session = self._get_session()

                async with asyncio.timeout(self.timeout):
                    async with session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                    ) as response:
                        if response.status == _HTTP_OK:
                            data = await response.json()
                            content: str | None = data.get("response")
                            if content is None:
                                return None
                            return OllamaResponse(
                                content=content,
                                eval_count=data.get("eval_count"),
                                eval_duration_ns=data.get("eval_duration"),
                                prompt_eval_count=data.get("prompt_eval_count"),
                                prompt_eval_duration_ns=data.get("prompt_eval_duration"),
                            )
                        else:
                            _LOGGER.error(
                                "Ollama API returned status %d: %s",
                                response.status,
                                await response.text(),
                            )
                            return None

            except aiohttp.ClientError as err:
                _LOGGER.error("Error communicating with Ollama: %s", err)
                return None
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected error calling Ollama: %s", err)
                return None

    async def chat(
        self, messages: list[dict[str, str]], context: dict[str, Any] | None = None
    ) -> OllamaResponse | None:
        """Chat with Ollama using conversation history.

        Args:
            messages: Chat message list in ``[{"role": ..., "content": ...}]``
                      format.
            context:  Optional Ollama context dict.

        Returns:
            An :class:`OllamaResponse` on success, ``None`` on error.
        """
        async with self._semaphore:
            try:
                payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                }

                if context:
                    payload["context"] = context

                session = self._get_session()

                async with asyncio.timeout(self.timeout):
                    async with session.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    ) as response:
                        if response.status == _HTTP_OK:
                            data = await response.json()
                            message = data.get("message", {})
                            content: str | None = message.get("content")
                            if content is None:
                                return None
                            return OllamaResponse(
                                content=content,
                                eval_count=data.get("eval_count"),
                                eval_duration_ns=data.get("eval_duration"),
                                prompt_eval_count=data.get("prompt_eval_count"),
                                prompt_eval_duration_ns=data.get("prompt_eval_duration"),
                            )
                        else:
                            _LOGGER.error(
                                "Ollama API returned status %d: %s",
                                response.status,
                                await response.text(),
                            )
                            return None

            except aiohttp.ClientError as err:
                _LOGGER.error("Error communicating with Ollama: %s", err)
                return None
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected error calling Ollama: %s", err)
                return None

    async def async_show_model(self) -> dict[str, Any] | None:
        """Fetch model metadata from the Ollama ``/api/show`` endpoint.

        Returns metadata such as parameter count, quantization level, context
        window size, and model family — used by :class:`AgentBenchmarker` for
        Layer 1 (metadata discovery).

        Returns:
            The full parsed JSON response dict from ``/api/show``, or ``None``
            on connection error or non-200 status.
        """
        try:
            session = self._get_session()

            async with asyncio.timeout(self.timeout):
                async with session.post(
                    f"{self.base_url}/api/show",
                    json={"name": self.model},
                ) as response:
                    if response.status == _HTTP_OK:
                        return await response.json()  # type: ignore[no-any-return]
                    _LOGGER.error("Ollama /api/show returned status %d", response.status)
                    return None

        except aiohttp.ClientError as err:
            _LOGGER.error("Error calling Ollama /api/show: %s", err)
            return None
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected error calling Ollama /api/show: %s", err)
            return None

    async def async_get_running_models(self) -> list[dict[str, Any]]:
        """Fetch currently loaded models from the Ollama ``/api/ps`` endpoint.

        Returns the list of model dicts reported by Ollama's process-status
        endpoint.  Each dict typically contains ``name``, ``size``, and
        ``size_vram`` fields.  Used as a pre-flight check before benchmarking
        to detect unexpected resource contention on shared hosts.

        Returns:
            List of model metadata dicts; empty list on any error or when no
            models are currently loaded.
        """
        data = await self._get_json("/api/ps")
        if data is None:
            return []
        models = data.get("models")
        return models if isinstance(models, list) else []

    async def async_get_version(self) -> str | None:
        """Fetch the Ollama server version from the ``/api/version`` endpoint.

        Used by :class:`~.benchmark_host_classifier.BenchmarkHostClassifier`
        to help fingerprint distinct Ollama server instances (two URLs that
        return the same version string from the same resolved host are almost
        certainly the same server).

        Returns:
            Version string such as ``"0.5.1"``, or ``None`` on any error.
        """
        data = await self._get_json("/api/version")
        if data is None:
            return None
        version = data.get("version")
        return str(version) if version is not None else None

    async def close(self) -> None:
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None
