"""Ollama client for NeuralBridge."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)
_HTTP_OK = 200


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str, model: str, timeout: int = 30) -> None:
        """Initialize the Ollama client."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str | None:
        """Generate a response from Ollama."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }

            if context:
                payload["context"] = context

            if self._session is None:
                self._session = aiohttp.ClientSession()

            async with asyncio.timeout(self.timeout):
                async with self._session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    if response.status == _HTTP_OK:
                        data = await response.json()
                        return data.get("response")
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
    ) -> str | None:
        """Chat with Ollama using conversation history."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }

            if context:
                payload["context"] = context

            if self._session is None:
                self._session = aiohttp.ClientSession()

            async with asyncio.timeout(self.timeout):
                async with self._session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    if response.status == _HTTP_OK:
                        data = await response.json()
                        message = data.get("message", {})
                        return message.get("content")
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

    async def close(self) -> None:
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None
