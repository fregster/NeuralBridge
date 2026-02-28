"""Ollama direct-HTTP router backend.

Sends classification prompts to an Ollama server using
:class:`~.ollama_client.OllamaClient` and returns the raw text response.
"""

from __future__ import annotations

import logging
from typing import Any

from ..const import (
    CONF_AGENT_NAME,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_TIMEOUT,
    DEFAULT_ROUTER_TIMEOUT,
)
from ..ollama_client import OllamaClient

_LOGGER = logging.getLogger(__name__)


class OllamaRouterBackend:
    """Routes classification prompts to an Ollama server via direct HTTP.

    Maintains a per-agent :class:`OllamaClient` cache keyed by agent ID.
    Each unique (url, model, timeout) triplet is instantiated at most once.

    Responsibility: Send *prompt* to an Ollama model and return raw text.
    """

    def __init__(self, shared_clients: dict[str, OllamaClient] | None = None) -> None:
        """Initialise with an optional shared Ollama client cache.

        Args:
            shared_clients: An existing dict to use as the client cache.  When
                            provided the backend and caller share the same object
                            so mutations are visible to both sides.  When omitted
                            an empty private dict is created.
        """
        self._ollama_clients: dict[str, OllamaClient] = (
            shared_clients if shared_clients is not None else {}
        )

    async def call(self, router_config: dict[str, Any], prompt: str) -> str | None:
        """Send *prompt* to the configured Ollama router model.

        Args:
            router_config: Router agent configuration dict containing at
                           minimum ``CONF_OLLAMA_URL``, ``CONF_OLLAMA_MODEL``,
                           and optionally ``CONF_TIMEOUT``.
            prompt:        The fully-formatted classification prompt.

        Returns:
            Raw text response from the Ollama model, or ``None`` when required
            configuration is absent or the request fails.
        """
        agent_id: str = router_config.get("id", "")
        agent_name: str = router_config.get(CONF_AGENT_NAME, "Unknown")
        ollama_url: str = router_config.get(CONF_OLLAMA_URL, "")
        ollama_model: str = router_config.get(CONF_OLLAMA_MODEL, "")
        timeout: int = router_config.get(CONF_TIMEOUT, DEFAULT_ROUTER_TIMEOUT)

        if not ollama_url or not ollama_model:
            _LOGGER.warning(
                "Router '%s' (Ollama) is missing URL or model — applying fallback", agent_name
            )
            return None

        if agent_id not in self._ollama_clients:
            self._ollama_clients[agent_id] = OllamaClient(ollama_url, ollama_model, timeout)

        ollama_resp = await self._ollama_clients[agent_id].generate(prompt)
        return ollama_resp.content if ollama_resp is not None else None
