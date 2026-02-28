"""Router backend protocol for NeuralBridge.

Defines :class:`RouterBackendProtocol` — the structural interface that every
router back-end must implement.  Concrete implementations live in the
``router_backends/`` sub-package:

* :class:`.router_backends.OllamaRouterBackend` — Ollama direct-HTTP backend.
* :class:`.router_backends.IntegratedRouterBackend` — HA conversation-service
  backend.
"""

from __future__ import annotations

from typing import Any, Protocol


class RouterBackendProtocol(Protocol):
    """Structural interface for a router agent back-end.

    A router back-end is responsible for sending *prompt* to its underlying
    model and returning the raw text response.  It is completely stateless
    with respect to conversation history — each call is independent.

    Implementations must be async-safe and must **never** log the prompt
    content at INFO or above (the prompt may contain user text / PII).
    """

    async def call(self, router_config: dict[str, Any], prompt: str) -> str | None:
        """Send *prompt* to the router model and return the raw text response.

        Args:
            router_config: Configuration dict for this router agent (contains
                           URL, model, entity_id, timeout, etc.).
            prompt:        The fully-formatted classification prompt.

        Returns:
            Raw text response from the model, or ``None`` when the backend
            cannot fulfil the request (missing config, network error, timeout).
        """
        ...
