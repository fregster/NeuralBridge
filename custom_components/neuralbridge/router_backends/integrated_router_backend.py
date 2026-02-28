"""HA conversation-service router backend.

Routes classification prompts to an installed Home Assistant conversation
entity (e.g. the HA Ollama integration, Gemini, OpenAI Conversation) via the
``conversation.process`` service.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN

from ..const import (
    CONF_AGENT_NAME,
    CONF_ENTITY_ID,
    CONF_TIMEOUT,
    DEFAULT_ROUTER_TIMEOUT,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class IntegratedRouterBackend:
    """Routes classification prompts to a HA conversation entity.

    Uses the ``conversation.process`` service with a fresh ``conversation_id``
    on every call — routing must be stateless.  Returns the first speech text
    from the response for JSON parsing by the caller.

    Responsibility: Bridge between RouterEngine and HA conversation agents.
    """

    def __init__(self, hass: "HomeAssistant") -> None:
        """Initialise with the Home Assistant instance.

        Args:
            hass: Home Assistant instance used for service calls and state lookups.
        """
        self._hass = hass

    async def call(self, router_config: dict[str, Any], prompt: str) -> str | None:
        """Send *prompt* to the configured HA conversation entity.

        Args:
            router_config: Router agent configuration dict containing at
                           minimum ``CONF_ENTITY_ID`` and optionally
                           ``CONF_TIMEOUT``.
            prompt:        The fully-formatted classification prompt.

        Returns:
            Speech text from the conversation response, or ``None`` when the
            entity is absent, not configured, or the call errors/times out.
        """
        agent_name: str = router_config.get(CONF_AGENT_NAME, "Unknown")
        entity_id: str = router_config.get(CONF_ENTITY_ID, "")
        timeout: int = router_config.get(CONF_TIMEOUT, DEFAULT_ROUTER_TIMEOUT)

        if not entity_id:
            _LOGGER.warning(
                "Router '%s' has no entity_id configured — applying fallback", agent_name
            )
            return None

        if not self._hass.states.get(entity_id):
            _LOGGER.warning("Routing agent entity %s not found", entity_id)
            return None

        try:
            async with asyncio.timeout(timeout):
                response = await self._hass.services.async_call(
                    CONVERSATION_DOMAIN,
                    "process",
                    {
                        "text": prompt,
                        "agent_id": entity_id,
                        # Fresh conversation each time — routing must be stateless
                        "conversation_id": None,
                    },
                    blocking=True,
                    return_response=True,
                )
        except asyncio.TimeoutError:
            _LOGGER.warning("Routing agent %s timed out after %d seconds", entity_id, timeout)
            return None
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error calling routing agent %s: %s", entity_id, err)
            return None

        return self._extract_speech_from_response(response)

    @staticmethod
    def _extract_speech_from_response(response: Any) -> str | None:
        """Extract the speech text from a ``conversation.process`` service call response.

        Traverses ``response["response"]["speech"]["plain"]["speech"]``.
        Returns the speech text if present and non-empty, otherwise ``None``.

        Args:
            response: The raw return value from ``hass.services.async_call``.

        Returns:
            The speech text string, or None if absent, empty, or malformed.
        """
        try:
            speech_text = response["response"]["speech"]["plain"]["speech"]
            return str(speech_text) if speech_text else None
        except (KeyError, TypeError, AttributeError):
            return None
