"""High-stakes confirmation handler for NeuralBridge."""

from __future__ import annotations

import hmac
import logging
from typing import TYPE_CHECKING, Any

from .const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    CONF_AGENT_NAME,
    CONF_HIGH_STAKES_DOMAINS,
    CONF_HIGH_STAKES_ENABLED,
    CONF_HIGH_STAKES_SECRET,
    CONF_HIGH_STAKES_SECRET_ENABLED,
    DEFAULT_HIGH_STAKES_DOMAINS,
    DEFAULT_HIGH_STAKES_ENABLED,
    DEFAULT_HIGH_STAKES_SECRET,
    DEFAULT_HIGH_STAKES_SECRET_ENABLED,
    EVENT_HIGH_STAKES_TRIGGERED,
)

if TYPE_CHECKING:
    from homeassistant.components.conversation import ConversationInput, ConversationResult

    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


class HighStakesHandler:
    """Handles high-stakes domain detection and confirmation flows.

    Detects when LOCAL_HA results target high-stakes domains and manages the
    passphrase / yes-no confirmation lifecycle.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this instance.
        """
        self._agent = agent

    async def _resolve_high_stakes_confirmation(
        self,
        user_input: "ConversationInput",
        hs_pending: "tuple[ConversationResult, list[str]]",
    ) -> "ConversationResult | None":
        """Resolve a pending high-stakes action confirmation.

        Compares the user's input against the configured passphrase (if enabled)
        or a simple yes/no.  Clears the pending cache entry regardless of outcome.

        Args:
            user_input: The latest user turn.
            hs_pending: Tuple of (original ConversationResult, entity_ids).

        Returns:
            The original ConversationResult on confirmation, an error result on
            denial/wrong passphrase, or None if the input is unrecognised.
        """
        original_result, _entity_ids = hs_pending
        conv_id = user_input.conversation_id or "default"
        config = self._agent._get_config()
        secret_enabled = config.get(
            CONF_HIGH_STAKES_SECRET_ENABLED, DEFAULT_HIGH_STAKES_SECRET_ENABLED
        )
        secret: str = config.get(CONF_HIGH_STAKES_SECRET, DEFAULT_HIGH_STAKES_SECRET)
        user_text = user_input.text.strip()

        if secret_enabled:
            # Passphrase mode — constant-time comparison prevents timing side-channel
            if hmac.compare_digest(user_text.strip().lower(), secret.lower()):
                await self._agent._high_stakes_cache.clear_pending(conv_id)
                return original_result
            # Wrong passphrase
            await self._agent._high_stakes_cache.clear_pending(conv_id)
            return self._agent._create_error_result(
                self._agent._localized("responses", "high_stakes_cancelled"),
                user_input.conversation_id,
            )
        else:
            # Simple yes/no mode
            if user_text.lower() == "yes":
                await self._agent._high_stakes_cache.clear_pending(conv_id)
                return original_result
            if user_text.lower() == "no":
                await self._agent._high_stakes_cache.clear_pending(conv_id)
                return self._agent._create_error_result(
                    self._agent._localized("responses", "high_stakes_cancelled"),
                    user_input.conversation_id,
                )
            # Not a recognised response — leave pending, return None to fall through
            return None

    async def _check_high_stakes(
        self,
        agent_config: dict[str, Any],
        result: "ConversationResult",
        user_input: "ConversationInput",
    ) -> "ConversationResult | None":
        """Intercept LOCAL_HA results that target high-stakes domains (Feature 4).

        Domain matching uses entity IDs captured from the proxy's
        ``_local_ha_targets`` dict.  Response-text keyword scanning is not used
        to avoid false-positive triggers on read-only queries.

        Args:
            agent_config: Configuration of the agent that produced the result.
            result: The ConversationResult that will be intercepted if necessary.
            user_input: The user's original conversation input.

        Returns:
            A confirmation-prompt ConversationResult if interception occurred,
            otherwise None.
        """
        config = self._agent._get_config()
        if not config.get(CONF_HIGH_STAKES_ENABLED, DEFAULT_HIGH_STAKES_ENABLED):
            return None

        agent_type = agent_config.get("agent_type")
        if agent_type not in (AGENT_TYPE_LOCAL_HA, AGENT_TYPE_INTEGRATED):
            return None

        conv_key = user_input.conversation_id or "default"
        hs_domains: list[str] = config.get(CONF_HIGH_STAKES_DOMAINS, DEFAULT_HIGH_STAKES_DOMAINS)

        # Use captured entity IDs from the LLM proxy — the only reliable detection method.
        entity_ids = self._agent._llm_proxy.consume_local_ha_targets(conv_key)
        matched = [eid for eid in entity_ids if eid.split(".")[0] in hs_domains]

        if not matched:
            return None

        # Store the result pending confirmation and fire the HA event
        await self._agent._high_stakes_cache.store_pending(conv_key, result, list(matched))
        self._agent.hass.bus.async_fire(
            EVENT_HIGH_STAKES_TRIGGERED,
            {
                "agent_name": agent_config.get(CONF_AGENT_NAME, "Unknown"),
                "agent_id": agent_config.get("id", ""),
                "entity_ids": list(matched),
                "secret_required": config.get(
                    CONF_HIGH_STAKES_SECRET_ENABLED, DEFAULT_HIGH_STAKES_SECRET_ENABLED
                ),
            },
        )

        _LOGGER.info("High-stakes confirmation required for: %s", ", ".join(matched))

        secret_enabled = config.get(
            CONF_HIGH_STAKES_SECRET_ENABLED, DEFAULT_HIGH_STAKES_SECRET_ENABLED
        )
        if secret_enabled:
            prompt = self._agent._localized("responses", "high_stakes_passphrase_prompt")
        else:
            prompt = self._agent._localized("responses", "high_stakes_confirmation")
        return self._agent._create_result(prompt, user_input.conversation_id)
