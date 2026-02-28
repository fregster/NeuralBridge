"""Confirmation flows for NeuralBridge — guard rails, high-stakes checks, preferences."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .const import EVENT_ANNOUNCE_SENT
from .guard_rail_action_handler import GuardRailActionHandler
from .high_stakes_handler import HighStakesHandler
from .preference_confirmation_handler import PreferenceConfirmationHandler
from .router_engine import _ANNOUNCE_PREVIEW_MAX_LEN

if TYPE_CHECKING:
    from homeassistant.components.conversation import ConversationInput, ConversationResult

    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


@runtime_checkable
class ConfirmationHandlerProtocol(Protocol):
    """Interface contract for confirmation flow objects.

    Concrete implementations coordinate guard rail checks, high-stakes
    confirmation flows, and preference suggestion confirmations.
    """

    async def _handle_confirmation_check(
        self, user_input: "ConversationInput"
    ) -> "ConversationResult | None":
        """Handle a confirmation for a pending guard rail or high-stakes response."""
        ...

    async def _check_guardrails(
        self,
        agent_config: dict[str, Any],
        result: "ConversationResult",
        conversation_id: str | None,
    ) -> "ConversationResult | None":
        """Check guard rails on agent output."""
        ...

    async def _check_high_stakes(
        self,
        agent_config: dict[str, Any],
        result: "ConversationResult",
        user_input: "ConversationInput",
    ) -> "ConversationResult | None":
        """Intercept LOCAL_HA results that target high-stakes domains."""
        ...


# ---------------------------------------------------------------------------
# Thin coordinator class
# ---------------------------------------------------------------------------


class ConfirmationFlows:
    """Coordinates guard rail, high-stakes, preferences, and broadcast logic.

    Delegates to specialised sub-handlers for each concern while exposing a
    single stable interface for :class:`NeuralBridgeAgent`.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this instance.
        """
        self._agent = agent
        self._high_stakes_handler = HighStakesHandler(agent)
        self._pref_handler = PreferenceConfirmationHandler(agent)
        self._guard_rail_handler = GuardRailActionHandler(agent)

    async def _handle_confirmation_check(
        self, user_input: "ConversationInput"
    ) -> "ConversationResult | None":
        """Handle a confirmation for a pending guard rail or high-stakes response.

        Checks both the guard rail pending cache (yes/no only) and the
        high-stakes pending cache (passphrase or yes/no, depending on config).

        Args:
            user_input: The user's conversation input.

        Returns:
            A ConversationResult if a pending response was resolved, else None.
        """
        conv_id = user_input.conversation_id or "default"

        # --- High-stakes confirmation check (Feature 4) ---
        hs_pending = await self._agent._high_stakes_cache.get_pending(conv_id)
        if hs_pending is not None:
            hs_result = await self._high_stakes_handler._resolve_high_stakes_confirmation(
                user_input, hs_pending
            )
            if hs_result is not None:
                return hs_result

        # --- Feature 15: Preference suggestion confirmation ---
        pref_result = await self._pref_handler._resolve_preference_confirmation(user_input)
        if pref_result is not None:
            return pref_result

        # --- Guard rail confirmation check (yes/no only) ---
        if user_input.text.lower() not in ("yes", "no"):
            return None

        pending = await self._agent._guard_rail_cache.get_pending_response(conv_id)
        if not pending:
            return None

        response_text, _ = pending
        await self._agent._guard_rail_cache.clear_pending_response(conv_id)
        if user_input.text.lower() == "yes":
            return self._agent._create_result(response_text, user_input.conversation_id)
        return self._agent._create_error_result(
            self._agent._localized("responses", "guard_rail_blocked"), user_input.conversation_id
        )

    # ── Thin delegations (keep signatures stable for conversation.py) ──────

    async def _resolve_high_stakes_confirmation(
        self,
        user_input: "ConversationInput",
        hs_pending: "tuple[ConversationResult, list[str]]",
    ) -> "ConversationResult | None":
        """Delegate to HighStakesHandler."""
        return await self._high_stakes_handler._resolve_high_stakes_confirmation(
            user_input, hs_pending
        )

    async def _resolve_preference_confirmation(
        self, user_input: "ConversationInput"
    ) -> "ConversationResult | None":
        """Delegate to PreferenceConfirmationHandler."""
        return await self._pref_handler._resolve_preference_confirmation(user_input)

    def _fire_preferences_updated(self) -> None:
        """Delegate to PreferenceConfirmationHandler."""
        self._pref_handler._fire_preferences_updated()

    async def _analyse_and_suggest(
        self,
        user_input: "ConversationInput",
        result: "ConversationResult",
    ) -> "ConversationResult":
        """Delegate to PreferenceConfirmationHandler."""
        return await self._pref_handler._analyse_and_suggest(user_input, result)

    async def _apply_guard_rail_action(
        self,
        action: str,
        result: "ConversationResult",
        response_text: str,
        conversation_id: str | None,
        guard_rail_result: Any,
    ) -> "ConversationResult | None":
        """Delegate to GuardRailActionHandler."""
        return await self._guard_rail_handler._apply_guard_rail_action(
            action, result, response_text, conversation_id, guard_rail_result
        )

    async def _check_guardrails(
        self,
        agent_config: dict[str, Any],
        result: "ConversationResult",
        conversation_id: str | None,
    ) -> "ConversationResult | None":
        """Delegate to GuardRailActionHandler."""
        return await self._guard_rail_handler._check_guardrails(
            agent_config, result, conversation_id
        )

    async def _check_high_stakes(
        self,
        agent_config: dict[str, Any],
        result: "ConversationResult",
        user_input: "ConversationInput",
    ) -> "ConversationResult | None":
        """Delegate to HighStakesHandler."""
        return await self._high_stakes_handler._check_high_stakes(agent_config, result, user_input)

    async def _get_guard_rail_agent_config(self) -> dict[str, Any] | None:
        """Delegate to GuardRailActionHandler."""
        return await self._guard_rail_handler._get_guard_rail_agent_config()

    def _find_tts_entity(self) -> str | None:
        """Return the entity_id of the first TTS entity, or None."""
        for entity_id in self._agent.hass.states.async_entity_ids():
            if entity_id.startswith("tts."):
                return entity_id
        return None

    async def _send_broadcast_announcement(
        self,
        text: str,
        media_players: list[str],
        user_input: "ConversationInput",
    ) -> "ConversationResult":
        """Send *text* to every player in *media_players* via TTS.

        Args:
            text: The announcement message to speak.
            media_players: List of ``media_player.*`` entity IDs.
            user_input: Original conversation input (used for conversation_id).

        Returns:
            A confirmation result, or an error result when no TTS entity exists.
        """
        tts_entity = self._agent._find_tts_entity()
        if not tts_entity:
            return self._agent._create_error_result(
                "No text-to-speech (tts) entity found.", user_input.conversation_id
            )

        for player in media_players:
            await self._agent.hass.services.async_call(
                "tts",
                "speak",
                {
                    "entity_id": tts_entity,
                    "media_player_entity_id": player,
                    "message": text,
                    "cache": False,
                },
            )

        count = len(media_players)
        preview = (
            text[:_ANNOUNCE_PREVIEW_MAX_LEN] + "\u2026"
            if len(text) > _ANNOUNCE_PREVIEW_MAX_LEN
            else text
        )
        self._agent.hass.bus.async_fire(
            EVENT_ANNOUNCE_SENT,
            {"media_player_count": count, "text_preview": preview},
        )
        return self._agent._create_result(
            f"Message sent to {count} speaker{'s' if count != 1 else '.'}",
            user_input.conversation_id,
        )
