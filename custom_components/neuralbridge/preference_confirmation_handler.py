"""Preference suggestion confirmation handler for NeuralBridge."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import (
    CONF_ADAPTIVE_LEARNING_ENABLED,
    CONF_PREFERENCE_AUTO_CONFIRM_CORRECTIONS,
    CONF_PREFERENCE_SUGGESTION_COOLDOWN_DAYS,
    DEFAULT_ADAPTIVE_LEARNING_ENABLED,
    DEFAULT_PREFERENCE_AUTO_CONFIRM_CORRECTIONS,
    DEFAULT_PREFERENCE_SUGGESTION_COOLDOWN_DAYS,
    EVENT_PREFERENCE_LEARNED,
    PREFERENCE_CONFIRM_WORDS,
    PREFERENCE_REJECT_WORDS,
    SIGNAL_PREFERENCES_UPDATED,
)
from .preference_analyser import PreferenceAnalyser

if TYPE_CHECKING:
    from homeassistant.components.conversation import ConversationInput, ConversationResult

    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


class PreferenceConfirmationHandler:
    """Handles preference suggestion detection, confirmation, and rejection flows.

    Analyses user input for learnable preferences and manages the pending
    suggestion lifecycle until the user confirms or rejects.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this instance.
        """
        self._agent = agent

    def _fire_preferences_updated(self) -> None:
        """Fire the preferences-updated dispatcher signal for this config entry."""
        signal = SIGNAL_PREFERENCES_UPDATED.format(entry_id=self._agent._config_entry.entry_id)
        async_dispatcher_send(self._agent.hass, signal)

    async def _resolve_preference_confirmation(
        self, user_input: "ConversationInput"
    ) -> "ConversationResult | None":
        """Resolve a pending preference suggestion confirmation.

        Checks whether the user's text is a confirmation or rejection word.
        Only acts when there is a pending suggestion for this conversation ID.

        Args:
            user_input: The user's latest input.

        Returns:
            A ConversationResult acknowledging the decision, or None if no
            pending suggestion exists.
        """
        if self._agent._preference_memory is None:
            return None

        conv_id = user_input.conversation_id or "default"
        suggestion = self._agent._pending_preference_suggestions.get(conv_id)
        if suggestion is None:
            return None

        text_lower = user_input.text.strip().lower()
        is_confirm = text_lower in PREFERENCE_CONFIRM_WORDS
        is_reject = text_lower in PREFERENCE_REJECT_WORDS

        if not is_confirm and not is_reject:
            return None

        # Consume the pending suggestion regardless of decision
        del self._agent._pending_preference_suggestions[conv_id]

        if is_confirm:
            await self._agent._preference_memory.confirm(suggestion.key)
            self._agent.hass.bus.async_fire(
                EVENT_PREFERENCE_LEARNED,
                {
                    "key": suggestion.key,
                    "value": suggestion.value,
                    "category": suggestion.category,
                    "action": "confirmed",
                },
            )
            self._fire_preferences_updated()
            ack = f"✅ Done — I'll default to {suggestion.value} from now on."
            return self._agent._create_result(ack, user_input.conversation_id)

        # Rejection: remove the tentative entry
        await self._agent._preference_memory.reject(suggestion.key)
        self._fire_preferences_updated()
        return self._agent._create_result(
            "👍 No problem, I won't save that.", user_input.conversation_id
        )

    async def _analyse_and_suggest(
        self,
        user_input: "ConversationInput",
        result: "ConversationResult",
    ) -> "ConversationResult":
        """Analyse the user's text for learnable preferences and append suggestions.

        For auto-store patterns the preference is stored immediately.  For softer
        detections a suggestion is held pending until the user confirms.

        Args:
            user_input: The original user input (text to analyse).
            result: The agent's ConversationResult to potentially augment.

        Returns:
            The (possibly augmented) ConversationResult.
        """
        if self._agent._preference_memory is None:
            return result

        config = self._agent._get_config()
        if not config.get(CONF_ADAPTIVE_LEARNING_ENABLED, DEFAULT_ADAPTIVE_LEARNING_ENABLED):
            return result

        cooldown_days = int(
            config.get(
                CONF_PREFERENCE_SUGGESTION_COOLDOWN_DAYS,
                DEFAULT_PREFERENCE_SUGGESTION_COOLDOWN_DAYS,
            )
        )
        auto_confirm_corrections = config.get(
            CONF_PREFERENCE_AUTO_CONFIRM_CORRECTIONS, DEFAULT_PREFERENCE_AUTO_CONFIRM_CORRECTIONS
        )

        suggestion = PreferenceAnalyser().detect(user_input.text)
        if suggestion is None:
            return result

        key = suggestion.key
        conv_id = user_input.conversation_id or "default"

        # Respect cooldown unless this auto-stores (first-time correction)
        if not suggestion.auto_store and self._agent._preference_memory.is_suggestion_cooling_down(
            key, cooldown_days
        ):
            return result

        should_auto_confirm = suggestion.auto_store and auto_confirm_corrections
        await self._agent._preference_memory.upsert(
            key,
            suggestion.value,
            suggestion.category,
            confidence=suggestion.confidence,
            confirmed=should_auto_confirm,
        )
        await self._agent._preference_memory.record_suggestion(key)

        response_text = self._agent._extract_response_text(result)
        if should_auto_confirm:
            self._agent.hass.bus.async_fire(
                EVENT_PREFERENCE_LEARNED,
                {
                    "key": key,
                    "value": suggestion.value,
                    "category": suggestion.category,
                    "action": "auto_confirmed",
                },
            )
            self._fire_preferences_updated()
            ack = f"\n\n✅ Got it — I'll default to {suggestion.value} from now on."
            return self._agent._create_result(response_text + ack, user_input.conversation_id)

        # Soft suggestion — store pending and append question
        self._agent._pending_preference_suggestions[conv_id] = suggestion
        prompt = f"\n\n💡 {suggestion.suggestion_text}"
        return self._agent._create_result(response_text + prompt, user_input.conversation_id)
