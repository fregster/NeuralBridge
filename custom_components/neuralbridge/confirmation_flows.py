"""Confirmation flows for NeuralBridge — guard rails, high-stakes checks, preferences."""

from __future__ import annotations

import hmac
import logging
import re
from typing import TYPE_CHECKING, Any

from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    CONF_ADAPTIVE_LEARNING_ENABLED,
    CONF_AGENT_NAME,
    CONF_GUARD_RAIL_ACTION,
    CONF_GUARD_RAIL_AI_THRESHOLD,
    CONF_GUARD_RAIL_DETOXIFY_THRESHOLD,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_GUARD_RAIL_RULES,
    CONF_GUARD_RAIL_USE_DETOXIFY,
    CONF_HIGH_STAKES_DOMAINS,
    CONF_HIGH_STAKES_ENABLED,
    CONF_HIGH_STAKES_SECRET,
    CONF_HIGH_STAKES_SECRET_ENABLED,
    CONF_PREFERENCE_AUTO_CONFIRM_CORRECTIONS,
    CONF_PREFERENCE_SUGGESTION_COOLDOWN_DAYS,
    DEFAULT_ADAPTIVE_LEARNING_ENABLED,
    DEFAULT_GUARD_RAIL_ACTION,
    DEFAULT_GUARD_RAIL_AI_THRESHOLD,
    DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD,
    DEFAULT_GUARD_RAIL_ENABLED,
    DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    DEFAULT_GUARD_RAIL_USE_DETOXIFY,
    DEFAULT_HIGH_STAKES_DOMAINS,
    DEFAULT_HIGH_STAKES_ENABLED,
    DEFAULT_HIGH_STAKES_SECRET,
    DEFAULT_HIGH_STAKES_SECRET_ENABLED,
    DEFAULT_PREFERENCE_AUTO_CONFIRM_CORRECTIONS,
    DEFAULT_PREFERENCE_SUGGESTION_COOLDOWN_DAYS,
    EVENT_ANNOUNCE_SENT,
    EVENT_GUARD_RAIL_TRIGGERED,
    EVENT_HIGH_STAKES_TRIGGERED,
    EVENT_PREFERENCE_LEARNED,
    GUARD_RAIL_ACTION_BLOCK,
    GUARD_RAIL_ACTION_NOTIFY_ASK,
    GUARD_RAIL_ACTION_WARN,
    PREFERENCE_CONFIRM_WORDS,
    PREFERENCE_REJECT_WORDS,
    SIGNAL_PREFERENCES_UPDATED,
)
from .guard_rail import GuardRailChecker
from .preference_analyser import PreferenceAnalyser
from .router_engine import _ANNOUNCE_PREVIEW_MAX_LEN

if TYPE_CHECKING:
    from homeassistant.components.conversation import ConversationInput, ConversationResult

    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


class ConfirmationFlows:
    """Encapsulates guard rail, high-stakes, preferences, and broadcast logic.

    Access to agent state is via ``self._agent``; within-class method calls
    stay as ``self.<method>()``.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this instance.
        """
        self._agent = agent

    async def _handle_confirmation_check(
        self, user_input: ConversationInput
    ) -> ConversationResult | None:
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
            hs_result = await self._agent._resolve_high_stakes_confirmation(user_input, hs_pending)
            if hs_result is not None:
                return hs_result

        # --- Feature 15: Preference suggestion confirmation ---
        pref_result = await self._agent._resolve_preference_confirmation(user_input)
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

    async def _resolve_high_stakes_confirmation(
        self,
        user_input: ConversationInput,
        hs_pending: tuple[ConversationResult, list[str]],
    ) -> ConversationResult | None:
        """Resolve a pending high-stakes action confirmation.

        Compares the user's input against the configured passphrase (if enabled)
        or a simple yes/no.  Clears the pending cache entry regardless of
        outcome.

        Args:
            user_input: The latest user turn.
            hs_pending: Tuple of (original ConversationResult, entity_ids).

        Returns:
            The original ConversationResult on confirmation, an error result on
            denial/wrong passphrase, or None if the input is not yet a
            recognisable confirmation (allowing the caller to fall through).
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

    async def _resolve_preference_confirmation(
        self, user_input: ConversationInput
    ) -> ConversationResult | None:
        """Resolve a pending preference suggestion confirmation.

        Checks whether the user's text is a confirmation or rejection word.
        Only acts when there is a pending suggestion for this conversation ID.

        Args:
            user_input: The user's latest input.

        Returns:
            A ConversationResult acknowledging the decision, or None if there
            is nothing to resolve.
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
            self._agent._fire_preferences_updated()
            ack = f"✅ Done — I'll default to {suggestion.value} from now on."
            return self._agent._create_result(ack, user_input.conversation_id)

        # Rejection: remove the tentative entry
        await self._agent._preference_memory.reject(suggestion.key)
        self._agent._fire_preferences_updated()
        return self._agent._create_result(
            "👍 No problem, I won't save that.", user_input.conversation_id
        )

    def _fire_preferences_updated(self) -> None:
        """Fire the preferences-updated dispatcher signal for this config entry."""
        signal = SIGNAL_PREFERENCES_UPDATED.format(entry_id=self._agent._config_entry.entry_id)
        async_dispatcher_send(self._agent.hass, signal)

    async def _analyse_and_suggest(
        self,
        user_input: ConversationInput,
        result: ConversationResult,
    ) -> ConversationResult:
        """Analyse the user's text for learnable preferences and append suggestions.

        For auto-store patterns (emphatic corrections) the preference is stored
        immediately and an acknowledgement is appended to the response.

        For softer detections a suggestion is held in ``_pending_preference_suggestions``
        and a prompt is appended asking the user to confirm.

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

        # Upsert a tentative entry (confirmed=False until the user says yes, or
        # auto_store + auto_confirm_corrections flips it to confirmed here).
        should_auto_confirm = suggestion.auto_store and auto_confirm_corrections
        await self._agent._preference_memory.upsert(
            key,
            suggestion.value,
            suggestion.category,
            confidence=suggestion.confidence,
            confirmed=should_auto_confirm,
        )
        await self._agent._preference_memory.record_suggestion(key)

        # Build augmented response text
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
            self._agent._fire_preferences_updated()
            ack = f"\n\n✅ Got it — I'll default to {suggestion.value} from now on."
            return self._agent._create_result(response_text + ack, user_input.conversation_id)

        # Soft suggestion — store pending and append question
        self._agent._pending_preference_suggestions[conv_id] = suggestion
        prompt = f"\n\n💡 {suggestion.suggestion_text}"
        return self._agent._create_result(response_text + prompt, user_input.conversation_id)

    async def _apply_guard_rail_action(
        self,
        action: str,
        result: ConversationResult,
        response_text: str,
        conversation_id: str | None,
        guard_rail_result: Any,
    ) -> ConversationResult | None:
        """Apply the configured guard rail action after a triggered rule.

        Args:
            action: The configured guard rail action key.
            result: The conversation result to modify or replace.
            response_text: The extracted response text.
            conversation_id: Conversation ID for pending-response caching.
            guard_rail_result: The guard rail check result.

        Returns:
            ConversationResult if the action produces a response, None otherwise.
        """
        if action == GUARD_RAIL_ACTION_BLOCK:
            return self._agent._create_error_result(
                self._agent._localized("responses", "guard_rail_blocked"), conversation_id
            )
        if action == GUARD_RAIL_ACTION_WARN:
            prefix = self._agent._localized("responses", "guard_rail_warning_prefix")
            result.response.async_set_speech(f"{prefix}{response_text}")
            return None
        if action == GUARD_RAIL_ACTION_NOTIFY_ASK:
            await self._agent._guard_rail_cache.store_pending_response(
                conversation_id or "default", response_text, guard_rail_result
            )
            return self._agent._create_result(
                self._agent._localized("responses", "guard_rail_notify_ask"), conversation_id
            )
        return None

    async def _check_guardrails(
        self,
        agent_config: dict[str, Any],
        result: ConversationResult,
        conversation_id: str | None,
    ) -> ConversationResult | None:
        """Check guard rails on agent output and fire an HA event if triggered (#13).

        Args:
            agent_config: Configuration of the agent that produced the result.
            result: The conversation result to check.
            conversation_id: Conversation ID for pending-response caching.

        Returns:
            ConversationResult if a guard rail action produced a response, None otherwise.
        """
        config = self._agent._get_config()

        guard_rail_enabled = config.get(CONF_GUARD_RAIL_ENABLED, DEFAULT_GUARD_RAIL_ENABLED)
        agent_guard_rail_enabled = agent_config.get(
            CONF_GUARD_RAIL_ENABLED_FOR_AGENT, DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
        )
        if not guard_rail_enabled or not agent_guard_rail_enabled:
            return None

        response_text = self._agent._extract_response_text(result)
        if not response_text:
            return None

        current_rules: dict[str, list[str]] | None = config.get(CONF_GUARD_RAIL_RULES)
        if current_rules != self._agent._guard_rail_rules_snapshot:
            self._agent._guard_rail_checker = None
            self._agent._guard_rail_rules_snapshot = current_rules

        if self._agent._guard_rail_checker is None:
            ai_threshold = config.get(CONF_GUARD_RAIL_AI_THRESHOLD, DEFAULT_GUARD_RAIL_AI_THRESHOLD)
            use_detoxify = config.get(CONF_GUARD_RAIL_USE_DETOXIFY, DEFAULT_GUARD_RAIL_USE_DETOXIFY)
            detoxify_threshold = config.get(
                CONF_GUARD_RAIL_DETOXIFY_THRESHOLD, DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD
            )
            self._agent._guard_rail_checker = GuardRailChecker(
                rules=current_rules,
                ai_threshold=ai_threshold,
                use_detoxify=use_detoxify,
                detoxify_threshold=detoxify_threshold,
            )
            await self._agent._guard_rail_checker.async_initialize()

        guard_rail_agent_config = await self._agent._get_guard_rail_agent_config()
        guard_rail_result = await self._agent._guard_rail_checker.check_output(
            response_text,
            use_ai=guard_rail_agent_config is not None,
            ai_agent_config=guard_rail_agent_config,
        )

        if guard_rail_result.is_safe:
            return None

        action = config.get(CONF_GUARD_RAIL_ACTION, DEFAULT_GUARD_RAIL_ACTION)
        agent_name = agent_config.get(CONF_AGENT_NAME, "Unknown")

        _LOGGER.warning(
            "Guard rail flagged output from agent %s: category=%s, confidence=%.2f, action=%s",
            agent_name,
            guard_rail_result.category,
            guard_rail_result.confidence,
            action,
        )

        # Fire HA event so users can build automations on guard rail triggers (#13)
        self._agent.hass.bus.async_fire(
            EVENT_GUARD_RAIL_TRIGGERED,
            {
                "agent_name": agent_name,
                "agent_id": agent_config.get("id", ""),
                "category": guard_rail_result.category,
                "confidence": round(guard_rail_result.confidence, 3),
                "action": action,
                "reason": guard_rail_result.reason,
            },
        )

        return await self._agent._apply_guard_rail_action(
            action, result, response_text, conversation_id, guard_rail_result
        )

    async def _check_high_stakes(
        self,
        agent_config: dict[str, Any],
        result: ConversationResult,
        user_input: ConversationInput,
    ) -> ConversationResult | None:
        """Intercept LOCAL_HA results that target high-stakes domains (Feature 4).

        Domain matching uses two complementary strategies:
          1. Entity IDs captured from the proxy's ``_local_ha_targets`` dict
             (populated by :class:`LLMAgentProxy` after each INTEGRATED call).
          2. Response-text keyword scan: checks whether any configured domain
             name appears as a standalone word in the agent's response speech.
             This is the fallback when no entity IDs were captured (e.g., the
             HA agent returned a result with no ``data.targets``).

        When the successful agent is LOCAL_HA (or EXISTING in assist mode) AND
        the targeted entities include a domain listed in
        ``CONF_HIGH_STAKES_DOMAINS``, the original result is stored in
        ``_high_stakes_cache`` and a confirmation prompt is returned instead.
        The caller may then confirm on the next turn via
        ``_resolve_high_stakes_confirmation``.

        Args:
            agent_config: Configuration of the agent that produced the result.
            result: The ConversationResult that will be intercepted if necessary.
            user_input: The user's original conversation input.

        Returns:
            A confirmation-prompt ConversationResult if interception occurred,
            otherwise None (meaning the original result should be used).
        """
        config = self._agent._get_config()
        if not config.get(CONF_HIGH_STAKES_ENABLED, DEFAULT_HIGH_STAKES_ENABLED):
            return None

        agent_type = agent_config.get("agent_type")
        if agent_type not in (AGENT_TYPE_LOCAL_HA, AGENT_TYPE_INTEGRATED):
            return None

        conv_key = user_input.conversation_id or "default"
        hs_domains: list[str] = config.get(CONF_HIGH_STAKES_DOMAINS, DEFAULT_HIGH_STAKES_DOMAINS)

        # Strategy 1: use captured entity IDs from the LLM proxy
        entity_ids = self._agent._llm_proxy.consume_local_ha_targets(conv_key)
        matched = [eid for eid in entity_ids if eid.split(".")[0] in hs_domains]

        # Strategy 2: keyword scan in response text when no entity IDs are available
        if not matched:
            response_text = self._agent._extract_response_text(result)
            lower_text = response_text.lower()
            matched = [d for d in hs_domains if re.search(rf"\b{re.escape(d)}\b", lower_text)]

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

        _LOGGER.info(
            "High-stakes confirmation required for: %s",
            ", ".join(matched),
        )

        secret_enabled = config.get(
            CONF_HIGH_STAKES_SECRET_ENABLED, DEFAULT_HIGH_STAKES_SECRET_ENABLED
        )
        if secret_enabled:
            prompt = self._agent._localized("responses", "high_stakes_passphrase_prompt")
        else:
            prompt = self._agent._localized("responses", "high_stakes_confirmation")
        return self._agent._create_result(prompt, user_input.conversation_id)

    async def _get_guard_rail_agent_config(self) -> dict[str, Any] | None:
        """Return the configuration for the designated guard rail agent.

        Returns:
            Agent configuration dict, or None if not configured.
        """
        from .const import CONF_AGENTS  # noqa: PLC0415

        config = self._agent._get_config()
        guard_rail_agent_id = config.get("guard_rail_agent_id")
        if not guard_rail_agent_id:
            return None

        for agent in config.get(CONF_AGENTS, []):
            agent_config: dict[str, Any] = agent
            if agent_config.get("id") == guard_rail_agent_id:
                return agent_config

        return None

    def _find_tts_entity(self) -> str | None:
        """Return the entity_id of the first TTS entity, or None.

        Iterates over all current HA states and returns the first entity
        whose ID starts with ``tts.``.

        Returns:
            A TTS entity_id string, or ``None`` if none is registered.
        """
        for entity_id in self._agent.hass.states.async_entity_ids():
            if entity_id.startswith("tts."):
                return entity_id
        return None

    async def _send_broadcast_announcement(
        self,
        text: str,
        media_players: list[str],
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Send *text* to every player in *media_players* via TTS.

        Calls the ``tts.speak`` HA service once per media player, then fires
        the :data:`EVENT_ANNOUNCE_SENT` event so other automations can react.

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
