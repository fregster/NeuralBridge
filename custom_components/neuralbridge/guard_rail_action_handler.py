"""Guard rail action handler for NeuralBridge."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .const import (
    CONF_AGENT_NAME,
    CONF_GUARD_RAIL_ACTION,
    CONF_GUARD_RAIL_AI_THRESHOLD,
    CONF_GUARD_RAIL_DETOXIFY_THRESHOLD,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_GUARD_RAIL_RULES,
    CONF_GUARD_RAIL_USE_DETOXIFY,
    DEFAULT_GUARD_RAIL_ACTION,
    DEFAULT_GUARD_RAIL_AI_THRESHOLD,
    DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD,
    DEFAULT_GUARD_RAIL_ENABLED,
    DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    DEFAULT_GUARD_RAIL_USE_DETOXIFY,
    EVENT_GUARD_RAIL_TRIGGERED,
    GUARD_RAIL_ACTION_BLOCK,
    GUARD_RAIL_ACTION_NOTIFY_ASK,
    GUARD_RAIL_ACTION_WARN,
)
from .guard_rail import GuardRailChecker

if TYPE_CHECKING:
    from homeassistant.components.conversation import ConversationResult

    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


class GuardRailActionHandler:
    """Handles guard rail checks and action dispatch on agent output.

    Manages checker lifecycle, fires HA events on triggers, and applies the
    configured block / warn / notify-ask action.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this instance.
        """
        self._agent = agent

    async def _apply_guard_rail_action(
        self,
        action: str,
        result: "ConversationResult",
        response_text: str,
        conversation_id: str | None,
        guard_rail_result: Any,
    ) -> "ConversationResult | None":
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
        result: "ConversationResult",
        conversation_id: str | None,
    ) -> "ConversationResult | None":
        """Check guard rails on agent output and fire an HA event if triggered.

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

        guard_rail_agent_config = await self._get_guard_rail_agent_config()
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

        return await self._apply_guard_rail_action(
            action, result, response_text, conversation_id, guard_rail_result
        )

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
