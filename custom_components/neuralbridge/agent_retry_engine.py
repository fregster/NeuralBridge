"""Agent retry and tracking logic for the NeuralBridge pipeline.

Extracts the retry/circuit-breaker portion of ``PipelineExecutor`` into a
focused component that is easier to test and modify independently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import (
    CONF_AGENT_NAME,
    CONF_MAX_RETRIES,
    CONF_RETRY_BASE_DELAY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    SIGNAL_STATS_UPDATED,
)
from .router_engine import RouterDecision, _is_unhelpful_response

if TYPE_CHECKING:
    from homeassistant.components.conversation import ConversationInput, ConversationResult

    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class AgentRetryEngineProtocol(Protocol):
    """Protocol for agent retry and tracking logic."""

    async def _try_agent_with_tracking(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult | None:
        """Try a single agent with circuit breaker guard, then retry logic."""
        ...

    async def _try_agent_with_retries(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        agent_id: str,
        agent_name: str,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult | None:
        """Attempt an agent call with exponential back-off on failure."""
        ...

    def _record_agent_failure(self, agent_id: str, timed_out: bool) -> None:
        """Record a failure or timeout for circuit breaker, stats, and dispatcher."""
        ...


class AgentRetryEngine:
    """Handles circuit-breaker guarding, exponential back-off retries, and failure recording.

    All agent state (circuit breaker, statistics, etc.) is accessed via
    ``self._agent``.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this engine.
        """
        self._agent = agent

    async def _try_agent_with_tracking(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult | None:
        """Try a single agent with circuit breaker guard, then retry logic.

        Args:
            agent_config:    Configuration dict for the agent to try.
            user_input:      The user's conversation input.
            router_decision: Optional routing decision carrying ``relevant_sensors``
                             for targeted sensor injection.

        Returns:
            ConversationResult on success, None if the agent failed or was skipped.
        """
        agent_id: str = agent_config.get("id", "")
        agent_name: str = agent_config.get(CONF_AGENT_NAME, "Unknown")

        if self._agent._circuit_breaker.is_open(agent_id):
            _LOGGER.warning(
                "Agent %s circuit is tripped — skipping until cooldown expires", agent_name
            )
            return None

        return await self._try_agent_with_retries(
            agent_config, user_input, agent_id, agent_name, router_decision
        )

    async def _try_agent_with_retries(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        agent_id: str,
        agent_name: str,
        router_decision: RouterDecision | None = None,
    ) -> ConversationResult | None:
        """Attempt an agent call with exponential back-off on failure.

        Args:
            agent_config:    Configuration dict for the agent to try.
            user_input:      The user's conversation input.
            agent_id:        Unique identifier for the agent.
            agent_name:      Display name of the agent (for logging).
            router_decision: Optional routing decision carrying ``relevant_sensors``
                             for targeted sensor injection.

        Returns:
            ConversationResult on success, None if all attempts are exhausted.
        """
        config = self._agent._get_config()
        max_retries = int(config.get(CONF_MAX_RETRIES, DEFAULT_MAX_RETRIES))
        retry_base_delay = float(config.get(CONF_RETRY_BASE_DELAY, DEFAULT_RETRY_BASE_DELAY))

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = retry_base_delay * (2 ** (attempt - 1))
                _LOGGER.debug(
                    "Retry %d/%d for agent %s — waiting %.1fs before next attempt",
                    attempt,
                    max_retries,
                    agent_name,
                    delay,
                )
                await asyncio.sleep(delay)

            self._agent._statistics.record_request(agent_id, agent_name)
            start_time = time.monotonic()
            result, timed_out = await self._agent._try_agent(
                agent_config, user_input, router_decision
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000

            if result is not None:
                response_text = self._agent._pipeline._extract_response_text(result)
                if _is_unhelpful_response(response_text):
                    _LOGGER.info(
                        "Agent %s cannot answer — sentinel detected, will re-route",
                        agent_name,
                    )
                    # Return the canonical sentinel result so _try_processing_agents
                    # can distinguish re-route from real failure and add a preamble.
                    from .const import CANNOT_ANSWER_SENTINEL  # noqa: PLC0415

                    return self._agent._create_result(
                        CANNOT_ANSWER_SENTINEL, user_input.conversation_id
                    )
                return await self._agent._pipeline._handle_successful_result(
                    agent_config, result, user_input, agent_id, elapsed_ms
                )

            self._record_agent_failure(agent_id, timed_out)
            if self._agent._circuit_breaker.is_open(agent_id):
                _LOGGER.warning(
                    "Circuit tripped for agent %s after attempt %d — stopping retries",
                    agent_name,
                    attempt + 1,
                )
                break

        return None

    def _record_agent_failure(self, agent_id: str, timed_out: bool) -> None:
        """Record a failure or timeout for circuit breaker, stats, and dispatcher.

        Args:
            agent_id: Unique identifier for the agent that failed.
            timed_out: True if the failure was a timeout; False for other errors.
        """
        if timed_out:
            self._agent._circuit_breaker.record_timeout(agent_id)
            self._agent._statistics.record_timeout(agent_id)
        else:
            self._agent._circuit_breaker.record_failure(agent_id)
            self._agent._statistics.record_failure(agent_id)
        async_dispatcher_send(
            self._agent.hass,
            SIGNAL_STATS_UPDATED.format(entry_id=self._agent._config_entry.entry_id),
        )
