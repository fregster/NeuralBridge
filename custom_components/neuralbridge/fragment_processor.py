"""Compound-command fragment processing for the NeuralBridge pipeline.

When the user sends a compound command (e.g. "Turn on the lights and play jazz"),
``FragmentProcessor`` splits the input and processes each part independently,
combining the results into a single response.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .const import COMPOUND_COMMAND_SEPARATOR

if TYPE_CHECKING:
    from homeassistant.components.conversation import ConversationInput, ConversationResult

    from .conversation import NeuralBridgeAgent

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class FragmentProcessorProtocol(Protocol):
    """Protocol for compound-command fragment processing."""

    async def _process_compound_fragments(
        self,
        fragments: list[str],
        processing_agents: list[dict[str, Any]],
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Process each compound command fragment independently and combine results."""
        ...


class FragmentProcessor:
    """Processes compound command fragments and combines the results.

    A compound command is detected earlier in the pipeline and split into
    individual fragment strings.  This class dispatches each fragment through
    the normal agent pipeline and joins all the responses.
    """

    def __init__(self, agent: "NeuralBridgeAgent") -> None:
        """Initialise with a back-reference to the owning agent.

        Args:
            agent: The :class:`NeuralBridgeAgent` that owns this processor.
        """
        self._agent = agent

    async def _process_compound_fragments(
        self,
        fragments: list[str],
        processing_agents: list[dict[str, Any]],
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Process each compound command fragment independently and combine results.

        Each fragment is sent to the processing agents in turn.  If a fragment
        fails (no result), a localised fallback string is used so no fragment is
        silently dropped.  All per-fragment responses are joined with the
        ``COMPOUND_COMMAND_SEPARATOR`` (`` · ``).

        Args:
            fragments: Individual command strings from _split_compound_input.
            processing_agents: Priority-sorted processing agent configs.
            user_input: Original ConversationInput (text will be replaced per fragment).

        Returns:
            A ConversationResult whose speech text is the joined responses.
        """
        speech_parts: list[str] = []
        fallback_text = self._agent._localized("responses", "fallback")
        for fragment in fragments:
            fragment_input = dataclasses.replace(user_input, text=fragment)
            result = await self._agent._try_processing_agents(processing_agents, fragment_input)
            fragment_speech = self._agent._extract_response_text(result)
            speech_parts.append(fragment_speech if fragment_speech else fallback_text)
        combined = COMPOUND_COMMAND_SEPARATOR.join(speech_parts)
        return self._agent._create_result(combined, user_input.conversation_id)
