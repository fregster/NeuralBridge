"""Integration tests for NeuralBridge full conversation flow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


@pytest.mark.skip(reason="Integration tests pending implementation")
async def test_end_to_end_conversation_flow(hass: HomeAssistant) -> None:
    """Test complete conversation flow through all tiers."""
    # TODO: Implement end-to-end integration test
    pass
