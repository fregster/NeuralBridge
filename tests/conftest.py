"""Fixtures for NeuralBridge tests."""

from __future__ import annotations

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.neuralbridge.const import CONF_AGENTS, DOMAIN


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Return a mock config entry with the structure created by async_step_user."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge",
        data={CONF_AGENTS: []},
        unique_id="test_unique_id",
    )
