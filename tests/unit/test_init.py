"""Unit tests for NeuralBridge integration initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.neuralbridge import async_reload_entry, async_setup_entry, async_unload_entry
from custom_components.neuralbridge.agent_benchmark import AgentBenchmarker
from custom_components.neuralbridge.circuit_breaker import CircuitBreaker
from custom_components.neuralbridge.const import (
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_AGENTS,
    CONF_PRIORITY,
    DATA_BENCHMARKER,
    DATA_CIRCUIT_BREAKER,
    DATA_ENTITY_CONTEXT,
    DATA_PREFERENCE_MEMORY,
    DATA_SESSION_MEMORY,
    DATA_STATISTICS,
    DOMAIN,
    SERVICE_CLEAR_CONVERSATION,
    SERVICE_CLEAR_PREFERENCES,
    SERVICE_MANAGE_PREFERENCE,
    SERVICE_RUN_BENCHMARK,
)
from custom_components.neuralbridge.entity_context import EntityContextCache
from custom_components.neuralbridge.statistics import AgentStatistics

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.neuralbridge.session_memory import SessionMemory


async def test_setup_entry(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> None:
    """Test setting up the integration."""
    # Use _setup (defined below) which patches async_forward_entry_setups so the
    # test doesn't need the full HA integration loader to be active.
    result = await _setup(hass, mock_config_entry)
    assert result
    assert DOMAIN in hass.data


async def test_unload_entry(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> None:
    """Test unloading the integration."""
    await _setup(hass, mock_config_entry)
    result = await _unload(hass, mock_config_entry)
    assert result
    assert mock_config_entry.entry_id not in hass.data[DOMAIN]


# ---------------------------------------------------------------------------
# Helpers — mock platform loading so unit tests don't need the full HA loader
# ---------------------------------------------------------------------------

_SETUP_PATCH = "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"
_UNLOAD_PATCH = "homeassistant.config_entries.ConfigEntries.async_unload_platforms"


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> bool:
    """Set up a NeuralBridge entry with platform loading mocked out."""
    entry.add_to_hass(hass)
    with patch(_SETUP_PATCH, return_value=None):
        return await async_setup_entry(hass, entry)


async def _unload(hass: HomeAssistant, entry: MockConfigEntry) -> bool:
    """Unload a NeuralBridge entry with platform unloading mocked out."""
    with patch(_UNLOAD_PATCH, return_value=True):
        return await async_unload_entry(hass, entry)


# ---------------------------------------------------------------------------
# clear_conversation service — registration
# ---------------------------------------------------------------------------


async def test_clear_conversation_service_registered(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """clear_conversation service is registered after async_setup_entry."""
    await _setup(hass, mock_config_entry)

    assert hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION)


async def test_clear_conversation_service_not_registered_twice(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Calling async_setup_entry twice does not double-register the service."""
    await _setup(hass, mock_config_entry)
    # A second setup attempt (e.g. reload) must not raise
    await _setup(hass, mock_config_entry)

    assert hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION)


# ---------------------------------------------------------------------------
# clear_conversation service — handler clears session memory
# ---------------------------------------------------------------------------


async def test_clear_conversation_service_clears_session(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Calling clear_conversation removes the given conversation_id from SessionMemory."""
    await _setup(hass, mock_config_entry)

    # Seed session memory directly via hass.data
    memory: SessionMemory = hass.data[DOMAIN][mock_config_entry.entry_id][DATA_SESSION_MEMORY]
    memory.add_turn("conv-to-clear", "hello", "world")
    assert memory.session_count() == 1

    # Act: call the service
    await hass.services.async_call(
        DOMAIN,
        SERVICE_CLEAR_CONVERSATION,
        {"conversation_id": "conv-to-clear"},
        blocking=True,
    )

    assert memory.session_count() == 0


async def test_clear_conversation_service_noop_for_unknown_id(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Calling clear_conversation with an unknown ID does not raise."""
    await _setup(hass, mock_config_entry)

    # Should not raise even with a conversation_id that has no session
    await hass.services.async_call(
        DOMAIN,
        SERVICE_CLEAR_CONVERSATION,
        {"conversation_id": "nonexistent-session"},
        blocking=True,
    )


# ---------------------------------------------------------------------------
# clear_conversation service — unregistered on last entry unload
# ---------------------------------------------------------------------------


async def test_clear_conversation_service_unregistered_on_unload(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Service is removed from hass when the last entry is unloaded."""
    await _setup(hass, mock_config_entry)
    assert hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION)

    await _unload(hass, mock_config_entry)

    assert not hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION)


# ---------------------------------------------------------------------------
# async_reload_entry
# ---------------------------------------------------------------------------


async def test_async_reload_entry_calls_async_reload(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_reload_entry delegates to hass.config_entries.async_reload."""
    with patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(hass, mock_config_entry)

    mock_reload.assert_called_once_with(mock_config_entry.entry_id)


# ---------------------------------------------------------------------------
# DATA_CIRCUIT_BREAKER stored in hass.data
# ---------------------------------------------------------------------------


async def test_setup_entry_stores_circuit_breaker(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_setup_entry stores a CircuitBreaker instance in hass.data."""
    await _setup(hass, mock_config_entry)

    entry_data = hass.data[DOMAIN][mock_config_entry.entry_id]
    assert DATA_CIRCUIT_BREAKER in entry_data
    assert isinstance(entry_data[DATA_CIRCUIT_BREAKER], CircuitBreaker)


async def test_setup_entry_stores_statistics(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_setup_entry stores an AgentStatistics instance in hass.data."""
    await _setup(hass, mock_config_entry)

    entry_data = hass.data[DOMAIN][mock_config_entry.entry_id]
    assert DATA_STATISTICS in entry_data
    assert isinstance(entry_data[DATA_STATISTICS], AgentStatistics)


async def test_setup_entry_stores_entity_context_cache(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_setup_entry stores an EntityContextCache instance in hass.data."""
    await _setup(hass, mock_config_entry)

    entry_data = hass.data[DOMAIN][mock_config_entry.entry_id]
    assert DATA_ENTITY_CONTEXT in entry_data
    assert isinstance(entry_data[DATA_ENTITY_CONTEXT], EntityContextCache)


# ---------------------------------------------------------------------------
# Benchmark scheduling — re_benchmark_on_save
# ---------------------------------------------------------------------------


async def test_setup_entry_schedules_benchmark_when_re_benchmark_on_save(
    hass: HomeAssistant,
) -> None:
    """async_setup_entry creates a benchmark task when re_benchmark_on_save is True."""
    agent_cfg = {
        "id": "bench-agent-1",
        CONF_AGENT_TYPE: "ollama",
        CONF_AGENT_NAME: "Bench Agent",
        CONF_PRIORITY: 10,
        "url": "http://localhost:11434",
        "model": "test-model",
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge Bench",
        data={CONF_AGENTS: [agent_cfg]},
        unique_id="bench_test_entry",
    )

    scheduled_ids: list[str] = []

    original_ensure = AgentBenchmarker.ensure_profile

    def fake_ensure(self: AgentBenchmarker, cfg: dict) -> object:  # type: ignore[misc]
        profile = original_ensure(self, cfg)
        profile.re_benchmark_on_save = True
        return profile

    async def fake_schedule(self: AgentBenchmarker, agent_id: str, cfg: dict) -> None:  # type: ignore[misc]
        scheduled_ids.append(agent_id)

    with (
        patch(_SETUP_PATCH, return_value=None),
        patch.object(AgentBenchmarker, "ensure_profile", fake_ensure),
        patch.object(AgentBenchmarker, "async_schedule_benchmark", fake_schedule),
    ):
        entry.add_to_hass(hass)
        await async_setup_entry(hass, entry)
        await hass.async_block_till_done()

    assert "bench-agent-1" in scheduled_ids


# ---------------------------------------------------------------------------
# run_benchmark service handler
# ---------------------------------------------------------------------------


async def test_run_benchmark_service_triggers_schedule(hass: HomeAssistant) -> None:
    """run_benchmark service calls async_schedule_benchmark for matching agent."""
    agent_cfg = {
        "id": "svc-agent-1",
        CONF_AGENT_TYPE: "ollama",
        CONF_AGENT_NAME: "Svc Agent",
        CONF_PRIORITY: 10,
        "url": "http://localhost:11434",
        "model": "test-model",
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge Svc",
        data={CONF_AGENTS: [agent_cfg]},
        unique_id="svc_test_entry",
    )

    scheduled: list[str] = []

    async def fake_schedule(self: AgentBenchmarker, agent_id: str, cfg: dict) -> None:  # type: ignore[misc]
        scheduled.append(agent_id)

    with (
        patch(_SETUP_PATCH, return_value=None),
        patch.object(AgentBenchmarker, "async_schedule_benchmark", fake_schedule),
    ):
        entry.add_to_hass(hass)
        await async_setup_entry(hass, entry)
        # Ensure profile exists in the benchmarker
        nb: AgentBenchmarker = hass.data[DOMAIN][entry.entry_id][DATA_BENCHMARKER]
        nb.ensure_profile(agent_cfg)

        await hass.services.async_call(
            DOMAIN,
            SERVICE_RUN_BENCHMARK,
            {"agent_id": "svc-agent-1"},
            blocking=True,
        )

    assert "svc-agent-1" in scheduled


async def test_run_benchmark_service_no_op_when_profile_missing(hass: HomeAssistant) -> None:
    """run_benchmark service does nothing when agent has no profile."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge NoProfile",
        data={CONF_AGENTS: []},
        unique_id="noprofile_test_entry",
    )

    scheduled: list[str] = []

    async def fake_schedule(self: AgentBenchmarker, agent_id: str, cfg: dict) -> None:  # type: ignore[misc]
        scheduled.append(agent_id)

    with (
        patch(_SETUP_PATCH, return_value=None),
        patch.object(AgentBenchmarker, "async_schedule_benchmark", fake_schedule),
    ):
        entry.add_to_hass(hass)
        await async_setup_entry(hass, entry)

        await hass.services.async_call(
            DOMAIN,
            SERVICE_RUN_BENCHMARK,
            {"agent_id": "nonexistent-agent"},
            blocking=True,
        )

    assert "nonexistent-agent" not in scheduled


# ---------------------------------------------------------------------------
# __init__.py line 110 — continue when agent_id is empty
# ---------------------------------------------------------------------------


async def test_setup_entry_skips_agent_without_id(
    hass: "HomeAssistant",
) -> None:
    """async_setup_entry continues past agents that have no id field (line 110)."""
    agent_no_id = {
        # 'id' field intentionally absent — triggers the empty-id continue
        CONF_AGENT_TYPE: "ollama",
        CONF_AGENT_NAME: "No ID Agent",
        CONF_PRIORITY: 10,
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge No ID",
        data={CONF_AGENTS: [agent_no_id]},
        unique_id="no_id_test_entry",
    )

    async def fake_schedule(  # type: ignore[misc]
        self: AgentBenchmarker, agent_id: str, cfg: dict
    ) -> None:
        pass

    with (
        patch(_SETUP_PATCH, return_value=None),
        patch.object(AgentBenchmarker, "async_schedule_benchmark", fake_schedule),
    ):
        entry.add_to_hass(hass)
        result = await async_setup_entry(hass, entry)

    assert result  # setup should still succeed


# ---------------------------------------------------------------------------
# __init__.py line 177 — continue when DATA_BENCHMARKER is None
# ---------------------------------------------------------------------------


async def test_run_benchmark_service_skips_entry_with_none_benchmarker(
    hass: "HomeAssistant",
) -> None:
    """run_benchmark service skips entries where DATA_BENCHMARKER is None (line 177)."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge None Bench",
        data={CONF_AGENTS: []},
        unique_id="none_bench_test_entry",
    )

    with patch(_SETUP_PATCH, return_value=None):
        entry.add_to_hass(hass)
        await async_setup_entry(hass, entry)

    # Explicitly set DATA_BENCHMARKER to None so line 177 is hit
    hass.data[DOMAIN][entry.entry_id][DATA_BENCHMARKER] = None

    # Should not raise — the service handler just continues past this entry
    await hass.services.async_call(
        DOMAIN,
        SERVICE_RUN_BENCHMARK,
        {"agent_id": "any-agent"},
        blocking=True,
    )


# ---------------------------------------------------------------------------
# Preference services (Feature 15)
# ---------------------------------------------------------------------------


async def test_clear_preferences_service_clears_memory(
    hass: "HomeAssistant",
) -> None:
    """clear_preferences service invokes pm.clear() on the PreferenceMemory."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge Prefs Clear",
        data={CONF_AGENTS: []},
        unique_id="test_clear_prefs_entry",
    )

    with patch(_SETUP_PATCH, return_value=None):
        entry.add_to_hass(hass)
        await async_setup_entry(hass, entry)

    # Inject a mock PreferenceMemory
    mock_pm = AsyncMock()
    mock_pm.clear = AsyncMock(return_value=2)
    hass.data[DOMAIN][entry.entry_id][DATA_PREFERENCE_MEMORY] = mock_pm

    await hass.services.async_call(
        DOMAIN,
        SERVICE_CLEAR_PREFERENCES,
        {},
        blocking=True,
    )

    mock_pm.clear.assert_awaited_once()


async def test_clear_preferences_service_skips_entry_without_memory(
    hass: "HomeAssistant",
) -> None:
    """clear_preferences service is a no-op when DATA_PREFERENCE_MEMORY is None."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge Prefs None",
        data={CONF_AGENTS: []},
        unique_id="test_clear_prefs_none_entry",
    )

    with patch(_SETUP_PATCH, return_value=None):
        entry.add_to_hass(hass)
        await async_setup_entry(hass, entry)

    # Explicitly set preference memory to None
    hass.data[DOMAIN][entry.entry_id][DATA_PREFERENCE_MEMORY] = None

    # Should not raise
    await hass.services.async_call(
        DOMAIN,
        SERVICE_CLEAR_PREFERENCES,
        {},
        blocking=True,
    )


async def test_manage_preference_service_add(
    hass: "HomeAssistant",
) -> None:
    """manage_preference service with action='add' calls pm.upsert() and fires event."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge Manage Prefs Add",
        data={CONF_AGENTS: []},
        unique_id="test_manage_prefs_add_entry",
    )

    with patch(_SETUP_PATCH, return_value=None):
        entry.add_to_hass(hass)
        await async_setup_entry(hass, entry)

    mock_pm = AsyncMock()
    mock_pm.upsert = AsyncMock()
    mock_pm.reject = AsyncMock()
    hass.data[DOMAIN][entry.entry_id][DATA_PREFERENCE_MEMORY] = mock_pm

    await hass.services.async_call(
        DOMAIN,
        SERVICE_MANAGE_PREFERENCE,
        {"action": "add", "key": "news_source", "value": "BBC", "category": "source"},
        blocking=True,
    )

    mock_pm.upsert.assert_awaited_once()
    mock_pm.reject.assert_not_awaited()


async def test_manage_preference_service_remove(
    hass: "HomeAssistant",
) -> None:
    """manage_preference service with action='remove' calls pm.reject()."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge Manage Prefs Remove",
        data={CONF_AGENTS: []},
        unique_id="test_manage_prefs_remove_entry",
    )

    with patch(_SETUP_PATCH, return_value=None):
        entry.add_to_hass(hass)
        await async_setup_entry(hass, entry)

    mock_pm = AsyncMock()
    mock_pm.upsert = AsyncMock()
    mock_pm.reject = AsyncMock(return_value=True)
    hass.data[DOMAIN][entry.entry_id][DATA_PREFERENCE_MEMORY] = mock_pm

    await hass.services.async_call(
        DOMAIN,
        SERVICE_MANAGE_PREFERENCE,
        {"action": "remove", "key": "news_source"},
        blocking=True,
    )

    mock_pm.reject.assert_awaited_once_with("news_source")
    mock_pm.upsert.assert_not_awaited()


async def test_manage_preference_service_skips_entry_without_memory(
    hass: "HomeAssistant",
) -> None:
    """manage_preference service is a no-op when DATA_PREFERENCE_MEMORY is None."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge Manage Prefs No Mem",
        data={CONF_AGENTS: []},
        unique_id="test_manage_prefs_no_mem_entry",
    )

    with patch(_SETUP_PATCH, return_value=None):
        entry.add_to_hass(hass)
        await async_setup_entry(hass, entry)

    hass.data[DOMAIN][entry.entry_id][DATA_PREFERENCE_MEMORY] = None

    # Should not raise
    await hass.services.async_call(
        DOMAIN,
        SERVICE_MANAGE_PREFERENCE,
        {"action": "add", "key": "news_source", "value": "BBC", "category": "source"},
        blocking=True,
    )
