"""Unit tests for NeuralBridge config flow."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.neuralbridge.config_flow import (
    NeuralBridgeConfigFlow,
    NeuralBridgeOptionsFlowHandler,
)
from custom_components.neuralbridge.const import (
    AGENT_TYPE_EXISTING,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    CONF_AGENT_CACHE_ENABLED,
    CONF_AGENT_ENABLED,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_AGENTS,
    CONF_ENTITY_ID,
    CONF_GUARD_RAIL_ACTION,
    CONF_GUARD_RAIL_AI_THRESHOLD,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_LANGUAGE,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_PRIORITY,
    CONF_RESPONSE_CACHE_ENABLED,
    CONF_RESPONSE_CACHE_TTL,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DATA_RESPONSE_CACHE,
    DEFAULT_AGENT_CACHE_ENABLED,
    DEFAULT_GUARD_RAIL_ACTION,
    DEFAULT_GUARD_RAIL_AI_THRESHOLD,
    DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    DEFAULT_LANGUAGE,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    DOMAIN,
    GUARD_RAIL_ACTION_BLOCK,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_SESSION = "custom_components.neuralbridge.config_flow.aiohttp.ClientSession"


def _make_config_flow(hass: HomeAssistant) -> NeuralBridgeConfigFlow:
    """Create a NeuralBridgeConfigFlow with hass and flow attrs injected."""
    flow = NeuralBridgeConfigFlow()
    flow.hass = hass
    flow.flow_id = "test-flow-id"  # type: ignore[attr-defined]
    flow.handler = DOMAIN  # type: ignore[attr-defined]
    return flow


def _make_handler(
    mock_config_entry: MockConfigEntry, hass: HomeAssistant
) -> NeuralBridgeOptionsFlowHandler:
    """Create a NeuralBridgeOptionsFlowHandler with hass and flow attrs injected."""
    handler = NeuralBridgeOptionsFlowHandler(mock_config_entry)
    handler.hass = hass
    # flow_id and handler are normally set by the HA flow manager; provide
    # sentinel values so async_show_form / async_show_menu don't raise.
    handler.flow_id = "test-flow-id"  # type: ignore[attr-defined]
    handler.handler = DOMAIN  # type: ignore[attr-defined]
    return handler


def _entry_with_agents(hass: HomeAssistant, agents: list[dict[str, Any]]) -> MockConfigEntry:
    """Create a MockConfigEntry pre-populated with agents and register it with hass."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="NeuralBridge",
        data={CONF_AGENTS: agents},
    )
    entry.add_to_hass(hass)
    return entry


def _make_session_cm(status: int = 200, models: list[str] | None = None) -> MagicMock:
    """Build nested async-context-manager mocks for _validate_ollama_connection.

    Simulates::

        async with aiohttp.ClientSession() as session:
            async with session.get(...) as response:
                ...
    """
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value={"models": [{"name": m} for m in (models or [])]})

    get_cm = MagicMock()
    get_cm.__aenter__ = AsyncMock(return_value=mock_response)
    get_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=get_cm)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    return session_cm


def _ollama_agent(agent_id: str = "agent-1", **overrides: Any) -> dict[str, Any]:
    """Return a minimal Ollama agent config dict."""
    base: dict[str, Any] = {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "Test Ollama",
        CONF_PRIORITY: 10,
        CONF_OLLAMA_URL: DEFAULT_OLLAMA_URL,
        CONF_OLLAMA_MODEL: "llama3:8b",
        CONF_TIMEOUT: DEFAULT_TIMEOUT,
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_AGENT_CACHE_ENABLED: DEFAULT_AGENT_CACHE_ENABLED,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    }
    base.update(overrides)
    return base


def _existing_agent(agent_id: str = "agent-2", **overrides: Any) -> dict[str, Any]:
    """Return a minimal existing-integration agent config dict."""
    base: dict[str, Any] = {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "Test Existing",
        CONF_PRIORITY: 20,
        CONF_ENTITY_ID: "conversation.openai",
        CONF_TIMEOUT: DEFAULT_TIMEOUT,
        CONF_AGENT_CACHE_ENABLED: DEFAULT_AGENT_CACHE_ENABLED,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    }
    base.update(overrides)
    return base


def _local_agent(agent_id: str = "agent-3", **overrides: Any) -> dict[str, Any]:
    """Return a minimal local HA agent config dict."""
    base: dict[str, Any] = {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "Home Assistant",
        CONF_PRIORITY: 50,
        CONF_ENTITY_ID: "conversation.home_assistant",
        CONF_TIMEOUT: DEFAULT_TIMEOUT,
        CONF_AGENT_CACHE_ENABLED: DEFAULT_AGENT_CACHE_ENABLED,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test 1 — _validate_ollama_connection: invalid URL scheme
# ---------------------------------------------------------------------------


async def test_validate_ollama_connection_invalid_scheme(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Non-http/https scheme returns 'invalid_url_scheme' without making an HTTP call."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler._validate_ollama_connection("ftp://localhost:11434", "llama3")
    assert result == "invalid_url_scheme"


# ---------------------------------------------------------------------------
# Test 2 — _validate_ollama_connection: aiohttp.ClientError → 'cannot_connect'
# ---------------------------------------------------------------------------


async def test_validate_ollama_connection_client_error(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """aiohttp.ClientError raised during GET returns 'cannot_connect'."""
    handler = _make_handler(mock_config_entry, hass)

    get_cm = MagicMock()
    get_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("connection refused"))
    get_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=get_cm)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with patch(_PATCH_SESSION, return_value=session_cm):
        result = await handler._validate_ollama_connection("http://localhost:11434", "llama3")

    assert result == "cannot_connect"


# ---------------------------------------------------------------------------
# Test 3 — _validate_ollama_connection: non-200 status → 'cannot_connect'
# ---------------------------------------------------------------------------


async def test_validate_ollama_connection_non_200(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """HTTP 503 from /api/tags returns 'cannot_connect'."""
    handler = _make_handler(mock_config_entry, hass)
    with patch(_PATCH_SESSION, return_value=_make_session_cm(status=503)):
        result = await handler._validate_ollama_connection("http://localhost:11434", "llama3")
    assert result == "cannot_connect"


# ---------------------------------------------------------------------------
# Test 4 — _validate_ollama_connection: model absent → 'model_not_found'
# ---------------------------------------------------------------------------


async def test_validate_ollama_connection_model_not_found(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Requested model absent from the server's model list returns 'model_not_found'."""
    handler = _make_handler(mock_config_entry, hass)
    session_cm = _make_session_cm(status=200, models=["codellama", "mistral"])
    with patch(_PATCH_SESSION, return_value=session_cm):
        result = await handler._validate_ollama_connection("http://localhost:11434", "llama3")
    assert result == "model_not_found"


# ---------------------------------------------------------------------------
# Test 5 — _validate_ollama_connection: model present → None (success)
# ---------------------------------------------------------------------------


async def test_validate_ollama_connection_success(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Model present in server list returns None (no error)."""
    handler = _make_handler(mock_config_entry, hass)
    session_cm = _make_session_cm(status=200, models=["llama3", "codellama"])
    with patch(_PATCH_SESSION, return_value=session_cm):
        result = await handler._validate_ollama_connection("http://localhost:11434", "llama3")
    assert result is None


# ---------------------------------------------------------------------------
# Test 6 — _validate_ollama_connection: unexpected exception → 'unknown'
# ---------------------------------------------------------------------------


async def test_validate_ollama_connection_unexpected_error(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Unexpected exception during validation returns 'unknown'."""
    handler = _make_handler(mock_config_entry, hass)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("unexpected"))
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with patch(_PATCH_SESSION, return_value=session_cm):
        result = await handler._validate_ollama_connection("http://localhost:11434", "llama3")
    assert result == "unknown"


# ---------------------------------------------------------------------------
# Test 7 — Config flow: async_step_user shows form
# ---------------------------------------------------------------------------


async def test_user_step_shows_form(hass: HomeAssistant) -> None:
    """async_step_user(None) returns a FORM result with step_id='user'."""
    flow = _make_config_flow(hass)
    result = await flow.async_step_user(None)
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"


# ---------------------------------------------------------------------------
# Test 8 — Config flow: async_step_user calls async_create_entry correctly
# ---------------------------------------------------------------------------


async def test_user_step_creates_entry(hass: HomeAssistant) -> None:
    """Submitting the user step calls async_create_entry with NeuralBridge title."""
    flow = _make_config_flow(hass)
    with (
        patch.object(flow, "async_set_unique_id", return_value=None),
        patch.object(flow, "_abort_if_unique_id_configured"),
        patch.object(
            flow, "async_create_entry", return_value={"type": FlowResultType.CREATE_ENTRY}
        ) as mock_create,
    ):
        await flow.async_step_user({})

    mock_create.assert_called_once_with(
        title="NeuralBridge",
        data={CONF_AGENTS: [], CONF_LANGUAGE: DEFAULT_LANGUAGE},
    )


# ---------------------------------------------------------------------------
# Test 9 — Options flow: init shows menu
# ---------------------------------------------------------------------------


async def test_options_flow_init_shows_menu(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_init returns a MENU result with the 5 expected options."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_init()
    assert result["type"] == FlowResultType.MENU
    assert set(result["menu_options"]) == {
        "add_agent",
        "manage_agents",
        "configure_guard_rails",
        "advanced_settings",
        "language_settings",
    }


# ---------------------------------------------------------------------------
# Test 10 — Options flow: add_agent shows form when no input provided
# ---------------------------------------------------------------------------


async def test_options_flow_add_agent_shows_form(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_add_agent(None) shows the agent-type selection form."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_add_agent()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "add_agent"


# ---------------------------------------------------------------------------
# Test 11 — Options flow: selecting AGENT_TYPE_OLLAMA routes to configure_ollama
# ---------------------------------------------------------------------------


async def test_options_flow_add_agent_routes_to_ollama(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Selecting AGENT_TYPE_OLLAMA from add_agent routes to the configure_ollama form."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_add_agent({CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA})
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_ollama"


# ---------------------------------------------------------------------------
# Test 12 — Options flow: selecting AGENT_TYPE_EXISTING routes to configure_existing
# ---------------------------------------------------------------------------


async def test_options_flow_add_agent_routes_to_existing(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Selecting AGENT_TYPE_EXISTING routes to configure_existing form."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_add_agent({CONF_AGENT_TYPE: AGENT_TYPE_EXISTING})
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_existing"


# ---------------------------------------------------------------------------
# Test 13 — Options flow: selecting AGENT_TYPE_LOCAL_HA routes to configure_local
# ---------------------------------------------------------------------------


async def test_options_flow_add_agent_routes_to_local(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Selecting AGENT_TYPE_LOCAL_HA routes to configure_local form."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_add_agent({CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA})
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_local"


# ---------------------------------------------------------------------------
# Test 14 — configure_ollama: validation error re-shows form
# ---------------------------------------------------------------------------


async def test_configure_ollama_validation_error(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """When Ollama validation fails the form is re-shown with errors."""
    handler = _make_handler(mock_config_entry, hass)
    with patch.object(
        handler,
        "_validate_ollama_connection",
        new_callable=AsyncMock,
        return_value="cannot_connect",
    ):
        result = await handler.async_step_configure_ollama(
            {
                CONF_AGENT_NAME: "My Ollama",
                CONF_PRIORITY: 10,
                CONF_OLLAMA_URL: DEFAULT_OLLAMA_URL,
                CONF_OLLAMA_MODEL: "llama3:8b",
            }
        )
    assert result["type"] == FlowResultType.FORM
    assert result.get("errors", {}).get("base") == "cannot_connect"


# ---------------------------------------------------------------------------
# Test 15 — configure_ollama: success appends new agent to entry.data
# ---------------------------------------------------------------------------


async def test_configure_ollama_success(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Successful Ollama configuration appends the new agent."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)
    with (
        patch.object(
            handler, "_validate_ollama_connection", new_callable=AsyncMock, return_value=None
        ),
        patch.object(
            handler,
            "async_create_entry",
            return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
        ),
    ):
        await handler.async_step_configure_ollama(
            {
                CONF_AGENT_NAME: "My Ollama",
                CONF_PRIORITY: 10,
                CONF_OLLAMA_URL: DEFAULT_OLLAMA_URL,
                CONF_OLLAMA_MODEL: "llama3:8b",
            }
        )

    agents = mock_config_entry.data.get(CONF_AGENTS, [])
    assert len(agents) == 1
    assert agents[0][CONF_AGENT_TYPE] == AGENT_TYPE_OLLAMA
    assert agents[0][CONF_AGENT_NAME] == "My Ollama"


# ---------------------------------------------------------------------------
# Test 16 — configure_existing: success appends new agent
# ---------------------------------------------------------------------------


async def test_configure_existing_success(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Successful existing-agent configuration appends the new agent."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)
    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_configure_existing(
            {
                CONF_AGENT_NAME: "ChatGPT",
                CONF_PRIORITY: 20,
                CONF_ENTITY_ID: "conversation.openai",
                CONF_TIMEOUT: DEFAULT_TIMEOUT,
            }
        )

    agents = mock_config_entry.data.get(CONF_AGENTS, [])
    assert len(agents) == 1
    assert agents[0][CONF_AGENT_TYPE] == AGENT_TYPE_EXISTING


# ---------------------------------------------------------------------------
# Test 17 — configure_local: success appends new agent with fixed entity_id
# ---------------------------------------------------------------------------


async def test_configure_local_success(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Successful local-agent configuration sets the fixed entity_id."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)
    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_configure_local(
            {
                CONF_AGENT_NAME: "HA Local",
                CONF_PRIORITY: 50,
                CONF_TIMEOUT: DEFAULT_TIMEOUT,
            }
        )

    agents = mock_config_entry.data.get(CONF_AGENTS, [])
    assert len(agents) == 1
    assert agents[0][CONF_AGENT_TYPE] == AGENT_TYPE_LOCAL_HA
    assert agents[0][CONF_ENTITY_ID] == "conversation.home_assistant"


# ---------------------------------------------------------------------------
# Test 18 — manage_agents: no agents shows empty-schema form
# ---------------------------------------------------------------------------


async def test_manage_agents_no_agents(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """With no agents configured, manage_agents shows a form without a data_schema."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_manage_agents()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "manage_agents"
    assert result.get("data_schema") is None


# ---------------------------------------------------------------------------
# Test 19 — manage_agents: agents present shows dropdown
# ---------------------------------------------------------------------------


async def test_manage_agents_with_agents_shows_dropdown(hass: HomeAssistant) -> None:
    """With agents present, manage_agents shows a dropdown selector form."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    result = await handler.async_step_manage_agents()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "manage_agents"
    assert result["data_schema"] is not None


# ---------------------------------------------------------------------------
# Test 20 — manage_agents: submitting selection routes to action step
# ---------------------------------------------------------------------------


async def test_manage_agents_selection_routes_to_action(hass: HomeAssistant) -> None:
    """Selecting an agent stores its id and forwards to manage_agent_action."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)

    with patch.object(
        handler, "async_step_manage_agent_action", new_callable=AsyncMock
    ) as mock_action:
        mock_action.return_value = {"type": FlowResultType.FORM, "step_id": "manage_agent_action"}
        await handler.async_step_manage_agents({"agent_id": agent["id"]})

    assert handler._agent_data["_selected_agent_id"] == agent["id"]
    mock_action.assert_called_once()


# ---------------------------------------------------------------------------
# Test 21 — manage_agent_action: form shown for enabled agent
# ---------------------------------------------------------------------------


async def test_manage_agent_action_shows_form_enabled(hass: HomeAssistant) -> None:
    """Form for an enabled agent includes edit, disable, and delete options."""
    agent = _ollama_agent(**{CONF_AGENT_ENABLED: True})
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = agent["id"]

    result = await handler.async_step_manage_agent_action()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "manage_agent_action"


# ---------------------------------------------------------------------------
# Test 22 — manage_agent_action: form shown for disabled agent
# ---------------------------------------------------------------------------


async def test_manage_agent_action_shows_form_disabled(hass: HomeAssistant) -> None:
    """Form for a disabled agent shows the 'Enable' toggle variant."""
    agent = _ollama_agent(**{CONF_AGENT_ENABLED: False})
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = agent["id"]

    result = await handler.async_step_manage_agent_action()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "manage_agent_action"


# ---------------------------------------------------------------------------
# Test 23 — manage_agent_action: form shown when agent not found (fallback labels)
# ---------------------------------------------------------------------------


async def test_manage_agent_action_form_missing_agent(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """When agent_id matches nothing, form shows with 'Unknown' placeholder."""
    handler = _make_handler(mock_config_entry, hass)
    handler._agent_data["_selected_agent_id"] = "nonexistent"

    result = await handler.async_step_manage_agent_action()
    assert result["type"] == FlowResultType.FORM
    assert result["description_placeholders"]["agent_name"] == "Unknown"


# ---------------------------------------------------------------------------
# Test 24 — manage_agent_action: toggle disables enabled agent
# ---------------------------------------------------------------------------


async def test_manage_agent_action_toggle_disables(hass: HomeAssistant) -> None:
    """Toggling an enabled agent sets CONF_AGENT_ENABLED to False."""
    agent = _ollama_agent(**{CONF_AGENT_ENABLED: True})
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = agent["id"]

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_manage_agent_action({"action": "toggle"})

    assert entry.data[CONF_AGENTS][0][CONF_AGENT_ENABLED] is False


# ---------------------------------------------------------------------------
# Test 25 — manage_agent_action: toggle enables disabled agent
# ---------------------------------------------------------------------------


async def test_manage_agent_action_toggle_enables(hass: HomeAssistant) -> None:
    """Toggling a disabled agent sets CONF_AGENT_ENABLED to True."""
    agent = _ollama_agent(**{CONF_AGENT_ENABLED: False})
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = agent["id"]

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_manage_agent_action({"action": "toggle"})

    assert entry.data[CONF_AGENTS][0][CONF_AGENT_ENABLED] is True


# ---------------------------------------------------------------------------
# Test 26 — manage_agent_action: delete removes agent
# ---------------------------------------------------------------------------


async def test_manage_agent_action_delete(hass: HomeAssistant) -> None:
    """Delete action removes the selected agent from entry.data."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = agent["id"]

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_manage_agent_action({"action": "delete"})

    assert entry.data[CONF_AGENTS] == []


# ---------------------------------------------------------------------------
# Test 27 — manage_agent_action: edit routes to _route_to_edit_step
# ---------------------------------------------------------------------------


async def test_manage_agent_action_edit_dispatches_to_route(hass: HomeAssistant) -> None:
    """Edit action calls _route_to_edit_step instead of creating an entry."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = agent["id"]

    with patch.object(handler, "_route_to_edit_step", new_callable=AsyncMock) as mock_route:
        mock_route.return_value = {"type": FlowResultType.FORM, "step_id": "edit_agent_ollama"}
        await handler.async_step_manage_agent_action({"action": "edit"})

    mock_route.assert_called_once()


# ---------------------------------------------------------------------------
# Test 28 — _route_to_edit_step: missing agent returns create_entry
# ---------------------------------------------------------------------------


async def test_route_to_edit_step_missing_agent(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """When selected agent is not found, _route_to_edit_step returns create_entry."""
    handler = _make_handler(mock_config_entry, hass)
    handler._agent_data["_selected_agent_id"] = "ghost-id"

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ) as mock_create:
        await handler._route_to_edit_step()

    mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# Test 29 — _route_to_edit_step: Ollama type routes to edit_agent_ollama
# ---------------------------------------------------------------------------


async def test_route_to_edit_step_ollama(hass: HomeAssistant) -> None:
    """Ollama agent type routes to async_step_edit_agent_ollama."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = agent["id"]

    with patch.object(handler, "async_step_edit_agent_ollama", new_callable=AsyncMock) as mock_edit:
        mock_edit.return_value = {"type": FlowResultType.FORM, "step_id": "edit_agent_ollama"}
        await handler._route_to_edit_step()

    mock_edit.assert_called_once()
    assert handler._agent_data["_editing_agent"]["id"] == agent["id"]


# ---------------------------------------------------------------------------
# Test 30 — _route_to_edit_step: Existing type routes to edit_agent_existing
# ---------------------------------------------------------------------------


async def test_route_to_edit_step_existing(hass: HomeAssistant) -> None:
    """Existing-integration agent type routes to async_step_edit_agent_existing."""
    agent = _existing_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = agent["id"]

    with patch.object(
        handler, "async_step_edit_agent_existing", new_callable=AsyncMock
    ) as mock_edit:
        mock_edit.return_value = {"type": FlowResultType.FORM, "step_id": "edit_agent_existing"}
        await handler._route_to_edit_step()

    mock_edit.assert_called_once()


# ---------------------------------------------------------------------------
# Test 31 — _route_to_edit_step: Local HA type routes to edit_agent_local
# ---------------------------------------------------------------------------


async def test_route_to_edit_step_local(hass: HomeAssistant) -> None:
    """Local HA agent type routes to async_step_edit_agent_local."""
    agent = _local_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = agent["id"]

    with patch.object(handler, "async_step_edit_agent_local", new_callable=AsyncMock) as mock_edit:
        mock_edit.return_value = {"type": FlowResultType.FORM, "step_id": "edit_agent_local"}
        await handler._route_to_edit_step()

    mock_edit.assert_called_once()


# ---------------------------------------------------------------------------
# Test 32 — edit_agent_ollama: shows pre-populated form
# ---------------------------------------------------------------------------


async def test_edit_agent_ollama_shows_form(hass: HomeAssistant) -> None:
    """async_step_edit_agent_ollama without input returns a FORM."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    result = await handler.async_step_edit_agent_ollama()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "edit_agent_ollama"


# ---------------------------------------------------------------------------
# Test 33 — edit_agent_ollama: validation error re-shows form
# ---------------------------------------------------------------------------


async def test_edit_agent_ollama_validation_error(hass: HomeAssistant) -> None:
    """When Ollama validation fails the edit form is re-shown with errors."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    with patch.object(
        handler,
        "_validate_ollama_connection",
        new_callable=AsyncMock,
        return_value="model_not_found",
    ):
        result = await handler.async_step_edit_agent_ollama(
            {
                CONF_AGENT_NAME: "Updated",
                CONF_PRIORITY: 5,
                CONF_OLLAMA_URL: DEFAULT_OLLAMA_URL,
                CONF_OLLAMA_MODEL: "no_such_model",
            }
        )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "edit_agent_ollama"
    assert result.get("errors", {}).get("base") == "model_not_found"


# ---------------------------------------------------------------------------
# Test 34 — edit_agent_ollama: success updates agent in place
# ---------------------------------------------------------------------------


async def test_edit_agent_ollama_success(hass: HomeAssistant) -> None:
    """Successful edit updates the agent without changing id or type."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    with (
        patch.object(
            handler, "_validate_ollama_connection", new_callable=AsyncMock, return_value=None
        ),
        patch.object(
            handler,
            "async_create_entry",
            return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
        ),
    ):
        await handler.async_step_edit_agent_ollama(
            {
                CONF_AGENT_NAME: "Updated Ollama",
                CONF_PRIORITY: 3,
                CONF_OLLAMA_URL: "http://ollama.local:11434",
                CONF_OLLAMA_MODEL: "llama3:8b",
                CONF_TIMEOUT: 60,
                CONF_SYSTEM_PROMPT: "Be brief.",
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
            }
        )

    agents = entry.data[CONF_AGENTS]
    assert len(agents) == 1
    updated = agents[0]
    assert updated["id"] == agent["id"]
    assert updated[CONF_AGENT_TYPE] == AGENT_TYPE_OLLAMA
    assert updated[CONF_AGENT_NAME] == "Updated Ollama"
    assert updated[CONF_PRIORITY] == 3
    assert updated[CONF_SYSTEM_PROMPT] == "Be brief."
    assert updated[CONF_AGENT_CACHE_ENABLED] is False


# ---------------------------------------------------------------------------
# Test 35 — edit_agent_ollama: other agents preserved
# ---------------------------------------------------------------------------


async def test_edit_agent_ollama_other_agents_preserved(hass: HomeAssistant) -> None:
    """Editing agent-1 does not affect agent-2 in entry.data."""
    agent1 = _ollama_agent("agent-1")
    agent2 = _ollama_agent("agent-2", **{CONF_AGENT_NAME: "Second"})
    entry = _entry_with_agents(hass, [agent1, agent2])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent1)

    with (
        patch.object(
            handler, "_validate_ollama_connection", new_callable=AsyncMock, return_value=None
        ),
        patch.object(
            handler,
            "async_create_entry",
            return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
        ),
    ):
        await handler.async_step_edit_agent_ollama(
            {
                CONF_AGENT_NAME: "Renamed First",
                CONF_PRIORITY: 1,
                CONF_OLLAMA_URL: DEFAULT_OLLAMA_URL,
                CONF_OLLAMA_MODEL: "llama3:8b",
            }
        )

    agents = entry.data[CONF_AGENTS]
    assert len(agents) == 2
    names = {a[CONF_AGENT_NAME] for a in agents}
    assert "Renamed First" in names
    assert "Second" in names


# ---------------------------------------------------------------------------
# Test 36 — edit_agent_existing: shows pre-populated form
# ---------------------------------------------------------------------------


async def test_edit_agent_existing_shows_form(hass: HomeAssistant) -> None:
    """async_step_edit_agent_existing without input returns a FORM."""
    agent = _existing_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    result = await handler.async_step_edit_agent_existing()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "edit_agent_existing"


# ---------------------------------------------------------------------------
# Test 37 — edit_agent_existing: success updates agent in place
# ---------------------------------------------------------------------------


async def test_edit_agent_existing_success(hass: HomeAssistant) -> None:
    """Successful edit updates the existing agent without changing id or type."""
    agent = _existing_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_edit_agent_existing(
            {
                CONF_AGENT_NAME: "Updated ChatGPT",
                CONF_PRIORITY: 15,
                CONF_ENTITY_ID: "conversation.openai_gpt4",
                CONF_TIMEOUT: 45,
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: True,
            }
        )

    agents = entry.data[CONF_AGENTS]
    assert len(agents) == 1
    updated = agents[0]
    assert updated["id"] == agent["id"]
    assert updated[CONF_AGENT_TYPE] == AGENT_TYPE_EXISTING
    assert updated[CONF_AGENT_NAME] == "Updated ChatGPT"
    assert updated[CONF_ENTITY_ID] == "conversation.openai_gpt4"


# ---------------------------------------------------------------------------
# Test 38 — edit_agent_local: shows pre-populated form
# ---------------------------------------------------------------------------


async def test_edit_agent_local_shows_form(hass: HomeAssistant) -> None:
    """async_step_edit_agent_local without input returns a FORM."""
    agent = _local_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    result = await handler.async_step_edit_agent_local()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "edit_agent_local"


# ---------------------------------------------------------------------------
# Test 39 — edit_agent_local: success updates agent in place
# ---------------------------------------------------------------------------


async def test_edit_agent_local_success(hass: HomeAssistant) -> None:
    """Successful edit updates the local agent without changing id, type, or entity_id."""
    agent = _local_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_edit_agent_local(
            {
                CONF_AGENT_NAME: "HA Local v2",
                CONF_PRIORITY: 80,
                CONF_TIMEOUT: 20,
                CONF_AGENT_CACHE_ENABLED: True,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
            }
        )

    agents = entry.data[CONF_AGENTS]
    assert len(agents) == 1
    updated = agents[0]
    assert updated["id"] == agent["id"]
    assert updated[CONF_AGENT_NAME] == "HA Local v2"
    assert updated[CONF_PRIORITY] == 80
    # entity_id comes from the original agent dict spread via **agent
    assert updated[CONF_ENTITY_ID] == "conversation.home_assistant"


# ---------------------------------------------------------------------------
# Test 40 — advanced_settings: shows form
# ---------------------------------------------------------------------------


async def test_advanced_settings_shows_form(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_advanced_settings without input shows the settings form."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_advanced_settings()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "advanced_settings"


# ---------------------------------------------------------------------------
# Test 41 — advanced_settings: saves cache config
# ---------------------------------------------------------------------------


async def test_advanced_settings_saves_config(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Submitting advanced settings persists cache config to entry.data."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_advanced_settings(
            {
                CONF_RESPONSE_CACHE_ENABLED: False,
                CONF_RESPONSE_CACHE_TTL: 120,
                "purge_cache_now": False,
            }
        )

    assert mock_config_entry.data[CONF_RESPONSE_CACHE_ENABLED] is False
    assert mock_config_entry.data[CONF_RESPONSE_CACHE_TTL] == 120


# ---------------------------------------------------------------------------
# Test 42 — advanced_settings: purge cache calls invalidate()
# ---------------------------------------------------------------------------


async def test_advanced_settings_purges_cache(hass: HomeAssistant) -> None:
    """Setting purge_cache_now=True calls invalidate() then configure() on the cache."""
    mock_cache = MagicMock()
    mock_cache.invalidate.return_value = 3

    entry = _entry_with_agents(hass, [])
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {DATA_RESPONSE_CACHE: mock_cache}

    handler = _make_handler(entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_advanced_settings(
            {
                CONF_RESPONSE_CACHE_ENABLED: True,
                CONF_RESPONSE_CACHE_TTL: 300,
                "purge_cache_now": True,
            }
        )

    mock_cache.invalidate.assert_called_once()
    mock_cache.configure.assert_called_once_with(enabled=True, ttl_seconds=300)


# ---------------------------------------------------------------------------
# Test 43 — advanced_settings: no purge skips invalidate()
# ---------------------------------------------------------------------------


async def test_advanced_settings_configures_without_purge(hass: HomeAssistant) -> None:
    """When purge_cache_now=False, invalidate() is not called but configure() is."""
    mock_cache = MagicMock()

    entry = _entry_with_agents(hass, [])
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {DATA_RESPONSE_CACHE: mock_cache}

    handler = _make_handler(entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_advanced_settings(
            {
                CONF_RESPONSE_CACHE_ENABLED: True,
                CONF_RESPONSE_CACHE_TTL: 60,
                "purge_cache_now": False,
            }
        )

    mock_cache.invalidate.assert_not_called()
    mock_cache.configure.assert_called_once_with(enabled=True, ttl_seconds=60)


# ---------------------------------------------------------------------------
# Test 44 — advanced_settings: no cache object in hass.data is safe
# ---------------------------------------------------------------------------


async def test_advanced_settings_no_cache_object(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """When no cache object exists, settings are still saved without error."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_advanced_settings(
            {
                CONF_RESPONSE_CACHE_ENABLED: True,
                CONF_RESPONSE_CACHE_TTL: 300,
                "purge_cache_now": True,
            }
        )

    assert mock_config_entry.data[CONF_RESPONSE_CACHE_ENABLED] is True


# ---------------------------------------------------------------------------
# Test 45 — configure_guard_rails: shows form
# ---------------------------------------------------------------------------


async def test_configure_guard_rails_shows_form(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_configure_guard_rails without input shows the form."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_configure_guard_rails()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_guard_rails"


# ---------------------------------------------------------------------------
# Test 46 — configure_guard_rails: disabled saves without router check
# ---------------------------------------------------------------------------


async def test_configure_guard_rails_disabled(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Disabling guard rails saves without needing a router agent."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_configure_guard_rails(
            {
                CONF_GUARD_RAIL_ENABLED: False,
                CONF_GUARD_RAIL_ACTION: DEFAULT_GUARD_RAIL_ACTION,
                CONF_GUARD_RAIL_AI_THRESHOLD: DEFAULT_GUARD_RAIL_AI_THRESHOLD,
            }
        )

    assert mock_config_entry.data[CONF_GUARD_RAIL_ENABLED] is False


# ---------------------------------------------------------------------------
# Test 47 — configure_guard_rails: enabled with router agent saves
# ---------------------------------------------------------------------------


async def test_configure_guard_rails_enabled_with_router(hass: HomeAssistant) -> None:
    """Enabling guard rails with a priority-0 router saves guard_rail_agent_id."""
    router = _ollama_agent("router-1", **{CONF_PRIORITY: 0})
    entry = _entry_with_agents(hass, [router])
    handler = _make_handler(entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_configure_guard_rails(
            {
                CONF_GUARD_RAIL_ENABLED: True,
                CONF_GUARD_RAIL_ACTION: GUARD_RAIL_ACTION_BLOCK,
                CONF_GUARD_RAIL_AI_THRESHOLD: 0.8,
            }
        )

    assert entry.data[CONF_GUARD_RAIL_ENABLED] is True
    assert entry.data.get("guard_rail_agent_id") == router["id"]


# ---------------------------------------------------------------------------
# Test 48 — configure_guard_rails: enabled without router shows error
# ---------------------------------------------------------------------------


async def test_configure_guard_rails_enabled_no_router(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Enabling guard rails with no priority-0 agent re-shows the form with an error."""
    handler = _make_handler(mock_config_entry, hass)

    result = await handler.async_step_configure_guard_rails(
        {
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_ACTION: DEFAULT_GUARD_RAIL_ACTION,
            CONF_GUARD_RAIL_AI_THRESHOLD: DEFAULT_GUARD_RAIL_AI_THRESHOLD,
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_guard_rails"
    assert result.get("errors", {}).get("base") == "no_router_agent"


# ---------------------------------------------------------------------------
# Test 49 — language_settings: shows form
# ---------------------------------------------------------------------------


async def test_language_settings_shows_form(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_language_settings without input shows the form."""
    handler = _make_handler(mock_config_entry, hass)

    with patch(
        "custom_components.neuralbridge.config_flow.list_available_languages",
        return_value=[{"code": "en_gb", "name": "English (UK)"}],
    ):
        result = await handler.async_step_language_settings()

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "language_settings"


# ---------------------------------------------------------------------------
# Test 50 — language_settings: saves selected language
# ---------------------------------------------------------------------------


async def test_language_settings_saves_language(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Submitting a language saves it to entry.data."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_language_settings({"language": "en_us"})

    assert mock_config_entry.data[CONF_LANGUAGE] == "en_us"


def test_async_get_options_flow_returns_handler(
    mock_config_entry: MockConfigEntry,
) -> None:
    """NeuralBridgeConfigFlow.async_get_options_flow returns an options flow handler."""
    handler = NeuralBridgeConfigFlow.async_get_options_flow(mock_config_entry)
    assert isinstance(handler, NeuralBridgeOptionsFlowHandler)
