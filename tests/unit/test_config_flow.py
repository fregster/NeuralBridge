"""Unit tests for NeuralBridge config flow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
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
    AGENT_TYPE_WEB_SEARCH,
    CONF_AGENT_ASSIST_MODE,
    CONF_AGENT_CACHE_ENABLED,
    CONF_AGENT_ENABLED,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_AGENTS,
    CONF_DEFAULT_PROMPT,
    CONF_ENABLE_HOME_CONTROL,
    CONF_ENTITY_ID,
    CONF_FORCE_RESPONSE_LANGUAGE,
    CONF_GUARD_RAIL_ACTION,
    CONF_GUARD_RAIL_AI_THRESHOLD,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_GUARD_RAIL_RULES,
    CONF_IS_ROUTER,
    CONF_LANGUAGE,
    CONF_MAX_RETRIES,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_PRIORITY,
    CONF_RESPONSE_CACHE_ENABLED,
    CONF_RESPONSE_CACHE_TTL,
    CONF_RETRY_BASE_DELAY,
    CONF_ROUTER_CUSTOM_PROMPT,
    CONF_ROUTER_FALLBACK,
    CONF_ROUTER_LOG_LEVEL,
    CONF_SEARCH_ANSWERS_API_KEY,
    CONF_SEARCH_API_KEY,
    CONF_SEARCH_MAX_SNIPPET_LEN,
    CONF_SEARCH_PROVIDER,
    CONF_SEARCH_RESULT_COUNT,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DATA_RESPONSE_CACHE,
    DEFAULT_AGENT_CACHE_ENABLED,
    DEFAULT_DEFAULT_PROMPT,
    DEFAULT_ENABLE_HOME_CONTROL,
    DEFAULT_FORCE_RESPONSE_LANGUAGE,
    DEFAULT_GUARD_RAIL_ACTION,
    DEFAULT_GUARD_RAIL_AI_THRESHOLD,
    DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    DEFAULT_LANGUAGE,
    DEFAULT_OLLAMA_URL,
    DEFAULT_RESPONSE_CACHE_ENABLED,
    DEFAULT_RESPONSE_CACHE_TTL,
    DEFAULT_ROUTER_CUSTOM_PROMPT,
    DEFAULT_ROUTER_FALLBACK,
    DEFAULT_ROUTER_LOG_LEVEL,
    DEFAULT_ROUTER_TIMEOUT,
    DEFAULT_SEARCH_MAX_SNIPPET_LEN,
    DEFAULT_SEARCH_RESULT_COUNT,
    DEFAULT_SEARCH_TIMEOUT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    DOMAIN,
    GUARD_RAIL_ACTION_BLOCK,
    GUARD_RAIL_CATEGORY_HARMFUL,
    GUARD_RAIL_CATEGORY_PRIVACY,
    PRIORITY_ROUTER,
    ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
    ROUTER_FALLBACK_SKIP_ROUTING,
    ROUTER_LOG_LEVEL_NONE,
    SEARCH_PROVIDER_BRAVE,
    SEARCH_PROVIDER_BRAVE_ANSWERS,
    SEARCH_PROVIDER_BRAVE_COMBINED,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

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
    # Ensure the entry is registered with hass — required because OptionsFlow.config_entry
    # is a property that calls hass.config_entries.async_get_known_entry(_config_entry_id),
    # where _config_entry_id is itself a property returning self.handler.
    if not hass.config_entries.async_get_entry(mock_config_entry.entry_id):
        mock_config_entry.add_to_hass(hass)
    handler = NeuralBridgeOptionsFlowHandler()
    handler.hass = hass
    # flow_id is normally set by the HA flow manager.
    handler.flow_id = "test-flow-id"  # type: ignore[attr-defined]
    # handler must equal the config entry's entry_id — that is what the
    # OptionsFlow._config_entry_id property returns (return self.handler).
    handler.handler = mock_config_entry.entry_id  # type: ignore[attr-defined]
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
    """async_step_init returns a MENU result with all expected options."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_init()
    assert result["type"] == FlowResultType.MENU
    assert set(result["menu_options"]) == {
        "default_prompt",
        "add_agent",
        "configure_routing_agent",
        "manage_agents",
        "configure_guard_rails",
        "advanced_settings",
        "language_settings",
        "done",
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
# Test 20 — manage_agents: submitting selection routes directly to edit step
# ---------------------------------------------------------------------------


async def test_manage_agents_selection_routes_to_edit(hass: HomeAssistant) -> None:
    """Selecting an agent stores its id and routes directly to the edit step."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)

    with patch.object(handler, "_route_to_edit_step", new_callable=AsyncMock) as mock_route:
        mock_route.return_value = {"type": FlowResultType.FORM, "step_id": "edit_agent_ollama"}
        await handler.async_step_manage_agents({"agent_id": agent["id"]})

    assert handler._agent_data["_selected_agent_id"] == agent["id"]
    mock_route.assert_called_once()


# ---------------------------------------------------------------------------
# Test 21 — _route_to_edit_step: missing agent returns to main menu
# ---------------------------------------------------------------------------


async def test_route_to_edit_step_missing_agent(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """When selected agent is not found, _route_to_edit_step returns to main menu."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)
    handler._agent_data["_selected_agent_id"] = "ghost-id"

    result = await handler._route_to_edit_step()

    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "init"


# ---------------------------------------------------------------------------
# Test 22 — _route_to_edit_step: Ollama type routes to edit_agent_ollama
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
# Test 23 — _route_to_edit_step: Existing type routes to edit_agent_existing
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
# Test 24 — _route_to_edit_step: Local HA type routes to edit_agent_local
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
# Test 25 — edit_agent_ollama: shows pre-populated form
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
# Test 26 — edit_agent_ollama: validation error re-shows form
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
# Test 27 — edit_agent_ollama: success updates agent in place
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
# Test 28 — edit_agent_ollama: other agents preserved
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
# Test 29 — edit_agent_ollama: enabled field saved when explicitly set
# ---------------------------------------------------------------------------


async def test_edit_agent_ollama_saves_enabled_false(hass: HomeAssistant) -> None:
    """Editing Ollama agent with CONF_AGENT_ENABLED=False disables the agent."""
    agent = _ollama_agent(**{CONF_AGENT_ENABLED: True})
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
                CONF_AGENT_NAME: agent[CONF_AGENT_NAME],
                CONF_PRIORITY: agent[CONF_PRIORITY],
                CONF_OLLAMA_URL: DEFAULT_OLLAMA_URL,
                CONF_OLLAMA_MODEL: "llama3:8b",
                CONF_AGENT_ENABLED: False,
            }
        )

    assert entry.data[CONF_AGENTS][0][CONF_AGENT_ENABLED] is False


# ---------------------------------------------------------------------------
# Test 30 — edit_agent_ollama: delete_agent=True routes to confirmation step
# ---------------------------------------------------------------------------


async def test_edit_agent_ollama_delete_routes_to_confirm(hass: HomeAssistant) -> None:
    """Submitting with delete_agent=True routes to the confirm_delete_agent step."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    with patch.object(
        handler, "async_step_confirm_delete_agent", new_callable=AsyncMock
    ) as mock_confirm:
        mock_confirm.return_value = {"type": FlowResultType.FORM, "step_id": "confirm_delete_agent"}
        await handler.async_step_edit_agent_ollama(
            {
                CONF_AGENT_NAME: agent[CONF_AGENT_NAME],
                CONF_PRIORITY: agent[CONF_PRIORITY],
                CONF_OLLAMA_URL: DEFAULT_OLLAMA_URL,
                CONF_OLLAMA_MODEL: "llama3:8b",
                "delete_agent": True,
            }
        )

    mock_confirm.assert_called_once()


# ---------------------------------------------------------------------------
# Test 31 — edit_agent_existing: shows pre-populated form
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
# Test 32 — edit_agent_existing: success updates agent in place
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
# Test 33 — edit_agent_existing: enabled field saved when explicitly set
# ---------------------------------------------------------------------------


async def test_edit_agent_existing_saves_enabled_false(hass: HomeAssistant) -> None:
    """Editing existing agent with CONF_AGENT_ENABLED=False disables the agent."""
    agent = _existing_agent(**{CONF_AGENT_ENABLED: True})
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
                CONF_AGENT_NAME: agent[CONF_AGENT_NAME],
                CONF_PRIORITY: agent[CONF_PRIORITY],
                CONF_ENTITY_ID: agent[CONF_ENTITY_ID],
                CONF_AGENT_ENABLED: False,
            }
        )

    assert entry.data[CONF_AGENTS][0][CONF_AGENT_ENABLED] is False


# ---------------------------------------------------------------------------
# Test 34 — edit_agent_existing: delete_agent=True routes to confirmation step
# ---------------------------------------------------------------------------


async def test_edit_agent_existing_delete_routes_to_confirm(hass: HomeAssistant) -> None:
    """Submitting with delete_agent=True routes to the confirm_delete_agent step."""
    agent = _existing_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    with patch.object(
        handler, "async_step_confirm_delete_agent", new_callable=AsyncMock
    ) as mock_confirm:
        mock_confirm.return_value = {"type": FlowResultType.FORM, "step_id": "confirm_delete_agent"}
        await handler.async_step_edit_agent_existing(
            {
                CONF_AGENT_NAME: agent[CONF_AGENT_NAME],
                CONF_PRIORITY: agent[CONF_PRIORITY],
                CONF_ENTITY_ID: agent[CONF_ENTITY_ID],
                "delete_agent": True,
            }
        )

    mock_confirm.assert_called_once()


# ---------------------------------------------------------------------------
# Test 35 — edit_agent_local: shows pre-populated form
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
# Test 36 — edit_agent_local: success updates agent in place
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
# Test 37 — edit_agent_local: enabled field saved when explicitly set
# ---------------------------------------------------------------------------


async def test_edit_agent_local_saves_enabled_false(hass: HomeAssistant) -> None:
    """Editing local agent with CONF_AGENT_ENABLED=False disables the agent."""
    agent = _local_agent(**{CONF_AGENT_ENABLED: True})
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
                CONF_AGENT_NAME: agent[CONF_AGENT_NAME],
                CONF_PRIORITY: agent[CONF_PRIORITY],
                CONF_AGENT_ENABLED: False,
            }
        )

    assert entry.data[CONF_AGENTS][0][CONF_AGENT_ENABLED] is False


# ---------------------------------------------------------------------------
# Test 38 — edit_agent_local: delete_agent=True routes to confirmation step
# ---------------------------------------------------------------------------


async def test_edit_agent_local_delete_routes_to_confirm(hass: HomeAssistant) -> None:
    """Submitting with delete_agent=True routes to the confirm_delete_agent step."""
    agent = _local_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    with patch.object(
        handler, "async_step_confirm_delete_agent", new_callable=AsyncMock
    ) as mock_confirm:
        mock_confirm.return_value = {"type": FlowResultType.FORM, "step_id": "confirm_delete_agent"}
        await handler.async_step_edit_agent_local(
            {
                CONF_AGENT_NAME: agent[CONF_AGENT_NAME],
                CONF_PRIORITY: agent[CONF_PRIORITY],
                "delete_agent": True,
            }
        )

    mock_confirm.assert_called_once()


# ---------------------------------------------------------------------------
# Test 39 — confirm_delete_agent: shows form with agent name placeholder
# ---------------------------------------------------------------------------


async def test_confirm_delete_agent_shows_form(hass: HomeAssistant) -> None:
    """async_step_confirm_delete_agent without input shows the confirmation form."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    result = await handler.async_step_confirm_delete_agent()

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "confirm_delete_agent"
    assert result["description_placeholders"]["agent_name"] == agent[CONF_AGENT_NAME]


# ---------------------------------------------------------------------------
# Test 40 — confirm_delete_agent: confirm=True deletes the agent
# ---------------------------------------------------------------------------


async def test_confirm_delete_agent_confirmed_deletes(hass: HomeAssistant) -> None:
    """When confirm=True the agent is removed from entry.data."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_confirm_delete_agent({"confirm": True})

    assert entry.data[CONF_AGENTS] == []


# ---------------------------------------------------------------------------
# Test 41 — confirm_delete_agent: confirm=False cancels deletion
# ---------------------------------------------------------------------------


async def test_confirm_delete_agent_not_confirmed_keeps_agent(hass: HomeAssistant) -> None:
    """When confirm=False the agent is preserved and flow returns to main menu."""
    agent = _ollama_agent()
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(agent)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        result = await handler.async_step_confirm_delete_agent({"confirm": False})

    assert len(entry.data[CONF_AGENTS]) == 1
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "init"


# ---------------------------------------------------------------------------
# Test 42 — advanced_settings: shows form
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
# Test 43 — advanced_settings: saves cache config
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
# Test 44 — advanced_settings: purge cache calls invalidate()
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
# Test 45 — advanced_settings: no purge skips invalidate()
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
# Test 45b — advanced_settings: TTL=0 is accepted (effectively disables caching)
# ---------------------------------------------------------------------------


async def test_advanced_settings_ttl_zero_accepted(hass: HomeAssistant) -> None:
    """TTL=0 is a valid value that causes cached entries to expire immediately."""
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    mock_cache = MagicMock()
    mock_cache.invalidate.return_value = 0
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
                CONF_RESPONSE_CACHE_TTL: 0,
                "purge_cache_now": False,
            }
        )

    mock_cache.configure.assert_called_once_with(enabled=True, ttl_seconds=0)


# ---------------------------------------------------------------------------
# Test 46 — advanced_settings: no cache object in hass.data is safe
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
# Test 47 — configure_guard_rails: shows sub-menu
# ---------------------------------------------------------------------------


async def test_configure_guard_rails_shows_menu(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_configure_guard_rails shows the guard rails sub-menu."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_configure_guard_rails()
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "configure_guard_rails"
    assert set(result["menu_options"]) == {
        "guard_rails_settings",
        "configure_guard_rail_rules",
        "back_to_main",
    }


# ---------------------------------------------------------------------------
# Test 48 — guard_rails_settings: shows form
# ---------------------------------------------------------------------------


async def test_guard_rails_settings_shows_form(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_guard_rails_settings without input shows the form."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_guard_rails_settings()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "guard_rails_settings"


# ---------------------------------------------------------------------------
# Test 49 — guard_rails_settings: disabled saves without router check
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
        await handler.async_step_guard_rails_settings(
            {
                CONF_GUARD_RAIL_ENABLED: False,
                CONF_GUARD_RAIL_ACTION: DEFAULT_GUARD_RAIL_ACTION,
                CONF_GUARD_RAIL_AI_THRESHOLD: DEFAULT_GUARD_RAIL_AI_THRESHOLD,
            }
        )

    assert mock_config_entry.data[CONF_GUARD_RAIL_ENABLED] is False


# ---------------------------------------------------------------------------
# Test 50 — guard_rails_settings: enabled with router agent saves
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
        await handler.async_step_guard_rails_settings(
            {
                CONF_GUARD_RAIL_ENABLED: True,
                CONF_GUARD_RAIL_ACTION: GUARD_RAIL_ACTION_BLOCK,
                CONF_GUARD_RAIL_AI_THRESHOLD: 0.8,
            }
        )

    assert entry.data[CONF_GUARD_RAIL_ENABLED] is True
    assert entry.data.get("guard_rail_agent_id") == router["id"]


# ---------------------------------------------------------------------------
# Test 51 — guard_rails_settings: enabled without router shows error
# ---------------------------------------------------------------------------


async def test_configure_guard_rails_enabled_no_router(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Enabling guard rails with no priority-0 agent re-shows the form with an error."""
    handler = _make_handler(mock_config_entry, hass)

    result = await handler.async_step_guard_rails_settings(
        {
            CONF_GUARD_RAIL_ENABLED: True,
            CONF_GUARD_RAIL_ACTION: DEFAULT_GUARD_RAIL_ACTION,
            CONF_GUARD_RAIL_AI_THRESHOLD: DEFAULT_GUARD_RAIL_AI_THRESHOLD,
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "guard_rails_settings"
    assert result.get("errors", {}).get("base") == "no_router_agent"


# ---------------------------------------------------------------------------
# Test 52 — back_to_main: returns to init menu
# ---------------------------------------------------------------------------


async def test_back_to_main_returns_to_init(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_back_to_main redirects to the main options menu."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_back_to_main()
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "init"


# ---------------------------------------------------------------------------
# Test 53 — language_settings: shows form
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
# Test 54 — language_settings: saves selected language
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


# ---------------------------------------------------------------------------
# Test 56 — configure_guard_rail_rules: shows form
# ---------------------------------------------------------------------------


async def test_configure_guard_rail_rules_show_form(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_configure_guard_rail_rules without input shows the form."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_configure_guard_rail_rules()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_guard_rail_rules"


# ---------------------------------------------------------------------------
# Test 57 — configure_guard_rail_rules: saves valid patterns
# ---------------------------------------------------------------------------


async def test_configure_guard_rail_rules_save_valid_patterns(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Submitting valid regex patterns saves them under CONF_GUARD_RAIL_RULES."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_configure_guard_rail_rules(
            {
                GUARD_RAIL_CATEGORY_HARMFUL: "\\bharm\\b\nbomb",
                GUARD_RAIL_CATEGORY_PRIVACY: "ssn|social.security",
                "security": "",
                "inappropriate": "",
            }
        )

    saved = mock_config_entry.data[CONF_GUARD_RAIL_RULES]
    assert saved[GUARD_RAIL_CATEGORY_HARMFUL] == ["\\bharm\\b", "bomb"]
    assert saved[GUARD_RAIL_CATEGORY_PRIVACY] == ["ssn|social.security"]
    assert saved["security"] == []
    assert saved["inappropriate"] == []


# ---------------------------------------------------------------------------
# Test 58 — configure_guard_rail_rules: invalid regex returns error
# ---------------------------------------------------------------------------


async def test_configure_guard_rail_rules_invalid_regex(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """An invalid regex pattern returns a form with errors."""
    handler = _make_handler(mock_config_entry, hass)

    result = await handler.async_step_configure_guard_rail_rules(
        {
            GUARD_RAIL_CATEGORY_HARMFUL: "[invalid(regex",
            GUARD_RAIL_CATEGORY_PRIVACY: "",
            "security": "",
            "inappropriate": "",
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_regex"}


# ---------------------------------------------------------------------------
# Test 59 — configure_guard_rail_rules: empty fields save empty lists
# ---------------------------------------------------------------------------


async def test_configure_guard_rail_rules_save_empty(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Submitting empty fields saves empty lists for all categories."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_configure_guard_rail_rules(
            {
                GUARD_RAIL_CATEGORY_HARMFUL: "",
                GUARD_RAIL_CATEGORY_PRIVACY: "",
                "security": "",
                "inappropriate": "",
            }
        )

    saved = mock_config_entry.data[CONF_GUARD_RAIL_RULES]
    assert all(v == [] for v in saved.values())


# ---------------------------------------------------------------------------
# Test 60 — advanced_settings: form includes retry fields
# ---------------------------------------------------------------------------


async def test_advanced_settings_includes_retry_fields(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """advanced_settings form schema includes max_retries and retry_base_delay fields."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_advanced_settings()

    assert result["type"] == FlowResultType.FORM
    schema_keys = {str(k) for k in result["data_schema"].schema}
    assert CONF_MAX_RETRIES in schema_keys
    assert CONF_RETRY_BASE_DELAY in schema_keys


# ---------------------------------------------------------------------------
# Test 61 — advanced_settings: saves retry config
# ---------------------------------------------------------------------------


async def test_advanced_settings_saves_retry_config(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Submitting advanced settings persists max_retries and retry_base_delay."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_advanced_settings(
            {
                CONF_RESPONSE_CACHE_ENABLED: DEFAULT_RESPONSE_CACHE_ENABLED,
                CONF_RESPONSE_CACHE_TTL: DEFAULT_RESPONSE_CACHE_TTL,
                "purge_cache_now": False,
                CONF_MAX_RETRIES: 3,
                CONF_RETRY_BASE_DELAY: 2.0,
            }
        )

    assert mock_config_entry.data[CONF_MAX_RETRIES] == 3
    assert abs(mock_config_entry.data[CONF_RETRY_BASE_DELAY] - 2.0) < 0.001


# ---------------------------------------------------------------------------
# Test 62 — default_prompt: shows form pre-populated with current value
# ---------------------------------------------------------------------------


async def test_default_prompt_shows_form(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_default_prompt without input shows a multiline text form."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    result = await handler.async_step_default_prompt()

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "default_prompt"


# ---------------------------------------------------------------------------
# Test 63 — default_prompt: saves prompt and returns to main menu
# ---------------------------------------------------------------------------


async def test_default_prompt_saves_and_returns_to_menu(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Submitting a new default prompt saves it to entry.data and returns the menu."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)
    custom_prompt = "You are a terse robot."

    result = await handler.async_step_default_prompt({CONF_DEFAULT_PROMPT: custom_prompt})

    assert mock_config_entry.data[CONF_DEFAULT_PROMPT] == custom_prompt
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "init"


# ---------------------------------------------------------------------------
# Test 64 — default_prompt: no stored value shows built-in default text
# ---------------------------------------------------------------------------


async def test_default_prompt_uses_builtin_default(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Form pre-populates with DEFAULT_DEFAULT_PROMPT when none is stored."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    result = await handler.async_step_default_prompt()

    schema = result["data_schema"].schema
    field = next(k for k in schema if str(k) == CONF_DEFAULT_PROMPT)
    assert field.default() == DEFAULT_DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# Test 65 — async_step_done: closes the options flow
# ---------------------------------------------------------------------------


async def test_options_flow_done_closes_flow(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_done returns a CREATE_ENTRY result to close the options flow."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    result = await handler.async_step_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY


# ---------------------------------------------------------------------------
# Test 66 — advanced_settings: saves enable_home_control toggle
# ---------------------------------------------------------------------------


async def test_advanced_settings_saves_enable_home_control(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Submitting advanced_settings persists CONF_ENABLE_HOME_CONTROL to entry.data."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_advanced_settings(
            {
                CONF_ENABLE_HOME_CONTROL: False,
                CONF_RESPONSE_CACHE_ENABLED: True,
                CONF_RESPONSE_CACHE_TTL: 300,
                "purge_cache_now": False,
            }
        )

    assert mock_config_entry.data[CONF_ENABLE_HOME_CONTROL] is False


# ---------------------------------------------------------------------------
# Test 67 — advanced_settings: enable_home_control defaults to True when absent
# ---------------------------------------------------------------------------


async def test_advanced_settings_home_control_defaults_true(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """CONF_ENABLE_HOME_CONTROL defaults to DEFAULT_ENABLE_HOME_CONTROL when not submitted."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        # Submit without CONF_ENABLE_HOME_CONTROL key
        await handler.async_step_advanced_settings(
            {
                CONF_RESPONSE_CACHE_ENABLED: True,
                CONF_RESPONSE_CACHE_TTL: 300,
                "purge_cache_now": False,
            }
        )

    assert mock_config_entry.data[CONF_ENABLE_HOME_CONTROL] is DEFAULT_ENABLE_HOME_CONTROL


# ---------------------------------------------------------------------------
# Test 68 — configure_local: assist_mode saved in agent config
# ---------------------------------------------------------------------------


async def test_configure_local_saves_assist_mode(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_configure_local persists CONF_AGENT_ASSIST_MODE in the agent config."""
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
                CONF_PRIORITY: 10,
                CONF_TIMEOUT: DEFAULT_TIMEOUT,
                CONF_AGENT_ASSIST_MODE: True,
            }
        )

    agents = mock_config_entry.data.get(CONF_AGENTS, [])
    assert len(agents) == 1
    assert agents[0][CONF_AGENT_ASSIST_MODE] is True


# ---------------------------------------------------------------------------
# Test 69 — configure_local: assist_mode=False saved correctly
# ---------------------------------------------------------------------------


async def test_configure_local_saves_assist_mode_false(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """async_step_configure_local saves CONF_AGENT_ASSIST_MODE=False when submitted."""
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
                CONF_PRIORITY: 10,
                CONF_TIMEOUT: DEFAULT_TIMEOUT,
                CONF_AGENT_ASSIST_MODE: False,
            }
        )

    agents = mock_config_entry.data.get(CONF_AGENTS, [])
    assert len(agents) == 1
    assert agents[0][CONF_AGENT_ASSIST_MODE] is False


# ---------------------------------------------------------------------------
# Test 70 — edit_agent_local: assist_mode updated on save
# ---------------------------------------------------------------------------


async def test_edit_agent_local_saves_assist_mode(hass: HomeAssistant) -> None:
    """async_step_edit_agent_local persists updated CONF_AGENT_ASSIST_MODE."""
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
                CONF_AGENT_NAME: "HA Local",
                CONF_PRIORITY: 10,
                CONF_TIMEOUT: DEFAULT_TIMEOUT,
                CONF_AGENT_CACHE_ENABLED: True,
                CONF_AGENT_ASSIST_MODE: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
            }
        )

    updated = entry.data[CONF_AGENTS][0]
    assert updated[CONF_AGENT_ASSIST_MODE] is False


# ===========================================================================
# Tests 71-78: configure_routing_agent step
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 71 — configure_routing_agent: no existing router shows add form
# ---------------------------------------------------------------------------


async def test_configure_routing_agent_shows_add_form(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """configure_routing_agent shows the form when no routing agent exists."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    result = await handler.async_step_configure_routing_agent(None)

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_routing_agent"
    # In add mode the schema should NOT contain the delete_agent or enabled fields
    schema_keys = {k.schema if hasattr(k, "schema") else k for k in result["data_schema"].schema}
    assert "delete_agent" not in schema_keys
    assert CONF_AGENT_ENABLED not in schema_keys


# ---------------------------------------------------------------------------
# Test 72 — configure_routing_agent: existing router shows edit form with extra fields
# ---------------------------------------------------------------------------


async def test_configure_routing_agent_shows_edit_form_with_extra_fields(
    hass: HomeAssistant,
) -> None:
    """configure_routing_agent shows enabled and delete_agent fields when editing."""
    router = {
        "id": "router-1",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_IS_ROUTER: True,
        CONF_AGENT_NAME: "Cloud Router",
        CONF_PRIORITY: PRIORITY_ROUTER,
        CONF_AGENT_ENABLED: True,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
        CONF_ROUTER_LOG_LEVEL: DEFAULT_ROUTER_LOG_LEVEL,
        CONF_ROUTER_CUSTOM_PROMPT: DEFAULT_ROUTER_CUSTOM_PROMPT,
        CONF_ROUTER_FALLBACK: DEFAULT_ROUTER_FALLBACK,
    }
    entry = _entry_with_agents(hass, [router])
    handler = _make_handler(entry, hass)

    result = await handler.async_step_configure_routing_agent(None)

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_routing_agent"
    schema_keys = {k.schema if hasattr(k, "schema") else k for k in result["data_schema"].schema}
    assert "delete_agent" in schema_keys
    assert CONF_AGENT_ENABLED in schema_keys


# ---------------------------------------------------------------------------
# Test 73 — configure_routing_agent: empty entity_id returns entity_not_found
# ---------------------------------------------------------------------------


async def test_configure_routing_agent_empty_entity_id_error(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Submitting with an empty entity_id shows entity_not_found error."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)

    result = await handler.async_step_configure_routing_agent(
        {
            CONF_AGENT_NAME: "My Router",
            CONF_ENTITY_ID: "",
            CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
            CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
            CONF_ROUTER_CUSTOM_PROMPT: "",
            CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result.get("errors", {}).get("base") == "entity_not_found"


# ---------------------------------------------------------------------------
# Test 74 — configure_routing_agent: entity not in hass.states returns error
# ---------------------------------------------------------------------------


async def test_configure_routing_agent_entity_not_in_states_error(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Submitting with an entity_id not in hass.states shows entity_not_found error."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)
    # "conversation.nonexistent" is not registered with hass

    result = await handler.async_step_configure_routing_agent(
        {
            CONF_AGENT_NAME: "My Router",
            CONF_ENTITY_ID: "conversation.nonexistent",
            CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
            CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
            CONF_ROUTER_CUSTOM_PROMPT: "",
            CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result.get("errors", {}).get("base") == "entity_not_found"


# ---------------------------------------------------------------------------
# Test 75 — configure_routing_agent: valid add saves agent with CONF_IS_ROUTER=True
# ---------------------------------------------------------------------------


async def test_configure_routing_agent_valid_add_saves_router(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """A valid submission adds a routing agent with CONF_IS_ROUTER=True."""
    mock_config_entry.add_to_hass(hass)
    handler = _make_handler(mock_config_entry, hass)
    # Register the entity in hass.states so validation passes
    hass.states.async_set("conversation.gemini", "idle")

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_configure_routing_agent(
            {
                CONF_AGENT_NAME: "Cloud Router",
                CONF_ENTITY_ID: "conversation.gemini",
                CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
                CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
                CONF_ROUTER_CUSTOM_PROMPT: "",
                CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
            }
        )

    agents = mock_config_entry.data.get(CONF_AGENTS, [])
    assert len(agents) == 1
    router = agents[0]
    assert router[CONF_IS_ROUTER] is True
    assert router[CONF_AGENT_TYPE] == AGENT_TYPE_EXISTING
    assert router[CONF_AGENT_NAME] == "Cloud Router"
    assert router[CONF_ENTITY_ID] == "conversation.gemini"
    assert router[CONF_AGENT_CACHE_ENABLED] is False
    assert router[CONF_GUARD_RAIL_ENABLED_FOR_AGENT] is False


# ---------------------------------------------------------------------------
# Test 76 — configure_routing_agent: edit mode valid submit updates in place
# ---------------------------------------------------------------------------


async def test_configure_routing_agent_edit_updates_in_place(hass: HomeAssistant) -> None:
    """Editing an existing router updates it in place without appending new entry."""
    router = {
        "id": "router-edit-1",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_IS_ROUTER: True,
        CONF_AGENT_NAME: "Old Name",
        CONF_PRIORITY: PRIORITY_ROUTER,
        CONF_AGENT_ENABLED: True,
        CONF_ENTITY_ID: "conversation.old_gemini",
        CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
        CONF_ROUTER_LOG_LEVEL: DEFAULT_ROUTER_LOG_LEVEL,
        CONF_ROUTER_CUSTOM_PROMPT: DEFAULT_ROUTER_CUSTOM_PROMPT,
        CONF_ROUTER_FALLBACK: DEFAULT_ROUTER_FALLBACK,
        CONF_AGENT_CACHE_ENABLED: False,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
    }
    entry = _entry_with_agents(hass, [router])
    handler = _make_handler(entry, hass)
    # Simulate coming from manage_agents
    handler._agent_data["_editing_agent"] = dict(router)
    hass.states.async_set("conversation.new_gemini", "idle")

    with patch.object(
        handler,
        "async_create_entry",
        return_value={"type": FlowResultType.CREATE_ENTRY, "data": {}},
    ):
        await handler.async_step_configure_routing_agent(
            {
                CONF_AGENT_NAME: "New Name",
                CONF_AGENT_ENABLED: True,
                CONF_ENTITY_ID: "conversation.new_gemini",
                CONF_TIMEOUT: 10,
                CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
                CONF_ROUTER_CUSTOM_PROMPT: "custom",
                CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_SKIP_ROUTING,
                "delete_agent": False,
            }
        )

    agents = entry.data[CONF_AGENTS]
    # Still only one agent — the original was updated, not a new one appended
    assert len(agents) == 1
    updated = agents[0]
    assert updated["id"] == "router-edit-1"
    assert updated[CONF_AGENT_NAME] == "New Name"
    assert updated[CONF_ENTITY_ID] == "conversation.new_gemini"
    assert updated[CONF_ROUTER_FALLBACK] == ROUTER_FALLBACK_SKIP_ROUTING


# ---------------------------------------------------------------------------
# Test 77 — configure_routing_agent: delete_agent=True routes to confirm_delete
# ---------------------------------------------------------------------------


async def test_configure_routing_agent_delete_routes_to_confirm(
    hass: HomeAssistant,
) -> None:
    """Setting delete_agent=True in edit mode triggers the confirm_delete_agent step."""
    router = {
        "id": "router-del-1",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_IS_ROUTER: True,
        CONF_AGENT_NAME: "Doomed Router",
        CONF_PRIORITY: PRIORITY_ROUTER,
        CONF_AGENT_ENABLED: True,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
        CONF_ROUTER_LOG_LEVEL: DEFAULT_ROUTER_LOG_LEVEL,
        CONF_ROUTER_CUSTOM_PROMPT: DEFAULT_ROUTER_CUSTOM_PROMPT,
        CONF_ROUTER_FALLBACK: DEFAULT_ROUTER_FALLBACK,
        CONF_AGENT_CACHE_ENABLED: False,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
    }
    entry = _entry_with_agents(hass, [router])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = dict(router)

    result = await handler.async_step_configure_routing_agent(
        {
            CONF_AGENT_NAME: "Doomed Router",
            CONF_AGENT_ENABLED: True,
            CONF_ENTITY_ID: "conversation.gemini",
            CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
            CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
            CONF_ROUTER_CUSTOM_PROMPT: "",
            CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
            "delete_agent": True,
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "confirm_delete_agent"


# ---------------------------------------------------------------------------
# Test 78 — _route_to_edit_step: is_router agent routes to configure_routing_agent
# ---------------------------------------------------------------------------


async def test_route_to_edit_step_router_calls_configure_routing_agent(
    hass: HomeAssistant,
) -> None:
    """_route_to_edit_step dispatches to configure_routing_agent for a routing agent."""
    router = {
        "id": "route-router-1",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_IS_ROUTER: True,
        CONF_AGENT_NAME: "Route Test Router",
        CONF_PRIORITY: PRIORITY_ROUTER,
        CONF_AGENT_ENABLED: True,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
        CONF_ROUTER_LOG_LEVEL: DEFAULT_ROUTER_LOG_LEVEL,
        CONF_ROUTER_CUSTOM_PROMPT: DEFAULT_ROUTER_CUSTOM_PROMPT,
        CONF_ROUTER_FALLBACK: DEFAULT_ROUTER_FALLBACK,
        CONF_AGENT_CACHE_ENABLED: False,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
    }
    entry = _entry_with_agents(hass, [router])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = "route-router-1"

    result = await handler._route_to_edit_step()

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_routing_agent"


# ---------------------------------------------------------------------------
# Test 79 — configure_routing_agent: delete from main menu (no _editing_agent pre-set)
# ---------------------------------------------------------------------------


async def test_configure_routing_agent_delete_from_menu_sets_editing_agent(
    hass: HomeAssistant,
) -> None:
    """delete_agent=True via main menu (no _editing_agent pre-loaded) still routes
    to confirm_delete_agent and populates _editing_agent automatically."""
    router = {
        "id": "router-menu-del",
        CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
        CONF_IS_ROUTER: True,
        CONF_AGENT_NAME: "Main Menu Router",
        CONF_PRIORITY: PRIORITY_ROUTER,
        CONF_AGENT_ENABLED: True,
        CONF_ENTITY_ID: "conversation.gemini",
        CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
        CONF_ROUTER_LOG_LEVEL: DEFAULT_ROUTER_LOG_LEVEL,
        CONF_ROUTER_CUSTOM_PROMPT: DEFAULT_ROUTER_CUSTOM_PROMPT,
        CONF_ROUTER_FALLBACK: DEFAULT_ROUTER_FALLBACK,
        CONF_AGENT_CACHE_ENABLED: False,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
    }
    entry = _entry_with_agents(hass, [router])
    handler = _make_handler(entry, hass)
    # Intentionally do NOT set _editing_agent — simulates navigation from main menu

    result = await handler.async_step_configure_routing_agent(
        {
            CONF_AGENT_NAME: "Main Menu Router",
            CONF_AGENT_ENABLED: True,
            CONF_ENTITY_ID: "conversation.gemini",
            CONF_TIMEOUT: DEFAULT_ROUTER_TIMEOUT,
            CONF_ROUTER_LOG_LEVEL: ROUTER_LOG_LEVEL_NONE,
            CONF_ROUTER_CUSTOM_PROMPT: "",
            CONF_ROUTER_FALLBACK: ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
            "delete_agent": True,
        }
    )

    # _editing_agent should have been auto-populated
    assert handler._agent_data.get("_editing_agent") is not None
    assert handler._agent_data["_editing_agent"]["id"] == "router-menu-del"
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "confirm_delete_agent"


# ===========================================================================
# Web Search Agent — Config Flow Tests
# ===========================================================================

_PATCH_BRAVE_SESSION = "custom_components.neuralbridge.config_flow.aiohttp.ClientSession"


def _web_search_agent(agent_id: str = "ws-1", **overrides: Any) -> dict[str, Any]:
    """Return a minimal web search agent config dict."""
    base: dict[str, Any] = {
        "id": agent_id,
        CONF_AGENT_TYPE: AGENT_TYPE_WEB_SEARCH,
        CONF_AGENT_ENABLED: True,
        CONF_AGENT_NAME: "Brave Search",
        CONF_PRIORITY: 40,
        CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
        CONF_SEARCH_API_KEY: "existing-brave-key",
        CONF_SEARCH_RESULT_COUNT: DEFAULT_SEARCH_RESULT_COUNT,
        CONF_SEARCH_MAX_SNIPPET_LEN: DEFAULT_SEARCH_MAX_SNIPPET_LEN,
        CONF_TIMEOUT: DEFAULT_SEARCH_TIMEOUT,
        CONF_AGENT_CACHE_ENABLED: DEFAULT_AGENT_CACHE_ENABLED,
        CONF_GUARD_RAIL_ENABLED_FOR_AGENT: DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    }
    base.update(overrides)
    return base


def _make_brave_session_cm(status: int = 200) -> MagicMock:
    """Build nested async-context-manager mocks for _validate_brave_api_key."""
    mock_response = MagicMock()
    mock_response.status = status

    get_cm = MagicMock()
    get_cm.__aenter__ = AsyncMock(return_value=mock_response)
    get_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=get_cm)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    return session_cm


# ---------------------------------------------------------------------------
# add_agent → configure_web_search routing
# ---------------------------------------------------------------------------


async def test_add_agent_routes_to_configure_web_search(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Selecting AGENT_TYPE_WEB_SEARCH from add_agent routes to configure_web_search."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_add_agent({CONF_AGENT_TYPE: AGENT_TYPE_WEB_SEARCH})
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_web_search"


# ---------------------------------------------------------------------------
# configure_web_search — show form
# ---------------------------------------------------------------------------


async def test_configure_web_search_shows_form(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """configure_web_search with no input shows the form."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler.async_step_configure_web_search(None)
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_web_search"


# ---------------------------------------------------------------------------
# configure_web_search — success path
# ---------------------------------------------------------------------------


async def test_configure_web_search_success_saves_agent(
    hass: HomeAssistant,
) -> None:
    """Valid input with a passing key validation adds the agent and returns init."""
    entry = _entry_with_agents(hass, [])
    handler = _make_handler(entry, hass)
    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await handler.async_step_configure_web_search(
            {
                CONF_AGENT_NAME: "Brave Search",
                CONF_PRIORITY: 40,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
                CONF_SEARCH_API_KEY: "test-key",
                CONF_SEARCH_RESULT_COUNT: 5,
                CONF_SEARCH_MAX_SNIPPET_LEN: 200,
                CONF_TIMEOUT: 15,
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
            }
        )

    # Should redirect to init menu after saving
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "init"
    # Agent should be stored
    agents = handler.config_entry.data[CONF_AGENTS]
    web_agents = [a for a in agents if a.get(CONF_AGENT_TYPE) == AGENT_TYPE_WEB_SEARCH]
    assert len(web_agents) == 1
    assert web_agents[0][CONF_AGENT_NAME] == "Brave Search"
    assert web_agents[0][CONF_SEARCH_RESULT_COUNT] == 5


# ---------------------------------------------------------------------------
# configure_web_search — validation errors
# ---------------------------------------------------------------------------


async def test_configure_web_search_empty_api_key_shows_error(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Empty API key returns search_api_key_missing error and re-shows form."""
    handler = _make_handler(mock_config_entry, hass)
    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value="search_api_key_missing",
    ):
        result = await handler.async_step_configure_web_search(
            {
                CONF_AGENT_NAME: "Brave Search",
                CONF_PRIORITY: 40,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
                CONF_SEARCH_API_KEY: "",
            }
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "search_api_key_missing"


async def test_configure_web_search_invalid_api_key_shows_error(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Invalid API key returns invalid_api_key error and re-shows form."""
    handler = _make_handler(mock_config_entry, hass)
    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value="invalid_api_key",
    ):
        result = await handler.async_step_configure_web_search(
            {
                CONF_AGENT_NAME: "Brave Search",
                CONF_PRIORITY: 40,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
                CONF_SEARCH_API_KEY: "bad-key",
            }
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "invalid_api_key"


# ---------------------------------------------------------------------------
# _validate_web_search_connection
# ---------------------------------------------------------------------------


async def test_validate_web_search_connection_empty_key_returns_missing(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Empty API key returns 'search_api_key_missing' without making HTTP calls."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler._validate_web_search_connection(SEARCH_PROVIDER_BRAVE, "")
    assert result == "search_api_key_missing"


async def test_validate_web_search_connection_unknown_provider_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Unknown provider with a key is accepted without live validation."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler._validate_web_search_connection("future_provider", "some-key")
    assert result is None


async def test_validate_web_search_connection_brave_delegates_to_validate_key(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Brave provider delegates to _validate_brave_api_key."""
    handler = _make_handler(mock_config_entry, hass)
    with patch.object(
        handler,
        "_validate_brave_api_key",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_validate:
        result = await handler._validate_web_search_connection(SEARCH_PROVIDER_BRAVE, "my-key")
    mock_validate.assert_called_once_with("my-key")
    assert result is None


# ---------------------------------------------------------------------------
# _validate_brave_api_key — HTTP variants
# ---------------------------------------------------------------------------


async def test_validate_brave_api_key_http_200_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """HTTP 200 from Brave → None (success)."""
    handler = _make_handler(mock_config_entry, hass)
    with patch(_PATCH_BRAVE_SESSION, return_value=_make_brave_session_cm(200)):
        result = await handler._validate_brave_api_key("valid-key")
    assert result is None


async def test_validate_brave_api_key_http_401_returns_invalid(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """HTTP 401 from Brave → 'invalid_api_key'."""
    handler = _make_handler(mock_config_entry, hass)
    with patch(_PATCH_BRAVE_SESSION, return_value=_make_brave_session_cm(401)):
        result = await handler._validate_brave_api_key("bad-key")
    assert result == "invalid_api_key"


async def test_validate_brave_api_key_http_403_returns_invalid(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """HTTP 403 from Brave → 'invalid_api_key'."""
    handler = _make_handler(mock_config_entry, hass)
    with patch(_PATCH_BRAVE_SESSION, return_value=_make_brave_session_cm(403)):
        result = await handler._validate_brave_api_key("forbidden-key")
    assert result == "invalid_api_key"


async def test_validate_brave_api_key_http_500_returns_unreachable(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Non-200/401/403 status → 'search_api_unreachable'."""
    handler = _make_handler(mock_config_entry, hass)
    with patch(_PATCH_BRAVE_SESSION, return_value=_make_brave_session_cm(500)):
        result = await handler._validate_brave_api_key("key")
    assert result == "search_api_unreachable"


async def test_validate_brave_api_key_client_error_returns_unreachable(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """aiohttp.ClientError → 'search_api_unreachable'."""
    handler = _make_handler(mock_config_entry, hass)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("timeout"))
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with patch(_PATCH_BRAVE_SESSION, return_value=session_cm):
        result = await handler._validate_brave_api_key("key")
    assert result == "search_api_unreachable"


async def test_validate_brave_api_key_unexpected_exception_returns_unknown(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """An unexpected exception during validation returns 'unknown'."""
    handler = _make_handler(mock_config_entry, hass)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("unexpected!"))
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with patch(_PATCH_BRAVE_SESSION, return_value=session_cm):
        result = await handler._validate_brave_api_key("key")
    assert result == "unknown"


# ---------------------------------------------------------------------------
# edit_agent_web_search — show form
# ---------------------------------------------------------------------------


async def test_edit_agent_web_search_shows_form_with_existing_values(hass: HomeAssistant) -> None:
    """edit_agent_web_search with no input shows pre-filled form."""
    agent = _web_search_agent(agent_id="ws-edit-1")
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = agent

    result = await handler.async_step_edit_agent_web_search(None)
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "edit_agent_web_search"


async def test_edit_agent_web_search_invalid_new_key_shows_error(hass: HomeAssistant) -> None:
    """Providing an invalid new API key re-shows the form with an error."""
    agent = _web_search_agent(agent_id="ws-edit-4")
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = agent

    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value="invalid_api_key",
    ):
        result = await handler.async_step_edit_agent_web_search(
            {
                CONF_AGENT_NAME: "Brave Search",
                CONF_PRIORITY: 40,
                CONF_AGENT_ENABLED: True,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
                CONF_SEARCH_API_KEY: "bad-new-key",
                CONF_SEARCH_RESULT_COUNT: DEFAULT_SEARCH_RESULT_COUNT,
                CONF_SEARCH_MAX_SNIPPET_LEN: DEFAULT_SEARCH_MAX_SNIPPET_LEN,
                CONF_TIMEOUT: DEFAULT_SEARCH_TIMEOUT,
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
            }
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "invalid_api_key"


# ---------------------------------------------------------------------------
# edit_agent_web_search — save with new key
# ---------------------------------------------------------------------------


async def test_edit_agent_web_search_saves_with_new_api_key(hass: HomeAssistant) -> None:
    """Providing a new API key validates it and saves the updated agent."""
    agent = _web_search_agent(agent_id="ws-edit-2")
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = agent

    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_validate:
        result = await handler.async_step_edit_agent_web_search(
            {
                CONF_AGENT_NAME: "Brave Search Updated",
                CONF_PRIORITY: 50,
                CONF_AGENT_ENABLED: True,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
                CONF_SEARCH_API_KEY: "new-brave-key",
                CONF_SEARCH_RESULT_COUNT: 7,
                CONF_SEARCH_MAX_SNIPPET_LEN: 300,
                CONF_TIMEOUT: 20,
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
            }
        )

    mock_validate.assert_called_once_with(SEARCH_PROVIDER_BRAVE, "new-brave-key", "")
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "init"
    agents = handler.config_entry.data[CONF_AGENTS]
    saved = next(a for a in agents if a.get("id") == "ws-edit-2")
    assert saved[CONF_AGENT_NAME] == "Brave Search Updated"
    assert saved[CONF_SEARCH_API_KEY] == "new-brave-key"
    assert saved[CONF_SEARCH_RESULT_COUNT] == 7


# ---------------------------------------------------------------------------
# edit_agent_web_search — keep existing key when blank
# ---------------------------------------------------------------------------


async def test_edit_agent_web_search_keeps_existing_key_when_blank(hass: HomeAssistant) -> None:
    """Blank API key field retains the original key without calling validation."""
    agent = _web_search_agent(agent_id="ws-edit-3", **{CONF_SEARCH_API_KEY: "keep-this-key"})
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = agent

    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
    ) as mock_validate:
        result = await handler.async_step_edit_agent_web_search(
            {
                CONF_AGENT_NAME: "Brave Search",
                CONF_PRIORITY: 40,
                CONF_AGENT_ENABLED: True,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
                CONF_SEARCH_API_KEY: "",  # blank → keep existing
                CONF_SEARCH_RESULT_COUNT: DEFAULT_SEARCH_RESULT_COUNT,
                CONF_SEARCH_MAX_SNIPPET_LEN: DEFAULT_SEARCH_MAX_SNIPPET_LEN,
                CONF_TIMEOUT: DEFAULT_SEARCH_TIMEOUT,
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
            }
        )

    mock_validate.assert_not_called()
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "init"
    agents = handler.config_entry.data[CONF_AGENTS]
    saved = next(a for a in agents if a.get("id") == "ws-edit-3")
    assert saved[CONF_SEARCH_API_KEY] == "keep-this-key"


# ---------------------------------------------------------------------------
# edit_agent_web_search — delete agent
# ---------------------------------------------------------------------------


async def test_edit_agent_web_search_delete_routes_to_confirm(hass: HomeAssistant) -> None:
    """delete_agent=True in edit form routes to confirm_delete_agent."""
    agent = _web_search_agent(agent_id="ws-del-1")
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = agent

    result = await handler.async_step_edit_agent_web_search({"delete_agent": True})
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "confirm_delete_agent"


# ---------------------------------------------------------------------------
# _route_to_edit_step dispatches to edit_agent_web_search
# ---------------------------------------------------------------------------


async def test_route_to_edit_step_dispatches_to_edit_web_search(hass: HomeAssistant) -> None:
    """_route_to_edit_step with AGENT_TYPE_WEB_SEARCH calls edit_agent_web_search."""
    agent = _web_search_agent(agent_id="ws-route-1")
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_selected_agent_id"] = "ws-route-1"

    with patch.object(
        handler,
        "async_step_edit_agent_web_search",
        new_callable=AsyncMock,
        return_value={"type": FlowResultType.FORM, "step_id": "edit_agent_web_search"},
    ) as mock_edit:
        result = await handler._route_to_edit_step()

    mock_edit.assert_called_once_with()
    assert result["step_id"] == "edit_agent_web_search"


# ---------------------------------------------------------------------------
# _validate_web_search_connection — Brave Answers + Combined paths
# ---------------------------------------------------------------------------


async def test_validate_web_search_connection_brave_answers_empty_answers_key(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """brave_answers with empty answers_api_key returns 'search_answers_api_key_missing'."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler._validate_web_search_connection(SEARCH_PROVIDER_BRAVE_ANSWERS, "", "")
    assert result == "search_answers_api_key_missing"


async def test_validate_web_search_connection_brave_combined_empty_search_key(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """brave_combined with empty search key returns 'search_api_key_missing'."""
    handler = _make_handler(mock_config_entry, hass)
    result = await handler._validate_web_search_connection(
        SEARCH_PROVIDER_BRAVE_COMBINED, "", "answers-key"
    )
    assert result == "search_api_key_missing"


async def test_validate_web_search_connection_brave_combined_empty_answers_key(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """brave_combined with empty answers key returns 'search_answers_api_key_missing'."""
    handler = _make_handler(mock_config_entry, hass)
    with patch.object(
        handler, "_validate_brave_api_key", new_callable=AsyncMock, return_value=None
    ):
        result = await handler._validate_web_search_connection(
            SEARCH_PROVIDER_BRAVE_COMBINED, "search-key", ""
        )
    assert result == "search_answers_api_key_missing"


async def test_validate_web_search_connection_brave_answers_delegates_to_answers_key(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """brave_answers delegates to _validate_brave_answers_api_key."""
    handler = _make_handler(mock_config_entry, hass)
    with patch.object(
        handler,
        "_validate_brave_answers_api_key",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_validate:
        result = await handler._validate_web_search_connection(
            SEARCH_PROVIDER_BRAVE_ANSWERS, "", "my-answers-key"
        )
    mock_validate.assert_called_once_with("my-answers-key")
    assert result is None


async def test_validate_web_search_connection_brave_answers_returns_error_when_key_invalid(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """brave_answers propagates error returned by _validate_brave_answers_api_key."""
    handler = _make_handler(mock_config_entry, hass)
    with patch.object(
        handler,
        "_validate_brave_answers_api_key",
        new_callable=AsyncMock,
        return_value="invalid_answers_api_key",
    ):
        result = await handler._validate_web_search_connection(
            SEARCH_PROVIDER_BRAVE_ANSWERS, "", "bad-answers-key"
        )
    assert result == "invalid_answers_api_key"


async def test_validate_web_search_connection_brave_combined_validates_both_keys(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """brave_combined calls both _validate_brave_api_key and _validate_brave_answers_api_key."""
    handler = _make_handler(mock_config_entry, hass)
    with (
        patch.object(
            handler, "_validate_brave_api_key", new_callable=AsyncMock, return_value=None
        ) as mock_search,
        patch.object(
            handler, "_validate_brave_answers_api_key", new_callable=AsyncMock, return_value=None
        ) as mock_answers,
    ):
        result = await handler._validate_web_search_connection(
            SEARCH_PROVIDER_BRAVE_COMBINED, "s-key", "a-key"
        )
    mock_search.assert_called_once_with("s-key")
    mock_answers.assert_called_once_with("a-key")
    assert result is None


async def test_validate_web_search_connection_brave_combined_search_key_fails_returns_error(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """brave_combined stops at first failure — bad search key is returned immediately."""
    handler = _make_handler(mock_config_entry, hass)
    with (
        patch.object(
            handler,
            "_validate_brave_api_key",
            new_callable=AsyncMock,
            return_value="invalid_api_key",
        ),
        patch.object(
            handler, "_validate_brave_answers_api_key", new_callable=AsyncMock, return_value=None
        ) as mock_answers,
    ):
        result = await handler._validate_web_search_connection(
            SEARCH_PROVIDER_BRAVE_COMBINED, "bad-search-key", "a-key"
        )
    mock_answers.assert_not_awaited()
    assert result == "invalid_api_key"


# ---------------------------------------------------------------------------
# _validate_brave_answers_api_key — HTTP variants
# ---------------------------------------------------------------------------


async def test_validate_brave_answers_api_key_http_200_returns_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """HTTP 200 from Brave Answers → None (success)."""
    handler = _make_handler(mock_config_entry, hass)
    with patch(_PATCH_BRAVE_SESSION, return_value=_make_brave_session_cm(200)):
        result = await handler._validate_brave_answers_api_key("valid-answers-key")
    assert result is None


async def test_validate_brave_answers_api_key_http_401_returns_invalid(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """HTTP 401 from Brave Answers → 'invalid_answers_api_key'."""
    handler = _make_handler(mock_config_entry, hass)
    with patch(_PATCH_BRAVE_SESSION, return_value=_make_brave_session_cm(401)):
        result = await handler._validate_brave_answers_api_key("bad-answers-key")
    assert result == "invalid_answers_api_key"


async def test_validate_brave_answers_api_key_http_403_returns_invalid(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """HTTP 403 from Brave Answers → 'invalid_answers_api_key'."""
    handler = _make_handler(mock_config_entry, hass)
    with patch(_PATCH_BRAVE_SESSION, return_value=_make_brave_session_cm(403)):
        result = await handler._validate_brave_answers_api_key("forbidden-key")
    assert result == "invalid_answers_api_key"


async def test_validate_brave_answers_api_key_http_500_returns_unreachable(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Non-200/401/403 Answers status → 'search_api_unreachable'."""
    handler = _make_handler(mock_config_entry, hass)
    with patch(_PATCH_BRAVE_SESSION, return_value=_make_brave_session_cm(500)):
        result = await handler._validate_brave_answers_api_key("key")
    assert result == "search_api_unreachable"


async def test_validate_brave_answers_api_key_client_error_returns_unreachable(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """aiohttp.ClientError during Answers validation → 'search_api_unreachable'."""
    handler = _make_handler(mock_config_entry, hass)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("timeout"))
    session_cm.__aexit__ = AsyncMock(return_value=None)
    with patch(_PATCH_BRAVE_SESSION, return_value=session_cm):
        result = await handler._validate_brave_answers_api_key("key")
    assert result == "search_api_unreachable"


async def test_validate_brave_answers_api_key_unexpected_exception_returns_unknown(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Unexpected exception during Answers validation → 'unknown'."""
    handler = _make_handler(mock_config_entry, hass)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
    session_cm.__aexit__ = AsyncMock(return_value=None)
    with patch(_PATCH_BRAVE_SESSION, return_value=session_cm):
        result = await handler._validate_brave_answers_api_key("key")
    assert result == "unknown"


# ---------------------------------------------------------------------------
# configure_web_search — test_connection checkbox
# ---------------------------------------------------------------------------


async def test_configure_web_search_test_connection_success_returns_form(
    hass: HomeAssistant,
) -> None:
    """test_connection=True with passing validation returns form (does not save)."""
    entry = _entry_with_agents(hass, [])
    handler = _make_handler(entry, hass)
    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await handler.async_step_configure_web_search(
            {
                CONF_AGENT_NAME: "Brave Search",
                CONF_PRIORITY: 40,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
                CONF_SEARCH_API_KEY: "valid-key",
                "test_connection": True,
            }
        )

    # Must return the form, not init menu
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure_web_search"
    # Nothing saved
    assert handler.config_entry.data.get(CONF_AGENTS, []) == []


async def test_configure_web_search_test_connection_failure_shows_error(
    hass: HomeAssistant,
) -> None:
    """test_connection=True with failing validation shows the error on the form."""
    entry = _entry_with_agents(hass, [])
    handler = _make_handler(entry, hass)
    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value="invalid_api_key",
    ):
        result = await handler.async_step_configure_web_search(
            {
                CONF_AGENT_NAME: "Brave Search",
                CONF_PRIORITY: 40,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
                CONF_SEARCH_API_KEY: "bad-key",
                "test_connection": True,
            }
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "invalid_api_key"


async def test_configure_web_search_saves_answers_api_key(
    hass: HomeAssistant,
) -> None:
    """configure_web_search persists CONF_SEARCH_ANSWERS_API_KEY in agent config."""
    entry = _entry_with_agents(hass, [])
    handler = _make_handler(entry, hass)
    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await handler.async_step_configure_web_search(
            {
                CONF_AGENT_NAME: "Brave Answers Agent",
                CONF_PRIORITY: 40,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE_COMBINED,
                CONF_SEARCH_API_KEY: "search-key",
                CONF_SEARCH_ANSWERS_API_KEY: "answers-key",
                CONF_SEARCH_RESULT_COUNT: DEFAULT_SEARCH_RESULT_COUNT,
                CONF_SEARCH_MAX_SNIPPET_LEN: DEFAULT_SEARCH_MAX_SNIPPET_LEN,
                CONF_TIMEOUT: DEFAULT_SEARCH_TIMEOUT,
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
                "test_connection": False,
            }
        )

    agents = handler.config_entry.data[CONF_AGENTS]
    assert len(agents) == 1
    assert agents[0][CONF_SEARCH_ANSWERS_API_KEY] == "answers-key"
    assert agents[0][CONF_SEARCH_API_KEY] == "search-key"


# ---------------------------------------------------------------------------
# edit_agent_web_search — test_connection checkbox
# ---------------------------------------------------------------------------


async def test_edit_agent_web_search_test_connection_success_returns_form(
    hass: HomeAssistant,
) -> None:
    """test_connection=True with passing validation returns form (does not save)."""
    agent = _web_search_agent(agent_id="ws-tc-1")
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = agent

    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await handler.async_step_edit_agent_web_search(
            {
                CONF_AGENT_NAME: "Brave Search",
                CONF_PRIORITY: 40,
                CONF_AGENT_ENABLED: True,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE,
                CONF_SEARCH_API_KEY: "new-key",
                CONF_SEARCH_ANSWERS_API_KEY: "",
                CONF_SEARCH_RESULT_COUNT: DEFAULT_SEARCH_RESULT_COUNT,
                CONF_SEARCH_MAX_SNIPPET_LEN: DEFAULT_SEARCH_MAX_SNIPPET_LEN,
                CONF_TIMEOUT: DEFAULT_SEARCH_TIMEOUT,
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
                "test_connection": True,
            }
        )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "edit_agent_web_search"
    # Original agent unchanged
    saved = next(a for a in handler.config_entry.data[CONF_AGENTS] if a.get("id") == "ws-tc-1")
    assert saved[CONF_SEARCH_API_KEY] == "existing-brave-key"


async def test_edit_agent_web_search_test_connection_failure_shows_error(
    hass: HomeAssistant,
) -> None:
    """test_connection=True with failing validation shows error on edit form."""
    agent = _web_search_agent(agent_id="ws-tc-2")
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = agent

    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value="invalid_answers_api_key",
    ):
        result = await handler.async_step_edit_agent_web_search(
            {
                CONF_AGENT_NAME: "Brave Search",
                CONF_PRIORITY: 40,
                CONF_AGENT_ENABLED: True,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE_COMBINED,
                CONF_SEARCH_API_KEY: "s-key",
                CONF_SEARCH_ANSWERS_API_KEY: "bad-answers-key",
                "test_connection": True,
            }
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "invalid_answers_api_key"


async def test_edit_agent_web_search_saves_answers_api_key(hass: HomeAssistant) -> None:
    """edit_agent_web_search saves CONF_SEARCH_ANSWERS_API_KEY when a new key is provided."""
    agent = _web_search_agent(
        agent_id="ws-ans-1",
        **{CONF_SEARCH_ANSWERS_API_KEY: "old-answers-key"},
    )
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = agent

    with patch.object(
        handler,
        "_validate_web_search_connection",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await handler.async_step_edit_agent_web_search(
            {
                CONF_AGENT_NAME: "Brave Combined",
                CONF_PRIORITY: 40,
                CONF_AGENT_ENABLED: True,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE_COMBINED,
                CONF_SEARCH_API_KEY: "new-search-key",
                CONF_SEARCH_ANSWERS_API_KEY: "new-answers-key",
                CONF_SEARCH_RESULT_COUNT: DEFAULT_SEARCH_RESULT_COUNT,
                CONF_SEARCH_MAX_SNIPPET_LEN: DEFAULT_SEARCH_MAX_SNIPPET_LEN,
                CONF_TIMEOUT: DEFAULT_SEARCH_TIMEOUT,
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
                "test_connection": False,
            }
        )

    saved = next(a for a in handler.config_entry.data[CONF_AGENTS] if a.get("id") == "ws-ans-1")
    assert saved[CONF_SEARCH_ANSWERS_API_KEY] == "new-answers-key"
    assert saved[CONF_SEARCH_API_KEY] == "new-search-key"


async def test_edit_agent_web_search_keeps_existing_answers_key_when_blank(
    hass: HomeAssistant,
) -> None:
    """Blank answers key field retains the original answers key."""
    agent = _web_search_agent(
        agent_id="ws-ans-2",
        **{CONF_SEARCH_ANSWERS_API_KEY: "keep-this-answers-key"},
    )
    entry = _entry_with_agents(hass, [agent])
    handler = _make_handler(entry, hass)
    handler._agent_data["_editing_agent"] = agent

    with patch.object(
        handler, "_validate_web_search_connection", new_callable=AsyncMock
    ) as mock_validate:
        await handler.async_step_edit_agent_web_search(
            {
                CONF_AGENT_NAME: "Brave Combined",
                CONF_PRIORITY: 40,
                CONF_AGENT_ENABLED: True,
                CONF_SEARCH_PROVIDER: SEARCH_PROVIDER_BRAVE_COMBINED,
                CONF_SEARCH_API_KEY: "",  # blank → keep existing
                CONF_SEARCH_ANSWERS_API_KEY: "",  # blank → keep existing
                CONF_SEARCH_RESULT_COUNT: DEFAULT_SEARCH_RESULT_COUNT,
                CONF_SEARCH_MAX_SNIPPET_LEN: DEFAULT_SEARCH_MAX_SNIPPET_LEN,
                CONF_TIMEOUT: DEFAULT_SEARCH_TIMEOUT,
                CONF_AGENT_CACHE_ENABLED: False,
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
                "test_connection": False,
            }
        )

    mock_validate.assert_not_called()
    saved = next(a for a in handler.config_entry.data[CONF_AGENTS] if a.get("id") == "ws-ans-2")
    assert saved[CONF_SEARCH_ANSWERS_API_KEY] == "keep-this-answers-key"


# ===========================================================================
# Feature 10 — advanced_settings: force_response_language toggle
# ===========================================================================


async def test_advanced_settings_saves_force_response_language(hass: HomeAssistant) -> None:
    """Submitting advanced_settings with force_response_language=False persists the value."""
    entry = _entry_with_agents(hass, [])
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
                "purge_cache_now": False,
                CONF_FORCE_RESPONSE_LANGUAGE: False,
            }
        )

    assert entry.data[CONF_FORCE_RESPONSE_LANGUAGE] is False


async def test_advanced_settings_force_response_language_defaults_to_true(
    hass: HomeAssistant,
) -> None:
    """force_response_language defaults to True when not provided in the submitted form."""
    entry = _entry_with_agents(hass, [])
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
                "purge_cache_now": False,
                # CONF_FORCE_RESPONSE_LANGUAGE not provided → uses default
            }
        )

    assert entry.data.get(CONF_FORCE_RESPONSE_LANGUAGE, DEFAULT_FORCE_RESPONSE_LANGUAGE) is True


async def test_advanced_settings_shows_force_response_language_field(
    hass: HomeAssistant,
) -> None:
    """async_step_advanced_settings form includes the force_response_language field."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_FORCE_RESPONSE_LANGUAGE: False},
        options={},
    )
    entry.add_to_hass(hass)
    handler = _make_handler(entry, hass)

    result = await handler.async_step_advanced_settings()

    assert result["type"] == FlowResultType.FORM
    schema = result["data_schema"].schema
    keys = [k.schema if hasattr(k, "schema") else k for k in schema]
    assert CONF_FORCE_RESPONSE_LANGUAGE in keys
