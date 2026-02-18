"""Config flow for NeuralBridge integration."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
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
    CONF_GUARD_RAIL_DETOXIFY_THRESHOLD,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_GUARD_RAIL_RULES,
    CONF_GUARD_RAIL_USE_DETOXIFY,
    CONF_LANGUAGE,
    CONF_MAX_RETRIES,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_PRIORITY,
    CONF_RESPONSE_CACHE_ENABLED,
    CONF_RESPONSE_CACHE_TTL,
    CONF_RETRY_BASE_DELAY,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DATA_RESPONSE_CACHE,
    DEFAULT_AGENT_CACHE_ENABLED,
    DEFAULT_AGENT_ENABLED,
    DEFAULT_GUARD_RAIL_ACTION,
    DEFAULT_GUARD_RAIL_AI_THRESHOLD,
    DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD,
    DEFAULT_GUARD_RAIL_ENABLED,
    DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    DEFAULT_GUARD_RAIL_USE_DETOXIFY,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PRIORITY,
    DEFAULT_RESPONSE_CACHE_ENABLED,
    DEFAULT_RESPONSE_CACHE_TTL,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    DOMAIN,
    GUARD_RAIL_ACTION_BLOCK,
    GUARD_RAIL_ACTION_NOTIFY_ASK,
    GUARD_RAIL_ACTION_WARN,
    GUARD_RAIL_CATEGORY_HARMFUL,
    GUARD_RAIL_CATEGORY_INAPPROPRIATE,
    GUARD_RAIL_CATEGORY_PRIVACY,
    GUARD_RAIL_CATEGORY_SECURITY,
    PRIORITY_MAX,
    PRIORITY_MIN,
)
from .languages_loader import get_string, list_available_languages

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigFlowResult

_LOGGER = logging.getLogger(__name__)
_HTTP_OK = 200


class NeuralBridgeConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for NeuralBridge."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle the initial step — create integration instance."""
        if user_input is not None:
            await self.async_set_unique_id(DOMAIN)
            self._abort_if_unique_id_configured()

            return self.async_create_entry(
                title="NeuralBridge",
                data={CONF_AGENTS: [], CONF_LANGUAGE: DEFAULT_LANGUAGE},
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({}),
            description_placeholders={"docs_url": "https://github.com/pfrye/NeuralBridge"},
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> NeuralBridgeOptionsFlowHandler:
        """Get the options flow for this handler."""
        return NeuralBridgeOptionsFlowHandler(config_entry)


class NeuralBridgeOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for NeuralBridge."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialise options flow."""
        self.config_entry = config_entry
        self._agent_data: dict[str, Any] = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def _lang(self) -> str:
        """Return the currently configured language code."""
        return str(self.config_entry.data.get(CONF_LANGUAGE, DEFAULT_LANGUAGE))

    def _s(self, *keys: str) -> str:
        """Return a localised string for the current language.

        Args:
            *keys: Path within the language YAML (e.g. ``"agent_types"``, ``"ollama"``).

        Returns:
            Localised string, or empty string if the key path is not found.
        """
        return get_string(self._lang, *keys)

    # ── Main menu ─────────────────────────────────────────────────────────────

    async def async_step_init(self, _user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Manage the options — main menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "add_agent",
                "manage_agents",
                "configure_guard_rails",
                "configure_guard_rail_rules",
                "advanced_settings",
                "language_settings",
            ],
        )

    # ── Add agent ─────────────────────────────────────────────────────────────

    async def async_step_add_agent(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add a new agent — select type."""
        if user_input is not None:
            self._agent_data[CONF_AGENT_TYPE] = user_input[CONF_AGENT_TYPE]

            if user_input[CONF_AGENT_TYPE] == AGENT_TYPE_OLLAMA:
                return await self.async_step_configure_ollama()
            if user_input[CONF_AGENT_TYPE] == AGENT_TYPE_EXISTING:
                return await self.async_step_configure_existing()
            return await self.async_step_configure_local()

        return self.async_show_form(
            step_id="add_agent",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_TYPE): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[  # type: ignore[typeddict-item]
                                {
                                    "value": AGENT_TYPE_LOCAL_HA,
                                    "label": self._s("agent_types", "home_assistant"),
                                },
                                {
                                    "value": AGENT_TYPE_EXISTING,
                                    "label": self._s("agent_types", "existing_integration"),
                                },
                                {
                                    "value": AGENT_TYPE_OLLAMA,
                                    "label": self._s("agent_types", "ollama"),
                                },
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    # ── Ollama connection validation ───────────────────────────────────────────

    async def _validate_ollama_connection(self, url: str, model: str) -> str | None:
        """Validate an Ollama URL and model, returning an error key or None.

        Args:
            url: Ollama base URL.
            model: Model name to verify exists on the server.

        Returns:
            An error key string if validation fails, None if successful.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "invalid_url_scheme"

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"{url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response,
            ):
                if response.status != _HTTP_OK:
                    return "cannot_connect"
                data = await response.json()
                models = [m["name"] for m in data.get("models", [])]
                if model not in models:
                    return "model_not_found"
        except aiohttp.ClientError:
            return "cannot_connect"
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected error validating Ollama: %s", err)
            return "unknown"

        return None

    # ── Configure Ollama ──────────────────────────────────────────────────────

    async def async_step_configure_ollama(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure Ollama agent."""
        errors: dict[str, str] = {}

        if user_input is not None:
            error_key = await self._validate_ollama_connection(
                user_input[CONF_OLLAMA_URL], user_input[CONF_OLLAMA_MODEL]
            )
            if error_key:
                errors["base"] = error_key

            if not errors:
                agent_id = str(uuid4())
                agent_config = {
                    "id": agent_id,
                    CONF_AGENT_TYPE: AGENT_TYPE_OLLAMA,
                    CONF_AGENT_ENABLED: DEFAULT_AGENT_ENABLED,
                    CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                    CONF_PRIORITY: user_input[CONF_PRIORITY],
                    CONF_OLLAMA_URL: user_input[CONF_OLLAMA_URL],
                    CONF_OLLAMA_MODEL: user_input[CONF_OLLAMA_MODEL],
                    CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    CONF_SYSTEM_PROMPT: user_input.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                    CONF_AGENT_CACHE_ENABLED: user_input.get(
                        CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
                    ),
                    CONF_GUARD_RAIL_ENABLED_FOR_AGENT: user_input.get(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                        (
                            DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
                            if user_input[CONF_PRIORITY] > 0
                            else False
                        ),
                    ),
                }

                agents = list(self.config_entry.data.get(CONF_AGENTS, []))
                agents.append(agent_config)
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={**self.config_entry.data, CONF_AGENTS: agents},
                )
                return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="configure_ollama",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_NAME): str,
                    vol.Required(CONF_PRIORITY, default=DEFAULT_PRIORITY): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Required(CONF_OLLAMA_URL, default=DEFAULT_OLLAMA_URL): str,
                    vol.Required(CONF_OLLAMA_MODEL): str,
                    vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=120,
                            unit_of_measurement="seconds",
                        )
                    ),
                    vol.Optional(
                        CONF_SYSTEM_PROMPT, default=DEFAULT_SYSTEM_PROMPT
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            multiline=True,
                            type=selector.TextSelectorType.TEXT,
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_CACHE_ENABLED, default=DEFAULT_AGENT_CACHE_ENABLED
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                        default=DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
                    ): selector.BooleanSelector(),
                }
            ),
            errors=errors,
            description_placeholders={
                "priority_info": self._s("placeholders", "priority_info"),
            },
        )

    # ── Configure existing integration ────────────────────────────────────────

    async def async_step_configure_existing(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure existing integration agent."""
        if user_input is not None:
            agent_id = str(uuid4())
            agent_config = {
                "id": agent_id,
                CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
                CONF_AGENT_ENABLED: DEFAULT_AGENT_ENABLED,
                CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                CONF_PRIORITY: user_input[CONF_PRIORITY],
                CONF_ENTITY_ID: user_input[CONF_ENTITY_ID],
                CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                CONF_AGENT_CACHE_ENABLED: user_input.get(
                    CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
                ),
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: user_input.get(
                    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                    (
                        DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
                        if user_input[CONF_PRIORITY] > 0
                        else False
                    ),
                ),
            }

            agents = list(self.config_entry.data.get(CONF_AGENTS, []))
            agents.append(agent_config)
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={**self.config_entry.data, CONF_AGENTS: agents},
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="configure_existing",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_NAME): str,
                    vol.Required(CONF_PRIORITY, default=DEFAULT_PRIORITY): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Required(CONF_ENTITY_ID): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=CONVERSATION_DOMAIN,
                        )
                    ),
                    vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=120,
                            unit_of_measurement="seconds",
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_CACHE_ENABLED, default=DEFAULT_AGENT_CACHE_ENABLED
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                        default=DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
                    ): selector.BooleanSelector(),
                }
            ),
            description_placeholders={
                "priority_info": self._s("placeholders", "priority_info"),
            },
        )

    # ── Configure local HA agent ──────────────────────────────────────────────

    async def async_step_configure_local(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure local Home Assistant agent."""
        if user_input is not None:
            agent_id = str(uuid4())
            agent_config = {
                "id": agent_id,
                CONF_AGENT_TYPE: AGENT_TYPE_LOCAL_HA,
                CONF_AGENT_ENABLED: DEFAULT_AGENT_ENABLED,
                CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                CONF_PRIORITY: user_input[CONF_PRIORITY],
                CONF_ENTITY_ID: "conversation.home_assistant",
                CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                CONF_AGENT_CACHE_ENABLED: user_input.get(
                    CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
                ),
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: user_input.get(
                    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                    (
                        DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
                        if user_input[CONF_PRIORITY] > 0
                        else False
                    ),
                ),
            }

            agents = list(self.config_entry.data.get(CONF_AGENTS, []))
            agents.append(agent_config)
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={**self.config_entry.data, CONF_AGENTS: agents},
            )
            return self.async_create_entry(title="", data={})

        default_name = self._s("agent_management", "default_ha_agent_name")
        return self.async_show_form(
            step_id="configure_local",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_NAME, default=default_name): str,
                    vol.Required(CONF_PRIORITY, default=DEFAULT_PRIORITY): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=120,
                            unit_of_measurement="seconds",
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_CACHE_ENABLED, default=DEFAULT_AGENT_CACHE_ENABLED
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                        default=DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
                    ): selector.BooleanSelector(),
                }
            ),
            description_placeholders={
                "priority_info": self._s("placeholders", "priority_info"),
            },
        )

    # ── Manage agents ─────────────────────────────────────────────────────────

    async def async_step_manage_agents(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage existing agents — select an agent to act on."""
        agents = list(self.config_entry.data.get(CONF_AGENTS, []))

        if not agents:
            return self.async_show_form(
                step_id="manage_agents",
                description_placeholders={
                    "message": self._s("agent_management", "no_agents_yet"),
                },
            )

        if user_input is not None:
            self._agent_data["_selected_agent_id"] = user_input["agent_id"]
            return await self.async_step_manage_agent_action()

        enabled_label = self._s("agent_management", "status_enabled")
        disabled_label = self._s("agent_management", "status_disabled")

        agent_options: list[selector.SelectOptionDict] = [
            {
                "value": agent["id"],
                "label": (
                    f"{agent[CONF_AGENT_NAME]}  (P:{agent[CONF_PRIORITY]})  "
                    f"{enabled_label if agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED) else disabled_label}"
                ),
            }
            for agent in agents
        ]

        return self.async_show_form(
            step_id="manage_agents",
            data_schema=vol.Schema(
                {
                    vol.Required("agent_id"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=agent_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_manage_agent_action(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select and apply an action (edit / toggle / delete) for the selected agent."""
        agents = list(self.config_entry.data.get(CONF_AGENTS, []))
        agent_id: str = self._agent_data.get("_selected_agent_id", "")
        agent = next((a for a in agents if a.get("id") == agent_id), None)

        if user_input is not None and agent is not None:
            action = user_input.get("action")

            if action == "edit":
                return await self._route_to_edit_step()
            if action == "toggle":
                current_enabled = agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED)
                agent[CONF_AGENT_ENABLED] = not current_enabled
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={**self.config_entry.data, CONF_AGENTS: agents},
                )
            elif action == "delete":
                agents = [a for a in agents if a.get("id") != agent_id]
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={**self.config_entry.data, CONF_AGENTS: agents},
                )

            return self.async_create_entry(title="", data={})

        agent_name = agent.get(CONF_AGENT_NAME, "Unknown") if agent else "Unknown"
        is_enabled = agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED) if agent else True
        toggle_label = (
            self._s("agent_management", "action_disable")
            if is_enabled
            else self._s("agent_management", "action_enable")
        )

        return self.async_show_form(
            step_id="manage_agent_action",
            data_schema=vol.Schema(
                {
                    vol.Required("action"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[  # type: ignore[typeddict-item]
                                {
                                    "value": "edit",
                                    "label": self._s("agent_management", "action_edit"),
                                },
                                {"value": "toggle", "label": toggle_label},
                                {
                                    "value": "delete",
                                    "label": self._s("agent_management", "action_delete"),
                                },
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
            description_placeholders={"agent_name": agent_name},
        )

    async def _route_to_edit_step(self) -> ConfigFlowResult:
        """Dispatch to the type-specific edit step for the selected agent."""
        agents = list(self.config_entry.data.get(CONF_AGENTS, []))
        agent_id: str = self._agent_data.get("_selected_agent_id", "")
        agent = next((a for a in agents if a.get("id") == agent_id), None)

        if agent is None:
            return self.async_create_entry(title="", data={})

        self._agent_data["_editing_agent"] = dict(agent)
        agent_type = agent.get(CONF_AGENT_TYPE)

        if agent_type == AGENT_TYPE_OLLAMA:
            return await self.async_step_edit_agent_ollama()
        if agent_type == AGENT_TYPE_EXISTING:
            return await self.async_step_edit_agent_existing()
        return await self.async_step_edit_agent_local()

    # ── Edit Ollama agent ──────────────────────────────────────────────────────

    async def async_step_edit_agent_ollama(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit settings for an existing Ollama agent."""
        errors: dict[str, str] = {}
        agent = self._agent_data.get("_editing_agent", {})
        agent_id: str = agent.get("id", "")

        if user_input is not None:
            error_key = await self._validate_ollama_connection(
                user_input[CONF_OLLAMA_URL], user_input[CONF_OLLAMA_MODEL]
            )
            if error_key:
                errors["base"] = error_key

            if not errors:
                updated = {
                    **agent,
                    CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                    CONF_PRIORITY: user_input[CONF_PRIORITY],
                    CONF_OLLAMA_URL: user_input[CONF_OLLAMA_URL],
                    CONF_OLLAMA_MODEL: user_input[CONF_OLLAMA_MODEL],
                    CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    CONF_SYSTEM_PROMPT: user_input.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                    CONF_AGENT_CACHE_ENABLED: user_input.get(
                        CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
                    ),
                    CONF_GUARD_RAIL_ENABLED_FOR_AGENT: user_input.get(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT, DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
                    ),
                }
                agents = list(self.config_entry.data.get(CONF_AGENTS, []))
                agents = [updated if a.get("id") == agent_id else a for a in agents]
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={**self.config_entry.data, CONF_AGENTS: agents},
                )
                return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="edit_agent_ollama",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_NAME, default=agent.get(CONF_AGENT_NAME, "")): str,
                    vol.Required(
                        CONF_PRIORITY,
                        default=agent.get(CONF_PRIORITY, DEFAULT_PRIORITY),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Required(
                        CONF_OLLAMA_URL,
                        default=agent.get(CONF_OLLAMA_URL, DEFAULT_OLLAMA_URL),
                    ): str,
                    vol.Required(CONF_OLLAMA_MODEL, default=agent.get(CONF_OLLAMA_MODEL, "")): str,
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=agent.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=120,
                            unit_of_measurement="seconds",
                        )
                    ),
                    vol.Optional(
                        CONF_SYSTEM_PROMPT,
                        default=agent.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            multiline=True,
                            type=selector.TextSelectorType.TEXT,
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_CACHE_ENABLED,
                        default=agent.get(CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                        default=agent.get(
                            CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                            DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
                        ),
                    ): selector.BooleanSelector(),
                }
            ),
            errors=errors,
            description_placeholders={
                "priority_info": self._s("placeholders", "priority_info"),
            },
        )

    # ── Edit existing-integration agent ───────────────────────────────────────

    async def async_step_edit_agent_existing(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit settings for an existing-integration agent."""
        agent = self._agent_data.get("_editing_agent", {})
        agent_id: str = agent.get("id", "")

        if user_input is not None:
            updated = {
                **agent,
                CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                CONF_PRIORITY: user_input[CONF_PRIORITY],
                CONF_ENTITY_ID: user_input[CONF_ENTITY_ID],
                CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                CONF_AGENT_CACHE_ENABLED: user_input.get(
                    CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
                ),
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: user_input.get(
                    CONF_GUARD_RAIL_ENABLED_FOR_AGENT, DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
                ),
            }
            agents = list(self.config_entry.data.get(CONF_AGENTS, []))
            agents = [updated if a.get("id") == agent_id else a for a in agents]
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={**self.config_entry.data, CONF_AGENTS: agents},
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="edit_agent_existing",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_NAME, default=agent.get(CONF_AGENT_NAME, "")): str,
                    vol.Required(
                        CONF_PRIORITY,
                        default=agent.get(CONF_PRIORITY, DEFAULT_PRIORITY),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Required(
                        CONF_ENTITY_ID,
                        default=agent.get(CONF_ENTITY_ID, ""),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=CONVERSATION_DOMAIN,
                        )
                    ),
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=agent.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=120,
                            unit_of_measurement="seconds",
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_CACHE_ENABLED,
                        default=agent.get(CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                        default=agent.get(
                            CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                            DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
                        ),
                    ): selector.BooleanSelector(),
                }
            ),
            description_placeholders={
                "priority_info": self._s("placeholders", "priority_info"),
            },
        )

    # ── Edit local HA agent ───────────────────────────────────────────────────

    async def async_step_edit_agent_local(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit settings for a local Home Assistant agent."""
        agent = self._agent_data.get("_editing_agent", {})
        agent_id: str = agent.get("id", "")

        if user_input is not None:
            updated = {
                **agent,
                CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                CONF_PRIORITY: user_input[CONF_PRIORITY],
                CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                CONF_AGENT_CACHE_ENABLED: user_input.get(
                    CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
                ),
                CONF_GUARD_RAIL_ENABLED_FOR_AGENT: user_input.get(
                    CONF_GUARD_RAIL_ENABLED_FOR_AGENT, DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
                ),
            }
            agents = list(self.config_entry.data.get(CONF_AGENTS, []))
            agents = [updated if a.get("id") == agent_id else a for a in agents]
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={**self.config_entry.data, CONF_AGENTS: agents},
            )
            return self.async_create_entry(title="", data={})

        default_name = self._s("agent_management", "default_ha_agent_name")
        return self.async_show_form(
            step_id="edit_agent_local",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_AGENT_NAME,
                        default=agent.get(CONF_AGENT_NAME, default_name),
                    ): str,
                    vol.Required(
                        CONF_PRIORITY,
                        default=agent.get(CONF_PRIORITY, DEFAULT_PRIORITY),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=agent.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=120,
                            unit_of_measurement="seconds",
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_CACHE_ENABLED,
                        default=agent.get(CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                        default=agent.get(
                            CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                            DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
                        ),
                    ): selector.BooleanSelector(),
                }
            ),
            description_placeholders={
                "priority_info": self._s("placeholders", "priority_info"),
            },
        )

    # ── Advanced settings ─────────────────────────────────────────────────────

    async def async_step_advanced_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure global response cache and retry settings."""
        if user_input is not None:
            cache_enabled: bool = user_input.get(
                CONF_RESPONSE_CACHE_ENABLED, DEFAULT_RESPONSE_CACHE_ENABLED
            )
            cache_ttl: int = int(
                user_input.get(CONF_RESPONSE_CACHE_TTL, DEFAULT_RESPONSE_CACHE_TTL)
            )
            purge_now: bool = user_input.get("purge_cache_now", False)
            max_retries: int = int(user_input.get(CONF_MAX_RETRIES, DEFAULT_MAX_RETRIES))
            retry_base_delay: float = float(
                user_input.get(CONF_RETRY_BASE_DELAY, DEFAULT_RETRY_BASE_DELAY)
            )

            current_data = {
                **self.config_entry.data,
                CONF_RESPONSE_CACHE_ENABLED: cache_enabled,
                CONF_RESPONSE_CACHE_TTL: cache_ttl,
                CONF_MAX_RETRIES: max_retries,
                CONF_RETRY_BASE_DELAY: retry_base_delay,
            }
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data=current_data,
            )

            entry_data = self.hass.data.get(DOMAIN, {}).get(self.config_entry.entry_id, {})
            response_cache = entry_data.get(DATA_RESPONSE_CACHE)
            if response_cache is not None:
                if purge_now:
                    purged = response_cache.invalidate()
                    _LOGGER.info("Response cache purged: %d entries removed", purged)
                response_cache.configure(enabled=cache_enabled, ttl_seconds=cache_ttl)

            return self.async_create_entry(title="", data={})

        entry_data = self.config_entry.data
        current_enabled = entry_data.get(
            CONF_RESPONSE_CACHE_ENABLED, DEFAULT_RESPONSE_CACHE_ENABLED
        )
        current_ttl = entry_data.get(CONF_RESPONSE_CACHE_TTL, DEFAULT_RESPONSE_CACHE_TTL)
        current_max_retries = entry_data.get(CONF_MAX_RETRIES, DEFAULT_MAX_RETRIES)
        current_retry_delay = entry_data.get(CONF_RETRY_BASE_DELAY, DEFAULT_RETRY_BASE_DELAY)

        return self.async_show_form(
            step_id="advanced_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_RESPONSE_CACHE_ENABLED, default=current_enabled
                    ): selector.BooleanSelector(),
                    vol.Required(
                        CONF_RESPONSE_CACHE_TTL, default=current_ttl
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=60,
                            max=600,
                            step=30,
                            unit_of_measurement="seconds",
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional("purge_cache_now", default=False): selector.BooleanSelector(),
                    vol.Required(
                        CONF_MAX_RETRIES, default=current_max_retries
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            max=5,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),
                    vol.Required(
                        CONF_RETRY_BASE_DELAY, default=current_retry_delay
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.5,
                            max=5.0,
                            step=0.5,
                            unit_of_measurement="seconds",
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                }
            ),
        )

    # ── Guard rails ───────────────────────────────────────────────────────────

    async def async_step_configure_guard_rails(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure global guard rail settings."""
        if user_input is not None:
            current_data = {
                **self.config_entry.data,
                CONF_GUARD_RAIL_ENABLED: user_input[CONF_GUARD_RAIL_ENABLED],
                CONF_GUARD_RAIL_ACTION: user_input[CONF_GUARD_RAIL_ACTION],
                CONF_GUARD_RAIL_AI_THRESHOLD: user_input[CONF_GUARD_RAIL_AI_THRESHOLD],
                CONF_GUARD_RAIL_USE_DETOXIFY: user_input.get(
                    CONF_GUARD_RAIL_USE_DETOXIFY, DEFAULT_GUARD_RAIL_USE_DETOXIFY
                ),
                CONF_GUARD_RAIL_DETOXIFY_THRESHOLD: user_input.get(
                    CONF_GUARD_RAIL_DETOXIFY_THRESHOLD, DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD
                ),
            }

            if user_input[CONF_GUARD_RAIL_ENABLED]:
                router_agents = [
                    a for a in current_data.get(CONF_AGENTS, []) if a.get(CONF_PRIORITY) == 0
                ]

                if router_agents:
                    current_data["guard_rail_agent_id"] = router_agents[0]["id"]
                else:
                    return self.async_show_form(
                        step_id="configure_guard_rails",
                        errors={"base": "no_router_agent"},
                        description_placeholders={
                            "info": self._s("placeholders", "no_router_agent_error"),
                        },
                    )

            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data=current_data,
            )
            return self.async_create_entry(title="", data={})

        current_enabled = self.config_entry.data.get(
            CONF_GUARD_RAIL_ENABLED, DEFAULT_GUARD_RAIL_ENABLED
        )
        current_action = self.config_entry.data.get(
            CONF_GUARD_RAIL_ACTION, DEFAULT_GUARD_RAIL_ACTION
        )
        current_threshold = self.config_entry.data.get(
            CONF_GUARD_RAIL_AI_THRESHOLD, DEFAULT_GUARD_RAIL_AI_THRESHOLD
        )
        current_use_detoxify = self.config_entry.data.get(
            CONF_GUARD_RAIL_USE_DETOXIFY, DEFAULT_GUARD_RAIL_USE_DETOXIFY
        )
        current_detoxify_threshold = self.config_entry.data.get(
            CONF_GUARD_RAIL_DETOXIFY_THRESHOLD, DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD
        )

        return self.async_show_form(
            step_id="configure_guard_rails",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_GUARD_RAIL_ENABLED, default=current_enabled
                    ): selector.BooleanSelector(),
                    vol.Required(
                        CONF_GUARD_RAIL_ACTION, default=current_action
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[  # type: ignore[typeddict-item]
                                {
                                    "value": GUARD_RAIL_ACTION_NOTIFY_ASK,
                                    "label": self._s("guard_rail_actions", "notify_ask"),
                                },
                                {
                                    "value": GUARD_RAIL_ACTION_WARN,
                                    "label": self._s("guard_rail_actions", "warn"),
                                },
                                {
                                    "value": GUARD_RAIL_ACTION_BLOCK,
                                    "label": self._s("guard_rail_actions", "block"),
                                },
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(
                        CONF_GUARD_RAIL_AI_THRESHOLD, default=current_threshold
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.5,
                            max=1.0,
                            step=0.05,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_GUARD_RAIL_USE_DETOXIFY, default=current_use_detoxify
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_GUARD_RAIL_DETOXIFY_THRESHOLD,
                        default=current_detoxify_threshold,
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.5,
                            max=1.0,
                            step=0.05,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                }
            ),
            description_placeholders={
                "info": self._s("placeholders", "guard_rails_info"),
            },
        )

    # ── Guard rail rules ──────────────────────────────────────────────────────

    async def async_step_configure_guard_rail_rules(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure custom guard rail regex patterns per category."""
        categories = [
            GUARD_RAIL_CATEGORY_HARMFUL,
            GUARD_RAIL_CATEGORY_PRIVACY,
            GUARD_RAIL_CATEGORY_SECURITY,
            GUARD_RAIL_CATEGORY_INAPPROPRIATE,
        ]

        if user_input is not None:
            rules, errors = self._parse_guard_rail_rules(user_input, categories)
            if errors:
                return self._show_guard_rail_rules_form(categories, user_input, errors)

            current_data = {
                **self.config_entry.data,
                CONF_GUARD_RAIL_RULES: rules,
            }
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data=current_data,
            )
            return self.async_create_entry(title="", data={})

        saved_rules: dict[str, list[str]] = self.config_entry.data.get(CONF_GUARD_RAIL_RULES, {})
        prefilled = {cat: "\n".join(saved_rules.get(cat, [])) for cat in categories}
        return self._show_guard_rail_rules_form(categories, prefilled, {})

    def _parse_guard_rail_rules(
        self,
        user_input: dict[str, Any],
        categories: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Parse and validate user-supplied regex patterns.

        Args:
            user_input: Raw form data containing newline-delimited pattern strings.
            categories: List of category keys to process.

        Returns:
            A tuple of (rules dict, errors dict). errors is empty on success.
        """
        rules: dict[str, list[str]] = {}
        errors: dict[str, str] = {}
        for cat in categories:
            raw = user_input.get(cat, "")
            patterns = [p.strip() for p in raw.splitlines() if p.strip()]
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error:
                    errors["base"] = "invalid_regex"
                    return rules, errors
            rules[cat] = patterns
        return rules, errors

    def _show_guard_rail_rules_form(
        self,
        categories: list[str],
        values: dict[str, Any],
        errors: dict[str, str],
    ) -> ConfigFlowResult:
        """Show the guard rail rules form.

        Args:
            categories: List of category keys.
            values: Pre-filled values for each category field.
            errors: Validation errors to display.

        Returns:
            Form flow result.
        """
        schema_fields: dict[Any, Any] = {}
        for cat in categories:
            schema_fields[vol.Optional(cat, default=values.get(cat, ""))] = selector.TextSelector(
                selector.TextSelectorConfig(multiline=True)
            )
        return self.async_show_form(
            step_id="configure_guard_rail_rules",
            data_schema=vol.Schema(schema_fields),
            errors=errors,
            description_placeholders={
                "info": self._s("placeholders", "guard_rail_rules_info"),
            },
        )

    # ── Language settings ─────────────────────────────────────────────────────

    async def async_step_language_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure the display language for NeuralBridge."""
        if user_input is not None:
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={**self.config_entry.data, CONF_LANGUAGE: user_input[CONF_LANGUAGE]},
            )
            return self.async_create_entry(title="", data={})

        available = list_available_languages()
        language_options: list[selector.SelectOptionDict] = [
            {"value": lang["code"], "label": lang["name"]} for lang in available
        ]
        current_language = str(self.config_entry.data.get(CONF_LANGUAGE, DEFAULT_LANGUAGE))

        return self.async_show_form(
            step_id="language_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_LANGUAGE, default=current_language): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=language_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )
