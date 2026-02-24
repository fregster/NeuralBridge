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
    CONF_GUARD_RAIL_DETOXIFY_THRESHOLD,
    CONF_GUARD_RAIL_ENABLED,
    CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
    CONF_GUARD_RAIL_RULES,
    CONF_GUARD_RAIL_USE_DETOXIFY,
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
    DEFAULT_AGENT_ENABLED,
    DEFAULT_DEFAULT_PROMPT,
    DEFAULT_ENABLE_HOME_CONTROL,
    DEFAULT_FORCE_RESPONSE_LANGUAGE,
    DEFAULT_GUARD_RAIL_ACTION,
    DEFAULT_GUARD_RAIL_AI_THRESHOLD,
    DEFAULT_GUARD_RAIL_DETOXIFY_THRESHOLD,
    DEFAULT_GUARD_RAIL_ENABLED,
    DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
    DEFAULT_GUARD_RAIL_USE_DETOXIFY,
    DEFAULT_IS_ROUTER,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PRIORITY,
    DEFAULT_RESPONSE_CACHE_ENABLED,
    DEFAULT_RESPONSE_CACHE_TTL,
    DEFAULT_RETRY_BASE_DELAY,
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
    GUARD_RAIL_ACTION_NOTIFY_ASK,
    GUARD_RAIL_ACTION_WARN,
    GUARD_RAIL_CATEGORY_HARMFUL,
    GUARD_RAIL_CATEGORY_INAPPROPRIATE,
    GUARD_RAIL_CATEGORY_PRIVACY,
    GUARD_RAIL_CATEGORY_SECURITY,
    PRIORITY_MAX,
    PRIORITY_MIN_PROCESSING,
    PRIORITY_ROUTER,
    ROUTER_FALLBACK_BLOCK,
    ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
    ROUTER_FALLBACK_SKIP_ROUTING,
    ROUTER_LOG_LEVEL_COMPLEXITY,
    ROUTER_LOG_LEVEL_DEBUG,
    ROUTER_LOG_LEVEL_DEBUG_QUERY,
    ROUTER_LOG_LEVEL_NONE,
    SEARCH_PROVIDER_BRAVE,
    SEARCH_PROVIDER_BRAVE_ANSWERS,
    SEARCH_PROVIDER_BRAVE_COMBINED,
    SEARCH_PROVIDERS,
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
        config_entry: config_entries.ConfigEntry,  # noqa: ARG004
    ) -> NeuralBridgeOptionsFlowHandler:
        """Get the options flow for this handler."""
        return NeuralBridgeOptionsFlowHandler()


class NeuralBridgeOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for NeuralBridge."""

    def __init__(self) -> None:
        """Initialise options flow."""
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
        """Manage the options — run migration then show the main menu."""
        self._migrate_legacy_router_agents()
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "default_prompt",
                "add_agent",
                "configure_routing_agent",
                "manage_agents",
                "configure_guard_rails",
                "advanced_settings",
                "language_settings",
                "done",
            ],
        )

    # ── Legacy migration ──────────────────────────────────────────────────────

    def _migrate_legacy_router_agents(self) -> None:
        """Idempotently upgrade priority-0 agents to first-class routing agents.

        Any agent stored with ``priority == 0`` and no explicit ``is_router``
        key is a legacy router agent.  This method sets ``is_router = True`` on
        each such agent so the new routing logic recognises them correctly.

        The migration is idempotent — running it multiple times is safe.
        """
        agents: list[dict[str, Any]] = list(self.config_entry.data.get(CONF_AGENTS, []))
        changed = False
        updated: list[dict[str, Any]] = []
        for agent in agents:
            if agent.get(CONF_PRIORITY) == PRIORITY_ROUTER and CONF_IS_ROUTER not in agent:
                updated.append({**agent, CONF_IS_ROUTER: True})
                changed = True
            else:
                updated.append(agent)

        if changed:
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={**self.config_entry.data, CONF_AGENTS: updated},
            )
            _LOGGER.debug("Migrated legacy priority-0 router agents to is_router=True")

    # ── Default prompt ────────────────────────────────────────────────────────

    async def async_step_default_prompt(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure the global default system prompt.

        This prompt is used for all Ollama agents that do not have their own
        per-agent system prompt set.

        Args:
            user_input: Form data submitted by the user, or None on first load.

        Returns:
            Form result or redirect to main menu after saving.
        """
        if user_input is not None:
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={
                    **self.config_entry.data,
                    CONF_DEFAULT_PROMPT: user_input[CONF_DEFAULT_PROMPT],
                },
            )
            return await self.async_step_init()

        current_prompt: str = self.config_entry.data.get(
            CONF_DEFAULT_PROMPT, DEFAULT_DEFAULT_PROMPT
        )
        return self.async_show_form(
            step_id="default_prompt",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_DEFAULT_PROMPT, default=current_prompt
                    ): selector.TextSelector(selector.TextSelectorConfig(multiline=True)),
                }
            ),
        )

    # ── Configure routing agent (unified add / edit) ──────────────────────────

    async def async_step_configure_routing_agent(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure the single routing agent — handles both add and edit.

        Only one routing agent is permitted.  When opened from the main menu
        this step auto-loads the existing routing agent for editing.  When
        reached via ``manage_agents`` the agent is pre-loaded in
        ``_agent_data["_editing_agent"]``.

        The routing agent uses any installed Home Assistant conversation agent
        (Gemini, Claude, ChatGPT, etc.) as its back-end.  The classification
        prompt is sent as the user message and the JSON response is parsed to
        produce a routing decision.

        Args:
            user_input: Form data submitted by the user, or None on first load.

        Returns:
            Form result or redirect to main menu after saving.
        """
        errors: dict[str, str] = {}

        # ── Resolve the existing routing agent (if any) ──────────────────────
        agents: list[dict[str, Any]] = list(self.config_entry.data.get(CONF_AGENTS, []))
        existing_routers = [
            a
            for a in agents
            if a.get(CONF_IS_ROUTER, False) or a.get(CONF_PRIORITY) == PRIORITY_ROUTER
        ]
        # manage_agents path pre-loads agent into _editing_agent; fall back to
        # the first existing router when entering from the main menu.
        existing: dict[str, Any] | None = self._agent_data.get("_editing_agent") or (
            existing_routers[0] if existing_routers else None
        )
        is_editing = existing is not None
        agent_id: str = existing.get("id", "") if existing else ""

        # ── Handle form submission ────────────────────────────────────────────
        if user_input is not None:
            if is_editing and user_input.get("delete_agent"):
                # Ensure the agent is registered for the delete confirmation step
                if not self._agent_data.get("_editing_agent"):
                    self._agent_data["_editing_agent"] = existing
                return await self.async_step_confirm_delete_agent()

            entity_id: str = user_input.get(CONF_ENTITY_ID, "")
            if not entity_id or not self.hass.states.get(entity_id):
                errors["base"] = "entity_not_found"

            if not errors:
                custom_prompt: str = user_input.get(
                    CONF_ROUTER_CUSTOM_PROMPT, DEFAULT_ROUTER_CUSTOM_PROMPT
                ).strip()
                base: dict[str, Any] = existing or {}
                updated: dict[str, Any] = {
                    **base,
                    "id": agent_id or str(uuid4()),
                    CONF_AGENT_TYPE: AGENT_TYPE_EXISTING,
                    CONF_IS_ROUTER: True,
                    CONF_PRIORITY: PRIORITY_ROUTER,
                    CONF_AGENT_ENABLED: user_input.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED),
                    CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                    CONF_ENTITY_ID: entity_id,
                    CONF_TIMEOUT: int(user_input.get(CONF_TIMEOUT, DEFAULT_ROUTER_TIMEOUT)),
                    CONF_ROUTER_LOG_LEVEL: user_input.get(
                        CONF_ROUTER_LOG_LEVEL, DEFAULT_ROUTER_LOG_LEVEL
                    ),
                    CONF_ROUTER_CUSTOM_PROMPT: custom_prompt,
                    CONF_ROUTER_FALLBACK: user_input.get(
                        CONF_ROUTER_FALLBACK, DEFAULT_ROUTER_FALLBACK
                    ),
                    # Routing agents never use the response cache or guard rails
                    CONF_AGENT_CACHE_ENABLED: False,
                    CONF_GUARD_RAIL_ENABLED_FOR_AGENT: False,
                }
                if is_editing:
                    agents = [updated if a.get("id") == agent_id else a for a in agents]
                else:
                    agents.append(updated)
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={**self.config_entry.data, CONF_AGENTS: agents},
                )
                self._agent_data.pop("_editing_agent", None)
                return await self.async_step_init()

        # ── Build default values for the form ─────────────────────────────────
        default_name = (existing or {}).get(CONF_AGENT_NAME, "Routing Agent")
        default_entity = (existing or {}).get(CONF_ENTITY_ID, "")
        default_enabled = (existing or {}).get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED)
        default_timeout = (existing or {}).get(CONF_TIMEOUT, DEFAULT_ROUTER_TIMEOUT)
        default_log_level = (existing or {}).get(CONF_ROUTER_LOG_LEVEL, DEFAULT_ROUTER_LOG_LEVEL)
        default_custom_prompt = (existing or {}).get(
            CONF_ROUTER_CUSTOM_PROMPT, DEFAULT_ROUTER_CUSTOM_PROMPT
        )
        default_fallback = (existing or {}).get(CONF_ROUTER_FALLBACK, DEFAULT_ROUTER_FALLBACK)

        log_level_options: list[selector.SelectOptionDict] = [
            {
                "value": ROUTER_LOG_LEVEL_NONE,
                "label": self._s("router_log_levels", "none") or "None",
            },
            {
                "value": ROUTER_LOG_LEVEL_COMPLEXITY,
                "label": (self._s("router_log_levels", "complexity_only") or "Complexity only"),
            },
            {
                "value": ROUTER_LOG_LEVEL_DEBUG,
                "label": (self._s("router_log_levels", "debug_info") or "Debug information"),
            },
            {
                "value": ROUTER_LOG_LEVEL_DEBUG_QUERY,
                "label": (
                    self._s("router_log_levels", "debug_with_query")
                    or "Debug with query (logs PII)"
                ),
            },
        ]
        fallback_options: list[selector.SelectOptionDict] = [
            {
                "value": ROUTER_FALLBACK_DEFAULT_COMPLEXITY,
                "label": (
                    self._s("router_fallbacks", "default_complexity") or "Use default complexity"
                ),
            },
            {
                "value": ROUTER_FALLBACK_SKIP_ROUTING,
                "label": (
                    self._s("router_fallbacks", "skip_routing") or "Skip routing (try all agents)"
                ),
            },
            {
                "value": ROUTER_FALLBACK_BLOCK,
                "label": self._s("router_fallbacks", "block") or "Block the request",
            },
        ]

        schema_fields: dict[Any, Any] = {
            vol.Required(CONF_AGENT_NAME, default=default_name): str,
        }
        if is_editing:
            schema_fields[vol.Optional(CONF_AGENT_ENABLED, default=default_enabled)] = (
                selector.BooleanSelector()
            )
        entity_field: Any = (
            vol.Required(CONF_ENTITY_ID, default=default_entity)
            if default_entity
            else vol.Required(CONF_ENTITY_ID)
        )
        schema_fields[entity_field] = selector.EntitySelector(
            selector.EntitySelectorConfig(domain=CONVERSATION_DOMAIN)
        )
        schema_fields.update(
            {
                vol.Optional(CONF_TIMEOUT, default=default_timeout): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=2,
                        max=30,
                        unit_of_measurement="seconds",
                    )
                ),
                vol.Optional(
                    CONF_ROUTER_LOG_LEVEL, default=default_log_level
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=log_level_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_ROUTER_CUSTOM_PROMPT, default=default_custom_prompt
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        multiline=True,
                        type=selector.TextSelectorType.TEXT,
                    )
                ),
                vol.Optional(
                    CONF_ROUTER_FALLBACK, default=default_fallback
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=fallback_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )
        if is_editing:
            schema_fields[vol.Optional("delete_agent", default=False)] = selector.BooleanSelector()

        return self.async_show_form(
            step_id="configure_routing_agent",
            data_schema=vol.Schema(schema_fields),
            errors=errors,
            description_placeholders={
                "routing_agent_info": self._s("placeholders", "routing_agent_info"),
            },
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
            if user_input[CONF_AGENT_TYPE] == AGENT_TYPE_WEB_SEARCH:
                return await self.async_step_configure_web_search()
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
                                {
                                    "value": AGENT_TYPE_WEB_SEARCH,
                                    "label": self._s("agent_types", "web_search"),
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
                return await self.async_step_init()

        return self.async_show_form(
            step_id="configure_ollama",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_NAME): str,
                    vol.Required(CONF_PRIORITY, default=DEFAULT_PRIORITY): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN_PROCESSING,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Required(CONF_OLLAMA_URL, default=DEFAULT_OLLAMA_URL): str,
                    vol.Required(CONF_OLLAMA_MODEL): str,
                    vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=2,
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
            return await self.async_step_init()

        return self.async_show_form(
            step_id="configure_existing",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_NAME): str,
                    vol.Required(CONF_PRIORITY, default=DEFAULT_PRIORITY): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN_PROCESSING,
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
                            min=2,
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
                CONF_AGENT_ASSIST_MODE: user_input.get(CONF_AGENT_ASSIST_MODE, True),
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
            return await self.async_step_init()

        default_name = self._s("agent_management", "default_ha_agent_name")
        return self.async_show_form(
            step_id="configure_local",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_NAME, default=default_name): str,
                    vol.Required(CONF_PRIORITY, default=DEFAULT_PRIORITY): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN_PROCESSING,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=2,
                            max=120,
                            unit_of_measurement="seconds",
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_CACHE_ENABLED, default=DEFAULT_AGENT_CACHE_ENABLED
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_AGENT_ASSIST_MODE,
                        default=True,
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

    # ── Web Search connection validation ───────────────────────────────────────

    async def _validate_web_search_connection(
        self,
        provider: str,
        api_key: str,
        answers_api_key: str = "",
    ) -> str | None:
        """Validate web search API key(s) by running lightweight test queries.

        For ``brave_combined`` both keys are validated independently; the first
        failure encountered is returned.

        Args:
            provider: Provider key (e.g. ``SEARCH_PROVIDER_BRAVE``).
            api_key: Brave Search subscription token (never logged).
            answers_api_key: Brave Answers subscription token (never logged).
                             Required when ``provider`` is ``brave_answers`` or
                             ``brave_combined``.

        Returns:
            An error key string if validation fails, None on success.
        """
        if provider in (SEARCH_PROVIDER_BRAVE, SEARCH_PROVIDER_BRAVE_COMBINED):
            if not api_key:
                return "search_api_key_missing"
            err = await self._validate_brave_api_key(api_key)
            if err:
                return err

        if provider in (SEARCH_PROVIDER_BRAVE_ANSWERS, SEARCH_PROVIDER_BRAVE_COMBINED):
            if not answers_api_key:
                return "search_answers_api_key_missing"
            err = await self._validate_brave_answers_api_key(answers_api_key)
            if err:
                return err

        # All known providers validated above; unknown providers are accepted without live validation.
        return None

    async def _validate_brave_api_key(self, api_key: str) -> str | None:
        """Validate a Brave Search API key with a lightweight test query.

        Args:
            api_key: Brave Search subscription token (never logged).

        Returns:
            An error key string if validation fails, None on success.
        """
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        params: dict[str, Any] = {"q": "test", "count": 1}
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response,
            ):
                if response.status == _HTTP_OK:
                    return None
                if response.status in (401, 403):
                    return "invalid_api_key"
                return "search_api_unreachable"
        except aiohttp.ClientError:
            return "search_api_unreachable"
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected error validating web search: %s", err)
            return "unknown"

    async def _validate_brave_answers_api_key(self, api_key: str) -> str | None:
        """Validate a Brave Answers API key with a lightweight test query.

        Args:
            api_key: Brave Answers subscription token (never logged).

        Returns:
            An error key string if validation fails, None on success.
        """
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        params: dict[str, Any] = {"q": "test"}
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    "https://api.search.brave.com/res/v1/answer",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response,
            ):
                if response.status == _HTTP_OK:
                    return None
                if response.status in (401, 403):
                    return "invalid_answers_api_key"
                return "search_api_unreachable"
        except aiohttp.ClientError:
            return "search_api_unreachable"
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected error validating Brave Answers key: %s", err)
            return "unknown"

    # ── Configure web search agent ─────────────────────────────────────────────

    async def async_step_configure_web_search(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure a new web search agent.

        The user picks a provider, supplies an API key, and sets priority and
        result-count options.  The API key is validated against the provider
        before the agent is saved.

        Args:
            user_input: Form data submitted by the user, or None on first load.

        Returns:
            Form result or redirect to main menu after saving.
        """
        errors: dict[str, str] = {}
        connection_status: str = self._agent_data.pop("_web_search_test_status", "")

        if user_input is not None:
            provider: str = user_input.get(CONF_SEARCH_PROVIDER, SEARCH_PROVIDER_BRAVE)
            api_key: str = user_input.get(CONF_SEARCH_API_KEY, "")
            answers_api_key: str = user_input.get(CONF_SEARCH_ANSWERS_API_KEY, "")
            test_only: bool = bool(user_input.get("test_connection", False))

            error_key = await self._validate_web_search_connection(
                provider, api_key, answers_api_key
            )
            if error_key:
                errors["base"] = error_key

            if not errors and test_only:
                # Test passed — return to the form with a success indicator.
                self._agent_data["_web_search_test_status"] = self._s(
                    "placeholders", "connection_ok"
                )
                return await self.async_step_configure_web_search()

            if not errors:
                agent_config = {
                    "id": str(uuid4()),
                    CONF_AGENT_TYPE: AGENT_TYPE_WEB_SEARCH,
                    CONF_AGENT_ENABLED: DEFAULT_AGENT_ENABLED,
                    CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                    CONF_PRIORITY: user_input[CONF_PRIORITY],
                    CONF_SEARCH_PROVIDER: provider,
                    CONF_SEARCH_API_KEY: api_key,
                    CONF_SEARCH_ANSWERS_API_KEY: answers_api_key,
                    CONF_SEARCH_RESULT_COUNT: int(
                        user_input.get(CONF_SEARCH_RESULT_COUNT, DEFAULT_SEARCH_RESULT_COUNT)
                    ),
                    CONF_SEARCH_MAX_SNIPPET_LEN: int(
                        user_input.get(CONF_SEARCH_MAX_SNIPPET_LEN, DEFAULT_SEARCH_MAX_SNIPPET_LEN)
                    ),
                    "timeout": int(user_input.get(CONF_TIMEOUT, DEFAULT_SEARCH_TIMEOUT)),
                    CONF_AGENT_CACHE_ENABLED: user_input.get(
                        CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
                    ),
                    CONF_GUARD_RAIL_ENABLED_FOR_AGENT: user_input.get(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT, DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT
                    ),
                }
                agents = list(self.config_entry.data.get(CONF_AGENTS, []))
                agents.append(agent_config)
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={**self.config_entry.data, CONF_AGENTS: agents},
                )
                return await self.async_step_init()

        provider_options = self._build_search_provider_options()

        return self.async_show_form(
            step_id="configure_web_search",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_AGENT_NAME, default="Web Search"): str,
                    vol.Required(CONF_PRIORITY, default=DEFAULT_PRIORITY): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN_PROCESSING,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Required(
                        CONF_SEARCH_PROVIDER, default=SEARCH_PROVIDER_BRAVE
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=provider_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(CONF_SEARCH_API_KEY): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.PASSWORD,
                        )
                    ),
                    vol.Optional(CONF_SEARCH_ANSWERS_API_KEY, default=""): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.PASSWORD,
                        )
                    ),
                    vol.Optional(
                        CONF_SEARCH_RESULT_COUNT, default=DEFAULT_SEARCH_RESULT_COUNT
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            max=10,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_SEARCH_MAX_SNIPPET_LEN, default=DEFAULT_SEARCH_MAX_SNIPPET_LEN
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=50,
                            max=500,
                            step=50,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_TIMEOUT, default=DEFAULT_SEARCH_TIMEOUT
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=30,
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
                    vol.Optional("test_connection", default=False): selector.BooleanSelector(),
                }
            ),
            errors=errors,
            description_placeholders={
                "priority_info": self._s("placeholders", "priority_info"),
                "web_search_info": self._s("placeholders", "web_search_info"),
                "connection_status": connection_status,
            },
        )

    def _build_search_provider_options(self) -> list[selector.SelectOptionDict]:
        """Build the provider dropdown option list for web search forms.

        Returns:
            List of SelectOptionDict for the provider selector.
        """
        options: list[selector.SelectOptionDict] = []
        for prov in SEARCH_PROVIDERS:
            label = self._s("search_providers", prov) or prov.capitalize()
            options.append({"value": prov, "label": label})
        return options

    # ── Edit web search agent ─────────────────────────────────────────────────

    async def async_step_edit_agent_web_search(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit settings for an existing web search agent.

        The API key field is a password field left blank on load.  If the user
        submits without entering a value the existing key is retained.  If a
        new value is entered it is validated before saving.

        Args:
            user_input: Form data submitted by the user, or None on first load.

        Returns:
            Form result or redirect to main menu after saving.
        """
        errors: dict[str, str] = {}
        connection_status: str = self._agent_data.pop("_web_search_test_status", "")
        agent = self._agent_data.get("_editing_agent", {})
        agent_id: str = agent.get("id", "")

        if user_input is not None:
            if user_input.get("delete_agent"):
                return await self.async_step_confirm_delete_agent()

            provider: str = user_input.get(CONF_SEARCH_PROVIDER, SEARCH_PROVIDER_BRAVE)
            new_key: str = user_input.get(CONF_SEARCH_API_KEY, "").strip()
            new_answers_key: str = user_input.get(CONF_SEARCH_ANSWERS_API_KEY, "").strip()
            api_key: str = new_key if new_key else agent.get(CONF_SEARCH_API_KEY, "")
            answers_api_key: str = (
                new_answers_key if new_answers_key else agent.get(CONF_SEARCH_ANSWERS_API_KEY, "")
            )
            test_only: bool = bool(user_input.get("test_connection", False))

            # Validate whichever key(s) were changed (or both on a test request).
            keys_to_validate_search = new_key or test_only
            keys_to_validate_answers = new_answers_key or test_only
            if keys_to_validate_search or keys_to_validate_answers:
                error_key = await self._validate_web_search_connection(
                    provider,
                    api_key if keys_to_validate_search else "",
                    answers_api_key if keys_to_validate_answers else "",
                )
                if error_key:
                    errors["base"] = error_key

            if not errors and test_only:
                # Test passed — return to the form with a success indicator.
                self._agent_data["_web_search_test_status"] = self._s(
                    "placeholders", "connection_ok"
                )
                return await self.async_step_edit_agent_web_search()

            if not errors:
                updated = {
                    **agent,
                    CONF_AGENT_ENABLED: user_input.get(
                        CONF_AGENT_ENABLED, agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED)
                    ),
                    CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                    CONF_PRIORITY: user_input[CONF_PRIORITY],
                    CONF_SEARCH_PROVIDER: provider,
                    CONF_SEARCH_API_KEY: api_key,
                    CONF_SEARCH_ANSWERS_API_KEY: answers_api_key,
                    CONF_SEARCH_RESULT_COUNT: int(
                        user_input.get(CONF_SEARCH_RESULT_COUNT, DEFAULT_SEARCH_RESULT_COUNT)
                    ),
                    CONF_SEARCH_MAX_SNIPPET_LEN: int(
                        user_input.get(CONF_SEARCH_MAX_SNIPPET_LEN, DEFAULT_SEARCH_MAX_SNIPPET_LEN)
                    ),
                    "timeout": int(user_input.get(CONF_TIMEOUT, DEFAULT_SEARCH_TIMEOUT)),
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
                return await self.async_step_init()

        provider_options = self._build_search_provider_options()

        return self.async_show_form(
            step_id="edit_agent_web_search",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_AGENT_NAME, default=agent.get(CONF_AGENT_NAME, "Web Search")
                    ): str,
                    vol.Required(
                        CONF_PRIORITY, default=agent.get(CONF_PRIORITY, DEFAULT_PRIORITY)
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=PRIORITY_MIN_PROCESSING,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_ENABLED,
                        default=agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED),
                    ): selector.BooleanSelector(),
                    vol.Required(
                        CONF_SEARCH_PROVIDER,
                        default=agent.get(CONF_SEARCH_PROVIDER, SEARCH_PROVIDER_BRAVE),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=provider_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(CONF_SEARCH_API_KEY, default=""): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.PASSWORD,
                        )
                    ),
                    vol.Optional(CONF_SEARCH_ANSWERS_API_KEY, default=""): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.PASSWORD,
                        )
                    ),
                    vol.Optional(
                        CONF_SEARCH_RESULT_COUNT,
                        default=agent.get(CONF_SEARCH_RESULT_COUNT, DEFAULT_SEARCH_RESULT_COUNT),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            max=10,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_SEARCH_MAX_SNIPPET_LEN,
                        default=agent.get(
                            CONF_SEARCH_MAX_SNIPPET_LEN, DEFAULT_SEARCH_MAX_SNIPPET_LEN
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=50,
                            max=500,
                            step=50,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=agent.get("timeout", DEFAULT_SEARCH_TIMEOUT),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=30,
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
                    vol.Optional("test_connection", default=False): selector.BooleanSelector(),
                    vol.Optional("delete_agent", default=False): selector.BooleanSelector(),
                }
            ),
            errors=errors,
            description_placeholders={
                "priority_info": self._s("placeholders", "priority_info"),
                "web_search_info": self._s("placeholders", "web_search_info"),
                "connection_status": connection_status,
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
            return await self._route_to_edit_step()

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

    async def _route_to_edit_step(self) -> ConfigFlowResult:
        """Dispatch to the type-specific edit step for the selected agent."""
        agents = list(self.config_entry.data.get(CONF_AGENTS, []))
        agent_id: str = self._agent_data.get("_selected_agent_id", "")
        agent = next((a for a in agents if a.get("id") == agent_id), None)

        if agent is None:
            return await self.async_step_init()

        self._agent_data["_editing_agent"] = dict(agent)

        # Routing agents (first-class is_router flag or legacy priority==0) get
        # the unified routing agent configure step
        if agent.get(CONF_IS_ROUTER, False) or agent.get(CONF_PRIORITY) == PRIORITY_ROUTER:
            return await self.async_step_configure_routing_agent()

        agent_type = agent.get(CONF_AGENT_TYPE)

        if agent_type == AGENT_TYPE_OLLAMA:
            return await self.async_step_edit_agent_ollama()
        if agent_type == AGENT_TYPE_EXISTING:
            return await self.async_step_edit_agent_existing()
        if agent_type == AGENT_TYPE_WEB_SEARCH:
            return await self.async_step_edit_agent_web_search()
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
            if user_input.get("delete_agent"):
                return await self.async_step_confirm_delete_agent()

            error_key = await self._validate_ollama_connection(
                user_input[CONF_OLLAMA_URL], user_input[CONF_OLLAMA_MODEL]
            )
            if error_key:
                errors["base"] = error_key

            if not errors:
                updated = {
                    **agent,
                    CONF_AGENT_ENABLED: user_input.get(
                        CONF_AGENT_ENABLED, agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED)
                    ),
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
                return await self.async_step_init()

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
                            min=PRIORITY_MIN_PROCESSING,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_ENABLED,
                        default=agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED),
                    ): selector.BooleanSelector(),
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
                            min=2,
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
                    vol.Optional("delete_agent", default=False): selector.BooleanSelector(),
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
            if user_input.get("delete_agent"):
                return await self.async_step_confirm_delete_agent()

            updated = {
                **agent,
                CONF_AGENT_ENABLED: user_input.get(
                    CONF_AGENT_ENABLED, agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED)
                ),
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
            return await self.async_step_init()

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
                            min=PRIORITY_MIN_PROCESSING,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_ENABLED,
                        default=agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED),
                    ): selector.BooleanSelector(),
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
                            min=2,
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
                    vol.Optional("delete_agent", default=False): selector.BooleanSelector(),
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
            if user_input.get("delete_agent"):
                return await self.async_step_confirm_delete_agent()

            updated = {
                **agent,
                CONF_AGENT_ENABLED: user_input.get(
                    CONF_AGENT_ENABLED, agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED)
                ),
                CONF_AGENT_NAME: user_input[CONF_AGENT_NAME],
                CONF_PRIORITY: user_input[CONF_PRIORITY],
                CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                CONF_AGENT_CACHE_ENABLED: user_input.get(
                    CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED
                ),
                CONF_AGENT_ASSIST_MODE: user_input.get(CONF_AGENT_ASSIST_MODE, True),
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
            return await self.async_step_init()

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
                            min=PRIORITY_MIN_PROCESSING,
                            max=PRIORITY_MAX,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_ENABLED,
                        default=agent.get(CONF_AGENT_ENABLED, DEFAULT_AGENT_ENABLED),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=agent.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=2,
                            max=120,
                            unit_of_measurement="seconds",
                        )
                    ),
                    vol.Optional(
                        CONF_AGENT_CACHE_ENABLED,
                        default=agent.get(CONF_AGENT_CACHE_ENABLED, DEFAULT_AGENT_CACHE_ENABLED),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_AGENT_ASSIST_MODE,
                        default=agent.get(CONF_AGENT_ASSIST_MODE, True),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                        default=agent.get(
                            CONF_GUARD_RAIL_ENABLED_FOR_AGENT,
                            DEFAULT_GUARD_RAIL_ENABLED_FOR_AGENT,
                        ),
                    ): selector.BooleanSelector(),
                    vol.Optional("delete_agent", default=False): selector.BooleanSelector(),
                }
            ),
            description_placeholders={
                "priority_info": self._s("placeholders", "priority_info"),
            },
        )

    # ── Confirm agent deletion ─────────────────────────────────────────────────

    async def async_step_confirm_delete_agent(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask the user to confirm deletion of the selected agent.

        Args:
            user_input: Form data submitted by the user, or None on first load.

        Returns:
            Confirmation form, or redirect to main menu after acting on the choice.
        """
        agent = self._agent_data.get("_editing_agent", {})
        agent_id: str = agent.get("id", "")
        agent_name: str = agent.get(CONF_AGENT_NAME, "Unknown")

        if user_input is not None:
            if user_input.get("confirm"):
                agents = list(self.config_entry.data.get(CONF_AGENTS, []))
                agents = [a for a in agents if a.get("id") != agent_id]
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={**self.config_entry.data, CONF_AGENTS: agents},
                )
            return await self.async_step_init()

        return self.async_show_form(
            step_id="confirm_delete_agent",
            data_schema=vol.Schema(
                {
                    vol.Required("confirm", default=False): selector.BooleanSelector(),
                }
            ),
            description_placeholders={"agent_name": agent_name},
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
            enable_home_control: bool = user_input.get(
                CONF_ENABLE_HOME_CONTROL, DEFAULT_ENABLE_HOME_CONTROL
            )
            force_response_language: bool = user_input.get(
                CONF_FORCE_RESPONSE_LANGUAGE, DEFAULT_FORCE_RESPONSE_LANGUAGE
            )

            current_data = {
                **self.config_entry.data,
                CONF_RESPONSE_CACHE_ENABLED: cache_enabled,
                CONF_RESPONSE_CACHE_TTL: cache_ttl,
                CONF_MAX_RETRIES: max_retries,
                CONF_RETRY_BASE_DELAY: retry_base_delay,
                CONF_ENABLE_HOME_CONTROL: enable_home_control,
                CONF_FORCE_RESPONSE_LANGUAGE: force_response_language,
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

            return await self.async_step_init()

        entry_data = self.config_entry.data
        current_enabled = entry_data.get(
            CONF_RESPONSE_CACHE_ENABLED, DEFAULT_RESPONSE_CACHE_ENABLED
        )
        current_ttl = entry_data.get(CONF_RESPONSE_CACHE_TTL, DEFAULT_RESPONSE_CACHE_TTL)
        current_max_retries = entry_data.get(CONF_MAX_RETRIES, DEFAULT_MAX_RETRIES)
        current_retry_delay = entry_data.get(CONF_RETRY_BASE_DELAY, DEFAULT_RETRY_BASE_DELAY)
        current_home_control = entry_data.get(CONF_ENABLE_HOME_CONTROL, DEFAULT_ENABLE_HOME_CONTROL)
        current_force_lang = entry_data.get(
            CONF_FORCE_RESPONSE_LANGUAGE, DEFAULT_FORCE_RESPONSE_LANGUAGE
        )

        return self.async_show_form(
            step_id="advanced_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_ENABLE_HOME_CONTROL, default=current_home_control
                    ): selector.BooleanSelector(),
                    vol.Required(
                        CONF_FORCE_RESPONSE_LANGUAGE, default=current_force_lang
                    ): selector.BooleanSelector(),
                    vol.Required(
                        CONF_RESPONSE_CACHE_ENABLED, default=current_enabled
                    ): selector.BooleanSelector(),
                    vol.Required(
                        CONF_RESPONSE_CACHE_TTL, default=current_ttl
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
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
        self, _user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show the guard rails sub-menu."""
        return self.async_show_menu(
            step_id="configure_guard_rails",
            menu_options=[
                "guard_rails_settings",
                "configure_guard_rail_rules",
                "back_to_main",
            ],
        )

    async def async_step_guard_rails_settings(
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
                # Routing agents are identified by is_router flag; fall back to
                # legacy priority==0 for configs created before v0.3.0
                router_agents = [
                    a
                    for a in current_data.get(CONF_AGENTS, [])
                    if a.get(CONF_IS_ROUTER, DEFAULT_IS_ROUTER) or a.get(CONF_PRIORITY) == 0
                ]

                if router_agents:
                    current_data["guard_rail_agent_id"] = router_agents[0]["id"]
                else:
                    return self.async_show_form(
                        step_id="guard_rails_settings",
                        errors={"base": "no_router_agent"},
                        description_placeholders={
                            "info": self._s("placeholders", "no_router_agent_error"),
                        },
                    )

            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data=current_data,
            )
            return await self.async_step_init()

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
            step_id="guard_rails_settings",
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

    async def async_step_back_to_main(
        self, _user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Return to the main options menu from the guard rails sub-menu."""
        return await self.async_step_init()

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
            return await self.async_step_init()

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
            return await self.async_step_init()

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

    # ── Done ──────────────────────────────────────────────────────────────────

    async def async_step_done(self, _user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Close the options flow."""
        return self.async_create_entry(title="", data={})
