"""LLM agent proxy — unified execution layer for INTEGRATED and OLLAMA backends.

Both :data:`AGENT_TYPE_INTEGRATED` (Home Assistant conversation entity) and
:data:`AGENT_TYPE_OLLAMA` (direct HTTP to an Ollama server) share the same
prompt-enrichment pipeline:

* Verbosity hints (brief / verbose)
* Adaptive preference injection (Feature 15)
* Sensor context injection
* Area context prefix (Ollama only — HA handles it natively for INTEGRATED)
* Language passthrough
* Cannot-answer sentinel (Ollama only)

Only the *transport* differs:

* **INTEGRATED** — calls the HA ``conversation.process`` service and isolates the
  outer ChatLog to prevent duplicate history entries.
* **OLLAMA** — calls :class:`OllamaClient` directly with a structured messages list
  and manages session memory explicitly.

Usage inside :class:`NeuralBridgeAgent`::

    self._llm_proxy = LLMAgentProxy(
        hass=hass,
        config_getter=self._get_config,
        entity_context_cache=self._entity_context_cache,
        session_memory=self._session_memory,
        preference_memory=self._preference_memory,
        ollama_clients=self._ollama_clients,
        benchmarker_getter=self._get_benchmarker,
    )

    result = await self._llm_proxy.process(agent_config, user_input, router_decision)
    targets = self._llm_proxy.consume_local_ha_targets(conv_id)  # for high-stakes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from homeassistant.components.conversation import ConversationInput, ConversationResult
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.helpers import intent

from .const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    CONF_AGENT_ASSIST_MODE,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_ENTITY_ID,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_RESPONSE_VERBOSITY,
    CONF_TIMEOUT,
    DEFAULT_AGENT_ASSIST_MODE,
    DEFAULT_RESPONSE_VERBOSITY,
    DEFAULT_TIMEOUT,
    VERBOSITY_BRIEF,
)
from .prompt_builder import PromptBuilder, truncate_to_first_sentence
from .prompt_builder import get_language_name as get_language_name  # noqa: PLC0414

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.core import HomeAssistant

    from .agent_benchmark import AgentBenchmarker
    from .entity_context import EntityContextCache
    from .ollama_client import OllamaClient
    from .preference_memory import PreferenceMemory
    from .session_memory import SessionMemory

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMAgentProxyProtocol(Protocol):
    """Interface contract for LLM agent proxy objects.

    Concrete implementations route prompt-enriched user input to the appropriate
    backend (INTEGRATED HA conversation entity or direct Ollama HTTP).
    """

    async def process(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: Any = None,
    ) -> ConversationResult | None:
        """Enrich the query and dispatch to the appropriate backend."""
        ...

    def consume_local_ha_targets(self, conv_id: str) -> list[str]:
        """Pop and return entity IDs captured from the last INTEGRATED call."""
        ...

    def get_device_area(self, device_id: str | None) -> str | None:
        """Return the friendly area name for a device, or None if unavailable."""
        ...

    def build_pref_hint(
        self,
        query_text: str,
        for_router: bool = False,
    ) -> str | None:
        """Build a compact preference hint to prepend to agent / router input."""
        ...


# ---------------------------------------------------------------------------
# Public proxy class
# ---------------------------------------------------------------------------


class LLMAgentProxy:
    """Unified execution proxy for INTEGRATED (HA conversation entity) and OLLAMA backends.

    Centralises all prompt-enrichment logic so that new capabilities (verbosity,
    preference injection, sensor context, etc.) only need to be added in one place
    regardless of which transport the agent uses.

    The two backends differ only in how the enriched content is delivered:

    * **INTEGRATED** — enriched text is sent as the ``text`` field in a
      ``conversation.process`` service call.  The HA entity sees it as the user's
      message and replies through its own LLM pipeline.  ChatLog isolation is
      applied to prevent duplicate history writes (see :meth:`_process_integrated`).

    * **OLLAMA** — verbosity and language go into the **system prompt**; user text,
      area context, and sensor context go into the **user message**.  Session memory
      is managed explicitly since the Ollama HTTP API is stateless.

    Args:
        hass: Home Assistant instance.
        config_getter: Zero-argument callable returning the merged entry config dict.
        entity_context_cache: Shared entity-context/sensor-value cache.
        session_memory: Shared conversation session memory.
        preference_memory: Adaptive preference store, or None if APL is disabled.
        ollama_clients: Mutable dict of cached :class:`OllamaClient` instances,
            keyed by agent ID.  Shared with :class:`NeuralBridgeAgent` so the same
            client is reused for routing and processing calls to the same Ollama URL.
        benchmarker_getter: Zero-argument callable returning the
            :class:`AgentBenchmarker` (or None before setup completes).  Used to
            record real-world token-per-second telemetry from Ollama responses.
    """

    def __init__(
        self,
        hass: "HomeAssistant",
        config_getter: "Callable[[], dict[str, Any]]",
        entity_context_cache: "EntityContextCache",
        session_memory: "SessionMemory",
        preference_memory: "PreferenceMemory | None",
        ollama_clients: "dict[str, OllamaClient]",
        benchmarker_getter: "Callable[[], AgentBenchmarker | None]",
    ) -> None:
        """Initialise the proxy with its dependencies."""
        self._hass = hass
        self._get_config = config_getter
        self._entity_context_cache = entity_context_cache
        self._session_memory = session_memory
        self._preference_memory = preference_memory
        self._ollama_clients = ollama_clients
        self._benchmarker_getter = benchmarker_getter
        # Keyed by conv_id; written by _process_integrated for high-stakes checks.
        self._local_ha_targets: dict[str, list[str]] = {}
        # Prompt-construction delegated to PromptBuilder (R4 split).
        self._prompt_builder = PromptBuilder(
            hass, config_getter, entity_context_cache, preference_memory
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Keep _prompt_builder in sync when shared dependencies are reassigned.

        When test code (or live code) reassigns ``_hass``, ``_entity_context_cache``,
        or ``_preference_memory`` on this proxy, the same attribute on the inner
        :class:`PromptBuilder` is updated automatically so they always share the
        same dependency objects.
        """
        super().__setattr__(name, value)
        if name in ("_hass", "_entity_context_cache", "_preference_memory"):
            pb: Any = self.__dict__.get("_prompt_builder")
            if pb is not None:
                object.__setattr__(pb, name, value)

    # ── Public API ─────────────────────────────────────────────────────────

    async def process(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: Any = None,
    ) -> ConversationResult | None:
        """Enrich the query and dispatch to the appropriate backend.

        Supported agent types:

        * :data:`AGENT_TYPE_OLLAMA` — direct Ollama HTTP backend.
        * :data:`AGENT_TYPE_INTEGRATED` / :data:`AGENT_TYPE_LOCAL_HA` — HA
          conversation-service backend.

        Args:
            agent_config: The agent's configuration dictionary.
            user_input: The user's conversation input.
            router_decision: Optional :class:`RouterDecision` carrying
                ``relevant_sensors`` for targeted sensor injection.  Typed
                as ``Any`` to avoid a circular import with ``conversation.py``.

        Returns:
            :class:`ConversationResult` on success, ``None`` on any failure
            (missing config, network error, non-action_done in assist mode, etc.).
        """
        agent_type: str = agent_config.get(CONF_AGENT_TYPE, "")
        if agent_type == AGENT_TYPE_OLLAMA:
            return await self._process_ollama(agent_config, user_input, router_decision)
        if agent_type in (AGENT_TYPE_INTEGRATED, AGENT_TYPE_LOCAL_HA):
            return await self._process_integrated(agent_config, user_input, router_decision)
        _LOGGER.warning(
            "LLMAgentProxy.process: unknown agent type '%s' for agent '%s'",
            agent_type,
            agent_config.get(CONF_AGENT_NAME, "?"),
        )
        return None

    def consume_local_ha_targets(self, conv_id: str) -> list[str]:
        """Pop and return entity IDs captured from the last INTEGRATED call.

        Called by :meth:`NeuralBridgeAgent._check_high_stakes` so it can
        determine whether an INTEGRATED response targets a high-stakes domain.

        Args:
            conv_id: The conversation session ID.

        Returns:
            List of entity IDs targeted by the last INTEGRATED call, or an
            empty list when nothing was captured or the key has already been
            consumed.
        """
        return self._local_ha_targets.pop(conv_id, [])

    def get_device_area(self, device_id: str | None) -> str | None:
        """Return the friendly area name for a device, or None if unavailable.

        Delegates to :class:`PromptBuilder`.

        Args:
            device_id: The HA device ID from :attr:`ConversationInput.device_id`.

        Returns:
            The area name string, or None when the device has no area or is unknown.
        """
        return self._prompt_builder.get_device_area(device_id)

    def build_pref_hint(
        self,
        query_text: str,
        for_router: bool = False,
    ) -> str | None:
        """Build a compact preference hint to prepend to agent / router input.

        Delegates to :class:`PromptBuilder`.

        Args:
            query_text: The user query (reserved for future filtering).
            for_router: When ``True``, exclude format-only preferences.

        Returns:
            A bracket-delimited hint string, or ``None`` when no confirmed
            preferences exist.
        """
        return self._prompt_builder.build_pref_hint(query_text, for_router)

    def _render_ha_context(self, prompt: str) -> str:
        """Substitute ``{ha_*}`` tokens with live HA config values.

        Delegates to :class:`PromptBuilder`.

        Args:
            prompt: Raw prompt text.

        Returns:
            Rendered prompt string.
        """
        return self._prompt_builder.render_ha_context(prompt)

    def _create_result(
        self, response_text: str, conversation_id: str | None = None
    ) -> ConversationResult:
        """Create a successful :class:`ConversationResult`.

        Args:
            response_text: The speech text to return to the user.
            conversation_id: The conversation session ID to echo to the caller.

        Returns:
            A :class:`ConversationResult` with the given speech text.
        """
        response = intent.IntentResponse(language=self._hass.config.language)
        response.async_set_speech(response_text)
        return ConversationResult(response=response, conversation_id=conversation_id)

    # ── Ollama backend ─────────────────────────────────────────────────────

    async def _process_ollama(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: Any = None,
    ) -> ConversationResult | None:
        """Process input with a direct Ollama HTTP backend.

        Builds the full message list (system prompt + history + enriched user
        message) and calls the :class:`OllamaClient`.  Session memory is updated
        on success and real-world token telemetry is forwarded to the benchmarker.

        Args:
            agent_config: Configuration dict for the Ollama agent.
            user_input: The user's conversation input.
            router_decision: Optional routing decision for targeted sensor injection.

        Returns:
            :class:`ConversationResult` on success, ``None`` on any failure.
        """
        from .ollama_client import OllamaClient  # noqa: PLC0415

        agent_id: str = agent_config.get("id", "")
        ollama_url: str = agent_config.get(CONF_OLLAMA_URL, "") or ""
        ollama_model: str = agent_config.get(CONF_OLLAMA_MODEL, "") or ""
        timeout: int = agent_config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)

        if not ollama_url or not ollama_model:
            _LOGGER.error(
                "Ollama agent '%s' is missing URL or model",
                agent_config.get(CONF_AGENT_NAME, "?"),
            )
            return None

        system_prompt, has_full_sensor_injection = self._prompt_builder.build_system_prompt(
            agent_config, user_input
        )

        if agent_id not in self._ollama_clients:
            self._ollama_clients[agent_id] = OllamaClient(ollama_url, ollama_model, timeout)

        client: OllamaClient = self._ollama_clients[agent_id]

        # Session history + enriched user message
        history = self._session_memory.get_messages(user_input.conversation_id)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)

        # Enrich user text; skip sensor injection when system prompt already has
        # {ha_sensor_states} to avoid sending the same data twice.
        if has_full_sensor_injection:
            # Only add area and pref hints — sensor block already in system prompt
            area_ctx = self.get_device_area(user_input.device_id)
            base_text = f"[Area: {area_ctx}] {user_input.text}" if area_ctx else user_input.text
            pref_hint = self.build_pref_hint(base_text, for_router=False)
            user_text = f"{pref_hint}\n\n{base_text}" if pref_hint else base_text
        else:
            user_text = self._prompt_builder.build_user_text(
                agent_config,
                user_input,
                router_decision,
                include_area=True,
                verbosity_as_prefix=False,
            )

        messages.append({"role": "user", "content": user_text})

        ollama_resp = await client.chat(messages)
        if ollama_resp is None:
            return None

        response_text = ollama_resp.content
        # Passive telemetry — update rolling tps average in benchmark profile
        benchmarker = self._benchmarker_getter()
        if benchmarker is not None:
            benchmarker.update_realworld_telemetry(agent_id, ollama_resp)

        self._session_memory.add_turn(user_input.conversation_id, user_input.text, response_text)
        return self._create_result(response_text, user_input.conversation_id)

    # ── INTEGRATED / LOCAL_HA backend ─────────────────────────────────────

    async def _process_integrated(
        self,
        agent_config: dict[str, Any],
        user_input: ConversationInput,
        router_decision: Any = None,
    ) -> ConversationResult | None:
        """Process input with an existing Home Assistant conversation agent.

        The sub-agent is intentionally isolated from the outer ``current_chat_log``
        context variable.  In HA 2025.2+ the built-in HA agent (and some third-party
        agents) call ``async_add_assistant_content_without_tools`` internally, which
        would write a duplicate entry into the outer ChatLog that NeuralBridge
        manages.  Setting ``current_chat_log`` to ``None`` for the duration of the
        sub-call prevents that.

        Entity IDs targeted by the sub-agent are captured in
        :attr:`_local_ha_targets` (keyed by ``conv_id``) for subsequent
        high-stakes domain checking by :class:`NeuralBridgeAgent`.

        Args:
            agent_config: Configuration dict for the HA conversation agent.
            user_input: The user's conversation input.
            router_decision: Optional routing decision for targeted sensor injection.

        Returns:
            :class:`ConversationResult` on success, ``None`` on any failure or
            when assist-mode filtering rejects the response type.
        """
        entity_id: str | None = agent_config.get(CONF_ENTITY_ID)
        if entity_id is None or not self._hass.states.get(entity_id):
            _LOGGER.error(
                "Integrated agent '%s' has no entity_id configured or entity not found",
                agent_config.get(CONF_AGENT_NAME, "?"),
            )
            return None

        # Isolate the sub-agent from the outer ChatLog (see docstring).
        from homeassistant.components.conversation.chat_log import (  # noqa: PLC0415
            current_chat_log,
        )

        _chat_log_var = current_chat_log
        _chat_log_token = current_chat_log.set(None)

        response: Any = None
        agent_type: str = agent_config.get(CONF_AGENT_TYPE, "")
        config = self._get_config()
        verbosity_cfg: str = config.get(CONF_RESPONSE_VERBOSITY, DEFAULT_RESPONSE_VERBOSITY)

        query_text = self._prompt_builder.build_user_text(
            agent_config,
            user_input,
            router_decision,
            include_area=False,
            verbosity_as_prefix=True,
        )

        try:
            response = await self._hass.services.async_call(
                CONVERSATION_DOMAIN,
                "process",
                {
                    "text": query_text,
                    "agent_id": entity_id,
                    "conversation_id": user_input.conversation_id,
                    "language": user_input.language,
                },
                blocking=True,
                context=user_input.context,
                return_response=True,
            )
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error(
                "Error calling conversation agent '%s': %s",
                entity_id,
                err,
            )
        finally:
            _chat_log_var.reset(_chat_log_token)

        if response is None:
            return None
        if not response or "response" not in response:
            _LOGGER.error("Agent '%s' returned unexpected response shape", entity_id)
            return None

        response_section: Any = response["response"]

        # Assist-mode filter: LOCAL_HA agents with assist_mode=True only accept
        # action_done responses — any other type falls through to the next agent.
        assist_mode: bool = agent_config.get(CONF_AGENT_ASSIST_MODE, DEFAULT_AGENT_ASSIST_MODE)
        if assist_mode:
            response_type = str(response_section.get("response_type", ""))
            if response_type != "action_done":
                _LOGGER.debug(
                    "Agent '%s' assist mode: response_type '%s' is not action_done,"
                    " falling through to next agent",
                    entity_id,
                    response_type,
                )
                return None

        speech_text = str(response_section.get("speech", {}).get("plain", {}).get("speech", ""))
        if not speech_text:
            return None

        # Trim to first sentence for LOCAL_HA agents in brief mode.
        if agent_type == AGENT_TYPE_LOCAL_HA and verbosity_cfg == VERBOSITY_BRIEF:
            speech_text = truncate_to_first_sentence(speech_text)

        # Capture targeted entity IDs for high-stakes domain checks.
        conv_key = user_input.conversation_id or "default"
        targets_raw = response_section.get("data", {}).get("targets", [])
        self._local_ha_targets[conv_key] = [
            t["id"] for t in targets_raw if isinstance(t, dict) and "id" in t
        ]

        return self._create_result(speech_text, user_input.conversation_id)
