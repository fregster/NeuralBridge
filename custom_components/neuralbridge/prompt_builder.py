"""Prompt construction logic for NeuralBridge LLM agents.

This module was split from :mod:`llm_agent_proxy` in R4 of the code-restructure
plan.  It encapsulates all *pure prompt-construction* logic — string manipulation,
template rendering, preference injection, sensor context embedding — and is free
of any I/O.  Only the transport layer in :mod:`llm_agent_proxy` makes network
calls.

Classes:

* :class:`PromptBuilder` — stateful helper that accepts a fixed set of HA
  dependencies once and exposes prompt-construction methods that can be tested in
  isolation without a full :class:`LLMAgentProxy` instance.

Module-level helpers:

* :func:`get_language_name` — BCP-47 → human-readable name.
* :func:`truncate_to_first_sentence` — first-sentence extraction for brief mode.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr

from .const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    CANNOT_ANSWER_INSTRUCTION,
    CONF_AGENT_ASSIST_MODE,
    CONF_AGENT_TYPE,
    CONF_DEFAULT_PROMPT,
    CONF_FORCE_RESPONSE_LANGUAGE,
    CONF_LANGUAGE,
    CONF_RESPONSE_VERBOSITY,
    CONF_SYSTEM_PROMPT,
    DEFAULT_AGENT_ASSIST_MODE,
    DEFAULT_DEFAULT_PROMPT,
    DEFAULT_FORCE_RESPONSE_LANGUAGE,
    DEFAULT_LANGUAGE,
    DEFAULT_RESPONSE_VERBOSITY,
    DEFAULT_SYSTEM_PROMPT,
    VERBOSITY_BRIEF,
    VERBOSITY_INSTRUCTION_BRIEF,
    VERBOSITY_INSTRUCTION_VERBOSE,
    VERBOSITY_VERBOSE,
)
from .preference_memory import PREF_CATEGORY_FORMAT

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import HomeAssistant

    from .entity_context import EntityContextCache
    from .preference_memory import PreferenceMemory

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language helper
# ---------------------------------------------------------------------------

_LANG_CODE_TO_NAME: dict[str, str] = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "zh": "Chinese",
}

_SENTENCE_END_RE: re.Pattern[str] = re.compile(r"([.?!])\s+")


def get_language_name(language_code: str) -> str:
    """Return a human-readable language name for a BCP-47 language code.

    Normalises the code to a 2-letter base (e.g. ``"fr-FR"`` → ``"fr"``) before
    looking up in ``_LANG_CODE_TO_NAME``.  Falls back to the original code when
    the language is not in the map.

    Args:
        language_code: BCP-47 language tag (e.g. ``"en"``, ``"fr-FR"``).

    Returns:
        Human-readable name (e.g. ``"French"``) or the original code if unknown.
    """
    base = language_code.lower().split("-")[0].split("_")[0]
    return _LANG_CODE_TO_NAME.get(base, language_code)


def truncate_to_first_sentence(text: str) -> str:
    """Return the first sentence of *text*, stripping trailing whitespace.

    Splits on the first ``'. '``, ``'? '``, or ``'! '`` boundary.  If no
    sentence boundary is found the original *text* is returned unchanged.

    Args:
        text: Input text that may contain one or more sentences.

    Returns:
        The first sentence (including its terminal punctuation), stripped.
    """
    match = _SENTENCE_END_RE.search(text)
    if match:
        return text[: match.start(1) + 1].rstrip()
    return text


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Pure prompt-construction helper for NeuralBridge LLM agents.

    Accepts a fixed set of HA dependencies at construction time and exposes
    methods for building system prompts, enriched user text, HA-context rendering,
    and preference hints.  No I/O is performed — all methods are synchronous
    string-manipulation functions.

    Args:
        hass: Home Assistant instance.
        config_getter: Zero-argument callable returning the merged entry config dict.
        entity_context_cache: Shared entity-context/sensor-value cache.
        preference_memory: Adaptive preference store, or ``None`` when APL is
            disabled.
    """

    def __init__(
        self,
        hass: "HomeAssistant",
        config_getter: "Callable[[], dict[str, Any]]",
        entity_context_cache: "EntityContextCache",
        preference_memory: "PreferenceMemory | None",
    ) -> None:
        """Initialise with HA dependencies."""
        self._hass = hass
        self._get_config = config_getter
        self._entity_context_cache = entity_context_cache
        self._preference_memory = preference_memory

    # ── Public API ─────────────────────────────────────────────────────────

    def get_device_area(self, device_id: str | None) -> str | None:
        """Return the friendly area name for a device, or ``None`` if unavailable.

        Looks up the device in the HA device registry, then resolves its area
        via the area registry.

        Args:
            device_id: The HA device ID from :attr:`ConversationInput.device_id`.

        Returns:
            The area name string, or ``None`` when the device has no area or is
            unknown.
        """
        if not device_id:
            return None
        dev_registry = dr.async_get(self._hass)
        device = dev_registry.async_get(device_id)
        if device is None or not device.area_id:
            return None
        area_reg = ar.async_get(self._hass)
        area = area_reg.async_get_area(device.area_id)
        if area is None:
            return None
        return str(area.name)

    def build_pref_hint(
        self,
        query_text: str,
        for_router: bool = False,
    ) -> str | None:
        """Build a compact preference hint to prepend to agent / router input.

        When ``for_router`` is ``True`` only routing-relevant categories (source
        and location) are included so the router's compact classification model is
        not burdened with format preferences.  For processing agents all confirmed
        preferences are included.

        Args:
            query_text: The user query (reserved for future per-query filtering,
                not used in the current implementation).
            for_router: When ``True``, exclude format-only preferences.

        Returns:
            A bracket-delimited hint string, or ``None`` when no confirmed
            preferences exist or :attr:`_preference_memory` is ``None``.
        """
        del query_text  # reserved
        if self._preference_memory is None:
            return None

        confirmed = self._preference_memory.all_confirmed()
        if not confirmed:
            return None

        if for_router:
            confirmed = [e for e in confirmed if e.category != PREF_CATEGORY_FORMAT]
        if not confirmed:
            return None

        lines = [f"- {e.key}: {e.value}" for e in confirmed]
        header = "User context hints:" if for_router else "User preferences (apply these):"
        return "[" + header + "\n" + "\n".join(lines) + "]"

    def render_ha_context(self, prompt: str) -> str:
        """Substitute ``{ha_*}`` template tokens with live HA configuration values.

        Supported tokens:

        * ``{ha_location_name}`` — friendly name of the HA installation.
        * ``{ha_timezone}`` — IANA timezone string, e.g. ``"Europe/London"``.
        * ``{ha_unit_temperature}`` — temperature unit symbol, e.g. ``"°C"``.
        * ``{ha_sensor_states}`` — opt-in, Ollama direct only — a multi-line
          block of current sensor state values.

        Args:
            prompt: Raw prompt text, possibly containing ``{ha_*}`` tokens.

        Returns:
            Prompt with all recognised tokens replaced by their runtime values.
        """
        cfg = self._hass.config
        unit_temp = getattr(cfg.units, "temperature_unit", None)
        temperature_unit: str = (
            unit_temp.value if unit_temp is not None and hasattr(unit_temp, "value") else ""
        )
        result = (
            prompt.replace("{ha_location_name}", cfg.location_name or "")
            .replace("{ha_timezone}", cfg.time_zone or "")
            .replace("{ha_unit_temperature}", temperature_unit)
        )
        if "{ha_sensor_states}" in result:
            sensor_block = self._entity_context_cache.get_sensor_values(self._hass)
            result = result.replace("{ha_sensor_states}", sensor_block)
        return result

    def build_system_prompt(
        self,
        agent_config: dict[str, Any],
        user_input: "ConversationInput",
    ) -> tuple[str, bool]:
        """Build the rendered system prompt for the Ollama backend.

        Applies ``{ha_*}`` token substitution, language passthrough, verbosity
        instruction, and the cannot-answer sentinel.

        Args:
            agent_config: The Ollama agent's configuration dict.
            user_input: The user's conversation input (language used for passthrough).

        Returns:
            ``(system_prompt, has_full_sensor_injection)`` tuple where the boolean
            flag indicates whether ``{ha_sensor_states}`` was embedded in the raw
            template (to prevent a second targeted injection in the caller).
        """
        default_prompt = self._get_config().get(CONF_DEFAULT_PROMPT, DEFAULT_DEFAULT_PROMPT)
        raw_prompt: str = agent_config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT) or ""
        if not raw_prompt:
            raw_prompt = default_prompt or ""

        has_full_sensor_injection = "{ha_sensor_states}" in raw_prompt
        system_prompt = self.render_ha_context(raw_prompt)

        config = self._get_config()
        # Language passthrough
        force_lang: bool = config.get(CONF_FORCE_RESPONSE_LANGUAGE, DEFAULT_FORCE_RESPONSE_LANGUAGE)
        if force_lang and user_input.language:
            input_base = user_input.language.lower().split("-")[0].split("_")[0]
            conf_base = (
                str(config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)).lower().split("-")[0].split("_")[0]
            )
            if input_base != conf_base:
                lang_name = get_language_name(user_input.language)
                system_prompt = f"{system_prompt}\nRespond in {lang_name}.".strip()

        # Verbosity instruction appended to system prompt for Ollama
        verbosity: str = config.get(CONF_RESPONSE_VERBOSITY, DEFAULT_RESPONSE_VERBOSITY)
        if verbosity == VERBOSITY_BRIEF:
            system_prompt = f"{system_prompt}\n{VERBOSITY_INSTRUCTION_BRIEF}".strip()
        elif verbosity == VERBOSITY_VERBOSE:
            system_prompt = f"{system_prompt}\n{VERBOSITY_INSTRUCTION_VERBOSE}".strip()

        # Cannot-answer sentinel
        system_prompt = f"{system_prompt}\n{CANNOT_ANSWER_INSTRUCTION}".strip()
        return system_prompt, has_full_sensor_injection

    def build_user_text(
        self,
        agent_config: dict[str, Any],
        user_input: "ConversationInput",
        router_decision: Any = None,
        *,
        include_area: bool = False,
        verbosity_as_prefix: bool = False,
    ) -> str:
        """Build the enriched user-facing text block.

        Applies (in order):

        1. Verbosity prefix — ``[Brief response]`` / ``[Verbose response]`` — when
           ``verbosity_as_prefix`` is ``True`` (INTEGRATED agents).
        2. Preference hint block (all confirmed preferences, all categories).
        3. Area context prefix — ``[Area: Kitchen]`` — when ``include_area`` is
           ``True`` (Ollama agents only).
        4. Sensor injection — router-guided (specific sensors) or full fallback.

        Args:
            agent_config: The agent's configuration dict (used for agent_type).
            user_input: The user's conversation input.
            router_decision: Optional router decision for targeted sensor injection.
            include_area: When ``True`` prepend the device area to the user text.
            verbosity_as_prefix: When ``True`` prepend a brief/verbose hint to the
                user text (used for INTEGRATED agents).

        Returns:
            The fully enriched user text string.
        """
        config = self._get_config()
        agent_type: str = agent_config.get(CONF_AGENT_TYPE, "")

        # 1. Verbosity prefix (INTEGRATED path)
        verbosity_cfg: str = config.get(CONF_RESPONSE_VERBOSITY, DEFAULT_RESPONSE_VERBOSITY)
        if verbosity_as_prefix and agent_type == AGENT_TYPE_INTEGRATED:
            if verbosity_cfg == VERBOSITY_BRIEF:
                text = f"[Brief response] {user_input.text}"
            elif verbosity_cfg == VERBOSITY_VERBOSE:
                text = f"[Verbose response] {user_input.text}"
            else:
                text = user_input.text
        else:
            text = user_input.text

        # 2. Preference hint block
        pref_hint = self.build_pref_hint(text, for_router=False)
        if pref_hint:
            text = f"{pref_hint}\n\n{text}"

        # 3. Area context prefix (Ollama path)
        if include_area:
            area_ctx = self.get_device_area(user_input.device_id)
            if area_ctx:
                text = f"[Area: {area_ctx}] {text}"

        # 4. Sensor injection
        assist_mode = agent_config.get(CONF_AGENT_ASSIST_MODE, DEFAULT_AGENT_ASSIST_MODE)
        should_inject = (agent_type == AGENT_TYPE_INTEGRATED) or (
            agent_type == AGENT_TYPE_LOCAL_HA and not assist_mode
        )
        if agent_type == AGENT_TYPE_OLLAMA:
            should_inject = True

        if should_inject:
            relevant = (
                getattr(router_decision, "relevant_sensors", None) if router_decision else None
            )
            if relevant:
                sensor_block = self._entity_context_cache.get_sensor_values_for(
                    self._hass, relevant
                )
            else:
                sensor_block = self._entity_context_cache.get_sensor_values(self._hass)
            if sensor_block:
                text = f"{text}\n\n{sensor_block}"

        return text
