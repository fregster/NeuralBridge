"""Host-group classification for NeuralBridge agent benchmarking.

Determines whether agents share the same physical host so that
:class:`~.benchmark_scheduler.BenchmarkScheduler` can run them in **series**
(local/remote Ollama) or in **parallel** (cloud services) without exhausting
RAM, VRAM, or CPU on the inference host.

Classification logic
--------------------
* ``AGENT_TYPE_OLLAMA`` with a localhost / HA-host URL → ``"local"``
* ``AGENT_TYPE_OLLAMA`` with a distinct remote URL → ``"remote:<host>:<port>"``
* ``AGENT_TYPE_INTEGRATED`` backed by the HA Ollama integration on localhost
  → ``"local"`` (same semaphore pool as a direct Ollama agent on the same host)
* ``AGENT_TYPE_INTEGRATED`` backed by a known cloud provider
  → ``"cloud:<domain>"`` (effectively unlimited concurrency)
* ``AGENT_TYPE_INTEGRATED`` with unknown backing → ``"local"`` (conservative)
* ``AGENT_TYPE_WEB_SEARCH`` → ``"cloud:web_search"``
* ``AGENT_TYPE_LOCAL_HA``  → ``"local"`` (runs inside HA itself)

Per-group concurrency
---------------------
* ``"local"`` and ``"remote:…"`` groups share a ``Semaphore(1)`` — one
  benchmark at a time per physical host.
* ``"cloud:…"`` groups share a ``Semaphore(1000)`` — effectively unlimited.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    AGENT_TYPE_WEB_SEARCH,
    CONF_AGENT_TYPE,
    CONF_ENTITY_ID,
    CONF_OLLAMA_URL,
    DEFAULT_OLLAMA_URL,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hostnames that resolve to the local machine.
_LOOPBACK_HOSTS: frozenset[str] = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",  # noqa: S104  # nosec B104 — lookup set, not a bind address
    }
)

# HA integration domain names that are definitively cloud-hosted.
# These integrations never run local inference, so their benchmarks can always
# be executed in parallel without any shared-hardware resource contention.
CLOUD_INTEGRATION_DOMAINS: frozenset[str] = frozenset(
    {
        "openai_conversation",  # ChatGPT / OpenAI
        "google_generative_ai_conversation",  # Google Gemini
        "anthropic_conversation",  # Claude (Anthropic)
        "groq",  # Groq cloud inference
        "mistral",  # Mistral AI
        "together_ai",  # Together AI
        "perplexity_ai",  # Perplexity AI
        "openrouter",  # OpenRouter (routes to cloud LLMs)
        "aws_bedrock_conversation",  # Amazon Bedrock
        "azure_openai_conversation",  # Azure OpenAI
    }
)

# Exported sentinel strings so callers can compare without hard-coding literals.
HOST_GROUP_LOCAL: str = "local"
HOST_GROUP_CLOUD_UNKNOWN: str = "cloud:unknown"

# Concurrency limits per host-group class.
_LOCAL_CONCURRENCY: int = 1  # one benchmark at a time on local hardware
_CLOUD_CONCURRENCY: int = 1_000  # effectively unlimited for cloud services


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class BenchmarkHostClassifier:
    """Classifies agent configs into host-group buckets for serialised scheduling.

    Each host-group maps to an :class:`asyncio.Semaphore`.
    :class:`~.benchmark_scheduler.BenchmarkScheduler` acquires the semaphore
    before launching probes so local agents run one at a time while cloud
    agents fire concurrently.

    Args:
        hass: Home Assistant instance used for entity / config-entry lookups.
    """

    def __init__(self, hass: "HomeAssistant") -> None:
        """Initialise with a reference to Home Assistant.

        Args:
            hass: Home Assistant instance.
        """
        self._hass = hass
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify_agent(self, agent_config: dict[str, Any]) -> str:
        """Return the host-group string for *agent_config*.

        Args:
            agent_config: Agent configuration dict containing at minimum
                          ``agent_type`` and any type-specific keys.

        Returns:
            A host-group string — one of ``"local"``,
            ``"remote:<host>:<port>"``, or ``"cloud:<provider>"``.
        """
        agent_type: str = agent_config.get(CONF_AGENT_TYPE, "")

        if agent_type == AGENT_TYPE_OLLAMA:
            url: str = agent_config.get(CONF_OLLAMA_URL, DEFAULT_OLLAMA_URL)
            return self._classify_ollama_url(url)

        if agent_type == AGENT_TYPE_INTEGRATED:
            entity_id: str = agent_config.get(CONF_ENTITY_ID, "")
            return await self._classify_integrated_entity(entity_id)

        if agent_type == AGENT_TYPE_LOCAL_HA:
            return HOST_GROUP_LOCAL

        if agent_type == AGENT_TYPE_WEB_SEARCH:
            return "cloud:web_search"

        return HOST_GROUP_CLOUD_UNKNOWN

    def get_semaphore(self, host_group: str) -> asyncio.Semaphore:
        """Return the :class:`asyncio.Semaphore` for *host_group*, creating on demand.

        Cloud groups receive a high-limit semaphore (effectively unlimited);
        local and remote groups receive ``Semaphore(1)`` so probes run in series.

        Args:
            host_group: Host-group string returned by :meth:`classify_agent`.

        Returns:
            An :class:`asyncio.Semaphore` shared across all agents in the group.
        """
        if host_group not in self._semaphores:
            limit = _CLOUD_CONCURRENCY if host_group.startswith("cloud") else _LOCAL_CONCURRENCY
            self._semaphores[host_group] = asyncio.Semaphore(limit)
        return self._semaphores[host_group]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_ollama_url(self, url: str) -> str:
        """Classify an Ollama direct-HTTP URL as ``"local"`` or ``"remote:h:p"``.

        Args:
            url: Raw Ollama URL string configured by the user.

        Returns:
            ``"local"`` if the URL resolves to the HA host or a loopback
            address, otherwise ``"remote:<hostname>:<port>"``.
        """
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            port = parsed.port or 11434
        except ValueError:
            # Malformed URL — run conservatively in series with local agents.
            _LOGGER.debug("Malformed Ollama URL %r; treating as local", url)
            return HOST_GROUP_LOCAL

        if not host or host in _LOOPBACK_HOSTS or self._is_ha_host(host):
            return HOST_GROUP_LOCAL

        return f"remote:{host}:{port}"

    async def _classify_integrated_entity(self, entity_id: str) -> str:
        """Classify an INTEGRATED conversation entity by its backing config entry.

        Looks up the entity in HA's entity registry to find its config entry,
        then uses the config-entry domain to decide cloud vs. local.

        Args:
            entity_id: HA conversation entity ID, e.g. ``"conversation.ollama_..."``.

        Returns:
            A host-group string.
        """
        config_entry = self._find_config_entry_for_entity(entity_id)
        if config_entry is None:
            _LOGGER.debug(
                "No config entry found for INTEGRATED entity %s; defaulting to %s",
                entity_id,
                HOST_GROUP_CLOUD_UNKNOWN,
            )
            return HOST_GROUP_CLOUD_UNKNOWN

        domain: str = config_entry.domain

        if domain in CLOUD_INTEGRATION_DOMAINS:
            return f"cloud:{domain}"

        if domain == "ollama":
            url: str = config_entry.data.get("url", DEFAULT_OLLAMA_URL)
            return self._classify_ollama_url(url)

        # Unknown / third-party integration — run in series (conservative).
        _LOGGER.debug(
            "Unknown HA integration domain '%s' for entity %s; running in series (local)",
            domain,
            entity_id,
        )
        return HOST_GROUP_LOCAL

    def _find_config_entry_for_entity(self, entity_id: str) -> Any | None:
        """Look up the HA config entry that owns *entity_id*.

        Args:
            entity_id: The conversation entity ID to look up.

        Returns:
            The :class:`homeassistant.config_entries.ConfigEntry` if found,
            otherwise ``None``.
        """
        try:
            from homeassistant.helpers.entity_registry import (  # noqa: PLC0415
                async_get as async_get_entity_registry,
            )

            er = async_get_entity_registry(self._hass)
            reg_entry = er.async_get(entity_id)
            if reg_entry is None or reg_entry.config_entry_id is None:
                return None
            return self._hass.config_entries.async_get_entry(reg_entry.config_entry_id)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.debug("Entity registry lookup failed for %s: %s", entity_id, err)
            return None

    def _is_ha_host(self, hostname: str) -> bool:
        """Return ``True`` if *hostname* matches the HA server's own LAN address.

        Args:
            hostname: Lowercased hostname extracted from an Ollama URL.

        Returns:
            ``True`` if the hostname corresponds to this HA instance's LAN IP.
        """
        with contextlib.suppress(Exception):
            ha_api = getattr(self._hass.config, "api", None)
            if ha_api is not None:
                local_ip: str | None = getattr(ha_api, "local_ip", None)
                if local_ip and hostname == local_ip.lower():
                    return True
        return False
