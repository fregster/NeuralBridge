"""Unit tests for benchmark_host_classifier module."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from custom_components.neuralbridge.benchmark_host_classifier import (
    CLOUD_INTEGRATION_DOMAINS,
    HOST_GROUP_CLOUD_UNKNOWN,
    HOST_GROUP_LOCAL,
    BenchmarkHostClassifier,
)
from custom_components.neuralbridge.const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    AGENT_TYPE_WEB_SEARCH,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hass(local_ip: str | None = None) -> MagicMock:
    """Return a minimal mock hass with optional api.local_ip."""
    hass = MagicMock()
    if local_ip is not None:
        hass.config.api.local_ip = local_ip
    else:
        hass.config.api = None
    return hass


def _make_reg_entry(config_entry_id: str | None = "entry-1") -> MagicMock:
    entry = MagicMock()
    entry.config_entry_id = config_entry_id
    return entry


def _make_config_entry(domain: str, data: dict | None = None) -> MagicMock:
    entry = MagicMock()
    entry.domain = domain
    entry.data = data or {}
    return entry


# ---------------------------------------------------------------------------
# classify_agent — AGENT_TYPE_OLLAMA
# ---------------------------------------------------------------------------


async def test_classify_ollama_localhost_is_local() -> None:
    """Ollama URL with 'localhost' hostname classifies as local."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "http://localhost:11434"}
    )
    assert result == HOST_GROUP_LOCAL


async def test_classify_ollama_127_0_0_1_is_local() -> None:
    """Ollama URL with 127.0.0.1 classifies as local."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "http://127.0.0.1:11434"}
    )
    assert result == HOST_GROUP_LOCAL


async def test_classify_ollama_ipv6_loopback_is_local() -> None:
    """Ollama URL with IPv6 loopback ::1 classifies as local."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "http://[::1]:11434"}
    )
    assert result == HOST_GROUP_LOCAL


async def test_classify_ollama_any_address_is_local() -> None:
    """Ollama URL with 0.0.0.0 classifies as local."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "http://0.0.0.0:11434"}
    )
    assert result == HOST_GROUP_LOCAL


async def test_classify_ollama_remote_host_uses_remote_group() -> None:
    """Ollama URL pointing at a distinct remote host produces remote:host:port."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "http://192.168.1.100:11434"}
    )
    assert result == "remote:192.168.1.100:11434"


async def test_classify_ollama_remote_custom_port() -> None:
    """Remote Ollama URL with non-default port encodes port in host group."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "http://myserver.local:8080"}
    )
    assert result == "remote:myserver.local:8080"


async def test_classify_ollama_default_port_when_omitted() -> None:
    """Remote Ollama URL without explicit port defaults to Ollama default 11434."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "http://aibox.local"}
    )
    assert result == "remote:aibox.local:11434"


async def test_classify_ollama_ha_host_ip_is_local() -> None:
    """Ollama URL using the HA host's LAN IP is treated as local."""
    hass = _make_hass(local_ip="192.168.1.50")
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "http://192.168.1.50:11434"}
    )
    assert result == HOST_GROUP_LOCAL


async def test_classify_ollama_missing_url_defaults_to_local() -> None:
    """Missing ollama_url falls back to DEFAULT_OLLAMA_URL which is localhost → local."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent({"agent_type": AGENT_TYPE_OLLAMA})
    assert result == HOST_GROUP_LOCAL


async def test_classify_ollama_malformed_url_no_host_is_local() -> None:
    """An Ollama URL that parses to an empty hostname is conservatively local."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "not a url \x00\xff"}
    )
    assert result == HOST_GROUP_LOCAL


async def test_classify_ollama_invalid_port_is_local() -> None:
    """An Ollama URL with a non-numeric port triggers ValueError → treated as local."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent(
        # urlparse("http://host:notaport").port raises ValueError
        {"agent_type": AGENT_TYPE_OLLAMA, "ollama_url": "http://localhost:notaport"}
    )
    assert result == HOST_GROUP_LOCAL


# ---------------------------------------------------------------------------
# classify_agent — AGENT_TYPE_INTEGRATED (cloud providers)
# ---------------------------------------------------------------------------


async def test_classify_integrated_openai_is_cloud() -> None:
    """INTEGRATED entity backed by openai_conversation → cloud group."""
    hass = _make_hass()
    config_entry = _make_config_entry("openai_conversation")
    reg_entry = _make_reg_entry("entry-1")
    hass.config_entries.async_get_entry.return_value = config_entry

    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = reg_entry
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.openai"}
        )

    assert result == "cloud:openai_conversation"


async def test_classify_integrated_gemini_is_cloud() -> None:
    """INTEGRATED entity backed by google_generative_ai_conversation → cloud group."""
    hass = _make_hass()
    config_entry = _make_config_entry("google_generative_ai_conversation")
    reg_entry = _make_reg_entry("entry-g")
    hass.config_entries.async_get_entry.return_value = config_entry

    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = reg_entry
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.gemini"}
        )

    assert result == "cloud:google_generative_ai_conversation"


async def test_classify_integrated_all_cloud_domains_produce_cloud_group() -> None:
    """Every domain in CLOUD_INTEGRATION_DOMAINS maps to a cloud:* host group."""
    for domain in CLOUD_INTEGRATION_DOMAINS:
        hass = _make_hass()
        config_entry = _make_config_entry(domain)
        reg_entry = _make_reg_entry("entry-x")
        hass.config_entries.async_get_entry.return_value = config_entry

        classifier = BenchmarkHostClassifier(hass)

        with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
            mock_er = MagicMock()
            mock_er.async_get.return_value = reg_entry
            mock_er_fn.return_value = mock_er
            result = await classifier.classify_agent(
                {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.x"}
            )

        assert result == f"cloud:{domain}", f"Domain {domain!r} did not produce cloud group"


# ---------------------------------------------------------------------------
# classify_agent — AGENT_TYPE_INTEGRATED (Ollama HA integration)
# ---------------------------------------------------------------------------


async def test_classify_integrated_ollama_integration_local_url_is_local() -> None:
    """INTEGRATED entity backed by HA Ollama integration at localhost → local."""
    hass = _make_hass()
    config_entry = _make_config_entry("ollama", {"url": "http://localhost:11434"})
    reg_entry = _make_reg_entry("entry-ol")
    hass.config_entries.async_get_entry.return_value = config_entry

    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = reg_entry
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.ollama_llama3"}
        )

    assert result == HOST_GROUP_LOCAL


async def test_classify_integrated_ollama_integration_remote_url_is_remote() -> None:
    """INTEGRATED HA Ollama integration pointing at a remote host → remote group."""
    hass = _make_hass()
    config_entry = _make_config_entry("ollama", {"url": "http://ollamabox.local:11434"})
    reg_entry = _make_reg_entry("entry-ol2")
    hass.config_entries.async_get_entry.return_value = config_entry

    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = reg_entry
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.ollama_remote"}
        )

    assert result == "remote:ollamabox.local:11434"


async def test_classify_integrated_ollama_integration_no_url_defaults_local() -> None:
    """INTEGRATED HA Ollama integration with no URL in data defaults to localhost → local."""
    hass = _make_hass()
    config_entry = _make_config_entry("ollama", {})  # no 'url' key
    reg_entry = _make_reg_entry("entry-ol3")
    hass.config_entries.async_get_entry.return_value = config_entry

    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = reg_entry
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.ollama_nourl"}
        )

    assert result == HOST_GROUP_LOCAL


# ---------------------------------------------------------------------------
# classify_agent — AGENT_TYPE_INTEGRATED (unknown / fallback)
# ---------------------------------------------------------------------------


async def test_classify_integrated_unknown_domain_is_local() -> None:
    """INTEGRATED entity with an unknown backing domain is treated as local (conservative)."""
    hass = _make_hass()
    config_entry = _make_config_entry("my_custom_llm_integration")
    reg_entry = _make_reg_entry("entry-custom")
    hass.config_entries.async_get_entry.return_value = config_entry

    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = reg_entry
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.custom"}
        )

    assert result == HOST_GROUP_LOCAL


async def test_classify_integrated_no_entity_registry_entry_is_cloud_unknown() -> None:
    """INTEGRATED entity not found in entity registry returns cloud:unknown."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = None  # entity not found
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.missing"}
        )

    assert result == HOST_GROUP_CLOUD_UNKNOWN


async def test_classify_integrated_no_config_entry_id_is_cloud_unknown() -> None:
    """INTEGRATED entity registry entry with no config_entry_id returns cloud:unknown."""
    hass = _make_hass()
    reg_entry = _make_reg_entry(config_entry_id=None)
    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = reg_entry
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.orphan"}
        )

    assert result == HOST_GROUP_CLOUD_UNKNOWN


async def test_classify_integrated_config_entry_not_found_is_cloud_unknown() -> None:
    """INTEGRATED entity whose config entry cannot be resolved returns cloud:unknown."""
    hass = _make_hass()
    reg_entry = _make_reg_entry("entry-gone")
    hass.config_entries.async_get_entry.return_value = None  # entry deleted
    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = reg_entry
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.gone"}
        )

    assert result == HOST_GROUP_CLOUD_UNKNOWN


async def test_classify_integrated_entity_registry_exception_is_cloud_unknown() -> None:
    """When entity registry lookup raises, INTEGRATED falls back to cloud:unknown."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)

    with patch(
        "homeassistant.helpers.entity_registry.async_get",
        side_effect=RuntimeError("registry unavailable"),
    ):
        result = await classifier.classify_agent(
            {"agent_type": AGENT_TYPE_INTEGRATED, "entity_id": "conversation.error"}
        )

    assert result == HOST_GROUP_CLOUD_UNKNOWN


async def test_classify_integrated_missing_entity_id_is_cloud_unknown() -> None:
    """INTEGRATED agent with no entity_id key returns cloud:unknown."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_fn:
        mock_er = MagicMock()
        mock_er.async_get.return_value = None
        mock_er_fn.return_value = mock_er
        result = await classifier.classify_agent({"agent_type": AGENT_TYPE_INTEGRATED})

    assert result == HOST_GROUP_CLOUD_UNKNOWN


# ---------------------------------------------------------------------------
# classify_agent — other agent types
# ---------------------------------------------------------------------------


async def test_classify_local_ha_is_local() -> None:
    """AGENT_TYPE_LOCAL_HA always classifies as local."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent({"agent_type": AGENT_TYPE_LOCAL_HA})
    assert result == HOST_GROUP_LOCAL


async def test_classify_web_search_is_cloud() -> None:
    """AGENT_TYPE_WEB_SEARCH classifies as cloud:web_search."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent({"agent_type": AGENT_TYPE_WEB_SEARCH})
    assert result == "cloud:web_search"


async def test_classify_unknown_agent_type_is_cloud_unknown() -> None:
    """An unrecognised agent_type string returns cloud:unknown."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent({"agent_type": "banana"})
    assert result == HOST_GROUP_CLOUD_UNKNOWN


async def test_classify_missing_agent_type_is_cloud_unknown() -> None:
    """An agent config with no agent_type key returns cloud:unknown."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    result = await classifier.classify_agent({})
    assert result == HOST_GROUP_CLOUD_UNKNOWN


# ---------------------------------------------------------------------------
# get_semaphore — concurrency limits and sharing
# ---------------------------------------------------------------------------


def test_get_semaphore_local_has_concurrency_of_one() -> None:
    """Local host-group gets Semaphore(1) — one benchmark at a time."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    sem = classifier.get_semaphore(HOST_GROUP_LOCAL)
    assert sem._value == 1


def test_get_semaphore_remote_has_concurrency_of_one() -> None:
    """Remote host-group gets Semaphore(1) — one benchmark at a time per host."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    sem = classifier.get_semaphore("remote:192.168.1.100:11434")
    assert sem._value == 1


def test_get_semaphore_cloud_has_high_concurrency() -> None:
    """Cloud host-group gets a high-limit semaphore (effectively unlimited)."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    sem = classifier.get_semaphore("cloud:openai_conversation")
    assert sem._value == 1_000


def test_get_semaphore_cloud_unknown_has_high_concurrency() -> None:
    """cloud:unknown also gets the high-limit semaphore."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    sem = classifier.get_semaphore(HOST_GROUP_CLOUD_UNKNOWN)
    assert sem._value == 1_000


def test_get_semaphore_same_group_returns_same_object() -> None:
    """Calling get_semaphore twice with the same key returns the identical object."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    s1 = classifier.get_semaphore("remote:box:11434")
    s2 = classifier.get_semaphore("remote:box:11434")
    assert s1 is s2


def test_get_semaphore_different_groups_are_independent() -> None:
    """Different host-group strings receive independent semaphore objects."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    s_local = classifier.get_semaphore(HOST_GROUP_LOCAL)
    s_remote = classifier.get_semaphore("remote:box:11434")
    s_cloud = classifier.get_semaphore("cloud:openai_conversation")
    assert s_local is not s_remote
    assert s_local is not s_cloud
    assert s_remote is not s_cloud


async def test_local_semaphore_serialises_concurrent_requests() -> None:
    """Two tasks competing for the local semaphore run one after the other."""
    hass = _make_hass()
    classifier = BenchmarkHostClassifier(hass)
    sem = classifier.get_semaphore(HOST_GROUP_LOCAL)
    assert sem._value == 1

    order: list[str] = []

    async def task_a() -> None:
        async with sem:
            order.append("a-start")
            await asyncio.sleep(0)
            order.append("a-end")

    async def task_b() -> None:
        async with sem:
            order.append("b-start")
            await asyncio.sleep(0)
            order.append("b-end")

    await asyncio.gather(task_a(), task_b())
    # The two tasks must not interleave
    assert order in (
        ["a-start", "a-end", "b-start", "b-end"],
        ["b-start", "b-end", "a-start", "a-end"],
    )


# ---------------------------------------------------------------------------
# _is_ha_host
# ---------------------------------------------------------------------------


def test_is_ha_host_matches_api_local_ip() -> None:
    """_is_ha_host returns True when hostname equals hass.config.api.local_ip."""
    hass = _make_hass(local_ip="10.0.0.1")
    classifier = BenchmarkHostClassifier(hass)
    assert classifier._is_ha_host("10.0.0.1") is True


def test_is_ha_host_no_match_different_ip() -> None:
    """_is_ha_host returns False for an IP that differs from HA's LAN IP."""
    hass = _make_hass(local_ip="10.0.0.1")
    classifier = BenchmarkHostClassifier(hass)
    assert classifier._is_ha_host("10.0.0.2") is False


def test_is_ha_host_no_api_object_returns_false() -> None:
    """_is_ha_host returns False when hass.config.api is None."""
    hass = _make_hass(local_ip=None)  # sets api = None
    classifier = BenchmarkHostClassifier(hass)
    assert classifier._is_ha_host("10.0.0.1") is False


def test_is_ha_host_api_local_ip_none_returns_false() -> None:
    """_is_ha_host returns False when api.local_ip is None."""
    hass = MagicMock()
    hass.config.api.local_ip = None
    classifier = BenchmarkHostClassifier(hass)
    assert classifier._is_ha_host("10.0.0.1") is False


def test_is_ha_host_exception_returns_false() -> None:
    """_is_ha_host returns False when attribute access raises."""

    class _BrokenApi:
        @property
        def local_ip(self) -> str:
            raise RuntimeError("api broken")

    hass = MagicMock()
    hass.config.api = _BrokenApi()
    classifier = BenchmarkHostClassifier(hass)
    # Should not raise; returns False safely.
    assert classifier._is_ha_host("any") is False
