"""Unit tests for agent_benchmark module."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.neuralbridge.agent_benchmark import (
    AgentBenchmarker,
    BenchmarkHostClassifier,
    BenchmarkProfile,
    BenchmarkStatus,
    ScoreBreakdown,
    _compute_scores,
    _evaluate_probe,
)
from custom_components.neuralbridge.benchmark_probes import BENCHMARK_PROBES, BenchmarkProbe
from custom_components.neuralbridge.benchmark_scheduler import BenchmarkScheduler
from custom_components.neuralbridge.const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    AGENT_TYPE_WEB_SEARCH,
    BENCHMARK_PROBE_SUITE_VERSION,
    EVENT_BENCHMARK_COMPLETE,
    EVENT_BENCHMARK_FAILED,
    EVENT_BENCHMARK_STARTED,
)
from custom_components.neuralbridge.ollama_client import OllamaResponse

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _basic_profile(
    agent_id: str = "agent-1",
    agent_type: str = AGENT_TYPE_OLLAMA,
    version: int = BENCHMARK_PROBE_SUITE_VERSION,
    status: BenchmarkStatus = BenchmarkStatus.COMPLETE,
) -> BenchmarkProfile:
    """Return a ready-made BenchmarkProfile for use in tests."""
    return BenchmarkProfile(
        agent_id=agent_id,
        agent_name="Test Agent",
        agent_type=agent_type,
        probe_suite_version=version,
        status=status,
    )


def _ollama_agent_config(agent_id: str = "agent-1") -> dict:
    """Return a minimal Ollama agent config dict."""
    return {
        "id": agent_id,
        "agent_type": AGENT_TYPE_OLLAMA,
        "agent_name": "Ollama Agent",
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama3",
        "timeout": 30,
    }


# ---------------------------------------------------------------------------
# _evaluate_probe
# ---------------------------------------------------------------------------


def _make_probe(**kwargs) -> BenchmarkProbe:
    defaults = {
        "name": "test",
        "dimension": "factual",
        "messages": [{"role": "user", "content": "Q"}],
    }
    return BenchmarkProbe(**{**defaults, **kwargs})


def test_evaluate_probe_no_criteria_returns_true() -> None:
    """A probe with no criteria always passes."""
    probe = _make_probe()
    assert _evaluate_probe(probe, "anything at all") is True


def test_evaluate_probe_expected_exact_match() -> None:
    """Exact match check is case-insensitive."""
    probe = _make_probe(expected_exact="Paris")
    assert _evaluate_probe(probe, "paris") is True
    assert _evaluate_probe(probe, "PARIS") is True
    assert (
        _evaluate_probe(probe, "Paris ") is True
    )  # trailing space is stripped before compare → "Paris" == "Paris"


def test_evaluate_probe_expected_exact_no_match() -> None:
    """Exact match check fails when the text differs."""
    probe = _make_probe(expected_exact="Paris")
    assert _evaluate_probe(probe, "London") is False


def test_evaluate_probe_expected_exact_strips_whitespace() -> None:
    """Exact match trims leading/trailing whitespace from response."""
    probe = _make_probe(expected_exact="paris")
    assert _evaluate_probe(probe, "  paris  ") is True


def test_evaluate_probe_expected_contains_all_present() -> None:
    """All substrings must be present for expected_contains to pass."""
    probe = _make_probe(expected_contains=["hello", "world"])
    assert _evaluate_probe(probe, "Say hello to the world") is True


def test_evaluate_probe_expected_contains_partial_fail() -> None:
    """expected_contains fails when any substring is absent."""
    probe = _make_probe(expected_contains=["hello", "world"])
    assert _evaluate_probe(probe, "Say hello there") is False


def test_evaluate_probe_expected_contains_is_case_insensitive() -> None:
    """expected_contains comparison is case-insensitive."""
    probe = _make_probe(expected_contains=["JSON"])
    assert _evaluate_probe(probe, "please use json format") is True


def test_evaluate_probe_word_count_check_pass() -> None:
    """word_count_check passes for a single-word response."""
    probe = _make_probe(word_count_check=True)
    assert _evaluate_probe(probe, "Paris") is True


def test_evaluate_probe_word_count_check_fail_multi_word() -> None:
    """word_count_check fails for a multi-word response."""
    probe = _make_probe(word_count_check=True)
    assert _evaluate_probe(probe, "Paris, France") is False


def test_evaluate_probe_min_word_count_pass() -> None:
    """min_word_count passes when response has at least the required words."""
    probe = _make_probe(min_word_count=3)
    assert _evaluate_probe(probe, "one two three") is True
    assert _evaluate_probe(probe, "one two three four") is True


def test_evaluate_probe_min_word_count_fail() -> None:
    """min_word_count fails when response has fewer words than required."""
    probe = _make_probe(min_word_count=3)
    assert _evaluate_probe(probe, "one two") is False
    assert _evaluate_probe(probe, "one") is False


def test_evaluate_probe_max_word_count_pass() -> None:
    """max_word_count passes when response is within the word limit."""
    probe = _make_probe(max_word_count=5)
    assert _evaluate_probe(probe, "blue") is True
    assert _evaluate_probe(probe, "a ripe banana is yellow") is True


def test_evaluate_probe_max_word_count_fail() -> None:
    """max_word_count fails when response exceeds the word limit."""
    probe = _make_probe(max_word_count=3)
    assert _evaluate_probe(probe, "one two three four") is False


def test_evaluate_probe_starts_with_pass() -> None:
    """starts_with passes when response begins with the expected prefix."""
    probe = _make_probe(starts_with="{")
    assert _evaluate_probe(probe, '{"status": "ok"}') is True


def test_evaluate_probe_starts_with_fail() -> None:
    """starts_with fails when response has a preamble before the expected prefix."""
    probe = _make_probe(starts_with="{")
    assert _evaluate_probe(probe, 'Sure! Here is the JSON: {"status": "ok"}') is False


def test_evaluate_probe_starts_with_case_insensitive() -> None:
    """starts_with comparison is case-insensitive."""
    probe = _make_probe(starts_with="YES")
    assert _evaluate_probe(probe, "yes, that is correct") is True


def test_evaluate_probe_combined_criteria_all_must_pass() -> None:
    """When multiple criteria are set, all must pass simultaneously."""
    probe = _make_probe(expected_contains=["json"], word_count_check=True)
    # Single word containing "json" — passes both
    assert _evaluate_probe(probe, "json") is True
    # Multi-word — fails word_count_check
    assert _evaluate_probe(probe, "use json format") is False


# ---------------------------------------------------------------------------
# _compute_scores
# ---------------------------------------------------------------------------


def test_compute_scores_all_pass() -> None:
    """All probes passing yields maximum scores per dimension."""
    all_pass = {p.name: True for p in BENCHMARK_PROBES}
    scores = _compute_scores(all_pass)
    assert isinstance(scores, ScoreBreakdown)
    assert scores.score_instruction_following >= 1
    assert scores.score_reasoning >= 1
    assert scores.capability_score == (
        scores.score_instruction_following
        + scores.score_reasoning
        + scores.score_smart_home_intent
        + scores.score_factual
        + scores.score_memory
        + scores.score_structured_output
        + scores.score_creative_generation
        + scores.score_verbosity_calibration
        + scores.score_robustness
        + scores.score_safety_refusal
    )


def test_compute_scores_all_fail() -> None:
    """All probes failing yields zero total capability score."""
    all_fail = {p.name: False for p in BENCHMARK_PROBES}
    scores = _compute_scores(all_fail)
    assert scores.capability_score == 0
    assert all(
        s == 0
        for s in (
            scores.score_instruction_following,
            scores.score_reasoning,
            scores.score_smart_home_intent,
            scores.score_factual,
            scores.score_memory,
            scores.score_structured_output,
            scores.score_creative_generation,
            scores.score_verbosity_calibration,
            scores.score_robustness,
            scores.score_safety_refusal,
        )
    )


def test_compute_scores_empty_returns_zeros() -> None:
    """Empty probe_results dict yields all zeros."""
    scores = _compute_scores({})
    assert scores.capability_score == 0
    assert scores == ScoreBreakdown(
        score_instruction_following=0,
        score_reasoning=0,
        score_smart_home_intent=0,
        score_factual=0,
        score_memory=0,
        score_structured_output=0,
        score_creative_generation=0,
        score_verbosity_calibration=0,
        score_robustness=0,
        score_safety_refusal=0,
        capability_score=0,
    )


# ---------------------------------------------------------------------------
# BenchmarkProfile.to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_profile_round_trip_basic() -> None:
    """to_dict / from_dict round-trip preserves basic identity fields."""
    profile = BenchmarkProfile(
        agent_id="abc-123",
        agent_name="My Ollama",
        agent_type=AGENT_TYPE_OLLAMA,
        status=BenchmarkStatus.COMPLETE,
        probe_suite_version=BENCHMARK_PROBE_SUITE_VERSION,
        capability_score=5,
    )
    restored = BenchmarkProfile.from_dict(profile.to_dict())
    assert restored.agent_id == "abc-123"
    assert restored.agent_name == "My Ollama"
    assert restored.status == BenchmarkStatus.COMPLETE
    assert restored.capability_score == 5


def test_profile_from_dict_unknown_status_defaults_pending() -> None:
    """An unknown status string is replaced with PENDING on deserialisation."""
    data = {
        "agent_id": "x",
        "agent_name": "X",
        "agent_type": AGENT_TYPE_OLLAMA,
        "status": "nonexistent_status",
    }
    profile = BenchmarkProfile.from_dict(data)
    assert profile.status == BenchmarkStatus.PENDING


def test_profile_from_dict_missing_optional_fields_have_defaults() -> None:
    """from_dict tolerates missing optional fields and supplies defaults."""
    minimal = {"agent_id": "a", "agent_name": "A", "agent_type": AGENT_TYPE_LOCAL_HA}
    profile = BenchmarkProfile.from_dict(minimal)
    assert profile.realworld_tokens_per_sec is None
    assert profile.realworld_sample_count == 0
    assert profile.probe_results == {}


# ---------------------------------------------------------------------------
# AgentBenchmarker._check_version
# ---------------------------------------------------------------------------


def test_check_version_current_version_no_change() -> None:
    """Profile with up-to-date version is not invalidated."""
    profile = _basic_profile(version=BENCHMARK_PROBE_SUITE_VERSION)
    profile.capability_score = 7
    AgentBenchmarker._check_version(profile)
    assert profile.capability_score == 7
    assert profile.status == BenchmarkStatus.COMPLETE


def test_check_version_stale_clears_scores() -> None:
    """Profile with old version has scores cleared and status reset."""
    profile = _basic_profile(version=BENCHMARK_PROBE_SUITE_VERSION - 1)
    profile.capability_score = 7
    profile.median_latency_ms = 100.0
    profile.score_reasoning = 2
    profile.score_structured_output = 3
    profile.score_robustness = 2
    AgentBenchmarker._check_version(profile)
    assert profile.capability_score is None
    assert profile.median_latency_ms is None
    assert profile.score_reasoning is None
    assert profile.score_structured_output is None
    assert profile.score_robustness is None
    assert profile.status == BenchmarkStatus.PENDING
    assert profile.re_benchmark_on_save is True


def test_check_version_preserves_metadata() -> None:
    """Metadata fields are preserved when version is stale."""
    profile = _basic_profile(version=0)
    profile.model_family = "llama"
    profile.parameter_count_billions = 7.0
    profile.quantization = "Q4_K_M"
    AgentBenchmarker._check_version(profile)
    assert profile.model_family == "llama"
    assert profile.parameter_count_billions == 7.0
    assert profile.quantization == "Q4_K_M"


# ---------------------------------------------------------------------------
# AgentBenchmarker.ensure_profile / get_profile / remove_profile
# ---------------------------------------------------------------------------


async def test_ensure_profile_creates_new_profile(hass: HomeAssistant) -> None:
    """ensure_profile creates a new profile when agent_id is not known."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config("new-agent")
    profile = benchmarker.ensure_profile(config)
    assert profile.agent_id == "new-agent"


async def test_ensure_profile_is_idempotent(hass: HomeAssistant) -> None:
    """Calling ensure_profile twice returns the same object."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config("same-agent")
    p1 = benchmarker.ensure_profile(config)
    p2 = benchmarker.ensure_profile(config)
    assert p1 is p2


async def test_get_profile_returns_none_for_unknown(hass: HomeAssistant) -> None:
    """get_profile returns None for an agent_id that was never registered."""
    benchmarker = AgentBenchmarker(hass)
    assert benchmarker.get_profile("nope") is None


async def test_remove_profile_deletes_it(hass: HomeAssistant) -> None:
    """remove_profile removes the stored profile."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config("del-agent")
    benchmarker.ensure_profile(config)
    assert benchmarker.get_profile("del-agent") is not None
    benchmarker.remove_profile("del-agent")
    assert benchmarker.get_profile("del-agent") is None


async def test_remove_profile_cancels_pending_task(hass: HomeAssistant) -> None:
    """remove_profile cancels any pending benchmark task."""
    benchmarker = AgentBenchmarker(hass, warm_up_delay_seconds=9999)
    config = _ollama_agent_config("cancel-agent")
    benchmarker.ensure_profile(config)
    await benchmarker.async_schedule_benchmark("cancel-agent", config)
    task = benchmarker._pending_tasks.get("cancel-agent")
    assert task is not None and not task.done()
    benchmarker.remove_profile("cancel-agent")
    # Allow event loop to process the cancellation
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()


# ---------------------------------------------------------------------------
# AgentBenchmarker.cancel_pending / cancel_all_pending
# ---------------------------------------------------------------------------


async def test_cancel_pending_no_op_when_no_task(hass: HomeAssistant) -> None:
    """cancel_pending does not raise when no task is registered."""
    benchmarker = AgentBenchmarker(hass)
    benchmarker.cancel_pending("nonexistent")  # must not raise


async def test_cancel_all_pending_cancels_all(hass: HomeAssistant) -> None:
    """cancel_all_pending cancels every registered task."""
    benchmarker = AgentBenchmarker(hass, warm_up_delay_seconds=9999)
    for i in range(3):
        cfg = _ollama_agent_config(f"agent-{i}")
        benchmarker.ensure_profile(cfg)
        await benchmarker.async_schedule_benchmark(f"agent-{i}", cfg)

    assert len(benchmarker._pending_tasks) == 3
    benchmarker.cancel_all_pending()
    await asyncio.sleep(0)
    assert len(benchmarker._pending_tasks) == 0


# ---------------------------------------------------------------------------
# AgentBenchmarker.async_load / async_save
# ---------------------------------------------------------------------------


async def test_async_load_empty_store(hass: HomeAssistant) -> None:
    """async_load on an empty store leaves profiles dict empty."""
    benchmarker = AgentBenchmarker(hass)
    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value=None)
    benchmarker._store = mock_store
    await benchmarker.async_load()
    assert benchmarker._profiles == {}


async def test_async_load_restores_profiles(hass: HomeAssistant) -> None:
    """async_load deserialises stored profiles."""
    stored = {
        "agent-1": {
            "agent_id": "agent-1",
            "agent_name": "Loaded Agent",
            "agent_type": AGENT_TYPE_OLLAMA,
            "status": "complete",
            "probe_suite_version": BENCHMARK_PROBE_SUITE_VERSION,
            "re_benchmark_on_save": False,
        }
    }
    benchmarker = AgentBenchmarker(hass)
    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value=stored)
    benchmarker._store = mock_store
    await benchmarker.async_load()

    profile = benchmarker.get_profile("agent-1")
    assert profile is not None
    assert profile.agent_name == "Loaded Agent"
    assert profile.status == BenchmarkStatus.COMPLETE


async def test_async_load_invalidates_stale_profiles(hass: HomeAssistant) -> None:
    """async_load calls _check_version and resets stale profiles."""
    stored = {
        "stale-agent": {
            "agent_id": "stale-agent",
            "agent_name": "Stale",
            "agent_type": AGENT_TYPE_OLLAMA,
            "status": "complete",
            "probe_suite_version": 0,  # stale
            "capability_score": 8,
            "re_benchmark_on_save": False,
        }
    }
    benchmarker = AgentBenchmarker(hass)
    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value=stored)
    benchmarker._store = mock_store
    await benchmarker.async_load()

    profile = benchmarker.get_profile("stale-agent")
    assert profile is not None
    assert profile.status == BenchmarkStatus.PENDING
    assert profile.capability_score is None


async def test_async_save_serialises_all_profiles(hass: HomeAssistant) -> None:
    """async_save calls Store.async_save with all profiles serialised."""
    benchmarker = AgentBenchmarker(hass)
    benchmarker.ensure_profile(_ollama_agent_config("save-agent"))

    saved_data: dict = {}

    async def capture_save(data: dict) -> None:
        saved_data.update(data)

    mock_store = AsyncMock()
    mock_store.async_save = AsyncMock(side_effect=capture_save)
    benchmarker._store = mock_store
    await benchmarker.async_save()

    assert "save-agent" in saved_data


# ---------------------------------------------------------------------------
# AgentBenchmarker.update_realworld_telemetry
# ---------------------------------------------------------------------------


async def test_update_realworld_telemetry_first_sample(hass: HomeAssistant) -> None:
    """First real-world telemetry sample sets realworld_tokens_per_sec directly."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    profile = benchmarker.ensure_profile(config)
    profile.probe_suite_version = BENCHMARK_PROBE_SUITE_VERSION

    resp = OllamaResponse(content="ok", eval_count=50, eval_duration_ns=500_000_000)
    benchmarker.update_realworld_telemetry("agent-1", resp)

    assert profile.realworld_tokens_per_sec == pytest.approx(100.0)
    assert profile.realworld_sample_count == 1


async def test_update_realworld_telemetry_rolling_average(hass: HomeAssistant) -> None:
    """Subsequent samples update the rolling exponential moving average."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    profile = benchmarker.ensure_profile(config)
    profile.probe_suite_version = BENCHMARK_PROBE_SUITE_VERSION

    # First sample: 100 t/s
    r1 = OllamaResponse(content="a", eval_count=100, eval_duration_ns=1_000_000_000)
    benchmarker.update_realworld_telemetry("agent-1", r1)

    # Second sample: 200 t/s — result should be 0.2*200 + 0.8*100 = 120
    r2 = OllamaResponse(content="b", eval_count=200, eval_duration_ns=1_000_000_000)
    benchmarker.update_realworld_telemetry("agent-1", r2)

    assert profile.realworld_tokens_per_sec == pytest.approx(120.0)
    assert profile.realworld_sample_count == 2


async def test_update_realworld_telemetry_noop_for_unknown_agent(hass: HomeAssistant) -> None:
    """update_realworld_telemetry silently ignores unknown agent IDs."""
    benchmarker = AgentBenchmarker(hass)
    resp = OllamaResponse(content="ok", eval_count=50, eval_duration_ns=500_000_000)
    benchmarker.update_realworld_telemetry("no-such-agent", resp)  # must not raise


async def test_update_realworld_telemetry_version_guard(hass: HomeAssistant) -> None:
    """Telemetry is ignored when probe_suite_version does not match current constant."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    profile = benchmarker.ensure_profile(config)
    profile.probe_suite_version = BENCHMARK_PROBE_SUITE_VERSION - 1  # stale

    resp = OllamaResponse(content="ok", eval_count=50, eval_duration_ns=500_000_000)
    benchmarker.update_realworld_telemetry("agent-1", resp)

    assert profile.realworld_tokens_per_sec is None
    assert profile.realworld_sample_count == 0


async def test_update_realworld_telemetry_none_tps_noop(hass: HomeAssistant) -> None:
    """Telemetry is ignored when tokens_per_second returns None."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    profile = benchmarker.ensure_profile(config)
    profile.probe_suite_version = BENCHMARK_PROBE_SUITE_VERSION

    resp = OllamaResponse(content="ok")  # no eval fields → tps is None
    benchmarker.update_realworld_telemetry("agent-1", resp)

    assert profile.realworld_tokens_per_sec is None
    assert profile.realworld_sample_count == 0


# ---------------------------------------------------------------------------
# AgentBenchmarker._apply_probe_results
# ---------------------------------------------------------------------------


def test_apply_probe_results_sets_scores_and_latency() -> None:
    """_apply_probe_results computes scores and latency stats into the profile."""
    profile = _basic_profile(status=BenchmarkStatus.RUNNING)
    all_pass = {p.name: True for p in BENCHMARK_PROBES}
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    AgentBenchmarker._apply_probe_results(profile, all_pass, latencies)

    assert profile.status == BenchmarkStatus.COMPLETE
    assert profile.re_benchmark_on_save is False
    assert profile.capability_score is not None
    assert profile.median_latency_ms is not None
    assert profile.p95_latency_ms is not None


def test_apply_probe_results_single_latency() -> None:
    """_apply_probe_results handles a single latency value."""
    profile = _basic_profile(status=BenchmarkStatus.RUNNING)
    AgentBenchmarker._apply_probe_results(profile, {}, [99.0])
    assert profile.median_latency_ms == pytest.approx(99.0)
    assert profile.p95_latency_ms == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# AgentBenchmarker.async_run_benchmark — agent type dispatch
# ---------------------------------------------------------------------------


async def test_async_run_benchmark_skips_local_ha(hass: HomeAssistant) -> None:
    """async_run_benchmark skips LOCAL_HA agents."""
    benchmarker = AgentBenchmarker(hass)
    config = {"id": "ha-1", "agent_type": AGENT_TYPE_LOCAL_HA, "agent_name": "HA", "timeout": 5}
    benchmarker.ensure_profile(config)

    with patch.object(benchmarker, "async_save", new_callable=AsyncMock):
        await benchmarker.async_run_benchmark("ha-1", config)

    profile = benchmarker.get_profile("ha-1")
    assert profile is not None
    assert profile.status == BenchmarkStatus.SKIPPED


async def test_async_run_benchmark_skips_web_search(hass: HomeAssistant) -> None:
    """async_run_benchmark skips WEB_SEARCH agents."""
    benchmarker = AgentBenchmarker(hass)
    config = {"id": "ws-1", "agent_type": AGENT_TYPE_WEB_SEARCH, "agent_name": "WS", "timeout": 5}
    benchmarker.ensure_profile(config)

    with patch.object(benchmarker, "async_save", new_callable=AsyncMock):
        await benchmarker.async_run_benchmark("ws-1", config)

    profile = benchmarker.get_profile("ws-1")
    assert profile is not None
    assert profile.status == BenchmarkStatus.SKIPPED


async def test_async_run_benchmark_fires_started_event(hass: HomeAssistant) -> None:
    """async_run_benchmark fires EVENT_BENCHMARK_STARTED before probes run."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)

    events: list[dict] = []
    hass.bus.async_listen(EVENT_BENCHMARK_STARTED, lambda e: events.append(e.data))

    # Make _run_ollama_probes a no-op so benchmark completes immediately
    with (
        patch.object(benchmarker, "_run_ollama_probes", new_callable=AsyncMock),
        patch.object(benchmarker, "async_save", new_callable=AsyncMock),
    ):
        await benchmarker.async_run_benchmark("agent-1", config)

    await hass.async_block_till_done()
    assert any(e.get("agent_id") == "agent-1" for e in events)


async def test_async_run_benchmark_fires_complete_event(hass: HomeAssistant) -> None:
    """async_run_benchmark fires EVENT_BENCHMARK_COMPLETE on success."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)

    events: list[dict] = []
    hass.bus.async_listen(EVENT_BENCHMARK_COMPLETE, lambda e: events.append(e.data))

    with (
        patch.object(benchmarker, "_run_ollama_probes", new_callable=AsyncMock),
        patch.object(benchmarker, "async_save", new_callable=AsyncMock),
    ):
        await benchmarker.async_run_benchmark("agent-1", config)

    await hass.async_block_till_done()
    assert any(e.get("agent_id") == "agent-1" for e in events)


async def test_async_run_benchmark_fires_failed_event_on_exception(
    hass: HomeAssistant,
) -> None:
    """async_run_benchmark fires EVENT_BENCHMARK_FAILED when probes raise."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)

    events: list[dict] = []
    hass.bus.async_listen(EVENT_BENCHMARK_FAILED, lambda e: events.append(e.data))

    with (
        patch.object(
            benchmarker,
            "_run_ollama_probes",
            side_effect=RuntimeError("probe exploded"),
        ),
        patch.object(benchmarker, "async_save", new_callable=AsyncMock),
    ):
        await benchmarker.async_run_benchmark("agent-1", config)

    profile = benchmarker.get_profile("agent-1")
    assert profile is not None
    assert profile.status == BenchmarkStatus.FAILED
    assert "probe exploded" in (profile.error or "")
    await hass.async_block_till_done()
    assert any(e.get("agent_id") == "agent-1" for e in events)


async def test_async_run_benchmark_sets_version_before_probes(
    hass: HomeAssistant,
) -> None:
    """async_run_benchmark stamps probe_suite_version before running probes."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    profile = benchmarker.ensure_profile(config)
    assert profile.probe_suite_version == 0

    versions_seen: list[int] = []

    async def _capture_version(*args, **kwargs) -> None:
        versions_seen.append(benchmarker.get_profile("agent-1").probe_suite_version)  # type: ignore[union-attr]

    with (
        patch.object(benchmarker, "_run_ollama_probes", side_effect=_capture_version),
        patch.object(benchmarker, "async_save", new_callable=AsyncMock),
    ):
        await benchmarker.async_run_benchmark("agent-1", config)

    assert versions_seen and versions_seen[0] == BENCHMARK_PROBE_SUITE_VERSION


# ---------------------------------------------------------------------------
# AgentBenchmarker.async_run_metadata
# ---------------------------------------------------------------------------


async def test_async_run_metadata_noop_for_non_ollama(hass: HomeAssistant) -> None:
    """async_run_metadata is a no-op for non-Ollama agents."""
    benchmarker = AgentBenchmarker(hass)
    config = {
        "id": "ex-1",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Existing",
        "entity_id": "conversation.mock",
    }
    benchmarker.ensure_profile(config)
    # Should complete without calling the Ollama client at all
    await benchmarker.async_run_metadata("ex-1", config)
    profile = benchmarker.get_profile("ex-1")
    assert profile is not None
    assert profile.model_family is None


async def test_async_run_metadata_populates_model_fields(hass: HomeAssistant) -> None:
    """async_run_metadata fills model metadata from the /api/show response."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)

    show_resp = {
        "details": {
            "family": "llama",
            "quantization_level": "Q4_K_M",
            "parameter_size": "7B",
        },
        "size": 4_200_000_000,
        "modelinfo": {"general.context_length": 4096},
    }
    mock_client = AsyncMock()
    mock_client.async_show_model = AsyncMock(return_value=show_resp)

    with (
        patch.object(benchmarker, "_get_ollama_client", return_value=mock_client),
        patch.object(benchmarker, "async_save", new_callable=AsyncMock),
    ):
        await benchmarker.async_run_metadata("agent-1", config)

    profile = benchmarker.get_profile("agent-1")
    assert profile is not None
    assert profile.model_family == "llama"
    assert profile.quantization == "Q4_K_M"
    assert profile.parameter_count_billions == pytest.approx(7.0)
    assert profile.context_window == 4096
    assert profile.model_size_bytes == 4_200_000_000


async def test_async_run_metadata_handles_api_exception(hass: HomeAssistant) -> None:
    """async_run_metadata logs and swallows exceptions from async_show_model."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)

    mock_client = AsyncMock()
    mock_client.async_show_model = AsyncMock(side_effect=RuntimeError("Ollama down"))

    with patch.object(benchmarker, "_get_ollama_client", return_value=mock_client):
        await benchmarker.async_run_metadata("agent-1", config)  # must not raise

    profile = benchmarker.get_profile("agent-1")
    assert profile is not None
    assert profile.model_family is None  # not populated


# ---------------------------------------------------------------------------
# BenchmarkStatus StrEnum
# ---------------------------------------------------------------------------


def test_benchmark_status_values() -> None:
    """All expected BenchmarkStatus values exist."""
    assert BenchmarkStatus.PENDING == "pending"
    assert BenchmarkStatus.SCHEDULED == "scheduled"
    assert BenchmarkStatus.RUNNING == "running"
    assert BenchmarkStatus.COMPLETE == "complete"
    assert BenchmarkStatus.FAILED == "failed"
    assert BenchmarkStatus.SKIPPED == "skipped"


# ---------------------------------------------------------------------------
# remove_profile — client close
# ---------------------------------------------------------------------------


async def test_remove_profile_closes_existing_client(hass: HomeAssistant) -> None:
    """remove_profile schedules client.close() when a client exists."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)
    # Create an Ollama client by calling _get_ollama_client directly
    benchmarker._get_ollama_client("agent-1", config)
    assert "agent-1" in benchmarker._ollama_clients

    closed_calls: list[object] = []

    original_create_task = hass.async_create_task

    def _capture_task(coro):
        closed_calls.append(coro)
        return original_create_task(coro)

    with patch.object(hass, "async_create_task", side_effect=_capture_task):
        benchmarker.remove_profile("agent-1")

    assert len(closed_calls) == 1
    assert "agent-1" not in benchmarker._ollama_clients


# ---------------------------------------------------------------------------
# async_run_metadata — client is None / data is None / numeric params
# ---------------------------------------------------------------------------


async def test_async_run_metadata_client_is_none_returns_early(
    hass: HomeAssistant,
) -> None:
    """async_run_metadata returns early when no Ollama URL or model configured."""
    benchmarker = AgentBenchmarker(hass)
    # Config has correct type but no URL/model → _get_ollama_client returns None
    config = {
        "id": "no-url",
        "agent_type": AGENT_TYPE_OLLAMA,
        "agent_name": "Missing URL",
    }
    benchmarker.ensure_profile(config)
    await benchmarker.async_run_metadata("no-url", config)  # must not raise
    profile = benchmarker.get_profile("no-url")
    assert profile is not None
    assert profile.model_family is None  # nothing was set


async def test_async_run_metadata_data_none_returns_early(hass: HomeAssistant) -> None:
    """async_run_metadata returns early when async_show_model() returns None."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)

    mock_client = AsyncMock()
    mock_client.async_show_model = AsyncMock(return_value=None)

    with patch.object(benchmarker, "_get_ollama_client", return_value=mock_client):
        await benchmarker.async_run_metadata("agent-1", config)  # must not raise

    profile = benchmarker.get_profile("agent-1")
    assert profile is not None
    assert profile.model_family is None


async def test_async_run_metadata_numeric_parameter_size(hass: HomeAssistant) -> None:
    """async_run_metadata handles numeric parameter_size (e.g. float)."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)

    show_resp = {
        "details": {
            "family": "qwen",
            "quantization_level": "Q4_K_M",
            "parameter_size": 0.6,  # numeric float, not a string like "0.6B"
        },
        "size": 500_000_000,
        "modelinfo": {},
    }
    mock_client = AsyncMock()
    mock_client.async_show_model = AsyncMock(return_value=show_resp)

    with (
        patch.object(benchmarker, "_get_ollama_client", return_value=mock_client),
        patch.object(benchmarker, "async_save", new_callable=AsyncMock),
    ):
        await benchmarker.async_run_metadata("agent-1", config)

    profile = benchmarker.get_profile("agent-1")
    assert profile is not None
    assert profile.parameter_count_billions == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# _warm_up_delay_seconds property — getter and setter
# ---------------------------------------------------------------------------


def test_warm_up_delay_seconds_getter_and_setter(hass: HomeAssistant) -> None:
    """_warm_up_delay_seconds getter reads from and setter writes to the scheduler."""
    benchmarker = AgentBenchmarker(hass)
    # Default value forwarded from scheduler
    default = benchmarker._warm_up_delay_seconds
    assert isinstance(default, int)
    assert default > 0
    # Setter updates the underlying scheduler attribute
    benchmarker._warm_up_delay_seconds = 0
    assert benchmarker._warm_up_delay_seconds == 0


# ---------------------------------------------------------------------------
# async_schedule_benchmark — warm-up exception paths
# ---------------------------------------------------------------------------


async def test_async_schedule_benchmark_warmup_runs_benchmark(
    hass: HomeAssistant,
) -> None:
    """async_schedule_benchmark creates a task that eventually calls async_run_benchmark."""
    benchmarker = AgentBenchmarker(hass)
    benchmarker._warm_up_delay_seconds = 0  # skip the delay

    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)

    run_calls: list[str] = []

    async def fake_run(agent_id: str, _cfg: dict) -> None:
        run_calls.append(agent_id)

    with patch.object(benchmarker, "async_run_benchmark", side_effect=fake_run):
        await benchmarker.async_schedule_benchmark("agent-1", config)
        await hass.async_block_till_done()

    assert run_calls == ["agent-1"]


async def test_async_schedule_benchmark_warmup_non_cancelled_exception(
    hass: HomeAssistant,
) -> None:
    """_warm_up_then_probe logs (and does not propagate) non-cancellation errors."""
    benchmarker = AgentBenchmarker(hass)
    benchmarker._warm_up_delay_seconds = 0

    config = _ollama_agent_config()
    benchmarker.ensure_profile(config)

    async def boom(*_: object, **__: object) -> None:
        raise RuntimeError("warm-up boom")

    with patch.object(benchmarker, "async_run_benchmark", side_effect=boom):
        await benchmarker.async_schedule_benchmark("agent-1", config)
        # Task runs but error is swallowed — no exception propagates
        await hass.async_block_till_done()

    # No assertion needed — test passes if no exception was raised


# ---------------------------------------------------------------------------
# async_run_benchmark — EXISTING and unknown agent types
# ---------------------------------------------------------------------------


async def test_async_run_benchmark_existing_calls_run_existing_probes(
    hass: HomeAssistant,
) -> None:
    """async_run_benchmark calls _run_existing_probes for EXISTING agent type."""
    benchmarker = AgentBenchmarker(hass)
    config = {
        "id": "ex-1",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Gemini",
        "entity_id": "conversation.gemini",
        "timeout": 30,
    }
    benchmarker.ensure_profile(config)

    with (
        patch.object(benchmarker, "_run_existing_probes", new_callable=AsyncMock) as mock_ep,
        patch.object(benchmarker, "async_save", new_callable=AsyncMock),
    ):
        await benchmarker.async_run_benchmark("ex-1", config)
        await hass.async_block_till_done()

    mock_ep.assert_called_once()


async def test_async_run_benchmark_unknown_type_sets_failed_status(
    hass: HomeAssistant,
) -> None:
    """async_run_benchmark marks profile FAILED for an unknown agent type."""
    benchmarker = AgentBenchmarker(hass)
    config = {
        "id": "unk-1",
        "agent_type": "totally_custom_type",
        "agent_name": "Unknown",
        "timeout": 30,
    }
    benchmarker.ensure_profile(config)

    with patch.object(benchmarker, "async_save", new_callable=AsyncMock):
        await benchmarker.async_run_benchmark("unk-1", config)
        await hass.async_block_till_done()

    profile = benchmarker.get_profile("unk-1")
    assert profile is not None
    assert profile.status == BenchmarkStatus.FAILED
    assert profile.error is not None
    assert "totally_custom_type" in profile.error


# ---------------------------------------------------------------------------
# _get_ollama_client — direct tests
# ---------------------------------------------------------------------------


async def test_get_ollama_client_returns_none_missing_url(hass: HomeAssistant) -> None:
    """_get_ollama_client returns None when ollama_url is absent."""
    benchmarker = AgentBenchmarker(hass)
    config = {"id": "a-1", "agent_type": AGENT_TYPE_OLLAMA, "ollama_model": "llama3"}
    result = benchmarker._get_ollama_client("a-1", config)
    assert result is None


async def test_get_ollama_client_creates_and_caches_client(
    hass: HomeAssistant,
) -> None:
    """_get_ollama_client creates an OllamaClient and caches it for re-use."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()

    c1 = benchmarker._get_ollama_client("agent-1", config)
    c2 = benchmarker._get_ollama_client("agent-1", config)

    assert c1 is not None
    assert c1 is c2  # same instance returned on second call


# ---------------------------------------------------------------------------
# _run_single_ollama_probe — timeout / None response / success
# ---------------------------------------------------------------------------


async def test_run_single_ollama_probe_timeout(hass: HomeAssistant) -> None:
    """_run_single_ollama_probe returns passed=False on per-probe timeout."""
    benchmarker = AgentBenchmarker(hass)
    probe = BenchmarkProbe(
        name="slow",
        dimension="factual",
        messages=[{"role": "user", "content": "Q"}],
        timeout_seconds=1,
    )

    async def slow_chat(_messages: object) -> None:
        await asyncio.sleep(100)

    mock_client = AsyncMock()
    mock_client.chat = slow_chat
    result = await benchmarker._run_single_ollama_probe(mock_client, probe)

    assert result.passed is False
    assert result.probe_name == "slow"
    assert result.latency_ms >= 0


async def test_run_single_ollama_probe_none_response(hass: HomeAssistant) -> None:
    """_run_single_ollama_probe returns passed=False when chat() returns None."""
    benchmarker = AgentBenchmarker(hass)
    probe = BenchmarkProbe(
        name="empty",
        dimension="factual",
        messages=[{"role": "user", "content": "Q"}],
        expected_contains=["A"],
    )
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=None)

    result = await benchmarker._run_single_ollama_probe(mock_client, probe)

    assert result.passed is False
    assert result.probe_name == "empty"


async def test_run_single_ollama_probe_pass(hass: HomeAssistant) -> None:
    """_run_single_ollama_probe returns passed=True when response matches criteria."""
    benchmarker = AgentBenchmarker(hass)
    probe = BenchmarkProbe(
        name="factual-q",
        dimension="factual",
        messages=[{"role": "user", "content": "Capital of France?"}],
        expected_exact="Paris",
    )
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(
        return_value=OllamaResponse(
            content="Paris",
            eval_count=5,
            eval_duration_ns=500_000_000,
            prompt_eval_count=3,
            prompt_eval_duration_ns=300_000_000,
        )
    )

    result = await benchmarker._run_single_ollama_probe(mock_client, probe)

    assert result.passed is True
    assert result.eval_count == 5
    assert result.eval_duration_ns == 500_000_000


# ---------------------------------------------------------------------------
# _run_ollama_probes — full run
# ---------------------------------------------------------------------------


async def test_run_ollama_probes_aggregates_results(hass: HomeAssistant) -> None:
    """_run_ollama_probes runs all probes and aggregates results into profile."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config()
    profile = benchmarker.ensure_profile(config)

    async def _mock_single_probe(_client: object, probe: BenchmarkProbe) -> object:
        from custom_components.neuralbridge.agent_benchmark import ProbeResult  # noqa: PLC0415

        return ProbeResult(
            probe_name=probe.name,
            dimension=probe.dimension,
            passed=True,
            latency_ms=50.0,
            eval_count=10,
            eval_duration_ns=1_000_000_000,
            prompt_eval_count=5,
            prompt_eval_duration_ns=500_000_000,
        )

    with (
        patch.object(benchmarker, "_get_ollama_client", return_value=AsyncMock()),
        patch.object(
            benchmarker,
            "_run_single_ollama_probe",
            side_effect=_mock_single_probe,
        ),
    ):
        await benchmarker._run_ollama_probes("agent-1", config, profile)

    assert profile.probe_tokens_per_sec is not None
    assert profile.probe_tokens_per_sec == pytest.approx(10.0)
    assert profile.probe_prompt_tps is not None


# ---------------------------------------------------------------------------
# _run_existing_probes — success / timeout / exception / no user messages
# ---------------------------------------------------------------------------


async def test_run_existing_probes_success() -> None:
    """_run_existing_probes processes probe results from HA conversation service."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass)
    config = {
        "id": "ex-2",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Gemini2",
        "entity_id": "conversation.gemini",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    ha_response = {"response": {"speech": {"plain": {"speech": "Paris"}}}}
    mock_hass.services.async_call = AsyncMock(return_value=ha_response)

    await benchmarker._run_existing_probes("ex-2", config, profile)

    assert profile.capability_score is not None


async def test_run_existing_probes_timeout() -> None:
    """_run_existing_probes marks probe as failed on timeout."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass)
    config = {
        "id": "ex-3",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Slow",
        "entity_id": "conversation.slow",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    # Raise TimeoutError immediately (no actual sleeping)
    mock_hass.services.async_call = AsyncMock(side_effect=asyncio.TimeoutError())

    await benchmarker._run_existing_probes("ex-3", config, profile)

    # All probes timed out → capability_score should still be set (0)
    assert profile.capability_score is not None


async def test_run_existing_probes_exception() -> None:
    """_run_existing_probes marks probe as failed and continues on exception."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass)
    config = {
        "id": "ex-4",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Broken",
        "entity_id": "conversation.broken",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    mock_hass.services.async_call = AsyncMock(side_effect=RuntimeError("service down"))

    await benchmarker._run_existing_probes("ex-4", config, profile)

    assert profile.capability_score is not None


async def test_run_existing_probes_missing_entity_id_raises() -> None:
    """_run_existing_probes raises ValueError when entity_id is absent."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass)
    config = {
        "id": "ex-5",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "No Entity",
        "timeout": 30,
        # entity_id intentionally omitted
    }
    profile = benchmarker.ensure_profile(config)

    with pytest.raises(ValueError, match="no entity_id configured"):
        await benchmarker._run_existing_probes("ex-5", config, profile)


# ---------------------------------------------------------------------------
# _run_ollama_probes — line 697: raises ValueError when client is None
# ---------------------------------------------------------------------------


async def test_run_ollama_probes_raises_when_client_none() -> None:
    """_run_ollama_probes raises ValueError when _get_ollama_client returns None."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass)
    config = {
        "id": "no-url-agent",
        "agent_type": AGENT_TYPE_OLLAMA,
        "agent_name": "Missing URL",
        # No URL or model — _get_ollama_client will return None
    }
    profile = benchmarker.ensure_profile(config)

    with pytest.raises(ValueError, match="missing Ollama URL or model"):
        await benchmarker._run_ollama_probes("no-url-agent", config, profile)


# ---------------------------------------------------------------------------
# _run_existing_probes — lines 807-808: probe with no user messages
# ---------------------------------------------------------------------------


async def test_run_existing_probes_skips_probe_with_no_user_messages() -> None:
    """_run_existing_probes marks probe as failed when probe has no user messages."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass)
    config = {
        "id": "ex-no-user",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "No User Msg",
        "entity_id": "conversation.test",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    # Create a probe where all messages are assistant-only (no user messages)
    no_user_probe = BenchmarkProbe(
        name="no_user_probe",
        dimension="factual",
        messages=[{"role": "assistant", "content": "I can help you"}],
        expected_contains=["help"],
        timeout_seconds=5,
    )

    # Patch BENCHMARK_PROBES with only the no-user probe
    with patch(
        "custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES",
        [no_user_probe],
    ):
        await benchmarker._run_existing_probes("ex-no-user", config, profile)

    # Profile should still be finalized (with score 0 since probe failed)
    assert profile.capability_score is not None


# ---------------------------------------------------------------------------
# _run_existing_probes — lines 845-846: malformed HA response
# ---------------------------------------------------------------------------


async def test_run_existing_probes_malformed_response_falls_back_to_empty_string() -> None:
    """_run_existing_probes handles malformed HA service response gracefully."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass)
    config = {
        "id": "ex-malformed",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Malformed",
        "entity_id": "conversation.malformed",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    # Return a response that lacks the expected speech structure
    malformed_response = {"response": {"wrong_key": {}}}
    mock_hass.services.async_call = AsyncMock(return_value=malformed_response)

    await benchmarker._run_existing_probes("ex-malformed", config, profile)

    # Profile should still be finalized
    assert profile.capability_score is not None


# ---------------------------------------------------------------------------
# debug_probes logging
# ---------------------------------------------------------------------------


async def test_run_single_ollama_probe_debug_logs_response(hass: HomeAssistant) -> None:
    """_run_single_ollama_probe emits DEBUG log when debug_probes=True."""
    benchmarker = AgentBenchmarker(hass, debug_probes=True)
    probe = _make_probe(name="factual-q", expected_exact="Paris")
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(
        return_value=OllamaResponse(
            content="Paris",
            eval_count=5,
            eval_duration_ns=500_000_000,
        )
    )

    with patch("custom_components.neuralbridge.benchmark_probe_runner._LOGGER") as mock_log:
        result = await benchmarker._run_single_ollama_probe(mock_client, probe)

    assert result.passed is True
    mock_log.debug.assert_called_once()
    call_args = mock_log.debug.call_args[0]
    assert "BENCHMARK DEBUG" in call_args[0]
    assert "factual-q" in str(call_args)


async def test_run_single_ollama_probe_no_debug_when_flag_false(hass: HomeAssistant) -> None:
    """_run_single_ollama_probe does NOT emit DEBUG log when debug_probes=False."""
    benchmarker = AgentBenchmarker(hass)  # debug_probes defaults to False
    probe = _make_probe(name="factual-q", expected_exact="Paris")
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(
        return_value=OllamaResponse(
            content="Paris",
            eval_count=5,
            eval_duration_ns=500_000_000,
        )
    )

    with patch("custom_components.neuralbridge.benchmark_probe_runner._LOGGER") as mock_log:
        await benchmarker._run_single_ollama_probe(mock_client, probe)

    mock_log.debug.assert_not_called()


async def test_run_single_ollama_probe_debug_logs_none_response(hass: HomeAssistant) -> None:
    """_run_single_ollama_probe logs at DEBUG when response is None and debug_probes=True."""
    benchmarker = AgentBenchmarker(hass, debug_probes=True)
    probe = _make_probe(name="none-resp")
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=None)

    with patch("custom_components.neuralbridge.benchmark_probe_runner._LOGGER") as mock_log:
        result = await benchmarker._run_single_ollama_probe(mock_client, probe)

    assert result.passed is False
    mock_log.debug.assert_called_once()
    call_args = mock_log.debug.call_args[0]
    assert "BENCHMARK DEBUG" in call_args[0]


async def test_run_existing_probes_debug_logs_response() -> None:
    """_run_existing_probes emits one DEBUG log per probe plus one warm-up log when debug_probes=True."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass, debug_probes=True)
    config = {
        "id": "ex-debug",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Debug Agent",
        "entity_id": "conversation.debug",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    ha_response = {"response": {"speech": {"plain": {"speech": "Paris"}}}}
    mock_hass.services.async_call = AsyncMock(return_value=ha_response)

    with patch("custom_components.neuralbridge.benchmark_probe_runner._LOGGER") as mock_log:
        await benchmarker._run_existing_probes("ex-debug", config, profile)

    # 1 warm-up log + 1 BENCHMARK DEBUG log per scored probe
    assert mock_log.debug.call_count == len(BENCHMARK_PROBES) + 1
    warm_up_call = mock_log.debug.call_args_list[0][0]
    assert "warm-up" in warm_up_call[0].lower()
    first_probe_call = mock_log.debug.call_args_list[1][0]
    assert "BENCHMARK DEBUG" in first_probe_call[0]


async def test_run_existing_probes_debug_logs_timeout() -> None:
    """_run_existing_probes logs at DEBUG when a probe times out and debug_probes=True."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass, debug_probes=True)
    config = {
        "id": "ex-timeout-dbg",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Timeout Debug",
        "entity_id": "conversation.timeout",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    probe = BenchmarkProbe(
        name="timeout_probe",
        dimension="factual",
        messages=[{"role": "user", "content": "Slow question"}],
        expected_exact="answer",
        timeout_seconds=1,
    )
    mock_hass.services.async_call = AsyncMock(side_effect=asyncio.TimeoutError)

    with (
        patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]),
        patch("custom_components.neuralbridge.benchmark_probe_runner._LOGGER") as mock_log,
    ):
        await benchmarker._run_existing_probes("ex-timeout-dbg", config, profile)

    debug_msgs = [str(call) for call in mock_log.debug.call_args_list]
    assert any("BENCHMARK DEBUG" in msg and "timed out" in msg for msg in debug_msgs)


async def test_run_existing_probes_debug_logs_exception() -> None:
    """_run_existing_probes logs at DEBUG when a probe raises and debug_probes=True."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass, debug_probes=True)
    config = {
        "id": "ex-exc-dbg",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Exc Debug",
        "entity_id": "conversation.exc",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    probe = BenchmarkProbe(
        name="exc_probe",
        dimension="factual",
        messages=[{"role": "user", "content": "Failing question"}],
        expected_exact="answer",
        timeout_seconds=5,
    )
    mock_hass.services.async_call = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]),
        patch("custom_components.neuralbridge.benchmark_probe_runner._LOGGER") as mock_log,
    ):
        await benchmarker._run_existing_probes("ex-exc-dbg", config, profile)

    debug_msgs = [str(call) for call in mock_log.debug.call_args_list]
    assert any("BENCHMARK DEBUG" in msg and "boom" in msg for msg in debug_msgs)


def test_debug_probes_flag_passes_through_to_runner(hass: HomeAssistant) -> None:
    """AgentBenchmarker forwards debug_probes=True to its BenchmarkProbeRunner."""
    benchmarker = AgentBenchmarker(hass, debug_probes=True)
    assert benchmarker._probe_runner._debug_probe_responses is True


def test_debug_probes_flag_defaults_false(hass: HomeAssistant) -> None:
    """AgentBenchmarker defaults debug_probes to False."""
    benchmarker = AgentBenchmarker(hass)
    assert benchmarker._probe_runner._debug_probe_responses is False


# ---------------------------------------------------------------------------
# BenchmarkProfile.host_group — new field
# ---------------------------------------------------------------------------


def test_profile_host_group_defaults_none() -> None:
    """BenchmarkProfile.host_group defaults to None on construction."""
    profile = BenchmarkProfile(agent_id="a", agent_name="A", agent_type=AGENT_TYPE_OLLAMA)
    assert profile.host_group is None


def test_profile_host_group_round_trips_through_dict() -> None:
    """host_group serialises via to_dict and restores via from_dict."""
    profile = BenchmarkProfile(
        agent_id="a",
        agent_name="A",
        agent_type=AGENT_TYPE_OLLAMA,
        host_group="remote:192.168.1.100:11434",
    )
    restored = BenchmarkProfile.from_dict(profile.to_dict())
    assert restored.host_group == "remote:192.168.1.100:11434"


def test_profile_host_group_missing_in_stored_dict_defaults_none() -> None:
    """from_dict tolerates missing host_group key (backwards compat)."""
    data = {"agent_id": "a", "agent_name": "A", "agent_type": AGENT_TYPE_OLLAMA}
    profile = BenchmarkProfile.from_dict(data)
    assert profile.host_group is None


def test_profile_host_group_cloud_round_trips() -> None:
    """host_group with cloud: prefix round-trips correctly."""
    profile = BenchmarkProfile(
        agent_id="b",
        agent_name="B",
        agent_type=AGENT_TYPE_INTEGRATED,
        host_group="cloud:openai_conversation",
    )
    restored = BenchmarkProfile.from_dict(profile.to_dict())
    assert restored.host_group == "cloud:openai_conversation"


# ---------------------------------------------------------------------------
# AgentBenchmarker — host classifier wiring
# ---------------------------------------------------------------------------


def test_benchmarker_creates_host_classifier(hass: HomeAssistant) -> None:
    """AgentBenchmarker.__init__ creates a BenchmarkHostClassifier."""
    benchmarker = AgentBenchmarker(hass)
    assert isinstance(benchmarker._host_classifier, BenchmarkHostClassifier)


def test_benchmarker_passes_classifier_to_scheduler(hass: HomeAssistant) -> None:
    """The BenchmarkScheduler receives the BenchmarkHostClassifier from the benchmarker."""
    benchmarker = AgentBenchmarker(hass)
    assert benchmarker._scheduler._host_classifier is benchmarker._host_classifier


async def test_run_benchmark_sets_host_group_on_profile(hass: HomeAssistant) -> None:
    """async_run_benchmark classifies and persists host_group on the profile."""
    benchmarker = AgentBenchmarker(hass)
    benchmarker._warm_up_delay_seconds = 0

    config = _ollama_agent_config("hg-agent")
    profile = benchmarker.ensure_profile(config)
    assert profile.host_group is None

    with (
        patch.object(benchmarker, "_run_ollama_probes", new_callable=AsyncMock),
        patch.object(benchmarker, "async_save", new_callable=AsyncMock),
    ):
        await benchmarker.async_run_benchmark("hg-agent", config)

    # ollama_url in the config is http://localhost:11434 → "local"
    assert profile.host_group == "local"


async def test_run_benchmark_host_group_not_reclassified_when_already_set(
    hass: HomeAssistant,
) -> None:
    """async_run_benchmark does not overwrite an already-set host_group."""
    benchmarker = AgentBenchmarker(hass)
    benchmarker._warm_up_delay_seconds = 0

    config = _ollama_agent_config("hg-agent2")
    profile = benchmarker.ensure_profile(config)
    profile.host_group = "remote:192.168.1.5:11434"  # pre-set

    with (
        patch.object(benchmarker, "_run_ollama_probes", new_callable=AsyncMock),
        patch.object(benchmarker, "async_save", new_callable=AsyncMock),
    ):
        await benchmarker.async_run_benchmark("hg-agent2", config)

    assert profile.host_group == "remote:192.168.1.5:11434"  # unchanged


# ---------------------------------------------------------------------------
# BenchmarkProbeRunner._pre_flight_contention_check
# ---------------------------------------------------------------------------


async def test_pre_flight_warns_when_other_models_are_loaded(hass: HomeAssistant) -> None:
    """_pre_flight_contention_check logs a warning when other models are loaded."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config("pf-warn")
    profile = benchmarker.ensure_profile(config)

    probe = BenchmarkProbe(
        name="pf_probe",
        dimension="factual",
        messages=[{"role": "user", "content": "Q?"}],
        expected_exact="A",
        timeout_seconds=5,
    )

    # Mock client reports another model loaded alongside the one under test
    mock_client = MagicMock()
    mock_client.chat = AsyncMock(
        return_value=MagicMock(
            content="A",
            eval_count=None,
            eval_duration_ns=None,
            prompt_eval_count=None,
            prompt_eval_duration_ns=None,
        )
    )
    mock_client.async_get_running_models = AsyncMock(
        return_value=[
            {"name": "llama3"},  # the agent being benchmarked
            {"name": "mistral:7b"},  # another model → should trigger warning
        ]
    )

    with (
        patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]),
        patch.object(benchmarker._probe_runner, "_get_ollama_client", return_value=mock_client),
        patch("custom_components.neuralbridge.benchmark_probe_runner._LOGGER") as mock_log,
    ):
        await benchmarker._run_ollama_probes("pf-warn", config, profile)

    warning_calls = [str(c) for c in mock_log.warning.call_args_list]
    assert any("mistral:7b" in msg for msg in warning_calls)


async def test_pre_flight_no_warning_when_only_current_model_loaded(
    hass: HomeAssistant,
) -> None:
    """_pre_flight_contention_check does not warn when only the current model is loaded."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config("pf-ok")
    profile = benchmarker.ensure_profile(config)

    probe = BenchmarkProbe(
        name="pf_probe2",
        dimension="factual",
        messages=[{"role": "user", "content": "Q?"}],
        expected_exact="A",
        timeout_seconds=5,
    )

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(
        return_value=MagicMock(
            content="A",
            eval_count=None,
            eval_duration_ns=None,
            prompt_eval_count=None,
            prompt_eval_duration_ns=None,
        )
    )
    # Only the current model is running
    mock_client.async_get_running_models = AsyncMock(return_value=[{"name": "llama3"}])

    with (
        patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]),
        patch.object(benchmarker._probe_runner, "_get_ollama_client", return_value=mock_client),
        patch("custom_components.neuralbridge.benchmark_probe_runner._LOGGER") as mock_log,
    ):
        await benchmarker._run_ollama_probes("pf-ok", config, profile)

    assert mock_log.warning.call_count == 0


async def test_pre_flight_no_warning_when_no_models_loaded(hass: HomeAssistant) -> None:
    """_pre_flight_contention_check does not warn when Ollama reports no loaded models."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config("pf-empty")
    profile = benchmarker.ensure_profile(config)

    probe = BenchmarkProbe(
        name="pf_probe3",
        dimension="factual",
        messages=[{"role": "user", "content": "Q?"}],
        expected_exact="A",
        timeout_seconds=5,
    )

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(
        return_value=MagicMock(
            content="A",
            eval_count=None,
            eval_duration_ns=None,
            prompt_eval_count=None,
            prompt_eval_duration_ns=None,
        )
    )
    mock_client.async_get_running_models = AsyncMock(return_value=[])

    with (
        patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]),
        patch.object(benchmarker._probe_runner, "_get_ollama_client", return_value=mock_client),
        patch("custom_components.neuralbridge.benchmark_probe_runner._LOGGER") as mock_log,
    ):
        await benchmarker._run_ollama_probes("pf-empty", config, profile)

    assert mock_log.warning.call_count == 0


async def test_pre_flight_exception_does_not_abort_probes(hass: HomeAssistant) -> None:
    """A failure in async_get_running_models does not stop the benchmark probes."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config("pf-exc")
    profile = benchmarker.ensure_profile(config)

    probe = BenchmarkProbe(
        name="pf_probe4",
        dimension="factual",
        messages=[{"role": "user", "content": "Q?"}],
        expected_exact="A",
        timeout_seconds=5,
    )

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(
        return_value=MagicMock(
            content="A",
            eval_count=None,
            eval_duration_ns=None,
            prompt_eval_count=None,
            prompt_eval_duration_ns=None,
        )
    )
    mock_client.async_get_running_models = AsyncMock(
        side_effect=RuntimeError("ps endpoint unavailable")
    )

    with (
        patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]),
        patch.object(benchmarker._probe_runner, "_get_ollama_client", return_value=mock_client),
    ):
        # Should complete without raising
        await benchmarker._run_ollama_probes("pf-exc", config, profile)

    # Probe still ran and profile was updated
    assert profile.probe_results.get("pf_probe4") is True


# ---------------------------------------------------------------------------
# BenchmarkScheduler — host_classifier semaphore gating
# ---------------------------------------------------------------------------


async def test_scheduler_with_classifier_acquires_semaphore(hass: HomeAssistant) -> None:
    """BenchmarkScheduler acquires the host-group semaphore before running probes."""
    acquired: list[bool] = []

    async def run_fn(agent_id: str, agent_config: dict) -> None:
        acquired.append(True)

    classifier = BenchmarkHostClassifier(hass)

    scheduler = BenchmarkScheduler(
        hass,
        run_benchmark_fn=run_fn,
        warm_up_delay_seconds=0,
        host_classifier=classifier,
    )
    with patch.object(classifier, "classify_agent", new=AsyncMock(return_value="local")):
        await scheduler.async_schedule_benchmark("sched-1", {"agent_type": AGENT_TYPE_OLLAMA})
        task = scheduler._pending_tasks["sched-1"]
        await task

    assert acquired == [True]


async def test_scheduler_without_classifier_still_runs(hass: HomeAssistant) -> None:
    """BenchmarkScheduler without a classifier runs the benchmark directly."""
    ran: list[str] = []

    async def run_fn(agent_id: str, agent_config: dict) -> None:
        ran.append(agent_id)

    scheduler = BenchmarkScheduler(
        hass,
        run_benchmark_fn=run_fn,
        warm_up_delay_seconds=0,
        host_classifier=None,
    )
    await scheduler.async_schedule_benchmark("no-cls", {"agent_type": AGENT_TYPE_OLLAMA})
    task = scheduler._pending_tasks["no-cls"]
    await task

    assert ran == ["no-cls"]


# ---------------------------------------------------------------------------
# Inter-probe delay and warm-up — BenchmarkProbeRunner / AgentBenchmarker
# ---------------------------------------------------------------------------


def test_inter_probe_delay_stored_in_probe_runner(hass: HomeAssistant) -> None:
    """inter_probe_delay_seconds is stored on BenchmarkProbeRunner."""
    benchmarker = AgentBenchmarker(hass, inter_probe_delay_seconds=7)
    assert benchmarker._probe_runner._inter_probe_delay_seconds == 7


def test_inter_probe_delay_defaults_to_zero(hass: HomeAssistant) -> None:
    """AgentBenchmarker defaults inter_probe_delay_seconds to 0."""
    benchmarker = AgentBenchmarker(hass)
    assert benchmarker._inter_probe_delay_seconds == 0


def test_inter_probe_delay_property_proxies_to_runner(hass: HomeAssistant) -> None:
    """AgentBenchmarker._inter_probe_delay_seconds reads and writes the runner's value."""
    benchmarker = AgentBenchmarker(hass)
    benchmarker._inter_probe_delay_seconds = 3
    assert benchmarker._probe_runner._inter_probe_delay_seconds == 3


async def test_run_ollama_probes_sends_warm_up_before_probes(hass: HomeAssistant) -> None:
    """_run_ollama_probes calls client.chat once for warm-up before scored probes."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config("wu-agent")
    profile = benchmarker.ensure_profile(config)

    probe = _make_probe(name="factual-wu", expected_exact="A")
    call_order: list[str] = []

    async def tracking_chat(messages: list) -> MagicMock:
        call_order.append("chat_called")
        return MagicMock(
            content="A",
            eval_count=None,
            eval_duration_ns=None,
            prompt_eval_count=None,
            prompt_eval_duration_ns=None,
        )

    mock_client = MagicMock()
    mock_client.chat = tracking_chat
    mock_client.async_get_running_models = AsyncMock(return_value=[])

    with (
        patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]),
        patch.object(benchmarker._probe_runner, "_get_ollama_client", return_value=mock_client),
    ):
        await benchmarker._run_ollama_probes("wu-agent", config, profile)

    # warm-up + 1 scored probe = 2 chat calls
    assert len(call_order) == 2


async def test_run_ollama_probes_warm_up_failure_ignored(hass: HomeAssistant) -> None:
    """_run_ollama_probes continues to scored probes even when warm-up chat raises."""
    benchmarker = AgentBenchmarker(hass)
    config = _ollama_agent_config("wu-fail-agent")
    profile = benchmarker.ensure_profile(config)

    probe = _make_probe(name="after-fail", expected_exact="ok")

    call_count = 0

    async def failing_then_ok(messages: list) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # warm-up request
            raise RuntimeError("model not yet ready")
        return MagicMock(
            content="ok",
            eval_count=None,
            eval_duration_ns=None,
            prompt_eval_count=None,
            prompt_eval_duration_ns=None,
        )

    mock_client = MagicMock()
    mock_client.chat = failing_then_ok
    mock_client.async_get_running_models = AsyncMock(return_value=[])

    with (
        patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]),
        patch.object(benchmarker._probe_runner, "_get_ollama_client", return_value=mock_client),
    ):
        await benchmarker._run_ollama_probes("wu-fail-agent", config, profile)

    # Scored probe ran despite warm-up failure
    assert profile.probe_results.get("after-fail") is True


async def test_run_ollama_probes_sleeps_between_probes_when_delay_nonzero(
    hass: HomeAssistant,
) -> None:
    """_run_ollama_probes calls asyncio.sleep after warm-up and between each scored probe."""
    benchmarker = AgentBenchmarker(hass, inter_probe_delay_seconds=3)
    config = _ollama_agent_config("sleep-agent")
    profile = benchmarker.ensure_profile(config)

    probe_a = _make_probe(name="p-a", expected_exact="a")
    probe_b = _make_probe(name="p-b", expected_exact="b")

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(
        side_effect=[
            MagicMock(
                content="any",
                eval_count=None,
                eval_duration_ns=None,  # warm-up
                prompt_eval_count=None,
                prompt_eval_duration_ns=None,
            ),
            MagicMock(
                content="a",
                eval_count=None,
                eval_duration_ns=None,  # probe_a
                prompt_eval_count=None,
                prompt_eval_duration_ns=None,
            ),
            MagicMock(
                content="b",
                eval_count=None,
                eval_duration_ns=None,  # probe_b
                prompt_eval_count=None,
                prompt_eval_duration_ns=None,
            ),
        ]
    )
    mock_client.async_get_running_models = AsyncMock(return_value=[])

    with (
        patch(
            "custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe_a, probe_b]
        ),
        patch.object(benchmarker._probe_runner, "_get_ollama_client", return_value=mock_client),
        patch(
            "custom_components.neuralbridge.benchmark_probe_runner.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        await benchmarker._run_ollama_probes("sleep-agent", config, profile)

    # sleep after warm-up (1) + sleep before probe_b (1) = 2 calls, each with 3 s
    assert mock_sleep.call_count == 2
    for call in mock_sleep.call_args_list:
        assert call.args[0] == 3


async def test_run_ollama_probes_no_sleep_when_delay_zero(hass: HomeAssistant) -> None:
    """_run_ollama_probes does not call asyncio.sleep when inter_probe_delay_seconds=0."""
    benchmarker = AgentBenchmarker(hass, inter_probe_delay_seconds=0)
    config = _ollama_agent_config("no-sleep-agent")
    profile = benchmarker.ensure_profile(config)

    probe_a = _make_probe(name="ns-a", expected_exact="a")
    probe_b = _make_probe(name="ns-b", expected_exact="b")

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(
        side_effect=[
            MagicMock(
                content="any",
                eval_count=None,
                eval_duration_ns=None,
                prompt_eval_count=None,
                prompt_eval_duration_ns=None,
            ),
            MagicMock(
                content="a",
                eval_count=None,
                eval_duration_ns=None,
                prompt_eval_count=None,
                prompt_eval_duration_ns=None,
            ),
            MagicMock(
                content="b",
                eval_count=None,
                eval_duration_ns=None,
                prompt_eval_count=None,
                prompt_eval_duration_ns=None,
            ),
        ]
    )
    mock_client.async_get_running_models = AsyncMock(return_value=[])

    with (
        patch(
            "custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe_a, probe_b]
        ),
        patch.object(benchmarker._probe_runner, "_get_ollama_client", return_value=mock_client),
        patch(
            "custom_components.neuralbridge.benchmark_probe_runner.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        await benchmarker._run_ollama_probes("no-sleep-agent", config, profile)

    assert mock_sleep.call_count == 0


async def test_run_existing_probes_sends_warm_up_before_probes() -> None:
    """_run_existing_probes calls the HA conversation service once for warm-up."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass)
    config = {
        "id": "ex-wu",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "WarmUp Agent",
        "entity_id": "conversation.warmup",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    probe = BenchmarkProbe(
        name="ex-wu-probe",
        dimension="factual",
        messages=[{"role": "user", "content": "Q?"}],
        expected_exact="A",
        timeout_seconds=5,
    )
    ha_response = {"response": {"speech": {"plain": {"speech": "A"}}}}
    mock_hass.services.async_call = AsyncMock(return_value=ha_response)

    with patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]):
        await benchmarker._run_existing_probes("ex-wu", config, profile)

    # warm-up + 1 scored probe = 2 calls to services.async_call
    assert mock_hass.services.async_call.call_count == 2


async def test_run_existing_probes_warm_up_failure_ignored() -> None:
    """_run_existing_probes continues to scored probes even when warm-up service raises."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass)
    config = {
        "id": "ex-wu-fail",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "WarmUp Fail Agent",
        "entity_id": "conversation.wufail",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    probe = BenchmarkProbe(
        name="ex-after-fail",
        dimension="factual",
        messages=[{"role": "user", "content": "Q?"}],
        expected_exact="A",
        timeout_seconds=5,
    )
    ha_response = {"response": {"speech": {"plain": {"speech": "A"}}}}
    call_count = 0

    async def failing_then_ok(*args: object, **kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("service unavailable during warm-up")
        return ha_response

    mock_hass.services.async_call = failing_then_ok

    with patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]):
        await benchmarker._run_existing_probes("ex-wu-fail", config, profile)

    assert profile.probe_results.get("ex-after-fail") is True


async def test_run_existing_probes_sleeps_between_probes_when_delay_nonzero() -> None:
    """_run_existing_probes calls asyncio.sleep after warm-up and between each scored probe."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass, inter_probe_delay_seconds=2)
    config = {
        "id": "ex-sleep",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "Sleep Agent",
        "entity_id": "conversation.sleep",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    probe_a = BenchmarkProbe(
        name="ex-s-a",
        dimension="factual",
        messages=[{"role": "user", "content": "Q1?"}],
        expected_exact="A1",
        timeout_seconds=5,
    )
    probe_b = BenchmarkProbe(
        name="ex-s-b",
        dimension="factual",
        messages=[{"role": "user", "content": "Q2?"}],
        expected_exact="A2",
        timeout_seconds=5,
    )
    ha_response_a = {"response": {"speech": {"plain": {"speech": "A1"}}}}
    ha_response_b = {"response": {"speech": {"plain": {"speech": "A2"}}}}
    mock_hass.services.async_call = AsyncMock(
        side_effect=[ha_response_a, ha_response_a, ha_response_b]  # warm-up + 2 probes
    )

    with (
        patch(
            "custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe_a, probe_b]
        ),
        patch(
            "custom_components.neuralbridge.benchmark_probe_runner.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        await benchmarker._run_existing_probes("ex-sleep", config, profile)

    # sleep after warm-up (1) + sleep before probe_b (1) = 2 calls, each with 2 s
    assert mock_sleep.call_count == 2
    for call in mock_sleep.call_args_list:
        assert call.args[0] == 2


async def test_run_existing_probes_no_sleep_when_delay_zero() -> None:
    """_run_existing_probes does not call asyncio.sleep when inter_probe_delay_seconds=0."""
    mock_hass = MagicMock()
    benchmarker = AgentBenchmarker(mock_hass, inter_probe_delay_seconds=0)
    config = {
        "id": "ex-nosleep",
        "agent_type": AGENT_TYPE_INTEGRATED,
        "agent_name": "No-Sleep Agent",
        "entity_id": "conversation.nosleep",
        "timeout": 30,
    }
    profile = benchmarker.ensure_profile(config)

    probe = BenchmarkProbe(
        name="ex-ns-a",
        dimension="factual",
        messages=[{"role": "user", "content": "Q?"}],
        expected_exact="A",
        timeout_seconds=5,
    )
    ha_response = {"response": {"speech": {"plain": {"speech": "A"}}}}
    mock_hass.services.async_call = AsyncMock(return_value=ha_response)

    with (
        patch("custom_components.neuralbridge.agent_benchmark.BENCHMARK_PROBES", [probe]),
        patch(
            "custom_components.neuralbridge.benchmark_probe_runner.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        await benchmarker._run_existing_probes("ex-nosleep", config, profile)

    assert mock_sleep.call_count == 0
