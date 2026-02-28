"""Telemetry update and profile version-checking helpers.

These free functions are extracted from
:class:`~.agent_benchmark.AgentBenchmarker` as part of the Stage 3a refactor
so that telemetry logic has a single canonical home.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .benchmark_models import _TELEMETRY_ALPHA, BenchmarkProfile, BenchmarkStatus
from .const import BENCHMARK_PROBE_SUITE_VERSION

if TYPE_CHECKING:
    from .ollama_client import OllamaResponse

_LOGGER = logging.getLogger(__name__)


def _check_version(profile: BenchmarkProfile) -> None:
    """Invalidate *profile* in-place if its ``probe_suite_version`` is stale.

    A version of ``0`` (sentinel for "never benchmarked") is also treated as
    stale.  Metadata fields (``model_family``, ``parameter_count_billions``,
    ``quantization``, ``context_window``, ``model_size_bytes``) are preserved —
    metadata discovery does not need to re-run.

    Args:
        profile: The :class:`~.benchmark_models.BenchmarkProfile` to check and
                 possibly mutate in-place.
    """
    if profile.probe_suite_version == BENCHMARK_PROBE_SUITE_VERSION:
        return

    _LOGGER.info(
        "Agent %s benchmark invalidated: probe suite updated to v%d",
        profile.agent_name,
        BENCHMARK_PROBE_SUITE_VERSION,
    )
    profile.status = BenchmarkStatus.PENDING
    profile.re_benchmark_on_save = True
    profile.error = None
    profile.median_latency_ms = None
    profile.p95_latency_ms = None
    profile.probe_tokens_per_sec = None
    profile.probe_prompt_tps = None
    profile.score_instruction_following = None
    profile.score_reasoning = None
    profile.score_smart_home_intent = None
    profile.score_factual = None
    profile.score_memory = None
    profile.score_structured_output = None
    profile.score_creative_generation = None
    profile.score_verbosity_calibration = None
    profile.score_robustness = None
    profile.score_safety_refusal = None
    profile.capability_score = None
    profile.probe_results = {}
    # Metadata (model_family, parameter_count_billions, etc.) is preserved


def update_realworld_telemetry(profile: BenchmarkProfile, response: OllamaResponse) -> None:
    """Update the rolling real-world tokens/sec average (Layer 3).

    Uses an exponentially weighted moving average:
    ``new = 0.2 * sample + 0.8 * existing``.

    Only applied when the stored ``probe_suite_version`` matches the current
    constant — prevents mixing telemetry from different probe eras.

    Args:
        profile:  The :class:`~.benchmark_models.BenchmarkProfile` to update
                  in-place.
        response: The :class:`~.ollama_client.OllamaResponse` from real traffic.
    """
    if profile.probe_suite_version != BENCHMARK_PROBE_SUITE_VERSION:
        return

    tps = response.tokens_per_second
    if tps is None:
        return

    if profile.realworld_tokens_per_sec is None:
        profile.realworld_tokens_per_sec = tps
    else:
        profile.realworld_tokens_per_sec = (
            _TELEMETRY_ALPHA * tps + (1.0 - _TELEMETRY_ALPHA) * profile.realworld_tokens_per_sec
        )
    profile.realworld_sample_count += 1
