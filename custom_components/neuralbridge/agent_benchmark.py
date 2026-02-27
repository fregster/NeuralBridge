"""Agent benchmark profiling for NeuralBridge.

Implements three complementary layers of agent capability scoring:

* **Layer 1** — Metadata discovery from the Ollama ``/api/show`` endpoint
  (one-shot, near-instant, no inference required).
* **Layer 2** — Active probe suite: 27 pre-defined prompts across 10 capability
  dimensions, with deterministic pass/fail criteria, run asynchronously in the
  background.  Scores inform the router about each agent's strengths so it can
  direct requests to the most capable available model.
* **Layer 3** — Passive telemetry: rolling weighted average of real-traffic
  ``eval_*`` counters from every successful Ollama call.

All data is stored in HA storage under ``neuralbridge.benchmark`` and exposed
as sensor attributes on :class:`NeuralBridgeAgentSensor`.

See ``benchmark_probes.py`` for probe definitions and
``const.py`` for ``BENCHMARK_PROBE_SUITE_VERSION``.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from .benchmark_probes import BENCHMARK_PROBES, PROBES_BY_DIMENSION, BenchmarkProbe
from .const import (
    AGENT_TYPE_INTEGRATED,
    AGENT_TYPE_LOCAL_HA,
    AGENT_TYPE_OLLAMA,
    AGENT_TYPE_WEB_SEARCH,
    BENCHMARK_PROBE_SUITE_VERSION,
    BENCHMARK_STORAGE_KEY,
    BENCHMARK_STORAGE_VERSION,
    CONF_AGENT_NAME,
    CONF_AGENT_TYPE,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_TIMEOUT,
    DEFAULT_TIMEOUT,
    EVENT_BENCHMARK_COMPLETE,
    EVENT_BENCHMARK_FAILED,
    EVENT_BENCHMARK_STARTED,
)
from .ollama_client import OllamaClient, OllamaResponse
from .persistent_store import AsyncPersistentStore

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Rolling weighted average factor for passive telemetry (Layer 3)
_TELEMETRY_ALPHA: float = 0.2  # new sample weight


class ScoreBreakdown(NamedTuple):
    """Per-dimension and aggregate capability scores from a single benchmark run.

    Returned by :func:`_compute_scores` and consumed by
    :meth:`AgentBenchmarker._apply_probe_results`.
    """

    score_instruction_following: int
    score_reasoning: int
    score_smart_home_intent: int
    score_factual: int
    score_memory: int
    score_structured_output: int
    score_creative_generation: int
    score_verbosity_calibration: int
    score_robustness: int
    score_safety_refusal: int
    capability_score: int


class BenchmarkStatus(StrEnum):
    """Lifecycle status of a :class:`BenchmarkProfile`."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProbeResult:
    """Result of a single benchmark probe execution.

    Response text is deliberately excluded — probes contain no PII but we
    establish the principle of not persisting model outputs.

    Attributes:
        probe_name:             Name of the probe (e.g. ``"math_basic"``).
        dimension:              Capability dimension tested.
        passed:                 Whether the probe's pass criteria were met.
        latency_ms:             Wall-clock time to receive the response.
        eval_count:             Output token count (Ollama only).
        eval_duration_ns:       Output generation duration in nanoseconds
                                (Ollama only).
        prompt_eval_count:      Prompt token count (Ollama only).
        prompt_eval_duration_ns: Prompt evaluation duration in nanoseconds
                                 (Ollama only).
    """

    probe_name: str
    dimension: str
    passed: bool
    latency_ms: float
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None


@dataclass
class BenchmarkProfile:
    """Persisted capability and performance profile for a single agent.

    All fields are serialisable to JSON via :meth:`to_dict` / :meth:`from_dict`
    for storage in Home Assistant's ``Store``.

    The ``probe_suite_version`` field records which version of the probe
    definitions was used for the scores.  When the stored version does not
    match :attr:`BENCHMARK_PROBE_SUITE_VERSION` the profile is invalidated
    by :meth:`AgentBenchmarker._check_version`.
    """

    # Identity
    agent_id: str
    agent_name: str
    agent_type: str

    # Lifecycle
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    re_benchmark_on_save: bool = True
    probe_suite_version: int = 0  # 0 = sentinel for "never benchmarked"

    # Timing
    benchmark_timestamp: float | None = None
    benchmark_duration_ms: float | None = None
    error: str | None = None

    # Layer 1 — model metadata (Ollama only)
    model_name: str | None = None
    model_family: str | None = None
    parameter_count_billions: float | None = None
    quantization: str | None = None
    context_window: int | None = None
    model_size_bytes: int | None = None

    # Layer 2 — active probe results
    median_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    probe_tokens_per_sec: float | None = None  # Ollama only
    probe_prompt_tps: float | None = None  # Ollama only

    # Layer 2 — capability scores
    score_instruction_following: int | None = None  # 0-3
    score_reasoning: int | None = None  # 0-4
    score_smart_home_intent: int | None = None  # 0-5
    score_factual: int | None = None  # 0-2
    score_memory: int | None = None  # 0-3
    score_structured_output: int | None = None  # 0-3
    score_creative_generation: int | None = None  # 0-2
    score_verbosity_calibration: int | None = None  # 0-2
    score_robustness: int | None = None  # 0-2
    score_safety_refusal: int | None = None  # 0-1
    capability_score: int | None = None  # 0-27 (sum of all)

    # Layer 3 — passive telemetry (continuous enrichment)
    realworld_tokens_per_sec: float | None = None
    realworld_sample_count: int = 0

    # Raw probe pass/fail map (no response text stored)
    probe_results: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict for HA storage.

        Returns:
            Dictionary with all fields; ``BenchmarkStatus`` stored as its
            string value.
        """
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkProfile:
        """Deserialise from a stored dict, tolerating missing optional fields.

        Args:
            data: Dict loaded from HA storage (may be missing newer fields).

        Returns:
            A :class:`BenchmarkProfile` with defaults applied for any field
            not present in *data*.
        """
        # Convert stored status string back to enum; default to PENDING on unknown
        raw_status = data.get("status", BenchmarkStatus.PENDING.value)
        try:
            status = BenchmarkStatus(raw_status)
        except ValueError:
            status = BenchmarkStatus.PENDING

        return cls(
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            agent_type=data.get("agent_type", ""),
            status=status,
            re_benchmark_on_save=data.get("re_benchmark_on_save", True),
            probe_suite_version=data.get("probe_suite_version", 0),
            benchmark_timestamp=data.get("benchmark_timestamp"),
            benchmark_duration_ms=data.get("benchmark_duration_ms"),
            error=data.get("error"),
            model_name=data.get("model_name"),
            model_family=data.get("model_family"),
            parameter_count_billions=data.get("parameter_count_billions"),
            quantization=data.get("quantization"),
            context_window=data.get("context_window"),
            model_size_bytes=data.get("model_size_bytes"),
            median_latency_ms=data.get("median_latency_ms"),
            p95_latency_ms=data.get("p95_latency_ms"),
            probe_tokens_per_sec=data.get("probe_tokens_per_sec"),
            probe_prompt_tps=data.get("probe_prompt_tps"),
            score_instruction_following=data.get("score_instruction_following"),
            score_reasoning=data.get("score_reasoning"),
            score_smart_home_intent=data.get("score_smart_home_intent"),
            score_factual=data.get("score_factual"),
            score_memory=data.get("score_memory"),
            score_structured_output=data.get("score_structured_output"),
            score_creative_generation=data.get("score_creative_generation"),
            score_verbosity_calibration=data.get("score_verbosity_calibration"),
            score_robustness=data.get("score_robustness"),
            score_safety_refusal=data.get("score_safety_refusal"),
            capability_score=data.get("capability_score"),
            realworld_tokens_per_sec=data.get("realworld_tokens_per_sec"),
            realworld_sample_count=data.get("realworld_sample_count", 0),
            probe_results=data.get("probe_results", {}),
        )


def _evaluate_probe(probe: BenchmarkProbe, response_text: str) -> bool:
    """Evaluate whether an agent response satisfies a probe's pass criteria.

    Applies the six optional criteria in order:

    1. ``expected_exact`` — stripped, lowercased response must equal the
       target string exactly.
    2. ``expected_contains`` — all substrings in the list must appear in the
       lowercased response.
    3. ``word_count_check`` — the stripped response must be a single word.
    4. ``min_word_count`` — the stripped response must have at least this many
       whitespace-delimited words.
    5. ``max_word_count`` — the stripped response must have no more than this
       many whitespace-delimited words.
    6. ``starts_with`` — the stripped (lowercased) response must begin with
       this string (case-insensitive).

    All active criteria must pass simultaneously.

    Args:
        probe:         The :class:`BenchmarkProbe` definition.
        response_text: Raw text returned by the agent for this probe.

    Returns:
        ``True`` if all active criteria pass, ``False`` otherwise.
    """
    text = response_text.strip()
    lower = text.lower()
    word_count = len(text.split())

    if probe.expected_exact is not None and lower != probe.expected_exact.lower():
        return False

    if probe.expected_contains is not None:
        for substr in probe.expected_contains:
            if substr.lower() not in lower:
                return False

    if probe.word_count_check and word_count != 1:
        return False

    if probe.min_word_count is not None and word_count < probe.min_word_count:
        return False

    if probe.max_word_count is not None and word_count > probe.max_word_count:
        return False

    return probe.starts_with is None or lower.startswith(probe.starts_with.lower())


def _compute_scores(
    probe_results: dict[str, bool],
) -> ScoreBreakdown:
    """Compute per-dimension and aggregate capability scores from probe results.

    Args:
        probe_results: Mapping of probe name → pass (``True``) / fail
                       (``False``).

    Returns:
        A :class:`ScoreBreakdown` named tuple with one field per dimension and
        a ``capability_score`` total.
    """

    def _sum_dimension(dimension: str) -> int:
        names = PROBES_BY_DIMENSION.get(dimension, [])
        return sum(1 for n in names if probe_results.get(n, False))

    inst = _sum_dimension("instruction_following")
    reasoning = _sum_dimension("reasoning")
    intent = _sum_dimension("smart_home_intent")
    factual = _sum_dimension("factual")
    memory = _sum_dimension("memory")
    structured = _sum_dimension("structured_output")
    creative = _sum_dimension("creative_generation")
    verbosity = _sum_dimension("verbosity_calibration")
    robustness = _sum_dimension("robustness")
    safety = _sum_dimension("safety_refusal")
    total = (
        inst
        + reasoning
        + intent
        + factual
        + memory
        + structured
        + creative
        + verbosity
        + robustness
        + safety
    )
    return ScoreBreakdown(
        score_instruction_following=inst,
        score_reasoning=reasoning,
        score_smart_home_intent=intent,
        score_factual=factual,
        score_memory=memory,
        score_structured_output=structured,
        score_creative_generation=creative,
        score_verbosity_calibration=verbosity,
        score_robustness=robustness,
        score_safety_refusal=safety,
        capability_score=total,
    )


class AgentBenchmarker(AsyncPersistentStore[dict[str, Any]]):
    """Manages benchmark profiles for all configured NeuralBridge agents.

    Profiles are persisted via HA's :class:`~homeassistant.helpers.storage.Store`
    under the key ``neuralbridge.benchmark``.  On load, stale profiles (whose
    stored ``probe_suite_version`` does not match
    ``BENCHMARK_PROBE_SUITE_VERSION``) are automatically invalidated so that
    changed probe definitions always trigger a fresh run.

    Args:
        hass:                 Home Assistant instance.
        warm_up_delay_seconds: Seconds to sleep before running probes on a
                               newly registered agent.  Allows Ollama to load
                               the model into memory before the first probe
                               fires.
    """

    def __init__(self, hass: "HomeAssistant", warm_up_delay_seconds: int = 60) -> None:
        """Initialise the benchmarker.

        Args:
            hass:                  Home Assistant instance.
            warm_up_delay_seconds: Warm-up delay before first probe run.
        """
        super().__init__(hass, BENCHMARK_STORAGE_KEY, BENCHMARK_STORAGE_VERSION)
        self._hass = hass
        self._warm_up_delay_seconds = warm_up_delay_seconds
        self._profiles: dict[str, BenchmarkProfile] = {}
        self._pending_tasks: dict[str, asyncio.Task[None]] = {}
        self._ollama_clients: dict[str, OllamaClient] = {}

    # ------------------------------------------------------------------
    # Persistence — AsyncPersistentStore implementation
    # ------------------------------------------------------------------

    def _deserialise(self, data: dict[str, Any]) -> None:
        """Populate _profiles from raw stored data, invalidating stale entries.

        Args:
            data: The raw dictionary loaded from storage.
        """
        for agent_id, profile_data in data.items():
            profile = BenchmarkProfile.from_dict(profile_data)
            self._check_version(profile)
            self._profiles[agent_id] = profile

    def _serialise(self) -> dict[str, Any]:
        """Return a dict of all current profiles serialised for storage."""
        return {agent_id: p.to_dict() for agent_id, p in self._profiles.items()}

    # ------------------------------------------------------------------
    # Public API — storage
    # ------------------------------------------------------------------

    def ensure_profile(self, agent_config: dict[str, Any]) -> BenchmarkProfile:
        """Return the existing profile for *agent_config*, creating one if absent.

        Idempotent — calling this multiple times for the same agent ID is safe.

        Args:
            agent_config: Agent configuration dict (must contain ``"id"``).

        Returns:
            The :class:`BenchmarkProfile` for this agent.
        """
        agent_id: str = agent_config.get("id", "")
        if agent_id not in self._profiles:
            self._profiles[agent_id] = BenchmarkProfile(
                agent_id=agent_id,
                agent_name=agent_config.get(CONF_AGENT_NAME, ""),
                agent_type=agent_config.get(CONF_AGENT_TYPE, ""),
                model_name=agent_config.get(CONF_OLLAMA_MODEL),
                status=BenchmarkStatus.PENDING,
                re_benchmark_on_save=True,
                probe_suite_version=0,
            )
        return self._profiles[agent_id]

    def remove_profile(self, agent_id: str) -> None:
        """Remove the stored profile for *agent_id*.

        Also cancels any pending warm-up task for this agent.

        Args:
            agent_id: The unique agent identifier.
        """
        self.cancel_pending(agent_id)
        self._profiles.pop(agent_id, None)
        client = self._ollama_clients.pop(agent_id, None)
        if client is not None:
            self._hass.async_create_task(client.close())

    def get_profile(self, agent_id: str) -> BenchmarkProfile | None:
        """Return the profile for *agent_id*, or ``None`` if not found.

        Args:
            agent_id: The unique agent identifier.

        Returns:
            The :class:`BenchmarkProfile`, or ``None``.
        """
        return self._profiles.get(agent_id)

    def cancel_pending(self, agent_id: str) -> None:
        """Cancel a pending warm-up/probe task for *agent_id* if one exists.

        Args:
            agent_id: The unique agent identifier.
        """
        task = self._pending_tasks.pop(agent_id, None)
        if task is not None and not task.done():
            task.cancel()

    def cancel_all_pending(self) -> None:
        """Cancel all pending warm-up/probe tasks (called on integration unload)."""
        for agent_id in list(self._pending_tasks):
            self.cancel_pending(agent_id)

    # ------------------------------------------------------------------
    # Public API — benchmarking
    # ------------------------------------------------------------------

    async def async_run_metadata(self, agent_id: str, agent_config: dict[str, Any]) -> None:
        """Fetch Ollama model metadata (Layer 1) for the given agent.

        No-op for non-Ollama agent types.

        Args:
            agent_id:     The unique agent identifier.
            agent_config: Agent configuration dict.
        """
        if agent_config.get(CONF_AGENT_TYPE) != AGENT_TYPE_OLLAMA:
            return

        profile = self.ensure_profile(agent_config)
        client = self._get_ollama_client(agent_id, agent_config)
        if client is None:
            return

        try:
            data = await client.async_show_model()
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Metadata discovery failed for agent %s: %s", agent_id, err)
            return

        if data is None:
            return

        details = data.get("details", {})
        model_info = data.get("modelinfo", {})

        profile.model_family = details.get("family")
        profile.quantization = details.get("quantization_level")
        profile.model_size_bytes = data.get("size")

        # Parameter count — try to parse "7B" / "0.6B" style strings
        raw_params = details.get("parameter_size")
        if isinstance(raw_params, str):
            raw_params = raw_params.upper().replace("B", "").strip()
            import contextlib  # noqa: PLC0415

            with contextlib.suppress(ValueError):
                profile.parameter_count_billions = float(raw_params)
        elif isinstance(raw_params, (int, float)):
            profile.parameter_count_billions = float(raw_params)

        ctx = model_info.get("general.context_length")
        if isinstance(ctx, int):
            profile.context_window = ctx

        await self.async_save()

    async def async_schedule_benchmark(self, agent_id: str, agent_config: dict[str, Any]) -> None:
        """Schedule a warm-up then probe run as a background asyncio task.

        The task sleeps for ``self._warm_up_delay_seconds`` before probing.
        This gives Ollama time to load the model into memory.

        Args:
            agent_id:     The unique agent identifier.
            agent_config: Agent configuration dict.
        """
        self.cancel_pending(agent_id)

        async def _warm_up_then_probe() -> None:
            try:
                await asyncio.sleep(self._warm_up_delay_seconds)
                await self.async_run_benchmark(agent_id, agent_config)
            except asyncio.CancelledError:
                _LOGGER.debug("Benchmark warm-up cancelled for agent %s", agent_id)
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.error("Benchmark warm-up task error for agent %s: %s", agent_id, err)

        task = self._hass.async_create_task(_warm_up_then_probe())
        self._pending_tasks[agent_id] = task

    async def async_run_benchmark(self, agent_id: str, agent_config: dict[str, Any]) -> None:
        """Run the full probe suite for the specified agent.

        Dispatches to the correct implementation based on agent type:

        * ``AGENT_TYPE_OLLAMA`` — direct HTTP probes with performance metrics.
        * ``AGENT_TYPE_INTEGRATED`` — probes via HA ``conversation.process``
          service; latency only (no ``eval_*`` counters available).
        * ``AGENT_TYPE_LOCAL_HA`` / ``AGENT_TYPE_WEB_SEARCH`` — immediately
          skipped; scores are either meaningless or misleading for these types.

        Args:
            agent_id:     The unique agent identifier.
            agent_config: Agent configuration dict.
        """
        profile = self.ensure_profile(agent_config)
        agent_name: str = profile.agent_name
        agent_type: str = profile.agent_type

        # LOCAL_HA and WEB_SEARCH are skipped — see docstring rationale
        if agent_type in (AGENT_TYPE_LOCAL_HA, AGENT_TYPE_WEB_SEARCH):
            profile.status = BenchmarkStatus.SKIPPED
            profile.re_benchmark_on_save = False
            await self.async_save()
            _LOGGER.debug("Agent %s (%s) benchmark skipped", agent_name, agent_type)
            return

        # Stamp the version BEFORE any probe fires — ensures a profile saved
        # mid-run (interrupted shutdown) still matches the current suite version
        # on next load, but status=RUNNING/FAILED will trigger a re-queue.
        profile.probe_suite_version = BENCHMARK_PROBE_SUITE_VERSION
        profile.status = BenchmarkStatus.RUNNING
        profile.error = None
        await self.async_save()

        self._hass.bus.async_fire(
            EVENT_BENCHMARK_STARTED,
            {"agent_id": agent_id, "agent_name": agent_name, "agent_type": agent_type},
        )

        start_total = time.monotonic()
        try:
            if agent_type == AGENT_TYPE_OLLAMA:
                await self._run_ollama_probes(agent_id, agent_config, profile)
            elif agent_type == AGENT_TYPE_INTEGRATED:
                await self._run_existing_probes(agent_id, agent_config, profile)
            else:
                _LOGGER.warning("Unknown agent type for benchmark: %s", agent_type)
                profile.status = BenchmarkStatus.FAILED
                profile.error = f"Unknown agent type: {agent_type}"
                await self.async_save()
                return

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Benchmark failed for agent %s: %s", agent_name, err)
            profile.status = BenchmarkStatus.FAILED
            profile.error = str(err)
            profile.re_benchmark_on_save = True
            await self.async_save()
            self._hass.bus.async_fire(
                EVENT_BENCHMARK_FAILED,
                {"agent_id": agent_id, "agent_name": agent_name, "error": str(err)},
            )
            return

        profile.benchmark_duration_ms = (time.monotonic() - start_total) * 1000
        profile.benchmark_timestamp = time.time()

        self._hass.bus.async_fire(
            EVENT_BENCHMARK_COMPLETE,
            {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "capability_score": profile.capability_score,
                "median_latency_ms": profile.median_latency_ms,
                "probe_tokens_per_sec": profile.probe_tokens_per_sec,
                "probe_suite_version": profile.probe_suite_version,
                "status": profile.status.value,
            },
        )
        await self.async_save()

    def update_realworld_telemetry(self, agent_id: str, response: OllamaResponse) -> None:
        """Update the rolling real-world tokens/sec average (Layer 3).

        Uses an exponentially weighted moving average:
        ``new = 0.2 * sample + 0.8 * existing``.

        Only applied when the stored ``probe_suite_version`` matches the
        current constant — prevents mixing telemetry from different probe eras.

        Args:
            agent_id: The unique agent identifier.
            response: The :class:`OllamaResponse` from the real traffic call.
        """
        profile = self._profiles.get(agent_id)
        if profile is None:
            return

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

    # ------------------------------------------------------------------
    # Private helpers — version checking
    # ------------------------------------------------------------------

    @staticmethod
    def _check_version(profile: BenchmarkProfile) -> None:
        """Invalidate *profile* in-place if its probe_suite_version is stale.

        Pure function: no I/O, safe to call in ``async_load`` and in tests.

        A version of ``0`` (the sentinel for "never benchmarked despite being
        stored") is also treated as stale.

        Metadata fields (``model_family``, ``parameter_count_billions``,
        ``quantization``, ``context_window``, ``model_size_bytes``) are
        **preserved** — metadata discovery does not need to re-run.

        Args:
            profile: The :class:`BenchmarkProfile` to check and possibly
                     mutate in-place.
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
        # Clear all performance and score fields
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

    # ------------------------------------------------------------------
    # Private helpers — Ollama probes
    # ------------------------------------------------------------------

    def _get_ollama_client(
        self, agent_id: str, agent_config: dict[str, Any]
    ) -> OllamaClient | None:
        """Return a cached Ollama client for *agent_id*, creating one if needed.

        Returns ``None`` when required configuration is missing.

        Args:
            agent_id:     The unique agent identifier.
            agent_config: Agent configuration dict.

        Returns:
            :class:`OllamaClient` or ``None``.
        """
        ollama_url: str = agent_config.get(CONF_OLLAMA_URL, "")
        ollama_model: str = agent_config.get(CONF_OLLAMA_MODEL, "")
        timeout: int = agent_config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
        if not ollama_url or not ollama_model:
            return None
        if agent_id not in self._ollama_clients:
            self._ollama_clients[agent_id] = OllamaClient(ollama_url, ollama_model, timeout)
        return self._ollama_clients[agent_id]

    async def _run_ollama_probes(
        self,
        agent_id: str,
        agent_config: dict[str, Any],
        profile: BenchmarkProfile,
    ) -> None:
        """Run all probes against an Ollama agent and update *profile*.

        Args:
            agent_id:     Unique agent identifier.
            agent_config: Agent configuration dict.
            profile:      The :class:`BenchmarkProfile` to update in-place.
        """
        client = self._get_ollama_client(agent_id, agent_config)
        if client is None:
            raise ValueError(f"Agent {agent_id} is missing Ollama URL or model configuration")

        probe_results: dict[str, bool] = {}
        latencies: list[float] = []
        tps_samples: list[float] = []
        prompt_tps_samples: list[float] = []

        for probe in BENCHMARK_PROBES:
            result = await self._run_single_ollama_probe(client, probe)
            probe_results[probe.name] = result.passed
            latencies.append(result.latency_ms)
            if result.eval_count is not None and result.eval_duration_ns:
                tps = result.eval_count / (result.eval_duration_ns / 1e9)
                tps_samples.append(tps)
            if result.prompt_eval_count is not None and result.prompt_eval_duration_ns:
                ptps = result.prompt_eval_count / (result.prompt_eval_duration_ns / 1e9)
                prompt_tps_samples.append(ptps)

        self._apply_probe_results(profile, probe_results, latencies)
        if tps_samples:
            profile.probe_tokens_per_sec = statistics.median(tps_samples)
        if prompt_tps_samples:
            profile.probe_prompt_tps = statistics.median(prompt_tps_samples)

    async def _run_single_ollama_probe(
        self, client: OllamaClient, probe: BenchmarkProbe
    ) -> ProbeResult:
        """Run one probe against the Ollama client and return a :class:`ProbeResult`.

        Handles per-probe timeout: a timed-out probe is counted as a failure
        but does not prevent subsequent probes from running.

        Args:
            client: The :class:`OllamaClient` to use.
            probe:  The probe definition.

        Returns:
            A :class:`ProbeResult` (passed=False on timeout or error).
        """
        start = time.monotonic()
        try:
            async with asyncio.timeout(probe.timeout_seconds):
                response = await client.chat(probe.messages)
        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            _LOGGER.debug("Probe %s timed out after %ds", probe.name, probe.timeout_seconds)
            return ProbeResult(
                probe_name=probe.name,
                dimension=probe.dimension,
                passed=False,
                latency_ms=latency,
            )

        latency = (time.monotonic() - start) * 1000

        if response is None:
            return ProbeResult(
                probe_name=probe.name,
                dimension=probe.dimension,
                passed=False,
                latency_ms=latency,
            )

        passed = _evaluate_probe(probe, response.content)
        return ProbeResult(
            probe_name=probe.name,
            dimension=probe.dimension,
            passed=passed,
            latency_ms=latency,
            eval_count=response.eval_count,
            eval_duration_ns=response.eval_duration_ns,
            prompt_eval_count=response.prompt_eval_count,
            prompt_eval_duration_ns=response.prompt_eval_duration_ns,
        )

    # ------------------------------------------------------------------
    # Private helpers — EXISTING integration probes
    # ------------------------------------------------------------------

    async def _run_existing_probes(
        self,
        agent_id: str,
        agent_config: dict[str, Any],
        profile: BenchmarkProfile,
    ) -> None:
        """Run all probes against an EXISTING integration agent via HA conversation service.

        Performance metrics are latency only (no ``eval_*`` fields from HA
        conversation API).

        Args:
            agent_id:     Unique agent identifier.
            agent_config: Agent configuration dict.
            profile:      The :class:`BenchmarkProfile` to update in-place.
        """
        from homeassistant.components.conversation.const import (  # noqa: PLC0415
            DOMAIN as CONVERSATION_DOMAIN,
        )

        entity_id: str | None = agent_config.get("entity_id")
        if not entity_id:
            raise ValueError(f"EXISTING agent {agent_id} has no entity_id configured")

        probe_results: dict[str, bool] = {}
        latencies: list[float] = []

        for probe in BENCHMARK_PROBES:
            # Build a single-turn prompt from the last user message in the probe
            user_messages = [m for m in probe.messages if m.get("role") == "user"]
            if not user_messages:
                probe_results[probe.name] = False
                continue

            user_text = user_messages[-1]["content"]
            timeout = probe.timeout_seconds
            start = time.monotonic()

            try:
                async with asyncio.timeout(timeout):
                    response = await self._hass.services.async_call(
                        CONVERSATION_DOMAIN,
                        "process",
                        {
                            "text": user_text,
                            "agent_id": entity_id,
                            "conversation_id": None,
                        },
                        blocking=True,
                        return_response=True,
                    )
            except asyncio.TimeoutError:
                latency = (time.monotonic() - start) * 1000
                latencies.append(latency)
                probe_results[probe.name] = False
                continue
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.error("Probe %s failed for EXISTING agent: %s", probe.name, err)
                latencies.append((time.monotonic() - start) * 1000)
                probe_results[probe.name] = False
                continue

            latency = (time.monotonic() - start) * 1000
            latencies.append(latency)

            # Extract speech text
            try:
                speech = cast("dict[str, Any]", response)["response"]["speech"]["plain"]["speech"]
                response_text = str(speech) if speech else ""
            except (KeyError, TypeError, AttributeError):
                response_text = ""

            probe_results[probe.name] = (
                _evaluate_probe(probe, response_text) if response_text else False
            )

        self._apply_probe_results(profile, probe_results, latencies)
        # probe_tokens_per_sec and probe_prompt_tps remain None for EXISTING agents

    # ------------------------------------------------------------------
    # Private helpers — shared score application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_probe_results(
        profile: BenchmarkProfile,
        probe_results: dict[str, bool],
        latencies: list[float],
    ) -> None:
        """Apply probe results to *profile*, computing scores and latency stats.

        Args:
            profile:       The :class:`BenchmarkProfile` to update in-place.
            probe_results: Mapping of probe name → pass/fail.
            latencies:     List of per-probe latencies in milliseconds.
        """
        profile.probe_results = probe_results

        scores = _compute_scores(probe_results)
        profile.score_instruction_following = scores.score_instruction_following
        profile.score_reasoning = scores.score_reasoning
        profile.score_smart_home_intent = scores.score_smart_home_intent
        profile.score_factual = scores.score_factual
        profile.score_memory = scores.score_memory
        profile.score_structured_output = scores.score_structured_output
        profile.score_creative_generation = scores.score_creative_generation
        profile.score_verbosity_calibration = scores.score_verbosity_calibration
        profile.score_robustness = scores.score_robustness
        profile.score_safety_refusal = scores.score_safety_refusal
        profile.capability_score = scores.capability_score

        if latencies:
            profile.median_latency_ms = statistics.median(latencies)
            profile.p95_latency_ms = (
                float(sorted(latencies)[int(len(latencies) * 0.95)])
                if len(latencies) > 1
                else latencies[0]
            )

        profile.status = BenchmarkStatus.COMPLETE
        profile.re_benchmark_on_save = False
