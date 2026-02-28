"""Data models for agent benchmark profiling.

Contains the data classes, enums, NamedTuples, and pure functions that form
the domain model for NeuralBridge agent benchmarking.

Separated from :mod:`.agent_benchmark` as part of the Stage 3a god-class
refactor to keep each module focused on a single responsibility.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, NamedTuple

from .benchmark_probes import PROBES_BY_DIMENSION

_LOGGER = logging.getLogger(__name__)

# Rolling weighted average factor for passive telemetry (Layer 3)
_TELEMETRY_ALPHA: float = 0.2  # new sample weight


class ScoreBreakdown(NamedTuple):
    """Per-dimension and aggregate capability scores from a single benchmark run.

    Returned by :func:`_compute_scores` and consumed by
    :meth:`~.agent_benchmark.AgentBenchmarker._apply_probe_results`.
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
    by :func:`.benchmark_telemetry._check_version`.
    """

    # Identity
    agent_id: str
    agent_name: str
    agent_type: str
    #: Host-group string assigned by :class:`~.benchmark_host_classifier.BenchmarkHostClassifier`.
    #: ``None`` until the first benchmark run classifies the agent.
    #: Values: ``"local"``, ``"remote:<host>:<port>"``, or ``"cloud:<domain>"``.
    host_group: str | None = None

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
        raw_status = data.get("status", BenchmarkStatus.PENDING.value)
        try:
            status = BenchmarkStatus(raw_status)
        except ValueError:
            status = BenchmarkStatus.PENDING

        return cls(
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            agent_type=data.get("agent_type", ""),
            host_group=data.get("host_group"),
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


def _evaluate_probe(probe: Any, response_text: str) -> bool:
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
        probe:         The :class:`~.benchmark_probes.BenchmarkProbe` definition.
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
        probe_results: Mapping of probe name to pass (``True``) / fail
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
