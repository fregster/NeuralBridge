"""Benchmark probe sub-package for NeuralBridge.

This package centralises all benchmark probe definitions.  The probe list is
assembled here from two sub-modules:

* :mod:`probes.ollama_probes` — general LLM capability probes (instruction,
  reasoning, factual, memory, structured output, creative, verbosity,
  robustness, safety)
* :mod:`probes.integration_probes` — HA-integration capability probes
  (smart home intent)

Public re-exports — import from here or from the top-level
``benchmark_probes`` façade, which is the canonical public interface::

    from custom_components.neuralbridge.benchmark_probes import (
        BenchmarkProbe,
        BENCHMARK_PROBES,
        PROBES_BY_DIMENSION,
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProbe:
    """A single benchmark probe definition.

    Attributes:
        name:              Unique probe identifier (e.g. ``"instruction_exact"``).
        dimension:         Capability dimension this probe tests (e.g.
                           ``"instruction_following"``, ``"reasoning"``,
                           ``"smart_home_intent"``, ``"factual"``, ``"memory"``).
        messages:          Chat-format message list sent to the agent.  Each
                           entry is a ``{"role": ..., "content": ...}`` dict.
        expected_contains: List of substrings that must ALL appear in the
                           (lowercased) response for the probe to pass.
                           ``None`` means this check is not applied.
        expected_exact:    The entire stripped response (lowercased) must equal
                           this string for the probe to pass.
                           ``None`` means this check is not applied.
        word_count_check:  When ``True``, the stripped response must be exactly
                           one word (used by the conciseness probe).
        min_word_count:    When set, the stripped response must contain at least
                           this many whitespace-delimited words.
                           ``None`` means this check is not applied.
        max_word_count:    When set, the stripped response must contain no more
                           than this many whitespace-delimited words.
                           ``None`` means this check is not applied.
        starts_with:       When set, the stripped (lowercased) response must begin
                           with this string (case-insensitive).  Used by
                           structured-output probes to enforce preamble-free JSON.
                           ``None`` means this check is not applied.
        timeout_seconds:   Per-probe wall-clock timeout.  The probe is counted
                           as a failure if the agent exceeds this budget.
    """

    name: str
    dimension: str
    messages: list[dict[str, str]]
    expected_contains: list[str] | None = None
    expected_exact: str | None = None
    word_count_check: bool = False
    min_word_count: int | None = None
    max_word_count: int | None = None
    starts_with: str | None = None
    timeout_seconds: int = 30


# BenchmarkProbe is now defined; sub-modules can safely import it from this package.
from .integration_probes import INTEGRATION_PROBES  # noqa: E402
from .ollama_probes import OLLAMA_PROBES  # noqa: E402

# Full probe suite assembled from sub-modules.
# Probe ordering: Ollama / general LLM probes first, then HA integration probes.
BENCHMARK_PROBES: list[BenchmarkProbe] = OLLAMA_PROBES + INTEGRATION_PROBES

# Mapping from dimension name to the list of probe names in that dimension.
# Computed once at module load to avoid repeated iteration.
PROBES_BY_DIMENSION: dict[str, list[str]] = {}
for _probe in BENCHMARK_PROBES:
    PROBES_BY_DIMENSION.setdefault(_probe.dimension, []).append(_probe.name)

__all__ = [
    "BENCHMARK_PROBES",
    "INTEGRATION_PROBES",
    "OLLAMA_PROBES",
    "PROBES_BY_DIMENSION",
    "BenchmarkProbe",
]
