"""Unit tests for benchmark_probes module."""

from __future__ import annotations

import contextlib

from custom_components.neuralbridge.benchmark_probes import (
    BENCHMARK_PROBES,
    PROBES_BY_DIMENSION,
    BenchmarkProbe,
)

# ---------------------------------------------------------------------------
# BenchmarkProbe dataclass
# ---------------------------------------------------------------------------


def test_benchmark_probe_is_frozen() -> None:
    """BenchmarkProbe instances are immutable (frozen dataclass)."""
    probe = BenchmarkProbe(
        name="test",
        dimension="instruction_following",
        messages=[{"role": "user", "content": "hi"}],
    )
    with contextlib.suppress(Exception):
        probe.name = "modified"  # type: ignore[misc]
        raise AssertionError("Expected FrozenInstanceError")


def test_benchmark_probe_optional_fields_default_none() -> None:
    """Optional fields default to None / False when not provided."""
    probe = BenchmarkProbe(
        name="x",
        dimension="factual",
        messages=[{"role": "user", "content": "q"}],
    )
    assert probe.expected_contains is None
    assert probe.expected_exact is None
    assert probe.word_count_check is False
    assert probe.min_word_count is None
    assert probe.max_word_count is None
    assert probe.starts_with is None
    assert probe.timeout_seconds == 30


# ---------------------------------------------------------------------------
# BENCHMARK_PROBES list
# ---------------------------------------------------------------------------


def test_benchmark_probes_has_twenty_seven_probes() -> None:
    """Exactly 27 probes are defined in the canonical suite (v2)."""
    assert len(BENCHMARK_PROBES) == 27


def test_benchmark_probes_all_have_names() -> None:
    """Every probe has a non-empty name and dimension."""
    for probe in BENCHMARK_PROBES:
        assert probe.name, f"Probe missing name: {probe!r}"
        assert probe.dimension, f"Probe missing dimension: {probe!r}"


def test_benchmark_probes_names_are_unique() -> None:
    """Probe names are globally unique."""
    names = [p.name for p in BENCHMARK_PROBES]
    assert len(names) == len(set(names)), "Duplicate probe names found"


def test_benchmark_probes_dimensions_within_valid_set() -> None:
    """Every probe dimension belongs to the expected set of ten dimensions."""
    valid = {
        "instruction_following",
        "reasoning",
        "smart_home_intent",
        "factual",
        "memory",
        "structured_output",
        "creative_generation",
        "verbosity_calibration",
        "robustness",
        "safety_refusal",
    }
    for probe in BENCHMARK_PROBES:
        assert probe.dimension in valid, f"Unknown dimension: {probe.dimension}"


def test_benchmark_probes_each_has_messages() -> None:
    """Every probe defines at least one message."""
    for probe in BENCHMARK_PROBES:
        assert probe.messages, f"Probe '{probe.name}' has no messages"


def test_benchmark_probes_all_have_pass_criterion() -> None:
    """Every probe has at least one pass criterion so it can produce a result."""
    for probe in BENCHMARK_PROBES:
        has_criterion = (
            probe.expected_exact is not None
            or probe.expected_contains is not None
            or probe.word_count_check
        )
        assert has_criterion, f"Probe '{probe.name}' has no pass criterion"


# ---------------------------------------------------------------------------
# PROBES_BY_DIMENSION dict
# ---------------------------------------------------------------------------


def test_probes_by_dimension_keys_match_probe_dimensions() -> None:
    """PROBES_BY_DIMENSION contains exactly the dimension keys used by BENCHMARK_PROBES."""
    dims_in_probes = {p.dimension for p in BENCHMARK_PROBES}
    assert set(PROBES_BY_DIMENSION.keys()) == dims_in_probes


def test_probes_by_dimension_values_are_probe_names() -> None:
    """Every entry in PROBES_BY_DIMENSION maps to valid probe names."""
    all_probe_names = {p.name for p in BENCHMARK_PROBES}
    for dim, names in PROBES_BY_DIMENSION.items():
        for name in names:
            assert name in all_probe_names, f"Unknown probe name '{name}' under dim '{dim}'"


def test_probes_by_dimension_covers_all_probes() -> None:
    """Every probe name appears under its dimension in PROBES_BY_DIMENSION."""
    for probe in BENCHMARK_PROBES:
        assert probe.name in PROBES_BY_DIMENSION.get(
            probe.dimension, []
        ), f"Probe '{probe.name}' missing from PROBES_BY_DIMENSION['{probe.dimension}']"
