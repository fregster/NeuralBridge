"""Agent benchmark / profiling constants for NeuralBridge (Feature 14).

Covers: probe suite versioning, warm-up and inter-probe delays,
storage keys, HA events and services, and the hass.data key.

NOTE: CONF_AGENT_RE_BENCHMARK is intentionally absent — re_benchmark_on_save
is owned exclusively by BenchmarkProfile in HA storage, never by the agent
config dict.  See config flow for rationale.
"""

from __future__ import annotations

from typing import Final

from .core import DOMAIN

# ── Probe suite versioning ────────────────────────────────────────────────────
# Increment whenever any probe in benchmark_probes.py is added, modified, or
# removed.  All stored profiles whose probe_suite_version does not match will be
# automatically invalidated and re-queued on next HA start.  Never decrement.
BENCHMARK_PROBE_SUITE_VERSION: Final = 2

# ── Benchmark configuration keys ─────────────────────────────────────────────
CONF_BENCHMARK_WARM_UP_DELAY: Final = "benchmark_warm_up_delay"
DEFAULT_BENCHMARK_WARM_UP_DELAY: Final = 60  # seconds
CONF_BENCHMARK_INTER_PROBE_DELAY: Final = "benchmark_inter_probe_delay"
DEFAULT_BENCHMARK_INTER_PROBE_DELAY: Final = 5  # seconds between each probe request
CONF_BENCHMARK_DEBUG_LOGGING: Final = "benchmark_debug_logging"
DEFAULT_BENCHMARK_DEBUG_LOGGING: Final = False

# ── Storage ───────────────────────────────────────────────────────────────────
BENCHMARK_STORAGE_KEY: Final = f"{DOMAIN}.benchmark"
BENCHMARK_STORAGE_VERSION: Final = 1

# ── HA events ─────────────────────────────────────────────────────────────────
EVENT_BENCHMARK_STARTED: Final = f"{DOMAIN}_benchmark_started"
EVENT_BENCHMARK_COMPLETE: Final = f"{DOMAIN}_benchmark_complete"
EVENT_BENCHMARK_FAILED: Final = f"{DOMAIN}_benchmark_failed"

# ── HA services ───────────────────────────────────────────────────────────────
SERVICE_RUN_BENCHMARK: Final = "run_benchmark"

# ── hass.data sub-key ─────────────────────────────────────────────────────────
DATA_BENCHMARKER: Final = "benchmarker"
