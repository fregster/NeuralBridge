"""Benchmark probe definitions for NeuralBridge agent capability testing.

This module is the single location for all benchmark probe definitions.
``agent_benchmark.py`` imports :data:`BENCHMARK_PROBES` and :class:`BenchmarkProbe`
from here.

**IMPORTANT:** Increment ``BENCHMARK_PROBE_SUITE_VERSION`` in ``const.py``
whenever any probe in this file is added, modified, or removed.  All stored
``BenchmarkProfile`` records will be automatically invalidated on next HA
restart, triggering a fresh benchmark run for every agent.

Current suite version: 2 (must match ``BENCHMARK_PROBE_SUITE_VERSION`` in
``const.py``).

Suite v2 expands coverage from 8 probes / 5 dimensions to 27 probes / 10
dimensions, adding routing-specific and agent-capability dimensions so the
router can make informed decisions about which model to send each request to.

New dimensions added in v2:
  - ``structured_output``     — JSON compliance without preamble (routing prerequisite)
  - ``creative_generation``   — free-form text quality (route generation tasks here)
  - ``verbosity_calibration`` — respects length/format constraints (voice assistant fit)
  - ``robustness``            — resistant to nonsense input and negation confusion
  - ``safety_refusal``        — declines clearly harmful requests without the guard rail

Expanded dimensions in v2:
  - ``reasoning``         2 → 4 probes (adds syllogism + multi-step chain)
  - ``smart_home_intent`` 1 → 5 probes (adds lock, climate, cover, media domains)
  - ``factual``           1 → 2 probes (adds science/numeric recall)
  - ``memory``            1 → 3 probes (adds entity-context faithfulness + name recall)

.. note::
   Probe definitions have been extracted into the :mod:`probes` sub-package:

   * :mod:`probes.ollama_probes` — general LLM / Ollama capability probes
   * :mod:`probes.integration_probes` — HA-integration (smart home intent) probes

   This module is now a thin façade that re-exports the assembled
   :data:`BENCHMARK_PROBES` list and the :class:`BenchmarkProbe` dataclass for
   backward compatibility.
"""

from __future__ import annotations

# Re-export the dataclass and assembled probe list from the probes sub-package.
from .probes import BENCHMARK_PROBES as BENCHMARK_PROBES  # noqa: PLC0414
from .probes import PROBES_BY_DIMENSION as PROBES_BY_DIMENSION  # noqa: PLC0414
from .probes import BenchmarkProbe as BenchmarkProbe  # noqa: PLC0414

# Probes are intentionally deterministic (expected_exact / expected_contains)
# or use bounded word-count checks (min/max_word_count) so that scoring is
# reproducible without an oracle model.
#
# Dimension breakdown:
#   instruction_following  3 probes  max  3
#   reasoning              4 probes  max  4
#   smart_home_intent      5 probes  max  5
#   factual                2 probes  max  2
#   memory                 3 probes  max  3
#   structured_output      3 probes  max  3
#   creative_generation    2 probes  max  2
#   verbosity_calibration  2 probes  max  2
#   robustness             2 probes  max  2
#   safety_refusal         1 probe   max  1
#                                  ─────────
#   TOTAL                 27 probes  max 27
#
# Changing this list requires bumping BENCHMARK_PROBE_SUITE_VERSION in const.py.
# ---------------------------------------------------------------------------
