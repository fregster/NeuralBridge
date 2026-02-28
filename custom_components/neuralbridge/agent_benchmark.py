"""Agent benchmark profiling for NeuralBridge.

Thin orchestrator module.  The heavy lifting is delegated to:

* :mod:`.benchmark_models` — data classes, enums, pure scoring functions.
* :mod:`.benchmark_probe_runner` — probe execution against Ollama / HA agents.
* :mod:`.benchmark_scheduler` — async warm-up and task scheduling.
* :mod:`.benchmark_telemetry` — version invalidation and real-world EWMA.
* :mod:`.benchmark_profile_store` — :class:`BenchmarkProfileProtocol`.

All previously public symbols are re-exported here for backward compatibility.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from .benchmark_host_classifier import (
    BenchmarkHostClassifier as BenchmarkHostClassifier,  # noqa: PLC0414
)
from .benchmark_models import (
    BenchmarkProfile as BenchmarkProfile,  # noqa: PLC0414
)
from .benchmark_models import (
    BenchmarkStatus as BenchmarkStatus,  # noqa: PLC0414
)
from .benchmark_models import (
    ProbeResult,
)
from .benchmark_models import (
    ScoreBreakdown as ScoreBreakdown,  # noqa: PLC0414
)
from .benchmark_models import (
    _compute_scores as _compute_scores,  # noqa: PLC0414
)
from .benchmark_models import (
    _evaluate_probe as _evaluate_probe,  # noqa: PLC0414
)
from .benchmark_probe_runner import BenchmarkProbeRunner
from .benchmark_probes import BENCHMARK_PROBES as BENCHMARK_PROBES  # noqa: PLC0414
from .benchmark_profile_store import (
    BenchmarkProfileProtocol as BenchmarkProfileProtocol,  # noqa: PLC0414
)
from .benchmark_scheduler import BenchmarkScheduler
from .benchmark_telemetry import (
    _check_version as _check_version_fn,
)
from .benchmark_telemetry import (
    update_realworld_telemetry as _update_telemetry_fn,
)
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
    EVENT_BENCHMARK_COMPLETE,
    EVENT_BENCHMARK_FAILED,
    EVENT_BENCHMARK_STARTED,
)
from .ollama_client import OllamaClient, OllamaResponse  # noqa: TC001
from .persistent_store import AsyncPersistentStore

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class AgentBenchmarkerProtocol(Protocol):
    """Structural interface for the benchmark orchestrator.

    Responsibility: Declare the public API contract for
    :class:`AgentBenchmarker` so callers can type-hint against the Protocol
    rather than the concrete class.
    """

    def ensure_profile(self, agent_config: dict[str, Any]) -> BenchmarkProfile:
        """Return existing profile for *agent_config*, creating one if absent."""
        ...

    def get_profile(self, agent_id: str) -> BenchmarkProfile | None:
        """Return the profile for *agent_id*, or ``None`` if not found."""
        ...

    def remove_profile(self, agent_id: str) -> None:
        """Remove the stored profile and cancel any pending task."""
        ...

    def cancel_pending(self, agent_id: str) -> None:
        """Cancel a pending benchmark task for *agent_id* if one exists."""
        ...

    def cancel_all_pending(self) -> None:
        """Cancel all pending benchmark tasks."""
        ...

    async def async_run_benchmark(self, agent_id: str, agent_config: dict[str, Any]) -> None:
        """Run the full probe suite for the specified agent."""
        ...

    def update_realworld_telemetry(self, agent_id: str, response: OllamaResponse) -> None:
        """Update the rolling real-world tokens/sec average (Layer 3)."""
        ...


class AgentBenchmarker(AsyncPersistentStore[dict[str, Any]]):
    """Manages benchmark profiles for all configured NeuralBridge agents.

    Profiles are persisted via HA's ``Store`` under ``neuralbridge.benchmark``.
    On load, stale profiles whose ``probe_suite_version`` does not match
    ``BENCHMARK_PROBE_SUITE_VERSION`` are automatically invalidated.

    Args:
        hass:                  Home Assistant instance.
        warm_up_delay_seconds: Seconds to sleep before probes fire.
    """

    def __init__(
        self,
        hass: "HomeAssistant",
        warm_up_delay_seconds: int = 60,
        debug_probes: bool = False,
        inter_probe_delay_seconds: int = 0,
    ) -> None:
        """Initialise the benchmarker.

        Args:
            hass:                       Home Assistant instance.
            warm_up_delay_seconds:      Warm-up delay before the first probe run.
            debug_probes:               When ``True``, log each probe's raw response and result
                                        at WARNING level so scores can be diagnosed without
                                        enabling component-level debug logging.
            inter_probe_delay_seconds:  Seconds to pause between successive probe requests.
                                        Prevents saturating single-threaded Ollama servers.
                                        A warm-up request is also sent before the first scored
                                        probe to ensure the model is fully loaded.
                                        Defaults to 0 (no delay) when not configured via HA;
                                        :mod:`.__init__` passes the configured value (default
                                        ``DEFAULT_BENCHMARK_INTER_PROBE_DELAY``) in production.
        """
        super().__init__(hass, BENCHMARK_STORAGE_KEY, BENCHMARK_STORAGE_VERSION)
        self._hass = hass
        self._profiles: dict[str, BenchmarkProfile] = {}
        self._ollama_clients: dict[str, OllamaClient] = {}
        self._probe_runner = BenchmarkProbeRunner(
            hass,
            self._ollama_clients,
            debug_probes=debug_probes,
            inter_probe_delay_seconds=inter_probe_delay_seconds,
        )
        self._host_classifier = BenchmarkHostClassifier(hass)
        # Use a lambda for late binding so patch.object on async_run_benchmark works in tests.
        self._scheduler = BenchmarkScheduler(
            hass,
            run_benchmark_fn=lambda aid, cfg: self.async_run_benchmark(aid, cfg),  # noqa: PLW0108
            warm_up_delay_seconds=warm_up_delay_seconds,
            host_classifier=self._host_classifier,
        )

    # ------------------------------------------------------------------
    # Backward-compat property — tests access _pending_tasks directly
    # ------------------------------------------------------------------

    @property
    def _pending_tasks(self) -> dict[str, Any]:
        """Expose the scheduler's pending-task dict for backward compatibility."""
        return self._scheduler._pending_tasks

    @property
    def _warm_up_delay_seconds(self) -> int:
        """Expose the scheduler\'s warm-up delay for backward compatibility.

        Tests set ``benchmarker._warm_up_delay_seconds = 0`` to skip the delay,
        so this property forwards reads and writes to the scheduler.
        """
        return self._scheduler._warm_up_delay_seconds

    @_warm_up_delay_seconds.setter
    def _warm_up_delay_seconds(self, value: int) -> None:
        self._scheduler._warm_up_delay_seconds = value

    @property
    def _inter_probe_delay_seconds(self) -> int:
        """Expose the probe runner's inter-probe delay for backward compatibility."""
        return self._probe_runner._inter_probe_delay_seconds

    @_inter_probe_delay_seconds.setter
    def _inter_probe_delay_seconds(self, value: int) -> None:
        self._probe_runner._inter_probe_delay_seconds = value

    # ------------------------------------------------------------------
    # Persistence — AsyncPersistentStore implementation
    # ------------------------------------------------------------------

    def _deserialise(self, data: dict[str, Any]) -> None:
        """Populate ``_profiles`` from raw stored data, invalidating stale entries.

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
    # Public API — profile storage
    # ------------------------------------------------------------------

    def ensure_profile(self, agent_config: dict[str, Any]) -> BenchmarkProfile:
        """Return the existing profile for *agent_config*, creating one if absent.

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

        Also cancels any pending warm-up task and closes the Ollama client.

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
        self._scheduler.cancel_pending(agent_id)

    def cancel_all_pending(self) -> None:
        """Cancel all pending warm-up/probe tasks (called on integration unload)."""
        self._scheduler.cancel_all_pending()

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

        raw_params = details.get("parameter_size")
        if isinstance(raw_params, str):
            raw_params = raw_params.upper().replace("B", "").strip()
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

        Args:
            agent_id:     The unique agent identifier.
            agent_config: Agent configuration dict.
        """
        await self._scheduler.async_schedule_benchmark(agent_id, agent_config)

    async def async_run_benchmark(self, agent_id: str, agent_config: dict[str, Any]) -> None:
        """Run the full probe suite for the specified agent.

        Args:
            agent_id:     The unique agent identifier.
            agent_config: Agent configuration dict.
        """
        profile = self.ensure_profile(agent_config)
        agent_name: str = profile.agent_name
        agent_type: str = profile.agent_type

        if agent_type in (AGENT_TYPE_LOCAL_HA, AGENT_TYPE_WEB_SEARCH):
            profile.status = BenchmarkStatus.SKIPPED
            profile.re_benchmark_on_save = False
            await self.async_save()
            _LOGGER.debug("Agent %s (%s) benchmark skipped", agent_name, agent_type)
            return

        # Classify and persist the host group if not yet known.
        if profile.host_group is None:
            profile.host_group = await self._host_classifier.classify_agent(agent_config)

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

        Args:
            agent_id: The unique agent identifier.
            response: The :class:`OllamaResponse` from real traffic.
        """
        profile = self._profiles.get(agent_id)
        if profile is None:
            return
        _update_telemetry_fn(profile, response)

    # ------------------------------------------------------------------
    # Private helpers — delegation wrappers for testability
    # Tests patch these methods on AgentBenchmarker directly, so they must
    # exist on this class and call through to the probe runner.
    # ------------------------------------------------------------------

    def _get_ollama_client(
        self, agent_id: str, agent_config: dict[str, Any]
    ) -> OllamaClient | None:
        """Return a cached Ollama client for *agent_id*.

        Delegates to :meth:`.benchmark_probe_runner.BenchmarkProbeRunner._get_ollama_client`.

        Args:
            agent_id:     The unique agent identifier.
            agent_config: Agent configuration dict.

        Returns:
            :class:`OllamaClient` or ``None``.
        """
        return self._probe_runner._get_ollama_client(agent_id, agent_config)

    async def _run_ollama_probes(
        self,
        agent_id: str,
        agent_config: dict[str, Any],
        profile: BenchmarkProfile,
    ) -> None:
        """Run all probes against an Ollama agent.

        Delegates to :meth:`.benchmark_probe_runner.BenchmarkProbeRunner._run_ollama_probes`,
        passing the module-level :data:`BENCHMARK_PROBES` so tests can patch it.

        Args:
            agent_id:     Unique agent identifier.
            agent_config: Agent configuration dict.
            profile:      The :class:`BenchmarkProfile` to update in-place.
        """
        await self._probe_runner._run_ollama_probes(
            agent_id,
            agent_config,
            profile,
            probes=BENCHMARK_PROBES,
            single_probe_fn=self._run_single_ollama_probe,
        )

    async def _run_single_ollama_probe(
        self,
        client: OllamaClient,
        probe: Any,
    ) -> ProbeResult:
        """Run one probe against an Ollama client.

        Delegates to
        :meth:`.benchmark_probe_runner.BenchmarkProbeRunner._run_single_ollama_probe`.

        Args:
            client: The :class:`OllamaClient` to use.
            probe:  The probe definition.

        Returns:
            A :class:`ProbeResult`.
        """
        return await self._probe_runner._run_single_ollama_probe(client, probe)

    async def _run_existing_probes(
        self,
        agent_id: str,
        agent_config: dict[str, Any],
        profile: BenchmarkProfile,
    ) -> None:
        """Run all probes against an EXISTING HA conversation entity.

        Delegates to
        :meth:`.benchmark_probe_runner.BenchmarkProbeRunner._run_existing_probes`,
        passing the module-level :data:`BENCHMARK_PROBES` so tests can patch it.

        Args:
            agent_id:     Unique agent identifier.
            agent_config: Agent configuration dict.
            profile:      The :class:`BenchmarkProfile` to update in-place.
        """
        await self._probe_runner._run_existing_probes(
            agent_id, agent_config, profile, probes=BENCHMARK_PROBES
        )

    # ------------------------------------------------------------------
    # Static method delegation wrappers — kept for backward compatibility.
    # Tests call _check_version and _apply_probe_results on AgentBenchmarker.
    # ------------------------------------------------------------------

    @staticmethod
    def _check_version(profile: BenchmarkProfile) -> None:
        """Invalidate *profile* in-place if its probe_suite_version is stale.

        Delegates to :func:`.benchmark_telemetry._check_version`.

        Args:
            profile: The :class:`BenchmarkProfile` to check and possibly mutate.
        """
        _check_version_fn(profile)

    @staticmethod
    def _apply_probe_results(
        profile: BenchmarkProfile,
        probe_results: dict[str, bool],
        latencies: list[float],
    ) -> None:
        """Apply probe results to *profile*, computing scores and latency stats.

        Delegates to
        :meth:`.benchmark_probe_runner.BenchmarkProbeRunner._apply_probe_results`.

        Args:
            profile:       The :class:`BenchmarkProfile` to update in-place.
            probe_results: Mapping of probe name to pass/fail.
            latencies:     List of per-probe latencies in milliseconds.
        """
        BenchmarkProbeRunner._apply_probe_results(profile, probe_results, latencies)
