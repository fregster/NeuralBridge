"""Probe execution logic for NeuralBridge agent benchmarking.

:class:`BenchmarkProbeRunner` runs the benchmark probe suite against Ollama
agents (direct HTTP) or EXISTING integration agents (HA conversation service).

Separated from :mod:`.agent_benchmark` as part of the Stage 3a refactor.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import TYPE_CHECKING, Any, cast

from .benchmark_models import BenchmarkProfile, BenchmarkStatus, ProbeResult, _evaluate_probe
from .benchmark_probes import BENCHMARK_PROBES, BenchmarkProbe
from .const import CONF_OLLAMA_MODEL, CONF_OLLAMA_URL, CONF_TIMEOUT, DEFAULT_TIMEOUT
from .ollama_client import OllamaClient

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class BenchmarkProbeRunner:
    """Runs the benchmark probe suite against Ollama or HA conversation agents.

    Shares the ``ollama_clients`` dict with the owning
    :class:`~.agent_benchmark.AgentBenchmarker` so that client creation and
    cleanup remain centralised in the orchestrator.

    Responsibility: Execute individual probes and aggregate results into a
    :class:`~.benchmark_models.BenchmarkProfile`.
    """

    def __init__(
        self,
        hass: "HomeAssistant",
        ollama_clients: dict[str, OllamaClient],
        debug_probes: bool = False,
        inter_probe_delay_seconds: int = 5,
    ) -> None:
        """Initialise with a shared Ollama client cache.

        Args:
            hass:                    Home Assistant instance (for EXISTING integration probes).
            ollama_clients:          Shared dict of :class:`OllamaClient` objects keyed by
                                     agent ID.  Mutations are visible to the caller.
            debug_probes:            When ``True``, log each probe's raw response text and
                                     pass/fail result at WARNING level for debugging.
            inter_probe_delay_seconds: Seconds to pause between successive probe requests.
                                     Avoids saturating local Ollama servers that process
                                     one request at a time.  A warm-up request is also
                                     sent before the first scored probe to ensure the model
                                     is fully loaded.
        """
        self._hass = hass
        self._ollama_clients = ollama_clients
        self._debug_probe_responses = debug_probes
        self._inter_probe_delay_seconds = inter_probe_delay_seconds

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

    async def _pre_flight_contention_check(
        self,
        agent_id: str,
        agent_config: dict[str, Any],
        client: OllamaClient,
    ) -> None:
        """Query Ollama ``/api/ps`` and warn if other models are already loaded.

        The host-group semaphore in :class:`~.benchmark_scheduler.BenchmarkScheduler`
        prevents concurrent benchmarks of agents on the same host.  This check
        catches unexpected load from *outside* NeuralBridge (e.g. a user
        actively chatting while a benchmark runs) and logs a warning so the
        operator knows the results may not represent peak single-model
        performance.

        Args:
            agent_id:     Unique agent identifier (for log context).
            agent_config: Agent configuration dict.
            client:       The :class:`OllamaClient` to poll.
        """
        current_model: str = agent_config.get(CONF_OLLAMA_MODEL, "")
        try:
            running_models = await client.async_get_running_models()
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.debug("Pre-flight contention check failed for %s: %s", agent_id, err)
            return

        other_running = [
            m.get("name", "") for m in running_models if m.get("name") != current_model
        ]
        if other_running:
            _LOGGER.warning(
                "Benchmark pre-flight [%s]: %d other model(s) currently loaded on Ollama "
                "server: %s. Results may not reflect peak single-model performance.",
                agent_id,
                len(other_running),
                other_running,
            )

    async def _run_ollama_probes(
        self,
        agent_id: str,
        agent_config: dict[str, Any],
        profile: BenchmarkProfile,
        *,
        probes: list[BenchmarkProbe] | None = None,
        single_probe_fn: Any | None = None,
    ) -> None:
        """Run all probes against an Ollama agent and update *profile*.

        Args:
            agent_id:       Unique agent identifier.
            agent_config:   Agent configuration dict.
            profile:        The :class:`~.benchmark_models.BenchmarkProfile` to
                            update in-place.
            probes:         Override the probe list (tests inject a short list).
                            Defaults to :data:`.benchmark_probes.BENCHMARK_PROBES`.
            single_probe_fn: Override the per-probe callable ``(client, probe)
                            -> ProbeResult``.  Defaults to
                            :meth:`_run_single_ollama_probe`.  Injected by
                            :class:`~.agent_benchmark.AgentBenchmarker` so that
                            ``patch.object(benchmarker, '_run_single_ollama_probe')``
                            intercepts correctly.

        Raises:
            ValueError: If the agent is missing required Ollama configuration.
        """
        client = self._get_ollama_client(agent_id, agent_config)
        if client is None:
            raise ValueError(f"Agent {agent_id} is missing Ollama URL or model configuration")

        await self._pre_flight_contention_check(agent_id, agent_config, client)

        # Send a warm-up request before scored probes so the model is fully
        # loaded.  The result is intentionally discarded.
        await self._send_ollama_warm_up(client, agent_id)
        if self._inter_probe_delay_seconds > 0:
            await asyncio.sleep(self._inter_probe_delay_seconds)

        active_probes = probes if probes is not None else BENCHMARK_PROBES
        run_probe = (
            single_probe_fn if single_probe_fn is not None else self._run_single_ollama_probe
        )
        probe_results: dict[str, bool] = {}
        latencies: list[float] = []
        tps_samples: list[float] = []
        prompt_tps_samples: list[float] = []

        for index, probe in enumerate(active_probes):
            if index > 0 and self._inter_probe_delay_seconds > 0:
                await asyncio.sleep(self._inter_probe_delay_seconds)
            result = await run_probe(client, probe)
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

    async def _send_ollama_warm_up(self, client: OllamaClient, agent_id: str) -> None:
        """Send a discarded warm-up request to ensure the model is fully loaded.

        Many local Ollama setups load models lazily.  The first real request can
        take significantly longer than subsequent ones while the model loads from
        disk.  This method fires a minimal prompt and throws away the result so
        that all scored probes see consistent steady-state latency.

        Args:
            client:   The :class:`OllamaClient` to warm up.
            agent_id: Agent identifier used only for log context.
        """
        _LOGGER.debug("Sending warm-up request for agent %s", agent_id)
        try:
            warm_up_messages = [{"role": "user", "content": "Hello"}]
            await client.chat(warm_up_messages)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.debug("Warm-up request failed for agent %s (ignored): %s", agent_id, err)

    async def _run_single_ollama_probe(
        self, client: OllamaClient, probe: BenchmarkProbe
    ) -> ProbeResult:
        """Run one probe against the Ollama client and return a :class:`ProbeResult`.

        A timed-out probe is counted as a failure but does not stop subsequent
        probes from running.

        Args:
            client: The :class:`OllamaClient` to use.
            probe:  The probe definition.

        Returns:
            A :class:`~.benchmark_models.ProbeResult` (passed=False on timeout
            or error).
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
            if self._debug_probe_responses:
                _LOGGER.debug(
                    "[BENCHMARK DEBUG] Probe %s — response is None (API error or empty reply)",
                    probe.name,
                )
            return ProbeResult(
                probe_name=probe.name,
                dimension=probe.dimension,
                passed=False,
                latency_ms=latency,
            )

        passed = _evaluate_probe(probe, response.content)
        if self._debug_probe_responses:
            _LOGGER.debug(
                "[BENCHMARK DEBUG] Probe %s | passed: %s | response: %r",
                probe.name,
                passed,
                response.content[:500],
            )
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

    async def _run_existing_probes(
        self,
        agent_id: str,
        agent_config: dict[str, Any],
        profile: BenchmarkProfile,
        *,
        probes: list[BenchmarkProbe] | None = None,
    ) -> None:
        """Run probes against an EXISTING HA conversation entity.

        Performance metrics are latency only (no ``eval_*`` fields available
        from the HA conversation API).

        Args:
            agent_id:     Unique agent identifier.
            agent_config: Agent configuration dict (must contain ``entity_id``).
            profile:      The :class:`~.benchmark_models.BenchmarkProfile` to
                          update in-place.
            probes:       Override the probe list (used by tests for injection).
                          Defaults to the full :data:`.benchmark_probes.BENCHMARK_PROBES`.

        Raises:
            ValueError: If the agent has no ``entity_id`` configured.
        """
        from homeassistant.components.conversation.const import (  # noqa: PLC0415
            DOMAIN as CONVERSATION_DOMAIN,
        )

        entity_id: str | None = agent_config.get("entity_id")
        if not entity_id:
            raise ValueError(f"EXISTING agent {agent_id} has no entity_id configured")

        # Send a warm-up request before scored probes so the underlying model is
        # fully loaded (especially important for HA Ollama or other lazy-loading
        # integrations).  The result is intentionally discarded.
        await self._send_existing_warm_up(entity_id, agent_id, CONVERSATION_DOMAIN)
        if self._inter_probe_delay_seconds > 0:
            await asyncio.sleep(self._inter_probe_delay_seconds)

        active_probes = probes if probes is not None else BENCHMARK_PROBES
        probe_results: dict[str, bool] = {}
        latencies: list[float] = []

        for index, probe in enumerate(active_probes):
            if index > 0 and self._inter_probe_delay_seconds > 0:
                await asyncio.sleep(self._inter_probe_delay_seconds)
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
                        {"text": user_text, "agent_id": entity_id},
                        blocking=True,
                        return_response=True,
                    )
            except asyncio.TimeoutError:
                latency = (time.monotonic() - start) * 1000
                latencies.append(latency)
                probe_results[probe.name] = False
                if self._debug_probe_responses:
                    _LOGGER.debug(
                        "[BENCHMARK DEBUG] Probe %s — timed out waiting for HA agent",
                        probe.name,
                    )
                continue
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.debug("Probe %s failed for EXISTING agent: %s", probe.name, err)
                latencies.append((time.monotonic() - start) * 1000)
                probe_results[probe.name] = False
                if self._debug_probe_responses:
                    _LOGGER.debug("[BENCHMARK DEBUG] Probe %s — exception: %s", probe.name, err)
                continue

            latency = (time.monotonic() - start) * 1000
            latencies.append(latency)

            try:
                speech = cast("dict[str, Any]", response)["response"]["speech"]["plain"]["speech"]
                response_text = str(speech) if speech else ""
            except (KeyError, TypeError, AttributeError):
                response_text = ""

            probe_results[probe.name] = (
                _evaluate_probe(probe, response_text) if response_text else False
            )
            if self._debug_probe_responses:
                _LOGGER.debug(
                    "[BENCHMARK DEBUG] Probe %s | passed: %s | response: %r",
                    probe.name,
                    probe_results[probe.name],
                    response_text[:500],
                )

        self._apply_probe_results(profile, probe_results, latencies)

    async def _send_existing_warm_up(
        self, entity_id: str, agent_id: str, conversation_domain: str
    ) -> None:
        """Send a discarded warm-up request to an EXISTING HA conversation entity.

        Ensures the underlying model (e.g. HA Ollama integration) is fully loaded
        before scored probes begin, so latency measurements reflect steady-state
        performance rather than model-load overhead.

        Args:
            entity_id:           The HA conversation entity to warm up.
            agent_id:            Agent identifier used only for log context.
            conversation_domain: The HA conversation domain string.
        """
        _LOGGER.debug("Sending warm-up request for EXISTING agent %s", agent_id)
        try:
            await self._hass.services.async_call(
                conversation_domain,
                "process",
                {"text": "Hello", "agent_id": entity_id},
                blocking=True,
                return_response=True,
            )
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.debug(
                "Warm-up request failed for EXISTING agent %s (ignored): %s", agent_id, err
            )

    @staticmethod
    def _apply_probe_results(
        profile: BenchmarkProfile,
        probe_results: dict[str, bool],
        latencies: list[float],
    ) -> None:
        """Apply probe results to *profile*, computing scores and latency stats.

        Args:
            profile:       The :class:`~.benchmark_models.BenchmarkProfile` to
                           update in-place.
            probe_results: Mapping of probe name to pass/fail.
            latencies:     List of per-probe latencies in milliseconds.
        """
        from .benchmark_models import _compute_scores  # noqa: PLC0415

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
