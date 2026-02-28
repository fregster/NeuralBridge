"""Benchmark scheduling logic for NeuralBridge.

:class:`BenchmarkScheduler` owns the ``_pending_tasks`` dict and manages
warm-up delays before probe runs fire.  Extracted from
:class:`~.agent_benchmark.AgentBenchmarker` as part of Stage 3a to keep
scheduling concerns separate from orchestration.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from homeassistant.core import HomeAssistant

    from .benchmark_host_classifier import BenchmarkHostClassifier

_LOGGER = logging.getLogger(__name__)


class BenchmarkScheduler:
    """Schedules benchmark warm-up and probe tasks as background asyncio tasks.

    Owns the ``_pending_tasks`` dict; callers access it via
    :attr:`~.agent_benchmark.AgentBenchmarker._pending_tasks` which is exposed
    as a forwarding property on the orchestrator for backward compatibility.

    Responsibility: Manage delayed execution of benchmark runs without
    blocking the HA event loop.
    """

    def __init__(
        self,
        hass: "HomeAssistant",
        run_benchmark_fn: Callable[[str, dict[str, Any]], Awaitable[None]],
        warm_up_delay_seconds: int = 60,
        host_classifier: "BenchmarkHostClassifier | None" = None,
    ) -> None:
        """Initialise the scheduler.

        Args:
            hass:                  Home Assistant instance.
            run_benchmark_fn:      Async callable ``(agent_id, agent_config) -> None``
                                   invoked after the warm-up delay.
            warm_up_delay_seconds: Seconds to sleep before the probe run fires.
            host_classifier:       Optional :class:`~.benchmark_host_classifier
                                   .BenchmarkHostClassifier`.  When provided,
                                   each benchmark run acquires the per-host
                                   semaphore before executing so that agents on
                                   the same physical host run in series.
        """
        self._hass = hass
        self._run_benchmark_fn = run_benchmark_fn
        self._warm_up_delay_seconds = warm_up_delay_seconds
        self._host_classifier = host_classifier
        self._pending_tasks: dict[str, asyncio.Task[None]] = {}

    async def async_schedule_benchmark(self, agent_id: str, agent_config: dict[str, Any]) -> None:
        """Schedule a warm-up then probe run as a background asyncio task.

        Cancels any existing pending task for *agent_id* before scheduling so
        only one task is ever active per agent at a time.

        When a :class:`~.benchmark_host_classifier.BenchmarkHostClassifier` was
        provided at construction, the probe run acquires the per-host semaphore
        before executing.  This serialises benchmarks for agents that share a
        physical host (local Ollama, HA-integrated Ollama) while allowing cloud
        agents (ChatGPT, Gemini, etc.) to run concurrently.

        Args:
            agent_id:     The unique agent identifier.
            agent_config: Agent configuration dict forwarded to
                          :attr:`run_benchmark_fn`.
        """
        self.cancel_pending(agent_id)

        async def _warm_up_then_probe() -> None:
            try:
                await asyncio.sleep(self._warm_up_delay_seconds)
                if self._host_classifier is not None:
                    host_group = await self._host_classifier.classify_agent(agent_config)
                    semaphore = self._host_classifier.get_semaphore(host_group)
                    async with semaphore:
                        await self._run_benchmark_fn(agent_id, agent_config)
                else:
                    await self._run_benchmark_fn(agent_id, agent_config)
            except asyncio.CancelledError:
                _LOGGER.debug("Benchmark warm-up cancelled for agent %s", agent_id)
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.error("Benchmark warm-up task error for agent %s: %s", agent_id, err)

        task = self._hass.async_create_task(_warm_up_then_probe())
        self._pending_tasks[agent_id] = task

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
