"""Usage statistics tracking for NeuralBridge agents.

Tracks per-agent request counts, success/failure/timeout rates, and latency.
Statistics are held in memory and reset on Home Assistant restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentStats:
    """Statistics for a single agent."""

    requests: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds across successful calls.

        Returns:
            Average latency in ms, or 0.0 if no successful calls.
        """
        if self.successes == 0:
            return 0.0
        return round(self.total_latency_ms / self.successes, 1)

    @property
    def success_rate(self) -> float:
        """Fraction of requests that succeeded (0.0–1.0).

        Returns:
            Success rate as a float, or 0.0 if no requests recorded.
        """
        if self.requests == 0:
            return 0.0
        return round(self.successes / self.requests, 3)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dict for use in sensor attributes.

        Returns:
            Dictionary with all statistics values.
        """
        return {
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
        }


class AgentStatistics:
    """Aggregated statistics for all NeuralBridge agents.

    Instances of this class are stored in ``hass.data`` and shared between the
    conversation entity (writer) and the sensor entity (reader).
    """

    def __init__(self) -> None:
        """Initialise with empty statistics."""
        self._stats: dict[str, AgentStats] = {}
        self._agent_names: dict[str, str] = {}  # agent_id -> agent_name

    def record_request(self, agent_id: str, agent_name: str) -> None:
        """Record that a routing attempt was made to an agent.

        Args:
            agent_id: Unique identifier for the agent.
            agent_name: Human-readable agent name (used as attribute key).
        """
        self._agent_names[agent_id] = agent_name
        stats = self._stats.setdefault(agent_id, AgentStats())
        stats.requests += 1

    def record_success(self, agent_id: str, latency_ms: float) -> None:
        """Record a successful agent response.

        Args:
            agent_id: Unique identifier for the agent.
            latency_ms: Round-trip latency in milliseconds.
        """
        stats = self._stats.get(agent_id)
        if stats is None:
            return
        stats.successes += 1
        stats.total_latency_ms += latency_ms

    def record_failure(self, agent_id: str) -> None:
        """Record a failed agent call (error, not timeout).

        Args:
            agent_id: Unique identifier for the agent.
        """
        stats = self._stats.get(agent_id)
        if stats is None:
            return
        stats.failures += 1

    def record_timeout(self, agent_id: str) -> None:
        """Record an agent timeout.

        Args:
            agent_id: Unique identifier for the agent.
        """
        stats = self._stats.get(agent_id)
        if stats is None:
            return
        stats.timeouts += 1

    def total_requests(self) -> int:
        """Return the total number of routing attempts across all agents.

        Returns:
            Sum of all agents' request counts.
        """
        return sum(s.requests for s in self._stats.values())

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return a serialisable dict of all agent statistics.

        Keys are agent names; values are the result of ``AgentStats.to_dict()``.

        Returns:
            Dictionary keyed by agent name with stats dicts as values.
        """
        return {
            self._agent_names.get(agent_id, agent_id): stats.to_dict()
            for agent_id, stats in self._stats.items()
        }
