"""Usage statistics tracking for NeuralBridge agents.

Tracks per-agent request counts, success/failure/timeout rates, and latency.
Statistics are held in memory and reset on Home Assistant restart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentStats:
    """Statistics for a single agent."""

    requests: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    total_latency_ms: float = 0.0
    blocks: int = 0
    first_request_time: float | None = None
    intent_hints: dict[str, int] = field(default_factory=dict)

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
        """Fraction of requests that succeeded (0.0-1.0).

        Returns:
            Success rate as a float, or 0.0 if no requests recorded.
        """
        if self.requests == 0:
            return 0.0
        return round(self.successes / self.requests, 3)

    @property
    def queries_per_hour(self) -> float:
        """Estimated request rate in queries per hour since the first request.

        Returns:
            Queries per hour, or 0.0 if fewer than 1 second of data exists.
        """
        if self.requests == 0 or self.first_request_time is None:
            return 0.0
        elapsed_hours = (time.monotonic() - self.first_request_time) / 3600
        if elapsed_hours < 1 / 3600:
            return 0.0
        return round(self.requests / elapsed_hours, 1)

    @property
    def block_rate(self) -> float:
        """Fraction of requests that were blocked (router agents only).

        Returns:
            Block rate as a float (0.0-1.0), or 0.0 if no requests recorded.
        """
        if self.requests == 0:
            return 0.0
        return round(self.blocks / self.requests, 3)

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
            "blocks": self.blocks,
            "block_rate": self.block_rate,
            "queries_per_hour": self.queries_per_hour,
            "intent_hints": dict(self.intent_hints),
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
        if stats.requests == 0:
            stats.first_request_time = time.monotonic()
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

    def record_block(self, agent_id: str) -> None:
        """Record that a router agent blocked a request.

        Args:
            agent_id: Unique identifier for the router agent.
        """
        stats = self._stats.get(agent_id)
        if stats is None:
            return
        stats.blocks += 1

    def record_intent_hint(self, agent_id: str, hint: str) -> None:
        """Record an intent_hint returned by a router agent (Feature 8).

        Counts how many times each hint type (timer, reminder, todo,
        shopping_list, announce) has been seen for a given router agent.

        Args:
            agent_id: Unique identifier for the router agent.
            hint: The intent_hint string (e.g. ``"timer"``, ``"todo"``).
        """
        stats = self._stats.get(agent_id)
        if stats is None:
            return
        stats.intent_hints[hint] = stats.intent_hints.get(hint, 0) + 1

    def get_agent_stats(self, agent_id: str) -> AgentStats | None:
        """Return the statistics for a specific agent.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            AgentStats instance, or None if no stats have been recorded yet.
        """
        return self._stats.get(agent_id)

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
