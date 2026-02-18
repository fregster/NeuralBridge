"""Circuit breaker for NeuralBridge agent failure tracking.

Tracks consecutive failures per agent and temporarily skips agents that have
exceeded the failure threshold, preventing timeout accumulation when a service
is known-down.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

_LOGGER = logging.getLogger(__name__)


@dataclass
class CircuitState:
    """Mutable state for a single agent's circuit breaker."""

    failure_count: int = 0
    tripped: bool = False
    trip_time: float = field(default_factory=float)


class CircuitBreaker:
    """Circuit breaker that skips repeatedly-failing agents during cooldown.

    When an agent accumulates ``failure_threshold`` consecutive failures its
    circuit is *tripped*.  While tripped, ``is_open()`` returns True and callers
    should skip the agent.  After ``cooldown_seconds`` have elapsed the circuit
    resets automatically (half-open: one retry is allowed).

    A successful call always resets the agent's failure count regardless of
    whether the circuit is currently tripped.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: int = 60,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Consecutive failures before tripping (default 3).
            cooldown_seconds: Seconds to wait before allowing retry (default 60).
        """
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._states: dict[str, CircuitState] = {}

    def is_open(self, agent_id: str) -> bool:
        """Return True if the circuit is tripped and the agent should be skipped.

        After the cooldown period expires the state is reset so a retry can
        proceed (half-open behaviour).

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            True if the agent should be skipped, False otherwise.
        """
        state = self._states.get(agent_id)
        if state is None or not state.tripped:
            return False

        if time.monotonic() - state.trip_time > self._cooldown:
            _LOGGER.info(
                "Circuit breaker cooldown expired for agent %s, allowing retry",
                agent_id,
            )
            state.tripped = False
            state.failure_count = 0
            return False

        return True

    def record_success(self, agent_id: str) -> None:
        """Record a successful agent call and reset the circuit state.

        Args:
            agent_id: Unique identifier for the agent.
        """
        if agent_id in self._states:
            self._states[agent_id] = CircuitState()

    def record_failure(self, agent_id: str) -> None:
        """Record a failed agent call; trip the circuit if threshold is reached.

        Args:
            agent_id: Unique identifier for the agent.
        """
        state = self._states.setdefault(agent_id, CircuitState())
        state.failure_count += 1
        if state.failure_count >= self._failure_threshold and not state.tripped:
            state.tripped = True
            state.trip_time = time.monotonic()
            _LOGGER.warning(
                "Circuit breaker tripped for agent %s after %d failures; "
                "will retry in %d seconds",
                agent_id,
                state.failure_count,
                self._cooldown,
            )

    def record_timeout(self, agent_id: str) -> None:
        """Record an agent timeout (treated identically to a failure).

        Args:
            agent_id: Unique identifier for the agent.
        """
        self.record_failure(agent_id)

    def get_state(self, agent_id: str) -> CircuitState:
        """Return the current circuit state for an agent.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            The CircuitState for the agent (default state if not seen before).
        """
        return self._states.get(agent_id, CircuitState())

    def reset(self, agent_id: str) -> None:
        """Manually reset the circuit state for a single agent.

        Args:
            agent_id: Unique identifier for the agent.
        """
        self._states.pop(agent_id, None)

    def reset_all(self) -> None:
        """Manually reset all circuit states."""
        self._states.clear()
