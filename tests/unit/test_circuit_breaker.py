"""Unit tests for the circuit breaker module."""

from __future__ import annotations

from unittest.mock import patch

from custom_components.neuralbridge.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitState:
    """Tests for CircuitState dataclass."""

    def test_default_values(self) -> None:
        """Test CircuitState initialises with correct defaults."""
        state = CircuitState()
        assert state.failure_count == 0
        assert state.tripped is False
        assert state.trip_time == 0.0

    def test_custom_values(self) -> None:
        """Test CircuitState accepts custom values."""
        state = CircuitState(failure_count=2, tripped=True, trip_time=100.0)
        assert state.failure_count == 2
        assert state.tripped is True
        assert state.trip_time == 100.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_init_defaults(self) -> None:
        """Test CircuitBreaker initialises with correct defaults."""
        cb = CircuitBreaker()
        assert cb._failure_threshold == 3
        assert cb._cooldown == 60
        assert cb._states == {}

    def test_init_custom_values(self) -> None:
        """Test CircuitBreaker accepts custom threshold and cooldown."""
        cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=120)
        assert cb._failure_threshold == 5
        assert cb._cooldown == 120

    def test_is_open_unknown_agent(self) -> None:
        """Test is_open returns False for an agent with no recorded failures."""
        cb = CircuitBreaker()
        assert cb.is_open("agent-1") is False

    def test_is_open_below_threshold(self) -> None:
        """Test is_open returns False when failure count is below threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        assert cb.is_open("agent-1") is False

    def test_trips_at_threshold(self) -> None:
        """Test circuit trips exactly at the failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        assert cb.is_open("agent-1") is False
        cb.record_failure("agent-1")
        assert cb.is_open("agent-1") is True

    def test_is_open_true_during_cooldown(self) -> None:
        """Test is_open returns True while cooldown period is active."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            cb.record_failure("agent-1")
            cb.record_failure("agent-1")
            # 0 seconds elapsed — still within 60s cooldown
            assert cb.is_open("agent-1") is True

    def test_is_open_false_after_cooldown(self) -> None:
        """Test circuit resets and allows retry after cooldown expires."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            cb.record_failure("agent-1")
            cb.record_failure("agent-1")
            assert cb.is_open("agent-1") is True

        # 61 seconds later — cooldown has passed
        with patch("time.monotonic", return_value=1061.0):
            assert cb.is_open("agent-1") is False
            # State should be reset after cooldown
            state = cb.get_state("agent-1")
            assert state.tripped is False
            assert state.failure_count == 0

    def test_record_success_resets_state(self) -> None:
        """Test recording a success clears all failure state."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        assert cb.is_open("agent-1") is True

        cb.record_success("agent-1")
        assert cb.is_open("agent-1") is False
        assert cb.get_state("agent-1").failure_count == 0

    def test_record_success_unknown_agent(self) -> None:
        """Test recording success for an unknown agent does not raise."""
        cb = CircuitBreaker()
        cb.record_success("never-seen-agent")  # Should not raise
        assert cb.is_open("never-seen-agent") is False

    def test_record_timeout_counts_as_failure(self) -> None:
        """Test timeouts increment the failure counter and can trip the circuit."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_timeout("agent-1")
        assert cb.is_open("agent-1") is False
        cb.record_timeout("agent-1")
        assert cb.is_open("agent-1") is True

    def test_multiple_agents_independent(self) -> None:
        """Test circuit state is tracked independently per agent."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("agent-A")
        cb.record_failure("agent-A")
        # agent-A is tripped; agent-B should not be affected
        assert cb.is_open("agent-A") is True
        assert cb.is_open("agent-B") is False

    def test_get_state_unknown_agent(self) -> None:
        """Test get_state returns a default CircuitState for unknown agents."""
        cb = CircuitBreaker()
        state = cb.get_state("unknown")
        assert state.failure_count == 0
        assert state.tripped is False

    def test_get_state_after_failures(self) -> None:
        """Test get_state returns the current failure count."""
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        state = cb.get_state("agent-1")
        assert state.failure_count == 2
        assert state.tripped is False

    def test_reset_single_agent(self) -> None:
        """Test reset clears state for a specific agent only."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("agent-A")
        cb.record_failure("agent-A")
        cb.record_failure("agent-B")
        assert cb.is_open("agent-A") is True

        cb.reset("agent-A")
        assert cb.is_open("agent-A") is False
        # agent-B state should be unaffected
        assert cb.get_state("agent-B").failure_count == 1

    def test_reset_unknown_agent(self) -> None:
        """Test reset on unknown agent does not raise."""
        cb = CircuitBreaker()
        cb.reset("does-not-exist")  # Should not raise

    def test_reset_all(self) -> None:
        """Test reset_all clears all agent states."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("agent-A")
        cb.record_failure("agent-A")
        cb.record_failure("agent-B")
        cb.record_failure("agent-B")
        assert cb.is_open("agent-A") is True
        assert cb.is_open("agent-B") is True

        cb.reset_all()
        assert cb.is_open("agent-A") is False
        assert cb.is_open("agent-B") is False
        assert cb._states == {}

    def test_no_double_trip(self) -> None:
        """Test additional failures after tripping don't reset the trip time."""
        cb = CircuitBreaker(failure_threshold=2)
        with patch("time.monotonic", return_value=1000.0):
            cb.record_failure("agent-1")
            cb.record_failure("agent-1")
            original_trip_time = cb.get_state("agent-1").trip_time

        with patch("time.monotonic", return_value=1010.0):
            cb.record_failure("agent-1")  # Extra failure after trip
            # Trip time should not be updated by subsequent failures
            assert cb.get_state("agent-1").trip_time == original_trip_time
