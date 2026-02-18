"""Unit tests for the statistics module."""

from __future__ import annotations

from custom_components.neuralbridge.statistics import AgentStatistics, AgentStats


class TestAgentStats:
    """Tests for AgentStats dataclass."""

    def test_default_values(self) -> None:
        """Test AgentStats initialises with zero counters."""
        stats = AgentStats()
        assert stats.requests == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.timeouts == 0
        assert stats.total_latency_ms == 0.0

    def test_avg_latency_no_successes(self) -> None:
        """Test avg_latency_ms returns 0.0 when there are no successes."""
        stats = AgentStats(successes=0, total_latency_ms=500.0)
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_with_successes(self) -> None:
        """Test avg_latency_ms computes correctly."""
        stats = AgentStats(successes=4, total_latency_ms=1000.0)
        assert stats.avg_latency_ms == 250.0

    def test_avg_latency_rounding(self) -> None:
        """Test avg_latency_ms is rounded to 1 decimal place."""
        stats = AgentStats(successes=3, total_latency_ms=100.0)
        assert stats.avg_latency_ms == 33.3

    def test_success_rate_no_requests(self) -> None:
        """Test success_rate returns 0.0 when no requests have been made."""
        stats = AgentStats()
        assert stats.success_rate == 0.0

    def test_success_rate_all_success(self) -> None:
        """Test success_rate is 1.0 when all requests succeed."""
        stats = AgentStats(requests=5, successes=5)
        assert stats.success_rate == 1.0

    def test_success_rate_partial(self) -> None:
        """Test success_rate reflects partial success correctly."""
        stats = AgentStats(requests=4, successes=3)
        assert stats.success_rate == 0.75

    def test_to_dict_keys(self) -> None:
        """Test to_dict returns all expected keys."""
        stats = AgentStats(requests=10, successes=8, failures=1, timeouts=1, total_latency_ms=800.0)
        result = stats.to_dict()
        assert set(result.keys()) == {
            "requests",
            "successes",
            "failures",
            "timeouts",
            "avg_latency_ms",
            "success_rate",
        }

    def test_to_dict_values(self) -> None:
        """Test to_dict values match computed properties."""
        stats = AgentStats(requests=10, successes=8, failures=1, timeouts=1, total_latency_ms=800.0)
        result = stats.to_dict()
        assert result["requests"] == 10
        assert result["successes"] == 8
        assert result["failures"] == 1
        assert result["timeouts"] == 1
        assert result["avg_latency_ms"] == 100.0
        assert result["success_rate"] == 0.8


class TestAgentStatistics:
    """Tests for AgentStatistics aggregator."""

    def test_initial_state(self) -> None:
        """Test AgentStatistics starts empty."""
        agg = AgentStatistics()
        assert agg.total_requests() == 0
        assert agg.get_all() == {}

    def test_record_request_increments_count(self) -> None:
        """Test recording a request increments the request counter."""
        agg = AgentStatistics()
        agg.record_request("id-1", "Llama3")
        assert agg.total_requests() == 1

    def test_record_request_stores_agent_name(self) -> None:
        """Test recording a request stores the agent name for attribute keys."""
        agg = AgentStatistics()
        agg.record_request("id-1", "Llama3")
        agg.record_success("id-1", 300.0)
        result = agg.get_all()
        assert "Llama3" in result

    def test_record_success_updates_latency(self) -> None:
        """Test recording a success updates success count and latency."""
        agg = AgentStatistics()
        agg.record_request("id-1", "Llama3")
        agg.record_success("id-1", 200.0)
        result = agg.get_all()
        assert result["Llama3"]["successes"] == 1
        assert result["Llama3"]["avg_latency_ms"] == 200.0

    def test_record_success_unknown_agent(self) -> None:
        """Test recording success for an unknown agent does not raise."""
        agg = AgentStatistics()
        agg.record_success("unknown", 100.0)  # Should not raise

    def test_record_failure_increments_counter(self) -> None:
        """Test recording a failure increments the failure counter."""
        agg = AgentStatistics()
        agg.record_request("id-1", "Llama3")
        agg.record_failure("id-1")
        result = agg.get_all()
        assert result["Llama3"]["failures"] == 1

    def test_record_failure_unknown_agent(self) -> None:
        """Test recording failure for an unknown agent does not raise."""
        agg = AgentStatistics()
        agg.record_failure("unknown")  # Should not raise

    def test_record_timeout_increments_counter(self) -> None:
        """Test recording a timeout increments the timeout counter."""
        agg = AgentStatistics()
        agg.record_request("id-1", "Llama3")
        agg.record_timeout("id-1")
        result = agg.get_all()
        assert result["Llama3"]["timeouts"] == 1

    def test_record_timeout_unknown_agent(self) -> None:
        """Test recording timeout for an unknown agent does not raise."""
        agg = AgentStatistics()
        agg.record_timeout("unknown")  # Should not raise

    def test_total_requests_multiple_agents(self) -> None:
        """Test total_requests sums across all agents."""
        agg = AgentStatistics()
        agg.record_request("id-1", "Agent A")
        agg.record_request("id-1", "Agent A")
        agg.record_request("id-2", "Agent B")
        assert agg.total_requests() == 3

    def test_get_all_multiple_agents(self) -> None:
        """Test get_all returns a dict with an entry per agent."""
        agg = AgentStatistics()
        agg.record_request("id-1", "Llama3")
        agg.record_request("id-2", "ChatGPT")
        result = agg.get_all()
        assert "Llama3" in result
        assert "ChatGPT" in result

    def test_get_all_uses_latest_name(self) -> None:
        """Test get_all uses the most recently recorded agent name."""
        agg = AgentStatistics()
        agg.record_request("id-1", "Old Name")
        agg.record_request("id-1", "New Name")
        result = agg.get_all()
        assert "New Name" in result
        assert "Old Name" not in result

    def test_multiple_successes_accumulate_latency(self) -> None:
        """Test avg_latency_ms correctly averages multiple success latencies."""
        agg = AgentStatistics()
        agg.record_request("id-1", "Llama3")
        agg.record_success("id-1", 100.0)
        agg.record_request("id-1", "Llama3")
        agg.record_success("id-1", 300.0)
        result = agg.get_all()
        assert result["Llama3"]["avg_latency_ms"] == 200.0
        assert result["Llama3"]["successes"] == 2
