"""Protocol definition for the BenchmarkProfile storage contract.

Defines :class:`BenchmarkProfileProtocol` so that callers can program against
the interface rather than the concrete :class:`~.agent_benchmark.AgentBenchmarker`
class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .benchmark_models import BenchmarkProfile


@runtime_checkable
class BenchmarkProfileProtocol(Protocol):
    """Structural interface for benchmark profile storage operations.

    Responsibility: Define the minimal contract for managing
    :class:`~.benchmark_models.BenchmarkProfile` objects so that callers can
    remain decoupled from the concrete :class:`~.agent_benchmark.AgentBenchmarker`.
    """

    def ensure_profile(self, agent_config: dict[str, Any]) -> "BenchmarkProfile":
        """Return existing profile for *agent_config*, creating one if absent."""
        ...

    def get_profile(self, agent_id: str) -> "BenchmarkProfile | None":
        """Return the profile for *agent_id*, or ``None`` if not found."""
        ...

    def remove_profile(self, agent_id: str) -> None:
        """Remove the stored profile for *agent_id* and cancel pending tasks."""
        ...

    def cancel_pending(self, agent_id: str) -> None:
        """Cancel a pending warm-up/probe task for *agent_id* if one exists."""
        ...
