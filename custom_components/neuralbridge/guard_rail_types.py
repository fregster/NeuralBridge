"""Shared type definitions for the guard rail system.

Kept in a standalone module so that ``guard_rail.py``, ``toxicity_checker.py``,
and ``ai_safety_checker.py`` can all import ``GuardRailResult`` without creating
circular import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardRailResult:
    """Result from a guard rail check.

    Attributes:
        is_safe: True when the content passes all active guard rail stages.
        confidence: Classifier confidence in the result (0.0-1.0).
        category: Violated category string, or None when safe.
        reason: Human-readable explanation of the violation, or None when safe.
        matched_pattern: The regex pattern that triggered the rule, if applicable.
    """

    is_safe: bool
    confidence: float
    category: str | None = None
    reason: str | None = None
    matched_pattern: str | None = None
