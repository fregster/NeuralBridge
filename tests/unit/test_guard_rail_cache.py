"""Tests for the guard_rail_cache re-export module."""

from __future__ import annotations

import custom_components.neuralbridge.guard_rail_cache as guard_rail_cache_module
from custom_components.neuralbridge.guard_rail_cache import GuardRailCache, HighStakesCache


def test_guard_rail_cache_importable() -> None:
    """GuardRailCache is importable from guard_rail_cache."""
    cache = GuardRailCache()
    assert cache is not None


def test_high_stakes_cache_importable() -> None:
    """HighStakesCache is importable from guard_rail_cache."""
    cache = HighStakesCache()
    assert cache is not None


def test_guard_rail_cache_module_exports_expected_names() -> None:
    """__all__ includes both cache classes."""
    assert "GuardRailCache" in guard_rail_cache_module.__all__
    assert "HighStakesCache" in guard_rail_cache_module.__all__
