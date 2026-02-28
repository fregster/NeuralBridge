"""Tests for ToxicityChecker — stages 2 & 3 of the guard rail pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from custom_components.neuralbridge.guard_rail_types import GuardRailResult
from custom_components.neuralbridge.toxicity_checker import ToxicityChecker


class TestToxicityCheckerAsyncInitialize:
    """Tests for ToxicityChecker.async_initialize."""

    @pytest.mark.asyncio
    async def test_async_initialize_sets_initialized_flag(self) -> None:
        """async_initialize sets _initialized=True after running."""
        checker = ToxicityChecker(use_detoxify=False)
        assert checker._initialized is False

        await checker.async_initialize()

        assert checker._initialized is True

    @pytest.mark.asyncio
    async def test_async_initialize_is_idempotent(self) -> None:
        """Calling async_initialize a second time is a no-op."""
        checker = ToxicityChecker(use_detoxify=False)
        await checker.async_initialize()

        # Mark that init ran once; second call should skip the body.
        first_profanity = checker._profanity_available
        checker._profanity_available = not first_profanity  # flip to detect second run

        await checker.async_initialize()

        # Should still be the flipped value — the body was NOT re-entered.
        assert checker._profanity_available is not first_profanity

    @pytest.mark.asyncio
    async def test_async_initialize_without_better_profanity(self) -> None:
        """When better-profanity is not installed, _profanity_available is False."""
        checker = ToxicityChecker(use_detoxify=False)

        with patch.object(checker, "_init_profanity", return_value=False):
            await checker.async_initialize()

        assert checker._profanity_available is False
        assert checker._initialized is True

    @pytest.mark.asyncio
    async def test_async_initialize_with_profanity_library_available(self) -> None:
        """When better-profanity is installed, _profanity_available is True."""
        checker = ToxicityChecker(use_detoxify=False)

        with patch.object(checker, "_init_profanity", return_value=True):
            await checker.async_initialize()

        assert checker._profanity_available is True
        assert checker._initialized is True

    @pytest.mark.asyncio
    async def test_async_initialize_with_detoxify_enabled(self) -> None:
        """When use_detoxify=True, _init_detoxify is called and result stored."""
        checker = ToxicityChecker(use_detoxify=True)

        with (
            patch.object(checker, "_init_profanity", return_value=False),
            patch.object(checker, "_init_detoxify", return_value=True),
        ):
            await checker.async_initialize()

        assert checker._detoxify_available is True
        assert checker._initialized is True

    @pytest.mark.asyncio
    async def test_async_initialize_with_detoxify_disabled_skips_detoxify_call(self) -> None:
        """When use_detoxify=False, _init_detoxify is NOT called."""
        checker = ToxicityChecker(use_detoxify=False)

        with (
            patch.object(checker, "_init_profanity", return_value=False),
            patch.object(checker, "_init_detoxify", return_value=True) as mock_detox,
        ):
            await checker.async_initialize()

        mock_detox.assert_not_called()
        assert checker._detoxify_available is False


class TestToxicityCheckerCheck:
    """Tests for ToxicityChecker.check."""

    @pytest.mark.asyncio
    async def test_check_returns_safe_default_when_no_libraries_available(self) -> None:
        """Without profanity or detoxify, returns default safe result."""
        checker = ToxicityChecker()
        checker._profanity_available = False
        checker._detoxify_available = False

        result = await checker.check("hello world")

        assert result.is_safe is True
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_check_uses_profanity_when_available(self) -> None:
        """When profanity library available, delegates to _check_with_profanity."""
        checker = ToxicityChecker()
        checker._profanity_available = True
        checker._detoxify_available = False

        profanity_result = GuardRailResult(is_safe=False, confidence=0.8, category="inappropriate")

        with patch.object(checker, "_check_with_profanity", return_value=profanity_result):
            result = await checker.check("bad text")

        assert result.is_safe is False
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_check_early_exits_on_high_confidence(self) -> None:
        """Returns early without running detoxify when profanity confidence >= 0.9."""
        checker = ToxicityChecker()
        checker._profanity_available = True
        checker._detoxify_available = True

        high_conf_result = GuardRailResult(is_safe=False, confidence=0.95, category="inappropriate")

        with (
            patch.object(checker, "_check_with_profanity", return_value=high_conf_result),
            patch.object(checker, "_check_with_detoxify", new_callable=AsyncMock) as mock_detox,
        ):
            result = await checker.check("very bad text")

        mock_detox.assert_not_called()
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_check_uses_detoxify_when_available(self) -> None:
        """When detoxify is available, result is compared and best is returned."""
        checker = ToxicityChecker()
        checker._profanity_available = False
        checker._detoxify_available = True

        detox_result = GuardRailResult(is_safe=False, confidence=0.85, category="harmful")

        with patch.object(
            checker, "_check_with_detoxify", new_callable=AsyncMock, return_value=detox_result
        ):
            result = await checker.check("some text")

        assert result.confidence == 0.85
        assert result.is_safe is False

    @pytest.mark.asyncio
    async def test_check_returns_best_of_profanity_and_detoxify(self) -> None:
        """Returns the highest-confidence result across both stages."""
        checker = ToxicityChecker()
        checker._profanity_available = True
        checker._detoxify_available = True

        profanity_result = GuardRailResult(is_safe=False, confidence=0.65, category="inappropriate")
        detox_result = GuardRailResult(is_safe=False, confidence=0.88, category="harmful")

        with (
            patch.object(checker, "_check_with_profanity", return_value=profanity_result),
            patch.object(
                checker, "_check_with_detoxify", new_callable=AsyncMock, return_value=detox_result
            ),
        ):
            result = await checker.check("flagged text")

        assert result.confidence == 0.88
        assert result.category == "harmful"

    @pytest.mark.asyncio
    async def test_check_keeps_default_when_profanity_lower_confidence(self) -> None:
        """Profanity result below default confidence is not promoted."""
        checker = ToxicityChecker()
        checker._profanity_available = True
        checker._detoxify_available = False

        low_result = GuardRailResult(is_safe=True, confidence=0.4, category=None)

        with patch.object(checker, "_check_with_profanity", return_value=low_result):
            result = await checker.check("clean text")

        # Default was 0.5; low result (0.4) should NOT replace it.
        assert result.confidence == 0.5
        assert result.is_safe is True
