"""Toxicity checking stages for the guard rail pipeline.

Implements stages 2 (better-profanity word-list) and 3 (detoxify ML model)
of the four-stage guard rail pipeline.  Both stages are opt-in — profanity
checking requires ``pip install better-profanity``; detoxify requires
``pip install detoxify`` (~200 MB model).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar, Protocol, runtime_checkable

from .const import (
    GUARD_RAIL_CATEGORY_HARMFUL,
    GUARD_RAIL_CATEGORY_INAPPROPRIATE,
)
from .guard_rail_types import GuardRailResult

_LOGGER = logging.getLogger(__name__)

# Confidence threshold above which we skip further checks and return early.
_CONFIDENCE_EARLY_EXIT: float = 0.9


@runtime_checkable
class ToxicityCheckerProtocol(Protocol):
    """Protocol for toxicity checking (stages 2 & 3 of the guard rail pipeline)."""

    async def async_initialize(self) -> None:
        """Load ML models in a thread pool executor."""
        ...

    async def check(self, text: str) -> GuardRailResult:
        """Run profanity and/or detoxify checks on *text*.

        Returns:
            GuardRailResult — confidence ≥ 0.9 means upstream may short-circuit.
        """
        ...


class ToxicityChecker:
    """Handles better-profanity (stage 2) and detoxify ML (stage 3) checks.

    Lightweight constructor — call :meth:`async_initialize` once before
    calling :meth:`check`.
    """

    # Maps detoxify score keys to NeuralBridge guard rail categories.
    _DETOXIFY_CATEGORY_MAP: ClassVar[dict[str, str]] = {
        "toxicity": GUARD_RAIL_CATEGORY_HARMFUL,
        "severe_toxicity": GUARD_RAIL_CATEGORY_HARMFUL,
        "obscene": GUARD_RAIL_CATEGORY_INAPPROPRIATE,
        "threat": GUARD_RAIL_CATEGORY_HARMFUL,
        "insult": GUARD_RAIL_CATEGORY_INAPPROPRIATE,
        "identity_attack": GUARD_RAIL_CATEGORY_HARMFUL,
    }

    def __init__(
        self,
        use_detoxify: bool = False,
        detoxify_threshold: float = 0.7,
    ) -> None:
        """Initialise the toxicity checker.

        Args:
            use_detoxify: Whether to enable the detoxify ML model as stage 3.
                Requires ``pip install detoxify``.  Disabled by default.
            detoxify_threshold: Score above which a detoxify category is flagged
                (0.0-1.0, default 0.7).
        """
        self._use_detoxify = use_detoxify
        self._detoxify_threshold = detoxify_threshold
        self._profanity_available: bool = False
        self._detoxify_model: Any = None
        self._detoxify_available: bool = False
        self._initialized: bool = False

    async def async_initialize(self) -> None:
        """Load ML models in a thread pool executor — safe to await on event loop.

        Idempotent: subsequent calls return immediately without re-loading.
        """
        if self._initialized:
            return
        loop = asyncio.get_running_loop()
        self._profanity_available = await loop.run_in_executor(None, self._init_profanity)
        if self._use_detoxify:
            self._detoxify_available = await loop.run_in_executor(None, self._init_detoxify)
        self._initialized = True

    # ── Library initialisation ──────────────────────────────────────────────

    def _init_profanity(self) -> bool:
        """Load the better-profanity word list.

        Returns:
            True if the library is available and initialised, False otherwise.
        """
        try:
            from better_profanity import profanity  # noqa: PLC0415

            profanity.load_censor_words()
            _LOGGER.debug("better-profanity word list loaded")
            return True
        except ImportError:
            _LOGGER.debug("better-profanity not installed — profanity word-list checking disabled")
            return False
        except Exception as err:
            _LOGGER.warning("Failed to initialise better-profanity: %s", err)
            return False

    def _init_detoxify(self) -> bool:
        """Load the detoxify transformer model ('original', ~200 MB).

        Only called when use_detoxify=True. Install with: ``pip install detoxify``

        Returns:
            True if the model loaded successfully, False otherwise.
        """
        try:
            from detoxify import Detoxify  # noqa: PLC0415

            self._detoxify_model = Detoxify("original")
            _LOGGER.info("detoxify model loaded successfully")
            return True
        except ImportError:
            _LOGGER.warning(
                "detoxify is not installed. "
                "Install it with: pip install detoxify. "
                "ML-based toxicity checking is disabled."
            )
            return False
        except Exception as err:
            _LOGGER.warning("Failed to initialise detoxify model: %s", err)
            return False

    # ── Public check method ─────────────────────────────────────────────────

    async def check(self, text: str) -> GuardRailResult:
        """Run stages 2 and 3 of the guard rail pipeline.

        Stage 2: better-profanity word-list (synchronous call; only when
        ``_profanity_available``).
        Stage 3: detoxify ML model in thread executor (only when opt-in and
        ``_detoxify_available``).

        Args:
            text: Text to evaluate.

        Returns:
            The most significant GuardRailResult found across both stages.
            Returns is_safe=True, confidence=0.5 as a fail-open default.
        """
        best = GuardRailResult(is_safe=True, confidence=0.5)

        if self._profanity_available:
            profanity_result = self._check_with_profanity(text)
            if profanity_result.confidence > best.confidence:
                best = profanity_result
            if best.confidence >= _CONFIDENCE_EARLY_EXIT:
                return best

        if self._detoxify_available:
            detox_result = await self._check_with_detoxify(text)
            if detox_result.confidence > best.confidence:
                best = detox_result

        return best

    # ── Stage 2: better-profanity ───────────────────────────────────────────

    def _check_with_profanity(self, text: str) -> GuardRailResult:
        """Check text using the better-profanity word-list.

        Args:
            text: Text to check.

        Returns:
            GuardRailResult with confidence 0.9 on a match, 0.6 on clean text,
            or 0.5 (fail-open) on any unexpected error.
        """
        try:
            from better_profanity import profanity  # noqa: PLC0415

            if profanity.contains_profanity(text):
                _LOGGER.warning(
                    "Guard rail profanity check matched: category=%s",
                    GUARD_RAIL_CATEGORY_INAPPROPRIATE,
                )
                return GuardRailResult(
                    is_safe=False,
                    confidence=0.9,
                    category=GUARD_RAIL_CATEGORY_INAPPROPRIATE,
                    reason="Profanity detected by word-list filter",
                )
            return GuardRailResult(is_safe=True, confidence=0.6)
        except Exception as err:
            _LOGGER.warning("Error in profanity check — failing open: %s", err)
            return GuardRailResult(is_safe=True, confidence=0.5)

    # ── Stage 3: detoxify ML model ──────────────────────────────────────────

    async def _check_with_detoxify(self, text: str) -> GuardRailResult:
        """Check text using the detoxify transformer model.

        Runs ``model.predict()`` in a thread executor.

        Args:
            text: Text to check.

        Returns:
            GuardRailResult derived from detoxify toxicity scores.
        """
        if self._detoxify_model is None:
            return GuardRailResult(is_safe=True, confidence=0.5)
        try:
            loop = asyncio.get_running_loop()
            model = self._detoxify_model
            scores: dict[str, float] = await loop.run_in_executor(None, lambda: model.predict(text))
            return self._evaluate_detoxify_scores(scores)
        except Exception as err:
            _LOGGER.warning("Error in detoxify check — failing open: %s", err)
            return GuardRailResult(is_safe=True, confidence=0.5)

    def _evaluate_detoxify_scores(self, scores: dict[str, float]) -> GuardRailResult:
        """Convert a detoxify score dictionary into a GuardRailResult.

        Args:
            scores: Detoxify category -> float score (0.0-1.0) mapping.

        Returns:
            GuardRailResult reflecting the most significant detected category.
        """
        best_score = 0.0
        best_key: str | None = None

        for key, score in scores.items():
            if key in self._DETOXIFY_CATEGORY_MAP and score > best_score:
                best_score = score
                best_key = key

        if best_key is None or best_score < self._detoxify_threshold:
            return GuardRailResult(is_safe=True, confidence=0.6)

        nb_category = self._DETOXIFY_CATEGORY_MAP[best_key]
        _LOGGER.warning(
            "Guard rail detoxify matched: key=%s, category=%s, score=%.3f",
            best_key,
            nb_category,
            best_score,
        )
        return GuardRailResult(
            is_safe=False,
            confidence=0.9,
            category=nb_category,
            reason=f"Detoxify: {best_key} score {best_score:.3f}",
        )
