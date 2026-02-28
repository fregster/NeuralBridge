"""Guard rail system for content filtering and safety checks."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Protocol, runtime_checkable

from homeassistant.components.conversation import ConversationResult

from .const import (
    GUARD_RAIL_CATEGORY_HARMFUL,
    GUARD_RAIL_CATEGORY_INAPPROPRIATE,
    GUARD_RAIL_CATEGORY_PRIVACY,
    GUARD_RAIL_CATEGORY_SECURITY,
)
from .guard_rail_types import GuardRailResult  # noqa: F401 — re-exported for callers
from .pending_cache import PendingCache

# flake8: noqa — ToxicityChecker/AISafetyChecker imported after GuardRailResult
# to avoid circular import (they import GuardRailResult from guard_rail_types).
from .ai_safety_checker import AISafetyChecker  # noqa: E402
from .toxicity_checker import ToxicityChecker  # noqa: E402

_LOGGER = logging.getLogger(__name__)

_HIGH_CONFIDENCE: float = 0.9

# Default rule patterns for fast filtering
DEFAULT_HARMFUL_PATTERNS = [
    r"\b(?:kill|murder|suicide|harm yourself|self-harm)\b",
    r"\b(?:bomb|explosive|weapon)\s+(?:make|build|create|construct)",
    r"\bhow\s+to\s+(?:hack|break into|steal)",
]

DEFAULT_PRIVACY_PATTERNS = [
    r"\b(?:ssn|social security number)\b",
    r"\b(?:credit card|cc)\s*(?:number|#)?\s*:?\s*\d",
    r"\b(?:password|passwd|pwd)\s*:?\s*[^\s]+",
]

DEFAULT_SECURITY_PATTERNS = [
    r"\bapi[_-]?key\s*:?\s*['\"]?[a-zA-Z0-9_-]{20,}",
    r"\btoken\s*:?\s*['\"]?[a-zA-Z0-9_-]{20,}",
    r"\bsecret\s*:?\s*['\"]?[a-zA-Z0-9_-]{20,}",
]

DEFAULT_INAPPROPRIATE_PATTERNS = [
    r"\b(?:explicit|graphic)\s+(?:sexual|violent)\s+content\b",
]


@runtime_checkable
class GuardRailCheckerProtocol(Protocol):
    """Protocol for guard rail input/output checking."""

    async def async_initialize(self) -> None:
        """Load optional ML models off the event loop."""
        ...

    async def check_input(
        self,
        text: str,
        use_ai: bool = False,
        ai_agent_config: dict[str, Any] | None = None,
    ) -> GuardRailResult:
        """Check input text for safety issues."""
        ...

    async def check_output(
        self,
        text: str,
        use_ai: bool = False,
        ai_agent_config: dict[str, Any] | None = None,
    ) -> GuardRailResult:
        """Check output text for safety issues."""
        ...


class GuardRailChecker:
    """Guard rail checker with hybrid rule-based and AI filtering.

    Orchestrates a four-stage pipeline using delegated sub-checkers:
    1. Fast regex rule patterns (``_check_with_rules``, implemented here)
    2. better-profanity word-list (``_toxicity_checker._check_with_profanity``)
    3. detoxify ML model (``_toxicity_checker._check_with_detoxify``)
    4. Ollama AI (``_ai_checker.check``)

    Proxy properties and methods expose sub-checker internals so that
    existing code and tests can interact with the same attribute names
    as before the refactoring.
    """

    def __init__(
        self,
        rules: dict[str, list[str]] | None = None,
        ai_threshold: float = 0.7,
        use_detoxify: bool = False,
        detoxify_threshold: float = 0.7,
    ) -> None:
        """Initialise the guard rail checker.

        Args:
            rules: Custom rule patterns by category.
            ai_threshold: Confidence threshold below which Ollama AI is consulted.
            use_detoxify: Whether to enable the detoxify ML model (stage 3).
            detoxify_threshold: Score threshold for detoxify (0.0-1.0, default 0.7).
        """
        self._ai_threshold = ai_threshold
        self._rules = self._build_rules(rules)
        self._toxicity_checker = ToxicityChecker(
            use_detoxify=use_detoxify,
            detoxify_threshold=detoxify_threshold,
        )
        self._ai_checker = AISafetyChecker(ai_threshold=ai_threshold)

    # ── Sub-checker state proxies ───────────────────────────────────────────

    @property
    def _initialized(self) -> bool:
        """Return True after async_initialize has been called."""
        return self._toxicity_checker._initialized

    @_initialized.setter
    def _initialized(self, value: bool) -> None:
        self._toxicity_checker._initialized = value

    @property
    def _profanity_available(self) -> bool:
        """True when better-profanity loaded successfully."""
        return self._toxicity_checker._profanity_available

    @_profanity_available.setter
    def _profanity_available(self, value: bool) -> None:
        self._toxicity_checker._profanity_available = value

    @property
    def _detoxify_available(self) -> bool:
        """True when the detoxify model loaded successfully."""
        return self._toxicity_checker._detoxify_available

    @_detoxify_available.setter
    def _detoxify_available(self, value: bool) -> None:
        self._toxicity_checker._detoxify_available = value

    @property
    def _detoxify_model(self) -> Any:
        """The loaded detoxify model, or None."""
        return self._toxicity_checker._detoxify_model

    @_detoxify_model.setter
    def _detoxify_model(self, value: Any) -> None:
        self._toxicity_checker._detoxify_model = value

    @property
    def _detoxify_threshold(self) -> float:
        """Score threshold for the detoxify model (0.0-1.0)."""
        return self._toxicity_checker._detoxify_threshold

    # ── Initialisation ──────────────────────────────────────────────────────

    async def async_initialize(self) -> None:
        """Load ML models in a thread pool executor — safe to await on the loop.

        Idempotent: subsequent calls return immediately without re-loading.
        Before this method is awaited, ``_profanity_available`` and
        ``_detoxify_available`` are both ``False`` (safe degradation).
        """
        if self._initialized:
            return
        loop = asyncio.get_running_loop()
        self._profanity_available = await loop.run_in_executor(None, self._init_profanity)
        if self._toxicity_checker._use_detoxify:
            self._detoxify_available = await loop.run_in_executor(None, self._init_detoxify)
        self._initialized = True

    # ── Public check methods ────────────────────────────────────────────────

    async def check_input(
        self,
        text: str,
        use_ai: bool = False,
        ai_agent_config: dict[str, Any] | None = None,
    ) -> GuardRailResult:
        """Check input text for safety issues.

        Four-stage pipeline (each stage short-circuits if confidence >= 0.9):
        1. Regex rule patterns (harmful / privacy / security / inappropriate)
        2. better-profanity word-list (inappropriate category)
        3. detoxify ML model (opt-in via constructor ``use_detoxify=True``)
        4. Ollama AI (opt-in via ``use_ai=True`` with ``ai_agent_config``)

        All stages fail-open: errors return ``is_safe=True, confidence=0.5``.

        Args:
            text: Input text to check.
            use_ai: Whether to use Ollama AI-based checking as the final stage.
            ai_agent_config: Configuration for AI agent (required when use_ai=True).

        Returns:
            Guard rail check result.
        """
        if not text:
            return GuardRailResult(is_safe=True, confidence=1.0)

        # Stage 1: Fast regex rules (unchanged behaviour)
        rule_result = await self._check_with_rules(text)
        if rule_result.confidence >= _HIGH_CONFIDENCE:
            return rule_result

        # Stage 2: better-profanity word-list
        if self._profanity_available:
            profanity_result = self._check_with_profanity(text)
            if profanity_result.confidence >= _HIGH_CONFIDENCE:
                return profanity_result

        best_result = rule_result

        # Stage 3: detoxify ML model (opt-in, runs in thread executor)
        if self._detoxify_available and best_result.confidence < _HIGH_CONFIDENCE:
            detox_result = await self._check_with_detoxify(text)
            if detox_result.confidence > best_result.confidence:
                best_result = detox_result
            if best_result.confidence >= _HIGH_CONFIDENCE:
                return best_result

        # Stage 4: Ollama AI (opt-in, only when still uncertain)
        if use_ai and ai_agent_config and best_result.confidence < self._ai_threshold:
            ai_result = await self._check_with_ai(text, ai_agent_config)
            if ai_result.confidence > best_result.confidence:
                return ai_result

        return best_result

    async def check_output(
        self,
        text: str,
        use_ai: bool = False,
        ai_agent_config: dict[str, Any] | None = None,
    ) -> GuardRailResult:
        """Check output text for safety issues.

        Args:
            text: Output text to check.
            use_ai: Whether to use AI-based checking.
            ai_agent_config: Configuration for AI agent (if use_ai is True).

        Returns:
            Guard rail check result.
        """
        return await self.check_input(text, use_ai, ai_agent_config)

    # ── Stage 1: regex rules ────────────────────────────────────────────────

    def _build_rules(
        self, custom_rules: dict[str, list[str]] | None
    ) -> dict[str, list[re.Pattern[str]]]:
        """Build compiled regex patterns from rules.

        Args:
            custom_rules: Custom rule patterns by category.

        Returns:
            Dictionary of compiled patterns by category.
        """
        rules: dict[str, list[str]] = {
            GUARD_RAIL_CATEGORY_HARMFUL: DEFAULT_HARMFUL_PATTERNS.copy(),
            GUARD_RAIL_CATEGORY_PRIVACY: DEFAULT_PRIVACY_PATTERNS.copy(),
            GUARD_RAIL_CATEGORY_SECURITY: DEFAULT_SECURITY_PATTERNS.copy(),
            GUARD_RAIL_CATEGORY_INAPPROPRIATE: DEFAULT_INAPPROPRIATE_PATTERNS.copy(),
        }

        if custom_rules:
            for category, patterns in custom_rules.items():
                if category in rules:
                    rules[category].extend(patterns)
                else:
                    rules[category] = patterns

        compiled_rules: dict[str, list[re.Pattern[str]]] = {}
        for category, patterns in rules.items():
            compiled_rules[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled_rules

    async def _check_with_rules(self, text: str) -> GuardRailResult:
        """Check text using rule-based patterns.

        Args:
            text: Text to check.

        Returns:
            Guard rail check result.
        """
        for category, patterns in self._rules.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    _LOGGER.warning(
                        "Guard rail rule matched: category=%s, pattern=%s",
                        category,
                        pattern.pattern,
                    )
                    return GuardRailResult(
                        is_safe=False,
                        confidence=0.95,
                        category=category,
                        reason=f"Matched {category} pattern",
                        matched_pattern=pattern.pattern,
                    )
        return GuardRailResult(is_safe=True, confidence=0.6)

    # ── Stage 2 proxy methods (better-profanity) ────────────────────────────

    def _init_profanity(self) -> bool:
        """Delegate profanity library initialisation to ToxicityChecker."""
        return self._toxicity_checker._init_profanity()

    def _check_with_profanity(self, text: str) -> GuardRailResult:
        """Delegate profanity word-list check to ToxicityChecker."""
        return self._toxicity_checker._check_with_profanity(text)

    # ── Stage 3 proxy methods (detoxify) ────────────────────────────────────

    def _init_detoxify(self) -> bool:
        """Delegate detoxify model initialisation to ToxicityChecker."""
        return self._toxicity_checker._init_detoxify()

    async def _check_with_detoxify(self, text: str) -> GuardRailResult:
        """Delegate detoxify check to ToxicityChecker."""
        return await self._toxicity_checker._check_with_detoxify(text)

    def _evaluate_detoxify_scores(self, scores: dict[str, float]) -> GuardRailResult:
        """Delegate detoxify score evaluation to ToxicityChecker."""
        return self._toxicity_checker._evaluate_detoxify_scores(scores)

    # ── Stage 4 proxy methods (AI) ───────────────────────────────────────────

    async def _check_with_ai(self, text: str, ai_agent_config: dict[str, Any]) -> GuardRailResult:
        """Delegate AI safety check to AISafetyChecker."""
        return await self._ai_checker.check(text, ai_agent_config)

    def _create_safety_prompt(self, text: str) -> str:
        """Delegate safety prompt creation to AISafetyChecker."""
        return self._ai_checker._create_safety_prompt(text)

    def _parse_ai_response(self, response: str) -> GuardRailResult:
        """Delegate AI response parsing to AISafetyChecker."""
        return self._ai_checker._parse_ai_response(response)


class GuardRailCache(PendingCache[tuple[str, GuardRailResult]]):
    """Pending-response cache for guard-rail confirmation flow."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300) -> None:
        """Initialise the guard-rail pending cache.

        Args:
            max_size: Maximum number of concurrent pending entries.
            ttl_seconds: Time-to-live in seconds for each entry.
        """
        super().__init__(max_size=max_size, ttl_seconds=ttl_seconds, key_prefix="gr")

    async def store_pending_response(
        self,
        conversation_id: str,
        response: str,
        guard_rail_result: GuardRailResult,
    ) -> None:
        """Store a response pending user confirmation.

        Args:
            conversation_id: Unique conversation identifier.
            response: The response text to cache.
            guard_rail_result: The guard rail check result.
        """
        await self.store(conversation_id, (response, guard_rail_result))

    async def get_pending_response(
        self, conversation_id: str
    ) -> tuple[str, GuardRailResult] | None:
        """Get a pending response.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            Tuple of (response, guard_rail_result) if found, None otherwise.
        """
        return await self.get(conversation_id)

    async def clear_pending_response(self, conversation_id: str) -> None:
        """Clear a pending response.

        Args:
            conversation_id: Unique conversation identifier.
        """
        await self.clear(conversation_id)


class HighStakesCache(PendingCache[tuple[ConversationResult, list[str]]]):
    """Short-lived cache that holds a ConversationResult pending user confirmation.

    Used by the high-stakes confirmation flow (Feature 4) to remember the
    original LOCAL_HA result while waiting for the user to confirm or supply
    the required passphrase on their next turn.

    The default TTL is intentionally short (120 s) so that unanswered
    confirmation prompts do not linger indefinitely.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 120) -> None:
        """Initialise the high-stakes pending-result cache.

        Args:
            max_size: Maximum number of pending items to retain simultaneously.
            ttl_seconds: Seconds before an unconfirmed entry expires.
        """
        super().__init__(max_size=max_size, ttl_seconds=ttl_seconds, key_prefix="hs")

    async def store_pending(
        self,
        conversation_id: str,
        result: ConversationResult,
        entity_ids: list[str],
    ) -> None:
        """Store a pending high-stakes result awaiting confirmation.

        Args:
            conversation_id: Unique conversation identifier.
            result: The ConversationResult to hold pending confirmation.
            entity_ids: Entity IDs targeted by the action (for event firing).
        """
        await self.store(conversation_id, (result, entity_ids))

    async def get_pending(
        self, conversation_id: str
    ) -> tuple[ConversationResult, list[str]] | None:
        """Return the pending result and entity IDs for a conversation, or None.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            Tuple of (ConversationResult, entity_ids) if a non-expired entry
            exists, otherwise None.
        """
        return await self.get(conversation_id)

    async def clear_pending(self, conversation_id: str) -> None:
        """Remove any pending entry for the given conversation.

        Args:
            conversation_id: Unique conversation identifier.
        """
        await self.clear(conversation_id)
