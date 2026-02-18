"""Tests for the guard rail system."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.neuralbridge.const import (
    CONF_AGENT_TYPE,
    CONF_OLLAMA_MODEL,
    CONF_OLLAMA_URL,
    CONF_TIMEOUT,
    GUARD_RAIL_CATEGORY_HARMFUL,
    GUARD_RAIL_CATEGORY_INAPPROPRIATE,
    GUARD_RAIL_CATEGORY_PRIVACY,
    GUARD_RAIL_CATEGORY_SECURITY,
)
from custom_components.neuralbridge.guard_rail import (
    GuardRailCache,
    GuardRailChecker,
    GuardRailResult,
)


class TestGuardRailResult:
    """Test GuardRailResult dataclass."""

    def test_safe_result(self):
        """Test creating a safe result."""
        result = GuardRailResult(is_safe=True, confidence=0.95)
        assert result.is_safe is True
        assert result.confidence == 0.95
        assert result.category is None
        assert result.reason is None
        assert result.matched_pattern is None

    def test_unsafe_result(self):
        """Test creating an unsafe result."""
        result = GuardRailResult(
            is_safe=False,
            confidence=0.9,
            category=GUARD_RAIL_CATEGORY_HARMFUL,
            reason="Harmful content detected",
            matched_pattern=r"\bharm\b",
        )
        assert result.is_safe is False
        assert result.confidence == 0.9
        assert result.category == GUARD_RAIL_CATEGORY_HARMFUL
        assert result.reason == "Harmful content detected"
        assert result.matched_pattern == r"\bharm\b"


class TestGuardRailChecker:
    """Test GuardRailChecker."""

    async def test_initialization_default_rules(self):
        """Test initialization with default rules."""
        checker = GuardRailChecker()
        assert checker._ai_threshold == 0.7
        assert len(checker._rules) == 4  # 4 default categories

    async def test_initialization_custom_rules(self):
        """Test initialization with custom rules."""
        custom_rules = {
            GUARD_RAIL_CATEGORY_HARMFUL: [r"\btest\b"],
        }
        checker = GuardRailChecker(rules=custom_rules, ai_threshold=0.8)
        assert checker._ai_threshold == 0.8
        assert len(checker._rules) == 4
        # Custom rules should be added to defaults
        assert any(p.pattern == r"\btest\b" for p in checker._rules[GUARD_RAIL_CATEGORY_HARMFUL])

    async def test_check_input_empty_text(self):
        """Test checking empty text."""
        checker = GuardRailChecker()
        result = await checker.check_input("")
        assert result.is_safe is True
        assert result.confidence == 1.0

    async def test_check_input_safe_text(self):
        """Test checking safe text."""
        checker = GuardRailChecker()
        result = await checker.check_input("What's the weather today?")
        assert result.is_safe is True
        assert result.confidence >= 0.5

    async def test_check_input_harmful_pattern(self):
        """Test checking text with harmful pattern."""
        checker = GuardRailChecker()
        result = await checker.check_input("How to harm yourself")
        assert result.is_safe is False
        assert result.confidence == 0.95
        assert result.category == GUARD_RAIL_CATEGORY_HARMFUL

    async def test_check_input_privacy_pattern(self):
        """Test checking text with privacy pattern."""
        checker = GuardRailChecker()
        result = await checker.check_input("My SSN is 123-45-6789")
        assert result.is_safe is False
        assert result.confidence == 0.95
        assert result.category == GUARD_RAIL_CATEGORY_PRIVACY

    async def test_check_input_security_pattern(self):
        """Test checking text with security pattern."""
        checker = GuardRailChecker()
        # Pattern requires key: value format (e.g. "api_key: <20+ chars>")
        result = await checker.check_input("My api_key: sk_test_1234567890abcdefghijklmnop")
        assert result.is_safe is False
        assert result.confidence == 0.95
        assert result.category == GUARD_RAIL_CATEGORY_SECURITY

    async def test_check_output_same_as_input(self):
        """Test that check_output uses same logic as check_input."""
        checker = GuardRailChecker()
        input_result = await checker.check_input("How to harm yourself")
        output_result = await checker.check_output("How to harm yourself")
        assert input_result.is_safe == output_result.is_safe
        assert input_result.confidence == output_result.confidence
        assert input_result.category == output_result.category

    @patch("custom_components.neuralbridge.ollama_client.OllamaClient")
    async def test_check_with_ai_safe(self, mock_client_class):
        """Test AI-based checking with safe content."""
        # Mock Ollama client
        mock_client = AsyncMock()
        mock_client.generate.return_value = "SAFE"
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # ai_threshold=0.7 means "call AI when rule confidence < 0.7".
        # Safe text gets rule confidence 0.6, so AI is triggered (0.6 < 0.7).
        checker = GuardRailChecker(ai_threshold=0.7)
        ai_agent_config = {
            CONF_AGENT_TYPE: "ollama",
            CONF_OLLAMA_URL: "http://localhost:11434",
            CONF_OLLAMA_MODEL: "tinyllama",
            CONF_TIMEOUT: 30,
        }

        result = await checker.check_input(
            "What's the weather?",
            use_ai=True,
            ai_agent_config=ai_agent_config,
        )

        assert result.is_safe is True
        assert result.confidence >= 0.5
        mock_client.generate.assert_called_once()

    @patch("custom_components.neuralbridge.ollama_client.OllamaClient")
    async def test_check_with_ai_unsafe(self, mock_client_class):
        """Test AI-based checking with unsafe content."""
        # Mock Ollama client
        mock_client = AsyncMock()
        mock_client.generate.return_value = "UNSAFE: harmful - contains harmful content"
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # ai_threshold=0.7 so rules (confidence 0.6) trigger AI (0.6 < 0.7).
        checker = GuardRailChecker(ai_threshold=0.7)
        ai_agent_config = {
            CONF_AGENT_TYPE: "ollama",
            CONF_OLLAMA_URL: "http://localhost:11434",
            CONF_OLLAMA_MODEL: "tinyllama",
            CONF_TIMEOUT: 30,
        }

        result = await checker.check_input(
            "Some potentially harmful content",
            use_ai=True,
            ai_agent_config=ai_agent_config,
        )

        assert result.is_safe is False
        assert result.category == "harmful"
        # _parse_ai_response uppercases the response; compare case-insensitively.
        assert "harmful content" in result.reason.lower()

    @patch("custom_components.neuralbridge.ollama_client.OllamaClient")
    async def test_check_with_ai_timeout(self, mock_client_class):
        """Test AI-based checking with timeout."""
        # Mock Ollama client to timeout
        mock_client = AsyncMock()
        mock_client.generate.side_effect = asyncio.TimeoutError()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # ai_threshold=0.7 so AI is triggered for text with rule confidence 0.6.
        checker = GuardRailChecker(ai_threshold=0.7)
        ai_agent_config = {
            CONF_AGENT_TYPE: "ollama",
            CONF_OLLAMA_URL: "http://localhost:11434",
            CONF_OLLAMA_MODEL: "tinyllama",
            CONF_TIMEOUT: 30,
        }

        result = await checker.check_input(
            "Some text",
            use_ai=True,
            ai_agent_config=ai_agent_config,
        )

        # Fail open: AI timed out (confidence 0.5) < rule confidence (0.6), so
        # the rule result is returned — safe with the rule-based confidence.
        assert result.is_safe is True
        assert result.confidence >= 0.5

    async def test_check_with_ai_unsupported_agent_type(self):
        """Test AI-based checking with unsupported agent type."""
        # Default ai_threshold=0.7; rule confidence for "Some text" is 0.6 → AI triggered.
        checker = GuardRailChecker()
        ai_agent_config = {
            CONF_AGENT_TYPE: "unsupported",
        }

        result = await checker.check_input(
            "Some text",
            use_ai=True,
            ai_agent_config=ai_agent_config,
        )

        # AI returns confidence 0.5 (unsupported type), rules returned 0.6.
        # check_input returns whichever is MORE confident → rule result (0.6).
        assert result.is_safe is True
        assert result.confidence >= 0.5

    @patch("custom_components.neuralbridge.ollama_client.OllamaClient")
    async def test_check_with_ai_empty_response_returns_safe(self, mock_client_class):
        """Test AI-based checking when generate() returns empty string → safe."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = ""
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        checker = GuardRailChecker(ai_threshold=0.7)
        ai_agent_config = {
            CONF_AGENT_TYPE: "ollama",
            CONF_OLLAMA_URL: "http://localhost:11434",
            CONF_OLLAMA_MODEL: "tinyllama",
            CONF_TIMEOUT: 30,
        }

        result = await checker.check_input(
            "What is the weather?",
            use_ai=True,
            ai_agent_config=ai_agent_config,
        )

        assert result.is_safe is True
        assert result.confidence >= 0.5

    @patch("custom_components.neuralbridge.ollama_client.OllamaClient")
    async def test_check_with_ai_generic_exception_returns_safe(self, mock_client_class):
        """Test AI-based checking when generate() raises a generic exception → safe."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = RuntimeError("connection refused")
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        checker = GuardRailChecker(ai_threshold=0.7)
        ai_agent_config = {
            CONF_AGENT_TYPE: "ollama",
            CONF_OLLAMA_URL: "http://localhost:11434",
            CONF_OLLAMA_MODEL: "tinyllama",
            CONF_TIMEOUT: 30,
        }

        result = await checker.check_input(
            "What is the weather?",
            use_ai=True,
            ai_agent_config=ai_agent_config,
        )

        # Fail open — exception → AI result (confidence 0.5) < rule result (0.6)
        assert result.is_safe is True
        assert result.confidence >= 0.5

    def test_build_rules_new_category_added(self):
        """Custom rules with a brand-new category are added alongside defaults."""
        custom_rules = {
            "custom_category": [r"\bcustom_bad_word\b"],
        }
        checker = GuardRailChecker(rules=custom_rules)
        # Should have 5 categories now: the 4 defaults + 1 custom
        assert len(checker._rules) == 5
        assert "custom_category" in checker._rules

    def test_parse_ai_response_unsafe_no_details(self):
        """_parse_ai_response returns unsafe result when UNSAFE line has no colon details."""
        checker = GuardRailChecker()
        result = checker._parse_ai_response("UNSAFE")
        assert result.is_safe is False
        assert result.confidence >= 0.9

    def test_initialization_with_new_params(self):
        """GuardRailChecker accepts use_detoxify and detoxify_threshold constructor params."""
        checker = GuardRailChecker(use_detoxify=False, detoxify_threshold=0.6)
        assert checker._detoxify_threshold == pytest.approx(0.6)
        assert checker._detoxify_available is False
        assert checker._detoxify_model is None

    async def test_full_pipeline_regex_short_circuits_all_stages(self):
        """Regex match at stage 1 prevents profanity, detoxify, and AI stages from running."""
        mock_model = MagicMock()
        checker = GuardRailChecker()
        checker._detoxify_model = mock_model
        checker._detoxify_available = True

        # "harm yourself" triggers DEFAULT_HARMFUL_PATTERNS (confidence 0.95)
        result = await checker.check_input("How to harm yourself")

        assert result.is_safe is False
        assert result.confidence == pytest.approx(0.95)
        mock_model.predict.assert_not_called()


class TestGuardRailCache:
    """Test GuardRailCache."""

    async def test_initialization(self):
        """Test cache initialization."""
        cache = GuardRailCache(max_size=50, ttl_seconds=60)
        assert cache._max_size == 50
        assert cache._ttl_seconds == 60
        assert len(cache._cache) == 0

    async def test_store_and_get_pending_response(self):
        """Test storing and retrieving pending response."""
        cache = GuardRailCache()
        conversation_id = "test_conversation"
        response = "Test response"
        guard_rail_result = GuardRailResult(
            is_safe=False,
            confidence=0.9,
            category=GUARD_RAIL_CATEGORY_HARMFUL,
        )

        # Store response
        await cache.store_pending_response(conversation_id, response, guard_rail_result)

        # Retrieve response
        retrieved = await cache.get_pending_response(conversation_id)
        assert retrieved is not None
        retrieved_response, retrieved_result = retrieved
        assert retrieved_response == response
        assert retrieved_result.is_safe == guard_rail_result.is_safe
        assert retrieved_result.category == guard_rail_result.category

    async def test_get_nonexistent_pending_response(self):
        """Test retrieving non-existent pending response."""
        cache = GuardRailCache()
        result = await cache.get_pending_response("nonexistent")
        assert result is None

    async def test_clear_pending_response(self):
        """Test clearing pending response."""
        cache = GuardRailCache()
        conversation_id = "test_conversation"
        response = "Test response"
        guard_rail_result = GuardRailResult(is_safe=False, confidence=0.9)

        # Store response
        await cache.store_pending_response(conversation_id, response, guard_rail_result)

        # Clear response
        await cache.clear_pending_response(conversation_id)

        # Should no longer be retrievable
        result = await cache.get_pending_response(conversation_id)
        assert result is None

    async def test_cache_expiry(self):
        """Test cache entry expiry."""
        cache = GuardRailCache(ttl_seconds=1)  # 1 second TTL
        conversation_id = "test_conversation"
        response = "Test response"
        guard_rail_result = GuardRailResult(is_safe=False, confidence=0.9)

        # Store response
        await cache.store_pending_response(conversation_id, response, guard_rail_result)

        # Wait for expiry
        await asyncio.sleep(1.1)

        # Should be expired
        result = await cache.get_pending_response(conversation_id)
        assert result is None

    async def test_cache_max_size(self):
        """Test cache max size enforcement."""
        cache = GuardRailCache(max_size=3, ttl_seconds=300)
        guard_rail_result = GuardRailResult(is_safe=False, confidence=0.9)

        # Add 4 entries (exceeds max_size)
        for i in range(4):
            await cache.store_pending_response(
                f"conversation_{i}",
                f"response_{i}",
                guard_rail_result,
            )

        # Cache should only have 3 entries (oldest removed)
        assert len(cache._cache) == 3

        # Oldest entry should be removed
        result = await cache.get_pending_response("conversation_0")
        assert result is None

        # Newer entries should exist
        result = await cache.get_pending_response("conversation_3")
        assert result is not None

    async def test_cleanup_on_store(self):
        """Test cleanup is called on store."""
        cache = GuardRailCache(max_size=2, ttl_seconds=1)
        guard_rail_result = GuardRailResult(is_safe=False, confidence=0.9)

        # Add first entry
        await cache.store_pending_response("conversation_1", "response_1", guard_rail_result)

        # Wait for it to expire
        await asyncio.sleep(1.1)

        # Add second entry - should trigger cleanup
        await cache.store_pending_response("conversation_2", "response_2", guard_rail_result)

        # First entry should be cleaned up
        assert len(cache._cache) == 1
        result = await cache.get_pending_response("conversation_1")
        assert result is None


class TestGuardRailCheckerOptionalLibraries:
    """Tests for optional library integration: better-profanity and detoxify."""

    # ── _init_profanity ──────────────────────────────────────────────────────

    def test_init_profanity_import_error_returns_false(self):
        """_init_profanity returns False when better-profanity is not installed."""
        with patch.dict(sys.modules, {"better_profanity": None}):
            checker = GuardRailChecker()
        assert checker._profanity_available is False

    def test_init_profanity_generic_exception_returns_false(self):
        """_init_profanity returns False when load_censor_words raises an exception."""
        mock_prof = MagicMock()
        mock_prof.load_censor_words.side_effect = RuntimeError("load failed")
        mock_mod = MagicMock()
        mock_mod.profanity = mock_prof
        with patch.dict(sys.modules, {"better_profanity": mock_mod}):
            checker = GuardRailChecker()
        assert checker._profanity_available is False

    # ── _check_with_profanity ────────────────────────────────────────────────

    def test_check_with_profanity_detects_profanity(self):
        """_check_with_profanity returns unsafe when text contains profanity."""
        mock_prof = MagicMock()
        mock_prof.contains_profanity.return_value = True
        mock_mod = MagicMock()
        mock_mod.profanity = mock_prof
        checker = GuardRailChecker()
        checker._profanity_available = True
        with patch.dict(sys.modules, {"better_profanity": mock_mod}):
            result = checker._check_with_profanity("bad word text")
        assert result.is_safe is False
        assert result.confidence >= 0.9

    def test_check_with_profanity_exception_fails_open(self):
        """_check_with_profanity returns safe (fail-open) on any exception."""
        mock_prof = MagicMock()
        mock_prof.contains_profanity.side_effect = RuntimeError("check failed")
        mock_mod = MagicMock()
        mock_mod.profanity = mock_prof
        checker = GuardRailChecker()
        checker._profanity_available = True
        with patch.dict(sys.modules, {"better_profanity": mock_mod}):
            result = checker._check_with_profanity("some text")
        assert result.is_safe is True

    # ── check_input profanity stage ──────────────────────────────────────────

    async def test_check_input_returns_early_when_profanity_high_confidence(self):
        """check_input returns the profanity result when confidence >= 0.9."""
        checker = GuardRailChecker()
        checker._profanity_available = True
        high_conf = GuardRailResult(is_safe=False, confidence=0.9, category="inappropriate")
        with patch.object(checker, "_check_with_profanity", return_value=high_conf):
            result = await checker.check_input("some text")
        assert result.is_safe is False
        assert result.confidence >= 0.9

    # ── _init_detoxify ───────────────────────────────────────────────────────

    def test_init_detoxify_import_error_returns_false(self):
        """_init_detoxify returns False when detoxify is not installed (ImportError)."""
        # detoxify is not installed in the test env — ImportError is raised naturally
        checker = GuardRailChecker(use_detoxify=True)
        assert checker._detoxify_available is False

    def test_init_detoxify_success_returns_true(self):
        """_init_detoxify returns True and stores the model when detoxify loads."""
        mock_model_instance = MagicMock()
        mock_mod = MagicMock()
        mock_mod.Detoxify = MagicMock(return_value=mock_model_instance)
        with patch.dict(sys.modules, {"detoxify": mock_mod}):
            checker = GuardRailChecker(use_detoxify=True)
        assert checker._detoxify_available is True
        assert checker._detoxify_model is mock_model_instance

    def test_init_detoxify_generic_exception_returns_false(self):
        """_init_detoxify returns False when Detoxify() raises a non-ImportError."""
        mock_mod = MagicMock()
        mock_mod.Detoxify = MagicMock(side_effect=RuntimeError("model load failed"))
        with patch.dict(sys.modules, {"detoxify": mock_mod}):
            checker = GuardRailChecker(use_detoxify=True)
        assert checker._detoxify_available is False

    # ── check_input detoxify stage ───────────────────────────────────────────

    async def test_check_input_detoxify_high_confidence_returns_early(self):
        """check_input returns detoxify result early when its confidence is >= 0.9."""
        checker = GuardRailChecker()
        checker._detoxify_available = True
        checker._detoxify_model = MagicMock()
        high_conf = GuardRailResult(is_safe=False, confidence=0.9, category="harmful")
        with patch.object(
            checker, "_check_with_detoxify", new_callable=AsyncMock, return_value=high_conf
        ):
            result = await checker.check_input("some text")
        assert result.confidence >= 0.9

    async def test_check_input_runs_detoxify_stage_when_available(self):
        """check_input invokes detoxify when _detoxify_available is True."""
        checker = GuardRailChecker()
        checker._detoxify_available = True
        checker._detoxify_model = MagicMock()
        safe_result = GuardRailResult(is_safe=True, confidence=0.6)
        with patch.object(
            checker, "_check_with_detoxify", new_callable=AsyncMock, return_value=safe_result
        ):
            result = await checker.check_input("some text")
        assert result is not None

    # ── _check_with_detoxify ─────────────────────────────────────────────────

    async def test_check_with_detoxify_no_model_returns_safe(self):
        """_check_with_detoxify returns safe (confidence 0.5) when model is None."""
        checker = GuardRailChecker()
        checker._detoxify_model = None
        result = await checker._check_with_detoxify("text")
        assert result.is_safe is True

    async def test_check_with_detoxify_calls_model_predict(self):
        """_check_with_detoxify calls model.predict in an executor."""
        checker = GuardRailChecker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value={"toxicity": 0.1})
        checker._detoxify_model = mock_model
        checker._detoxify_available = True
        result = await checker._check_with_detoxify("safe text")
        mock_model.predict.assert_called_once_with("safe text")
        assert result is not None

    async def test_check_with_detoxify_exception_fails_open(self):
        """_check_with_detoxify returns safe (fail-open) when predict raises."""
        checker = GuardRailChecker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(side_effect=RuntimeError("predict failed"))
        checker._detoxify_model = mock_model
        checker._detoxify_available = True
        result = await checker._check_with_detoxify("text")
        assert result.is_safe is True

    # ── _evaluate_detoxify_scores ────────────────────────────────────────────

    def test_evaluate_detoxify_scores_above_threshold_returns_unsafe(self):
        """_evaluate_detoxify_scores returns unsafe when score exceeds threshold."""
        checker = GuardRailChecker(detoxify_threshold=0.7)
        scores = {"toxicity": 0.95, "obscene": 0.3}
        result = checker._evaluate_detoxify_scores(scores)
        assert result.is_safe is False
        assert result.confidence >= 0.9

    def test_evaluate_detoxify_scores_below_threshold_returns_safe(self):
        """_evaluate_detoxify_scores returns safe when all scores < threshold."""
        checker = GuardRailChecker(detoxify_threshold=0.7)
        scores = {"toxicity": 0.1, "obscene": 0.2}
        result = checker._evaluate_detoxify_scores(scores)
        assert result.is_safe is True
