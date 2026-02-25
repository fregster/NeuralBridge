"""Unit tests for the prompts_loader module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

from custom_components.neuralbridge.prompts_loader import load_prompt

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestLoadPrompt:
    """Tests for load_prompt."""

    def test_returns_file_content_when_file_exists(self, tmp_path: Path) -> None:
        """Returns the file content when the prompt file exists."""
        prompt_file = tmp_path / "my_prompt.txt"
        prompt_file.write_text("Hello, world!", encoding="utf-8")

        with patch("custom_components.neuralbridge.prompts_loader._PROMPTS_DIR", tmp_path):
            result = load_prompt("my_prompt.txt", fallback="fallback text")

        assert result == "Hello, world!"

    def test_strips_trailing_whitespace(self, tmp_path: Path) -> None:
        """Trailing whitespace and newlines are stripped from file content."""
        prompt_file = tmp_path / "padded.txt"
        prompt_file.write_text("Hello\n\n  ", encoding="utf-8")

        with patch("custom_components.neuralbridge.prompts_loader._PROMPTS_DIR", tmp_path):
            result = load_prompt("padded.txt")

        assert result == "Hello"

    def test_returns_fallback_when_file_missing(self, tmp_path: Path) -> None:
        """Returns the fallback string when the prompt file does not exist."""
        with patch("custom_components.neuralbridge.prompts_loader._PROMPTS_DIR", tmp_path):
            result = load_prompt("nonexistent.txt", fallback="my fallback")

        assert result == "my fallback"

    def test_returns_empty_string_fallback_by_default(self, tmp_path: Path) -> None:
        """Returns empty string when file is missing and no fallback is provided."""
        with patch("custom_components.neuralbridge.prompts_loader._PROMPTS_DIR", tmp_path):
            result = load_prompt("nonexistent.txt")

        assert result == ""

    def test_logs_warning_when_file_missing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warning is logged when the prompt file cannot be read."""
        with (
            patch("custom_components.neuralbridge.prompts_loader._PROMPTS_DIR", tmp_path),
            caplog.at_level(
                logging.WARNING, logger="custom_components.neuralbridge.prompts_loader"
            ),
        ):
            load_prompt("missing.txt", fallback="fallback")

        assert any("missing.txt" in record.message for record in caplog.records)

    def test_placeholder_preserved_in_content(self, tmp_path: Path) -> None:
        """Template placeholders like {user_text} are preserved verbatim."""
        prompt_file = tmp_path / "template.txt"
        prompt_file.write_text("Say hello to {user_text}", encoding="utf-8")

        with patch("custom_components.neuralbridge.prompts_loader._PROMPTS_DIR", tmp_path):
            result = load_prompt("template.txt")

        assert "{user_text}" in result

    def test_multiline_content_preserved(self, tmp_path: Path) -> None:
        """Multi-line prompts are returned with internal newlines intact."""
        content = "Line one\nLine two\n\nLine four"
        prompt_file = tmp_path / "multi.txt"
        prompt_file.write_text(content, encoding="utf-8")

        with patch("custom_components.neuralbridge.prompts_loader._PROMPTS_DIR", tmp_path):
            result = load_prompt("multi.txt")

        assert result == content
