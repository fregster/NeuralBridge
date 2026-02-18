"""Unit tests for the languages_loader module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

import custom_components.neuralbridge.languages_loader as ll
from custom_components.neuralbridge.languages_loader import (
    _load_language_file,
    _resolve_keys,
    get_language_data,
    get_string,
    list_available_languages,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_cache() -> dict:
    """Return an empty cache dict for use with patch.dict(..., clear=True)."""
    return {}


# ---------------------------------------------------------------------------
# _load_language_file
# ---------------------------------------------------------------------------


class TestLoadLanguageFile:
    """Tests for _load_language_file."""

    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        """Returns {} when the language file does not exist on disk."""
        with patch("custom_components.neuralbridge.languages_loader._LANGUAGES_DIR", tmp_path):
            result = _load_language_file("no_such_lang")
        assert result == {}

    def test_os_error_returns_empty_dict(self, tmp_path: Path) -> None:
        """Returns {} when an OSError occurs while reading the file."""
        # Create a real file so is_file() returns True, then make open() fail.
        lang_file = tmp_path / "bad_lang.yaml"
        lang_file.write_bytes(b"key: value")

        mock_path = MagicMock(spec=Path)
        mock_path.is_file.return_value = True
        mock_path.open.side_effect = OSError("permission denied")

        mock_dir = MagicMock(spec=Path)
        mock_dir.__truediv__ = MagicMock(return_value=mock_path)

        with patch("custom_components.neuralbridge.languages_loader._LANGUAGES_DIR", mock_dir):
            result = _load_language_file("bad_lang")
        assert result == {}

    def test_yaml_error_returns_empty_dict(self, tmp_path: Path) -> None:
        """Returns {} when yaml.safe_load raises YAMLError."""
        lang_file = tmp_path / "broken.yaml"
        lang_file.write_bytes(b"key: value")

        with (
            patch("custom_components.neuralbridge.languages_loader._LANGUAGES_DIR", tmp_path),
            patch(
                "custom_components.neuralbridge.languages_loader.yaml.safe_load",
                side_effect=yaml.YAMLError("bad yaml"),
            ),
        ):
            result = _load_language_file("broken")
        assert result == {}


# ---------------------------------------------------------------------------
# _resolve_keys
# ---------------------------------------------------------------------------


class TestResolveKeys:
    """Tests for _resolve_keys."""

    def test_returns_string_at_path(self) -> None:
        """Returns the string value at the resolved key path."""
        data = {"responses": {"greeting": "Hello"}}
        assert _resolve_keys(data, ("responses", "greeting")) == "Hello"

    def test_intermediate_key_missing_returns_none(self) -> None:
        """Returns None when an intermediate key is absent (node becomes None)."""
        data = {"responses": {"greeting": "Hello"}}
        # "missing_key" is not in the dict → node is None after first step
        assert _resolve_keys(data, ("responses", "missing_key", "deep")) is None

    def test_intermediate_node_not_dict_returns_none(self) -> None:
        """Returns None when traversal hits a non-dict node before consuming all keys."""
        data = {"responses": "flat_string"}
        # "responses" resolves to a string, not a dict — cannot go deeper
        assert _resolve_keys(data, ("responses", "nested_key")) is None

    def test_leaf_is_not_string_returns_none(self) -> None:
        """Returns None when the resolved leaf is not a string (e.g. a list)."""
        data = {"responses": {"list_val": [1, 2, 3]}}
        assert _resolve_keys(data, ("responses", "list_val")) is None


# ---------------------------------------------------------------------------
# get_language_data — fallback to default language
# ---------------------------------------------------------------------------


class TestGetLanguageData:
    """Tests for get_language_data."""

    def test_falls_back_to_default_when_language_fails(self) -> None:
        """When a language file is empty, get_language_data returns the default data."""
        default_data = {"responses": {"greeting": "Hello from en_gb"}}

        # Inject: "xx" -> {} (failed load), "en_gb" -> default_data
        with patch.dict(
            ll._cache,
            {"xx": {}, "en_gb": default_data},
            clear=False,
        ):
            result = get_language_data("xx")

        assert result == default_data

    def test_falls_back_and_loads_default_if_not_cached(self) -> None:
        """When language fails and en_gb is not cached, it loads and caches en_gb."""
        default_data = {"responses": {"key": "default value"}}

        with (
            patch.dict(ll._cache, {}, clear=True),
            patch.object(ll, "_load_language_file") as mock_load,
        ):
            # "xx" fails; "en_gb" succeeds
            mock_load.side_effect = lambda code: default_data if code == "en_gb" else {}
            result = get_language_data("xx")

        assert result == default_data


# ---------------------------------------------------------------------------
# get_string — key-level fallback
# ---------------------------------------------------------------------------


class TestGetString:
    """Tests for get_string."""

    def test_returns_value_from_requested_language(self) -> None:
        """Returns the string from the requested language when the key exists."""
        with patch.dict(
            ll._cache,
            {"fr": {"responses": {"greeting": "Bonjour"}}},
            clear=False,
        ):
            assert get_string("fr", "responses", "greeting") == "Bonjour"

    def test_key_level_fallback_to_en_gb(self) -> None:
        """Falls back to en_gb when the key is missing in the requested language."""
        with patch.dict(
            ll._cache,
            {
                "de": {"responses": {}},  # key missing in German
                "en_gb": {"responses": {"greeting": "Hello"}},
            },
            clear=False,
        ):
            result = get_string("de", "responses", "greeting")
        assert result == "Hello"

    def test_returns_default_when_key_absent_everywhere(self) -> None:
        """Returns the default argument when the key is absent in all languages."""
        with patch.dict(
            ll._cache,
            {
                "de": {},
                "en_gb": {},
            },
            clear=False,
        ):
            result = get_string("de", "responses", "no_such_key", default="fallback")
        assert result == "fallback"


# ---------------------------------------------------------------------------
# list_available_languages
# ---------------------------------------------------------------------------


class TestListAvailableLanguages:
    """Tests for list_available_languages."""

    def test_returns_empty_list_when_dir_missing(self, tmp_path: Path) -> None:
        """Returns [] when the languages directory does not exist."""
        nonexistent = tmp_path / "no_languages"

        with patch("custom_components.neuralbridge.languages_loader._LANGUAGES_DIR", nonexistent):
            result = list_available_languages()

        assert result == []

    def test_returns_language_entries_for_yaml_files(self, tmp_path: Path) -> None:
        """Returns one entry per YAML file with code and name from the file."""
        lang_file = tmp_path / "test_lang.yaml"
        lang_file.write_text("language:\n  name: Test Language\n", encoding="utf-8")

        with (
            patch("custom_components.neuralbridge.languages_loader._LANGUAGES_DIR", tmp_path),
            patch.dict(ll._cache, {}, clear=True),
        ):
            result = list_available_languages()

        assert len(result) == 1
        assert result[0]["code"] == "test_lang"
        assert result[0]["name"] == "Test Language"

    def test_uses_code_when_language_name_missing(self, tmp_path: Path) -> None:
        """Falls back to the language code as name when 'language.name' is absent."""
        lang_file = tmp_path / "nolang.yaml"
        lang_file.write_text("responses:\n  greeting: Hi\n", encoding="utf-8")

        with (
            patch("custom_components.neuralbridge.languages_loader._LANGUAGES_DIR", tmp_path),
            patch.dict(ll._cache, {}, clear=True),
        ):
            result = list_available_languages()

        assert len(result) == 1
        assert result[0]["code"] == "nolang"
        assert result[0]["name"] == "nolang"
