"""Translation completeness tests for NeuralBridge language files.

Verifies that every key present in the reference file (``en_gb.yaml``) also
exists and is non-empty in all other language files found in the
``languages/`` directory.  This ensures that partially-translated files are
caught early and no runtime fallback goes unnoticed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

# Absolute path to the languages directory so tests run from any working dir
_LANGUAGES_DIR: Path = (
    Path(__file__).parent.parent.parent / "custom_components" / "neuralbridge" / "languages"
)
_REFERENCE_FILE: Path = _LANGUAGES_DIR / "en_gb.yaml"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_yaml(file_path: Path) -> dict[str, Any]:
    """Load a YAML file and return its content as a dictionary.

    Args:
        file_path: Absolute path to the YAML file.

    Returns:
        Parsed content, guaranteed to be a dict (empty if file is invalid).
    """
    with file_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _collect_leaf_paths(data: Any, prefix: str = "") -> list[str]:
    """Recursively collect all dot-separated key paths that lead to a leaf value.

    Leaf values are any non-dict values (strings, numbers, booleans, etc.).

    Args:
        data: The (nested) object to traverse.
        prefix: Accumulated dot-separated key path from parent calls.

    Returns:
        Sorted list of dot-separated key path strings.
    """
    paths: list[str] = []
    if not isinstance(data, dict):
        return paths

    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.extend(_collect_leaf_paths(value, path))
        else:
            paths.append(path)

    return sorted(paths)


def _get_value_at_path(data: dict[str, Any], dot_path: str) -> Any:
    """Retrieve the value at a dot-separated key path within a nested dict.

    Args:
        data: Dictionary to traverse.
        dot_path: Dot-separated key path (e.g. ``"responses.fallback"``).

    Returns:
        The value at the path, or ``None`` if the path does not exist.
    """
    node: Any = data
    for key in dot_path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
        if node is None:
            return None
    return node


def _non_reference_language_files() -> list[Path]:
    """Return all language YAML files excluding the reference (en_gb.yaml).

    Returns:
        Sorted list of Path objects for non-reference language files.
    """
    return sorted(f for f in _LANGUAGES_DIR.glob("*.yaml") if f.name != _REFERENCE_FILE.name)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def reference_data() -> dict[str, Any]:
    """Load and return the en_gb.yaml reference data."""
    return _load_yaml(_REFERENCE_FILE)


@pytest.fixture(scope="module")
def reference_leaf_paths(reference_data: dict[str, Any]) -> list[str]:
    """Return the sorted list of all leaf key paths in the reference file."""
    return _collect_leaf_paths(reference_data)


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_reference_file_exists() -> None:
    """en_gb.yaml must exist and be a valid non-empty YAML mapping."""
    assert _REFERENCE_FILE.is_file(), f"Reference language file not found: {_REFERENCE_FILE}"
    data = _load_yaml(_REFERENCE_FILE)
    assert data, "en_gb.yaml is empty or not a valid YAML mapping"


def test_languages_directory_exists() -> None:
    """The languages directory must exist."""
    assert _LANGUAGES_DIR.is_dir(), f"Languages directory not found: {_LANGUAGES_DIR}"


def test_at_least_one_additional_language() -> None:
    """At least one language file other than en_gb.yaml must exist."""
    extra = _non_reference_language_files()
    assert extra, "No additional language files found in languages/ directory"


@pytest.mark.parametrize(
    "lang_file",
    _non_reference_language_files(),
    ids=lambda f: f.stem,
)
def test_language_file_is_valid_yaml(lang_file: Path) -> None:
    """Each language file must be a valid, non-empty YAML mapping.

    Args:
        lang_file: Path to the language YAML file under test.
    """
    data = _load_yaml(lang_file)
    assert isinstance(data, dict) and data, f"{lang_file.name}: must be a non-empty YAML mapping"


@pytest.mark.parametrize(
    "lang_file",
    _non_reference_language_files(),
    ids=lambda f: f.stem,
)
def test_language_metadata_present(lang_file: Path) -> None:
    """Each language file must declare language.name and language.code.

    Args:
        lang_file: Path to the language YAML file under test.
    """
    data = _load_yaml(lang_file)
    lang_meta = data.get("language", {})
    assert isinstance(lang_meta, dict), f"{lang_file.name}: 'language' section must be a mapping"
    assert lang_meta.get("name"), f"{lang_file.name}: 'language.name' is missing or empty"
    assert lang_meta.get("code"), f"{lang_file.name}: 'language.code' is missing or empty"


@pytest.mark.parametrize(
    "lang_file",
    _non_reference_language_files(),
    ids=lambda f: f.stem,
)
def test_translation_completeness(
    lang_file: Path,
    reference_leaf_paths: list[str],
) -> None:
    """Every key in en_gb.yaml must be present and non-empty in each language file.

    Missing or empty keys indicate an incomplete translation.  Runtime fallback
    to en_gb will cover them, but they should still be flagged for translators.

    Args:
        lang_file: Path to the language YAML file under test.
        reference_leaf_paths: All leaf key paths from the en_gb.yaml reference.
    """
    data = _load_yaml(lang_file)

    missing: list[str] = []
    empty: list[str] = []

    for path in reference_leaf_paths:
        value = _get_value_at_path(data, path)
        if value is None:
            missing.append(path)
        elif isinstance(value, str) and not value.strip():
            empty.append(path)

    issues: list[str] = []
    if missing:
        issues.append(
            f"  Missing ({len(missing)} key(s)):\n" + "\n".join(f"    - {k}" for k in missing)
        )
    if empty:
        issues.append(f"  Empty ({len(empty)} key(s)):\n" + "\n".join(f"    - {k}" for k in empty))

    assert not issues, f"{lang_file.name} has incomplete translations:\n" + "\n".join(issues)


@pytest.mark.parametrize(
    "lang_file",
    _non_reference_language_files(),
    ids=lambda f: f.stem,
)
def test_no_extra_keys(
    lang_file: Path,
    reference_leaf_paths: list[str],
) -> None:
    """Language files must not define keys that do not exist in en_gb.yaml.

    Extra keys are a sign of stale translations or typos in key names.

    Args:
        lang_file: Path to the language YAML file under test.
        reference_leaf_paths: All leaf key paths from the en_gb.yaml reference.
    """
    data = _load_yaml(lang_file)
    reference_set = set(reference_leaf_paths)
    lang_paths = set(_collect_leaf_paths(data))
    extra = sorted(lang_paths - reference_set)

    assert not extra, (
        f"{lang_file.name} contains {len(extra)} key(s) not present in en_gb.yaml "
        f"(possible typos or stale entries):\n" + "\n".join(f"  - {k}" for k in extra)
    )
