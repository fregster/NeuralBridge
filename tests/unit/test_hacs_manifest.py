"""HACS manifest validation tests for NeuralBridge.

Validates that ``hacs.json`` conforms to the HACS manifest schema before push,
catching errors that would otherwise only surface during GitHub Actions CI.

Reference: https://hacs.xyz/docs/publish/include#check-hacs-manifest
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT: Path = Path(__file__).parent.parent.parent
_HACS_MANIFEST: Path = _REPO_ROOT / "hacs.json"

# Keys permitted by the HACS manifest schema.
# Any key outside this set causes an "extra keys not allowed" validation failure.
# Reference: https://hacs.xyz/docs/publish/include#check-hacs-manifest
_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "content_in_root",
        "zip_release",
        "filename",
        "render_readme",
        "hide_default_branch",
        "homeassistant",
        "country",
        "category",
    }
)

_REQUIRED_KEYS: frozenset[str] = frozenset({"name"})

_BOOL_KEYS: frozenset[str] = frozenset(
    {"content_in_root", "zip_release", "render_readme", "hide_default_branch"}
)

_STR_KEYS: frozenset[str] = frozenset({"name", "filename", "homeassistant", "category"})

_LIST_KEYS: frozenset[str] = frozenset({"country"})


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_hacs_manifest() -> dict[str, Any]:
    """Load and parse ``hacs.json`` from the repository root.

    Returns:
        Parsed JSON content as a dictionary.
    """
    with _HACS_MANIFEST.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data  # type: ignore[no-any-return]


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_hacs_manifest_exists() -> None:
    """hacs.json must exist at the repository root."""
    assert _HACS_MANIFEST.is_file(), f"hacs.json not found at: {_HACS_MANIFEST}"


def test_hacs_manifest_is_valid_json() -> None:
    """hacs.json must be parseable as a valid JSON object."""
    data = _load_hacs_manifest()
    assert isinstance(data, dict), "hacs.json root must be a JSON object, not a list or scalar"


def test_hacs_manifest_required_keys() -> None:
    """hacs.json must contain all required keys."""
    data = _load_hacs_manifest()
    missing = _REQUIRED_KEYS - data.keys()
    assert not missing, f"hacs.json is missing required key(s): {sorted(missing)}"


def test_hacs_manifest_no_extra_keys() -> None:
    """hacs.json must not contain keys outside the HACS manifest schema.

    Extra keys produce an ``extra keys not allowed`` validation error in HACS CI.
    Common mistake: adding ``domains`` (a Home Assistant manifest field) to hacs.json.

    See: https://hacs.xyz/docs/publish/include#check-hacs-manifest
    """
    data = _load_hacs_manifest()
    extra = data.keys() - _ALLOWED_KEYS
    assert not extra, (
        f"hacs.json contains key(s) not allowed by the HACS schema: {sorted(extra)}\n"
        f"Allowed keys are: {sorted(_ALLOWED_KEYS)}"
    )


@pytest.mark.parametrize("key", sorted(_BOOL_KEYS))
def test_hacs_manifest_bool_fields(key: str) -> None:
    """Boolean fields in hacs.json must have ``bool`` values when present.

    Args:
        key: The boolean field name to validate.
    """
    data = _load_hacs_manifest()
    if key not in data:
        pytest.skip(f"Optional field '{key}' not present in hacs.json")
    assert isinstance(data[key], bool), (
        f"hacs.json field '{key}' must be a boolean, got {type(data[key]).__name__!r}"
    )


@pytest.mark.parametrize("key", sorted(_STR_KEYS))
def test_hacs_manifest_string_fields(key: str) -> None:
    """String fields in hacs.json must have non-empty string values when present.

    Args:
        key: The string field name to validate.
    """
    data = _load_hacs_manifest()
    if key not in data:
        pytest.skip(f"Optional field '{key}' not present in hacs.json")
    assert isinstance(data[key], str) and data[key].strip(), (
        f"hacs.json field '{key}' must be a non-empty string"
    )


@pytest.mark.parametrize("key", sorted(_LIST_KEYS))
def test_hacs_manifest_list_fields(key: str) -> None:
    """List fields in hacs.json must be non-empty lists of strings when present.

    Args:
        key: The list field name to validate.
    """
    data = _load_hacs_manifest()
    if key not in data:
        pytest.skip(f"Optional field '{key}' not present in hacs.json")
    value = data[key]
    assert isinstance(value, list) and value, (
        f"hacs.json field '{key}' must be a non-empty list"
    )
    assert all(isinstance(item, str) for item in value), (
        f"hacs.json field '{key}' must contain only strings"
    )
