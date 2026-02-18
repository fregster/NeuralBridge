"""Language file loader for NeuralBridge.

Loads YAML language files from the ``languages/`` sub-directory and provides
helpers for retrieving localised strings.  All loaded data is cached in-process
so disk I/O only occurs once per language per interpreter lifetime.

Fallback behaviour
------------------
1. If the requested language file does not exist or fails to load, the default
   language (``en_gb``) is used for **all** keys.
2. If a specific key is missing in the requested language file but the file
   itself loaded successfully, the default language is used as a **key-level**
   fallback, so partially-translated files degrade gracefully.
3. If the key is also absent in the default language, ``default`` (empty string
   by default) is returned.

Usage::

    from .languages_loader import get_string, list_available_languages

    text = get_string("en_gb", "responses", "fallback")
    languages = list_available_languages()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

_LOGGER = logging.getLogger(__name__)

_LANGUAGES_DIR: Path = Path(__file__).parent / "languages"
_DEFAULT_LANGUAGE: str = "en_gb"

# Module-level cache: lower-cased language_code -> loaded YAML dict (empty on error)
_cache: dict[str, dict[str, Any]] = {}


def _load_language_file(language_code: str) -> dict[str, Any]:
    """Load and parse a language YAML file from disk.

    Args:
        language_code: Normalised (lower-case) language code, e.g. ``"en_gb"``.

    Returns:
        Parsed YAML data as a dictionary, or an empty dict if the file is
        missing or cannot be parsed.
    """
    file_path = _LANGUAGES_DIR / f"{language_code}.yaml"
    if not file_path.is_file():
        _LOGGER.warning("NeuralBridge: language file not found: %s", file_path)
        return {}
    try:
        with file_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as err:
        _LOGGER.error("NeuralBridge: failed to load language file %s: %s", file_path, err)
        return {}


def get_language_data(language_code: str) -> dict[str, Any]:
    """Return the language data dict for *language_code*, using the cache.

    Falls back to the default language (``en_gb``) only when the requested
    language file is not found or could not be loaded at all.  For missing
    individual keys within a partially-translated file use :func:`get_string`,
    which applies key-level fallback.

    Args:
        language_code: Language code string (case-insensitive), e.g. ``"en_GB"``.

    Returns:
        Language data dictionary (may be empty if both the requested language
        and the default language fail to load).
    """
    code = language_code.lower()
    if code not in _cache:
        _cache[code] = _load_language_file(code)

    if not _cache[code] and code != _DEFAULT_LANGUAGE:
        if _DEFAULT_LANGUAGE not in _cache:
            _cache[_DEFAULT_LANGUAGE] = _load_language_file(_DEFAULT_LANGUAGE)
        return _cache[_DEFAULT_LANGUAGE]

    return _cache[code]


def _resolve_keys(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Navigate a nested dictionary by a sequence of keys.

    Args:
        data: Dictionary to traverse.
        keys: Sequence of keys forming the path into the hierarchy.

    Returns:
        The string value at the resolved path, or ``None`` if the path does
        not exist or does not resolve to a string.
    """
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
        if node is None:
            return None
    return node if isinstance(node, str) else None


def get_string(language_code: str, *keys: str, default: str = "") -> str:
    """Retrieve a localised string with key-level fallback to British English.

    Resolution order:
    1. Key path in the requested language file.
    2. Key path in ``en_gb.yaml`` (default language) if step 1 misses.
    3. *default* (empty string) if the key is absent everywhere.

    Example::

        get_string("fr", "responses", "fallback")
        # → "J'ai des difficultés à me connecter..."  (if translated)
        # → "I'm having trouble..."  (en_gb fallback if not yet translated)

    Args:
        language_code: Language code (case-insensitive), e.g. ``"fr"``.
        *keys: One or more keys forming the path into the YAML hierarchy.
        default: Value returned when the key path is absent in all languages.

    Returns:
        Localised string, or *default* if the path does not resolve.
    """
    code = language_code.lower()

    result = _resolve_keys(get_language_data(code), keys)
    if result is not None:
        return result

    # Key-level fallback to default language
    if code != _DEFAULT_LANGUAGE:
        fallback = _resolve_keys(get_language_data(_DEFAULT_LANGUAGE), keys)
        if fallback is not None:
            _LOGGER.debug(
                "NeuralBridge: key %r missing in '%s', using en_gb fallback",
                ".".join(keys),
                code,
            )
            return fallback

    return default


def list_available_languages() -> list[dict[str, str]]:
    """Return metadata for every language file found in the ``languages/`` directory.

    Returns:
        List of dicts, each containing ``"code"`` (file stem) and ``"name"``
        (display name from the YAML ``language.name`` key).  Sorted by file name.
    """
    if not _LANGUAGES_DIR.is_dir():
        _LOGGER.warning("NeuralBridge: languages directory not found: %s", _LANGUAGES_DIR)
        return []

    languages: list[dict[str, str]] = []
    for yaml_file in sorted(_LANGUAGES_DIR.glob("*.yaml")):
        code = yaml_file.stem
        data = get_language_data(code)
        lang_info = data.get("language", {})
        name: str = lang_info.get("name", code) if isinstance(lang_info, dict) else code
        languages.append({"code": code, "name": name if isinstance(name, str) else code})
    return languages
