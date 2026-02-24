"""Prompt file loader for NeuralBridge.

Loads prompt templates from the ``prompts/`` directory located alongside this
module.  A *fallback* string is returned if the file cannot be read, so that
the integration keeps working even when a prompt file is accidentally missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

_PROMPTS_DIR: Path = Path(__file__).parent / "prompts"


def load_prompt(filename: str, fallback: str = "") -> str:
    """Load a prompt template from the prompts directory.

    The file is read from ``custom_components/neuralbridge/prompts/<filename>``.
    Any trailing whitespace (including the final newline) is stripped so that
    callers receive a clean string without an unexpected trailing newline.

    Args:
        filename: Filename inside ``custom_components/neuralbridge/prompts/``,
            e.g. ``"router_classification.txt"``.
        fallback: String to return when the file cannot be read.  Defaults to
            an empty string.

    Returns:
        Prompt text with trailing whitespace stripped, or *fallback* on error.
    """
    path = _PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8").rstrip()
    except OSError:
        _LOGGER.warning(
            "Could not load prompt file %s; using built-in fallback",
            path,
        )
        return fallback
