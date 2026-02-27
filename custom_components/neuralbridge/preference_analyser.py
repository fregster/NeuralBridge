"""Preference detection for NeuralBridge (Feature 15 — Adaptive Preference Learning).

Analyses user query text to detect expressed preferences using lightweight
regex patterns.  No LLM or network calls are made — all detection is
synchronous and runs within the HA event loop budget.

Returns a :class:`PreferenceSuggestion` or ``None``.

Detectors run in this priority order:

    1. Correction signal (emphatic all-caps, "I meant") — auto-store
    2. Source preference  ("from BBC", "on CNN")
    3. Location disambiguation  ("in London England", "in Paris France")
    4. Format preference  ("24 hour", "Celsius", "miles")

Only the first matching detector result is returned per call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .preference_memory import PREF_CATEGORY_FORMAT, PREF_CATEGORY_LOCATION, PREF_CATEGORY_SOURCE

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class PreferenceSuggestion:
    """A detected preference that the system may offer to store.

    Attributes:
        key:             Preference key (e.g. ``"news_source"``, ``"location_london"``).
        value:           Detected value (e.g. ``"BBC"``, ``"London, England"``).
        category:        One of ``"source"``, ``"location"``, ``"format"``.
        confidence:      Detection confidence 0.0-1.0.
        auto_store:      When True the preference is stored immediately without
                         asking the user (used for emphatic correction signals).
        suggestion_text: The message suffix shown to the user (or the "Got it"
                         notice when ``auto_store`` is True).
    """

    key: str
    value: str
    category: str
    confidence: float
    auto_store: bool
    suggestion_text: str


# ---------------------------------------------------------------------------
# Known news sources
# ---------------------------------------------------------------------------

# Each tuple: (match_text, canonical_display_name)
_NEWS_SOURCES: tuple[tuple[str, str], ...] = (
    ("BBC", "BBC"),
    ("CNN", "CNN"),
    ("Sky News", "Sky News"),
    ("ITV News", "ITV News"),
    ("ITV", "ITV News"),
    ("The Guardian", "The Guardian"),
    ("Guardian", "The Guardian"),
    ("Daily Mail", "Daily Mail"),
    ("The Times", "The Times"),
    ("Reuters", "Reuters"),
    ("AP News", "AP News"),
    ("AP", "AP News"),
    ("Fox News", "Fox News"),
    ("NBC News", "NBC News"),
    ("NBC", "NBC News"),
    ("ABC News", "ABC News"),
    ("Bloomberg", "Bloomberg"),
    ("Sky Sports", "Sky Sports"),
)

# Canonical map: lowercase match text → display name
_SOURCE_CANONICAL: dict[str, str] = {src.lower(): display for src, display in _NEWS_SOURCES}

# Build compiled regex for "from <source>" or "on <source>"
_SOURCE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:from|on)\s+(" + "|".join(re.escape(src) for src, _ in _NEWS_SOURCES) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Location qualifiers
# ---------------------------------------------------------------------------

_COUNTRY_QUALIFIERS: tuple[str, ...] = (
    "England",
    "Scotland",
    "Wales",
    "Ireland",
    "Northern Ireland",
    "UK",
    "United Kingdom",
    "Britain",
    "Great Britain",
    "USA",
    "US",
    "America",
    "United States",
    "France",
    "Germany",
    "Spain",
    "Italy",
    "Australia",
    "Canada",
    "India",
    "China",
    "Japan",
    "Brazil",
    "Mexico",
    "Poland",
    "Portugal",
    "Netherlands",
    "Belgium",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
)

# Matches "in <City> <Qualifier>" — city may be one or two title-cased words
_LOCATION_PATTERN: re.Pattern[str] = re.compile(
    r"\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+("
    + "|".join(re.escape(q) for q in sorted(_COUNTRY_QUALIFIERS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Correction signal patterns
# ---------------------------------------------------------------------------

# All-caps word of 2+ letters (excluding common abbreviations that are benign)
_ALLCAPS_RE: re.Pattern[str] = re.compile(r"\b[A-Z]{2,}\b")
# Common ALL-CAPS words that should NOT trigger the correction detector
_ALLCAPS_IGNORE: frozenset[str] = frozenset(
    {"BBC", "CNN", "ITV", "NBC", "ABC", "AP", "TV", "UK", "US", "AM", "PM", "OK"}
)

_CORRECTION_PHRASE_RE: re.Pattern[str] = re.compile(
    r"\b(?:I\s+meant|actually\b|I\s+said|no[,]?\s+I\s+mean|not\s+\w+[,]\s+\w+)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Format preference rules
# ---------------------------------------------------------------------------

# Each entry: (compiled pattern, preference key, canonical value)
_FORMAT_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b24[- ]?hour\b", re.IGNORECASE), "time_format", "24h"),
    (re.compile(r"\b12[- ]?hour\b", re.IGNORECASE), "time_format", "12h"),
    (re.compile(r"\bcelsius\b|\bcentigrade\b", re.IGNORECASE), "temperature_unit", "celsius"),
    (re.compile(r"\bfahrenheit\b", re.IGNORECASE), "temperature_unit", "fahrenheit"),
    # "miles" but not "miles per hour" to avoid false positives on speed queries
    (re.compile(r"\bmiles?\b(?!\s+per\s+hour)", re.IGNORECASE), "distance_unit", "miles"),
    (re.compile(r"\bkilo(?:metres?|meters?)\b|\bkm\b", re.IGNORECASE), "distance_unit", "km"),
)

# ---------------------------------------------------------------------------
# Public analyser
# ---------------------------------------------------------------------------


class PreferenceAnalyser:
    """Detect expressed preferences in user query text.

    All methods are pure functions (no I/O, no mutable state).  Create a
    single instance and reuse it — it is safe to share across calls.

    Example::

        analyser = PreferenceAnalyser()
        suggestion = analyser.detect("What are the latest headlines from the BBC?")
        # PreferenceSuggestion(key="news_source", value="BBC", ...)
    """

    def detect(self, text: str) -> PreferenceSuggestion | None:
        """Analyse *text* and return the first detected preference.

        Runs detectors in priority order:
        correction → source → location → format.

        Args:
            text: Raw user query text.

        Returns:
            A :class:`PreferenceSuggestion` or ``None`` when no preference
            signal is found.
        """
        for detector in (
            self._detect_correction_with_location,
            self._detect_source_preference,
            self._detect_location_disambiguation,
            self._detect_format_preference,
        ):
            result = detector(text)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------
    # Correction detector
    # ------------------------------------------------------------------

    def _detect_correction_with_location(self, text: str) -> PreferenceSuggestion | None:
        """Detect an emphatic correction signal paired with location context.

        Triggers when the query contains an all-caps non-benign word or a
        correction phrase (e.g. "I meant"), AND a location pattern is also
        present.  When both are found the preference is auto-stored without
        asking the user.

        Args:
            text: Raw user query text.

        Returns:
            A :class:`PreferenceSuggestion` with ``auto_store=True``, or None.
        """
        is_emphatic = self._has_emphatic_signal(text)
        if not is_emphatic:
            return None
        loc = self._detect_location_disambiguation(text)
        if loc is not None:
            return PreferenceSuggestion(
                key=loc.key,
                value=loc.value,
                category=loc.category,
                confidence=0.85,
                auto_store=True,
                suggestion_text=f"✅ Got it — I'll default to {loc.value} from now on.",
            )
        return None

    @staticmethod
    def _has_emphatic_signal(text: str) -> bool:
        """Return True when *text* contains an all-caps non-benign or correction phrase.

        Args:
            text: User query text.

        Returns:
            True when an emphatic signal is detected.
        """
        caps_matches = _ALLCAPS_RE.findall(text)
        non_benign = [w for w in caps_matches if w not in _ALLCAPS_IGNORE]
        if non_benign:
            return True
        return bool(_CORRECTION_PHRASE_RE.search(text))

    # ------------------------------------------------------------------
    # Source detector
    # ------------------------------------------------------------------

    def _detect_source_preference(self, text: str) -> PreferenceSuggestion | None:
        """Detect an explicit news source preference.

        Matches the pattern ``"from <source>"`` or ``"on <source>"`` where
        ``<source>`` is a known news outlet (e.g. BBC, CNN, Sky News).

        Args:
            text: Raw user query text.

        Returns:
            A :class:`PreferenceSuggestion` for ``"news_source"``, or None.
        """
        match = _SOURCE_PATTERN.search(text)
        if match is None:
            return None
        source = _SOURCE_CANONICAL.get(match.group(1).lower(), match.group(1))
        return PreferenceSuggestion(
            key="news_source",
            value=source,
            category=PREF_CATEGORY_SOURCE,
            confidence=0.8,
            auto_store=False,
            suggestion_text=(f"💡 Next time would you like me to default to {source} for news?"),
        )

    # ------------------------------------------------------------------
    # Location detector
    # ------------------------------------------------------------------

    def _detect_location_disambiguation(self, text: str) -> PreferenceSuggestion | None:
        """Detect a location disambiguation qualifier.

        Matches ``"in <City> <Country>"`` patterns (e.g. "in London England",
        "in Paris France").  The city name may be one or two words.

        Args:
            text: Raw user query text.

        Returns:
            A :class:`PreferenceSuggestion` for ``"location_<city_slug>"``, or None.
        """
        match = _LOCATION_PATTERN.search(text)
        if match is None:
            return None
        city = match.group(1).strip().title()
        qualifier = match.group(2).strip()
        city_slug = city.lower().replace(" ", "_")
        full_location = f"{city}, {qualifier}"
        return PreferenceSuggestion(
            key=f"location_{city_slug}",
            value=full_location,
            category=PREF_CATEGORY_LOCATION,
            confidence=0.8,
            auto_store=False,
            suggestion_text=(
                f"💡 Next time would you like me to default to {full_location} "
                f"when you say {city!r}?"
            ),
        )

    # ------------------------------------------------------------------
    # Format detector
    # ------------------------------------------------------------------

    def _detect_format_preference(self, text: str) -> PreferenceSuggestion | None:
        """Detect a unit or format preference (time, temperature, distance).

        Checks against the ``_FORMAT_RULES`` table.  Returns on the first
        matching rule.

        Args:
            text: Raw user query text.

        Returns:
            A :class:`PreferenceSuggestion` for the matched format key, or None.
        """
        for pattern, key, value in _FORMAT_RULES:
            if pattern.search(text):
                display = value.replace("_", " ")
                return PreferenceSuggestion(
                    key=key,
                    value=value,
                    category=PREF_CATEGORY_FORMAT,
                    confidence=0.7,
                    auto_store=False,
                    suggestion_text=f"💡 Would you like me to always use {display}?",
                )
        return None
