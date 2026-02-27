"""Unit tests for preference_analyser.py (Feature 15 — Adaptive Preference Learning)."""

from __future__ import annotations

import pytest

from custom_components.neuralbridge.preference_analyser import (
    PreferenceAnalyser,
    PreferenceSuggestion,
)
from custom_components.neuralbridge.preference_memory import (
    PREF_CATEGORY_FORMAT,
    PREF_CATEGORY_LOCATION,
    PREF_CATEGORY_SOURCE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def analyser() -> PreferenceAnalyser:
    """Return a fresh PreferenceAnalyser instance."""
    return PreferenceAnalyser()


# ---------------------------------------------------------------------------
# detect() — top-level dispatcher
# ---------------------------------------------------------------------------


def test_detect_returns_none_for_unrecognised_text(analyser: PreferenceAnalyser) -> None:
    """detect() returns None when no preference signal is found."""
    result = analyser.detect("What is the weather today?")
    assert result is None


def test_detect_returns_none_for_empty_string(analyser: PreferenceAnalyser) -> None:
    """detect() returns None for an empty string."""
    result = analyser.detect("")
    assert result is None


def test_detect_source_preference_from_bbc(analyser: PreferenceAnalyser) -> None:
    """detect() recognises 'from BBC' as a news source preference."""
    result = analyser.detect("Get me the latest news from BBC")
    assert result is not None
    assert result.category == PREF_CATEGORY_SOURCE
    assert result.value == "BBC"
    assert result.key == "news_source"
    assert result.auto_store is False
    assert result.confidence > 0.5


def test_detect_source_preference_on_cnn(analyser: PreferenceAnalyser) -> None:
    """detect() recognises 'on CNN' as a news source preference."""
    result = analyser.detect("Show me headlines on CNN")
    assert result is not None
    assert result.value == "CNN"
    assert result.category == PREF_CATEGORY_SOURCE


def test_detect_location_disambiguation(analyser: PreferenceAnalyser) -> None:
    """detect() recognises 'in London England' as a location preference."""
    result = analyser.detect("What is the weather in London England?")
    assert result is not None
    assert result.category == PREF_CATEGORY_LOCATION
    assert "London" in result.value
    assert result.auto_store is False


def test_detect_format_preference_celsius(analyser: PreferenceAnalyser) -> None:
    """detect() recognises 'celsius' as a format preference."""
    result = analyser.detect("Show me the temperature in celsius")
    assert result is not None
    assert result.category == PREF_CATEGORY_FORMAT
    assert result.key == "temperature_unit"
    assert result.value == "celsius"


def test_detect_format_preference_fahrenheit(analyser: PreferenceAnalyser) -> None:
    """detect() recognises 'fahrenheit' as a format preference."""
    result = analyser.detect("I prefer the temperature in fahrenheit")
    assert result is not None
    assert result.category == PREF_CATEGORY_FORMAT
    assert result.value == "fahrenheit"


def test_detect_format_preference_24_hour(analyser: PreferenceAnalyser) -> None:
    """detect() recognises '24 hour' as a time format preference."""
    result = analyser.detect("Use 24 hour format please")
    assert result is not None
    assert result.key == "time_format"
    assert result.value == "24h"


def test_detect_format_preference_12_hour(analyser: PreferenceAnalyser) -> None:
    """detect() recognises '12-hour' as a time format preference."""
    result = analyser.detect("I prefer 12-hour clock")
    assert result is not None
    assert result.key == "time_format"
    assert result.value == "12h"


def test_detect_format_preference_miles(analyser: PreferenceAnalyser) -> None:
    """detect() recognises 'miles' as a distance format preference."""
    result = analyser.detect("Show me distances in miles")
    assert result is not None
    assert result.key == "distance_unit"
    assert result.value == "miles"


def test_detect_format_preference_kilometres(analyser: PreferenceAnalyser) -> None:
    """detect() recognises 'kilometres' as a distance format preference."""
    result = analyser.detect("Show distances in kilometres")
    assert result is not None
    assert result.key == "distance_unit"
    assert result.value == "km"


def test_detect_format_preference_km(analyser: PreferenceAnalyser) -> None:
    """detect() recognises 'km' as a distance format preference."""
    result = analyser.detect("Use km for distances")
    assert result is not None
    assert result.key == "distance_unit"
    assert result.value == "km"


def test_detect_miles_does_not_trigger_on_miles_per_hour(analyser: PreferenceAnalyser) -> None:
    """detect() does NOT match 'miles per hour' to avoid false positives."""
    result = analyser.detect("The car is going 60 miles per hour")
    assert result is None or result.key != "distance_unit"


def test_detect_correction_with_location_auto_stores(analyser: PreferenceAnalyser) -> None:
    """detect() sets auto_store=True when all-caps + location are both present."""
    result = analyser.detect("I mean in LONDON England, not London Canada")
    assert result is not None
    assert result.auto_store is True
    assert result.category == PREF_CATEGORY_LOCATION


def test_detect_correction_no_location_returns_none(analyser: PreferenceAnalyser) -> None:
    """_detect_correction_with_location returns None when emphatic but no location pattern."""
    result = analyser._detect_correction_with_location("I ACTUALLY want tea not coffee")
    assert result is None


def test_detect_priority_correction_before_source(analyser: PreferenceAnalyser) -> None:
    """Correction detector runs before source detector."""
    # All-caps BBC + London England → correction (location) wins, not source
    result = analyser.detect("BBC NEWS in London England")
    # BBC is in ALLCAPS_IGNORE so no emphatic signal; location should trigger
    assert result is not None
    assert result.category == PREF_CATEGORY_LOCATION


def test_detect_location_in_paris_france(analyser: PreferenceAnalyser) -> None:
    """detect() recognises 'in Paris France' as a location preference."""
    result = analyser.detect("What is happening in Paris France?")
    assert result is not None
    assert result.category == PREF_CATEGORY_LOCATION
    assert "Paris" in result.value
    assert "France" in result.value


# ---------------------------------------------------------------------------
# _has_emphatic_signal
# ---------------------------------------------------------------------------


def test_has_emphatic_signal_all_caps_non_benign(analyser: PreferenceAnalyser) -> None:
    """_has_emphatic_signal returns True for a non-benign all-caps word."""
    assert analyser._has_emphatic_signal("I want LONDON not the other one") is True


def test_has_emphatic_signal_all_caps_benign_only(analyser: PreferenceAnalyser) -> None:
    """_has_emphatic_signal returns False when all all-caps words are benign."""
    # BBC, CNN, ITV, NBC, ABC, AP, TV, UK, US, AM, PM, OK are benign
    assert analyser._has_emphatic_signal("Watch BBC TV in the UK") is False


def test_has_emphatic_signal_correction_phrase(analyser: PreferenceAnalyser) -> None:
    """_has_emphatic_signal returns True for 'I meant'."""
    assert analyser._has_emphatic_signal("I meant London England") is True


def test_has_emphatic_signal_no_signal(analyser: PreferenceAnalyser) -> None:
    """_has_emphatic_signal returns False for plain text with no caps or correction."""
    assert analyser._has_emphatic_signal("What is the weather like today?") is False


def test_has_emphatic_signal_correction_phrase_actually(analyser: PreferenceAnalyser) -> None:
    """_has_emphatic_signal returns True for 'actually'."""
    assert analyser._has_emphatic_signal("I actually meant Paris France") is True


# ---------------------------------------------------------------------------
# _detect_source_preference
# ---------------------------------------------------------------------------


def test_detect_source_preference_sky_news(analyser: PreferenceAnalyser) -> None:
    """_detect_source_preference returns a suggestion for 'from Sky News'."""
    result = analyser._detect_source_preference("News from Sky News tonight")
    assert result is not None
    assert result.value == "Sky News"


def test_detect_source_preference_guardian(analyser: PreferenceAnalyser) -> None:
    """_detect_source_preference normalises 'Guardian' to 'The Guardian'."""
    result = analyser._detect_source_preference("I read news from Guardian")
    assert result is not None
    assert result.value == "The Guardian"


def test_detect_source_preference_no_match(analyser: PreferenceAnalyser) -> None:
    """_detect_source_preference returns None for unrecognised text."""
    result = analyser._detect_source_preference("Just give me any news")
    assert result is None


def test_detect_source_preference_itv(analyser: PreferenceAnalyser) -> None:
    """_detect_source_preference returns 'ITV News' for 'on ITV'."""
    result = analyser._detect_source_preference("What is on ITV now?")
    assert result is not None
    assert result.value == "ITV News"


def test_detect_source_preference_bloomberg(analyser: PreferenceAnalyser) -> None:
    """_detect_source_preference returns Bloomberg for 'from Bloomberg'."""
    result = analyser._detect_source_preference("Get the news from Bloomberg")
    assert result is not None
    assert result.value == "Bloomberg"


# ---------------------------------------------------------------------------
# _detect_location_disambiguation
# ---------------------------------------------------------------------------


def test_detect_location_disambiguation_london_england(analyser: PreferenceAnalyser) -> None:
    """_detect_location_disambiguation matches 'in London England'."""
    result = analyser._detect_location_disambiguation("weather in London England")
    assert result is not None
    assert result.key == "location_london"
    assert "England" in result.value


def test_detect_location_disambiguation_no_match(analyser: PreferenceAnalyser) -> None:
    """_detect_location_disambiguation returns None for no qualifier."""
    result = analyser._detect_location_disambiguation("What is in London?")
    assert result is None


def test_detect_location_disambiguation_two_word_city(analyser: PreferenceAnalyser) -> None:
    """_detect_location_disambiguation matches a two-word city name."""
    result = analyser._detect_location_disambiguation("weather in New York USA")
    assert result is not None
    assert "New York" in result.value


def test_detect_location_disambiguation_suggestion_text(analyser: PreferenceAnalyser) -> None:
    """Location suggestion_text mentions the city and qualifier."""
    result = analyser._detect_location_disambiguation("in Madrid Spain")
    assert result is not None
    assert "Madrid" in result.suggestion_text


# ---------------------------------------------------------------------------
# _detect_format_preference
# ---------------------------------------------------------------------------


def test_detect_format_preference_no_match(analyser: PreferenceAnalyser) -> None:
    """_detect_format_preference returns None for text with no format hint."""
    result = analyser._detect_format_preference("What is the weather today?")
    assert result is None


def test_detect_format_preference_suggestion_text(analyser: PreferenceAnalyser) -> None:
    """Format suggestion_text contains a user-friendly unit description."""
    result = analyser._detect_format_preference("I like celsius for temperatures")
    assert result is not None
    assert "celsius" in result.suggestion_text.lower()


# ---------------------------------------------------------------------------
# PreferenceSuggestion dataclass
# ---------------------------------------------------------------------------


def test_preference_suggestion_fields_accessible() -> None:
    """PreferenceSuggestion fields are readable."""
    sug = PreferenceSuggestion(
        key="k",
        value="v",
        category=PREF_CATEGORY_SOURCE,
        confidence=0.9,
        auto_store=False,
        suggestion_text="Would you?",
    )
    assert sug.key == "k"
    assert sug.value == "v"
    assert sug.category == PREF_CATEGORY_SOURCE
    assert sug.confidence == 0.9
    assert sug.auto_store is False
    assert sug.suggestion_text == "Would you?"
