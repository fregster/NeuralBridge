"""Response cache for NeuralBridge.

Deduplicates identical conversation queries by caching successful agent
responses for a configurable TTL.  Enabled by default; can be toggled and
purged via the integration's Advanced Settings options flow.

Exact cache keys are SHA-256 hashes of the lower-cased, stripped input text
so capitalisation and leading/trailing whitespace differences are treated as
the same query.

When *semantic* (normalised) mode is enabled, :func:`_normalise_cache_key`
collapses filler phrases, punctuation, number-words and common morphological
suffixes so paraphrase variants like ``"What's the weather?"`` and
``"How is the weather today?"`` share the same cache entry. Semantic entries
use a shorter TTL (default 60 s) because the looser key means a hit is less
certainly identical to the stored query.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field

# ---------------------------------------------------------------------------
# Input size guards
# ---------------------------------------------------------------------------

_MAX_KEY_CHARS = 2_000  # Max chars of input text used as cache key
_MAX_VALUE_CHARS = 50_000  # Max chars of response text that will be cached

# ---------------------------------------------------------------------------
# Semantic (normalised) key helpers — Feature 9
# ---------------------------------------------------------------------------

_FILLER_PHRASES_RE = re.compile(
    r"\b(?:can\s+you|could\s+you|would\s+you|what(?:'s|\s+is)"
    r"|how(?:'s|\s+is)|tell\s+me|please|today|right\s+now|currently|now)\b",
    re.IGNORECASE,
)

_NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_TENS_THRESHOLD: int = 20  # Numbers >= 20 are tens (twenty, thirty, ...)
_UNITS_MAX: int = 9  # Units are numbers 1-9
_TENS: frozenset[str] = frozenset(w for w, v in _NUMBER_WORDS.items() if v >= _TENS_THRESHOLD)
_UNITS: frozenset[str] = frozenset(w for w, v in _NUMBER_WORDS.items() if 1 <= v <= _UNITS_MAX)

_SUFFIXES: tuple[str, ...] = ("ing", "ed", "s")

# Minimum stem length after suffix removal (prevents over-stripping short words)
_MIN_STEM: int = 3
# Minimum word length required before suffix stripping is attempted
_MIN_WORD_LEN_FOR_STEM: int = 4


def _normalise_number_words(words: list[str]) -> list[str]:
    """Replace number-word sequences with digit equivalents.

    Handles simple units (``one`` → ``1``), teens (``eleven`` → ``11``),
    plain tens (``twenty`` → ``20``), and compound tens-plus-units
    (``twenty two`` → ``22``).

    Args:
        words: Tokenised word list (already lowercased, punctuation stripped).

    Returns:
        New word list with number words replaced by digit strings.
    """
    result: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        if word in _TENS and i + 1 < len(words) and words[i + 1] in _UNITS:
            combined = _NUMBER_WORDS[word] + _NUMBER_WORDS[words[i + 1]]
            result.append(str(combined))
            i += 2
        elif word in _NUMBER_WORDS:
            result.append(str(_NUMBER_WORDS[word]))
            i += 1
        else:
            result.append(word)
            i += 1
    return result


def _normalise_cache_key(text: str) -> str:
    """Normalise *text* to a canonical form for semantic cache keying.

    Applies the following transformations in order:

    1. Truncate at :data:`_MAX_KEY_CHARS` to prevent RegEx/stemming CPU spikes.
    2. Lowercase the input.
    3. Remove filler phrases **before** punctuation stripping, so contractions
       like ``what's`` are matched correctly by the regex.
    4. Strip punctuation (non-word, non-whitespace characters → space).
    5. Normalise number words to digits.
    6. Strip ``-ing``, ``-ed`` and ``-s`` suffixes from words longer than four
       characters, preserving a minimum stem of :data:`_MIN_STEM` characters.
    7. Collapse extra whitespace and strip leading/trailing space.

    Args:
        text: Raw user input text.

    Returns:
        Normalised string suitable for use as a semantic cache-key seed.
    """
    # 0. Enforce input length cap
    if len(text) > _MAX_KEY_CHARS:
        text = text[:_MAX_KEY_CHARS]
    # 1. Lowercase
    normalised = text.lower()
    # 2. Remove filler phrases (before punctuation strip — preserves apostrophes)
    normalised = _FILLER_PHRASES_RE.sub(" ", normalised)
    # 3. Strip punctuation
    normalised = re.sub(r"[^\w\s]", " ", normalised)
    # 4. Normalise number words
    words = _normalise_number_words(normalised.split())
    # 5. Strip common suffixes from words > 4 chars
    stemmed: list[str] = []
    for word in words:
        stemmed_word = word
        if len(stemmed_word) > _MIN_WORD_LEN_FOR_STEM:
            for suffix in _SUFFIXES:
                stem = stemmed_word[: -len(suffix)] if stemmed_word.endswith(suffix) else ""
                if stem and len(stem) >= _MIN_STEM:
                    stemmed_word = stem
                    break
        stemmed.append(stemmed_word)
    # 6. Collapse whitespace
    return " ".join(stemmed)


# ---------------------------------------------------------------------------
# Cache dataclass and main class
# ---------------------------------------------------------------------------


@dataclass
class CachedResponse:
    """A single cached agent response."""

    response_text: str
    agent_name: str
    cached_at: float
    expires_at: float
    normalised_text: str | None = dataclass_field(default=None)


class ResponseCache:
    """Short-TTL response cache for deduplicated conversation answers.

    Instances are stored in ``hass.data`` so the options flow handler can
    call :meth:`invalidate` and :meth:`configure` without restarting the
    integration.

    When *semantic* is ``True`` callers should pass ``normalise=True`` to
    :meth:`get` and :meth:`store` so that paraphrase variants of the same
    question share a cache entry.
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: int = 300,
        max_size: int = 200,
        semantic: bool = False,
        semantic_ttl_seconds: int = 60,
    ) -> None:
        """Initialise the response cache.

        Args:
            enabled: Whether caching is active (default True).
            ttl_seconds: How long an exact-matched cached response is valid
                (default 300 s).
            max_size: Maximum number of cached entries before oldest are
                evicted (default 200).
            semantic: Whether semantic (normalised) keying is enabled
                (default False).
            semantic_ttl_seconds: TTL for semantically-keyed entries; shorter
                than the exact TTL because the looser key is less precise
                (default 60 s).
        """
        self._enabled = enabled
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._semantic = semantic
        self._semantic_ttl = semantic_ttl_seconds
        self._cache: dict[str, CachedResponse] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str, *, normalise: bool = False) -> str | None:
        """Return a cached response for *text*, or ``None`` on a miss.

        Expired entries are removed on access.

        When *normalise* is ``True``, the lookup key is derived from the
        normalised form of *text* (via :func:`_normalise_cache_key`), and the
        stored :attr:`CachedResponse.normalised_text` must match to avoid
        false hits from hash collisions between different normalised forms.

        Args:
            text: The raw user input text.
            normalise: If ``True``, use the semantic (normalised) cache key.

        Returns:
            Cached response string, or ``None`` if disabled/not found/expired.
        """
        if not self._enabled:
            return None

        if normalise:
            norm = _normalise_cache_key(text)
            key = hashlib.sha256(norm.encode()).hexdigest()
        else:
            key = self._make_key(text)

        entry = self._cache.get(key)
        if entry is None:
            return None

        if time.monotonic() > entry.expires_at:
            del self._cache[key]
            return None

        # For semantic keys, verify normalised text matches to prevent false hits
        if normalise and entry.normalised_text != norm:
            return None

        return entry.response_text

    def store(
        self,
        text: str,
        response: str,
        agent_name: str,
        *,
        normalise: bool = False,
    ) -> None:
        """Cache a successful agent response.

        Triggers internal cleanup (expire old entries, enforce max_size)
        before inserting the new entry.

        When *normalise* is ``True``, the entry is stored under the semantic
        key and uses the shorter :attr:`semantic_ttl`.

        Args:
            text: The raw user input text (used to derive the cache key).
            response: The agent response to cache.
            agent_name: Human-readable agent name (stored for diagnostics).
            normalise: If ``True``, use the semantic (normalised) cache key
                and semantic TTL.
        """
        if not self._enabled:
            return

        # Silently skip over-long inputs or responses to avoid memory and CPU waste.
        if len(text) > _MAX_KEY_CHARS or len(response) > _MAX_VALUE_CHARS:
            return

        self._cleanup()
        now = time.monotonic()

        if normalise:
            norm = _normalise_cache_key(text)
            key = hashlib.sha256(norm.encode()).hexdigest()
            ttl = self._semantic_ttl
            self._cache[key] = CachedResponse(
                response_text=response,
                agent_name=agent_name,
                cached_at=now,
                expires_at=now + ttl,
                normalised_text=norm,
            )
        else:
            key = self._make_key(text)
            self._cache[key] = CachedResponse(
                response_text=response,
                agent_name=agent_name,
                cached_at=now,
                expires_at=now + self._ttl,
            )

    def invalidate(self) -> int:
        """Purge all cached entries.

        Returns:
            The number of entries that were removed.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def configure(
        self,
        enabled: bool,
        ttl_seconds: int,
        *,
        semantic: bool | None = None,
        semantic_ttl_seconds: int | None = None,
    ) -> None:
        """Update cache configuration without restarting.

        Disabling the cache also purges all existing entries.

        Args:
            enabled: Whether caching should be active.
            ttl_seconds: New TTL for subsequent exact-matched entries.
            semantic: If provided, update the semantic-keying toggle.
            semantic_ttl_seconds: If provided, update the semantic TTL.
        """
        self._enabled = enabled
        self._ttl = ttl_seconds
        if semantic is not None:
            self._semantic = semantic
        if semantic_ttl_seconds is not None:
            self._semantic_ttl = semantic_ttl_seconds
        if not enabled:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return whether the cache is currently enabled."""
        return self._enabled

    @property
    def ttl(self) -> int:
        """Return the current exact-match TTL in seconds."""
        return self._ttl

    @property
    def semantic(self) -> bool:
        """Return whether semantic (normalised) cache keying is active."""
        return self._semantic

    @property
    def semantic_ttl(self) -> int:
        """Return the semantic-key TTL in seconds."""
        return self._semantic_ttl

    @property
    def size(self) -> int:
        """Return the current number of cached entries (including expired)."""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, text: str) -> str:
        """Derive an exact cache key from the input text.

        Keys are SHA-256 hashes of the lower-cased, stripped text so minor
        differences in capitalisation or whitespace map to the same entry.

        Args:
            text: Raw input text.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        normalised = text.lower().strip()
        return hashlib.sha256(normalised.encode()).hexdigest()

    def _cleanup(self) -> None:
        """Remove expired entries and enforce max_size by evicting oldest."""
        now = time.monotonic()
        self._cache = {k: v for k, v in self._cache.items() if v.expires_at > now}

        if len(self._cache) >= self._max_size:
            # Evict oldest entries until we are one below max_size
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].cached_at)
            evict_count = len(self._cache) - self._max_size + 1
            for key, _ in sorted_entries[:evict_count]:
                del self._cache[key]
