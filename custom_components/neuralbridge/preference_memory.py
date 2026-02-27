"""Persistent preference memory for NeuralBridge (Feature 15 — Adaptive Preference Learning).

Stores user preferences discovered from conversation patterns (e.g. preferred
news source, location disambiguation).  Data is persisted across HA restarts
via :class:`homeassistant.helpers.storage.Store`.

Only structured key→value pairs are stored — verbatim query text is NEVER
written to this store.  A configurable cap on total entries prevents unbounded
growth; unconfirmed entries are evicted first by age when the cap is reached.
Confirmed entries are never auto-evicted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .persistent_store import AsyncPersistentStore

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Preference category constants
PREF_CATEGORY_SOURCE: str = "source"  # e.g. news_source = BBC
PREF_CATEGORY_LOCATION: str = "location"  # e.g. location_london = London, England
PREF_CATEGORY_FORMAT: str = "format"  # e.g. time_format = 24h

PREF_STORAGE_VERSION: int = 1

# Security caps — clamp key/value lengths to prevent prompt injection bleed-through
_KEY_MAX_LEN: int = 50
_VALUE_MAX_LEN: int = 200


@dataclass
class PreferenceEntry:
    """A single stored user preference.

    Attributes:
        key:              Allowlisted slug identifying the preference (max 50 chars).
        value:            The stored value, sanitised and length-capped (max 200 chars).
        category:         One of ``"source"``, ``"location"``, ``"format"``.
        confirmed:        True only after the user explicitly confirmed this.
        confidence:       Float 0.0-1.0; becomes 1.0 on user confirmation.
        suggestion_count: Number of times this preference has been suggested.
        last_suggested:   ``time.time()`` timestamp of the last suggestion (0 = never).
        created_at:       ``time.time()`` timestamp of creation.
        updated_at:       ``time.time()`` timestamp of last write.
    """

    key: str
    value: str
    category: str
    confirmed: bool
    confidence: float
    suggestion_count: int
    last_suggested: float
    created_at: float
    updated_at: float


def _entry_from_dict(data: dict[str, Any]) -> PreferenceEntry:
    """Deserialise a :class:`PreferenceEntry` from a plain dict.

    Args:
        data: Raw dict loaded from HA storage.

    Returns:
        The deserialised :class:`PreferenceEntry`.

    Raises:
        KeyError: If required keys are missing.
        TypeError: If field types are incompatible.
        ValueError: If numeric conversion fails.
    """
    now = time.time()
    return PreferenceEntry(
        key=str(data["key"]),
        value=str(data["value"]),
        category=str(data["category"]),
        confirmed=bool(data["confirmed"]),
        confidence=float(data["confidence"]),
        suggestion_count=int(data.get("suggestion_count", 0)),
        last_suggested=float(data.get("last_suggested", 0.0)),
        created_at=float(data.get("created_at", now)),
        updated_at=float(data.get("updated_at", now)),
    )


class PreferenceMemory(AsyncPersistentStore[dict[str, Any]]):
    """Persistent store for learned user preference entries.

    Backed by HA's :class:`homeassistant.helpers.storage.Store`.  Preferences
    survive HA restarts.  Unconfirmed entries are evicted by oldest-updated-at
    when the entry cap is reached; confirmed entries are never auto-evicted.

    This class is NOT thread-safe — it is designed to run exclusively on the
    HA event loop and must not be called from threads.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        max_entries: int = 25,
    ) -> None:
        """Initialise preference memory.

        Args:
            hass:        Home Assistant instance (used to build the Store).
            entry_id:    Config entry ID; namespaces the storage file so each
                         NeuralBridge config entry has its own preferences.
            max_entries: Maximum number of stored entries before eviction
                         (default 25, range 1-500).
        """
        super().__init__(hass, f"neuralbridge.preferences_{entry_id}", PREF_STORAGE_VERSION)
        self._max_entries: int = max(1, max_entries)
        self._entries: dict[str, PreferenceEntry] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _deserialise(self, data: dict[str, Any]) -> None:
        """Populate _entries from raw stored data.

        Args:
            data: The raw dictionary from HA storage.
        """
        for key, entry_dict in data.get("entries", {}).items():
            if not isinstance(entry_dict, dict):
                continue
            try:
                self._entries[key] = _entry_from_dict(entry_dict)
            except (KeyError, TypeError, ValueError):
                _LOGGER.debug("Skipping corrupt preference entry for key: %s", key)

    def _serialise(self) -> dict[str, Any]:
        """Return a serialisable dict of all current preference entries."""
        return {"entries": {k: asdict(v) for k, v in self._entries.items()}}

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: str) -> PreferenceEntry | None:
        """Return an entry by key, or None if not found.

        Args:
            key: The preference key to look up.

        Returns:
            The :class:`PreferenceEntry` or ``None``.
        """
        return self._entries.get(key)

    def all_confirmed(self) -> list[PreferenceEntry]:
        """Return all confirmed entries sorted by creation date (oldest first).

        Returns:
            List of user-confirmed :class:`PreferenceEntry` objects.
        """
        return sorted(
            (e for e in self._entries.values() if e.confirmed),
            key=lambda e: e.created_at,
        )

    def all_entries(self) -> list[PreferenceEntry]:
        """Return all entries (confirmed and pending) sorted by creation date.

        Returns:
            All stored :class:`PreferenceEntry` objects, oldest first.
        """
        return sorted(self._entries.values(), key=lambda e: e.created_at)

    def is_suggestion_cooling_down(self, key: str, cooldown_days: int = 7) -> bool:
        """Return True if *key* was suggested within the cooldown window.

        Prevents repeatedly asking the same preference question if the user
        has not acted on a previous suggestion.

        Args:
            key:           Preference key to check.
            cooldown_days: Minimum days between suggestions (default 7).

        Returns:
            True if the cooldown is still active, False otherwise.
        """
        entry = self._entries.get(key)
        if entry is None or entry.last_suggested == 0.0:
            return False
        return (time.time() - entry.last_suggested) < (cooldown_days * 86400)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert(
        self,
        key: str,
        value: str,
        category: str,
        confidence: float,
        confirmed: bool = False,
    ) -> PreferenceEntry:
        """Create or update a preference entry, then persist.

        When updating an existing entry the confidence is only increased —
        never decreased — and the ``confirmed`` flag is sticky (once True,
        stays True).

        Args:
            key:        Preference key (truncated to 50 chars for safety).
            value:      Preference value (truncated to 200 chars for safety).
            category:   ``"source"``, ``"location"``, or ``"format"``.
            confidence: Float 0.0-1.0.  Clamped to valid range.
            confirmed:  Set True to mark as user-confirmed immediately.

        Returns:
            The created or updated :class:`PreferenceEntry`.
        """
        key = key[:_KEY_MAX_LEN]
        value = value[:_VALUE_MAX_LEN]
        confidence = max(0.0, min(1.0, confidence))
        now = time.time()
        existing = self._entries.get(key)
        if existing is not None:
            entry = PreferenceEntry(
                key=key,
                value=value,
                category=category,
                confirmed=confirmed or existing.confirmed,
                confidence=max(confidence, existing.confidence),
                suggestion_count=existing.suggestion_count,
                last_suggested=existing.last_suggested,
                created_at=existing.created_at,
                updated_at=now,
            )
        else:
            self._evict_if_needed()
            entry = PreferenceEntry(
                key=key,
                value=value,
                category=category,
                confirmed=confirmed,
                confidence=confidence,
                suggestion_count=0,
                last_suggested=0.0,
                created_at=now,
                updated_at=now,
            )
        self._entries[key] = entry
        await self.async_save()
        return entry

    async def confirm(self, key: str) -> bool:
        """Mark an entry as user-confirmed, raising confidence to 1.0.

        Args:
            key: Preference key to confirm.

        Returns:
            True if the entry was found and updated; False if not found.
        """
        entry = self._entries.get(key)
        if entry is None:
            return False
        self._entries[key] = PreferenceEntry(
            key=entry.key,
            value=entry.value,
            category=entry.category,
            confirmed=True,
            confidence=1.0,
            suggestion_count=entry.suggestion_count,
            last_suggested=entry.last_suggested,
            created_at=entry.created_at,
            updated_at=time.time(),
        )
        await self.async_save()
        return True

    async def reject(self, key: str) -> bool:
        """Remove an entry (user said no).

        Args:
            key: Preference key to remove.

        Returns:
            True if the entry existed and was removed; False if not found.
        """
        if key not in self._entries:
            return False
        del self._entries[key]
        await self.async_save()
        return True

    async def record_suggestion(self, key: str) -> None:
        """Increment suggestion_count and record the current timestamp.

        Should be called each time a suggestion is shown to the user.  The
        timestamp powers the cooldown check.

        Args:
            key: Preference key that was just suggested.
        """
        entry = self._entries.get(key)
        if entry is None:
            return
        self._entries[key] = PreferenceEntry(
            key=entry.key,
            value=entry.value,
            category=entry.category,
            confirmed=entry.confirmed,
            confidence=entry.confidence,
            suggestion_count=entry.suggestion_count + 1,
            last_suggested=time.time(),
            created_at=entry.created_at,
            updated_at=entry.updated_at,
        )
        await self.async_save()

    async def clear(self) -> int:
        """Remove all stored preferences.

        Returns:
            The number of entries that were cleared.
        """
        count = len(self._entries)
        self._entries.clear()
        await self.async_save()
        return count

    @property
    def max_entries(self) -> int:
        """Return the current maximum entries cap."""
        return self._max_entries

    def set_max_entries(self, max_entries: int) -> None:
        """Update the maximum entry cap at runtime.

        Called when the admin changes ``CONF_PREFERENCE_MAX_ENTRIES`` in the
        options flow.  Does not trigger immediate eviction — excess entries
        will be evicted lazily on the next :meth:`upsert` call.

        Args:
            max_entries: New maximum (clamped to minimum 1).
        """
        self._max_entries = max(1, max_entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict one entry when at capacity — unconfirmed first, then oldest.

        Eviction policy (in order of preference):
          1. Oldest unconfirmed entry (lowest ``updated_at``).
          2. Oldest confirmed entry when all entries are confirmed.
        """
        if len(self._entries) < self._max_entries:
            return
        unconfirmed = sorted(
            (e for e in self._entries.values() if not e.confirmed),
            key=lambda e: e.updated_at,
        )
        if unconfirmed:
            del self._entries[unconfirmed[0].key]
            return
        # All confirmed — evict the one least recently updated
        oldest = min(self._entries.values(), key=lambda e: e.updated_at)
        del self._entries[oldest.key]
