"""Base class for HA-backed async persistent stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from homeassistant.helpers.storage import Store

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

T = TypeVar("T")
_STORAGE_VERSION = 1


class AsyncPersistentStore(ABC, Generic[T]):
    """Abstract base for objects persisted via :class:`homeassistant.helpers.storage.Store`.

    Subclasses implement :meth:`_deserialise` and :meth:`_serialise`; this base
    class manages the :class:`Store` lifecycle and provides
    :meth:`async_load` / :meth:`async_save`.

    Typical usage::

        class MyStore(AsyncPersistentStore[MyData]):
            def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
                super().__init__(hass, f"neuralbridge.mystore_{entry_id}")
                self._data: dict[str, MyData] = {}

            def _deserialise(self, data: dict[str, Any]) -> None:
                for k, v in data.items():
                    self._data[k] = MyData.from_dict(v)

            def _serialise(self) -> dict[str, Any]:
                return {k: v.to_dict() for k, v in self._data.items()}

    Not thread-safe; designed for single-threaded asyncio use only.
    """

    def __init__(
        self,
        hass: "HomeAssistant",
        store_key: str,
        version: int = _STORAGE_VERSION,
    ) -> None:
        """Initialise the persistent store.

        Args:
            hass: Home Assistant instance.
            store_key: Unique key identifying the storage file.
            version: Storage schema version (used by HA migration helpers).
        """
        self._store: Store[dict[str, Any]] = Store(hass, version, store_key)

    async def async_load(self) -> None:
        """Load persisted data and populate internal state via :meth:`_deserialise`.

        Does nothing if the store is empty, or returns a non-dict value.
        """
        raw = await self._store.async_load()
        if isinstance(raw, dict) and raw:
            self._deserialise(raw)

    async def async_save(self) -> None:
        """Serialise internal state via :meth:`_serialise` and persist it."""
        await self._store.async_save(self._serialise())

    @abstractmethod
    def _deserialise(self, data: dict[str, Any]) -> None:
        """Populate instance state from the raw persisted dict.

        Args:
            data: The raw dictionary loaded from storage (guaranteed non-falsy).
        """

    @abstractmethod
    def _serialise(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representing the current instance state.

        Returns:
            Dict that will be written to HA storage.
        """
