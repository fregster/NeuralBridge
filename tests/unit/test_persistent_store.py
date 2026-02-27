"""Unit tests for the AsyncPersistentStore base class."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.neuralbridge.persistent_store import AsyncPersistentStore

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------


class SimpleStore(AsyncPersistentStore):
    """Minimal concrete implementation used only in tests."""

    def __init__(self, hass, key: str = "test.key") -> None:
        """Initialise with a mock store."""
        with patch("custom_components.neuralbridge.persistent_store.Store") as MockStore:
            mock = AsyncMock()
            MockStore.return_value = mock
            super().__init__(hass, key)
        self._data: dict[str, str] = {}

    def _deserialise(self, data: dict[str, Any]) -> None:
        """Load string data from dict."""
        self._data.update(data.get("items", {}))

    def _serialise(self) -> dict[str, Any]:
        """Serialise current data to dict."""
        return {"items": dict(self._data)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncPersistentStoreInit:
    """Test that the base class initialises the Store correctly."""

    def test_store_is_created_on_init(self, hass):
        """The Store instance is available after __init__."""
        with patch("custom_components.neuralbridge.persistent_store.Store") as MockStore:
            mock_store = AsyncMock()
            MockStore.return_value = mock_store
            store = SimpleStore.__new__(SimpleStore)
            AsyncPersistentStore.__init__(store, hass, "some.key", version=2)
        MockStore.assert_called_once_with(hass, 2, "some.key")
        assert store._store is mock_store

    def test_default_version_is_one(self, hass):
        """Default storage version is 1."""
        with patch("custom_components.neuralbridge.persistent_store.Store") as MockStore:
            MockStore.return_value = AsyncMock()
            store_obj = SimpleStore.__new__(SimpleStore)
            AsyncPersistentStore.__init__(store_obj, hass, "a.key")
        MockStore.assert_called_once_with(hass, 1, "a.key")


class TestAsyncPersistentStoreLoad:
    """Test async_load behaviour."""

    async def test_async_load_empty_store_is_noop(self, hass):
        """async_load does nothing when the store returns None."""
        s = SimpleStore(hass)
        s._store.async_load = AsyncMock(return_value=None)
        await s.async_load()
        assert s._data == {}

    async def test_async_load_falsy_store_is_noop(self, hass):
        """async_load does nothing when the store returns an empty dict."""
        s = SimpleStore(hass)
        s._store.async_load = AsyncMock(return_value={})
        await s.async_load()
        assert s._data == {}

    async def test_async_load_calls_deserialise_with_raw_data(self, hass):
        """async_load passes raw data from the store to _deserialise."""
        s = SimpleStore(hass)
        s._store.async_load = AsyncMock(return_value={"items": {"key1": "value1"}})
        await s.async_load()
        assert s._data == {"key1": "value1"}

    async def test_async_load_non_dict_items_is_noop(self, hass):
        """async_load skips _deserialise when the store returns a non-dict."""
        s = SimpleStore(hass)
        s._store.async_load = AsyncMock(return_value=[1, 2, 3])
        # The base class guards against non-dict raw values.
        await s.async_load()
        assert s._data == {}


class TestAsyncPersistentStoreSave:
    """Test async_save behaviour."""

    async def test_async_save_calls_store_async_save(self, hass):
        """async_save passes the serialised dict to Store.async_save."""
        s = SimpleStore(hass)
        s._data = {"hello": "world"}
        await s.async_save()
        s._store.async_save.assert_called_once_with({"items": {"hello": "world"}})

    async def test_async_save_empty_data(self, hass):
        """async_save works with no data stored."""
        s = SimpleStore(hass)
        await s.async_save()
        s._store.async_save.assert_called_once_with({"items": {}})

    async def test_async_load_then_save_round_trip(self, hass):
        """Data survives a serialise → deserialise round-trip."""
        s = SimpleStore(hass)
        s._store.async_load = AsyncMock(return_value={"items": {"x": "y"}})
        await s.async_load()
        assert s._data == {"x": "y"}

        saved: dict = {}

        def capture(d: dict) -> None:
            saved.update(d)

        s._store.async_save = AsyncMock(side_effect=capture)
        await s.async_save()
        assert saved == {"items": {"x": "y"}}
