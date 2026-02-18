"""Unit tests for the session memory module."""

from __future__ import annotations

from unittest.mock import patch

from custom_components.neuralbridge.session_memory import (
    ConversationSession,
    ConversationTurn,
    SessionMemory,
)


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_fields(self) -> None:
        """Test ConversationTurn stores role, content, and timestamp."""
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.timestamp > 0

    def test_custom_timestamp(self) -> None:
        """Test ConversationTurn accepts a custom timestamp."""
        turn = ConversationTurn(role="assistant", content="hi", timestamp=42.0)
        assert turn.timestamp == 42.0


class TestConversationSession:
    """Tests for ConversationSession dataclass."""

    def test_defaults(self) -> None:
        """Test ConversationSession initialises with empty turns."""
        session = ConversationSession()
        assert session.turns == []
        assert session.last_active > 0


class TestSessionMemory:
    """Tests for SessionMemory class."""

    def test_init_defaults(self) -> None:
        """Test SessionMemory initialises with correct defaults."""
        mem = SessionMemory()
        assert mem._max_turns == 20
        assert mem._ttl == 1800
        assert mem._max_sessions == 100

    def test_get_messages_none_conversation_id(self) -> None:
        """Test get_messages returns [] for None conversation_id."""
        mem = SessionMemory()
        assert mem.get_messages(None) == []

    def test_get_messages_empty_string_conversation_id(self) -> None:
        """Test get_messages returns [] for empty string conversation_id."""
        mem = SessionMemory()
        assert mem.get_messages("") == []

    def test_get_messages_unknown_session(self) -> None:
        """Test get_messages returns [] for an unknown conversation_id."""
        mem = SessionMemory()
        assert mem.get_messages("conv-123") == []

    def test_add_turn_then_get_messages(self) -> None:
        """Test message history is returned in correct format after a turn."""
        mem = SessionMemory()
        mem.add_turn("conv-1", "What is 2+2?", "4")
        messages = mem.get_messages("conv-1")
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "What is 2+2?"}
        assert messages[1] == {"role": "assistant", "content": "4"}

    def test_add_turn_none_conversation_id_is_noop(self) -> None:
        """Test add_turn with None conversation_id does not store anything."""
        mem = SessionMemory()
        mem.add_turn(None, "hello", "hi")
        assert mem.session_count() == 0

    def test_add_turn_empty_string_is_noop(self) -> None:
        """Test add_turn with empty string conversation_id does not store anything."""
        mem = SessionMemory()
        mem.add_turn("", "hello", "hi")
        assert mem.session_count() == 0

    def test_multiple_turns_ordered(self) -> None:
        """Test that multiple turns are ordered oldest-first."""
        mem = SessionMemory()
        mem.add_turn("conv-1", "Q1", "A1")
        mem.add_turn("conv-1", "Q2", "A2")
        messages = mem.get_messages("conv-1")
        assert len(messages) == 4
        assert messages[0]["content"] == "Q1"
        assert messages[1]["content"] == "A1"
        assert messages[2]["content"] == "Q2"
        assert messages[3]["content"] == "A2"

    def test_trim_at_max_turns(self) -> None:
        """Test that oldest turns are dropped when max_turns is exceeded."""
        mem = SessionMemory(max_turns=2)  # Keep last 2 pairs = 4 messages
        mem.add_turn("conv-1", "Q1", "A1")
        mem.add_turn("conv-1", "Q2", "A2")
        mem.add_turn("conv-1", "Q3", "A3")  # This pushes Q1/A1 out
        messages = mem.get_messages("conv-1")
        assert len(messages) == 4
        assert messages[0]["content"] == "Q2"
        assert messages[1]["content"] == "A2"
        assert messages[2]["content"] == "Q3"

    def test_clear_session(self) -> None:
        """Test clear_session removes all history for the conversation."""
        mem = SessionMemory()
        mem.add_turn("conv-1", "hello", "hi")
        mem.clear_session("conv-1")
        assert mem.get_messages("conv-1") == []
        assert mem.session_count() == 0

    def test_clear_session_unknown(self) -> None:
        """Test clear_session on an unknown id does not raise."""
        mem = SessionMemory()
        mem.clear_session("does-not-exist")  # Should not raise

    def test_session_count(self) -> None:
        """Test session_count reflects the number of active sessions."""
        mem = SessionMemory()
        assert mem.session_count() == 0
        mem.add_turn("conv-1", "a", "b")
        mem.add_turn("conv-2", "c", "d")
        assert mem.session_count() == 2

    def test_ttl_expiry(self) -> None:
        """Test expired sessions are not returned by get_messages."""
        mem = SessionMemory(ttl_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            mem.add_turn("conv-1", "hello", "hi")

        # 61 seconds later — session is expired
        with patch("time.monotonic", return_value=1061.0):
            messages = mem.get_messages("conv-1")
            assert messages == []

    def test_ttl_not_expired(self) -> None:
        """Test sessions within TTL are still accessible."""
        mem = SessionMemory(ttl_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            mem.add_turn("conv-1", "hello", "hi")

        with patch("time.monotonic", return_value=1059.0):
            messages = mem.get_messages("conv-1")
            assert len(messages) == 2

    def test_expired_session_removed_from_store(self) -> None:
        """Test expired sessions are removed from internal storage."""
        mem = SessionMemory(ttl_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            mem.add_turn("conv-1", "hello", "hi")

        with patch("time.monotonic", return_value=1061.0):
            mem.get_messages("conv-1")  # Triggers cleanup
            assert mem.session_count() == 0

    def test_max_sessions_evicts_oldest(self) -> None:
        """Test the oldest session is evicted when max_sessions is exceeded."""
        mem = SessionMemory(max_sessions=2)
        with patch("time.monotonic", return_value=1000.0):
            mem.add_turn("conv-old", "a", "b")
        with patch("time.monotonic", return_value=1001.0):
            mem.add_turn("conv-new", "c", "d")
        # Assertions are inside the patch block so that is_expired() uses the same
        # fake clock; otherwise real time.monotonic() would make all sessions appear
        # expired (real uptime >> patched timestamps + TTL).
        with patch("time.monotonic", return_value=1002.0):
            mem.add_turn("conv-newer", "e", "f")
            assert mem.get_messages("conv-old") == []
            assert len(mem.get_messages("conv-new")) == 2
            assert len(mem.get_messages("conv-newer")) == 2

    def test_cleanup_removes_expired_on_add(self) -> None:
        """Test expired sessions are cleaned up when a new turn is added."""
        mem = SessionMemory(ttl_seconds=60)
        with patch("time.monotonic", return_value=1000.0):
            mem.add_turn("conv-expired", "x", "y")

        assert mem.session_count() == 1

        with patch("time.monotonic", return_value=1061.0):
            mem.add_turn("conv-new", "a", "b")
            assert mem.session_count() == 1  # expired removed, new one added

    def test_independent_conversations(self) -> None:
        """Test that separate conversation_ids have independent histories."""
        mem = SessionMemory()
        mem.add_turn("conv-A", "Question A", "Answer A")
        mem.add_turn("conv-B", "Question B", "Answer B")
        msgs_a = mem.get_messages("conv-A")
        msgs_b = mem.get_messages("conv-B")
        assert msgs_a[0]["content"] == "Question A"
        assert msgs_b[0]["content"] == "Question B"
        assert len(msgs_a) == 2
        assert len(msgs_b) == 2

    def test_evict_oldest_session_noop_when_empty(self) -> None:
        """_evict_oldest_session does nothing when the session store is empty."""
        mem = SessionMemory()
        # Must not raise even when there are no sessions
        mem._evict_oldest_session()
        assert mem.session_count() == 0
