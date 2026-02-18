"""Conversation session memory for NeuralBridge (Ollama agents only).

Maintains per-conversation message history so Ollama agents can provide
contextually aware responses across multiple turns.  History is stored in
memory only and is lost on Home Assistant restart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    """A single user/assistant exchange."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class ConversationSession:
    """All turns belonging to a single conversation."""

    turns: list[ConversationTurn] = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)


class SessionMemory:
    """In-memory store for Ollama conversation context.

    Only Ollama agents use this class.  Existing HA integrations (Gemini,
    ChatGPT, etc.) manage their own conversation history internally.

    Sessions expire after ``ttl_seconds`` of inactivity.  Turns within a
    session are capped at ``max_turns`` pairs; the oldest turns are dropped
    when the limit is exceeded.  The total number of live sessions is capped
    at ``max_sessions``; the oldest session is dropped when the limit would
    be exceeded.
    """

    def __init__(
        self,
        max_turns: int = 20,
        ttl_seconds: int = 1800,
        max_sessions: int = 100,
    ) -> None:
        """Initialise session memory.

        Args:
            max_turns: Maximum number of user/assistant turn *pairs* per session
                (default 20, i.e. 40 individual messages).
            ttl_seconds: Inactivity TTL before a session is expired (default 1800).
            max_sessions: Maximum concurrent sessions before oldest is evicted
                (default 100).
        """
        self._max_turns = max_turns
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._sessions: dict[str, ConversationSession] = {}

    def get_messages(self, conversation_id: str | None) -> list[dict[str, Any]]:
        """Return the message history for an Ollama ``chat()`` call.

        Args:
            conversation_id: HA conversation identifier.  ``None`` means the
                request is stateless; an empty list is returned.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts ordered oldest-first.
            Returns an empty list for stateless or expired sessions.
        """
        if not conversation_id:
            return []

        session = self._sessions.get(conversation_id)
        if session is None:
            return []

        if self._is_expired(session):
            del self._sessions[conversation_id]
            return []

        return [{"role": t.role, "content": t.content} for t in session.turns]

    def add_turn(
        self,
        conversation_id: str | None,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Record a completed user/assistant exchange.

        Trims history to ``max_turns`` pairs (oldest removed first) and
        triggers session cleanup to enforce TTL and max_sessions limits.

        Args:
            conversation_id: HA conversation identifier.  ``None`` is a no-op.
            user_text: The user's input text.
            assistant_text: The assistant's response text.
        """
        if not conversation_id:
            return

        self._cleanup_sessions()

        session = self._sessions.get(conversation_id)
        if session is None:
            if len(self._sessions) >= self._max_sessions:
                self._evict_oldest_session()
            session = ConversationSession()
            self._sessions[conversation_id] = session

        session.turns.append(ConversationTurn(role="user", content=user_text))
        session.turns.append(ConversationTurn(role="assistant", content=assistant_text))
        session.last_active = time.monotonic()

        # Trim to max_turns pairs (each pair = 2 messages)
        max_messages = self._max_turns * 2
        if len(session.turns) > max_messages:
            session.turns = session.turns[-max_messages:]

    def clear_session(self, conversation_id: str) -> None:
        """Remove all history for a specific conversation.

        Args:
            conversation_id: HA conversation identifier.
        """
        self._sessions.pop(conversation_id, None)

    def session_count(self) -> int:
        """Return the number of currently tracked sessions.

        Returns:
            Count of live (not yet expired) sessions.
        """
        return len(self._sessions)

    def _is_expired(self, session: ConversationSession) -> bool:
        """Return True if the session has exceeded its TTL.

        Args:
            session: The session to check.

        Returns:
            True if the session is expired.
        """
        return time.monotonic() - session.last_active > self._ttl

    def _cleanup_sessions(self) -> None:
        """Remove all expired sessions."""
        expired = [cid for cid, s in self._sessions.items() if self._is_expired(s)]
        for cid in expired:
            del self._sessions[cid]

    def _evict_oldest_session(self) -> None:
        """Remove the session with the oldest last_active timestamp."""
        if not self._sessions:
            return
        oldest = min(self._sessions, key=lambda cid: self._sessions[cid].last_active)
        del self._sessions[oldest]
