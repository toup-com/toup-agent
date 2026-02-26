"""
Transcript Export — Per-session conversation transcript persistence.

Generates structured transcript files from session message history.
Supports Markdown, JSON, and plain text export formats.

Usage:
    from app.agent.transcript import get_transcript_manager

    mgr = get_transcript_manager()
    mgr.append("session-123", role="user", content="Hello!")
    mgr.append("session-123", role="assistant", content="Hi there!")
    md = mgr.export("session-123", fmt="markdown")
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


@dataclass
class TranscriptMessage:
    """A single message in a transcript."""
    role: str  # user | assistant | system | tool
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class SessionTranscript:
    """Full transcript for a session."""
    session_id: str
    messages: List[TranscriptMessage] = field(default_factory=list)
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def word_count(self) -> int:
        return sum(len(m.content.split()) for m in self.messages)

    @property
    def duration_seconds(self) -> float:
        if len(self.messages) < 2:
            return 0.0
        return self.messages[-1].timestamp - self.messages[0].timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "message_count": self.message_count,
            "word_count": self.word_count,
            "duration_seconds": round(self.duration_seconds, 1),
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages],
        }


class TranscriptManager:
    """
    Manages per-session transcripts.

    Stores messages in memory and supports export to multiple formats.
    For production, this can be backed by the database.
    """

    def __init__(self, max_sessions: int = 200):
        self._transcripts: Dict[str, SessionTranscript] = {}
        self._max_sessions = max_sessions

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: float = 0.0,
    ) -> int:
        """
        Append a message to a session transcript.

        Returns the message index.
        """
        if session_id not in self._transcripts:
            if len(self._transcripts) >= self._max_sessions:
                self._evict_oldest()
            self._transcripts[session_id] = SessionTranscript(session_id=session_id)

        msg = TranscriptMessage(
            role=role,
            content=content,
            timestamp=timestamp or time.time(),
            metadata=metadata or {},
        )
        self._transcripts[session_id].messages.append(msg)
        return len(self._transcripts[session_id].messages) - 1

    def get(self, session_id: str) -> Optional[SessionTranscript]:
        """Get a session transcript."""
        return self._transcripts.get(session_id)

    def export(self, session_id: str, fmt: str = "markdown") -> Optional[str]:
        """
        Export a session transcript.

        Args:
            session_id: Session identifier
            fmt: Export format — "markdown", "json", or "text"

        Returns:
            Formatted transcript string, or None if session not found.
        """
        transcript = self._transcripts.get(session_id)
        if not transcript:
            return None

        if fmt == "json":
            return json.dumps(transcript.to_dict(), indent=2, default=str)

        elif fmt == "markdown":
            return self._export_markdown(transcript)

        else:  # text
            return self._export_text(transcript)

    def _export_markdown(self, t: SessionTranscript) -> str:
        lines = [
            f"# Session Transcript: {t.session_id}",
            f"",
            f"- **Messages:** {t.message_count}",
            f"- **Words:** {t.word_count}",
            f"- **Duration:** {t.duration_seconds:.0f}s",
            f"",
            f"---",
            f"",
        ]

        for msg in t.messages:
            role_icon = {"user": "👤", "assistant": "🤖", "system": "⚙️", "tool": "🔧"}.get(msg.role, "❓")
            lines.append(f"### {role_icon} {msg.role.capitalize()}")
            lines.append(f"")
            lines.append(msg.content)
            lines.append(f"")

        return "\n".join(lines)

    def _export_text(self, t: SessionTranscript) -> str:
        lines = [f"Session: {t.session_id}", f"Messages: {t.message_count}", ""]
        for msg in t.messages:
            lines.append(f"[{msg.role}] {msg.content}")
            lines.append("")
        return "\n".join(lines)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all tracked sessions with summary info."""
        return [
            {
                "session_id": t.session_id,
                "message_count": t.message_count,
                "word_count": t.word_count,
                "duration_seconds": round(t.duration_seconds, 1),
                "created_at": t.created_at,
            }
            for t in self._transcripts.values()
        ]

    def delete(self, session_id: str) -> bool:
        """Delete a session transcript."""
        return self._transcripts.pop(session_id, None) is not None

    def clear(self) -> int:
        """Clear all transcripts. Returns count cleared."""
        count = len(self._transcripts)
        self._transcripts.clear()
        return count

    def _evict_oldest(self) -> None:
        """Remove the oldest transcript to make room."""
        if not self._transcripts:
            return
        oldest = min(self._transcripts.values(), key=lambda t: t.created_at)
        self._transcripts.pop(oldest.session_id, None)
        logger.debug(f"[TRANSCRIPT] Evicted oldest session: {oldest.session_id}")

    @property
    def session_count(self) -> int:
        return len(self._transcripts)


# ── Singleton ────────────────────────────────────────────
_manager: Optional[TranscriptManager] = None


def get_transcript_manager() -> TranscriptManager:
    """Get the global transcript manager."""
    global _manager
    if _manager is None:
        _manager = TranscriptManager()
    return _manager
