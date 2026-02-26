"""
Thread Manager — Cross-channel thread operations.

Provides unified thread management across channels that support
threading (Discord threads, Slack threads, Telegram topics, etc.).

Each channel adapter maps its native threading to this abstraction.

Usage:
    from app.agent.thread_manager import get_thread_manager

    mgr = get_thread_manager()
    thread = await mgr.create_thread("discord", channel_id="123", title="Bug Report")
    await mgr.reply_to_thread(thread.thread_id, "Looking into it...")
    threads = await mgr.list_threads("discord", channel_id="123")
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class ThreadState(str, Enum):
    OPEN = "open"
    ARCHIVED = "archived"
    LOCKED = "locked"
    DELETED = "deleted"


@dataclass
class ThreadInfo:
    """Information about a thread."""
    thread_id: str
    channel_type: str
    channel_id: str
    title: str
    state: ThreadState = ThreadState.OPEN
    parent_message_id: Optional[str] = None
    creator_id: Optional[str] = None
    message_count: int = 0
    created_at: float = 0.0
    last_message_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "channel_type": self.channel_type,
            "channel_id": self.channel_id,
            "title": self.title,
            "state": self.state.value,
            "parent_message_id": self.parent_message_id,
            "message_count": self.message_count,
            "created_at": self.created_at,
        }


@dataclass
class ThreadMessage:
    """A message in a thread."""
    message_id: str
    thread_id: str
    content: str
    sender_id: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "content": self.content,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
        }


# Type for channel-specific thread handler
ThreadHandler = Callable[..., Coroutine[Any, Any, Any]]


class ThreadManager:
    """
    Manages threads across all channels.

    Channel adapters register handlers for create/list/reply/archive.
    The ThreadManager provides a unified API.
    """

    def __init__(self):
        self._threads: Dict[str, ThreadInfo] = {}
        self._messages: Dict[str, List[ThreadMessage]] = {}
        self._handlers: Dict[str, Dict[str, ThreadHandler]] = {}
        self._counter: int = 0

    def register_handler(
        self,
        channel_type: str,
        operation: str,
        handler: ThreadHandler,
    ) -> None:
        """
        Register a channel-specific thread handler.

        Args:
            channel_type: Channel type (discord, slack, telegram, etc.)
            operation: Operation (create, list, reply, archive, delete)
            handler: Async handler function
        """
        if channel_type not in self._handlers:
            self._handlers[channel_type] = {}
        self._handlers[channel_type][operation] = handler
        logger.info(f"[THREAD] Registered {operation} handler for {channel_type}")

    async def create_thread(
        self,
        channel_type: str,
        channel_id: str,
        title: str,
        *,
        parent_message_id: Optional[str] = None,
        creator_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThreadInfo:
        """
        Create a new thread.

        If the channel has a registered create handler, delegates to it.
        Otherwise creates a local thread record.
        """
        self._counter += 1
        thread_id = f"thread_{channel_type}_{self._counter}"

        thread = ThreadInfo(
            thread_id=thread_id,
            channel_type=channel_type,
            channel_id=channel_id,
            title=title,
            parent_message_id=parent_message_id,
            creator_id=creator_id,
            metadata=metadata or {},
        )

        # Delegate to channel handler if available
        handler = self._handlers.get(channel_type, {}).get("create")
        if handler:
            result = await handler(
                channel_id=channel_id,
                title=title,
                parent_message_id=parent_message_id,
            )
            if isinstance(result, dict) and "thread_id" in result:
                thread.thread_id = result["thread_id"]

        self._threads[thread.thread_id] = thread
        self._messages[thread.thread_id] = []
        logger.info(f"[THREAD] Created {thread.thread_id}: {title}")
        return thread

    async def reply_to_thread(
        self,
        thread_id: str,
        content: str,
        *,
        sender_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThreadMessage:
        """
        Reply to a thread.

        Args:
            thread_id: The thread to reply to.
            content: Message content.
            sender_id: Who sent the reply.
        """
        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found")

        if thread.state == ThreadState.LOCKED:
            raise ValueError(f"Thread {thread_id} is locked")

        msg_id = f"msg_{thread_id}_{thread.message_count + 1}"
        message = ThreadMessage(
            message_id=msg_id,
            thread_id=thread_id,
            content=content,
            sender_id=sender_id,
            metadata=metadata or {},
        )

        # Delegate to channel handler
        handler = self._handlers.get(thread.channel_type, {}).get("reply")
        if handler:
            await handler(thread_id=thread_id, content=content, sender_id=sender_id)

        self._messages.setdefault(thread_id, []).append(message)
        thread.message_count += 1
        thread.last_message_at = time.time()
        return message

    async def list_threads(
        self,
        channel_type: Optional[str] = None,
        channel_id: Optional[str] = None,
        *,
        state: Optional[ThreadState] = None,
        limit: int = 50,
    ) -> List[ThreadInfo]:
        """List threads, optionally filtered by channel and state."""
        threads = list(self._threads.values())

        if channel_type:
            threads = [t for t in threads if t.channel_type == channel_type]
        if channel_id:
            threads = [t for t in threads if t.channel_id == channel_id]
        if state:
            threads = [t for t in threads if t.state == state]

        # Sort by most recent activity
        threads.sort(key=lambda t: t.last_message_at or t.created_at, reverse=True)
        return threads[:limit]

    async def get_thread_messages(
        self,
        thread_id: str,
        *,
        limit: int = 100,
    ) -> List[ThreadMessage]:
        """Get messages from a thread."""
        if thread_id not in self._threads:
            raise ValueError(f"Thread {thread_id} not found")
        return self._messages.get(thread_id, [])[-limit:]

    async def archive_thread(self, thread_id: str) -> bool:
        """Archive a thread."""
        thread = self._threads.get(thread_id)
        if not thread:
            return False
        thread.state = ThreadState.ARCHIVED

        handler = self._handlers.get(thread.channel_type, {}).get("archive")
        if handler:
            await handler(thread_id=thread_id)

        return True

    async def lock_thread(self, thread_id: str) -> bool:
        """Lock a thread (no more replies allowed)."""
        thread = self._threads.get(thread_id)
        if not thread:
            return False
        thread.state = ThreadState.LOCKED
        return True

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread."""
        thread = self._threads.pop(thread_id, None)
        if not thread:
            return False
        self._messages.pop(thread_id, None)
        thread.state = ThreadState.DELETED
        return True

    def get_thread(self, thread_id: str) -> Optional[ThreadInfo]:
        """Get thread info by ID."""
        return self._threads.get(thread_id)

    def stats(self) -> Dict[str, Any]:
        """Get thread statistics."""
        by_channel: Dict[str, int] = {}
        by_state: Dict[str, int] = {}
        total_messages = 0

        for t in self._threads.values():
            by_channel[t.channel_type] = by_channel.get(t.channel_type, 0) + 1
            by_state[t.state.value] = by_state.get(t.state.value, 0) + 1
            total_messages += t.message_count

        return {
            "total_threads": len(self._threads),
            "total_messages": total_messages,
            "by_channel": by_channel,
            "by_state": by_state,
            "handlers_registered": {
                ch: list(ops.keys())
                for ch, ops in self._handlers.items()
            },
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[ThreadManager] = None


def get_thread_manager() -> ThreadManager:
    """Get the global thread manager."""
    global _manager
    if _manager is None:
        _manager = ThreadManager()
    return _manager
