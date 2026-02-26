"""
Gmail Pub/Sub — Gmail event triggers via Google Pub/Sub.

Watches Gmail inboxes for new messages and triggers agent actions.
Uses Google Cloud Pub/Sub push subscriptions.

Usage:
    from app.agent.gmail_pubsub import get_gmail_watcher

    watcher = get_gmail_watcher()
    watcher.add_watch("user@gmail.com", credentials={...})
    await watcher.process_notification(pubsub_payload)
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class WatchState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    ERROR = "error"


class TriggerAction(str, Enum):
    NOTIFY_AGENT = "notify_agent"    # Send to agent for processing
    AUTO_REPLY = "auto_reply"        # Auto-reply with agent response
    EXTRACT = "extract"              # Extract and store info
    FORWARD = "forward"              # Forward to another channel


@dataclass
class GmailWatch:
    """A Gmail inbox watch configuration."""
    email: str
    watch_id: str = ""
    state: WatchState = WatchState.ACTIVE
    label_filter: str = "INBOX"
    action: TriggerAction = TriggerAction.NOTIFY_AGENT
    agent_id: str = "default"
    history_id: int = 0
    created_at: float = 0.0
    expires_at: float = 0.0
    message_count: int = 0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if not self.watch_id:
            self.watch_id = f"watch_{int(self.created_at)}"

    @property
    def is_expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "watch_id": self.watch_id,
            "state": self.state.value,
            "label_filter": self.label_filter,
            "action": self.action.value,
            "agent_id": self.agent_id,
            "message_count": self.message_count,
        }


@dataclass
class GmailEvent:
    """A Gmail notification event."""
    email: str
    history_id: int
    message_ids: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    subject: str = ""
    sender: str = ""
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "history_id": self.history_id,
            "message_ids": self.message_ids,
            "subject": self.subject,
            "sender": self.sender,
            "snippet": self.snippet,
        }


class GmailPubSubWatcher:
    """
    Gmail Pub/Sub watcher for inbox triggers.

    Monitors Gmail inboxes via Google Cloud Pub/Sub and triggers
    agent actions when new emails arrive.
    """

    def __init__(self):
        self._watches: Dict[str, GmailWatch] = {}
        self._handlers: Dict[TriggerAction, List[Callable]] = {}
        self._events: List[GmailEvent] = []
        self._max_events: int = 100

    def add_watch(
        self,
        email: str,
        *,
        label_filter: str = "INBOX",
        action: TriggerAction = TriggerAction.NOTIFY_AGENT,
        agent_id: str = "default",
        ttl_hours: int = 168,  # 7 days (Google max)
    ) -> GmailWatch:
        """Add a Gmail inbox watch."""
        watch = GmailWatch(
            email=email,
            label_filter=label_filter,
            action=action,
            agent_id=agent_id,
            expires_at=time.time() + (ttl_hours * 3600),
        )
        self._watches[email] = watch
        logger.info(f"[GMAIL] Added watch for {email} ({action.value})")
        return watch

    def remove_watch(self, email: str) -> bool:
        """Remove a Gmail watch."""
        return self._watches.pop(email, None) is not None

    def get_watch(self, email: str) -> Optional[GmailWatch]:
        """Get a watch by email."""
        return self._watches.get(email)

    def pause_watch(self, email: str) -> bool:
        """Pause a watch."""
        watch = self._watches.get(email)
        if watch:
            watch.state = WatchState.PAUSED
            return True
        return False

    def resume_watch(self, email: str) -> bool:
        """Resume a paused watch."""
        watch = self._watches.get(email)
        if watch and watch.state == WatchState.PAUSED:
            watch.state = WatchState.ACTIVE
            return True
        return False

    async def process_notification(self, payload: Dict[str, Any]) -> Optional[GmailEvent]:
        """
        Process a Pub/Sub push notification from Gmail.

        Expects the standard Google Pub/Sub push format.
        """
        message = payload.get("message", {})
        data_b64 = message.get("data", "")

        try:
            if data_b64:
                data = json.loads(base64.b64decode(data_b64))
            else:
                data = payload.get("data", {})
        except (json.JSONDecodeError, Exception):
            data = {}

        email = data.get("emailAddress", "")
        history_id = data.get("historyId", 0)

        watch = self._watches.get(email)
        if not watch:
            logger.warning(f"[GMAIL] No watch for {email}")
            return None

        if watch.state != WatchState.ACTIVE:
            return None

        event = GmailEvent(
            email=email,
            history_id=history_id,
            timestamp=time.time(),
        )

        watch.history_id = history_id
        watch.message_count += 1

        # Store event
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        return event

    def register_handler(self, action: TriggerAction, handler: Callable) -> None:
        """Register a handler for a trigger action."""
        if action not in self._handlers:
            self._handlers[action] = []
        self._handlers[action].append(handler)

    def list_watches(self, state: Optional[WatchState] = None) -> List[Dict[str, Any]]:
        """List all watches."""
        watches = list(self._watches.values())
        if state:
            watches = [w for w in watches if w.state == state]
        return [w.to_dict() for w in watches]

    def recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent Gmail events."""
        return [e.to_dict() for e in self._events[-limit:]]

    def stats(self) -> Dict[str, Any]:
        by_state: Dict[str, int] = {}
        total_messages = 0
        for w in self._watches.values():
            by_state[w.state.value] = by_state.get(w.state.value, 0) + 1
            total_messages += w.message_count

        return {
            "total_watches": len(self._watches),
            "by_state": by_state,
            "total_messages": total_messages,
            "recent_events": len(self._events),
        }


# ── Singleton ────────────────────────────────────────────
_watcher: Optional[GmailPubSubWatcher] = None


def get_gmail_watcher() -> GmailPubSubWatcher:
    """Get the global Gmail Pub/Sub watcher."""
    global _watcher
    if _watcher is None:
        _watcher = GmailPubSubWatcher()
    return _watcher
