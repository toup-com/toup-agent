"""
Event Bus — Central pub/sub system for decoupled component communication.

Allows any component to publish events and any other component to subscribe,
without direct coupling. Supports sync and async handlers, wildcard subscriptions,
and event history for replay.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger("hexbrain.event_bus")


class EventPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """An event published to the bus."""
    topic: str
    data: Any = None
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time()*1000)}")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
        }


@dataclass
class Subscription:
    """A subscription to events on a topic."""
    topic: str
    handler: Callable
    subscriber_id: str = ""
    is_async: bool = False
    priority: EventPriority = EventPriority.NORMAL


class EventBus:
    """
    Central event bus for pub/sub communication.
    
    Features:
    - Topic-based routing with wildcard support (e.g., "agent.*")
    - Sync and async handlers
    - Event history for replay
    - Priority-based delivery
    - Dead letter queue for failed deliveries
    """

    def __init__(self, history_size: int = 1000):
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._history: deque = deque(maxlen=history_size)
        self._dead_letters: deque = deque(maxlen=100)
        self._stats = {
            "published": 0,
            "delivered": 0,
            "failed": 0,
        }
        self._running = False

    def subscribe(
        self,
        topic: str,
        handler: Callable,
        subscriber_id: str = "",
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Subscription:
        """Subscribe to events on a topic. Supports wildcards like 'agent.*'."""
        is_async = asyncio.iscoroutinefunction(handler)
        sub = Subscription(
            topic=topic,
            handler=handler,
            subscriber_id=subscriber_id,
            is_async=is_async,
            priority=priority,
        )
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        self._subscriptions[topic].append(sub)
        logger.debug(f"Subscribed {subscriber_id or 'anonymous'} to '{topic}'")
        return sub

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Remove a subscription."""
        subs = self._subscriptions.get(subscription.topic, [])
        if subscription in subs:
            subs.remove(subscription)
            return True
        return False

    def unsubscribe_all(self, subscriber_id: str) -> int:
        """Remove all subscriptions for a subscriber."""
        count = 0
        for topic, subs in self._subscriptions.items():
            before = len(subs)
            self._subscriptions[topic] = [s for s in subs if s.subscriber_id != subscriber_id]
            count += before - len(self._subscriptions[topic])
        return count

    async def publish(self, event: Event) -> int:
        """Publish an event. Returns number of handlers notified."""
        self._stats["published"] += 1
        self._history.append(event)

        handlers = self._get_matching_handlers(event.topic)
        delivered = 0

        # Sort by priority
        priority_order = {EventPriority.CRITICAL: 0, EventPriority.HIGH: 1, EventPriority.NORMAL: 2, EventPriority.LOW: 3}
        handlers.sort(key=lambda s: priority_order.get(s.priority, 2))

        for sub in handlers:
            try:
                if sub.is_async:
                    await sub.handler(event)
                else:
                    sub.handler(event)
                delivered += 1
            except Exception as e:
                logger.error(f"Event handler error for '{event.topic}' in {sub.subscriber_id}: {e}")
                self._dead_letters.append({"event": event.to_dict(), "error": str(e), "subscriber": sub.subscriber_id})
                self._stats["failed"] += 1

        self._stats["delivered"] += delivered
        return delivered

    async def emit(self, topic: str, data: Any = None, source: str = "") -> int:
        """Convenience method to publish an event by topic string."""
        event = Event(topic=topic, data=data, source=source)
        return await self.publish(event)

    def _get_matching_handlers(self, topic: str) -> List[Subscription]:
        """Get all handlers matching a topic, including wildcards."""
        handlers = []
        for pattern, subs in self._subscriptions.items():
            if self._topic_matches(pattern, topic):
                handlers.extend(subs)
        return handlers

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """Check if a topic matches a pattern. Supports '*' wildcard."""
        if pattern == topic:
            return True
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".") or topic == prefix
        return False

    def get_history(self, topic: Optional[str] = None, limit: int = 50) -> List[dict]:
        """Get recent event history, optionally filtered by topic."""
        events = list(self._history)
        if topic:
            events = [e for e in events if self._topic_matches(topic, e.topic)]
        return [e.to_dict() for e in events[-limit:]]

    def get_dead_letters(self, limit: int = 20) -> List[dict]:
        """Get recent failed event deliveries."""
        return list(self._dead_letters)[-limit:]

    def get_stats(self) -> dict:
        """Get event bus statistics."""
        return {
            **self._stats,
            "topics": list(self._subscriptions.keys()),
            "total_subscriptions": sum(len(s) for s in self._subscriptions.values()),
            "history_size": len(self._history),
            "dead_letters": len(self._dead_letters),
        }

    def list_topics(self) -> List[str]:
        """List all topics with active subscriptions."""
        return list(self._subscriptions.keys())

    def clear_history(self):
        """Clear event history."""
        self._history.clear()


# ── Singleton ──
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus singleton."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
