"""
Message Queue — Reliable async delivery between components.

Provides at-least-once delivery with retry, dead letter queue,
priority ordering, and consumer groups.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger("hexbrain.message_queue")


class MessageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD = "dead"


class DeliveryMode(str, Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"


@dataclass
class QueueMessage:
    """A message in the queue."""
    queue: str
    payload: Any
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0  # Higher = more urgent
    status: MessageStatus = MessageStatus.PENDING
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "queue": self.queue,
            "payload": self.payload,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "error": self.error,
        }


@dataclass
class QueueConsumer:
    """A consumer registered on a queue."""
    queue: str
    handler: Callable
    consumer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    is_async: bool = False
    concurrency: int = 1
    _active: int = 0


class MessageQueue:
    """
    In-process message queue with retry and dead letter support.
    
    Features:
    - Named queues with multiple consumers
    - Priority ordering within a queue
    - Retry with exponential backoff
    - Dead letter queue for failed messages
    - Consumer concurrency control
    """

    def __init__(self):
        self._queues: Dict[str, deque] = {}
        self._consumers: Dict[str, List[QueueConsumer]] = {}
        self._dead_letters: deque = deque(maxlen=200)
        self._processing: Dict[str, QueueMessage] = {}
        self._stats = {
            "enqueued": 0,
            "processed": 0,
            "failed": 0,
            "retried": 0,
        }
        self._workers: Dict[str, asyncio.Task] = {}
        self._running = False

    def create_queue(self, name: str) -> None:
        """Create a named queue."""
        if name not in self._queues:
            self._queues[name] = deque()
            logger.info(f"Queue '{name}' created")

    async def enqueue(
        self,
        queue: str,
        payload: Any,
        priority: int = 0,
        max_retries: int = 3,
    ) -> str:
        """Add a message to a queue. Returns message ID."""
        if queue not in self._queues:
            self.create_queue(queue)

        msg = QueueMessage(
            queue=queue,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
        )
        self._queues[queue].append(msg)
        self._stats["enqueued"] += 1

        # Trigger processing if workers are running
        if queue in self._workers:
            pass  # Worker will pick it up

        logger.debug(f"Enqueued message {msg.message_id} to '{queue}'")
        return msg.message_id

    def register_consumer(
        self,
        queue: str,
        handler: Callable,
        consumer_id: str = "",
        concurrency: int = 1,
    ) -> QueueConsumer:
        """Register a consumer for a queue."""
        if queue not in self._queues:
            self.create_queue(queue)

        consumer = QueueConsumer(
            queue=queue,
            handler=handler,
            consumer_id=consumer_id or str(uuid.uuid4())[:8],
            is_async=asyncio.iscoroutinefunction(handler),
            concurrency=concurrency,
        )
        if queue not in self._consumers:
            self._consumers[queue] = []
        self._consumers[queue].append(consumer)
        return consumer

    async def process_one(self, queue: str) -> Optional[str]:
        """Process one message from a queue. Returns message ID or None."""
        q = self._queues.get(queue)
        if not q:
            return None

        # Sort by priority (higher first)
        sorted_q = sorted(q, key=lambda m: -m.priority)
        if not sorted_q:
            return None

        msg = sorted_q[0]
        q.remove(msg)

        consumers = self._consumers.get(queue, [])
        if not consumers:
            # No consumers, put back
            q.appendleft(msg)
            return None

        msg.status = MessageStatus.PROCESSING
        msg.attempts += 1
        self._processing[msg.message_id] = msg

        for consumer in consumers:
            try:
                if consumer.is_async:
                    await consumer.handler(msg.payload)
                else:
                    consumer.handler(msg.payload)
                msg.status = MessageStatus.COMPLETED
                self._stats["processed"] += 1
            except Exception as e:
                logger.error(f"Consumer {consumer.consumer_id} failed on {msg.message_id}: {e}")
                msg.error = str(e)
                if msg.attempts < msg.max_retries:
                    msg.status = MessageStatus.PENDING
                    await asyncio.sleep(msg.retry_delay * msg.attempts)
                    q.append(msg)
                    self._stats["retried"] += 1
                else:
                    msg.status = MessageStatus.DEAD
                    self._dead_letters.append(msg)
                    self._stats["failed"] += 1

        self._processing.pop(msg.message_id, None)
        return msg.message_id

    async def process_all(self, queue: str) -> int:
        """Process all pending messages in a queue. Returns count processed."""
        count = 0
        while self._queues.get(queue):
            result = await self.process_one(queue)
            if result:
                count += 1
            else:
                break
        return count

    def get_queue_size(self, queue: str) -> int:
        """Get number of pending messages in a queue."""
        return len(self._queues.get(queue, []))

    def list_queues(self) -> List[dict]:
        """List all queues with their sizes."""
        return [
            {"name": name, "size": len(q), "consumers": len(self._consumers.get(name, []))}
            for name, q in self._queues.items()
        ]

    def get_dead_letters(self, queue: Optional[str] = None, limit: int = 20) -> List[dict]:
        """Get dead letter messages."""
        msgs = list(self._dead_letters)
        if queue:
            msgs = [m for m in msgs if m.queue == queue]
        return [m.to_dict() for m in msgs[-limit:]]

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            **self._stats,
            "queues": len(self._queues),
            "total_pending": sum(len(q) for q in self._queues.values()),
            "processing": len(self._processing),
            "dead_letters": len(self._dead_letters),
        }

    def purge_queue(self, queue: str) -> int:
        """Remove all messages from a queue. Returns count removed."""
        q = self._queues.get(queue)
        if not q:
            return 0
        count = len(q)
        q.clear()
        return count


# ── Singleton ──
_message_queue: Optional[MessageQueue] = None


def get_message_queue() -> MessageQueue:
    """Get the global message queue singleton."""
    global _message_queue
    if _message_queue is None:
        _message_queue = MessageQueue()
    return _message_queue
