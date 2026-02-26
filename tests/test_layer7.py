"""
Layer 7 Tests — Event Bus, Message Queue, Structured Logging, Channel Stubs.

Tests: 65+ tests covering all Layer 7 backend components.
"""

import asyncio
import json
import logging
import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch


# ═══════════════════════════════════════════════════════════════
# 1. EVENT BUS
# ═══════════════════════════════════════════════════════════════

class TestEventBusCore(unittest.TestCase):
    """Test Event Bus core functionality."""

    def test_import(self):
        from app.agent.event_bus import EventBus, Event, EventPriority, get_event_bus
        self.assertIsNotNone(EventBus)
        self.assertIsNotNone(get_event_bus)

    def test_event_creation(self):
        from app.agent.event_bus import Event, EventPriority
        evt = Event(topic="test.created", data={"key": "value"}, source="test")
        self.assertEqual(evt.topic, "test.created")
        self.assertEqual(evt.data, {"key": "value"})
        self.assertEqual(evt.priority, EventPriority.NORMAL)
        self.assertIn("evt_", evt.event_id)

    def test_event_to_dict(self):
        from app.agent.event_bus import Event
        evt = Event(topic="test", data="hello")
        d = evt.to_dict()
        self.assertEqual(d["topic"], "test")
        self.assertEqual(d["data"], "hello")
        self.assertIn("event_id", d)
        self.assertIn("timestamp", d)

    def test_priority_enum(self):
        from app.agent.event_bus import EventPriority
        self.assertEqual(EventPriority.LOW, "low")
        self.assertEqual(EventPriority.NORMAL, "normal")
        self.assertEqual(EventPriority.HIGH, "high")
        self.assertEqual(EventPriority.CRITICAL, "critical")


class TestEventBusSubscription(unittest.TestCase):
    """Test subscribe/unsubscribe."""

    def test_subscribe(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        handler = MagicMock()
        sub = bus.subscribe("test.topic", handler, subscriber_id="sub1")
        self.assertEqual(sub.topic, "test.topic")
        self.assertEqual(sub.subscriber_id, "sub1")

    def test_unsubscribe(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        handler = MagicMock()
        sub = bus.subscribe("test.topic", handler)
        result = bus.unsubscribe(sub)
        self.assertTrue(result)

    def test_unsubscribe_not_found(self):
        from app.agent.event_bus import EventBus, Subscription
        bus = EventBus()
        fake_sub = Subscription(topic="nope", handler=lambda e: None)
        result = bus.unsubscribe(fake_sub)
        self.assertFalse(result)

    def test_unsubscribe_all(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        handler = MagicMock()
        bus.subscribe("a", handler, subscriber_id="x")
        bus.subscribe("b", handler, subscriber_id="x")
        bus.subscribe("c", handler, subscriber_id="y")
        count = bus.unsubscribe_all("x")
        self.assertEqual(count, 2)


class TestEventBusPublish(unittest.TestCase):
    """Test event publishing and delivery."""

    def test_publish_sync_handler(self):
        from app.agent.event_bus import EventBus, Event
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e.data))
        event = Event(topic="test", data="hello")
        count = asyncio.get_event_loop().run_until_complete(bus.publish(event))
        self.assertEqual(count, 1)
        self.assertEqual(received, ["hello"])

    def test_publish_async_handler(self):
        from app.agent.event_bus import EventBus, Event
        bus = EventBus()
        received = []
        async def handler(e):
            received.append(e.data)
        bus.subscribe("test", handler)
        event = Event(topic="test", data="async_hello")
        asyncio.get_event_loop().run_until_complete(bus.publish(event))
        self.assertEqual(received, ["async_hello"])

    def test_emit_convenience(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        received = []
        bus.subscribe("quick", lambda e: received.append(e.data))
        asyncio.get_event_loop().run_until_complete(bus.emit("quick", data="fast"))
        self.assertEqual(received, ["fast"])

    def test_wildcard_subscription(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        received = []
        bus.subscribe("agent.*", lambda e: received.append(e.topic))
        asyncio.get_event_loop().run_until_complete(bus.emit("agent.started"))
        asyncio.get_event_loop().run_until_complete(bus.emit("agent.stopped"))
        asyncio.get_event_loop().run_until_complete(bus.emit("memory.stored"))
        self.assertEqual(received, ["agent.started", "agent.stopped"])

    def test_star_catches_all(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e.topic))
        asyncio.get_event_loop().run_until_complete(bus.emit("anything"))
        asyncio.get_event_loop().run_until_complete(bus.emit("else.here"))
        self.assertEqual(len(received), 2)

    def test_publish_with_priority(self):
        from app.agent.event_bus import EventBus, EventPriority
        bus = EventBus()
        order = []
        bus.subscribe("test", lambda e: order.append("high"), priority=EventPriority.HIGH)
        bus.subscribe("test", lambda e: order.append("low"), priority=EventPriority.LOW)
        asyncio.get_event_loop().run_until_complete(bus.emit("test"))
        self.assertEqual(order[0], "high")

    def test_handler_error_goes_to_dead_letter(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        def bad_handler(e):
            raise ValueError("boom")
        bus.subscribe("test", bad_handler, subscriber_id="broken")
        asyncio.get_event_loop().run_until_complete(bus.emit("test"))
        dead = bus.get_dead_letters()
        self.assertTrue(len(dead) > 0)
        self.assertIn("boom", dead[0]["error"])


class TestEventBusHistory(unittest.TestCase):
    """Test event history and stats."""

    def test_history(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        asyncio.get_event_loop().run_until_complete(bus.emit("a", data=1))
        asyncio.get_event_loop().run_until_complete(bus.emit("b", data=2))
        history = bus.get_history()
        self.assertEqual(len(history), 2)

    def test_history_filter_by_topic(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        asyncio.get_event_loop().run_until_complete(bus.emit("a"))
        asyncio.get_event_loop().run_until_complete(bus.emit("b"))
        asyncio.get_event_loop().run_until_complete(bus.emit("a"))
        history = bus.get_history(topic="a")
        self.assertEqual(len(history), 2)

    def test_stats(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        asyncio.get_event_loop().run_until_complete(bus.emit("test"))
        stats = bus.get_stats()
        self.assertEqual(stats["published"], 1)
        self.assertEqual(stats["delivered"], 1)
        self.assertIn("topics", stats)

    def test_list_topics(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        topics = bus.list_topics()
        self.assertIn("a", topics)
        self.assertIn("b", topics)

    def test_clear_history(self):
        from app.agent.event_bus import EventBus
        bus = EventBus()
        asyncio.get_event_loop().run_until_complete(bus.emit("test"))
        bus.clear_history()
        self.assertEqual(len(bus.get_history()), 0)

    def test_singleton(self):
        from app.agent.event_bus import get_event_bus
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        self.assertIs(bus1, bus2)


# ═══════════════════════════════════════════════════════════════
# 2. MESSAGE QUEUE
# ═══════════════════════════════════════════════════════════════

class TestMessageQueueCore(unittest.TestCase):
    """Test Message Queue core functionality."""

    def test_import(self):
        from app.agent.message_queue import MessageQueue, QueueMessage, MessageStatus, get_message_queue
        self.assertIsNotNone(MessageQueue)

    def test_message_status_enum(self):
        from app.agent.message_queue import MessageStatus
        self.assertEqual(MessageStatus.PENDING, "pending")
        self.assertEqual(MessageStatus.COMPLETED, "completed")
        self.assertEqual(MessageStatus.DEAD, "dead")

    def test_delivery_mode_enum(self):
        from app.agent.message_queue import DeliveryMode
        self.assertEqual(DeliveryMode.AT_MOST_ONCE, "at_most_once")
        self.assertEqual(DeliveryMode.AT_LEAST_ONCE, "at_least_once")

    def test_create_queue(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        mq.create_queue("test_q")
        queues = mq.list_queues()
        self.assertEqual(len(queues), 1)
        self.assertEqual(queues[0]["name"], "test_q")

    def test_enqueue(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        msg_id = asyncio.get_event_loop().run_until_complete(
            mq.enqueue("orders", {"item": "widget"})
        )
        self.assertIsNotNone(msg_id)
        self.assertEqual(mq.get_queue_size("orders"), 1)

    def test_enqueue_auto_creates_queue(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        asyncio.get_event_loop().run_until_complete(mq.enqueue("auto_q", "data"))
        self.assertEqual(mq.get_queue_size("auto_q"), 1)


class TestMessageQueueProcessing(unittest.TestCase):
    """Test message processing."""

    def test_process_one(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        processed = []
        mq.register_consumer("q", lambda payload: processed.append(payload))
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", "hello"))
        result = asyncio.get_event_loop().run_until_complete(mq.process_one("q"))
        self.assertIsNotNone(result)
        self.assertEqual(processed, ["hello"])

    def test_process_async_consumer(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        processed = []
        async def handler(payload):
            processed.append(payload)
        mq.register_consumer("q", handler)
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", "async_data"))
        asyncio.get_event_loop().run_until_complete(mq.process_one("q"))
        self.assertEqual(processed, ["async_data"])

    def test_process_all(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        count = []
        mq.register_consumer("q", lambda p: count.append(p))
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", 1))
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", 2))
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", 3))
        n = asyncio.get_event_loop().run_until_complete(mq.process_all("q"))
        self.assertEqual(n, 3)
        self.assertEqual(count, [1, 2, 3])

    def test_no_consumers_keeps_message(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", "data"))
        result = asyncio.get_event_loop().run_until_complete(mq.process_one("q"))
        self.assertIsNone(result)  # No consumer, can't process
        self.assertEqual(mq.get_queue_size("q"), 1)  # Still there

    def test_failed_consumer_retries(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        call_count = [0]
        def flaky(payload):
            call_count[0] += 1
            if call_count[0] <= 1:
                raise ValueError("transient error")
        mq.register_consumer("q", flaky)
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", "retry_me", max_retries=3))
        asyncio.get_event_loop().run_until_complete(mq.process_one("q"))
        # Should have retried — message back in queue
        stats = mq.get_stats()
        self.assertTrue(stats["retried"] > 0 or stats["failed"] > 0)

    def test_dead_letter_after_max_retries(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        def always_fail(payload):
            raise ValueError("always fails")
        mq.register_consumer("q", always_fail)
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", "doomed", max_retries=1))
        asyncio.get_event_loop().run_until_complete(mq.process_one("q"))
        dead = mq.get_dead_letters()
        self.assertTrue(len(dead) > 0)

    def test_purge_queue(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", 1))
        asyncio.get_event_loop().run_until_complete(mq.enqueue("q", 2))
        count = mq.purge_queue("q")
        self.assertEqual(count, 2)
        self.assertEqual(mq.get_queue_size("q"), 0)


class TestMessageQueueStats(unittest.TestCase):
    """Test queue statistics and listing."""

    def test_stats(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        stats = mq.get_stats()
        self.assertIn("enqueued", stats)
        self.assertIn("processed", stats)
        self.assertIn("queues", stats)

    def test_list_queues(self):
        from app.agent.message_queue import MessageQueue
        mq = MessageQueue()
        mq.create_queue("a")
        mq.create_queue("b")
        queues = mq.list_queues()
        names = [q["name"] for q in queues]
        self.assertIn("a", names)
        self.assertIn("b", names)

    def test_singleton(self):
        from app.agent.message_queue import get_message_queue
        q1 = get_message_queue()
        q2 = get_message_queue()
        self.assertIs(q1, q2)


# ═══════════════════════════════════════════════════════════════
# 3. STRUCTURED LOGGING
# ═══════════════════════════════════════════════════════════════

class TestStructuredLogging(unittest.TestCase):
    """Test structured logging system."""

    def test_import(self):
        from app.agent.structured_logging import (
            Subsystem, StructuredFormatter, SubsystemLogger,
            get_subsystem_logger, set_request_context, generate_request_id,
        )
        self.assertIsNotNone(Subsystem)
        self.assertIsNotNone(StructuredFormatter)

    def test_subsystem_enum(self):
        from app.agent.structured_logging import Subsystem
        self.assertEqual(Subsystem.AGENT, "agent")
        self.assertEqual(Subsystem.MEMORY, "memory")
        self.assertEqual(Subsystem.CHANNEL, "channel")
        self.assertEqual(Subsystem.TOOL, "tool")
        self.assertEqual(Subsystem.API, "api")
        self.assertEqual(Subsystem.DB, "db")
        self.assertEqual(Subsystem.VOICE, "voice")
        self.assertEqual(Subsystem.CANVAS, "canvas")

    def test_get_subsystem_logger(self):
        from app.agent.structured_logging import get_subsystem_logger, Subsystem
        log = get_subsystem_logger(Subsystem.AGENT)
        self.assertIsNotNone(log)

    def test_logger_methods(self):
        from app.agent.structured_logging import get_subsystem_logger, Subsystem
        log = get_subsystem_logger(Subsystem.TOOL)
        # Should not raise
        log.debug("test debug")
        log.info("test info")
        log.warning("test warning")
        log.error("test error")

    def test_generate_request_id(self):
        from app.agent.structured_logging import generate_request_id
        rid = generate_request_id()
        self.assertTrue(len(rid) > 0)
        self.assertTrue(len(rid) <= 12)

    def test_set_request_context(self):
        from app.agent.structured_logging import set_request_context, request_id_var, user_id_var
        set_request_context(request_id="req123", user_id="user456")
        self.assertEqual(request_id_var.get(), "req123")
        self.assertEqual(user_id_var.get(), "user456")

    def test_structured_formatter_json(self):
        from app.agent.structured_logging import StructuredFormatter
        fmt = StructuredFormatter()
        record = logging.LogRecord(
            name="hexbrain.test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        record.subsystem = "agent"
        output = fmt.format(record)
        data = json.loads(output)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["subsystem"], "agent")
        self.assertEqual(data["message"], "test message")

    def test_convenience_loggers(self):
        from app.agent.structured_logging import agent_log, memory_log, channel_log, tool_log, api_log, db_log
        self.assertIsNotNone(agent_log)
        self.assertIsNotNone(memory_log)
        self.assertIsNotNone(channel_log)
        self.assertIsNotNone(tool_log)
        self.assertIsNotNone(api_log)
        self.assertIsNotNone(db_log)


# ═══════════════════════════════════════════════════════════════
# 4. CHANNEL STUBS
# ═══════════════════════════════════════════════════════════════

class TestChannelStubs(unittest.TestCase):
    """Test all channel adapter stubs."""

    def test_signal_import(self):
        from app.agent.channels.signal_channel import SignalChannel
        ch = SignalChannel()
        self.assertEqual(ch.name, "signal")
        self.assertFalse(ch.is_connected)

    def test_teams_import(self):
        from app.agent.channels.teams_channel import TeamsChannel
        ch = TeamsChannel()
        self.assertEqual(ch.name, "teams")

    def test_matrix_import(self):
        from app.agent.channels.matrix_channel import MatrixChannel
        ch = MatrixChannel()
        self.assertEqual(ch.name, "matrix")

    def test_line_import(self):
        from app.agent.channels.line_channel import LINEChannel
        ch = LINEChannel()
        self.assertEqual(ch.name, "line")

    def test_google_chat_import(self):
        from app.agent.channels.google_chat_channel import GoogleChatChannel
        ch = GoogleChatChannel()
        self.assertEqual(ch.name, "google_chat")

    def test_imessage_import(self):
        from app.agent.channels.imessage_channel import IMessageChannel
        ch = IMessageChannel()
        self.assertEqual(ch.name, "imessage")

    def test_channel_connect_returns_false(self):
        """Stub channels should return False (not implemented)."""
        from app.agent.channels.signal_channel import SignalChannel
        ch = SignalChannel()
        result = asyncio.get_event_loop().run_until_complete(ch.connect())
        self.assertFalse(result)

    def test_channel_disconnect(self):
        from app.agent.channels.teams_channel import TeamsChannel
        ch = TeamsChannel()
        result = asyncio.get_event_loop().run_until_complete(ch.disconnect())
        self.assertTrue(result)

    def test_channel_send_message_stub(self):
        from app.agent.channels.matrix_channel import MatrixChannel
        ch = MatrixChannel()
        result = asyncio.get_event_loop().run_until_complete(
            ch.send_message("ch1", "hello")
        )
        self.assertIsNone(result)

    def test_channel_send_reaction_stub(self):
        from app.agent.channels.line_channel import LINEChannel
        ch = LINEChannel()
        result = asyncio.get_event_loop().run_until_complete(
            ch.send_reaction("ch1", "msg1", "👍")
        )
        self.assertFalse(result)

    def test_channel_status(self):
        from app.agent.channels.google_chat_channel import GoogleChatChannel
        ch = GoogleChatChannel()
        status = ch.status
        self.assertEqual(status["channel"], "google_chat")
        self.assertEqual(status["type"], "stub")
        self.assertFalse(status["connected"])

    def test_channel_list_channels(self):
        from app.agent.channels.imessage_channel import IMessageChannel
        ch = IMessageChannel()
        channels = asyncio.get_event_loop().run_until_complete(ch.list_channels())
        self.assertEqual(channels, [])

    def test_channel_on_message_handler(self):
        from app.agent.channels.signal_channel import SignalChannel
        ch = SignalChannel()
        handler = MagicMock()
        ch.on_message(handler)
        self.assertEqual(ch._handlers.get("message"), handler)

    def test_channel_get_info(self):
        from app.agent.channels.teams_channel import TeamsChannel
        ch = TeamsChannel()
        info = asyncio.get_event_loop().run_until_complete(ch.get_channel_info("ch1"))
        self.assertIn("name", info)
        self.assertEqual(info["type"], "stub")

    def test_all_channels_have_consistent_interface(self):
        """Verify all channel stubs have the same methods."""
        from app.agent.channels.signal_channel import SignalChannel
        from app.agent.channels.teams_channel import TeamsChannel
        from app.agent.channels.matrix_channel import MatrixChannel
        from app.agent.channels.line_channel import LINEChannel
        from app.agent.channels.google_chat_channel import GoogleChatChannel
        from app.agent.channels.imessage_channel import IMessageChannel

        required_methods = ["connect", "disconnect", "send_message", "send_reaction",
                           "edit_message", "delete_message", "get_channel_info", "list_channels"]

        for cls in [SignalChannel, TeamsChannel, MatrixChannel, LINEChannel, GoogleChatChannel, IMessageChannel]:
            ch = cls()
            for method in required_methods:
                self.assertTrue(hasattr(ch, method), f"{cls.__name__} missing {method}")


# ═══════════════════════════════════════════════════════════════
# 5. TOPIC MATCHING EDGE CASES
# ═══════════════════════════════════════════════════════════════

class TestTopicMatching(unittest.TestCase):
    """Test event bus topic matching edge cases."""

    def test_exact_match(self):
        from app.agent.event_bus import EventBus
        self.assertTrue(EventBus._topic_matches("foo", "foo"))

    def test_no_match(self):
        from app.agent.event_bus import EventBus
        self.assertFalse(EventBus._topic_matches("foo", "bar"))

    def test_wildcard_match(self):
        from app.agent.event_bus import EventBus
        self.assertTrue(EventBus._topic_matches("agent.*", "agent.started"))
        self.assertTrue(EventBus._topic_matches("agent.*", "agent.stopped"))
        self.assertFalse(EventBus._topic_matches("agent.*", "memory.stored"))

    def test_star_all(self):
        from app.agent.event_bus import EventBus
        self.assertTrue(EventBus._topic_matches("*", "anything"))
        self.assertTrue(EventBus._topic_matches("*", "some.nested.topic"))

    def test_wildcard_prefix_only(self):
        from app.agent.event_bus import EventBus
        self.assertTrue(EventBus._topic_matches("agent.*", "agent"))


if __name__ == "__main__":
    unittest.main()
