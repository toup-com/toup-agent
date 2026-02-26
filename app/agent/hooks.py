"""
Plugin Hook System — Lifecycle event bus for the agent runtime.

Provides 12+ lifecycle hooks that plugins and channels can subscribe to.
Hooks run in registration order. Each hook handler is an async callable.

Usage:
    from app.agent.hooks import hook_bus, HookEvent

    # Register a handler
    @hook_bus.on(HookEvent.BEFORE_AGENT_START)
    async def my_handler(context: dict):
        print("Agent is about to start!")

    # Or register programmatically
    hook_bus.register(HookEvent.AFTER_TOOL_CALL, my_handler)

    # Emit an event
    await hook_bus.emit(HookEvent.BEFORE_AGENT_START, {"user_id": "..."})
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# Callback type for hook handlers
HookHandler = Callable[[Dict[str, Any]], Coroutine[Any, Any, Optional[Dict[str, Any]]]]


class HookEvent(str, Enum):
    """All lifecycle hook events."""

    # Agent lifecycle
    BEFORE_AGENT_START = "before_agent_start"
    AGENT_END = "agent_end"

    # Tool lifecycle
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"

    # Message lifecycle
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENDING = "message_sending"
    MESSAGE_SENT = "message_sent"

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Context management
    BEFORE_COMPACTION = "before_compaction"
    AFTER_COMPACTION = "after_compaction"

    # Memory lifecycle
    MEMORY_CREATED = "memory_created"
    MEMORY_MERGED = "memory_merged"
    MEMORY_DECAYED = "memory_decayed"

    # Channel lifecycle
    CHANNEL_CONNECTED = "channel_connected"
    CHANNEL_DISCONNECTED = "channel_disconnected"

    # System
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


class HookBus:
    """
    Central event bus for lifecycle hooks.

    Handlers are called in registration order. If a handler returns a dict,
    it is merged into the context for subsequent handlers (pipeline pattern).
    """

    def __init__(self):
        self._handlers: Dict[HookEvent, List[HookHandler]] = {
            event: [] for event in HookEvent
        }
        self._global_handlers: List[HookHandler] = []

    def register(self, event: HookEvent, handler: HookHandler) -> None:
        """Register a handler for a specific event."""
        self._handlers[event].append(handler)
        logger.debug("[HOOKS] Registered handler for %s: %s", event.value, handler.__name__)

    def register_global(self, handler: HookHandler) -> None:
        """Register a handler that receives ALL events."""
        self._global_handlers.append(handler)

    def on(self, event: HookEvent):
        """Decorator to register a hook handler."""
        def decorator(func: HookHandler):
            self.register(event, func)
            return func
        return decorator

    def unregister(self, event: HookEvent, handler: HookHandler) -> None:
        """Remove a specific handler from an event."""
        try:
            self._handlers[event].remove(handler)
        except ValueError:
            pass

    def clear(self, event: Optional[HookEvent] = None) -> None:
        """Clear all handlers for an event, or all events if None."""
        if event:
            self._handlers[event] = []
        else:
            for ev in HookEvent:
                self._handlers[ev] = []
            self._global_handlers = []

    async def emit(self, event: HookEvent, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Emit a hook event, calling all registered handlers.

        Args:
            event: The lifecycle event to emit.
            context: Data passed to each handler. Handlers can modify it.

        Returns:
            The final context dict after all handlers have run.
        """
        ctx = dict(context or {})
        ctx["_event"] = event.value

        handlers = self._handlers.get(event, []) + self._global_handlers

        for handler in handlers:
            try:
                result = await handler(ctx)
                if isinstance(result, dict):
                    ctx.update(result)
            except Exception:
                logger.exception(
                    "[HOOKS] Handler %s failed for event %s",
                    handler.__name__,
                    event.value,
                )

        return ctx

    def handler_count(self, event: Optional[HookEvent] = None) -> int:
        """Return the number of registered handlers."""
        if event:
            return len(self._handlers.get(event, []))
        return sum(len(h) for h in self._handlers.values()) + len(self._global_handlers)

    def status(self) -> Dict[str, int]:
        """Return handler counts per event (for admin/debug)."""
        return {
            event.value: len(handlers)
            for event, handlers in self._handlers.items()
            if handlers
        }


# Singleton instance
hook_bus = HookBus()


def get_hook_bus() -> HookBus:
    """Get the global hook bus instance."""
    return hook_bus
