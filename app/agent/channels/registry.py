"""
Channel Registry — Manages all active channel adapters.

The orchestrator uses the registry to:
* Start / stop all channels as a group.
* Route outbound messages to the correct channel.
* Enumerate active connections for admin / gateway APIs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.agent.channels.base import BaseChannel, ChannelType

logger = logging.getLogger(__name__)


class ChannelRegistry:
    """Process-wide registry of channel adapters.

    All methods are classmethods so callers in `agent_main.py` can do
    `ChannelRegistry.register(channel)` without first having to obtain
    or maintain an instance reference. The previous instance-method
    design produced a long-running silent bug where every call site
    raised `missing 1 required positional argument: 'channel'`, was
    swallowed by the surrounding try/except, and the registry never
    populated — breaking outbound routing for any code that looked up
    channels by type.
    """

    _channels: Dict[ChannelType, BaseChannel] = {}

    @classmethod
    def _set(cls, channel: BaseChannel) -> None:
        cls._channels[channel.channel_type] = channel
        logger.info("[REGISTRY] Registered channel: %s", channel.channel_type.value)

    @classmethod
    def register(cls, channel: BaseChannel) -> bool:
        """Add a channel adapter to the registry.

        REFUSES a silent replace. The previous body logged a warning and
        overwrote the slot, which orphaned a LIVE adapter: nothing called
        `stop()` on the evicted object, so its SSE stream, sidecar and
        inbound dispatch all kept running for the life of the process and
        every WhatsApp message was handled twice. A caller that genuinely
        means to swap the adapter must say so via `replace()`.

        Returns True when the slot now holds `channel`.
        """
        incumbent = cls._channels.get(channel.channel_type)
        if incumbent is not None and incumbent is not channel:
            logger.error(
                "[REGISTRY] refusing to replace live %s adapter (%s) with %s — "
                "call ChannelRegistry.replace() or unregister() first",
                channel.channel_type.value,
                incumbent.__class__.__name__,
                channel.__class__.__name__,
            )
            return False
        cls._set(channel)
        return True

    @classmethod
    async def replace(cls, channel: BaseChannel) -> None:
        """Stop + drop the incumbent, then register `channel` in its slot.

        The caller is responsible for holding whatever lock serializes
        restarts — this method does not create one.
        """
        incumbent = cls._channels.pop(channel.channel_type, None)
        if incumbent is not None and incumbent is not channel:
            try:
                await incumbent.stop()
            except Exception:
                logger.exception(
                    "[REGISTRY] stop failed while replacing %s",
                    channel.channel_type.value,
                )
        cls._set(channel)

    @classmethod
    def unregister(cls, channel_type: ChannelType) -> Optional[BaseChannel]:
        """Remove a channel adapter from the registry and return it.

        Does NOT stop the adapter — callers that already tore it down
        (agent_main's restart path) need only the slot cleared, and they
        used to reach into `_channels` directly to do it.
        """
        removed = cls._channels.pop(channel_type, None)
        if removed is not None:
            logger.info("[REGISTRY] Unregistered channel: %s", channel_type.value)
        return removed

    @classmethod
    def get(cls, channel_type: ChannelType) -> Optional[BaseChannel]:
        """Retrieve a channel adapter by type."""
        return cls._channels.get(channel_type)

    @classmethod
    def all(cls) -> List[BaseChannel]:
        """Return all registered channel adapters."""
        return list(cls._channels.values())

    @classmethod
    def types(cls) -> List[str]:
        """Return names of all registered channel types."""
        return [c.channel_type.value for c in cls._channels.values()]

    @classmethod
    async def start_all(cls) -> None:
        """Start all registered channels."""
        for ch in cls._channels.values():
            try:
                await ch.start()
                logger.info("[REGISTRY] Started channel: %s", ch.channel_type.value)
            except Exception:
                logger.exception("[REGISTRY] Failed to start %s", ch.channel_type.value)

    @classmethod
    async def stop_all(cls) -> None:
        """Stop all registered channels gracefully."""
        for ch in cls._channels.values():
            try:
                await ch.stop()
                logger.info("[REGISTRY] Stopped channel: %s", ch.channel_type.value)
            except Exception:
                logger.exception("[REGISTRY] Failed to stop %s", ch.channel_type.value)

    @classmethod
    def status(cls) -> List[Dict]:
        """Return status summary for each channel (for admin API)."""
        results = []
        for ch in cls._channels.values():
            results.append({
                "type": ch.channel_type.value,
                "class": ch.__class__.__name__,
            })
        return results
