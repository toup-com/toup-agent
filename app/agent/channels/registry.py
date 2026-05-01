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
    def register(cls, channel: BaseChannel) -> None:
        """Add a channel adapter to the registry."""
        if channel.channel_type in cls._channels:
            logger.warning(
                "[REGISTRY] Replacing existing %s adapter", channel.channel_type.value
            )
        cls._channels[channel.channel_type] = channel
        logger.info("[REGISTRY] Registered channel: %s", channel.channel_type.value)

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
