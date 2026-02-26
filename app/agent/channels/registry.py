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
    """Singleton-style registry for channel adapters."""

    def __init__(self):
        self._channels: Dict[ChannelType, BaseChannel] = {}

    def register(self, channel: BaseChannel) -> None:
        """Add a channel adapter to the registry."""
        if channel.channel_type in self._channels:
            logger.warning(
                "[REGISTRY] Replacing existing %s adapter", channel.channel_type.value
            )
        self._channels[channel.channel_type] = channel
        logger.info("[REGISTRY] Registered channel: %s", channel.channel_type.value)

    def get(self, channel_type: ChannelType) -> Optional[BaseChannel]:
        """Retrieve a channel adapter by type."""
        return self._channels.get(channel_type)

    def all(self) -> List[BaseChannel]:
        """Return all registered channel adapters."""
        return list(self._channels.values())

    def types(self) -> List[str]:
        """Return names of all registered channel types."""
        return [c.channel_type.value for c in self._channels.values()]

    async def start_all(self) -> None:
        """Start all registered channels."""
        for ch in self._channels.values():
            try:
                await ch.start()
                logger.info("[REGISTRY] Started channel: %s", ch.channel_type.value)
            except Exception:
                logger.exception("[REGISTRY] Failed to start %s", ch.channel_type.value)

    async def stop_all(self) -> None:
        """Stop all registered channels gracefully."""
        for ch in self._channels.values():
            try:
                await ch.stop()
                logger.info("[REGISTRY] Stopped channel: %s", ch.channel_type.value)
            except Exception:
                logger.exception("[REGISTRY] Failed to stop %s", ch.channel_type.value)

    def status(self) -> List[Dict]:
        """Return status summary for each channel (for admin API)."""
        results = []
        for ch in self._channels.values():
            results.append({
                "type": ch.channel_type.value,
                "class": ch.__class__.__name__,
            })
        return results
