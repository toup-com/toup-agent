"""
Signal Messenger Channel Adapter — Stub implementation.

Implements the channel interface for Signal Messenger using signal-cli.
This is a structural stub ready for production implementation.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("hexbrain.channel.signal")


class SignalChannel:
    """
    Signal Messenger channel adapter.
    
    Status: STUB — structural implementation, needs signal-cli integration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "signal"
        self.display_name = "Signal Messenger"
        self.connected = False
        self._handlers: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """Connect to Signal Messenger. Returns True on success."""
        logger.info(f"[{self.display_name}] Connecting...")
        # TODO: Implement signal-cli connection
        logger.warning(f"[{self.display_name}] Stub — not yet implemented")
        return False

    async def disconnect(self) -> bool:
        """Disconnect from Signal Messenger."""
        self.connected = False
        return True

    async def send_message(
        self,
        channel_id: str,
        text: str,
        reply_to: Optional[str] = None,
        media: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Send a message. Returns message ID."""
        logger.warning(f"[{self.display_name}] send_message stub called")
        return None

    async def send_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> bool:
        """React to a message."""
        return False

    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        new_text: str,
    ) -> bool:
        """Edit a sent message."""
        return False

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
    ) -> bool:
        """Delete a message."""
        return False

    async def get_channel_info(self, channel_id: str) -> Optional[dict]:
        """Get channel/group info."""
        return {"name": "signal", "type": "stub", "connected": self.connected}

    async def list_channels(self) -> List[dict]:
        """List available channels/groups."""
        return []

    def on_message(self, handler):
        """Register a message handler."""
        self._handlers["message"] = handler

    def on_reaction(self, handler):
        """Register a reaction handler."""
        self._handlers["reaction"] = handler

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def status(self) -> dict:
        return {
            "channel": self.name,
            "display_name": self.display_name,
            "connected": self.connected,
            "type": "stub",
        }
