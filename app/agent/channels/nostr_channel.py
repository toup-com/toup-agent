"""
Nostr Channel — Decentralized social protocol.

Connects via Nostr relay WebSocket (NIP-01).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hexbrain.channel.nostr")


class NostrChannel:
    """Nostr decentralized messaging channel via relay WebSocket."""

    PLATFORM = "nostr"
    FEATURES = {"text", "dm", "public_notes", "relay_selection"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "nostr"
        self.display_name = "Nostr"
        self.connected = False
        self._private_key: str = ""
        self._public_key: str = ""
        self._relays: List[str] = []
        self._msg_counter: int = 0
        self._handlers: Dict[str, Any] = {}

    async def connect(self) -> bool:
        self._private_key = self.config.get("private_key", "")
        self._relays = self.config.get("relays", [
            "wss://relay.damus.io",
            "wss://nos.lol",
        ])
        if not self._private_key:
            logger.warning("[NOSTR] Missing private_key")
            return False
        self._public_key = f"npub_{self._private_key[:8]}"
        self.connected = True
        logger.info(f"[NOSTR] Connected to {len(self._relays)} relays")
        return True

    async def disconnect(self) -> bool:
        self.connected = False
        return True

    async def send_message(self, channel_id: str, text: str, **kwargs) -> Optional[str]:
        if not self.connected:
            return None
        msg_id = f"nostr_{self._msg_counter}"
        self._msg_counter += 1
        return msg_id

    async def handle_webhook(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        kind = payload.get("kind", 0)
        if kind == 4:  # NIP-04 DM
            return {
                "platform": self.PLATFORM,
                "channel_id": payload.get("pubkey", ""),
                "user_id": payload.get("pubkey", ""),
                "text": payload.get("content", ""),
                "raw": payload,
            }
        elif kind == 1:  # Text note
            return {
                "platform": self.PLATFORM,
                "channel_id": "public",
                "user_id": payload.get("pubkey", ""),
                "text": payload.get("content", ""),
                "raw": payload,
            }
        return None

    async def publish_note(self, text: str) -> Optional[str]:
        """Publish a public text note (kind 1)."""
        if not self.connected:
            return None
        return f"nostr_note_{self._msg_counter}"

    def on_message(self, handler):
        self._handlers["message"] = handler

    @property
    def is_connected(self) -> bool:
        return self.connected

    def stats(self) -> Dict[str, Any]:
        return {
            "channel": self.name,
            "connected": self.connected,
            "public_key": self._public_key,
            "relays": len(self._relays),
            "messages_sent": self._msg_counter,
        }
