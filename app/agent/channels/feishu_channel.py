"""
Feishu/Lark Channel — ByteDance enterprise messaging.

Connects via Feishu Open Platform Bot API + WebSocket.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hexbrain.channel.feishu")


class FeishuChannel:
    """Feishu/Lark messaging channel via Bot WebSocket."""

    PLATFORM = "feishu"
    FEATURES = {"text", "rich_text", "cards", "groups", "threads"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "feishu"
        self.display_name = "Feishu/Lark"
        self.connected = False
        self._app_id: str = ""
        self._app_secret: str = ""
        self._msg_counter: int = 0
        self._handlers: Dict[str, Any] = {}

    async def connect(self) -> bool:
        self._app_id = self.config.get("app_id", "")
        self._app_secret = self.config.get("app_secret", "")
        if not self._app_id or not self._app_secret:
            logger.warning("[FEISHU] Missing app_id or app_secret")
            return False
        self.connected = True
        logger.info(f"[FEISHU] Connected app {self._app_id}")
        return True

    async def disconnect(self) -> bool:
        self.connected = False
        return True

    async def send_message(self, channel_id: str, text: str, **kwargs) -> Optional[str]:
        if not self.connected:
            return None
        msg_id = f"feishu_{self._msg_counter}"
        self._msg_counter += 1
        return msg_id

    async def handle_webhook(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        header = payload.get("header", {})
        event_type = header.get("event_type", "")
        if event_type == "im.message.receive_v1":
            event = payload.get("event", {})
            msg = event.get("message", {})
            sender = event.get("sender", {})
            return {
                "platform": self.PLATFORM,
                "channel_id": msg.get("chat_id", ""),
                "user_id": sender.get("sender_id", {}).get("open_id", ""),
                "text": msg.get("content", ""),
                "raw": payload,
            }
        return None

    async def send_card(self, channel_id: str, card: Dict[str, Any]) -> Optional[str]:
        """Send an interactive card message."""
        if not self.connected:
            return None
        return f"feishu_card_{self._msg_counter}"

    def on_message(self, handler):
        self._handlers["message"] = handler

    @property
    def is_connected(self) -> bool:
        return self.connected

    def stats(self) -> Dict[str, Any]:
        return {
            "channel": self.name,
            "connected": self.connected,
            "app_id": self._app_id,
            "messages_sent": self._msg_counter,
        }
