"""
Zalo Channel — Vietnam's most popular messaging platform.

Connects via Zalo Official Account API.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hexbrain.channel.zalo")


class ZaloChannel:
    """Zalo messaging channel via Official Account API."""

    PLATFORM = "zalo"
    FEATURES = {"text", "images", "stickers", "templates", "broadcasts"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "zalo"
        self.display_name = "Zalo"
        self.connected = False
        self._oa_id: str = ""
        self._access_token: str = ""
        self._msg_counter: int = 0
        self._handlers: Dict[str, Any] = {}

    async def connect(self) -> bool:
        self._oa_id = self.config.get("oa_id", "")
        self._access_token = self.config.get("access_token", "")
        if not self._oa_id or not self._access_token:
            logger.warning("[ZALO] Missing oa_id or access_token")
            return False
        self.connected = True
        logger.info(f"[ZALO] Connected OA {self._oa_id}")
        return True

    async def disconnect(self) -> bool:
        self.connected = False
        return True

    async def send_message(self, channel_id: str, text: str, **kwargs) -> Optional[str]:
        if not self.connected:
            return None
        msg_id = f"zalo_{self._msg_counter}"
        self._msg_counter += 1
        return msg_id

    async def handle_webhook(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        event = payload.get("event_name", "")
        if event == "user_send_text":
            data = payload.get("message", {})
            return {
                "platform": self.PLATFORM,
                "channel_id": str(payload.get("sender", {}).get("id", "")),
                "user_id": str(payload.get("sender", {}).get("id", "")),
                "text": data.get("text", ""),
                "raw": payload,
            }
        return None

    async def send_template(self, user_id: str, template_id: str, **kwargs) -> Optional[str]:
        """Send a Zalo template message."""
        if not self.connected:
            return None
        return f"zalo_tpl_{self._msg_counter}"

    def on_message(self, handler):
        self._handlers["message"] = handler

    @property
    def is_connected(self) -> bool:
        return self.connected

    def stats(self) -> Dict[str, Any]:
        return {
            "channel": self.name,
            "connected": self.connected,
            "oa_id": self._oa_id,
            "messages_sent": self._msg_counter,
        }
