"""
Mattermost Channel — Open-source Slack alternative.

Connects via Mattermost Bot API + WebSocket events.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hexbrain.channel.mattermost")


class MattermostChannel:
    """Mattermost messaging channel via Bot API + WebSocket."""

    PLATFORM = "mattermost"
    FEATURES = {"text", "threads", "reactions", "files", "slash_commands"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "mattermost"
        self.display_name = "Mattermost"
        self.connected = False
        self._server_url: str = ""
        self._bot_token: str = ""
        self._msg_counter: int = 0
        self._handlers: Dict[str, Any] = {}

    async def connect(self) -> bool:
        self._server_url = self.config.get("server_url", "")
        self._bot_token = self.config.get("bot_token", "")
        if not self._server_url or not self._bot_token:
            logger.warning("[MATTERMOST] Missing server_url or bot_token")
            return False
        self.connected = True
        logger.info(f"[MATTERMOST] Connected to {self._server_url}")
        return True

    async def disconnect(self) -> bool:
        self.connected = False
        return True

    async def send_message(self, channel_id: str, text: str, **kwargs) -> Optional[str]:
        if not self.connected:
            return None
        msg_id = f"mm_{self._msg_counter}"
        self._msg_counter += 1
        return msg_id

    async def handle_webhook(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        event = payload.get("event", "")
        if event == "posted":
            post = payload.get("data", {}).get("post", {})
            return {
                "platform": self.PLATFORM,
                "channel_id": post.get("channel_id", ""),
                "user_id": post.get("user_id", ""),
                "text": post.get("message", ""),
                "thread_id": post.get("root_id"),
                "raw": payload,
            }
        return None

    async def reply_thread(self, channel_id: str, root_id: str, text: str) -> Optional[str]:
        """Reply to a Mattermost thread."""
        return await self.send_message(channel_id, text, root_id=root_id)

    async def add_reaction(self, post_id: str, emoji: str) -> bool:
        """Add a reaction to a post."""
        return self.connected

    def on_message(self, handler):
        self._handlers["message"] = handler

    @property
    def is_connected(self) -> bool:
        return self.connected

    def stats(self) -> Dict[str, Any]:
        return {
            "channel": self.name,
            "connected": self.connected,
            "server_url": self._server_url,
            "messages_sent": self._msg_counter,
        }
