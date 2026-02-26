"""
Slack Channel Adapter — Connects Slack to HexBrain agent runtime.

Uses slack-bolt (Socket Mode) to handle:
- DM conversations
- Channel messages with @mention
- Thread replies
- File uploads
- Reactions
- Slash commands (/hexbrain)
"""

import asyncio
import logging
from typing import Optional

from app.agent.channels.base import (
    BaseChannel,
    ChannelType,
    InboundMessage,
)

logger = logging.getLogger(__name__)


class SlackChannel(BaseChannel):
    """
    Slack channel adapter using slack-bolt (Socket Mode).

    Requires:
    - SLACK_BOT_TOKEN (xoxb-...)
    - SLACK_APP_TOKEN (xapp-...) for Socket Mode
    """

    def __init__(self, bot_token: str, app_token: str, allowed_channels: Optional[list] = None):
        super().__init__(ChannelType.SLACK)
        self.bot_token = bot_token
        self.app_token = app_token
        self.allowed_channels = set(allowed_channels or [])
        self._app = None
        self._handler = None
        self._task = None
        self._bot_user_id = None

    async def start(self) -> None:
        """Connect to Slack via Socket Mode and start listening."""
        try:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        except ImportError:
            logger.error("[SLACK] slack-bolt not installed. Run: pip install slack-bolt")
            return

        self._app = AsyncApp(token=self.bot_token)

        # Get bot user ID for mention detection
        try:
            from slack_sdk.web.async_client import AsyncWebClient
            client = AsyncWebClient(token=self.bot_token)
            auth = await client.auth_test()
            self._bot_user_id = auth["user_id"]
            logger.info("[SLACK] Bot user: %s (ID: %s)", auth.get("user", ""), self._bot_user_id)
        except Exception as e:
            logger.warning("[SLACK] Could not get bot user ID: %s", e)

        @self._app.event("message")
        async def handle_message(event, say, client):
            await self._on_message(event, say, client)

        @self._app.event("app_mention")
        async def handle_mention(event, say, client):
            await self._on_message(event, say, client)

        # Start Socket Mode handler
        self._handler = AsyncSocketModeHandler(self._app, self.app_token)
        self._task = asyncio.create_task(self._handler.start_async())
        logger.info("[SLACK] Channel started (Socket Mode)")

    async def _on_message(self, event: dict, say, client):
        """Handle an incoming Slack message event."""
        # Ignore bot messages
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts")

        # ACL check
        if self.allowed_channels and channel_id not in self.allowed_channels:
            return

        # Clean bot mention from text
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        # Handle file attachments
        media_paths = []
        for file_info in event.get("files", []):
            if file_info.get("mimetype", "").startswith("image/"):
                media_paths.append(file_info.get("url_private", ""))

        # Get user info
        display_name = user_id
        try:
            user_info = await client.users_info(user=user_id)
            profile = user_info.get("user", {}).get("profile", {})
            display_name = profile.get("display_name") or profile.get("real_name") or user_id
        except Exception:
            pass

        inbound = InboundMessage(
            channel=ChannelType.SLACK,
            channel_user_id=user_id,
            channel_chat_id=channel_id,
            text=text,
            media_paths=media_paths,
            username=user_id,
            display_name=display_name,
            raw={"event": event, "say": say, "thread_ts": thread_ts},
        )

        await self.dispatch(inbound)

    async def stop(self) -> None:
        """Disconnect from Slack."""
        if self._handler:
            await self._handler.close_async()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[SLACK] Channel stopped")

    async def send_text(self, chat_id: str, text: str, parse_mode: Optional[str] = None) -> None:
        """Send a text message to a Slack channel/DM."""
        if not self._app:
            return

        try:
            from slack_sdk.web.async_client import AsyncWebClient
            client = AsyncWebClient(token=self.bot_token)
            await client.chat_postMessage(channel=chat_id, text=text)
        except Exception:
            logger.exception("[SLACK] Failed to send message to %s", chat_id)

    async def send_typing(self, chat_id: str) -> None:
        """Slack doesn't have a direct typing indicator API for bots."""
        pass  # Slack shows typing automatically during API calls

    async def send_file(self, chat_id: str, file_path: str, caption: Optional[str] = None) -> None:
        """Send a file to a Slack channel."""
        if not self._app:
            return

        try:
            from slack_sdk.web.async_client import AsyncWebClient
            client = AsyncWebClient(token=self.bot_token)
            await client.files_upload_v2(
                channel=chat_id,
                file=file_path,
                initial_comment=caption,
            )
        except Exception:
            logger.exception("[SLACK] Failed to send file")

    async def send_photo(self, chat_id: str, photo_path: str, caption: Optional[str] = None) -> None:
        """Send a photo to a Slack channel."""
        await self.send_file(chat_id, photo_path, caption)
