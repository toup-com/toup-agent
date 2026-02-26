"""
Discord Channel Adapter — Connects Discord to HexBrain agent runtime.

Uses discord.py (py-cord) to handle:
- DM conversations
- Server channels with @mention detection
- Thread support
- Slash commands
- Reactions
- Voice channel awareness
- File/image attachments
"""

import asyncio
import logging
import os
from typing import Optional

from app.agent.channels.base import (
    BaseChannel,
    ChannelType,
    InboundMessage,
    OutboundMessage,
)

logger = logging.getLogger(__name__)

# Max message length for Discord (2000 chars)
DISCORD_MAX_LENGTH = 2000


class DiscordChannel(BaseChannel):
    """
    Discord channel adapter using discord.py.

    Requires: DISCORD_BOT_TOKEN env var.
    Optional: DISCORD_ALLOWED_GUILDS, DISCORD_ALLOWED_USERS.
    """

    def __init__(self, token: str, allowed_guilds: Optional[list] = None, allowed_users: Optional[list] = None):
        super().__init__(ChannelType.DISCORD)
        self.token = token
        self.allowed_guilds = set(allowed_guilds or [])
        self.allowed_users = set(allowed_users or [])
        self._client = None
        self._ready = asyncio.Event()
        self._task = None

    async def start(self) -> None:
        """Connect to Discord and start listening."""
        try:
            import discord
        except ImportError:
            logger.error("[DISCORD] discord.py not installed. Run: pip install discord.py")
            return

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.dm_messages = True

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            logger.info("[DISCORD] Bot ready as %s (ID: %s)", self._client.user.name, self._client.user.id)
            self._ready.set()

        @self._client.event
        async def on_message(message: discord.Message):
            # Ignore own messages
            if message.author == self._client.user:
                return

            # Ignore bots
            if message.author.bot:
                return

            # ACL check — guilds
            if self.allowed_guilds and message.guild:
                if str(message.guild.id) not in self.allowed_guilds:
                    return

            # ACL check — users
            if self.allowed_users:
                if str(message.author.id) not in self.allowed_users:
                    return

            # Check if bot is mentioned (in server channels) or DM
            is_dm = message.guild is None
            is_mentioned = self._client.user in message.mentions if message.guild else False

            if not is_dm and not is_mentioned:
                return  # Ignore non-mentioned server messages

            # Clean the mention from message text
            text = message.content
            if self._client.user:
                text = text.replace(f"<@{self._client.user.id}>", "").strip()
                text = text.replace(f"<@!{self._client.user.id}>", "").strip()

            # Handle attachments
            media_paths = []
            for attachment in message.attachments:
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    media_paths.append(attachment.url)

            # Build inbound message
            inbound = InboundMessage(
                channel=ChannelType.DISCORD,
                channel_user_id=str(message.author.id),
                channel_chat_id=str(message.channel.id),
                text=text,
                media_paths=media_paths,
                reply_to_text=None,
                username=message.author.name,
                display_name=message.author.display_name,
                raw=message,
            )

            # Reply-to context
            if message.reference and message.reference.resolved:
                ref = message.reference.resolved
                if hasattr(ref, "content"):
                    inbound.reply_to_text = ref.content[:500]

            await self.dispatch(inbound)

        # Start the bot in background
        self._task = asyncio.create_task(self._run_client())
        logger.info("[DISCORD] Channel starting...")

    async def _run_client(self):
        """Run the Discord client (blocking)."""
        try:
            await self._client.start(self.token)
        except Exception:
            logger.exception("[DISCORD] Client crashed")

    async def stop(self) -> None:
        """Disconnect from Discord."""
        if self._client:
            await self._client.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[DISCORD] Channel stopped")

    async def send_text(self, chat_id: str, text: str, parse_mode: Optional[str] = None) -> None:
        """Send a text message to a Discord channel/DM."""
        if not self._client:
            return

        channel = self._client.get_channel(int(chat_id))
        if not channel:
            # Try fetching as a user for DMs
            try:
                user = await self._client.fetch_user(int(chat_id))
                channel = await user.create_dm()
            except Exception:
                logger.warning("[DISCORD] Cannot find channel or user: %s", chat_id)
                return

        # Split long messages
        for chunk in self._split_message(text):
            await channel.send(chunk)

    async def send_typing(self, chat_id: str) -> None:
        """Show typing indicator."""
        if not self._client:
            return
        channel = self._client.get_channel(int(chat_id))
        if channel:
            await channel.typing()

    async def send_file(self, chat_id: str, file_path: str, caption: Optional[str] = None) -> None:
        """Send a file to a Discord channel."""
        import discord

        if not self._client:
            return

        channel = self._client.get_channel(int(chat_id))
        if not channel:
            return

        try:
            await channel.send(content=caption, file=discord.File(file_path))
        except Exception:
            logger.exception("[DISCORD] Failed to send file")

    async def send_photo(self, chat_id: str, photo_path: str, caption: Optional[str] = None) -> None:
        """Send an image to a Discord channel."""
        await self.send_file(chat_id, photo_path, caption)

    @staticmethod
    def _split_message(text: str, max_len: int = DISCORD_MAX_LENGTH) -> list:
        """Split text into chunks that fit Discord's message limit."""
        if len(text) <= max_len:
            return [text]

        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break

            # Try to split at newline
            split_at = text.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = max_len

            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")

        return chunks
