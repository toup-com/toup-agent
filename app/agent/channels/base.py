"""
Channel Base — Abstract interface for messaging channels.

Every channel adapter (Telegram, Discord, Slack, Web, …) implements this
interface so the agent runtime can interact with any channel uniformly.

Design Principles
-----------------
* Channel-specific logic lives **only** inside the adapter subclass.
* The agent runtime calls `channel.send_text()`, `channel.send_voice()`, etc.
* Inbound messages go through `on_message` / `on_photo` callbacks set by
  the orchestrator at startup.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data objects
# ------------------------------------------------------------------

class ChannelType(str, Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    WEB = "web"
    WHATSAPP = "whatsapp"
    SIGNAL = "signal"
    MATRIX = "matrix"
    API = "api"  # raw HTTP/REST callers


@dataclass
class InboundMessage:
    """Normalised inbound message from any channel."""

    channel: ChannelType
    channel_user_id: str          # Platform-specific user identifier
    channel_chat_id: str          # Platform-specific chat/group identifier
    text: str = ""
    media_paths: List[str] = field(default_factory=list)
    reply_to_text: Optional[str] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    raw: Any = None               # Original platform event for edge cases


@dataclass
class OutboundMessage:
    """Normalised outbound message to any channel."""

    channel_chat_id: str
    text: str = ""
    voice_path: Optional[str] = None
    photo_path: Optional[str] = None
    file_path: Optional[str] = None
    parse_mode: Optional[str] = None  # "HTML", "Markdown", None


# Callback type: async def on_message(msg: InboundMessage) -> None
MessageCallback = Callable[[InboundMessage], Coroutine[Any, Any, None]]


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------

class BaseChannel(ABC):
    """
    Abstract base for a messaging channel adapter.

    Subclasses must implement:
    * ``start()``  — Connect to the platform and begin listening.
    * ``stop()``   — Gracefully disconnect.
    * ``send_text()`` — Send a text message to a chat.
    * ``send_typing()`` — Show typing indicator.

    Optional overrides:
    * ``send_voice()`` — Send audio/voice note.
    * ``send_photo()`` — Send an image.
    * ``send_file()``  — Send an arbitrary file.
    """

    channel_type: ChannelType
    on_message: Optional[MessageCallback] = None

    def __init__(self, channel_type: ChannelType):
        self.channel_type = channel_type
        self.on_message = None

    def set_message_callback(self, callback: MessageCallback):
        """Set the handler that receives normalised inbound messages."""
        self.on_message = callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Connect and start polling / webhook listener."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully disconnect from the platform."""

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    @abstractmethod
    async def send_text(self, chat_id: str, text: str, parse_mode: Optional[str] = None) -> None:
        """Send a text message to a specific chat."""

    @abstractmethod
    async def send_typing(self, chat_id: str) -> None:
        """Show a typing / "working" indicator."""

    async def send_voice(self, chat_id: str, audio_path: str) -> None:
        """Send a voice note / audio file. Default: unsupported."""
        logger.warning("[%s] send_voice not implemented", self.channel_type.value)

    async def send_photo(self, chat_id: str, photo_path: str, caption: Optional[str] = None) -> None:
        """Send a photo / image. Default: unsupported."""
        logger.warning("[%s] send_photo not implemented", self.channel_type.value)

    async def send_file(self, chat_id: str, file_path: str, caption: Optional[str] = None) -> None:
        """Send an arbitrary file. Default: unsupported."""
        logger.warning("[%s] send_file not implemented", self.channel_type.value)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def dispatch(self, msg: InboundMessage) -> None:
        """
        Forward a normalised inbound message to the registered callback.
        Typically called by the adapter's internal event handler.
        """
        if self.on_message:
            await self.on_message(msg)
        else:
            logger.warning("[%s] No message callback set — dropping message", self.channel_type.value)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type={self.channel_type.value}>"
