"""
Cross-Channel Messaging — Send, react, edit, delete, pin messages
across any connected channel (Telegram, Discord, Slack, WhatsApp).

The agent can use these operations via the `message` tool to interact
with users across channels without needing channel-specific tools.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def send_cross_channel(
    channel: str,
    target: str,
    text: str,
    *,
    reply_to: Optional[str] = None,
    thread_id: Optional[str] = None,
    bot_refs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Send a message to any connected channel.

    Args:
        channel: Channel type (telegram, discord, slack, whatsapp)
        target: Target identifier (chat_id, channel_id, phone number)
        text: Message text
        reply_to: Message ID to reply to
        thread_id: Thread/topic ID
        bot_refs: Dict of bot references {telegram_bot, discord_bot, ...}

    Returns:
        {"ok": True, "message_id": "...", "channel": "..."} or error
    """
    bot_refs = bot_refs or {}

    try:
        if channel == "telegram":
            bot = bot_refs.get("telegram_bot")
            if not bot or not bot.bot:
                return {"ok": False, "error": "Telegram bot not connected"}
            kwargs = {"chat_id": int(target), "text": text}
            if reply_to:
                kwargs["reply_to_message_id"] = int(reply_to)
            if thread_id:
                kwargs["message_thread_id"] = int(thread_id)
            msg = await bot.bot.send_message(**kwargs)
            return {"ok": True, "message_id": str(msg.message_id), "channel": "telegram"}

        elif channel == "discord":
            return {"ok": False, "error": "Discord send not yet wired — channel adapter needed"}

        elif channel == "slack":
            return {"ok": False, "error": "Slack send not yet wired — channel adapter needed"}

        elif channel == "whatsapp":
            return {"ok": False, "error": "WhatsApp send not yet wired — channel adapter needed"}

        else:
            return {"ok": False, "error": f"Unknown channel: {channel}"}

    except Exception as e:
        logger.error(f"[CROSS-CHANNEL] Send failed: {channel}/{target}: {e}")
        return {"ok": False, "error": str(e)}


async def react_cross_channel(
    channel: str,
    target: str,
    message_id: str,
    emoji: str,
    *,
    bot_refs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add a reaction to a message on any channel."""
    bot_refs = bot_refs or {}

    try:
        if channel == "telegram":
            bot = bot_refs.get("telegram_bot")
            if not bot or not bot.bot:
                return {"ok": False, "error": "Telegram bot not connected"}
            from telegram import ReactionTypeEmoji
            await bot.bot.set_message_reaction(
                chat_id=int(target),
                message_id=int(message_id),
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
            return {"ok": True, "channel": "telegram", "emoji": emoji}

        elif channel == "discord":
            return {"ok": False, "error": "Discord react not yet wired"}

        else:
            return {"ok": False, "error": f"Reactions not supported on {channel}"}

    except Exception as e:
        logger.error(f"[CROSS-CHANNEL] React failed: {channel}/{target}/{message_id}: {e}")
        return {"ok": False, "error": str(e)}


async def edit_cross_channel(
    channel: str,
    target: str,
    message_id: str,
    new_text: str,
    *,
    bot_refs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Edit a message on any channel."""
    bot_refs = bot_refs or {}

    try:
        if channel == "telegram":
            bot = bot_refs.get("telegram_bot")
            if not bot or not bot.bot:
                return {"ok": False, "error": "Telegram bot not connected"}
            await bot.bot.edit_message_text(
                chat_id=int(target),
                message_id=int(message_id),
                text=new_text,
            )
            return {"ok": True, "channel": "telegram"}

        else:
            return {"ok": False, "error": f"Edit not supported on {channel}"}

    except Exception as e:
        logger.error(f"[CROSS-CHANNEL] Edit failed: {e}")
        return {"ok": False, "error": str(e)}


async def delete_cross_channel(
    channel: str,
    target: str,
    message_id: str,
    *,
    bot_refs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Delete a message on any channel."""
    bot_refs = bot_refs or {}

    try:
        if channel == "telegram":
            bot = bot_refs.get("telegram_bot")
            if not bot or not bot.bot:
                return {"ok": False, "error": "Telegram bot not connected"}
            await bot.bot.delete_message(
                chat_id=int(target),
                message_id=int(message_id),
            )
            return {"ok": True, "channel": "telegram"}

        else:
            return {"ok": False, "error": f"Delete not supported on {channel}"}

    except Exception as e:
        logger.error(f"[CROSS-CHANNEL] Delete failed: {e}")
        return {"ok": False, "error": str(e)}


async def pin_cross_channel(
    channel: str,
    target: str,
    message_id: str,
    *,
    bot_refs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pin a message on any channel."""
    bot_refs = bot_refs or {}

    try:
        if channel == "telegram":
            bot = bot_refs.get("telegram_bot")
            if not bot or not bot.bot:
                return {"ok": False, "error": "Telegram bot not connected"}
            await bot.bot.pin_chat_message(
                chat_id=int(target),
                message_id=int(message_id),
            )
            return {"ok": True, "channel": "telegram"}

        else:
            return {"ok": False, "error": f"Pin not supported on {channel}"}

    except Exception as e:
        logger.error(f"[CROSS-CHANNEL] Pin failed: {e}")
        return {"ok": False, "error": str(e)}
