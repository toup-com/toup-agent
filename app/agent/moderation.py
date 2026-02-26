"""
Moderation Tools — Group management operations across channels.

Supports: timeout, kick, ban, unban, role management via Telegram Bot API.
Discord/Slack moderation can be added as channel adapters are wired.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def moderate_user(
    action: str,
    channel: str,
    chat_id: str,
    user_id: str,
    *,
    duration_seconds: int = 0,
    reason: str = "",
    bot_refs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a moderation action on a user.

    Args:
        action: timeout | kick | ban | unban | mute | unmute
        channel: telegram | discord | slack
        chat_id: Group/channel identifier
        user_id: Target user identifier
        duration_seconds: Duration for timeout/mute (0 = permanent for ban)
        reason: Reason for the action
        bot_refs: Bot references dict

    Returns:
        {"ok": True/False, "action": "...", ...}
    """
    bot_refs = bot_refs or {}

    try:
        if channel == "telegram":
            return await _moderate_telegram(
                action, chat_id, user_id,
                duration_seconds=duration_seconds,
                reason=reason,
                bot_refs=bot_refs,
            )
        elif channel == "discord":
            return {"ok": False, "error": "Discord moderation not yet wired"}
        elif channel == "slack":
            return {"ok": False, "error": "Slack moderation not yet wired"}
        else:
            return {"ok": False, "error": f"Moderation not supported on {channel}"}

    except Exception as e:
        logger.error(f"[MODERATION] {action} failed on {channel}: {e}")
        return {"ok": False, "error": str(e)}


async def _moderate_telegram(
    action: str,
    chat_id: str,
    user_id: str,
    *,
    duration_seconds: int = 0,
    reason: str = "",
    bot_refs: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute Telegram-specific moderation."""
    tg_bot = bot_refs.get("telegram_bot")
    if not tg_bot or not tg_bot.bot:
        return {"ok": False, "error": "Telegram bot not connected"}

    bot = tg_bot.bot
    cid = int(chat_id)
    uid = int(user_id)

    if action == "kick":
        await bot.ban_chat_member(chat_id=cid, user_id=uid)
        # Unban immediately to kick without permanent ban
        await bot.unban_chat_member(chat_id=cid, user_id=uid, only_if_banned=True)
        log_msg = f"Kicked user {uid} from {cid}"
        if reason:
            log_msg += f" (reason: {reason})"
        logger.info(f"[MODERATION] {log_msg}")
        return {"ok": True, "action": "kick", "user_id": user_id, "reason": reason}

    elif action == "ban":
        import datetime
        until = None
        if duration_seconds > 0:
            until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=duration_seconds)
        await bot.ban_chat_member(chat_id=cid, user_id=uid, until_date=until)
        logger.info(f"[MODERATION] Banned user {uid} from {cid} for {duration_seconds}s")
        return {"ok": True, "action": "ban", "user_id": user_id, "duration": duration_seconds}

    elif action == "unban":
        await bot.unban_chat_member(chat_id=cid, user_id=uid, only_if_banned=True)
        logger.info(f"[MODERATION] Unbanned user {uid} from {cid}")
        return {"ok": True, "action": "unban", "user_id": user_id}

    elif action == "timeout" or action == "mute":
        import datetime
        if duration_seconds <= 0:
            duration_seconds = 300  # Default 5 minutes
        until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=duration_seconds)
        from telegram import ChatPermissions
        await bot.restrict_chat_member(
            chat_id=cid,
            user_id=uid,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until,
        )
        logger.info(f"[MODERATION] Muted user {uid} in {cid} for {duration_seconds}s")
        return {"ok": True, "action": "mute", "user_id": user_id, "duration": duration_seconds}

    elif action == "unmute":
        from telegram import ChatPermissions
        await bot.restrict_chat_member(
            chat_id=cid,
            user_id=uid,
            permissions=ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_send_other_messages=True,
                can_add_web_page_previews=True,
            ),
        )
        logger.info(f"[MODERATION] Unmuted user {uid} in {cid}")
        return {"ok": True, "action": "unmute", "user_id": user_id}

    else:
        return {"ok": False, "error": f"Unknown moderation action: {action}"}
