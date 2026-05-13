"""Shared write path for trigger-generated Messages.

Mirror of `app.agent.routines.message_writer` but pins
`channel="trigger"` on the Conversation + Message rows. Mission
Control + the WS broadcast surface treat the two channels the same;
the distinction matters for routing fan-out (routine vs event) and
for the Day-as-Chat filter "show me trigger fires only."
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)


async def write_trigger_message(
    db: AsyncSession,
    *,
    user_id: str,
    content: str,
    source: str,                  # e.g. "email_received"
    trigger_id: Optional[str] = None,
    title: Optional[str] = None,
    model_used: Optional[str] = None,
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
    tz_override: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Tuple[str, Optional[str]]:
    """Create the Conversation + Message pair, bump DayChat counters,
    commit. Returns (message_id, day_chat_id).

    Identical shape to `write_routine_message`; channel pinned to
    "trigger". Broadcasting is the caller's responsibility (so the
    broadcast event can include the just-returned id).
    """
    from app.agent.conversation_resolver import resolve_or_create_day_conversation
    from app.db.message_helpers import resolve_day_chat_id_for_now
    from app.db.models import DayChat, Message

    day_chat_id = await resolve_day_chat_id_for_now(
        db, user_id, tz_override=tz_override,
    )

    convo_meta: dict = {"trigger_source": source}
    if trigger_id:
        convo_meta["trigger_id"] = trigger_id
    if extra_metadata:
        convo_meta.update(extra_metadata)

    # Reading A (Day-as-Chat): one trigger Conversation per
    # (user, day_chat, channel). See app/agent/conversation_resolver.py.
    conv = await resolve_or_create_day_conversation(
        db,
        user_id=user_id,
        day_chat_id=day_chat_id,
        channel="trigger",
        title=title or f"Trigger: {source}",
        metadata=convo_meta,
    )

    msg_id = str(uuid.uuid4())
    msg = Message(
        id=msg_id,
        conversation_id=conv.id,
        day_chat_id=day_chat_id,
        role="assistant",
        content=content,
        channel="trigger",
        source=source,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        model_used=model_used,
    )
    db.add(msg)
    await db.flush()

    if day_chat_id:
        dc = await db.get(DayChat, day_chat_id)
        if dc:
            dc.message_count = (dc.message_count or 0) + 1
            tt = (tokens_prompt or 0) + (tokens_completion or 0)
            if tt:
                dc.total_tokens = (dc.total_tokens or 0) + tt
            dc.last_message_at = datetime.utcnow()

    await db.commit()
    logger.info(
        "[trigger_writer] wrote source=%s user_id=%s msg_id=%s day_chat_id=%s tokens=%d",
        source, user_id[:8], msg_id, day_chat_id,
        (tokens_prompt or 0) + (tokens_completion or 0),
    )
    return msg_id, day_chat_id


async def broadcast_trigger_message(
    user_id: str,
    *,
    message_id: str,
    day_chat_id: Optional[str],
    source: str,
    content: str,
    model_used: Optional[str] = None,
    delivery_channels: Optional[list[str]] = None,
    trigger_name: Optional[str] = None,
) -> int:
    """WS broadcast + optional fan-out to Telegram / WhatsApp. Reuses
    the routines `channel_dispatcher` because the cross-channel
    plumbing is identical — only the originating channel name
    differs."""
    try:
        from app.api.ws_chat import broadcast_to_user
    except Exception as e:
        logger.debug("[trigger_writer] broadcast skipped — ws_chat unavailable: %s", e)
        ws_count = 0
    else:
        event = {
            "type": "message",
            "id": message_id,
            "day_chat_id": day_chat_id,
            "role": "assistant",
            "channel": "trigger",
            "source": source,
            "content": content,
            "model_used": model_used,
            "created_at": datetime.utcnow().isoformat(),
        }
        try:
            ws_count = await broadcast_to_user(user_id, event)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("[trigger_writer] broadcast failed: %s", e)
            ws_count = 0

    extras = [c for c in (delivery_channels or []) if c and c != "website"]
    if extras:
        try:
            from app.agent.routines.channel_dispatcher import deliver_to_extra_channels
            from app.db.database import async_session_maker
            results = await deliver_to_extra_channels(
                user_id=user_id,
                delivery_channels=extras,
                routine_name=trigger_name or source,
                content=content,
                db_session_maker=async_session_maker,
            )
            if results:
                logger.info(
                    "[trigger_writer] extra channel fan-out user=%s results=%s",
                    user_id[:8], results,
                )
        except Exception as e:  # pragma: no cover — top-level guard
            logger.warning(
                "[trigger_writer] extra channel dispatch failed: %s: %s",
                type(e).__name__, e,
            )

    return ws_count
