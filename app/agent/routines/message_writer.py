"""Shared write path for routine-generated Messages.

Both the success path (handler) and the nudge path (runner) post a
Message via this helper so the row shape stays consistent — same
`channel="routine"`, same Conversation creation, same DayChat counter
bumps, same live WS broadcast.

Day-as-Chat rule: `Message.day_chat_id` is canonical. Always resolved
via `resolve_day_chat_id_for_now()` — never trust `Conversation.day_chat_id`,
which is a loose hint for long-lived sessions.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)


async def write_routine_message(
    db: AsyncSession,
    *,
    user_id: str,
    content: str,
    source: str,
    routine_id: Optional[str] = None,
    title: Optional[str] = None,
    model_used: Optional[str] = None,
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
    tz_override: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Tuple[str, Optional[str]]:
    """Create a Conversation + Message pair representing one routine
    post, bump DayChat counters, commit. Returns (message_id, day_chat_id).

    Does NOT broadcast — the caller does that AFTER receiving the id so
    the broadcast event can carry it. Broadcast is also explicitly
    isolated so a test environment without ws_chat doesn't error here.
    """
    from app.agent.conversation_resolver import resolve_or_create_day_conversation
    from app.db.message_helpers import resolve_day_chat_id_for_now
    from app.db.models import DayChat, Message

    day_chat_id = await resolve_day_chat_id_for_now(db, user_id, tz_override=tz_override)

    convo_meta = {"routine_source": source}
    if routine_id:
        convo_meta["routine_id"] = routine_id
    if extra_metadata:
        convo_meta.update(extra_metadata)

    # Reading A (Day-as-Chat): one routine Conversation per
    # (user, day_chat, channel). All routine fires on the same day
    # append to the same row. See app/agent/conversation_resolver.py.
    conv = await resolve_or_create_day_conversation(
        db,
        user_id=user_id,
        day_chat_id=day_chat_id,
        channel="routine",
        title=title or f"Routine: {source}",
        metadata=convo_meta,
    )

    msg_id = str(uuid.uuid4())
    msg = Message(
        id=msg_id,
        conversation_id=conv.id,
        day_chat_id=day_chat_id,
        role="assistant",
        content=content,
        channel="routine",
        source=source,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        model_used=model_used,
    )
    db.add(msg)
    await db.flush()

    # DayChat counters — same bookkeeping as agent_runner._save_messages
    # (lines 2596-2606). Without these, the per-day budget logs drift
    # and the UI count gets stuck.
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
        "[routine_writer] wrote source=%s user_id=%s msg_id=%s day_chat_id=%s tokens=%d",
        source, user_id, msg_id, day_chat_id,
        (tokens_prompt or 0) + (tokens_completion or 0),
    )
    return msg_id, day_chat_id


async def broadcast_routine_message(
    user_id: str,
    *,
    message_id: str,
    day_chat_id: Optional[str],
    source: str,
    content: str,
    model_used: Optional[str] = None,
    delivery_channels: Optional[list[str]] = None,
    routine_name: Optional[str] = None,
    extra: Optional[dict] = None,
):
    """Push a routine Message to any live WS clients AND (optionally) to
    extra outbound channels the routine asked for (Telegram / WhatsApp).

    Return shape (Ticket 2.5 — was `int` pre-fix):
      {
        "ws_count": int,   # WS frames pushed
        "channel_results": {
          "website": {"status": "delivered", "message_id": ..., "ws_count": ..., "day_chat_id": ...},
          "telegram": {"status": "delivered" | "skipped" | "failed", ...},
          "whatsapp": {...},
        },
      }

    Callers that previously expected an int can use `out["ws_count"]`.

    `delivery_channels` is the list the user picked when they created
    the routine. The 'website' slug is served by this broadcast (the
    chat UI subscribes to the WS event AND/OR fetches via
    `/api/messages/since`); other slugs route through
    `channel_dispatcher.deliver_to_extra_channels_detailed`.
    """
    channel_results: dict[str, dict[str, Any]] = {}

    try:
        from app.api.ws_chat import broadcast_to_user
    except Exception as e:
        logger.debug("[routine_writer] broadcast skipped — ws_chat unavailable: %s", e)
        ws_count = 0
    else:
        event = {
            "type": "message",
            "id": message_id,
            "day_chat_id": day_chat_id,
            "role": "assistant",
            "channel": "routine",
            "source": source,
            "content": content,
            "model_used": model_used,
            # Ticket 2.5 — flag so the chat UI's parser can render the
            # routine card vs a normal assistant bubble.
            "routine_message": True,
            "created_at": datetime.utcnow().isoformat(),
        }
        if extra:
            # Additive card/chip keys (e.g. R29's fix_chip) — the
            # persisted metadata_json is the durable copy; this keeps
            # the live frame and the reload agreeing.
            event.update(extra)
        try:
            ws_count = await broadcast_to_user(user_id, event)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("[routine_writer] broadcast failed: %s", e)
            ws_count = 0

    # Website always counts as delivered when the Message row landed.
    # ws_count == 0 just means the user wasn't viewing the day-chat
    # live; they'll see the row on next /messages/since fetch.
    channel_results["website"] = {
        "status": "delivered",
        "message_id": message_id,
        "day_chat_id": day_chat_id,
        "ws_count": ws_count,
    }

    # Optional fan-out to extra channels. Best-effort by design — the
    # canonical Day-as-Chat record already landed via write_routine_message,
    # so a Telegram outage must not block or fail the routine.
    extras = [c for c in (delivery_channels or []) if c and c != "website"]
    if extras:
        try:
            from .channel_dispatcher import deliver_to_extra_channels_detailed
            from app.db.database import async_session_maker
            extras_results = await deliver_to_extra_channels_detailed(
                user_id=user_id,
                delivery_channels=extras,
                routine_name=routine_name or source,
                content=content,
                db_session_maker=async_session_maker,
            )
            if extras_results:
                channel_results.update(extras_results)
                logger.info(
                    "[routine_writer] extra channel fan-out user=%s results=%s",
                    user_id,
                    # Log a compact summary so the line stays grep-friendly.
                    {k: (v or {}).get("status") for k, v in extras_results.items()},
                )
        except Exception as e:  # pragma: no cover — top-level guard
            logger.warning(
                "[routine_writer] extra channel dispatch failed: %s: %s",
                type(e).__name__, e,
            )

    return {"ws_count": ws_count, "channel_results": channel_results}
