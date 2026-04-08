"""
Day Chats API — list and retrieve day-level conversation containers.

These endpoints work in read-only mode regardless of USE_DAY_CHAT_CONTEXT flag:
return data if backfill has completed, return empty list if not. Never error on
the flag being off. This lets the frontend deploy independently of the flag flip.
"""

import logging
from datetime import date as Date, datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select, and_, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.db.models import User, Conversation, Message
from app.db.models.day_chat import DayChat
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/day-chats", tags=["Day Chats"])


@router.get("")
async def list_day_chats(
    before: Optional[str] = Query(None, description="Cursor: ISO date (YYYY-MM-DD), return days before this"),
    limit: int = Query(30, ge=1, le=90, description="Max day chats to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's day chats, newest first, cursor-paginated.

    Pagination: pass ?before=2026-04-01&limit=30 to get the 30 days before April 1.
    Default: most recent 30 days. Max limit: 90.

    Works in read-only mode regardless of feature flag state. Returns empty list
    if no day_chats exist (backfill hasn't run yet).
    """
    query = (
        select(DayChat)
        .where(DayChat.user_id == current_user.id)
        .order_by(DayChat.local_date.desc())
        .limit(limit)
    )

    if before:
        try:
            before_date = Date.fromisoformat(before)
            query = query.where(DayChat.local_date < before_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'before' date. Use YYYY-MM-DD.")

    result = await db.execute(query)
    day_chats = result.scalars().all()

    # TODO: collapse to single GROUP BY query — currently N+1 (one channel query
    # per day chat). For 90 days that's 91 queries. Fix when telemetry shows it matters.
    items = []
    for dc in day_chats:
        # Get distinct channels for this day chat
        ch_result = await db.execute(
            select(distinct(Conversation.channel))
            .where(Conversation.day_chat_id == dc.id)
        )
        channels = sorted([r[0] for r in ch_result.all() if r[0]])

        items.append({
            "id": dc.id,
            "local_date": dc.local_date.isoformat(),
            "message_count": dc.message_count or 0,
            "channels_active": channels,
            "last_message_at": dc.last_message_at.isoformat() if dc.last_message_at else None,
            "summary_status": dc.summary_status or "up_to_date",
        })

    # TODO: add next_cursor field so frontend doesn't have to derive it from
    # the last item's local_date. Works but ugly.
    return JSONResponse(content=items)


@router.get("/{date_str}/messages")
async def get_day_chat_messages(
    date_str: str,
    limit: int = Query(500, ge=1, le=2000),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all messages for a given local date, across all channels.

    Messages are in strict chronological order. Each message includes its
    channel and conversation_id so the frontend can show session dividers
    and channel badges.

    Content is RAW — no [channel time] annotations. Annotations are LLM-only
    and never exposed to the frontend or persisted.

    Returns 404 if no day chat exists for this date.
    Returns empty array if the day chat exists but has no messages.

    Works in read-only mode regardless of feature flag state.
    """
    try:
        target_date = Date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Look up day chat by (user_id, local_date)
    dc = (await db.execute(
        select(DayChat).where(
            and_(DayChat.user_id == current_user.id, DayChat.local_date == target_date)
        )
    )).scalar_one_or_none()

    if not dc:
        # Fall back to date-range scan (for when backfill hasn't run yet)
        # This mirrors the existing /sessions/by-date/{date}/messages logic
        from datetime import timezone as tz
        user_tz_name = current_user.timezone if hasattr(current_user, 'timezone') else None
        if user_tz_name:
            try:
                import zoneinfo
                user_tz = zoneinfo.ZoneInfo(user_tz_name)
                # Midnight in user's timezone, converted to UTC
                local_midnight = datetime(target_date.year, target_date.month, target_date.day, tzinfo=user_tz)
                day_start = local_midnight.astimezone(tz.utc).replace(tzinfo=None)
            except (KeyError, Exception):
                day_start = datetime(target_date.year, target_date.month, target_date.day)
        else:
            day_start = datetime(target_date.year, target_date.month, target_date.day)

        day_end = day_start + timedelta(days=1)

        sessions_result = await db.execute(
            select(Conversation.id, Conversation.channel)
            .where(
                and_(
                    Conversation.user_id == current_user.id,
                    Conversation.started_at >= day_start,
                    Conversation.started_at < day_end,
                )
            )
        )
        session_rows = sessions_result.all()
        if not session_rows:
            raise HTTPException(status_code=404, detail=f"No conversations found for {date_str}")

        session_ids = [r[0] for r in session_rows]
        channel_map = {r[0]: r[1] for r in session_rows}

        msgs_result = await db.execute(
            select(Message)
            .where(Message.conversation_id.in_(session_ids))
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        messages = msgs_result.scalars().all()

        return JSONResponse(content=[
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "channel": channel_map.get(m.conversation_id, "web"),
                "conversation_id": m.conversation_id,
            }
            for m in messages
        ])

    # Day chat exists — load messages via day_chat_id (fast path)
    msgs_result = await db.execute(
        select(Message, Conversation.channel)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.day_chat_id == dc.id)
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    rows = msgs_result.all()

    return JSONResponse(content=[
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
            "channel": channel or "web",
            "conversation_id": msg.conversation_id,
        }
        for msg, channel in rows
    ])
