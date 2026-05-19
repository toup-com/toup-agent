"""
Day Chats API — list and retrieve day-level conversation containers.

These endpoints work in read-only mode regardless of USE_DAY_CHAT_CONTEXT flag:
return data if backfill has completed, return empty list if not. Never error on
the flag being off. This lets the frontend deploy independently of the flag flip.
"""

import logging
from datetime import date as Date, datetime, timedelta
from typing import Optional, List, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select, and_, func, distinct, update
from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.db.models import User, Conversation, Message
from app.db.models.day_chat import DayChat
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/day-chats", tags=["Day Chats"])

# When this router is mounted in `platform_main`, the platform DB
# does NOT have the agent-only tables (`day_chats`, `conversations`,
# `messages`) — they live in each tenant's agent DB. The endpoints
# below proxy to the user's agent when one exists; when it doesn't,
# the SELECT fallback would crash with `UndefinedTableError`. The
# helper `_safe_local` swallows that specific schema-shape error and
# yields an empty result, so a brand-new platform user (no agent
# provisioned yet) gets a quiet 200 [] instead of a red 500 in their
# devtools console. Caught 2026-05-06 in the post-Install live retest.
_MISSING_TABLE_ERRORS: tuple = (ProgrammingError, OperationalError)


_PREVIEW_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _attachment_urls(message_id: str, att: dict) -> dict:
    """Compute download_url and (when applicable) preview_url for a stored
    attachment. Mirrors the live WS event payload built by ws_chat.py so
    REST-loaded history renders identically to live messages."""
    from app.config import settings as _settings
    aid = att.get("id", "")
    mime = att.get("mime_type", "")
    out = {"download_url": f"{_settings.api_prefix}/files/{message_id}/{aid}"}
    if mime in _PREVIEW_MIMES or mime.startswith("image/"):
        out["preview_url"] = f"{_settings.api_prefix}/files/{message_id}/{aid}/preview?format=html"
    return out


def _serialize_attachments(msg: Message) -> Optional[List[dict]]:
    """Return Message.attachments as a client-safe list (strip storage_path,
    enrich with download_url + preview_url) or None when there are none.
    Handles both native-JSON driver returns (list) and legacy TEXT-stored
    JSON strings."""
    import json as _json
    raw = getattr(msg, "attachments", None)
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, list):
        return None
    return [
        {
            **{k: v for k, v in att.items() if k != "storage_path"},
            **_attachment_urls(msg.id, att),
        }
        for att in raw
        if isinstance(att, dict)
    ]


def _serialize_media(msg: Message) -> Optional[dict]:
    """Extract the media payload persisted in metadata_json by agent_runner.
    Shape: {"type": "youtube"|"netflix", "video_id": "...", "title": "..."}.
    Returns None when absent or malformed."""
    import json as _json
    raw = getattr(msg, "metadata_json", None)
    if not raw:
        return None
    try:
        parsed = _json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    media = parsed.get("media")
    return media if isinstance(media, dict) else None


def _serialize_tool_events(msg: Message) -> Optional[List[dict]]:
    """Extract the ToolPillRow records persisted in metadata_json by
    agent_runner. Shape per record: {tool, started_at_ms,
    completed_at_ms, summary}. Returns None when absent or malformed
    so the frontend ToolPillRow component skips rendering entirely
    for legacy (pre-feature) messages instead of showing an empty
    pill row."""
    import json as _json
    raw = getattr(msg, "metadata_json", None)
    if not raw:
        return None
    try:
        parsed = _json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    events = parsed.get("tool_events")
    if not isinstance(events, list) or not events:
        return None
    # Defensive: drop any record missing the required keys instead of
    # poisoning the whole list. The agent always writes well-formed
    # records but a hand-edited row in production should degrade
    # gracefully, not 500 every history load.
    return [
        e for e in events
        if isinstance(e, dict) and "tool" in e and "started_at_ms" in e
    ] or None


# ── Agent proxy (platform mode proxies to user's VPS agent) ──────────

async def _get_agent_proxy_info(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
    """Return (agent_url, agent_api_key) if the user has a remote agent."""
    try:
        from app.db.models import AgentConfig
        async with db.begin_nested():
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(AgentConfig.user_id == user_id, AgentConfig.deploy_status == "active")
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass
    return None


async def _proxy_day_chats(agent_url: str, agent_api_key: str, path: str = "", params: dict = None):
    """Proxy a day-chats request to the VPS agent."""
    url = f"{agent_url}/api/day-chats/{path}" if path else f"{agent_url}/api/day-chats"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers={"X-Agent-Key": agent_api_key}, params=params or {})
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.warning("Day-chats proxy %s failed: %s", url, e)
    return None


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
    # Platform mode: proxy to user's VPS agent
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        params = {"limit": limit}
        if before:
            params["before"] = before
        data = await _proxy_day_chats(proxy[0], proxy[1], "", params)
        if data is not None:
            return JSONResponse(content=data)

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

    try:
        result = await db.execute(query)
        day_chats = result.scalars().all()
    except _MISSING_TABLE_ERRORS as exc:
        # Platform DB has no `day_chats` (AGENT_ONLY table). User
        # without an active proxy → no day chats yet. Return empty
        # so the chat shell can render its empty state instead of
        # surfacing a 500 in devtools.
        await db.rollback()
        logger.info("[day_chats] local SELECT skipped (table absent in this DB): %s", str(exc)[:120])
        return JSONResponse(content=[])

    # ── Self-healing: fix DayChats with future local_date ──
    # This can happen when a user's timezone was NULL (defaulting to UTC) and
    # messages were sent after midnight UTC but before midnight local. The DayChat
    # got created with tomorrow's UTC date instead of today's local date.
    # Fix: re-resolve the date using the user's current timezone and merge.
    user_tz = getattr(current_user, 'timezone', None)
    if user_tz:
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(user_tz)
            local_today = datetime.now(tz).date()
            _healed = False
            for dc in list(day_chats):  # copy list — we may delete entries
                if dc.local_date > local_today:
                    _healed = True
                    logger.warning(
                        "[day_chats] Future-dated DayChat detected: id=%s local_date=%s > today=%s (tz=%s). Rebucketing.",
                        dc.id[:8], dc.local_date, local_today, user_tz,
                    )
                    # Check if a DayChat for today already exists
                    existing_today = (await db.execute(
                        select(DayChat).where(
                            and_(DayChat.user_id == current_user.id, DayChat.local_date == local_today)
                        )
                    )).scalar_one_or_none()

                    if existing_today:
                        # Merge: move all messages from the future DayChat to today's
                        await db.execute(
                            update(Message)
                            .where(Message.day_chat_id == dc.id)
                            .values(day_chat_id=existing_today.id)
                        )
                        # Update conversations too
                        await db.execute(
                            update(Conversation)
                            .where(Conversation.day_chat_id == dc.id)
                            .values(day_chat_id=existing_today.id)
                        )
                        # Delete the orphaned future DayChat
                        await db.delete(dc)
                        await db.commit()
                        logger.info("[day_chats] Merged future DayChat %s into %s (today)", dc.id[:8], existing_today.id[:8])
                    else:
                        # No DayChat for today — just fix the date
                        dc.local_date = local_today
                        dc.timezone = user_tz
                        await db.commit()
                        logger.info("[day_chats] Fixed future DayChat %s: %s → %s", dc.id[:8], dc.local_date, local_today)

            if _healed:
                # Re-query after healing to get correct data
                result = await db.execute(
                    select(DayChat)
                    .where(DayChat.user_id == current_user.id)
                    .order_by(DayChat.local_date.desc())
                    .limit(limit)
                )
                day_chats = result.scalars().all()
        except Exception as _heal_err:
            logger.warning("[day_chats] Self-healing failed: %s", _heal_err)

    # TODO: collapse to single GROUP BY query — currently N+1 (one channel query
    # per day chat). For 90 days that's 91 queries. Fix when telemetry shows it matters.
    items = []
    for dc in day_chats:
        # Get distinct channels from MESSAGES (not Conversations) — because Telegram
        # sessions are long-lived and their Conversation.day_chat_id may point to a
        # different day. Message.day_chat_id is canonical for day membership.
        ch_result = await db.execute(
            select(distinct(Conversation.channel))
            .select_from(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Message.day_chat_id == dc.id)
        )
        channels = sorted([r[0] for r in ch_result.all() if r[0]])

        # Count from messages table (canonical), not from DayChat.message_count (cached, can drift)
        msg_count_result = await db.execute(
            select(func.count()).select_from(Message).where(Message.day_chat_id == dc.id)
        )
        msg_count = msg_count_result.scalar() or 0

        items.append({
            "id": dc.id,
            "local_date": dc.local_date.isoformat(),
            "message_count": msg_count,
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
    # Platform mode: proxy to user's VPS agent
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_day_chats(proxy[0], proxy[1], f"{date_str}/messages", {"limit": limit})
        if data is not None:
            return JSONResponse(content=data)

    try:
        target_date = Date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Look up day chat by (user_id, local_date). When this router is
    # mounted in platform_main, the platform DB lacks the agent-only
    # `day_chats` table — return empty rather than 500'ing the client.
    try:
        dc = (await db.execute(
            select(DayChat).where(
                and_(DayChat.user_id == current_user.id, DayChat.local_date == target_date)
            )
        )).scalar_one_or_none()
    except _MISSING_TABLE_ERRORS as exc:
        await db.rollback()
        logger.info(
            "[day_chats] messages SELECT skipped (table absent in this DB): %s",
            str(exc)[:120],
        )
        return JSONResponse(content=[])

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

        try:
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
        except _MISSING_TABLE_ERRORS as exc:
            await db.rollback()
            logger.info(
                "[day_chats] messages fallback skipped (conversations table absent): %s",
                str(exc)[:120],
            )
            return JSONResponse(content=[])
        session_rows = sessions_result.all()
        if not session_rows:
            # No conversations for this day. Return empty list rather
            # than 404 — the chat shell uses these endpoints to render
            # past-day dividers; an empty day is a valid state, not a
            # client error. (Pre-fix: 404 surfaced as a red devtools
            # entry on every day-chat poll for new users.)
            return JSONResponse(content=[])

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
                "attachments": _serialize_attachments(m),
                "media": _serialize_media(m),
                "tool_events": _serialize_tool_events(m),
                "reply_to_message_id": getattr(m, "reply_to_message_id", None),
            }
            for m in messages
        ])

    # Day chat exists — load messages via day_chat_id (fast path).
    # Wrapped: if `day_chats` lives on platform DB (legacy artifact) but
    # `messages`/`conversations` don't, this query would 500. Guard
    # mirrors the missing-table handling above.
    try:
        msgs_result = await db.execute(
            select(Message, Conversation.channel)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Message.day_chat_id == dc.id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        rows = msgs_result.all()
    except _MISSING_TABLE_ERRORS:
        await db.rollback()
        return JSONResponse(content=[])

    return JSONResponse(content=[
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
            "channel": channel or "web",
            "conversation_id": msg.conversation_id,
            "attachments": _serialize_attachments(msg),
            "media": _serialize_media(msg),
            "tool_events": _serialize_tool_events(msg),
            "reply_to_message_id": getattr(msg, "reply_to_message_id", None),
        }
        for msg, channel in rows
    ])


@router.get("/app-conversation/{app_id}")
async def resolve_app_conversation(
    app_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Resolve the existing Conversation for an app in today's day chat.

    Returns the conversation_id and message count if found, 404 if no
    conversation exists yet (user hasn't chatted with this app today).

    Used by the frontend to restore app conversation state on re-open.
    """
    import json as _json

    # Platform mode: proxy to user's VPS agent
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_day_chats(proxy[0], proxy[1], f"app-conversation/{app_id}")
        if data is not None:
            return JSONResponse(content=data)

    # Find today's day chat
    user_tz = getattr(current_user, 'timezone', None)
    if user_tz:
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(user_tz)
            local_today = datetime.now(tz).date()
        except (KeyError, Exception):
            local_today = datetime.utcnow().date()
    else:
        local_today = datetime.utcnow().date()

    dc = (await db.execute(
        select(DayChat).where(
            and_(DayChat.user_id == current_user.id, DayChat.local_date == local_today)
        )
    )).scalar_one_or_none()

    if not dc:
        raise HTTPException(status_code=404, detail="No day chat for today")

    # Find app conversation in today's day chat
    candidates = (await db.execute(
        select(Conversation).where(and_(
            Conversation.user_id == current_user.id,
            Conversation.day_chat_id == dc.id,
            Conversation.channel == "app",
        ))
    )).scalars().all()

    for conv in candidates:
        try:
            meta = _json.loads(conv.metadata_json or "{}")
            if meta.get("app_id") == app_id:
                msg_count = (await db.execute(
                    select(func.count()).select_from(Message)
                    .where(Message.conversation_id == conv.id)
                )).scalar() or 0
                return JSONResponse(content={
                    "conversation_id": conv.id,
                    "message_count": msg_count,
                })
        except (ValueError, TypeError):
            continue

    raise HTTPException(status_code=404, detail="No app conversation found for today")
