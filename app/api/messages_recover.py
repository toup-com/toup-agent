"""
Messages recovery endpoint (Pre-A — never-sleep plan).

Backstop for chat WS reconnects: when the browser drops mid-stream and
reopens the WebSocket, the assistant's response — which the agent
already finished generating and persisted — would otherwise be lost
(the streamed frames went into a dead WS). The frontend calls
`GET /api/messages/since/<message_id>` on every WS reopen, gets back
any messages whose `created_at >=` the seed message's created_at, and
merges them into local state (dedupe by id).

Mounted on BOTH platform and agent. Platform-mode proxies to the
user's VPS agent (same pattern as day_chats.py); agent-mode reads
the local agent DB.

Shape matches the existing `/api/day-chats/{date}/messages` endpoint
exactly (see `day_chats.py`) so the frontend's merge can dedupe by id
without per-source adapters. Capped at 500 to bound the payload — a
genuine reconnect after >500 missed messages is pathological and the
chat should reload fully via the day-chat fetch instead.
"""

import logging
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select, and_
from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.db.models import User, Conversation, Message
from app.api.auth import get_current_user
from app.api.day_chats import _serialize_attachments, _serialize_media

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/messages", tags=["messages"])

# Same swallow-set day_chats uses: when this router is mounted on
# platform_main, the platform DB has no `messages`/`conversations`
# tables — return empty rather than 500'ing devtools red.
_MISSING_TABLE_ERRORS: tuple = (ProgrammingError, OperationalError)


async def _get_agent_proxy_info(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
    """Return (agent_url, agent_api_key) if the user has a remote agent.

    Mirrors `day_chats._get_agent_proxy_info` but local copy avoids a
    cross-import dependency (platform mounts both routers, the import
    chain matters for mount-order in agent mode where this file's
    `from app.api.day_chats import` would resolve fine but pulling in
    its proxy helper is cleaner kept local).
    """
    try:
        from app.db.models import AgentConfig
        async with db.begin_nested():
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
                    AgentConfig.user_id == user_id, AgentConfig.deploy_status == "active"
                )
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass
    return None


async def _proxy_messages_since(
    agent_url: str, agent_api_key: str, message_id: str, limit: int
):
    """Proxy a messages-since request to the VPS agent."""
    url = f"{agent_url}/api/messages/since/{message_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                headers={"X-Agent-Key": agent_api_key},
                params={"limit": limit},
            )
            if resp.status_code == 200:
                return resp.json()
            # 404 propagates as None → caller returns 404 below; we
            # don't pretend the message exists by returning [].
            if resp.status_code == 404:
                return "404"
    except Exception as e:
        logger.warning("messages_recover proxy %s failed: %s", url, e)
    return None


@router.get("/since/{message_id}")
async def messages_since(
    message_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return messages for the current user with `created_at >=` the
    seed message's `created_at`, EXCLUSIVE of the seed itself. Strict
    chronological order. Caller-owned messages only.

    Platform mode: proxies to user's VPS agent.
    Agent mode: queries local DB.

    The seed-message ownership check is mandatory: without it, a leaked
    message id from any other tenant would let the caller list their own
    feed but seeded by an attacker-supplied timestamp. We require the
    seed to belong to the caller; if it doesn't, return 404 with the
    same shape a non-existent id would produce — no oracle.
    """
    # Platform mode: proxy
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_messages_since(proxy[0], proxy[1], message_id, limit)
        if data == "404":
            raise HTTPException(status_code=404, detail="Message not found")
        if data is not None:
            return JSONResponse(content=data)
        # Proxy unreachable — fall through. In platform mode the local
        # tables don't exist, so the SELECT below will trip the
        # missing-table guard and return []. That's a degraded but
        # safe response: chat keeps showing what it had, no crash.

    # Agent mode (or platform fallback): local query.
    try:
        seed = (
            await db.execute(
                select(Message, Conversation.user_id)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(Message.id == message_id)
            )
        ).first()
    except _MISSING_TABLE_ERRORS as exc:
        await db.rollback()
        logger.info(
            "[messages_recover] seed lookup skipped (tables absent): %s",
            str(exc)[:120],
        )
        return JSONResponse(content=[])

    if not seed:
        raise HTTPException(status_code=404, detail="Message not found")

    seed_msg, seed_owner = seed
    if seed_owner != current_user.id:
        raise HTTPException(status_code=404, detail="Message not found")

    seed_created_at = seed_msg.created_at
    if seed_created_at is None:
        return JSONResponse(content=[])

    try:
        rows = (
            await db.execute(
                select(Message, Conversation.channel)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(
                    and_(
                        Conversation.user_id == current_user.id,
                        Message.created_at >= seed_created_at,
                        Message.id != message_id,
                    )
                )
                .order_by(Message.created_at.asc(), Message.id.asc())
                .limit(limit)
            )
        ).all()
    except _MISSING_TABLE_ERRORS as exc:
        await db.rollback()
        logger.info(
            "[messages_recover] tail lookup skipped (tables absent): %s",
            str(exc)[:120],
        )
        return JSONResponse(content=[])

    # Bulk-resolve reply targets so each replying row can render its
    # quoted card on first paint — see api/day_chats.py for rationale.
    from app.agent.reply_quote import resolve_reply_targets_for_serialization
    reply_targets = await resolve_reply_targets_for_serialization(
        db, [msg for msg, _ in rows]
    )

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
            "reply_to_message_id": getattr(msg, "reply_to_message_id", None),
            "reply_to": reply_targets.get(msg.id),
        }
        for msg, channel in rows
    ])
