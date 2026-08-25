"""
Sessions API - Conversation session management

Sessions track conversation history with the agent.
Each session maintains:
- Message history (user + assistant messages)
- Token usage statistics
- Channel information (api, telegram, discord, web)
- Metadata for context
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy.orm import selectinload

# Mirrors day_chats.py — `conversations`/`messages` are AGENT_ONLY_TABLES,
# so any local fall-back query on the platform DB raises UndefinedTable.
# Without this guard, a failed agent proxy (401, timeout, agent down) cascades
# into a 500 instead of an empty-but-valid response, breaking the chat shell's
# multi-day load.
_MISSING_TABLE_ERRORS: tuple = (ProgrammingError, OperationalError)
from typing import Optional, Tuple
import json
import httpx

from app.db import get_db, Conversation, Message, User, AgentConfig
from app.schemas import (
    SessionCreate, SessionResponse, SessionWithMessages, SessionListResponse,
    ChatMessageResponse, SessionMessageCreate
)
from app.api.auth import get_current_user
from app.api.message_cards import (
    attach_run_to_cards,
    job_card_fields,
    load_build_jobs,
    public_text,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


# ── Agent proxy helpers (same pattern as stats.py) ────────────────────

async def _get_agent_proxy_info(
    user_id: str, db: AsyncSession
) -> Optional[Tuple[str, str]]:
    """Return (agent_url, agent_api_key) if the user has a remote agent."""
    try:
        async with db.begin_nested():
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(
                    AgentConfig.user_id == user_id,
                    AgentConfig.deploy_status == "active",
                )
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass  # agent_configs table may not exist — savepoint rolled back
    return None


async def _proxy_sessions(
    agent_url: str, agent_api_key: str, path: str, params: Optional[dict] = None
):
    """Proxy a sessions request to the VPS agent.

    TKT-LAT-007: uses the shared agent_http client.
    """
    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url}/api/sessions/{path}" if path else f"{agent_url}/api/sessions"
    try:
        client = get_agent_http_client()
        resp = await client.get(
            url,
            headers={"X-Agent-Key": agent_api_key},
            params=params or {},
            timeout=10.0,
        )
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Agent sessions proxy %s returned %s", url, resp.status_code)
    except Exception as e:
        logger.warning("Agent sessions proxy %s failed: %s", url, e)
    return None


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new conversation session.
    
    Sessions are containers for conversations with the agent.
    You can optionally provide a title and channel.
    """
    # Resolve day_chat_id so voice/API-created sessions join the Day-as-Chat
    _day_chat_id = None
    try:
        from app.agent.day_chat_resolver import get_or_create_day_chat
        _user_tz = getattr(current_user, 'timezone', None)
        _dc = await get_or_create_day_chat(db, current_user.id, tz_name=_user_tz)
        _day_chat_id = _dc.id
    except Exception:
        pass

    # SessionCreate.channel DEFAULTS to "api" — a channel governed by the
    # partial unique index ix_conversations_system_channel_per_day. With a
    # real day_chat_id stamped, a blind insert 500s on the user's 2nd
    # default-channel create of the same local day (same bug class the
    # runner hit, 543739ab). Route system channels through the canonical
    # resolver: it reuses the day's existing thread and recovers from the
    # insert race. Reuse still returns 201 with the existing row.
    from app.agent.conversation_resolver import (
        SYSTEM_CHANNELS,
        resolve_or_create_day_conversation,
    )
    if request.channel in SYSTEM_CHANNELS and _day_chat_id is not None:
        session = await resolve_or_create_day_conversation(
            db,
            user_id=current_user.id,
            day_chat_id=_day_chat_id,
            channel=request.channel,
            title=request.title,
            metadata=request.metadata,
        )
        await db.commit()
        await db.refresh(session)
        return _session_to_response(session)

    # Create session (Conversation model)
    session = Conversation(
        user_id=current_user.id,
        title=request.title,
        channel=request.channel,
        metadata_json=json.dumps(request.metadata) if request.metadata else None,
        is_active=True,
        day_chat_id=_day_chat_id,
    )

    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    return _session_to_response(session)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    channel: Optional[str] = None,
    active_only: bool = False,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List conversation sessions for the current user.

    Optionally filter by channel or active status.
    Sessions are ordered by most recent first.
    """
    # Try proxying to remote agent first
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        params = {"limit": limit, "offset": offset}
        if channel:
            params["channel"] = channel
        if active_only:
            params["active_only"] = "true"
        data = await _proxy_sessions(proxy[0], proxy[1], "", params)
        if data is not None:
            # Merge platform-local sessions (voice + browser) into proxy response
            # These channels run on the platform, not the VPS agent, so their
            # sessions are stored in the platform DB and must be merged manually.
            # Only merge on the first page to avoid duplicates across pagination.
            local_channels = {"voice", "web"}
            if offset == 0 and (channel is None or channel in local_channels):
                try:
                    merge_channels = [channel] if channel else list(local_channels)
                    local_conditions = [
                        Conversation.user_id == current_user.id,
                        Conversation.channel.in_(merge_channels),
                    ]
                    if active_only:
                        local_conditions.append(Conversation.is_active == True)
                    local_result = await db.execute(
                        select(Conversation)
                        .where(and_(*local_conditions))
                        .order_by(Conversation.updated_at.desc())
                        .limit(50)
                    )
                    local_sessions = local_result.scalars().all()
                    if local_sessions:
                        local_list = [_session_to_response(s).model_dump(mode="json") for s in local_sessions]
                        # Merge into proxy response
                        proxy_sessions = data.get("sessions", data) if isinstance(data, dict) else data
                        if isinstance(proxy_sessions, list):
                            existing_ids = {s.get("id") for s in proxy_sessions}
                            for ls in local_list:
                                if ls["id"] not in existing_ids:
                                    proxy_sessions.append(ls)
                            proxy_sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
                        elif isinstance(data, dict) and "sessions" in data:
                            existing_ids = {s.get("id") for s in data["sessions"]}
                            for ls in local_list:
                                if ls["id"] not in existing_ids:
                                    data["sessions"].append(ls)
                            data["sessions"].sort(key=lambda s: s.get("updated_at", ""), reverse=True)
                            data["total_count"] = len(data["sessions"])
                except Exception as e:
                    logger.warning("Failed to merge local sessions: %s", e)
            return JSONResponse(content=data)

    # Build query
    conditions = [Conversation.user_id == current_user.id]
    
    if channel:
        conditions.append(Conversation.channel == channel)
    
    if active_only:
        conditions.append(Conversation.is_active == True)
    
    # Count total
    count_query = select(func.count(Conversation.id)).where(and_(*conditions))
    total_result = await db.execute(count_query)
    total_count = total_result.scalar()
    
    # Get sessions
    query = (
        select(Conversation)
        .where(and_(*conditions))
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    
    result = await db.execute(query)
    sessions = result.scalars().all()
    
    return SessionListResponse(
        sessions=[_session_to_response(s) for s in sessions],
        total_count=total_count
    )


@router.get("/{session_id}", response_model=SessionWithMessages)
async def get_session(
    session_id: str,
    include_messages: bool = True,
    message_limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific session with its message history.

    Messages are ordered chronologically (oldest first).
    """
    # Check if this session exists locally first (voice sessions are in platform DB)
    local_check = await db.execute(
        select(Conversation.id).where(
            and_(
                Conversation.id == session_id,
                Conversation.user_id == current_user.id,
            )
        )
    )
    is_local = local_check.scalar_one_or_none() is not None

    if not is_local:
        # Try proxying to remote agent
        proxy = await _get_agent_proxy_info(current_user.id, db)
        if proxy:
            params = {"include_messages": str(include_messages).lower(), "message_limit": message_limit}
            data = await _proxy_sessions(proxy[0], proxy[1], session_id, params)
            if data is not None:
                return JSONResponse(content=data)

    # Build query with optional message loading
    query = select(Conversation).where(
        and_(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id
        )
    )
    
    if include_messages:
        query = query.options(selectinload(Conversation.messages))
    
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Convert to response
    response_dict = _session_to_response(session).model_dump()
    
    # Add messages if included
    if include_messages and session.messages:
        messages = session.messages[:message_limit]
        # Look up BuildJob status for job card messages
        build_jobs = await load_build_jobs(db, messages)
        from app.agent.reply_quote import resolve_reply_targets_for_serialization
        reply_targets = await resolve_reply_targets_for_serialization(db, messages)
        _channels = {session.id: session.channel}
        response_dict["messages"] = attach_run_to_cards([
            _message_to_response(m, build_jobs, reply_targets, _channels) for m in messages
        ])
    else:
        response_dict["messages"] = []
    
    return SessionWithMessages(**response_dict)


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update session title or metadata."""
    query = select(Conversation).where(
        and_(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id
        )
    )
    
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if title is not None:
        session.title = title
    
    if metadata is not None:
        session.metadata_json = json.dumps(metadata)
    
    await db.commit()
    await db.refresh(session)
    
    return _session_to_response(session)


@router.post("/{session_id}/end", response_model=SessionResponse)
async def end_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    End a conversation session.
    
    Sets is_active=False and records end timestamp.
    """
    query = select(Conversation).where(
        and_(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id
        )
    )
    
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    session.is_active = False
    session.ended_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(session)
    
    return _session_to_response(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a session and all its messages.
    
    This is permanent and cannot be undone.
    """
    query = select(Conversation).where(
        and_(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id
        )
    )
    
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    await db.delete(session)
    await db.commit()


@router.get("/{session_id}/messages", response_model=list[ChatMessageResponse])
async def get_session_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get messages from a session with pagination.

    Messages are ordered chronologically (oldest first).
    """
    # Check if this session exists in local platform DB first (voice sessions)
    local_check = await db.execute(
        select(Conversation.id).where(
            and_(
                Conversation.id == session_id,
                Conversation.user_id == current_user.id,
            )
        )
    )
    is_local = local_check.scalar_one_or_none() is not None

    if not is_local:
        # Try proxying to remote agent
        proxy = await _get_agent_proxy_info(current_user.id, db)
        if proxy:
            params = {"limit": limit, "offset": offset}
            data = await _proxy_sessions(proxy[0], proxy[1], f"{session_id}/messages", params)
            if data is not None:
                return JSONResponse(content=data)

    # Verify session ownership
    session_query = select(Conversation.id, Conversation.channel).where(
        and_(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id
        )
    )
    session_result = await db.execute(session_query)
    session_row = session_result.first()
    if not session_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    conversation_channels = {session_row[0]: session_row[1]}

    # Get messages
    query = (
        select(Message)
        .where(Message.conversation_id == session_id)
        .order_by(Message.created_at.asc())
        .offset(offset)
        .limit(limit)
    )

    result = await db.execute(query)
    messages = result.scalars().all()

    # Look up BuildJob status for any job card messages
    build_jobs = await load_build_jobs(db, messages)

    # Bulk-resolve reply targets so each row's reply_to card paints
    # immediately instead of flashing the "(message not in current view)"
    # stub. See api/day_chats.py for the same pattern.
    from app.agent.reply_quote import resolve_reply_targets_for_serialization
    reply_targets = await resolve_reply_targets_for_serialization(db, messages)

    return attach_run_to_cards([
        _message_to_response(m, build_jobs, reply_targets, conversation_channels)
        for m in messages
    ])


# ── Persisted tool records off an untrusted body ──────────────────────────
# This route is reached with the user's own credentials from the platform
# relay, so the body is not hostile — but it is not the agent runner either,
# and `metadata_json` is read back by every client. Keep only the keys
# `day_chats._serialize_tool_events` and the clients actually read, bound the
# list, and require the two fields that function requires — a record missing
# them is dropped there anyway, so admitting it here only stores garbage.
_TOOL_EVENT_KEYS = {
    "tool", "started_at_ms", "completed_at_ms", "summary",
    "step_index", "url", "domains", "urls", "sources", "ok",
    # Round 13: the rest of the step attribution a CHAT record already
    # carries (agent_runner stamps StepTracker.event_fields onto every
    # record). `step_index` alone is an index into a step list the reader
    # has no other way to find — without the job id and the totals beside
    # it, a voice record could not be bucketed the way a chat one is.
    "job_id", "step_name", "steps_total", "job_type", "call_id",
    # Round 18: the human status line the backend chose for this tool, and
    # the app a `present_app` record handed over. Both are written by
    # `agent_runner` onto a chat record; an allowlist that omits them makes
    # a record ingested through this route render differently from the same
    # work done in chat — a row labelled "create app file" beside one
    # labelled "Building your app".
    # …and, for the same reason, WHICH BUILD produced it: the build card in a
    # reopened thread is drawn from this id, and dropping it here would make an
    # ingested record render as the generic "N actions" rail.
    "label", "app_slug", "job_id",
}
_TOOL_EVENTS_MAX = 40


def _clean_tool_events(events) -> Optional[list]:
    if not isinstance(events, list) or not events:
        return None
    out = []
    for e in events[:_TOOL_EVENTS_MAX]:
        if not isinstance(e, dict) or "tool" not in e or "started_at_ms" not in e:
            continue
        out.append({k: v for k, v in e.items() if k in _TOOL_EVENT_KEYS})
    return out or None


def _build_metadata(media, tool_events) -> Optional[str]:
    """The message's metadata_json, in AgentRunner._save_messages' shape.

    One writer for both keys: they used to be mutually exclusive here (media
    replaced the whole blob), so a voice turn that played a song AND used a
    tool could only ever persist one of them.
    """
    meta = {}
    if media:
        meta["media"] = media
    if tool_events:
        meta["tool_events"] = tool_events
    return json.dumps(meta) if meta else None


@router.post("/{session_id}/messages", response_model=ChatMessageResponse, status_code=status.HTTP_201_CREATED)
async def create_session_message(
    session_id: str,
    body: Optional[SessionMessageCreate] = None,
    role: str = "user",
    content: str = "",
    model_used: Optional[str] = None,
    day_chat_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a message to a session.

    Used by the platform voice route to persist voice messages
    on the user's VPS database (not platform DB).

    Accepts the fields either as a JSON body or as query parameters. The
    query form is what shipped originally; the body was added because voice
    transcripts are too long (and too often non-Latin) to survive a URL. Body
    wins when both are present.
    """
    import uuid as _uuid

    media = None
    tool_events = None
    if body is not None:
        role = body.role or role
        content = body.content or content
        model_used = body.model_used or model_used
        day_chat_id = body.day_chat_id or day_chat_id
        media = body.media or None
        tool_events = _clean_tool_events(body.tool_events)

    # Verify session ownership
    session_query = select(Conversation).where(
        and_(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id,
        )
    )
    result = await db.execute(session_query)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content is required"
        )

    # Resolve day_chat_id from the CURRENT TIME first (mirrors
    # AgentRunner._save_messages): session.day_chat_id points at the day
    # the session was CREATED, so preferring it filed post-local-midnight
    # voice messages into yesterday's day chat until UTC midnight. An
    # explicit day_chat_id param still wins (caller intent); the session's
    # stamp is only the degraded-path fallback.
    _day_chat_id = day_chat_id
    if not _day_chat_id:
        from app.db.message_helpers import resolve_day_chat_id_for_now
        _day_chat_id = await resolve_day_chat_id_for_now(
            db, current_user.id,
            tz_override=getattr(current_user, 'timezone', None),
        )
    if not _day_chat_id:
        _day_chat_id = session.day_chat_id
    # Backfill the session's day_chat_id if it was missing
    if _day_chat_id and not session.day_chat_id:
        session.day_chat_id = _day_chat_id

    msg = Message(
        id=str(_uuid.uuid4()),
        conversation_id=session_id,
        role=role,
        content=content.replace("\x00", ""),
        model_used=model_used,
        day_chat_id=_day_chat_id,
        # Denormalized per-message channel (was left NULL on this path,
        # so voice rows rendered channel-less in day history).
        channel=session.channel,
        # Same shape AgentRunner._save_messages writes, so the clients need no
        # new rendering path: a voice-started song gets the same Toup card a
        # chat-started one does, and a voice RUN gets the same steps, actions
        # and sources a typed run does (`day_chats._serialize_tool_events`
        # reads both through one function).
        metadata_json=_build_metadata(media, tool_events),
    )
    db.add(msg)

    session.message_count = (session.message_count or 0) + 1
    session.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(msg)

    # Hand the row to any open chat socket live, WITH its media. Voice rows
    # used to reach the phone only on the next resync (as plain text until
    # then), so a song the agent genuinely started rendered as a bare
    # transcript line while it played. Fire-and-forget — live delivery must
    # never be on the persist path.
    try:
        from app.api.ws_chat import broadcast_to_user
        _frame = {
            "type": "message",
            "id": msg.id,
            "role": msg.role,
            # LIVE delivery gets the same guard the history readers get.
            # This body arrives from the platform relay rather than the
            # agent runner, and a frame is the one message surface that
            # bypasses every serializer — leaving it raw would have kept
            # exactly one door open on the way out (see message_cards.py).
            "content": public_text(msg.role, msg.content),
            "created_at": (msg.created_at.isoformat() + "Z")
            if getattr(msg, "created_at", None) else datetime.utcnow().isoformat() + "Z",
            "channel": session.channel,
        }
        if _day_chat_id:
            _frame["day_chat_id"] = _day_chat_id
        if media:
            _frame["media"] = media
        if tool_events:
            # Live delivery carries the run too, or an open thread shows the
            # voice turn as a bare bubble until the next resync and then
            # silently grows a run card under the reader.
            _frame["tool_events"] = tool_events
        import asyncio as _asyncio
        _asyncio.create_task(broadcast_to_user(current_user.id, _frame))
    except Exception:
        logger.warning("[sessions] live message broadcast failed", exc_info=True)

    return _message_to_response(msg, None, None, {session.id: session.channel})


def _session_to_response(session: Conversation) -> SessionResponse:
    """Convert Conversation model to SessionResponse."""
    metadata = None
    if session.metadata_json:
        try:
            metadata = json.loads(session.metadata_json)
        except json.JSONDecodeError:
            metadata = None
    
    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        channel=session.channel,
        is_active=session.is_active,
        started_at=session.started_at,
        ended_at=session.ended_at,
        updated_at=session.updated_at,
        message_count=session.message_count,
        total_tokens=session.total_tokens,
        metadata=metadata
    )


def _message_to_response(
    message: Message,
    build_jobs: dict = None,
    reply_targets: Optional[dict] = None,
    conversation_channels: Optional[dict] = None,
) -> ChatMessageResponse:
    """Convert Message model to ChatMessageResponse.

    ``reply_targets`` is the bulk-resolved map from
    ``resolve_reply_targets_for_serialization``: ``{replier_id: target_dict}``.
    Optional — single-message callers (POST .../messages) pass None and
    the resulting row just won't have a ``reply_to`` payload (the path
    doesn't carry one anyway).

    ``conversation_channels`` is ``{conversation_id: channel}``. It exists
    because ``Message.channel`` is only written for system senders (routine /
    trigger / subagent / app_builder) — a voice turn leaves it NULL and carries
    "voice" on its Conversation. Conversation first is also what
    api/day_chats.py serializes, so the two paths agree on every row instead of
    the fallback quietly reporting a different channel than the primary.
    """
    # Local import: sessions ↔ day_chats would cycle at module load.
    from app.api.day_chats import _serialize_tool_events

    memories_retrieved = None
    if message.memories_retrieved_json:
        try:
            memories_retrieved = json.loads(message.memories_retrieved_json)
        except json.JSONDecodeError:
            memories_retrieved = None

    # Parse message metadata (media cards, etc.)
    msg_metadata = None
    if getattr(message, 'metadata_json', None):
        try:
            msg_metadata = json.loads(message.metadata_json)
        except (json.JSONDecodeError, TypeError):
            pass

    # Generated-file attachments (JSON column — already a Python list).
    # Strip storage_path — internal key, not for the client.
    #
    # The URLs are built by day_chats._attachment_urls, deliberately, not by
    # hand here. This route is the FALLBACK the mobile client takes whenever
    # /api/day-chats fails, and an attachment that arrives here missing the
    # `thumb_url` day-chats advertises loads the multi-megabyte original into
    # a card sized for a tenth of it — the fallback would be visibly slower
    # than the path it stands in for, which is the one shape a fallback must
    # not have. width/height/has_thumb already ride through untouched.
    attachments_list = None
    raw_atts = getattr(message, 'attachments', None)
    # Belt-and-braces: some drivers may return a JSON string even on a JSON column.
    if isinstance(raw_atts, str):
        try:
            raw_atts = json.loads(raw_atts)
        except (json.JSONDecodeError, TypeError):
            raw_atts = None
    if isinstance(raw_atts, list):
        from app.api.day_chats import _attachment_urls
        attachments_list = [
            {
                **{k: v for k, v in att.items() if k != "storage_path"},
                **_attachment_urls(message.id, att),
            }
            for att in raw_atts
            if isinstance(att, dict)
        ]

    # Alias the real model id before it leaves the API (docs/security/
    # audit-2026.md MI-2). Flag-gated (default off).
    _model_used = message.model_used
    from app.config import settings as _settings
    if _settings.security_leak_filter and _model_used:
        from app.services.model_alias import public_model_label
        _model_used = public_model_label(_model_used)

    resp = dict(
        id=message.id,
        role=message.role,
        # Through the guard, not raw. See api/message_cards.py — a marker
        # row must never reach a client as text on ANY of the four
        # readers, and this one only ever blanked the role it knew about.
        content=public_text(message.role, message.content),
        created_at=message.created_at,
        tokens_prompt=message.tokens_prompt,
        tokens_completion=message.tokens_completion,
        model_used=_model_used,
        memories_retrieved=memories_retrieved,
        processing_time_ms=message.processing_time_ms,
        media=msg_metadata.get("media") if msg_metadata else None,
        # The app this turn published ({"slug": …}). Same parity rule as
        # every other metadata key on this row: the mobile client reads
        # `m.app_artifact?.slug` FIRST and only falls back to parsing the
        # tool prose, so a serializer that omits it sends the client back
        # to the string it is no longer given.
        app_artifact=msg_metadata.get("app_artifact") if msg_metadata else None,
        pending_action=msg_metadata.get("pending_action") if msg_metadata else None,
        # Automations setup cards (Round 26) — same parity rule as the
        # neighbors above; api/day_chats.py and api/messages_recover.py
        # carry the same two keys.
        automation_connector_card=(
            msg_metadata.get("automation_connector_card")
            if msg_metadata else None
        ),
        automation_grant_card=(
            msg_metadata.get("automation_grant_card")
            if msg_metadata else None
        ),
        # Round 29 session chips/cards — same parity rule as above.
        draft_card=msg_metadata.get("draft_card") if msg_metadata else None,
        memory_update=(
            msg_metadata.get("memory_update") if msg_metadata else None
        ),
        fix_chip=msg_metadata.get("fix_chip") if msg_metadata else None,
        # Operator notice (admin dispatch). The clients fall back from
        # /api/day-chats/.../messages to these session routes whenever
        # day-chats fails, so a field emitted by only one serializer
        # vanishes on the fallback and the notice arrives as a plain
        # assistant bubble owned by the agent — exactly what the
        # feature exists to prevent. api/day_chats.py and
        # api/messages_recover.py carry the same key.
        admin_notice=msg_metadata.get("admin_notice") if msg_metadata else None,
        # Same parity rule, and it bit for real (founder recording,
        # 2026-08-19): tool_events was day-chats-only, so a resync that
        # landed on this fallback stripped every reply's tools — the app's
        # reminder card vanished, the fire row un-folded, and the thread
        # visibly flickered for one frame while completely idle.
        tool_events=_serialize_tool_events(message),
        attachments=attachments_list,
        channel=(
            (conversation_channels or {}).get(message.conversation_id)
            or getattr(message, "channel", None)
        ),
        reply_to_message_id=getattr(message, "reply_to_message_id", None),
        reply_to=(reply_targets or {}).get(message.id),
    )

    # Enrich job card messages with current BuildJob status. The projection
    # itself lives in api/message_cards.py — this route used to be the only
    # reader that had one, which is exactly why the other three leaked the
    # marker (see that module's header).
    resp.update(job_card_fields(message, build_jobs))

    return ChatMessageResponse(**resp)


# ── Batch messages by date ───────────────────────────────────────────
@router.get("/by-date/{date_str}/messages")
async def get_messages_by_date(
    date_str: str,
    limit: int = Query(200, ge=1, le=500),
    tz_offset: int = Query(0),  # Client timezone offset in minutes (e.g. -210 for UTC+3:30)
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all messages for all sessions on a specific date (YYYY-MM-DD).
    Returns messages from the 10 most recent sessions, ordered chronologically.
    tz_offset adjusts the date boundary to match the client's local day.
    """
    from datetime import date as date_type, timedelta

    try:
        target_date = date_type.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Try proxying to remote agent first (pass tz_offset through)
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_sessions(proxy[0], proxy[1], f"by-date/{date_str}/messages", {"limit": limit, "tz_offset": tz_offset})
        if data is not None:
            return JSONResponse(content=data)

    # Local: find sessions for this date, adjusted for client timezone.
    # tz_offset is in minutes from UTC (e.g. -210 means UTC+3:30).
    # day_start in UTC = midnight local + offset
    tz_delta = timedelta(minutes=tz_offset)
    day_start = datetime(target_date.year, target_date.month, target_date.day) + tz_delta
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
            .order_by(Conversation.updated_at.desc())
            .limit(10)
        )
    except _MISSING_TABLE_ERRORS:
        # Platform DB doesn't carry conversations/messages (AGENT_ONLY_TABLES).
        # Roll back and return empty so the chat shell renders normally.
        await db.rollback()
        return JSONResponse(content=[])
    conversation_channels = {r[0]: r[1] for r in sessions_result.fetchall()}
    session_ids = list(conversation_channels.keys())

    if not session_ids:
        return JSONResponse(content=[])

    # Get all messages for these sessions in one query
    try:
        messages_result = await db.execute(
            select(Message)
            .where(Message.conversation_id.in_(session_ids))
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
    except _MISSING_TABLE_ERRORS:
        await db.rollback()
        return JSONResponse(content=[])
    messages = messages_result.scalars().all()

    # Enrich with build job data
    build_jobs = await load_build_jobs(db, messages)

    from app.agent.reply_quote import resolve_reply_targets_for_serialization
    reply_targets = await resolve_reply_targets_for_serialization(db, messages)

    return JSONResponse(content=[
        r.model_dump(mode="json") for r in attach_run_to_cards([
            _message_to_response(m, build_jobs, reply_targets, conversation_channels)
            for m in messages
        ])
    ])
