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
from sqlalchemy.orm import selectinload
from typing import Optional, Tuple
import json
import httpx

from app.db import get_db, Conversation, Message, User, AgentConfig
from app.schemas import (
    SessionCreate, SessionResponse, SessionWithMessages, SessionListResponse,
    ChatMessageResponse
)
from app.api.auth import get_current_user

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
    """Proxy a sessions request to the VPS agent."""
    url = f"{agent_url}/api/sessions/{path}" if path else f"{agent_url}/api/sessions"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                headers={"X-Agent-Key": agent_api_key},
                params=params or {},
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
    # Create session (Conversation model)
    session = Conversation(
        user_id=current_user.id,
        title=request.title,
        channel=request.channel,
        metadata_json=json.dumps(request.metadata) if request.metadata else None,
        is_active=True
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
        build_jobs = {}
        job_ids = []
        for m in messages:
            if m.role == "job":
                try:
                    meta = json.loads(m.content) if m.content else {}
                    jid = meta.get("job_id")
                    if jid:
                        job_ids.append(jid)
                except (json.JSONDecodeError, TypeError):
                    pass
        if job_ids:
            from app.db.models import BuildJob
            bj_result = await db.execute(
                select(BuildJob).where(BuildJob.id.in_(job_ids))
            )
            for bj in bj_result.scalars().all():
                build_jobs[bj.id] = bj
        response_dict["messages"] = [_message_to_response(m, build_jobs) for m in messages]
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
    session_query = select(Conversation.id).where(
        and_(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id
        )
    )
    session_result = await db.execute(session_query)
    if not session_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
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
    build_jobs = {}
    job_ids = []
    for m in messages:
        if m.role == "job":
            try:
                meta = json.loads(m.content) if m.content else {}
                jid = meta.get("job_id")
                if jid:
                    job_ids.append(jid)
            except (json.JSONDecodeError, TypeError):
                pass
    if job_ids:
        from app.db.models import BuildJob
        bj_result = await db.execute(
            select(BuildJob).where(BuildJob.id.in_(job_ids))
        )
        for bj in bj_result.scalars().all():
            build_jobs[bj.id] = bj

    return [_message_to_response(m, build_jobs) for m in messages]


@router.post("/{session_id}/messages", response_model=ChatMessageResponse, status_code=status.HTTP_201_CREATED)
async def create_session_message(
    session_id: str,
    role: str = "user",
    content: str = "",
    model_used: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a message to a session.

    Used by the platform voice route to persist voice messages
    on the user's VPS database (not platform DB).
    """
    import uuid as _uuid

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

    msg = Message(
        id=str(_uuid.uuid4()),
        conversation_id=session_id,
        role=role,
        content=content.replace("\x00", ""),
        model_used=model_used,
    )
    db.add(msg)

    session.message_count = (session.message_count or 0) + 1
    session.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(msg)

    return _message_to_response(msg)


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


def _message_to_response(message: Message, build_jobs: dict = None) -> ChatMessageResponse:
    """Convert Message model to ChatMessageResponse."""
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

    resp = dict(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        tokens_prompt=message.tokens_prompt,
        tokens_completion=message.tokens_completion,
        model_used=message.model_used,
        memories_retrieved=memories_retrieved,
        processing_time_ms=message.processing_time_ms,
        media=msg_metadata.get("media") if msg_metadata else None,
    )

    # Enrich job card messages with current BuildJob status
    if message.role == "job":
        try:
            job_meta = json.loads(message.content) if message.content else {}
        except (json.JSONDecodeError, TypeError):
            job_meta = {}
        job_id = job_meta.get("job_id", "")
        resp["job_id"] = job_id
        resp["job_name"] = job_meta.get("job_name", "App Build")
        resp["content"] = ""  # Don't expose raw JSON to frontend
        if build_jobs and job_id in build_jobs:
            bj = build_jobs[job_id]
            resp["job_status"] = bj.status
            resp["job_app_id"] = bj.app_id
            # Use title as name if stored name is generic
            if resp["job_name"] == "App Build" and bj.title:
                resp["job_name"] = bj.title.replace("Build: ", "")
            try:
                steps = json.loads(bj.steps_json) if bj.steps_json else []
                completed = sum(1 for s in steps if s.get("status") == "completed")
                resp["job_total_steps"] = len(steps)
                resp["job_completed_steps"] = completed
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            resp["job_status"] = job_meta.get("job_status", "completed")

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

    sessions_result = await db.execute(
        select(Conversation.id)
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
    session_ids = [r[0] for r in sessions_result.fetchall()]

    if not session_ids:
        return JSONResponse(content=[])

    # Get all messages for these sessions in one query
    messages_result = await db.execute(
        select(Message)
        .where(Message.conversation_id.in_(session_ids))
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    messages = messages_result.scalars().all()

    # Enrich with build job data
    build_jobs = {}
    job_ids = [
        m.metadata_json and __import__("json").loads(m.metadata_json).get("job_id")
        for m in messages if m.role == "job"
    ]
    job_ids = [j for j in job_ids if j]
    if job_ids:
        from app.db.models import BuildJob
        bj_result = await db.execute(select(BuildJob).where(BuildJob.id.in_(job_ids)))
        for bj in bj_result.scalars().all():
            build_jobs[bj.id] = bj

    return JSONResponse(content=[
        _message_to_response(m, build_jobs).model_dump(mode="json") for m in messages
    ])
