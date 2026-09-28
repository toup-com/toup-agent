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
import unicodedata
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
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
    ChatMessageResponse, SessionMessageCreate,
    # The nothing-heard projection. One implementation for all five client
    # serializers, declared beside the wire field it enforces.
    public_heard_text,
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
    # WHERE THE DATA ACTUALLY IS. `serving_locally()` is true in an agent
    # container and in a monolith/dev run — the AGENT_ONLY tables are in THIS
    # process's database and there is nothing to proxy to. Without this the
    # agent can resolve its OWN `agent_configs` row and proxy to itself over
    # the public internet: harmless while a failed hop fell through to the
    # local SELECT, and a 503 over a perfectly readable local database now that
    # it does not. `tenant_proxy.agent_proxy_info` has always had this guard;
    # these two copies never did.
    from app.api.tenant_proxy import serving_locally
    if serving_locally():
        return None
    # NOT gated on `deploy_status == "active"` — see the note in
    # `day_chats._get_agent_proxy_info`. A container mid-redeploy, or one a
    # stale-deploy sweep marked "error" 15 minutes later, is very often still
    # up and holding the user's entire history; skipping the proxy for those
    # users served them the platform's own empty tables instead.
    try:
        async with db.begin_nested():
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(
                    AgentConfig.user_id == user_id,
                )
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass  # agent_configs table may not exist — savepoint rolled back
    return None


class SessionsAgentUnreachable(Exception):
    """The tenant owns these sessions and did not answer. Never swallowed.

    Same defect as `day_chats.AgentUnreachable`, and the same 2026-08-31
    incident: this module is the FALLBACK the mobile client reaches for when
    `/api/day-chats` fails, so a silent `None` here meant both the primary and
    the recovery path answered "you have no history" while the tenant was
    merely slow. See `day_chats.AgentUnreachable` for the full account.
    """

    def __init__(self, detail: str, reason: str = "agent_unreachable"):
        super().__init__(detail)
        self.detail = detail
        # See `day_chats.AgentUnreachable.reason` — `agent_error` means the
        # tenant answered 500, which is an answer and is not retried.
        self.reason = reason


class SessionsAgentSaidNo(Exception):
    """A 4xx from the tenant — an answer, forwarded verbatim."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


# The WHOLE LADDER has to fit inside the mobile client's 15 s abort — see the
# note on `day_chats._PROXY_TIMEOUT_S`. 2 × 6 s + 0.4 s = 12.4 s worst case.
_SESSIONS_PROXY_TIMEOUT_S = 6.0
_SESSIONS_PROXY_ATTEMPTS = 2
_SESSIONS_PROXY_BACKOFF_S = 0.4


def _sessions_unreachable(exc: Exception) -> HTTPException:
    """See `day_chats._unreachable` — same shape, same reason. The cause is
    logged, never sent: a user cannot act on "ReadTimeout()"."""
    reason = getattr(exc, "reason", "agent_unreachable")
    logger.warning("[sessions] answering 503 %s: %s",
                   reason, getattr(exc, "detail", exc))
    return HTTPException(
        status_code=503,
        detail="Your agent did not answer in time. Your history is safe — try again.",
        headers={"Retry-After": "2", "X-Toup-Reason": reason},
    )


def _sessions_4xx(e: "SessionsAgentSaidNo") -> HTTPException:
    """See `day_chats._agent_4xx` — same reason, same shape.

    A 401/403 FROM THE TENANT AGENT is the platform's own `X-Agent-Key` being
    stale, NOT a bad user JWT; forwarded verbatim it signs the user out of the
    app (`api.ts` treats any 401 as an expired token). `/api/sessions` is the
    FALLBACK the mobile client reaches for when `/api/day-chats` fails, so a
    forwarded 401 here signs the user out on the recovery path too. Map 401/403
    to 503 + `Retry-After` + `X-Toup-Reason: agent_key_stale`; forward every
    other 4xx (the agent's real answer) verbatim. A platform-auth 401 is raised
    before the tenant call and never reaches here.
    """
    if e.status in (401, 403):
        logger.warning(
            "[sessions] tenant answered %s — platform agent key stale; "
            "answering 503 agent_key_stale (forwarding would sign the user out)",
            e.status,
        )
        return HTTPException(
            status_code=503,
            detail="Your agent is still starting up. Please try again in a moment.",
            headers={"Retry-After": "3", "X-Toup-Reason": "agent_key_stale"},
        )
    return HTTPException(status_code=e.status, detail=e.body or "Agent declined")


async def _proxy_sessions(
    agent_url: str, agent_api_key: str, path: str, params: Optional[dict] = None
):
    """Proxy a sessions request to the VPS agent.

    TKT-LAT-007: uses the shared agent_http client.

    Raises `SessionsAgentUnreachable` on transport failure or a 5xx, and
    `SessionsAgentSaidNo` on a 4xx. It never returns `None`: the caller cannot
    tell an empty account from an unread one, and every caller below sits in
    front of a screen that renders those two identically.
    """
    import asyncio

    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url}/api/sessions/{path}" if path else f"{agent_url}/api/sessions"
    last = "unknown"
    for attempt in range(1, _SESSIONS_PROXY_ATTEMPTS + 1):
        try:
            client = get_agent_http_client()
            resp = await client.get(
                url,
                headers={"X-Agent-Key": agent_api_key},
                params=params or {},
                timeout=_SESSIONS_PROXY_TIMEOUT_S,
            )
            if resp.status_code == 200:
                return resp.json()
            if 400 <= resp.status_code < 500:
                raise SessionsAgentSaidNo(resp.status_code, resp.text[:300])
            # A deterministic tenant 500 is an ANSWER; retrying it is the same
            # answer at twice the load. Only 502/503/504 and transport errors
            # (a hop failing in FRONT of the tenant) stay in the retry set.
            # `day_chats._proxy_day_chats` makes the same split — the two
            # doors have to agree or the fallback is slower than the path it
            # stands in for.
            if resp.status_code == 500:
                logger.warning(
                    "Agent sessions proxy %s: tenant answered 500 — not retried "
                    "(deterministic)", url,
                )
                raise SessionsAgentUnreachable("HTTP 500", reason="agent_error")
            last = f"HTTP {resp.status_code}"
        except (SessionsAgentSaidNo, SessionsAgentUnreachable):
            raise
        except Exception as e:
            # `repr`, not `str`: httpx timeout exceptions stringify to "".
            last = repr(e)
        if attempt < _SESSIONS_PROXY_ATTEMPTS:
            logger.info("Agent sessions proxy %s attempt %d/%d failed (%s) — retrying",
                        url, attempt, _SESSIONS_PROXY_ATTEMPTS, last)
            await asyncio.sleep(_SESSIONS_PROXY_BACKOFF_S)
    logger.warning("Agent sessions proxy %s failed after %d attempts: %s",
                   url, _SESSIONS_PROXY_ATTEMPTS, last)
    raise SessionsAgentUnreachable(last)


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
    # `le` was 100, and the shipped mobile client asks for 120 and 360:
    # `api.getDayChats(n)` falls back to `getSessions(n * 4)` when day-chats
    # fails, so the whole recovery path had been answering 422 in production —
    # proven in the 2026-08-31 trail, two 422s inside the same second as the
    # day-chats timeout that summoned them. The client is clamped in the same
    # round, but binaries already on phones are not, so the ceiling moves here
    # too. A session row is small and this route is offset-paginated.
    limit: int = Query(20, ge=1, le=500),
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
        # The tenant is authoritative for this list. A failed hop used to fall
        # through to the platform DB, which holds only the voice/web sessions
        # merged below — so an unreachable tenant answered 200 with a partial
        # list that read as the whole account. 503 instead.
        try:
            data = await _proxy_sessions(proxy[0], proxy[1], "", params)
        except SessionsAgentSaidNo as e:
            raise _sessions_4xx(e)
        except SessionsAgentUnreachable as e:
            raise _sessions_unreachable(e)
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
            # `is_local` already proved this session is NOT in the platform
            # DB, so falling through on a failed hop reaches a query that can
            # only 404 — "session not found" for a session that exists.
            try:
                data = await _proxy_sessions(proxy[0], proxy[1], session_id, params)
            except SessionsAgentSaidNo as e:
                raise _sessions_4xx(e)
            except SessionsAgentUnreachable as e:
                raise _sessions_unreachable(e)
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
            # Same as above: not local, so the fall-through can only 404.
            try:
                data = await _proxy_sessions(proxy[0], proxy[1], f"{session_id}/messages", params)
            except SessionsAgentSaidNo as e:
                raise _sessions_4xx(e)
            except SessionsAgentUnreachable as e:
                raise _sessions_unreachable(e)
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
        # (created_at, id) — the tiebreaker `day_context_loader` already has.
        # Without it two rows sharing a timestamp come back in a different
        # order on two fetches.
        # COALESCE(occurred_at, created_at) — see day_chats.py (round 46, A12).
        .order_by(
            func.coalesce(Message.occurred_at, Message.created_at).asc(),
            Message.id.asc(),
        )
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


# Voice provenance (R48 §8). The relay stamps WHERE a row came from so the
# thread can tell a spoken reply apart from a delegated answer, and so a row
# that was cut off says so. Allowlisted for the same reason `_TOOL_EVENT_KEYS`
# is: this dict is request-body input that ends up inside a row every client
# renders, and an open-ended blob there is an unbounded write.
_VOICE_KEYS = {
    "source",          # live_spoken | delegated | fast_path | …
    "delegation_id",
    "model",
    "played_ms",
    "generated_chars",
    "interrupted",
    "superseded",
    "cancelled",
    "epoch",
    # GPT-Live causal/timing provenance. These are bounded scalar identifiers
    # and provider-clock positions, never transcript or model text.
    "turn_id",
    # Exact caller turns consumed by a task, in spoken order. The app may
    # coalesce only these persisted user rows after the voice socket closes.
    "request_turn_ids",
    "assistant_turn_id",
    "parent_user_turn_id",
    "task_id",
    "start_ms",
    "end_ms",
    "clock",
    # Blocker B / contract v0.2 §`message.voice`. The record's own shape number
    # and WHICH of the two rows a delegated turn produced:
    #   task_result          — the complete backend answer, never truncated,
    #                          and therefore carrying NO playback numbers
    #                          (they describe the spoken paraphrase, not this);
    #   assistant_transcript — what the caller actually heard for one epoch.
    # A reader without `record_kind` cannot tell the two apart and falls back
    # to legacy single-row semantics, which is why the relay must keep
    # degrading correctly while a tenant is still on an image without these
    # keys — until this set ships, they are dropped here and the drop is
    # logged rather than silent.
    # `heard_chars` is a COUNT, never the words: "how much of the epoch this
    # client displayed". It is not an acoustic claim and is never estimated
    # from `played_ms`. The text itself is the row's own content — no key here
    # ever carries transcript, which is what `_VOICE_STR_MAX` below enforces
    # (`task_title`, at the end, is the task's bounded NAME, not a transcript).
    "version",
    "record_kind",
    "heard_chars",
    # …and the relay's SECOND way of saying "nothing of this epoch stands".
    # A transcript row already written with text, then contradicted by a
    # validated empty receipt, cannot be un-written (the caller may have read
    # it in the thread) and cannot be rewritten to empty content (this very
    # route 400s an empty body). It is re-stamped with this flag instead, and
    # `schemas.heard_nothing` — the one projection every reader funnels
    # through — stops serving the text. Dropped here, the row would keep
    # serving words that were never played: the flag must survive the
    # allowlist or the retraction never reaches a client at all.
    "transcript_retracted",
    # R12.2: did the provider actually SAY this delegated answer out loud?
    # False is not a failure — the result is in the thread either way — but
    # nothing may claim audio that never happened, and the row is the only
    # place that survives the call.
    "spoken",
    # R2 §D: the task this one replaces, refines or continues — an id, so a
    # reader can link a correction to the request it corrected after the call.
    "related_task_id",
    # R2 AF6: the task's NAME — the same string the live `delegation` frame
    # carried as `title`, which headed the call's card — so the day chat names
    # a voice run the way the call did instead of by the nearest (fragment)
    # user row. The ONE key here that carries words rather than an id, count or
    # flag, and they are the caller's own request as already shown on the
    # card: see `_VOICE_TEXT_KEYS` for how it is sanitized and bounded.
    "task_title",
}
#: A provenance value is an id, a count or a flag — never prose. Anything
#: longer is not provenance and is dropped rather than truncated, so a caller
#: cannot smuggle a paragraph in under a known key.
_VOICE_STR_MAX = 120
#: …with ONE exception, and it is not a style choice. `delegation_id` is
#: COMPARED, by `voice_jobs.VoiceTurnJob._row_is_proof`, against a copy of the
#: same provider-supplied id that reached the agent bounded to 64 (the relay
#: body, then `ChatRequest(max_length=64)`). Stored at a different length the
#: two can never match, and a delivered answer closes its card `cancelled` —
#: silently, in the safe-looking direction. So this key is TRUNCATED to the
#: same bound rather than kept long or dropped when oversize; 64 characters of
#: an id is not the paragraph the rule above exists to stop.
_VOICE_ID_MAX = 64
_VOICE_BOUNDED_IDS = {"delegation_id"}
#: …and the one key that is WORDS: `task_title` (AF6). Sanitized to one line —
#: a control character (a newline, a NUL) becomes a space and whitespace runs
#: collapse, so a name cannot break the card it heads; a ZWNJ is not
#: whitespace and survives — then bounded by `_VOICE_STR_MAX` with the rule
#: above: dropped, not truncated. The relay already cuts the title at a word
#: boundary to 80 (`live_voice_protocol.request_title`), so anything longer is
#: not a title it wrote, and a half-cut one would be a new fragment. Empty
#: after cleaning is dropped too: an empty name still reads as a name, and
#: without one the app keeps its own ask-derived title.
_VOICE_TEXT_KEYS = {"task_title"}
_VOICE_REQUEST_TURN_IDS_MAX = 16  # live_voice_protocol.REQUEST_TURN_IDS_MAX


def _clean_voice_text(value) -> str:
    if not isinstance(value, str):
        return ""
    if len(value) > 4 * _VOICE_STR_MAX:
        # Request-body input: no per-character pass over a value this far past
        # the ceiling. Returned as-is, it is longer than `_VOICE_STR_MAX`
        # whatever its whitespace, so the caller drops it as oversize.
        return value
    one_line = "".join(
        " " if unicodedata.category(ch) == "Cc" else ch for ch in value
    )
    return " ".join(one_line.split())


def _clean_voice(voice) -> Optional[dict]:
    """Key-allowlist and bound the body's `voice` provenance dict.

    A dropped key is LOGGED, by name. The allowlist is the right default for
    request-body input, but the producer is our own relay: the failure mode
    that matters is "L1 added a key and it vanished", and a silent drop makes
    that indistinguishable from a relay that never sent it. Names and counts
    only — a provenance VALUE is an id the row already carries, and the rule in
    this round is that nothing logs content.
    """
    if not isinstance(voice, dict) or not voice:
        return None
    out: dict = {}
    dropped: list = []
    for k, v in voice.items():
        if k not in _VOICE_KEYS:
            dropped.append(str(k)[:40])
            continue
        if k == "request_turn_ids":
            if (
                isinstance(v, list)
                and 0 < len(v) <= _VOICE_REQUEST_TURN_IDS_MAX
                and all(isinstance(turn_id, str) and 0 < len(turn_id) <= _VOICE_STR_MAX
                        for turn_id in v)
                and len(set(v)) == len(v)
            ):
                out[k] = list(v)
            else:
                dropped.append(f"{k}:invalid")
            continue
        if k in _VOICE_BOUNDED_IDS and isinstance(v, str):
            out[k] = v[:_VOICE_ID_MAX]
            continue
        if k in _VOICE_TEXT_KEYS:
            text = _clean_voice_text(v)
            if text and len(text) <= _VOICE_STR_MAX:
                out[k] = text
            else:
                dropped.append(f"{str(k)[:40]}:{'oversize' if text else 'empty'}")
            continue
        if isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
            out[k] = v
        elif isinstance(v, str) and len(v) <= _VOICE_STR_MAX:
            out[k] = v
        else:
            dropped.append(f"{str(k)[:40]}:oversize")
    if dropped:
        logger.warning(
            "[sessions] voice provenance dropped %d key(s): %s",
            len(dropped), ",".join(sorted(dropped)[:12]),
        )
    return out or None


def _clean_tool_events(events) -> Optional[list]:
    if not isinstance(events, list) or not events:
        return None
    out = []
    for e in events[:_TOOL_EVENTS_MAX]:
        if not isinstance(e, dict) or "tool" not in e or "started_at_ms" not in e:
            continue
        out.append({k: v for k, v in e.items() if k in _TOOL_EVENT_KEYS})
    return out or None


# The fields an attachment dict may carry on the way IN. `Message.attachments`
# had only server-controlled writers until C8 opened it to a request body, and
# one of its fields is a capability: `files.py` reads `storage_path` back and
# streams whatever object key it names (files.py `_load_attachment` →
# `download_file`), checking only that the caller owns the parent Conversation.
# Sibling untrusted input in this file is allowlisted for exactly this reason
# (`_TOOL_EVENT_KEYS`); this is the same rule for the same class of field.
_ATTACHMENT_KEYS = {
    "id", "attachment_id", "filename", "mime_type", "size_bytes", "created_at",
    "width", "height", "has_thumb", "kind", "role", "intent", "ingest",
    # Deliberately last and deliberately conditional — see `_clean_attachments`.
    "storage_path",
}
# A13: MAX_ATTACHMENTS_PER_TURN.
_ATTACHMENTS_MAX = 8


def _clean_attachments(atts, *, user_id: str, trusted: bool) -> Optional[list]:
    """Key-allowlist the request body's attachment dicts.

    `trusted` is "this request authenticated with the agent key", i.e. it came
    from the platform voice relay, which is the only intended producer. An
    ordinary JWT caller keeps every descriptive field and loses `storage_path`:
    naming an arbitrary object key under the workspace would otherwise mint
    itself a download of a file it never produced — including, on a recycled
    pool slot, one a previous tenant left behind.

    Even for the relay the key must stay inside the user's own scope
    (`doc_generators._persist` writes `{user_id}/{att_id}_{filename}`), because
    the relay forwards what the agent returned and a compromised or confused
    producer is exactly what a capability field must survive.
    """
    if not isinstance(atts, list) or not atts:
        return None
    out = []
    for a in atts[:_ATTACHMENTS_MAX]:
        if not isinstance(a, dict):
            continue
        clean = {k: v for k, v in a.items() if k in _ATTACHMENT_KEYS}
        key = clean.get("storage_path")
        if key is not None:
            ok = (
                trusted and isinstance(key, str) and key
                and not key.startswith("/") and ".." not in key
                and key.split("/", 1)[0] in (user_id, "shared")
            )
            if not ok:
                clean.pop("storage_path", None)
        out.append(clean)
    return out or None


def _build_metadata(media, tool_events, app_artifact=None, voice=None) -> Optional[str]:
    """The message's metadata_json, in AgentRunner._save_messages' shape.

    One writer for every key: they used to be mutually exclusive here (media
    replaced the whole blob), so a voice turn that played a song AND used a
    tool could only ever persist one of them.
    """
    meta = {}
    if media:
        meta["media"] = media
    if tool_events:
        meta["tool_events"] = tool_events
    if app_artifact:
        # Same key AgentRunner._save_messages writes; the clients read only it.
        meta["app_artifact"] = app_artifact
    if voice:
        meta["voice"] = voice
    return json.dumps(meta) if meta else None


def _merge_metadata(existing: Optional[str], incoming: Optional[str]) -> Optional[str]:
    """Metadata for a REWRITE of a row that already exists.

    R48 made repeated writes of one key the normal case (a Live utterance is
    revised in place as it is spoken), and the revisions after the first carry
    provenance and nothing else. A whole-blob replace would therefore delete
    the media card and the run rail off a row that had them — the same defect
    the `metadata_json is None → skip` rule above already guards for the
    no-metadata case, one step further along. So a rewrite only ever overwrites
    the keys it actually carries; a key it is silent about is kept.

    The merge is TOP-LEVEL ONLY, and for `voice` that is deliberate rather than
    an oversight — a rewrite REPLACES the whole provenance object.

    Blocker B D2 requires the delegated row's playback numbers (`played_ms`,
    `interrupted`, `epoch`, `start_ms`, `end_ms`) to be GONE once that row is
    re-stamped `record_kind: "task_result"`: they describe the spoken
    paraphrase, and the complete backend answer is a different, longer string.
    A deep/additive merge cannot express a removal, so it would resurrect those
    numbers under the one key whose whole purpose is that it does not carry
    them — a row claiming playback facts about text it is not. Replacement is
    the only semantics that can.

    The cost is the mirror case: a partial revision loses the keys the earlier
    write established. That is a producer contract, not a hope — the relay
    writes the FULL voice object on every write of a row (contract v0.2,
    `message.voice`), so there is no partial revision to lose anything. It is
    pinned by `tests/test_voice_record_metadata.py`, which asserts both halves:
    siblings (`media`, `tool_events`) survive a voice-only rewrite, and `voice`
    itself does not accumulate.
    """
    if incoming is None:
        return None
    try:
        new = json.loads(incoming)
    except (TypeError, ValueError):
        return None
    if not isinstance(new, dict):
        return None
    old = {}
    if existing:
        try:
            _parsed = json.loads(existing)
            if isinstance(_parsed, dict):
                old = _parsed
        except (TypeError, ValueError):
            old = {}
    # A repaired Live socket claims one automatic read-aloud attempt directly
    # on the durable answer row. The original detached relay may still submit
    # a later full-voice revision with `spoken:false`; preserve only this
    # monotonic claim when it names the SAME delegation. This does not restore
    # any playback fields that the full-voice replacement intentionally drops.
    previous_voice = old.get("voice") if isinstance(old.get("voice"), dict) else {}
    incoming_voice = new.get("voice") if isinstance(new.get("voice"), dict) else None
    if (
        incoming_voice is not None
        and previous_voice.get("recovery_speech_claimed") is True
        and previous_voice.get("delegation_id")
        and previous_voice.get("delegation_id") == incoming_voice.get("delegation_id")
    ):
        incoming_voice["recovery_speech_claimed"] = True
    old.update(new)
    return json.dumps(old) if old else None


@router.post("/{session_id}/messages", response_model=ChatMessageResponse, status_code=status.HTTP_201_CREATED)
async def create_session_message(
    session_id: str,
    body: Optional[SessionMessageCreate] = None,
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a message to a session.

    Used by the platform voice route to persist voice messages
    on the user's VPS database (not platform DB).

    **BODY ONLY.** `role`/`content`/`model_used`/`day_chat_id` used to be
    declared here as bare scalars with defaults, i.e. as QUERY PARAMETERS,
    with the body merely winning when both were present. That left the
    transcript leak open on the RECEIVING end after the sender was fixed: any
    caller still using the old form puts the user's spoken sentence into the
    request line, which uvicorn logs and the fleet ships to Loki (observed
    2026-09-15, fifteen such lines inside one four-minute window). It also made
    the `role` pattern on `SessionMessageCreate` unenforceable, because an
    unvalidated query value won whenever the body omitted the field. There is
    no in-tree producer of the query form, and an out-of-tree one now gets a
    loud 400 instead of a silent leak.
    """
    import uuid as _uuid

    role = "user"
    content = ""
    model_used = None
    day_chat_id = None
    media = None
    tool_events = None
    attachments = None
    app_artifact = None
    client_msg_id = None
    occurred_at = None
    revision = None
    voice = None
    # "Authenticated with the agent key", i.e. from the platform voice relay —
    # the only intended producer of `attachments`. See `_clean_attachments`.
    _trusted = False
    if request is not None:
        try:
            import secrets as _secrets
            from app.config import settings as _cfg
            _key = request.headers.get("x-agent-key") or ""
            _trusted = bool(
                _cfg.agent_api_key
                and _secrets.compare_digest(_key, _cfg.agent_api_key)
            )
        except Exception:  # noqa: BLE001 — auth already passed; this only widens
            _trusted = False
    if body is not None:
        role = body.role or role
        content = body.content or content
        model_used = body.model_used or model_used
        day_chat_id = body.day_chat_id or day_chat_id
        media = body.media or None
        tool_events = _clean_tool_events(body.tool_events)
        attachments = _clean_attachments(
            body.attachments, user_id=current_user.id, trusted=_trusted,
        )
        app_artifact = body.app_artifact or None
        client_msg_id = (body.client_msg_id or None)
        occurred_at = body.occurred_at
        voice = _clean_voice(getattr(body, "voice", None))
        # Absent stays absent: `message_frame` only puts `revision` on the wire
        # when it is not None, and the clients read absent as "not a
        # correction". A negative counter is noise, never a correction.
        _rev = getattr(body, "revision", None)
        revision = (
            int(_rev)
            if isinstance(_rev, int) and not isinstance(_rev, bool) and _rev >= 0
            else None
        )

    # The pydantic pattern is only a gate if nothing can route round it. `role`
    # is half of the UPSERT key and is stamped on the live frame; an arbitrary
    # string here is a role the clients have no rendering path for.
    if role not in ("user", "assistant"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role",
        )

    # Verify session ownership
    session_query = select(Conversation).where(
        and_(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id,
        )
    )
    # A KEYED write is a read-modify-write with no unique constraint behind it
    # (`client_msg_id` is deliberately non-unique — a chat turn stamps one value
    # on both of its rows), so two overlapping POSTs of the same key both find
    # nothing and both INSERT. That was latent while no producer ever repeated a
    # key; R48's revise-in-place makes a repeated key the normal case, and the
    # duplicate would be two rows holding different prefixes of one sentence.
    # Locking the CONVERSATION row makes the lookup-then-write atomic per
    # session. It is not an extra lock: the write below updates
    # `session.message_count` on this same row, so Postgres takes it either way
    # — this only takes it earlier, in the same order, which is why it cannot
    # deadlock where today does not. Postgres only; SQLite serialises writes
    # itself and MySQL is not a target.
    if client_msg_id and db.bind is not None and db.bind.dialect.name == "postgresql":
        session_query = session_query.with_for_update()
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
    #
    # A closure because it has to be RE-RUNNABLE. `get_or_create_day_chat`
    # INSERTs the DayChat and only `flush()`es it — no commit — so every
    # `db.rollback()` in the degradation paths below destroys a row the
    # Message's `day_chat_id` FOREIGN KEY still points at. The follow-up commit
    # then raises a FK violation that neither handler expects, turning the
    # designed graceful degradation into a 500 and losing the voice turn. The
    # resolver is idempotent and self-heals its own id cache when the row it
    # remembers is gone, so re-resolving after a rollback is the whole fix.
    # Read off the ORM row ONCE, here, while it is still loaded: `rollback()`
    # expires every instance in the session, and an expired attribute on an
    # AsyncSession raises MissingGreenlet rather than lazy-loading. A closure
    # that touched `current_user.timezone` after a rollback would 500 on the
    # very path written to degrade.
    _owner_id = current_user.id
    _owner_tz = getattr(current_user, 'timezone', None)

    async def _resolve_day():
        _d = day_chat_id
        if not _d:
            from app.db.message_helpers import resolve_day_chat_id_for_now
            _d = await resolve_day_chat_id_for_now(
                db, _owner_id, tz_override=_owner_tz,
            )
        if not _d and session is not None:
            _d = session.day_chat_id
        return _d

    _day_chat_id = await _resolve_day()
    # Backfill the session's day_chat_id if it was missing
    if _day_chat_id and not session.day_chat_id:
        session.day_chat_id = _day_chat_id

    # Exactly-once (R46 C1): a persist that is retried — a relay reconnect, a
    # re-delivered provider event — must land on the SAME row rather than a
    # second copy of the same utterance. The key is the caller's; with none, a
    # plain insert, which is what every pre-R46 caller gets.
    # The mapped attribute can also be absent outright when this platform build
    # is newer than the ORM it is running against (a mixed rollout), so the
    # identity fields are probed on the class, not assumed.
    _has_cmid = hasattr(Message, "client_msg_id")
    _has_occurred = hasattr(Message, "occurred_at")
    existing = None
    if client_msg_id and _has_cmid:
        try:
            existing = (await db.execute(select(Message).where(and_(
                Message.conversation_id == session_id,
                Message.client_msg_id == client_msg_id,
                # ROLE is part of the key. `client_msg_id` is explicitly NOT
                # unique (db/models/conversation.py): a chat turn stamps the
                # same value on its user row AND its assistant row. Keyed on the
                # pair alone, the second POST of such a turn would overwrite the
                # first row's role and content instead of appending — losing the
                # user's message. Ordered so `.first()` is deterministic if a
                # duplicate pair already exists from before this guard.
                Message.role == role,
            )).order_by(Message.created_at.asc(), Message.id.asc()))).scalars().first()
        except Exception:
            # The column has not reached this tenant DB yet (agent DBs self-heal
            # their schema, they do not migrate). Degrade to an insert; the
            # defensive retry below drops the field from the INSERT too.
            await db.rollback()
            existing = None
            # rollback() EXPIRES every instance in the session, and the next
            # statements read `session.message_count` / `session.channel` — an
            # expired attribute on an AsyncSession raises MissingGreenlet, so
            # this degradation path 500'd instead of degrading. Reload the row
            # the way ws_chat.py does after its own rollback, and re-apply the
            # day_chat_id backfill the rollback undid.
            session = await db.get(Conversation, session_id)
            if session is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Session not found",
                )
            # The rollback also discarded the DayChat the resolver had just
            # INSERTed-and-flushed, and `_day_chat_id` still names it. Inserting
            # the Message against that id is a dangling FK.
            _day_chat_id = await _resolve_day()
            if _day_chat_id and not session.day_chat_id:
                session.day_chat_id = _day_chat_id

    _occurred = occurred_at
    if _occurred is not None and _occurred.tzinfo is not None:
        # Message.created_at/occurred_at are naive UTC everywhere else.
        _occurred = _occurred.astimezone(timezone.utc).replace(tzinfo=None)
    if _occurred is not None:
        # Unvalidated client input that drives the thread's sort key
        # (COALESCE(occurred_at, created_at)). A far-future stamp pins a row to
        # the bottom of the day forever. Out of a day's reach in either
        # direction it is not a stamp, it is noise — and C1 already defines
        # null as "unknown", which sorts on created_at.
        _skew = abs((datetime.utcnow() - _occurred).total_seconds())
        if _skew > 86400:
            _occurred = None

    _msg_kwargs = dict(
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
        metadata_json=_build_metadata(media, tool_events, app_artifact, voice),
    )
    if attachments:
        # The COLUMN, not metadata_json — GET /api/files/{message_id}/{aid}
        # authorizes against it and every history serializer already reads it.
        _msg_kwargs["attachments"] = attachments
    _identity_kwargs = {}
    if client_msg_id and _has_cmid:
        _identity_kwargs["client_msg_id"] = client_msg_id
    if _occurred is not None and _has_occurred:
        _identity_kwargs["occurred_at"] = _occurred

    if existing is not None:
        msg = existing
        for _k, _v in _msg_kwargs.items():
            if _k == "id":
                continue
            if _k == "metadata_json":
                # Degrade the way `attachments` does, not the opposite way. A
                # replayed assistant persist after a relay reconnect carries no
                # media and no tool_events by construction (both are per-socket
                # state), so an unconditional write NULLed the media card and
                # the run rail off a row that already had them. A rewrite that
                # DOES carry metadata is merged for the same reason — see
                # `_merge_metadata`.
                _merged = _merge_metadata(getattr(msg, "metadata_json", None), _v)
                if _merged is None:
                    continue
                setattr(msg, _k, _merged)
                continue
            setattr(msg, _k, _v)
        if _occurred is not None and _has_occurred:
            msg.occurred_at = _occurred
        session.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(msg)
    else:
        msg = Message(**_msg_kwargs, **_identity_kwargs)
        db.add(msg)
        session.message_count = (session.message_count or 0) + 1
        session.updated_at = datetime.utcnow()
        try:
            await db.commit()
        except Exception as _commit_err:
            # Same defensive retry ws_chat runs for reply_to_message_id: a
            # tenant whose ALTER has not run yet rejects the INSERT with
            # "column client_msg_id of relation messages does not exist".
            # Losing the identity fields degrades ordering; losing the
            # transcript loses the turn.
            _err = str(_commit_err).lower()
            if _identity_kwargs and any(k in _err for k in _identity_kwargs):
                logger.warning(
                    "[sessions] message identity columns missing on this tenant; "
                    "persisting without client_msg_id/occurred_at",
                )
                await db.rollback()
                # Same trap as the lookup's rollback above: the rollback expires
                # `session`, and the retry reads session.message_count two lines
                # down — MissingGreenlet, a 500 where a degraded insert was
                # intended. Reload, and re-apply the backfill the rollback undid.
                session = await db.get(Conversation, session_id)
                if session is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Session not found",
                    )
                # …and the DayChat the resolver INSERTed-and-flushed went with
                # it, so `_msg_kwargs["day_chat_id"]` names a row that no longer
                # exists. Re-resolve BEFORE rebuilding the Message or the retry
                # trades a missing-column error for a foreign-key one.
                _day_chat_id = await _resolve_day()
                _msg_kwargs["day_chat_id"] = _day_chat_id
                if _day_chat_id and not session.day_chat_id:
                    session.day_chat_id = _day_chat_id
                msg = Message(**_msg_kwargs)
                db.add(msg)
                session.message_count = (session.message_count or 0) + 1
                session.updated_at = datetime.utcnow()
                db.add(session)
                await db.commit()
            else:
                raise
        await db.refresh(msg)

    # Hand the row to any open chat socket live, WITH its media. Voice rows
    # used to reach the phone only on the next resync (as plain text until
    # then), so a song the agent genuinely started rendered as a bare
    # transcript line while it played. Fire-and-forget — live delivery must
    # never be on the persist path.
    try:
        from app.api.ws_chat import broadcast_to_user
        from app.api.message_frames import message_frame
        from app.api.day_chats import _serialize_attachments
        # The frame describes the COMMITTED row, not the request body. They
        # diverge on a rewrite: a revision carries provenance and a longer
        # sentence, not the media card the first write established, and the
        # merge above deliberately kept that card on the row. Sending the
        # body's view would hand the client a frame with no media for a row
        # that has one — and the client replaces in place on `revision`, so the
        # card would vanish from an open thread until the next refetch.
        _frame_media, _frame_tools, _frame_app = media, tool_events, app_artifact
        if existing is not None:
            try:
                _committed = json.loads(getattr(msg, "metadata_json", None) or "{}")
            except (TypeError, ValueError):
                _committed = {}
            if isinstance(_committed, dict):
                _frame_media = media or _committed.get("media")
                _frame_tools = tool_events or _committed.get("tool_events")
                _frame_app = app_artifact or _committed.get("app_artifact")
        _frame = message_frame(
            msg,
            channel=session.channel,
            day_chat_id=_day_chat_id,
            media=_frame_media,
            tool_events=_frame_tools,
            # From the COMMITTED row, not from the request body: the wire form
            # of an attachment is message-scoped (download_url / preview_url /
            # thumb_url / kind) and the body carries none of it. Handing the raw
            # dicts to the client rendered a file produced during a voice turn
            # as a dead card until the next history refetch — the sibling writer
            # (voice_tasks._persist_message) has always passed enriched dicts.
            attachments=(
                _serialize_attachments(msg)
                if (attachments
                    or (existing is not None and getattr(msg, "attachments", None)))
                else None
            ),
            app_artifact=_frame_app,
            # The caller's monotonic counter for this key. Without it an open
            # ChatScreen de-dupes the corrected row away and keeps the first
            # fragment of the sentence until the next full refetch.
            revision=revision,
        )
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

    ``conversation_channels`` is ``{conversation_id: channel}`` — the
    FALLBACK for a row whose own ``Message.channel`` is NULL. Message first,
    Conversation second, then "web": the same order ``api/day_chats.py``
    resolves, which is what makes this route and the primary one agree on
    every row. (The old docstring justified Conversation-first by saying
    ``Message.channel`` is only written for system senders. That was already
    untrue for runner-saved rows, and it stopped being true altogether once
    the ws_chat presave stamped the channel on both inserts — a WhatsApp turn
    and a mobile turn can share a Conversation, and only the row knows which
    it is.)
    """
    # Local import: sessions ↔ day_chats would cycle at module load.
    from app.api.day_chats import _serialize_meta_card, _serialize_tool_events

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

    # Read ONCE, above the payload: the body's projection and the `voice` key
    # itself must be built from the same object, or a row could be blanked
    # while its provenance says it was heard (or the reverse).
    _voice = _serialize_meta_card(message, "voice")

    resp = dict(
        id=message.id,
        role=message.role,
        source=message.source,
        background=message.source == "attachment_analysis",
        # Through the guards, not raw — TWO of them, and neither subsumes the
        # other. `public_text` answers "may this ROLE's body be rendered at
        # all": a marker row must never reach a client as text on ANY of the
        # four readers, and this one only ever blanked the role it knew about
        # (see api/message_cards.py). `public_heard_text` answers "did the
        # caller hear any of it" for a voice transcript row whose stored text
        # the persistence API would not let the relay write empty (see
        # schemas.py). Composed, never either alone.
        content=public_heard_text(
            public_text(message.role, message.content), _voice,
        ),
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
        # R30 §4.10: the once-per-run notification card — same parity
        # rule; the ONLY automation presence in the main chat (D-05).
        automation_notification=(
            msg_metadata.get("automation_notification")
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
        # Voice provenance (R48 §8, contract v0.2 `message.voice`). Until now
        # this key was WRITE-ONLY end to end: the relay minted it,
        # `_clean_voice` allowlisted it, `_build_metadata` persisted it — and
        # not one reader sent it back, so the contract was false at its very
        # first read and the app could not see the field at all. Read through
        # the same helper the other three readers use, so the isinstance guard
        # and the absent→null behaviour are one implementation; same parity
        # rule as every neighbour above (api/day_chats.py and
        # api/messages_recover.py carry this key too, or it disappears the
        # moment a client takes its fallback path). NULL on every legacy row,
        # which is every row not written by the voice relay. Serialized
        # UNPROJECTED even when the body above was blanked: `heard_chars`,
        # `interrupted` and `record_kind` are how the client renders the empty
        # row honestly instead of as an unexplained blank bubble.
        voice=_voice,
        attachments=attachments_list,
        channel=(
            getattr(message, "channel", None)
            or (conversation_channels or {}).get(message.conversation_id)
        ),
        reply_to_message_id=getattr(message, "reply_to_message_id", None),
        reply_to=(reply_targets or {}).get(message.id),
        # Turn identity (round 46, C1). Same parity rule as every neighbour
        # above: the clients fall back from /api/day-chats to these session
        # routes, and a field emitted by only one serializer vanishes on the
        # fallback — here that would mean the thread silently losing the
        # ability to pair its own optimistic row and going back to matching
        # on byte-equal content. NULL is UNKNOWN, never "not mine".
        client_msg_id=getattr(message, "client_msg_id", None),
        occurred_at=getattr(message, "occurred_at", None),
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
    Get all messages for all sessions on a specific date (YYYY-MM-DD),
    ordered chronologically. `tz_offset` adjusts the date boundary to
    match the client's local day.

    **A day is bounded by its MESSAGES, not by a session count.**

    This used to take the 10 most recently-UPDATED conversations of the
    day and return their messages. That is a silent, moving truncation:
    a day is a fixed set of rows, and which of them you could see
    depended on how many conversations had been touched since — so a
    row that was there an hour ago is gone now, with no error and no
    marker, and nothing about the response says a session was dropped.

    R31-D caught it on 2026-08-26 with two captures of the same date at
    the same offset: 30 rows at 17:35Z, 19 at 18:47Z, 11 dropped and 0
    added. The eleven were all three `automation_notification` cards and
    eight rows of ordinary conversation about reminders — the founder
    talking to his agent, invisible in his own day. They were never
    deleted: `/api/day-chats/{date}/messages`, which does not have this
    limit, returned all 51 rows throughout.

    Automations made it VISIBLE rather than caused it: each automation
    thread mints a conversation, so an active day now has more than ten,
    and `updated_at DESC` puts the freshly-touched automation rows at
    the top and pushes the user's own morning off the end.

    So the session cap is gone. The `limit` parameter — which the caller
    controls and which already bounds the response — does the bounding,
    against the day's messages directly.
    """
    from datetime import date as date_type, timedelta

    try:
        target_date = date_type.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Try proxying to remote agent first (pass tz_offset through)
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        # This route IS the mobile client's fallback when /api/day-chats fails.
        # Answering it with a silent empty list is how one slow tenant became
        # "my messages were deleted".
        try:
            data = await _proxy_sessions(
                proxy[0], proxy[1], f"by-date/{date_str}/messages",
                {"limit": limit, "tz_offset": tz_offset},
            )
        except SessionsAgentSaidNo as e:
            raise _sessions_4xx(e)
        except SessionsAgentUnreachable as e:
            raise _sessions_unreachable(e)
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
        # NEWEST `limit`, re-sorted ascending for the client.
        #
        # With the session cap gone a busy day can exceed `limit`, and
        # `ORDER BY created_at ASC LIMIT n` would then return the day's
        # FIRST n rows — so the user's most recent messages become the
        # ones that disappear, which is the worse half of the same
        # defect. Take the tail and reverse it.
        messages_result = await db.execute(
            select(Message)
            .where(Message.conversation_id.in_(session_ids))
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
    except _MISSING_TABLE_ERRORS:
        await db.rollback()
        return JSONResponse(content=[])
    messages = list(reversed(messages_result.scalars().all()))

    # Enrich with build job data
    build_jobs = await load_build_jobs(db, messages)

    from app.agent.reply_quote import resolve_reply_targets_for_serialization
    reply_targets = await resolve_reply_targets_for_serialization(db, messages)

    # R31-D: say when the answer is clipped.
    #
    # The session cap this route used to carry was invisible — the
    # response looked complete whether or not it was, which is what let
    # eleven rows go missing for hours without anyone reading an error.
    # Bounding by `limit` instead of by session count is correct, but it
    # has exactly the same property: at 500 rows a busy day truncates
    # and the body still says nothing about it. D's census now prints
    # `<-- AT LIMIT` for that reason; a consumer should not have to
    # infer it from the row count matching the parameter.
    #
    # Headers rather than a wrapper object: the body is a bare array and
    # four clients read it that way, so a shape change here is a
    # migration. `X-Truncated` is loud for anything that looks and inert
    # for anything that does not.
    truncated = len(messages) >= limit
    if truncated:
        logger.warning(
            "[sessions] by-date TRUNCATED user=%s date=%s limit=%d — the "
            "day has more rows than were returned",
            str(current_user.id)[:8], date_str, limit,
        )
    return JSONResponse(
        content=[
            r.model_dump(mode="json") for r in attach_run_to_cards([
                _message_to_response(m, build_jobs, reply_targets,
                                     conversation_channels)
                for m in messages
            ])
        ],
        headers={
            "X-Returned-Count": str(len(messages)),
            "X-Truncated": "true" if truncated else "false",
        },
    )
