"""
Public API v1 — Programmatic access to HexBrain via API keys.

Endpoints:
  POST /api/v1/chat           — Send a message and get a response
  POST /api/v1/chat/stream    — Send a message and stream response (SSE)
  GET  /api/v1/sessions       — List sessions
  GET  /api/v1/sessions/{id}  — Get session messages
  POST /api/v1/memories/search — Search memories
  GET  /api/v1/skills         — List loaded skills

  POST /api/v1/keys           — Create a new API key
  GET  /api/v1/keys           — List your API keys
  DELETE /api/v1/keys/{id}    — Revoke an API key

Authentication:
  Header: Authorization: Bearer hx_...
  API keys are prefixed with "hx_" and hashed with SHA-256 for storage.

Rate limiting:
  Per-key configurable, default 60 requests/minute.
  Tracked in-memory with sliding window.
"""

import hashlib
import json
import logging
import secrets
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db, async_session_maker
from app.db.models import ApiKey, Conversation, Message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Public API v1"])

# References set at startup
_agent_runner = None
_skill_loader = None


def set_api_v1_refs(agent_runner, skill_loader=None):
    """Set references to the agent runner and skill loader (called from main.py lifespan)."""
    global _agent_runner, _skill_loader
    _agent_runner = agent_runner
    _skill_loader = skill_loader


# ======================================================================
# Rate limiter (in-memory sliding window)
# ======================================================================

_rate_windows: Dict[str, List[float]] = defaultdict(list)
_RATE_WINDOW_SECONDS = 60


def _check_rate_limit(key_id: str, limit: int) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    window = _rate_windows[key_id]

    # Remove timestamps outside the window
    cutoff = now - _RATE_WINDOW_SECONDS
    _rate_windows[key_id] = [t for t in window if t > cutoff]
    window = _rate_windows[key_id]

    if len(window) >= limit:
        return False

    window.append(now)
    return True


# ======================================================================
# Auth dependency
# ======================================================================

def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def get_api_key_user(request: Request, db: AsyncSession = Depends(get_db)) -> str:
    """
    Dependency: Extract API key from Authorization header, validate, rate-limit.
    Returns user_id.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer hx_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Format: Authorization: Bearer hx_...",
        )

    raw_key = auth.removeprefix("Bearer ").strip()
    key_hash = _hash_key(raw_key)

    result = await db.execute(
        select(ApiKey).where(
            and_(
                ApiKey.key_hash == key_hash,
                ApiKey.is_active == True,
            )
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    # Check expiration
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired")

    # Rate limit
    if not _check_rate_limit(api_key.id, api_key.rate_limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({api_key.rate_limit}/min)",
        )

    # Update last_used
    api_key.last_used_at = datetime.utcnow()
    await db.commit()

    return api_key.user_id


# ======================================================================
# Request/Response schemas
# ======================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    session_id: Optional[str] = None
    model: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    session_id: str
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    model: str = ""
    tool_calls: int = 0
    processing_time_ms: int = 0


class SessionSummary(BaseModel):
    id: str
    channel: str
    is_active: bool
    message_count: int
    total_tokens: int
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str
    model_used: Optional[str] = None


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    brain_type: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    rate_limit: int = Field(default=60, ge=1, le=1000)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class KeyOut(BaseModel):
    id: str
    name: str
    key_prefix: str
    rate_limit: int
    is_active: bool
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: str


class CreateKeyResponse(BaseModel):
    key: str  # Only returned on creation
    id: str
    name: str
    key_prefix: str


# ======================================================================
# Chat endpoints
# ======================================================================

@router.post("/chat", response_model=ChatResponse)
async def api_chat(
    req: ChatRequest,
    user_id: str = Depends(get_api_key_user),
):
    """Send a message to HexBrain and get a response."""
    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        response = await _agent_runner.run(
            user_message=req.message,
            user_id=user_id,
            session_id=req.session_id,
            model_override=req.model,
        )

        return ChatResponse(
            text=response.text,
            session_id=response.session_id,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            tokens_total=response.tokens_total,
            model=response.model,
            tool_calls=len(response.tool_calls),
            processing_time_ms=response.processing_time_ms,
        )
    except Exception as e:
        logger.exception(f"API chat error for user {user_id}")
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")


@router.post("/chat/stream")
async def api_chat_stream(
    req: ChatRequest,
    user_id: str = Depends(get_api_key_user),
):
    """Send a message and stream the response as Server-Sent Events."""
    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")

    async def generate():
        try:
            async def on_text_chunk(chunk: str):
                data = json.dumps({"type": "text_chunk", "text": chunk})
                yield f"data: {data}\n\n"

            async def on_tool_start(tool_name: str):
                data = json.dumps({"type": "tool_start", "tool": tool_name})
                yield f"data: {data}\n\n"

            async def on_tool_end(tool_name: str, summary: str):
                data = json.dumps({"type": "tool_end", "tool": tool_name, "summary": summary})
                yield f"data: {data}\n\n"

            # We need to collect chunks for SSE because callbacks are coroutines
            chunks: list[str] = []
            tool_events: list[dict] = []

            async def collect_text(chunk: str):
                chunks.append(chunk)

            async def collect_tool_start(tool_name: str):
                tool_events.append({"type": "tool_start", "tool": tool_name})

            async def collect_tool_end(tool_name: str, summary: str):
                tool_events.append({"type": "tool_end", "tool": tool_name, "summary": summary})

            response = await _agent_runner.run(
                user_message=req.message,
                user_id=user_id,
                session_id=req.session_id,
                model_override=req.model,
                on_text_chunk=collect_text,
                on_tool_start=collect_tool_start,
                on_tool_end=collect_tool_end,
            )

            # Emit collected events
            for event in tool_events:
                yield f"data: {json.dumps(event)}\n\n"

            # Emit final result
            done = {
                "type": "done",
                "text": response.text,
                "session_id": response.session_id,
                "tokens_input": response.tokens_input,
                "tokens_output": response.tokens_output,
                "model": response.model,
                "tool_calls": len(response.tool_calls),
                "processing_time_ms": response.processing_time_ms,
            }
            yield f"data: {json.dumps(done)}\n\n"

        except Exception as e:
            error = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ======================================================================
# Sessions
# ======================================================================

@router.get("/sessions", response_model=List[SessionSummary])
async def api_list_sessions(
    limit: int = 20,
    active_only: bool = False,
    user_id: str = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
):
    """List conversation sessions."""
    query = select(Conversation).where(Conversation.user_id == user_id)
    if active_only:
        query = query.where(Conversation.is_active == True)
    query = query.order_by(Conversation.updated_at.desc()).limit(limit)

    result = await db.execute(query)
    sessions = result.scalars().all()

    return [
        SessionSummary(
            id=s.id,
            channel=s.channel or "api",
            is_active=s.is_active,
            message_count=s.message_count,
            total_tokens=s.total_tokens,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
async def api_session_messages(
    session_id: str,
    limit: int = 50,
    user_id: str = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
):
    """Get messages from a specific session."""
    # Verify ownership
    result = await db.execute(
        select(Conversation).where(
            and_(Conversation.id == session_id, Conversation.user_id == user_id)
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = list(reversed(result.scalars().all()))

    return [
        MessageOut(
            role=m.role,
            content=m.content,
            created_at=m.created_at.isoformat(),
            model_used=m.model_used,
        )
        for m in messages
    ]


# ======================================================================
# Memory search
# ======================================================================

@router.post("/memories/search")
async def api_memory_search(
    req: MemorySearchRequest,
    user_id: str = Depends(get_api_key_user),
):
    """Search memories via the API."""
    try:
        from app.services.embedding_service import get_embedding_service
        from app.services.memory_service import MemoryService

        emb = get_embedding_service()
        embedding = emb.embed(req.query)

        async with async_session_maker() as db:
            svc = MemoryService(db)
            results = await svc.search_memories_by_embedding(
                user_id=user_id,
                embedding=embedding,
                limit=req.limit,
                min_similarity=0.1,
                brain_types=[req.brain_type] if req.brain_type else None,
            )

        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


# ======================================================================
# Skills
# ======================================================================

@router.get("/skills")
async def api_list_skills(user_id: str = Depends(get_api_key_user)):
    """List all loaded skills and their tools."""
    if not _skill_loader:
        return {"skills": [], "count": 0}

    return {
        "skills": _skill_loader.get_summary(),
        "count": _skill_loader.loaded_count,
    }


# ======================================================================
# API Key management (uses JWT auth, not API key auth)
# ======================================================================

async def _get_jwt_user(request: Request, db: AsyncSession = Depends(get_db)) -> str:
    """Get user from JWT token (for key management endpoints)."""
    from app.api.auth import get_current_user
    user = await get_current_user(
        credentials=request.headers.get("Authorization", "").removeprefix("Bearer "),
        db=db,
    )
    return user.id


@router.post("/keys", response_model=CreateKeyResponse)
async def create_api_key(
    req: CreateKeyRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_api_key_user),
):
    """Create a new API key. The raw key is only returned once."""
    # Generate key
    raw_key = f"hx_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    key_prefix = raw_key[:10]

    expires_at = None
    if req.expires_in_days:
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(days=req.expires_in_days)

    api_key = ApiKey(
        user_id=user_id,
        name=req.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        rate_limit=req.rate_limit,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return CreateKeyResponse(
        key=raw_key,
        id=api_key.id,
        name=api_key.name,
        key_prefix=key_prefix,
    )


@router.get("/keys", response_model=List[KeyOut])
async def list_api_keys(
    user_id: str = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
):
    """List your API keys (without the actual key values)."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()

    return [
        KeyOut(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            rate_limit=k.rate_limit,
            is_active=k.is_active,
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
            created_at=k.created_at.isoformat(),
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user_id: str = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke (deactivate) an API key."""
    result = await db.execute(
        select(ApiKey).where(
            and_(ApiKey.id == key_id, ApiKey.user_id == user_id)
        )
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.is_active = False
    await db.commit()

    return {"status": "revoked", "id": key_id}
