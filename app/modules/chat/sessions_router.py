"""
Sessions API - Conversation session management

Sessions track conversation history with the agent.
Each session maintains:
- Message history (user + assistant messages)
- Token usage statistics
- Channel information (api, telegram, discord, web)
- Metadata for context
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload
from typing import Optional
import json

from app.db import get_db, Conversation, Message, User
from app.schemas import (
    SessionCreate, SessionResponse, SessionWithMessages, SessionListResponse,
    ChatMessageResponse
)
from app.api.auth import get_current_user

router = APIRouter(prefix="/sessions", tags=["sessions"])


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
        response_dict["messages"] = [_message_to_response(m) for m in messages]
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
    
    return [_message_to_response(m) for m in messages]


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


def _message_to_response(message: Message) -> ChatMessageResponse:
    """Convert Message model to ChatMessageResponse."""
    memories_retrieved = None
    if message.memories_retrieved_json:
        try:
            memories_retrieved = json.loads(message.memories_retrieved_json)
        except json.JSONDecodeError:
            memories_retrieved = None
    
    return ChatMessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        tokens_prompt=message.tokens_prompt,
        tokens_completion=message.tokens_completion,
        model_used=message.model_used,
        memories_retrieved=memories_retrieved,
        processing_time_ms=message.processing_time_ms
    )
