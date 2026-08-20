"""
Chat API - Main chat orchestration endpoint

This is the core of the Toup Agent Platform.
The chat endpoint orchestrates:
1. Session management (create/continue conversations)
2. Memory retrieval (find relevant memories)
3. Prompt building (system prompt from identities + context)
4. LLM completion (generate response)
5. Memory extraction (learn from conversation)
6. Token tracking (usage analytics)
"""

import json
import time
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from app.db import get_db, Conversation, Message, Memory, User
from app.schemas import (
    ChatRequest, ChatResponse, ChatMessageResponse, MemoryWithScore, MemoryResponse,
    BrainType, MemoryCreate, MemoryType, MemoryLevel
)
from app.api.auth import get_current_user
from app.services.llm_service import get_llm_service
from app.services.prompt_builder import get_prompt_builder
from app.services.memory_service import MemoryService
from app.services.embedding_service import get_embedding_service
from app.config import settings

from app.services.memory_log import describe_memory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Main chat endpoint - send a message and get an AI response.
    
    This endpoint orchestrates the entire chat flow:
    1. Creates or retrieves a session
    2. Retrieves relevant memories based on the query
    3. Builds a system prompt from identities and memories
    4. Generates a response using the LLM
    5. Extracts memories from the conversation
    6. Saves everything and returns the response
    
    Features:
    - Automatic memory retrieval from all brain types
    - Identity-based system prompts (soul, user profile, instructions)
    - Automatic memory extraction from conversations
    - Token tracking for cost analysis
    - Session continuity with message history
    - **Streaming support**: Set `stream: true` for Server-Sent Events
    """
    # If streaming requested, use streaming endpoint
    if request.stream:
        return await _chat_stream(request, current_user, db)
    
    return await _chat_complete(request, current_user, db)


async def _chat_complete(
    request: ChatRequest,
    current_user: User,
    db: AsyncSession
) -> ChatResponse:
    """Non-streaming chat completion"""
    start_time = time.time()
    
    # Initialize services
    llm_service = get_llm_service()
    prompt_builder = get_prompt_builder()
    embedding_service = get_embedding_service()
    memory_service = MemoryService(db)
    
    # 1. Get or create session
    session, is_new_session = await _get_or_create_session(
        db=db,
        user_id=current_user.id,
        session_id=request.session_id
    )
    
    # 2. Get conversation history
    history = await _get_conversation_history(
        db=db,
        session_id=session.id,
        max_messages=settings.max_history_messages
    )
    
    # 3. Retrieve relevant memories (if enabled)
    retrieved_memories: List[dict] = []
    memory_ids: List[str] = []
    
    if request.include_memories and request.memory_limit > 0:
        retrieved_memories = await _retrieve_memories(
            db=db,
            user_id=current_user.id,
            query=request.message,
            memory_service=memory_service,
            embedding_service=embedding_service,
            limit=request.memory_limit,
            min_similarity=request.min_similarity,
            brain_types=request.brain_types
        )
        memory_ids = [m.get("id") for m in retrieved_memories]
    
    # 4. Build messages with system prompt
    messages = await prompt_builder.build_messages(
        user_id=current_user.id,
        db=db,
        user_message=request.message,
        conversation_history=history,
        memories=retrieved_memories,
        max_history_messages=settings.max_history_messages
    )
    
    # 5. Generate LLM response
    model = request.model or settings.default_model
    temperature = request.temperature if request.temperature is not None else settings.temperature
    max_tokens = request.max_tokens or settings.max_tokens
    
    try:
        llm_response = await llm_service.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"LLM completion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {str(e)}"
        )
    
    # 6. Calculate processing time
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    # 7. Save user message
    user_msg = Message(
        conversation_id=session.id,
        role="user",
        content=request.message
    )
    db.add(user_msg)
    
    # 8. Save assistant message with metadata
    assistant_msg = Message(
        conversation_id=session.id,
        role="assistant",
        content=llm_response.content,
        tokens_prompt=llm_response.tokens_prompt,
        tokens_completion=llm_response.tokens_completion,
        model_used=llm_response.model,
        memories_retrieved_json=json.dumps(memory_ids) if memory_ids else None,
        processing_time_ms=processing_time_ms
    )
    db.add(assistant_msg)
    
    # 9. Update session stats
    session.message_count += 2
    session.total_tokens += llm_response.tokens_total
    session.updated_at = datetime.utcnow()
    
    # 10. Memory: nothing is extracted here (v3 §2.1.6).
    #
    # Same excision as app/api/chat.py — see the note there. This module is
    # additionally not mounted by ANY entrypoint (agent_main mounts
    # app.api.chat; platform_main mounts neither), so it was a dormant fourth
    # writer rather than a live one.
    extracted_memories: List[MemoryResponse] = []

    # 11. Commit all changes
    await db.commit()
    await db.refresh(session)
    await db.refresh(assistant_msg)
    
    # 12. Build response
    return ChatResponse(
        session_id=session.id,
        message_id=assistant_msg.id,
        response=llm_response.content,
        tokens_prompt=llm_response.tokens_prompt,
        tokens_completion=llm_response.tokens_completion,
        tokens_total=llm_response.tokens_total,
        model_used=llm_response.model,
        processing_time_ms=processing_time_ms,
        memories_retrieved=[
            MemoryWithScore(
                id=m.get("id"),
                content=m.get("content", ""),
                summary=m.get("summary"),
                brain_type=m.get("brain_type", "user"),
                category=m.get("category", "context"),
                memory_type=m.get("memory_type", "fact"),
                importance=m.get("importance", 0.5),
                confidence=m.get("confidence", 1.0),
                created_at=m.get("created_at", datetime.utcnow()),
                updated_at=m.get("updated_at", datetime.utcnow()),
                last_accessed_at=m.get("last_accessed_at"),
                access_count=m.get("access_count", 0),
                source_type=m.get("source_type", "api"),
                similarity_score=m.get("similarity_score", 0.0)
            )
            for m in retrieved_memories
        ],
        memories_extracted=extracted_memories,
        is_new_session=is_new_session,
        session_message_count=session.message_count
    )


async def _chat_stream(
    request: ChatRequest,
    current_user: User,
    db: AsyncSession
) -> StreamingResponse:
    """
    Streaming chat completion using Server-Sent Events (SSE).
    
    Event types:
    - "memories": Retrieved memories (sent first)
    - "content": Token chunks from LLM
    - "done": Final message with metadata
    - "error": Error message
    """
    import asyncio
    
    async def _stream_body(db):
        start_time = time.time()
        full_response = ""
        
        try:
            # Initialize services
            llm_service = get_llm_service()
            prompt_builder = get_prompt_builder()
            embedding_service = get_embedding_service()
            memory_service = MemoryService(db)
            
            # 1. Get or create session
            session, is_new_session = await _get_or_create_session(
                db=db,
                user_id=current_user.id,
                session_id=request.session_id
            )
            
            # 2. Get conversation history
            history = await _get_conversation_history(
                db=db,
                session_id=session.id,
                max_messages=settings.max_history_messages
            )
            
            # 3. Retrieve memories
            retrieved_memories: List[dict] = []
            memory_ids: List[str] = []
            
            if request.include_memories and request.memory_limit > 0:
                retrieved_memories = await _retrieve_memories(
                    db=db,
                    user_id=current_user.id,
                    query=request.message,
                    memory_service=memory_service,
                    embedding_service=embedding_service,
                    limit=request.memory_limit,
                    min_similarity=request.min_similarity,
                    brain_types=request.brain_types
                )
                memory_ids = [m.get("id") for m in retrieved_memories]
                
                # Send memories event first
                if retrieved_memories:
                    memories_data = json.dumps({
                        "type": "memories",
                        "data": retrieved_memories[:5]  # Send top 5 for display
                    })
                    yield f"data: {memories_data}\n\n"
            
            # 4. Build messages
            messages = await prompt_builder.build_messages(
                user_id=current_user.id,
                db=db,
                user_message=request.message,
                conversation_history=history,
                memories=retrieved_memories,
                max_history_messages=settings.max_history_messages
            )
            
            # 5. Stream LLM response
            model = request.model or settings.default_model
            temperature = request.temperature if request.temperature is not None else settings.temperature
            max_tokens = request.max_tokens or settings.max_tokens
            
            # Save user message first
            user_msg = Message(
                conversation_id=session.id,
                role="user",
                content=request.message
            )
            db.add(user_msg)
            await db.flush()
            
            # Stream tokens
            async for chunk in llm_service.stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                if chunk.content:
                    full_response += chunk.content
                    chunk_data = json.dumps({
                        "type": "content",
                        "data": chunk.content
                    })
                    yield f"data: {chunk_data}\n\n"
                
                if chunk.is_final:
                    break
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Estimate tokens (since streaming doesn't give exact counts)
            prompt_tokens = llm_service.count_message_tokens(messages)
            completion_tokens = llm_service.count_tokens(full_response)
            
            # Save assistant message
            assistant_msg = Message(
                conversation_id=session.id,
                role="assistant",
                content=full_response,
                tokens_prompt=prompt_tokens,
                tokens_completion=completion_tokens,
                model_used=model,
                memories_retrieved_json=json.dumps(memory_ids) if memory_ids else None,
                processing_time_ms=processing_time_ms
            )
            db.add(assistant_msg)
            
            # Update session stats
            session.message_count += 2
            session.total_tokens += prompt_tokens + completion_tokens
            session.updated_at = datetime.utcnow()
            
            await db.commit()
            
            # Memory extraction was here; see the note on the sync route.

            # Send done event with metadata
            done_data = json.dumps({
                "type": "done",
                "data": {
                    "session_id": session.id,
                    "message_id": assistant_msg.id,
                    "tokens_prompt": prompt_tokens,
                    "tokens_completion": completion_tokens,
                    "tokens_total": prompt_tokens + completion_tokens,
                    "model_used": model,
                    "processing_time_ms": processing_time_ms,
                    "is_new_session": is_new_session,
                    "memories_count": len(retrieved_memories)
                }
            })
            yield f"data: {done_data}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_data = json.dumps({
                "type": "error",
                "data": str(e)
            })
            yield f"data: {error_data}\n\n"
    
    async def generate_stream():
        """Own the session — see the note in app/api/chat.py.

        This module is an unmounted duplicate of app/api/chat.py, so the same
        defect here is latent rather than live. Fixed anyway: a dormant copy of
        a known leak is a regression waiting for whoever mounts it.
        """
        from app.db.database import async_session_maker as _own_maker
        async with _own_maker() as _own_db:
            async for _evt in _stream_body(_own_db):
                yield _evt

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


async def _get_or_create_session(
    db: AsyncSession,
    user_id: str,
    session_id: Optional[str] = None
) -> tuple[Conversation, bool]:
    """Get existing session or create a new one."""
    is_new = False
    
    if session_id:
        # Try to find existing session
        query = select(Conversation).where(
            and_(
                Conversation.id == session_id,
                Conversation.user_id == user_id
            )
        )
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if session:
            # Start a new session if the day changed (daily rotation)
            if session.started_at:
                from datetime import datetime, timezone
                now_utc = datetime.now(timezone.utc)
                started = session.started_at.replace(tzinfo=timezone.utc) if session.started_at.tzinfo is None else session.started_at
                if started.date() != now_utc.date():
                    logger.info(f"Session {session_id} is from {started.date()}, creating new session for today")
                else:
                    return session, False
            else:
                return session, False
        else:
            # Session not found, create new
            logger.warning(f"Session {session_id} not found, creating new")
    
    # Create new session
    session = Conversation(
        user_id=user_id,
        title=None,  # Will be set later or by user
        channel="api",
        is_active=True
    )
    db.add(session)
    await db.flush()  # Get the ID
    
    return session, True


async def _get_conversation_history(
    db: AsyncSession,
    session_id: str,
    max_messages: int = 20
) -> List[dict]:
    """Get recent conversation history for context."""
    query = (
        select(Message)
        .where(Message.conversation_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(max_messages)
    )
    
    result = await db.execute(query)
    messages = list(reversed(result.scalars().all()))  # Oldest first
    
    return [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]


async def _retrieve_memories(
    db: AsyncSession,
    user_id: str,
    query: str,
    memory_service: "MemoryService",
    embedding_service,
    limit: int = 15,
    min_similarity: float = 0.1,
    brain_types: Optional[List[BrainType]] = None
) -> List[dict]:
    """Retrieve relevant memories for the query."""
    try:
        # Generate embedding for query (sync method)
        query_embedding = embedding_service.embed(query)
        
        # Build brain type filter
        brain_type_values = None
        if brain_types:
            brain_type_values = [bt.value for bt in brain_types]
        
        # Search memories
        memories = await memory_service.search_memories_by_embedding(
            user_id=user_id,
            embedding=query_embedding,
            limit=limit,
            min_similarity=min_similarity,
            brain_types=brain_type_values
        )
        
        return memories
        
    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}")
        return []


@router.get("/history/{session_id}", response_model=List[ChatMessageResponse])
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get chat history for a session."""
    # Verify ownership
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
        .limit(limit)
    )
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    return [
        ChatMessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at,
            tokens_prompt=msg.tokens_prompt,
            tokens_completion=msg.tokens_completion,
            model_used=msg.model_used,
            memories_retrieved=json.loads(msg.memories_retrieved_json) if msg.memories_retrieved_json else None,
            processing_time_ms=msg.processing_time_ms
        )
        for msg in messages
    ]
