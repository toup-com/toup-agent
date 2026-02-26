"""Ingestion pipeline endpoints - process conversations and extract memories"""

import json
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db import get_db, Conversation, Message, Memory, Entity, EntityLink
from app.schemas import (
    IngestMessageRequest, IngestConversationRequest, IngestResponse,
    MemoryCreate, MemoryResponse, MemoryCategory, MemoryType
)
from app.api.auth import get_current_user
from app.api.memories import memory_to_response
from app.services import get_memory_extractor, get_embedding_service
from app.services.memory_service import MemoryService

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("/message", response_model=IngestResponse)
async def ingest_message(
    request: IngestMessageRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest a single conversation turn (user message + assistant response).
    Extracts memories, entities, and creates embeddings.
    """
    memory_service = MemoryService(db)
    extractor = get_memory_extractor()
    embedding_service = get_embedding_service()
    
    # Get or create conversation
    conversation_id = request.conversation_id
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    else:
        # Create new conversation
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            title=request.user_message[:100] if len(request.user_message) > 100 else request.user_message,
            started_at=datetime.utcnow(),
        )
        db.add(conversation)
        await db.flush()
        conversation_id = conversation.id
    
    # Store messages
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="user",
        content=request.user_message,
        embedding_json=embedding_service.embed_to_json(request.user_message),
    )
    db.add(user_msg)
    
    assistant_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="assistant",
        content=request.assistant_response,
        embedding_json=embedding_service.embed_to_json(request.assistant_response),
    )
    db.add(assistant_msg)
    
    # Update conversation message count
    conversation.message_count += 2
    await db.flush()
    
    memories_created = []
    entities_created = 0
    
    if request.extract_memories:
        # Extract memories from the conversation turn
        extracted = extractor.extract_memories(
            request.user_message,
            request.assistant_response
        )
        
        for ext_memory in extracted:
            # Create memory
            memory_data = MemoryCreate(
                content=ext_memory.content,
                summary=ext_memory.summary,
                category=ext_memory.category,
                memory_type=ext_memory.memory_type,
                importance=ext_memory.importance,
                confidence=ext_memory.confidence,
                tags=ext_memory.tags,
                metadata=ext_memory.metadata,
            )
            
            memory = await memory_service.create_memory(
                current_user.id,
                memory_data,
                source_message_id=user_msg.id
            )
            memories_created.append(memory)
            
            # Create/link entities
            for ent_data in ext_memory.entities:
                # Handle both dict and ExtractedEntity objects
                if hasattr(ent_data, 'name'):
                    ent_name = ent_data.name
                    ent_type = ent_data.entity_type
                else:
                    ent_name = ent_data["name"]
                    ent_type = ent_data["type"]
                    
                entity = await _get_or_create_entity(
                    db,
                    current_user.id,
                    ent_name,
                    ent_type,
                    embedding_service
                )
                
                # Link entity to memory
                link = EntityLink(
                    memory_id=memory.id,
                    entity_id=entity.id,
                    role="mentioned"
                )
                db.add(link)
                entities_created += 1
    
    await db.commit()
    
    return IngestResponse(
        conversation_id=conversation_id,
        messages_ingested=2,
        memories_extracted=len(memories_created),
        entities_extracted=entities_created,
        memories=[memory_to_response(m) for m in memories_created]
    )


@router.post("/conversation", response_model=IngestResponse)
async def ingest_conversation(
    request: IngestConversationRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest a full conversation with multiple messages.
    Processes all messages and extracts memories.
    """
    memory_service = MemoryService(db)
    extractor = get_memory_extractor()
    embedding_service = get_embedding_service()
    
    # Create conversation
    conversation = Conversation(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        title=request.title or (request.messages[0].content[:100] if request.messages else "Untitled"),
        started_at=datetime.utcnow(),
    )
    db.add(conversation)
    await db.flush()
    
    all_memories = []
    total_entities = 0
    
    # Process messages in pairs (user + assistant)
    messages_stored = []
    for msg_data in request.messages:
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation.id,
            role=msg_data.role,
            content=msg_data.content,
            embedding_json=embedding_service.embed_to_json(msg_data.content),
        )
        db.add(msg)
        messages_stored.append(msg)
    
    conversation.message_count = len(messages_stored)
    await db.flush()
    
    if request.extract_memories:
        # Process pairs of user-assistant messages
        for i in range(0, len(messages_stored) - 1, 2):
            if i + 1 < len(messages_stored):
                user_msg = messages_stored[i]
                assistant_msg = messages_stored[i + 1]
                
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    extracted = extractor.extract_memories(
                        user_msg.content,
                        assistant_msg.content
                    )
                    
                    for ext_memory in extracted:
                        memory_data = MemoryCreate(
                            content=ext_memory.content,
                            summary=ext_memory.summary,
                            category=ext_memory.category,
                            memory_type=ext_memory.memory_type,
                            importance=ext_memory.importance,
                            confidence=ext_memory.confidence,
                            tags=ext_memory.tags,
                            metadata=ext_memory.metadata,
                        )
                        
                        memory = await memory_service.create_memory(
                            current_user.id,
                            memory_data,
                            source_message_id=user_msg.id
                        )
                        all_memories.append(memory)
                        
                        for ent_data in ext_memory.entities:
                            entity = await _get_or_create_entity(
                                db,
                                current_user.id,
                                ent_data["name"],
                                ent_data["type"],
                                embedding_service
                            )
                            link = EntityLink(
                                memory_id=memory.id,
                                entity_id=entity.id,
                                role="mentioned"
                            )
                            db.add(link)
                            total_entities += 1
    
    await db.commit()
    
    return IngestResponse(
        conversation_id=conversation.id,
        messages_ingested=len(messages_stored),
        memories_extracted=len(all_memories),
        entities_extracted=total_entities,
        memories=[memory_to_response(m) for m in all_memories]
    )


async def _get_or_create_entity(
    db: AsyncSession,
    user_id: str,
    name: str,
    entity_type: str,
    embedding_service
) -> Entity:
    """Get existing entity or create new one"""
    result = await db.execute(
        select(Entity).where(
            Entity.user_id == user_id,
            Entity.name == name,
            Entity.entity_type == entity_type
        )
    )
    entity = result.scalar_one_or_none()
    
    if entity:
        entity.mention_count += 1
        entity.last_seen_at = datetime.utcnow()
    else:
        entity = Entity(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            entity_type=entity_type,
            embedding_json=embedding_service.embed_to_json(name),
        )
        db.add(entity)
    
    await db.flush()
    return entity
