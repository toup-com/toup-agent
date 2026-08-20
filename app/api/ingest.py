"""Ingestion endpoints — store a conversation. They do NOT write memory.

v3 (docs/memory/rebuild-2026-08-v3.md §2.1): these two routes used to be the
sixth writer, running the extractor's rule-based half and calling
`MemoryService.create_memory` directly. Both extraction blocks are severed;
what remains is conversation + message storage with embeddings, which is what
an external ingestion client is actually importing. See the long note inside
`ingest_message` for the callers checked and why the response shape is kept.

TOPOLOGY (see app/api/tenant_proxy.py)
    Identical defect and identical fix to `app/api/documents.py`, which is
    mounted one line below this router in platform_main.py. `conversations`,
    `messages`, `memories`, `entities` and `entity_links` are ALL AGENT_ONLY:
    platform-mode `init_db` never creates them and no alembic revision does
    either, so before this router was proxied `POST /api/ingest/message` raised
    `UndefinedTableError: relation "conversations" does not exist` on the only
    app that mounted it.

    Registered on BOTH apps now. On the platform each handler forwards to the
    user's agent with X-Agent-Key; on the agent `serving_locally()` is true and
    the handlers run against the tenant store the agent actually reads.
"""

import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db import get_db, Conversation, Message
from app.schemas import (
    IngestMessageRequest, IngestConversationRequest, IngestResponse,
)
from app.api.auth import get_current_user
from app.api.tenant_proxy import agent_proxy_info, proxy_to_agent
from app.services import get_embedding_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("/message", response_model=IngestResponse)
async def ingest_message(
    request: IngestMessageRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Store one conversation turn (user message + assistant response).

    Memory is NOT written here — the curator writes it from the agent's own
    turn. `memories_extracted` is always 0.
    """
    # Route to the tenant BEFORE touching this session — every table below is
    # AGENT_ONLY. No local fallback: a turn "ingested" into the platform DB is
    # invisible to the agent's hybrid_search forever.
    proxy = await agent_proxy_info(current_user.id, db)
    if proxy:
        data = await proxy_to_agent(
            proxy[0], proxy[1], "ingest/message", "POST",
            json_body=request.model_dump(mode="json", exclude_none=True),
        )
        if data is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Your agent ingested the message but returned no details.",
            )
        return JSONResponse(content=data)

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
    
    # ── NO EXTRACTION HERE (v3 §2.1) ─────────────────────────────────
    #
    # This route was the SIXTH writer, and it survived the convergence pass
    # because it does not import anything the retirement grep looked for: it
    # calls the extractor's RULE-BASED half (`extract_memories`, regexes, no
    # LLM) and writes through `MemoryService.create_memory` directly. It was
    # still minting `MemoryCategory.ACTIVE_TASK` rows — the category whose
    # entire surface v3 deleted — with `extract_memories` defaulting to True
    # on a route mounted by BOTH agent_main and platform_main.
    #
    # It cannot feed a memory FILE, so this was never a correctness bug in
    # the product; it was a truth bug in the invariant. "Every producer
    # converges on the curator" has to be true of every producer, and
    # `tests/test_curator_producers.py` now walks the AST of app/ so this
    # class cannot hide again.
    #
    # Callers checked before severing: `frontend/src/store.ts`'s
    # `ingestDemoMessage` (zero call sites — the demo helper it fed is gone)
    # and `backend/app/scripts/demo_mode.py` (an operator script). No mobile
    # caller, no production client.
    #
    # MESSAGE STORAGE STAYS. Same treatment as the REST /chat routes: this
    # endpoint's other half — conversation + messages + embeddings — is
    # genuinely useful and is what an external ingestion client is actually
    # importing. `memories_extracted` is now always 0 and `memories` always
    # empty; the response SHAPE is unchanged so an existing client does not
    # break, it just stops being told about rows nothing can read.

    await db.commit()
    
    return IngestResponse(
        conversation_id=conversation_id,
        messages_ingested=2,
        memories_extracted=0,
        entities_extracted=0,
        memories=[],
    )


@router.post("/conversation", response_model=IngestResponse)
async def ingest_conversation(
    request: IngestConversationRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Store a full conversation with multiple messages.

    Memory is NOT written here — see `ingest_message`.
    """
    proxy = await agent_proxy_info(current_user.id, db)
    if proxy:
        data = await proxy_to_agent(
            proxy[0], proxy[1], "ingest/conversation", "POST",
            json_body=request.model_dump(mode="json", exclude_none=True),
        )
        if data is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Your agent ingested the conversation but returned no details.",
            )
        return JSONResponse(content=data)

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
    
    # Process and store the messages. No extraction — see /ingest/message.
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

    await db.commit()
    
    return IngestResponse(
        conversation_id=conversation.id,
        messages_ingested=len(messages_stored),
        memories_extracted=0,
        entities_extracted=0,
        memories=[],
    )

# `_get_or_create_entity` was deleted with the extraction blocks: its only two
# callers were the entity fan-out inside them. Entities and their links are
# still written by the conversation path's own extractor
# (`memory_service.store_entity_relationship`, MCP `entity_relationship_create`)
# — this route never needed a second, private way to make one.
