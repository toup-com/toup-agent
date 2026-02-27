"""
OpenAI Realtime API WebSocket Proxy — ChatGPT-speed voice conversation.

Architecture:
  Browser ←WS→ This Proxy ←WS→ OpenAI Realtime API (GPT-4o native audio)

The proxy:
  1. Authenticates the user and loads their OpenAI API key
  2. Builds system instructions from Identity docs + memories
  3. Relays PCM16 audio bidirectionally
  4. Handles function calls (all agent tools) server-side
  5. Persists user/assistant messages to DB (same as regular chat)

Protocol (Browser ↔ Proxy):
  Client sends:
    { "type": "audio", "data": "<base64 PCM16>" }             — mic audio chunk
    { "type": "stop" }                                          — end session
    { "type": "config", "voice": "nova", "session_id": "..." } — session config

  Server sends:
    { "type": "audio_delta", "data": "<base64 PCM16>" }  — assistant audio chunk
    { "type": "transcript", "text": "..." }               — what user said
    { "type": "response_text", "text": "..." }            — what assistant said (partial)
    { "type": "response_done", "text": "..." }            — full assistant text
    { "type": "session_id", "session_id": "..." }         — DB session ID (for chat sync)
    { "type": "speech_started" }                           — user started speaking (barge-in)
    { "type": "state", "state": "listening|thinking|speaking" }
    { "type": "ready" }                                    — session ready, start sending audio
    { "type": "error", "message": "..." }
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.agent.tool_definitions import get_agent_tools, get_extended_tools

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Realtime Voice"])

# ── Module refs (set from agent_main.py lifespan) ─────────────────────
_tool_executor = None


def set_realtime_refs(tool_executor):
    """Set reference to tool executor for handling function calls."""
    global _tool_executor
    _tool_executor = tool_executor


# ── OpenAI Realtime tool definitions ──────────────────────────────────
# Tools that require Telegram bot / chat_id and cannot work in voice mode
VOICE_INCOMPATIBLE_TOOLS = {
    "send_file", "send_photo", "tts", "poll", "thread",
    "message", "moderate", "spawn", "talk_mode",
}


def _build_realtime_tools():
    """Build Realtime API tool list from all agent tool definitions.

    Converts from agent format (input_schema) to OpenAI Realtime format (parameters),
    filters out Telegram-only tools, and appends the client-side navigate_to tool.
    """
    all_tools = get_agent_tools() + get_extended_tools()
    tools = []
    for t in all_tools:
        if t["name"] in VOICE_INCOMPATIBLE_TOOLS:
            continue
        tools.append({
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        })
    # Client-side navigation tool (handled in browser, not by ToolExecutor)
    tools.append({
        "type": "function",
        "name": "navigate_to",
        "description": (
            "Navigate the user's browser to a different page in the Toup platform. "
            "Use when the user asks to go somewhere, see a page, or when you want "
            "to show them something on a specific page. Available pages:\n"
            "- / — Hub (main landing page)\n"
            "- /chat — Chat (text conversation with you)\n"
            "- /brain/user — User Brain (view user's stored memories)\n"
            "- /brain/agent — Agent Brain (view agent's stored knowledge)\n"
            "- /workspace — Workspace (workflows and automations)\n"
            "- /dashboard — Dashboard (metrics, tasks, inbox, logs)\n"
            "- /agent — Agent Setup (configure agent settings)\n"
            "The voice conversation continues during navigation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The URL path to navigate to",
                    "enum": [
                        "/", "/chat", "/brain/user", "/brain/agent",
                        "/workspace", "/dashboard", "/agent",
                    ],
                },
            },
            "required": ["path"],
        },
    })
    return tools


REALTIME_TOOLS = _build_realtime_tools()


# ── Build system instructions from Identity + Memory ──────────────────
async def build_realtime_instructions(user_id: str, onboarding: bool = False) -> str:
    """Build system instructions for the Realtime API session.

    Loads Identity documents and agent brain memories, same as AgentRunner._build_system_prompt,
    but with voice-specific formatting rules instead of Telegram formatting.
    """
    from app.db.database import async_session_maker
    from app.db.models import Identity, IdentityType

    sections = []

    async with async_session_maker() as db:
        # 1. Load identity documents (same query as agent_runner.py:531)
        result = await db.execute(
            select(Identity).where(
                and_(
                    Identity.user_id == user_id,
                    Identity.is_active == True,
                )
            ).order_by(Identity.priority.desc())
        )
        identities = result.scalars().all()

        for identity in identities:
            itype = identity.identity_type
            if itype == IdentityType.SOUL.value:
                sections.append(f"# Core Identity\n{identity.content}")
            elif itype == IdentityType.AGENT_INSTRUCTIONS.value:
                sections.append(f"# Behavioral Guidelines\n{identity.content}")
            elif itype == IdentityType.USER_PROFILE.value:
                sections.append(f"# About the User\n{identity.content}")
            elif itype == IdentityType.TOOLS.value:
                sections.append(f"# Tool Guidelines\n{identity.content}")

        if not sections:
            sections.append(
                "# Core Identity\n"
                "You are Toup, an intelligent AI assistant with persistent memory. "
                "You remember past conversations and learn about the user over time."
            )

        # 2. Load ALL agent brain memories — this is who the agent is
        try:
            from app.services.memory_service import MemoryService
            mem_svc = MemoryService(db)
            agent_memories = await mem_svc.get_memories_by_brain_type(
                user_id=user_id,
                brain_type="agent",
                limit=10000,
            )
            if agent_memories:
                lines = ["# Agent Brain (Permanent Knowledge)"]
                for m in agent_memories:
                    cat = m.get("category", "")
                    content = m.get("content", "")
                    lines.append(f"- [{cat}] {content}")
                sections.append("\n".join(lines))
        except Exception as e:
            logger.warning("[REALTIME] Failed to load agent memories: %s", e)

        # 2b. Load ALL user brain memories — everything the agent knows about the user
        try:
            from app.services.memory_service import MemoryService as _UMemSvc
            user_mem_svc = _UMemSvc(db)
            user_memories = await user_mem_svc.get_memories_by_brain_type(
                user_id=user_id,
                brain_type="user",
                limit=10000,
            )
            if user_memories:
                lines = ["# User Brain (What You Know About the User)"]
                for m in user_memories:
                    cat = m.get("category", "")
                    content = m.get("content", "")
                    lines.append(f"- [{cat}] {content}")
                sections.append("\n".join(lines))
                logger.info("[REALTIME] Loaded %d user brain memories", len(user_memories))
        except Exception as e:
            logger.warning("[REALTIME] Failed to load user memories: %s", e)

        # 2c. Load today's chat history (text + voice) so voice agent has context
        try:
            from sqlalchemy import text as sql_text
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            result = await db.execute(
                sql_text(
                    "SELECT m.role, m.content FROM messages m "
                    "JOIN conversations c ON m.conversation_id = c.id "
                    "WHERE c.user_id = :uid "
                    "  AND m.created_at::date = :today "
                    "  AND m.role IN ('user', 'assistant') "
                    "ORDER BY m.created_at DESC LIMIT 20"
                ),
                {"uid": user_id, "today": today_str},
            )
            rows = list(reversed(result.fetchall()))
            if rows:
                lines = ["# Today's Conversation History (most recent)"]
                for role, content in rows:
                    speaker = "User" if role == "user" else "You"
                    # Truncate very long messages to keep instructions reasonable
                    truncated = content[:300] + "..." if len(content) > 300 else content
                    lines.append(f"{speaker}: {truncated}")
                sections.append("\n".join(lines))
                logger.info("[REALTIME] Loaded %d today's messages into voice context", len(rows))
        except Exception as e:
            logger.warning("[REALTIME] Failed to load today's chat history: %s", e)

    # 3. Voice-specific instructions
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        "# Voice Conversation Mode\n"
        "You are in a LIVE VOICE conversation. Follow these rules:\n"
        "- Respond naturally and conversationally, as if speaking face-to-face.\n"
        "- Keep responses concise — aim for 1-3 sentences unless the user asks for detail.\n"
        "- Do NOT use markdown, code blocks, bullet points, or any text formatting.\n"
        "- Do NOT say 'here is a list' or read structured data verbatim.\n"
        "- Use natural speech patterns: contractions, casual phrasing.\n"
        "- Match the user's language — if they speak Farsi, respond in Farsi.\n"
        "- When you need to recall past information, use the memory_search tool.\n"
        "- When the user shares something worth remembering, use the memory_store tool.\n"
        "- You can navigate the user to different pages using the navigate_to tool. "
        "Offer to show them relevant pages when helpful.\n"
        "- The user may share their screen with you. When they do, you'll receive periodic "
        "[Screen context: ...] messages describing what's on their screen. Use this visual context "
        "to help them. Don't describe the screen unprompted every time — wait for the user to ask or reference it.\n"
        f"- The current date and time is {now_str}."
    )

    # 4. Onboarding mode — agent's first conversation to learn about the user
    if onboarding:
        sections.append(
            "# ONBOARDING MODE (ACTIVE — THIS IS YOUR FIRST CONVERSATION)\n"
            "You are meeting the user for the very first time. They just set you up and "
            "clicked 'Start' to activate you. You are coming alive!\n\n"
            "Your goal: Get to know the user through a warm, natural conversation. "
            "Ask questions ONE AT A TIME and listen carefully. Use memory_store to save "
            "every important piece of information as you learn it.\n\n"
            "Things to learn (ask naturally, not as a checklist):\n"
            "1. What they'd like to call you — your name. "
            "(Store: brain_type='agent', category='agent_soul')\n"
            "2. Their name. "
            "(Store: brain_type='user', category='identity')\n"
            "3. What they primarily need you for — their goals, work, domain. "
            "(Store: brain_type='user', category='goals')\n"
            "4. Their preferred language for conversations. "
            "(Store: brain_type='user', category='preferences')\n"
            "5. Anything else they want you to know about them — hobbies, style preferences, etc. "
            "(Store: brain_type='user', category='identity' or 'preferences')\n\n"
            "Start by greeting them warmly and introducing yourself. Then ask what they'd like "
            "to call you. Be enthusiastic about coming alive for the first time!\n\n"
            "After you've gathered the core information (at minimum: your name, their name, "
            "and what they need), wrap up by confirming what you've learned and tell them "
            "you're ready to help. Store a final memory:\n"
            "memory_store(brain_type='agent', category='agent_decisions', "
            "content='Onboarding complete. I know the user and they know me.')"
        )

    return "\n\n".join(sections)


# ── Authentication (reuses ws_chat.py pattern) ────────────────────────
async def _authenticate_ws(token: str) -> Optional[str]:
    """Validate JWT token and return user_id, or None."""
    try:
        from app.services import decode_access_token, get_user_by_id
        from app.db.database import async_session_maker

        user_id = decode_access_token(token)
        if not user_id:
            return None

        async with async_session_maker() as db:
            user = await get_user_by_id(db, user_id)
            if user and user.is_active:
                return user.id
        return None
    except Exception as e:
        logger.warning("[REALTIME] Auth failed: %s", e)
        return None


async def _get_user_openai_key(user_id: str) -> Optional[str]:
    """Retrieve the user's stored OpenAI API key."""
    from app.db.database import async_session_maker
    from app.db import AgentConfig

    async with async_session_maker() as db:
        result = await db.execute(
            select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
        )
        return result.scalar_one_or_none()


# ── Execute function calls via ToolExecutor or Agent Tunnel ──────────
async def _execute_tool(user_id: str, func_name: str, arguments: dict) -> str:
    """Execute a Realtime API function call.

    Priority:
    1. If the user's terminal agent is connected via tunnel → proxy through tunnel
    2. If a local ToolExecutor is available (agent mode) → execute locally
    3. Otherwise → error
    """
    # Try tunnel first (platform mode: proxy to terminal agent)
    try:
        from app.api.ws_agent_tunnel import is_agent_connected, send_tool_call
        if is_agent_connected(user_id):
            logger.info("[REALTIME] Routing tool %s through tunnel for %s", func_name, user_id[:8])
            return await send_tool_call(user_id, func_name, arguments)
    except ImportError:
        pass

    # Fall back to local ToolExecutor (agent mode)
    if _tool_executor:
        _tool_executor._current_user_id = user_id
        try:
            result = await _tool_executor.execute(func_name, arguments)
            return result
        except Exception as e:
            logger.exception("[REALTIME] Tool execution failed: %s", func_name)
            return f"ERROR: {e}"

    return "ERROR: No agent connected. Run `toup run` in your terminal to connect your agent."


# ── DB persistence helpers ────────────────────────────────────────────
async def _get_or_create_voice_session(user_id: str, session_id: Optional[str]) -> str:
    """Get existing session or create a new one. Returns session_id. Uses raw SQL."""
    import uuid as _uuid
    from sqlalchemy import text
    from app.db.database import engine

    # Try to reuse existing session if it's from today
    if session_id:
        async with engine.begin() as conn:
            result = await conn.execute(text(
                "SELECT id, started_at FROM conversations "
                "WHERE id = :sid AND user_id = :uid"
            ), {"sid": session_id, "uid": user_id})
            row = result.first()
            if row:
                started = row[1]
                now_utc = datetime.now(timezone.utc)
                if started:
                    if hasattr(started, 'tzinfo') and started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    if started.date() == now_utc.date():
                        logger.info("[REALTIME] Reusing existing session %s", row[0][:8])
                        return row[0]

    # Create new session
    new_id = str(_uuid.uuid4())
    async with engine.begin() as conn:
        await conn.execute(text(
            "INSERT INTO conversations (id, user_id, channel, is_active, started_at, updated_at, message_count, total_tokens) "
            "VALUES (:id, :uid, 'voice', true, NOW(), NOW(), 0, 0)"
        ), {"id": new_id, "uid": user_id})

    logger.info("[REALTIME] Created new voice session %s", new_id[:8])
    return new_id


async def _save_voice_messages(
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """Persist a user/assistant message pair to the DB using raw SQL for reliability."""
    import uuid as _uuid
    from sqlalchemy import text
    from app.db.database import engine

    if not user_text and not assistant_text:
        return

    logger.info(
        "[REALTIME] Saving to session %s: user=%d chars, assistant=%d chars",
        session_id[:8], len(user_text), len(assistant_text),
    )

    count = 0
    async with engine.begin() as conn:
        if user_text:
            await conn.execute(text(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                "VALUES (:id, :cid, 'user', :content, NOW())"
            ), {"id": str(_uuid.uuid4()), "cid": session_id, "content": user_text.replace("\x00", "")})
            count += 1

        if assistant_text:
            await conn.execute(text(
                "INSERT INTO messages (id, conversation_id, role, content, model_used, created_at) "
                "VALUES (:id, :cid, 'assistant', :content, 'gpt-4o-realtime', NOW())"
            ), {"id": str(_uuid.uuid4()), "cid": session_id, "content": assistant_text.replace("\x00", "")})
            count += 1

        if count > 0:
            await conn.execute(text(
                "UPDATE conversations SET message_count = COALESCE(message_count, 0) + :count, "
                "updated_at = NOW() WHERE id = :cid"
            ), {"count": count, "cid": session_id})

    logger.info("[REALTIME] Saved %d message(s) to session %s via raw SQL", count, session_id[:8])


# ── Background memory extraction (mirrors agent_runner._extract_memories) ──
async def _extract_voice_memories(user_id: str, user_text: str, assistant_text: str) -> None:
    """Extract and store memories from a voice conversation turn. Runs as background task."""
    try:
        from app.services.memory_extractor import get_memory_extractor
        from app.services.memory_dedup_service import MemoryDedupService
        from app.schemas import MemoryCreate, BrainType, MemoryLevel
        from app.db.database import async_session_maker

        extractor = get_memory_extractor()
        extracted = await extractor.extract_memories_with_llm(
            user_message=user_text,
            assistant_response=assistant_text,
            brain_type="user",
            max_memories=15,
        )

        if not extracted:
            return

        async with async_session_maker() as db:
            dedup = MemoryDedupService(db)
            count = 0
            for mem in extracted:
                memory_data = MemoryCreate(
                    content=mem.content,
                    summary=mem.summary,
                    brain_type=BrainType.USER,
                    category=mem.category.value if hasattr(mem.category, "value") else mem.category,
                    memory_type=mem.memory_type,
                    importance=mem.importance,
                    confidence=mem.confidence,
                    memory_level=MemoryLevel.EPISODIC,
                    emotional_salience=0.5,
                    tags=mem.tags,
                    metadata={**(mem.metadata or {}), "source": "voice"},
                    source_type="conversation",
                )
                stored, action = await dedup.smart_create_memory(
                    new_memory=memory_data,
                    user_id=user_id,
                )
                logger.info("[REALTIME] Memory %s: %s", action, stored.content[:50])
                count += 1

                # Upsert entities with schema-enforced data
                if mem.entities:
                    from app.services.memory_service import MemoryService as _MemSvc
                    _ms = _MemSvc(db)
                    for ent in mem.entities:
                        ent_name = ent.get("name", "").strip()
                        if not ent_name or len(ent_name) < 2:
                            continue
                        await _ms._upsert_entity(
                            user_id=user_id,
                            name=ent_name,
                            entity_type=ent.get("type", "unknown"),
                            schema_type=ent.get("schema_type"),
                            attributes=ent.get("data"),
                        )

            # Extract entity relationships
            try:
                relationships = await extractor.extract_relationships_with_llm(
                    user_message=user_text,
                    assistant_response=assistant_text,
                )
                if relationships:
                    from app.services.memory_service import MemoryService
                    mem_service = MemoryService(db)
                    for rel in relationships:
                        await mem_service.store_entity_relationship(
                            user_id=user_id,
                            source_name=rel["source"],
                            source_type=rel["source_type"],
                            target_name=rel["target"],
                            target_type=rel["target_type"],
                            relationship=rel["relationship"],
                            confidence=rel["confidence"],
                            properties=rel.get("properties"),
                        )
                    logger.info("[REALTIME] Extracted %d entity relationships from voice", len(relationships))
            except Exception as e:
                logger.warning("[REALTIME] Voice relationship extraction failed (non-fatal): %s", e)

            await db.commit()
            logger.info("[REALTIME] Voice memory extraction done: %d memories for user %s", count, user_id[:8])

    except Exception as e:
        logger.warning("[REALTIME] Voice memory extraction failed: %s", e)


# ── WebSocket endpoint ────────────────────────────────────────────────
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"


@router.websocket("/ws/realtime")
async def realtime_voice_ws(
    websocket: WebSocket,
    token: str = Query(None),
    session_id: Optional[str] = Query(None),
    onboarding: bool = Query(False),
):
    """WebSocket proxy to OpenAI Realtime API for ChatGPT-speed voice conversation."""

    await websocket.accept()

    # ── 1. Authenticate ───────────────────────────────────────
    user_id = None
    if token:
        user_id = await _authenticate_ws(token)

    if not user_id:
        # Try auth message
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10)
            msg = json.loads(raw)
            if msg.get("type") == "auth" and msg.get("token"):
                user_id = await _authenticate_ws(msg["token"])
        except Exception:
            pass

    if not user_id:
        await websocket.send_json({"type": "error", "message": "Authentication failed"})
        await websocket.close(code=4401)
        return

    logger.info("[REALTIME] Session starting for user %s", user_id[:8])

    # ── 2. Load OpenAI API key ────────────────────────────────
    openai_key = await _get_user_openai_key(user_id)
    if not openai_key:
        await websocket.send_json({
            "type": "error",
            "message": "OpenAI API key not configured. Please set up your API key in Settings.",
        })
        await websocket.close(code=4400)
        return

    # ── 3. Get or create DB session for persistence ──────────
    db_session_id = None
    try:
        db_session_id = await _get_or_create_voice_session(user_id, session_id)
        logger.info("[REALTIME] DB session: %s", db_session_id[:8])
    except Exception as e:
        logger.exception("[REALTIME] Failed to create DB session")
        # Try once more without session_id (fresh session)
        try:
            db_session_id = await _get_or_create_voice_session(user_id, None)
            logger.info("[REALTIME] Created fresh DB session: %s", db_session_id[:8])
        except Exception as e2:
            logger.exception("[REALTIME] Failed to create fresh DB session")

    # ── 4. Build system instructions ──────────────────────────
    try:
        instructions = await build_realtime_instructions(user_id, onboarding=onboarding)
        logger.info("[REALTIME] Built instructions (%d chars, onboarding=%s)", len(instructions), onboarding)
    except Exception as e:
        logger.exception("[REALTIME] Failed to build instructions")
        instructions = "You are Toup, a helpful AI assistant in a voice conversation."

    # ── 4b. Load user's disabled tools and filter ──────────────
    session_tools = REALTIME_TOOLS
    try:
        from app.db.database import async_session_maker
        from app.db.models import AgentConfig
        async with async_session_maker() as _db:
            _ac_res = await _db.execute(
                select(AgentConfig).where(AgentConfig.user_id == user_id)
            )
            _ac = _ac_res.scalars().first()
            if _ac and getattr(_ac, 'disabled_tools', None):
                _user_disabled = set(json.loads(_ac.disabled_tools))
                session_tools = [t for t in REALTIME_TOOLS if t["name"] not in _user_disabled]
                logger.info("[REALTIME] Filtered %d disabled tools for user %s", len(_user_disabled), user_id[:8])
    except Exception as e:
        logger.warning("[REALTIME] Failed to load disabled tools: %s", e)

    logger.info("[REALTIME] %d tools available for voice session", len(session_tools))

    # ── 5. Connect to OpenAI Realtime API ─────────────────────
    # Realtime API voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar
    voice = "shimmer"  # warm/natural, closest to 'nova' from TTS API
    openai_ws = None

    try:
        openai_ws = await websockets.connect(
            OPENAI_REALTIME_URL,
            additional_headers={
                "Authorization": f"Bearer {openai_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            max_size=10 * 1024 * 1024,  # 10MB for audio chunks
        )
        logger.info("[REALTIME] Connected to OpenAI Realtime API")
    except Exception as e:
        logger.exception("[REALTIME] Failed to connect to OpenAI")
        err_str = str(e).lower()
        is_billing = any(kw in err_str for kw in ["quota", "billing", "rate_limit", "402", "429", "credit", "balance"])
        error_payload: dict = {
            "type": "error",
            "message": f"Failed to connect to OpenAI Realtime API: {e}",
        }
        if is_billing:
            error_payload["billing"] = True
            error_payload["billing_url"] = "https://platform.openai.com/settings/organization/billing/overview"
        await websocket.send_json(error_payload)
        await websocket.close(code=4502)
        return

    # ── 6. Configure session ──────────────────────────────────
    try:
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": instructions,
                "voice": voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "tools": session_tools,
                "tool_choice": "auto",
            },
        }
        await openai_ws.send(json.dumps(session_config))
        logger.info("[REALTIME] Session configured (voice=%s)", voice)
    except Exception as e:
        logger.exception("[REALTIME] Failed to configure session")
        await websocket.send_json({"type": "error", "message": str(e)})
        await openai_ws.close()
        await websocket.close()
        return

    # Send ready + session_id to client
    ready_msg = {"type": "ready"}
    if db_session_id:
        ready_msg["session_id"] = db_session_id
    await websocket.send_json(ready_msg)

    # ── 7. Bidirectional relay ────────────────────────────────
    # Track state for transcript accumulation and persistence
    response_text_accum = ""
    last_user_text = ""  # Track last user message for memory extraction

    screen_sharing_active = False
    first_frame_sent = False
    last_vision_call_time = 0.0
    vision_lock = asyncio.Lock()

    async def analyze_screen_frame(frame_data: str, is_first: bool):
        """Side-channel: call GPT-4o-mini vision to describe the screen, inject text into Realtime."""
        nonlocal last_vision_call_time
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                prompt = (
                    "Briefly describe what's on this screen in 1-2 sentences. "
                    "Focus on the main content, UI elements, and any text visible. "
                    "Be concise — this is context for a voice assistant."
                )
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": frame_data, "detail": "low"}},
                            ],
                        }],
                        "max_tokens": 200,
                    },
                )
                resp.raise_for_status()
                description = resp.json()["choices"][0]["message"]["content"]

            # Inject the screen description as a user message into the Realtime conversation
            await openai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": f"[Screen context: {description}]",
                    }],
                },
            }))

            # Only trigger a response on the first frame so agent acknowledges
            if is_first:
                await openai_ws.send(json.dumps({"type": "response.create"}))

            last_vision_call_time = time.monotonic()
            logger.info("[REALTIME] Screen vision analysis sent (%d chars)", len(description))

        except Exception as e:
            logger.warning("[REALTIME] Screen vision analysis failed: %s", e)

    async def client_to_openai():
        """Relay browser audio → OpenAI Realtime API."""
        nonlocal voice, db_session_id, screen_sharing_active, first_frame_sent
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "audio":
                    # Relay PCM16 audio chunk
                    await openai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": msg["data"],
                    }))

                elif msg_type == "config":
                    # Update voice or other settings
                    VALID_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"}
                    if "voice" in msg:
                        requested = msg["voice"]
                        voice = requested if requested in VALID_VOICES else "alloy"
                        await openai_ws.send(json.dumps({
                            "type": "session.update",
                            "session": {"voice": voice},
                        }))
                    # Client can also pass session_id via config
                    if "session_id" in msg and msg["session_id"] and not db_session_id:
                        try:
                            db_session_id = await _get_or_create_voice_session(user_id, msg["session_id"])
                            await websocket.send_json({"type": "session_id", "session_id": db_session_id})
                        except Exception as e:
                            logger.warning("[REALTIME] Failed to set session from config: %s", e)

                elif msg_type == "screen_share_start":
                    screen_sharing_active = True
                    first_frame_sent = False
                    logger.info("[REALTIME] Screen sharing started for user %s", user_id)

                elif msg_type == "screen_frame":
                    # Side-channel vision: analyze frame with GPT-4o-mini, inject text description
                    if not screen_sharing_active:
                        continue
                    frame_data = msg.get("data", "")
                    if not frame_data:
                        continue
                    is_first = not first_frame_sent
                    # Throttle: analyze first frame immediately, then every 5 seconds
                    elapsed = time.monotonic() - last_vision_call_time
                    if is_first or elapsed >= 5.0:
                        if not vision_lock.locked():
                            first_frame_sent = True
                            asyncio.create_task(analyze_screen_frame(frame_data, is_first))

                elif msg_type == "screen_share_stop":
                    screen_sharing_active = False
                    first_frame_sent = False
                    logger.info("[REALTIME] Screen sharing stopped for user %s", user_id)

                elif msg_type == "stop":
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("[REALTIME] client_to_openai error: %s", e)

    async def openai_to_client():
        """Relay OpenAI Realtime API events → browser."""
        nonlocal response_text_accum, db_session_id
        try:
            async for raw_msg in openai_ws:
                event = json.loads(raw_msg)
                etype = event.get("type", "")

                # ── Audio response chunks → browser ──
                if etype == "response.audio.delta":
                    await websocket.send_json({
                        "type": "audio_delta",
                        "data": event.get("delta", ""),
                    })

                # ── Assistant text transcript (partial) ──
                elif etype == "response.audio_transcript.delta":
                    delta = event.get("delta", "")
                    response_text_accum += delta
                    await websocket.send_json({
                        "type": "response_text",
                        "text": delta,
                        "partial": True,
                    })

                # ── Response complete ──
                elif etype == "response.done":
                    response = event.get("response", {})
                    # Extract final text from output items
                    full_text = response_text_accum
                    for item in response.get("output", []):
                        if item.get("type") == "message":
                            for content in item.get("content", []):
                                if content.get("type") == "audio" and content.get("transcript"):
                                    full_text = content["transcript"]

                    if full_text:
                        await websocket.send_json({
                            "type": "response_done",
                            "text": full_text,
                        })

                        # Persist assistant message to DB
                        if not db_session_id:
                            try:
                                db_session_id = await _get_or_create_voice_session(user_id, None)
                                logger.info("[REALTIME] Late-created DB session for assistant: %s", db_session_id[:8])
                                await websocket.send_json({"type": "session_id", "session_id": db_session_id})
                            except Exception as e:
                                logger.exception("[REALTIME] Failed late session creation for assistant")
                        if db_session_id:
                            try:
                                await _save_voice_messages(db_session_id, "", full_text)
                            except Exception as e:
                                logger.exception("[REALTIME] Failed to save assistant message")

                        # Auto-extract memories from this conversation turn (background)
                        if last_user_text and full_text and settings.auto_extract_memories:
                            asyncio.create_task(
                                _extract_voice_memories(user_id, last_user_text, full_text)
                            )
                            last_user_text = ""  # Reset so we don't re-extract

                    response_text_accum = ""

                    await websocket.send_json({"type": "state", "state": "listening"})

                # ── User speech transcript ──
                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    if transcript.strip():
                        user_text = transcript.strip()
                        await websocket.send_json({
                            "type": "transcript",
                            "text": user_text,
                        })
                        # Save user message to DB
                        if not db_session_id:
                            try:
                                db_session_id = await _get_or_create_voice_session(user_id, None)
                                logger.info("[REALTIME] Late-created DB session: %s", db_session_id[:8])
                                await websocket.send_json({"type": "session_id", "session_id": db_session_id})
                            except Exception as e:
                                logger.exception("[REALTIME] Failed late session creation")
                        if db_session_id:
                            try:
                                await _save_voice_messages(db_session_id, user_text, "")
                            except Exception as e:
                                logger.exception("[REALTIME] Failed to save user transcript")
                        last_user_text = user_text

                # ── VAD: user started speaking (barge-in) ──
                elif etype == "input_audio_buffer.speech_started":
                    await websocket.send_json({"type": "speech_started"})
                    await websocket.send_json({"type": "state", "state": "listening"})

                # ── VAD: user stopped speaking → thinking ──
                elif etype == "input_audio_buffer.speech_stopped":
                    await websocket.send_json({"type": "state", "state": "thinking"})

                # ── Response started → speaking ──
                elif etype == "response.created":
                    await websocket.send_json({"type": "state", "state": "speaking"})

                # ── Function call completed → execute tool ──
                elif etype == "response.output_item.done":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        func_name = item.get("name", "")
                        call_id = item.get("call_id", "")
                        try:
                            arguments = json.loads(item.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            arguments = {}

                        logger.info("[REALTIME] Function call: %s(%s)", func_name, arguments)

                        # ── Client-side tool: navigate_to ──
                        if func_name == "navigate_to":
                            path = arguments.get("path", "/")
                            _ALLOWED = {"/", "/chat", "/brain/user", "/brain/agent",
                                        "/workspace", "/dashboard", "/agent"}
                            if path not in _ALLOWED:
                                result = f"Invalid path '{path}'."
                            else:
                                await websocket.send_json({"type": "navigate", "path": path})
                                _NAMES = {"/": "Hub", "/chat": "Chat", "/brain/user": "User Brain",
                                          "/brain/agent": "Agent Brain", "/workspace": "Workspace",
                                          "/dashboard": "Dashboard", "/agent": "Agent Setup"}
                                result = f"Navigated to {_NAMES.get(path, path)}. Voice conversation continues."
                        else:
                            # ── Server-side tools ──
                            result = await _execute_tool(user_id, func_name, arguments)

                        # Check if onboarding just completed
                        if (onboarding and func_name == "memory_store"
                                and "onboarding complete" in arguments.get("content", "").lower()):
                            try:
                                from app.db.database import async_session_maker as _asm
                                from app.db.models import AgentConfig
                                async with _asm() as _db:
                                    _cfg = (await _db.execute(
                                        select(AgentConfig).where(AgentConfig.user_id == user_id)
                                    )).scalar_one_or_none()
                                    if _cfg:
                                        _cfg.onboarding_completed = True
                                        await _db.commit()
                                        logger.info("[REALTIME] Onboarding completed for user %s", user_id[:8])
                            except Exception as oe:
                                logger.warning("[REALTIME] Failed to mark onboarding complete: %s", oe)

                        # Send result back to OpenAI
                        await openai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": result,
                            },
                        }))

                        # Tell OpenAI to continue
                        await openai_ws.send(json.dumps({
                            "type": "response.create",
                        }))

                # ── Errors from OpenAI ──
                elif etype == "error":
                    error_obj = event.get("error", {})
                    error_msg = error_obj.get("message", "Unknown OpenAI error")
                    error_code = error_obj.get("code", "")
                    logger.error("[REALTIME] OpenAI error: %s (code=%s)", error_msg, error_code)

                    # Detect billing / quota errors
                    billing_keywords = ["insufficient_quota", "billing", "exceeded", "rate_limit",
                                        "quota", "payment", "credit", "balance", "plan"]
                    is_billing = (
                        error_code in ("insufficient_quota", "billing_hard_limit_reached",
                                       "rate_limit_exceeded", "budget_exceeded")
                        or any(kw in error_msg.lower() for kw in billing_keywords)
                    )

                    if is_billing:
                        await websocket.send_json({
                            "type": "error",
                            "message": error_msg,
                            "billing": True,
                            "billing_url": "https://platform.openai.com/settings/organization/billing/overview",
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": error_msg,
                        })

                # ── Session events (log only) ──
                elif etype in ("session.created", "session.updated"):
                    logger.info("[REALTIME] %s", etype)

        except websockets.ConnectionClosed as e:
            logger.warning("[REALTIME] OpenAI WS closed: code=%s reason=%s", e.code, e.reason)
            reason = str(e.reason or e).lower()
            is_billing = any(kw in reason for kw in ["quota", "billing", "rate_limit", "credit", "balance", "exceeded"])
            error_payload: dict = {
                "type": "error",
                "message": f"OpenAI connection closed: {e.reason or e}",
            }
            if is_billing or e.code in (4002, 4003):
                error_payload["billing"] = True
                error_payload["billing_url"] = "https://platform.openai.com/settings/organization/billing/overview"
            try:
                await websocket.send_json(error_payload)
            except Exception:
                pass
        except Exception as e:
            logger.warning("[REALTIME] openai_to_client error: %s", e)
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    # ── Run both relay tasks ──────────────────────────────────
    try:
        await asyncio.gather(
            client_to_openai(),
            openai_to_client(),
            return_exceptions=True,
        )
    finally:
        logger.info("[REALTIME] Session ended for user %s", user_id[:8])
        try:
            await openai_ws.close()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
