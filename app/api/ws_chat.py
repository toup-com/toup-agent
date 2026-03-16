"""
WebSocket Chat Endpoint — Real-time streaming chat via WebSocket.

Protocol:
  Client sends JSON:
    { "type": "message", "text": "...", "session_id": "..." }
    { "type": "ping" }

  Server sends JSON:
    { "type": "text_chunk", "text": "..." }
    { "type": "tool_start", "tool": "..." }
    { "type": "tool_end", "tool": "...", "summary": "..." }
    { "type": "done", "session_id": "...", "tokens": {...}, "model": "..." }
    { "type": "error", "message": "..." }
    { "type": "pong" }

Authentication:
  Connect with token as query param: ws://host/api/ws/chat?token=JWT_TOKEN
  Or send as first message: { "type": "auth", "token": "JWT_TOKEN" }
"""

import asyncio
import json
import logging
import sys
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from app.config import settings

logger = logging.getLogger(__name__)

# ── User WebSocket broadcast registry ────────────────────────────────
# Maps user_id → list of asyncio.Queues (one per active WS connection).
# Background tasks (e.g. app builder) push events here, and the WS
# handler forwards them to the client.
_user_ws_queues: Dict[str, List[asyncio.Queue]] = {}


async def broadcast_to_user(user_id: str, event: dict) -> int:
    """Push an event to all WebSocket connections for a user.
    Returns number of connections that received the event."""
    queues = _user_ws_queues.get(user_id, [])
    sent = 0
    for q in queues:
        try:
            q.put_nowait(event)
            sent += 1
        except asyncio.QueueFull:
            pass  # Drop if client is slow
    return sent


def _register_ws_queue(user_id: str, queue: asyncio.Queue) -> None:
    """Register a queue for a user's WebSocket connection."""
    if user_id not in _user_ws_queues:
        _user_ws_queues[user_id] = []
    _user_ws_queues[user_id].append(queue)


def _unregister_ws_queue(user_id: str, queue: asyncio.Queue) -> None:
    """Unregister a queue when WebSocket disconnects."""
    queues = _user_ws_queues.get(user_id, [])
    try:
        queues.remove(queue)
    except ValueError:
        pass
    if not queues:
        _user_ws_queues.pop(user_id, None)

# ── Onboarding prompt ────────────────────────────────────────────────
_ONBOARDING_TRIGGER = (
    "[SYSTEM: ONBOARDING — FIRST CONVERSATION]\n"
    "The user just set up their AI agent and is opening the chat for the very first time. "
    "You are coming alive for the first time. Greet them warmly.\n\n"
    "Your FIRST question MUST be: 'What is your name, and what is my name?'\n"
    "Wait for their answer, then:\n"
    "- Store user's name: memory_store(brain_type='user', category='identity', "
    "content='User name: <name>')\n"
    "- Store your name: memory_store(brain_type='agent', category='agent_soul', "
    "content='My name is <name>')\n\n"
    "Then continue naturally, ONE question at a time:\n"
    "- What they primarily need you for — goals, work domain. "
    "Store: brain_type='user', category='goals'\n"
    "- Their preferred language. Store: brain_type='user', category='preferences'\n"
    "- How they want you to communicate — formal/casual, concise/detailed. "
    "Store: brain_type='agent', category='agent_soul'\n"
    "- Any behavioral rules (always/never do). Store: brain_type='agent', category='agent_soul'\n\n"
    "After gathering core info (names, goals, language, personality), summarize what you learned. "
    "Then store: memory_store(brain_type='agent', category='agent_decisions', "
    "content='Onboarding complete. I know the user and they know me.')\n\n"
    "Be warm, enthusiastic, conversational. Ask ONE question at a time."
)

# ── ANSI helpers for terminal activity display ────────────────────────
_CYAN_BOLD = "\033[1;36m"
_GREEN_BOLD = "\033[1;32m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_IS_TTY = sys.stdout.isatty()


def _tprint(msg: str) -> None:
    """Print to terminal only when stdout is a TTY (not piped/redirected)."""
    if _IS_TTY:
        print(msg, flush=True)

router = APIRouter(tags=["WebSocket Chat"])

# References set at startup
_agent_runner = None
_skill_loader = None


def set_ws_refs(agent_runner, skill_loader=None):
    """Set references to the agent runner and skill loader (called from main.py lifespan)."""
    global _agent_runner, _skill_loader
    _agent_runner = agent_runner
    _skill_loader = skill_loader


async def _authenticate_ws(token: str) -> Optional[str]:
    """Validate a JWT token and return the user_id, or None."""
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
        logger.warning(f"WS auth failed: {e}")
        return None


@router.websocket("/ws/chat")
async def ws_chat(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    agent_key: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time chat with the agent.

    Supports streaming text chunks, tool call indicators, and session management.
    Auth: JWT token (query param or first message) OR agent_key (for platform proxy).
    """
    await websocket.accept()
    user_id: Optional[str] = None

    # Try agent_key auth first (platform proxy mode)
    if agent_key and settings.agent_api_key and agent_key == settings.agent_api_key:
        user_id = settings.user_id
        if user_id:
            # Ensure stub user exists (same as auth.py agent mode)
            from app.db.database import async_session_maker as _sm
            from app.db.models import User
            async with _sm() as _db:
                from app.services.auth_service import get_user_by_id
                u = await get_user_by_id(_db, user_id)
                if not u:
                    u = User(id=user_id, email=f"{user_id[:8]}@agent.local", hashed_password="", name="Agent Owner")
                    _db.add(u)
                    await _db.commit()

    # Try JWT query-param auth
    if not user_id and token:
        user_id = await _authenticate_ws(token)

    try:
        # If not authenticated via query param, expect auth message
        if not user_id:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                msg = json.loads(raw)
                if msg.get("type") == "auth" and msg.get("token"):
                    user_id = await _authenticate_ws(msg["token"])
            except asyncio.TimeoutError:
                pass

        if not user_id:
            await websocket.send_json({"type": "error", "message": "Authentication required"})
            await websocket.close(code=4001, reason="Unauthorized")
            return

        if not _agent_runner:
            await websocket.send_json({"type": "error", "message": "Agent not available"})
            await websocket.close(code=4503, reason="Service unavailable")
            return

        logger.info(f"[WS] Authenticated user: {user_id}")

        # Register broadcast queue for this connection
        broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        _register_ws_queue(user_id, broadcast_queue)

        async def _broadcast_reader():
            """Forward broadcast events to this WebSocket."""
            try:
                while True:
                    event = await broadcast_queue.get()
                    try:
                        await websocket.send_json(event)
                    except Exception:
                        break
            except asyncio.CancelledError:
                pass

        broadcast_task = asyncio.create_task(_broadcast_reader())

        # Main message loop
        try:
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    logger.info(f"[WS] Client disconnected: {user_id}")
                    return

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if msg_type != "message":
                    await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
                    continue

                text = msg.get("text", "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "message": "Empty message"})
                    continue

                # ── Onboarding trigger ──────────────────────────────
                is_onboarding_msg = False
                if text == "__ONBOARDING_START__":
                    from app.db.database import async_session_maker
                    from app.db.models import AgentConfig
                    async with async_session_maker() as _db:
                        _cfg = (await _db.execute(
                            select(AgentConfig).where(AgentConfig.user_id == user_id)
                        )).scalar_one_or_none()
                        if _cfg and _cfg.onboarding_completed:
                            text = "Hello!"
                        else:
                            text = _ONBOARDING_TRIGGER
                            is_onboarding_msg = True

                session_id = msg.get("session_id")
                model = msg.get("model")
                channel = msg.get("channel")  # e.g. "mobile", "web", "app"

                # If message comes from inside a built app, prepend context
                app_id_from_msg = msg.get("app_id")
                if channel == "app" and app_id_from_msg:
                    try:
                        from app.db.database import async_session_maker
                        from app.db.models import App
                        async with async_session_maker() as _db:
                            _app = await _db.get(App, app_id_from_msg)
                            if _app:
                                _slug_safe = _app.slug.replace('-', '_')
                                _is_layer2 = msg.get("layer2") or False
                                _base_ctx = (
                                    f"[CONTEXT: The user is chatting from inside their '{_app.name}' app. "
                                    f"You are their in-app assistant.\n"
                                    f"- Be conversational and helpful. Greet naturally when they say hi.\n"
                                    f"- NEVER mention internal details (SQLite, bridges, connections, file paths, agent infrastructure).\n"
                                    f"- You have these app tools: app_{_slug_safe}__navigate (change screens), "
                                    f"app_{_slug_safe}__read_file / app_{_slug_safe}__write_file (edit the app), "
                                    f"app_{_slug_safe}__query_db (read/write app data).\n"
                                    f"- When the user asks to change something in the app (UI, content, settings), "
                                    f"use write_file/query_db to make it happen.\n"
                                    f"- Suggest helpful actions as [[Button Label]] chips.\n"
                                )
                                if _is_layer2:
                                    # Layer 2 trigger — audit-first deep customization
                                    _base_ctx += (
                                        f"\n- LAYER 2 CUSTOMIZATION MODE activated.\n"
                                        f"  STEP 1 (SILENT): Use app_{_slug_safe}__read_file to read EVERY key file — "
                                        f"App.tsx, all screen components, database/seed files, data constants, config files. "
                                        f"Do NOT tell the user you are reading files. Do NOT expose paths or technical details.\n"
                                        f"  As you read, identify: placeholder/demo data, shallow features, generic defaults, "
                                        f"hardcoded content that should be personalized, missing functionality.\n"
                                        f"  STEP 2: Ask 10+ questions that reference SPECIFIC things you found in the code. "
                                        f"Example: 'I found 500 vocabulary words but they are all general — should I focus on your field?' "
                                        f"NEVER ask generic onboarding questions (target score, test date, color theme — Layer 1 already did that). "
                                        f"Every question MUST have [[option]] buttons on the NEXT LINE — buttons must be inline with their question, "
                                        f"NOT collected at the end.\n"
                                        f"  STEP 3: Apply changes with write_file/query_db. Show brief progress after each edit.\n"
                                    )
                                text = f"{_base_ctx}]\n\n{text}"
                    except Exception as e:
                        logger.warning(f"[WS] Failed to load app context: {e}")

                # Terminal activity: show user message
                _tprint(f"\n{_CYAN_BOLD} user {_RESET} {text}")

                # Stream callbacks
                async def on_text_chunk(chunk: str):
                    try:
                        await websocket.send_json({"type": "text_chunk", "text": chunk})
                    except Exception:
                        pass

                async def on_tool_start(tool_name: str):
                    _tprint(f"{_DIM}  ⚙ {tool_name}{_RESET}")
                    try:
                        await websocket.send_json({"type": "tool_start", "tool": tool_name})
                    except Exception:
                        pass

                async def on_tool_end(tool_name: str, summary: str):
                    short = summary[:120] + "..." if len(summary) > 120 else summary
                    # Collapse to single line for terminal readability
                    short = short.replace("\n", " ")
                    _tprint(f"{_DIM}  ✓ {tool_name}: {short}{_RESET}")
                    try:
                        await websocket.send_json({"type": "tool_end", "tool": tool_name, "summary": summary})
                    except Exception:
                        pass

                # Run agent
                try:
                    response = await _agent_runner.run(
                        user_message=text,
                        user_id=user_id,
                        session_id=session_id,
                        channel=channel,
                        on_text_chunk=on_text_chunk,
                        on_tool_start=on_tool_start,
                        on_tool_end=on_tool_end,
                        model_override=model,
                        save_user_message=not is_onboarding_msg,
                    )

                    # Terminal activity: show agent response summary
                    resp_preview = response.text[:200].replace("\n", " ")
                    if len(response.text) > 200:
                        resp_preview += "..."
                    _tprint(f"{_GREEN_BOLD} agent {_RESET} {resp_preview}")
                    _tprint(
                        f"{_DIM}  ({response.tokens_total or 0} tokens, "
                        f"{response.processing_time_ms or 0}ms, "
                        f"{response.model or '?'}){_RESET}"
                    )

                    # Check if onboarding just completed (agent stored the signal memory)
                    for tc in response.tool_calls:
                        if tc.get("name") == "memory_store":
                            tc_content = (tc.get("input") or {}).get("content", "")
                            if "onboarding complete" in tc_content.lower():
                                try:
                                    from app.db.database import async_session_maker
                                    from app.db.models import AgentConfig
                                    async with async_session_maker() as _db:
                                        _cfg = (await _db.execute(
                                            select(AgentConfig).where(AgentConfig.user_id == user_id)
                                        )).scalar_one_or_none()
                                        if _cfg:
                                            _cfg.onboarding_completed = True
                                            await _db.commit()
                                            logger.info(f"[WS] Onboarding completed for user {user_id}")
                                except Exception as e:
                                    logger.warning(f"[WS] Failed to mark onboarding complete: {e}")
                                break

                    await websocket.send_json({
                        "type": "done",
                        "text": response.text,
                        "session_id": response.session_id,
                        "tokens": {
                            "input": response.tokens_input,
                            "output": response.tokens_output,
                            "total": response.tokens_total,
                        },
                        "model": response.model,
                        "tool_calls": len(response.tool_calls),
                        "processing_time_ms": response.processing_time_ms,
                    })

                except asyncio.CancelledError:
                    await websocket.send_json({"type": "error", "message": "Request cancelled"})
                except Exception as e:
                    logger.exception(f"[WS] Agent error for {user_id}")
                    _tprint(f"\033[1;31m  ✗ Error: {e}{_RESET}")
                    await websocket.send_json({"type": "error", "message": f"Agent error: {type(e).__name__}: {e}"})
        finally:
            # Clean up broadcast queue and task
            broadcast_task.cancel()
            _unregister_ws_queue(user_id, broadcast_queue)

    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected: {user_id}")
    except Exception as e:
        logger.exception(f"[WS] Unexpected error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=4500)
        except Exception:
            pass
