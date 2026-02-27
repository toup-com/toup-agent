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
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from app.config import settings

logger = logging.getLogger(__name__)

# ── Onboarding prompt ────────────────────────────────────────────────
_ONBOARDING_TRIGGER = (
    "[SYSTEM: ONBOARDING — FIRST CONVERSATION]\n"
    "The user just set up their AI agent and is opening the chat for the very first time. "
    "You are coming alive for the first time. Greet them warmly.\n\n"
    "Your goals for this conversation:\n"
    "1. Ask what they'd like to call you (your name). Once they answer, use memory_store "
    "(brain_type='agent', category='agent_soul') to save it.\n"
    "2. Ask what they primarily need you for. Store with memory_store "
    "(brain_type='user', category='goals').\n"
    "3. Ask their name so you can address them personally. Store with memory_store "
    "(brain_type='user', category='identity').\n\n"
    "Be conversational and warm. Ask ONE question at a time — don't overwhelm them. "
    "After you've learned all three, confirm you're ready and store a final memory: "
    "memory_store(brain_type='agent', category='agent_decisions', "
    "content='Onboarding complete. I know the user and they know me.')\n\n"
    "Start by greeting them and asking what they'd like to call you."
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
):
    """
    WebSocket endpoint for real-time chat with the HexBrain agent.

    Supports streaming text chunks, tool call indicators, and session management.
    """
    await websocket.accept()
    user_id: Optional[str] = None

    # Try query-param auth first
    if token:
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

        # Main message loop
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
                    on_text_chunk=on_text_chunk,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    model_override=model,
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

    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected: {user_id}")
    except Exception as e:
        logger.exception(f"[WS] Unexpected error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=4500)
        except Exception:
            pass
