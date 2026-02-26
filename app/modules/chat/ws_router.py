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
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)

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

            session_id = msg.get("session_id")
            model = msg.get("model")

            # Stream callbacks
            async def on_text_chunk(chunk: str):
                try:
                    await websocket.send_json({"type": "text_chunk", "text": chunk})
                except Exception:
                    pass

            async def on_tool_start(tool_name: str):
                try:
                    await websocket.send_json({"type": "tool_start", "tool": tool_name})
                except Exception:
                    pass

            async def on_tool_end(tool_name: str, summary: str):
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
