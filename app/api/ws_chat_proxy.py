"""
WebSocket Chat Proxy — Platform proxies chat WS to user's Agent VPS.

The browser connects to wss://toup.ai/api/ws/chat.
The platform authenticates via JWT, looks up the agent's URL + API key,
opens a WebSocket to the agent, and relays messages bidirectionally.

This avoids mixed-content issues (HTTPS→WS) and JWT secret mismatches.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select

from app.config import settings
from app.services.auth_service import decode_access_token, get_user_by_id
from app.db.database import async_session_maker
from app.db.models import AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket Chat Proxy"])


async def _authenticate_ws(token: str) -> Optional[str]:
    """Validate a JWT token and return the user_id, or None."""
    try:
        user_id = decode_access_token(token)
        if not user_id:
            return None
        async with async_session_maker() as db:
            user = await get_user_by_id(db, user_id)
            if user and user.is_active:
                return user.id
        return None
    except Exception as e:
        logger.warning("WS proxy auth failed: %s", e)
        return None


async def _get_agent_ws_info(user_id: str) -> Optional[tuple[str, str]]:
    """Return (agent_ws_url, agent_api_key) for the user's deployed agent."""
    async with async_session_maker() as db:
        result = await db.execute(
            select(AgentConfig.agent_url, AgentConfig.agent_api_key)
            .where(
                AgentConfig.user_id == user_id,
                AgentConfig.deploy_status == "active",
            )
        )
        row = result.first()
        if row and row.agent_url and row.agent_api_key:
            # Convert http:// to ws://
            ws_url = row.agent_url.replace("https://", "wss://").replace("http://", "ws://")
            if not ws_url.endswith("/"):
                ws_url += "/api/ws/chat"
            else:
                ws_url += "api/ws/chat"
            return (ws_url, row.agent_api_key)
    return None


@router.websocket("/ws/chat")
async def ws_chat_proxy(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    WebSocket proxy: browser ↔ platform ↔ agent VPS.

    Authenticates the user, finds their agent, and relays messages.
    """
    await websocket.accept()
    user_id: Optional[str] = None

    # Authenticate via query param
    if token:
        user_id = await _authenticate_ws(token)

    # If not authenticated, try first message
    if not user_id:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("type") == "auth" and msg.get("token"):
                user_id = await _authenticate_ws(msg["token"])
        except (asyncio.TimeoutError, json.JSONDecodeError):
            pass

    if not user_id:
        await websocket.send_json({"type": "error", "message": "Authentication required"})
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Look up agent info
    agent_info = await _get_agent_ws_info(user_id)
    if not agent_info:
        await websocket.send_json({"type": "error", "message": "No active agent found. Deploy your agent first."})
        await websocket.close(code=4404, reason="No agent")
        return

    agent_ws_url, agent_api_key = agent_info
    full_url = f"{agent_ws_url}?agent_key={agent_api_key}"

    logger.info("[WS Proxy] User %s → Agent %s", user_id, agent_ws_url)

    # Connect to the agent's WebSocket
    try:
        import websockets
    except ImportError:
        await websocket.send_json({"type": "error", "message": "WebSocket proxy not available"})
        await websocket.close(code=4500)
        return

    agent_ws = None
    try:
        agent_ws = await asyncio.wait_for(
            websockets.connect(full_url, max_size=10 * 1024 * 1024),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "Agent connection timed out"})
        await websocket.close(code=4504, reason="Agent timeout")
        return
    except Exception as e:
        logger.warning("[WS Proxy] Agent connect failed: %s", e)
        await websocket.send_json({"type": "error", "message": f"Cannot reach agent: {e}"})
        await websocket.close(code=4502, reason="Agent unreachable")
        return

    logger.info("[WS Proxy] Connected to agent for user %s", user_id)

    # Relay messages bidirectionally
    async def browser_to_agent():
        """Forward messages from browser → agent."""
        try:
            while True:
                raw = await websocket.receive_text()
                await agent_ws.send(raw)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await agent_ws.close()

    async def agent_to_browser():
        """Forward messages from agent → browser."""
        try:
            async for raw in agent_ws:
                await websocket.send_text(raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    # Run both directions concurrently
    try:
        await asyncio.gather(
            browser_to_agent(),
            agent_to_browser(),
        )
    except Exception as e:
        logger.info("[WS Proxy] Session ended for user %s: %s", user_id, e)
    finally:
        if agent_ws and not agent_ws.closed:
            await agent_ws.close()
        logger.info("[WS Proxy] Disconnected user %s", user_id)
