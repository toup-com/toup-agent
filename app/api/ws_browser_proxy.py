"""
WebSocket Browser Proxy — Platform proxies browser WS to user's Agent VPS.

The browser connects to wss://toup.ai/api/ws/browser.
The platform authenticates via JWT, looks up the agent's URL + API key,
opens a WebSocket to the agent's /ws/browser endpoint, and relays messages.
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
router = APIRouter(tags=["WebSocket Browser Proxy"])


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
        logger.warning("WS browser proxy auth failed: %s", e)
        return None


async def _get_agent_ws_info(user_id: str) -> Optional[tuple[str, str]]:
    """Return (agent_ws_url, agent_api_key) for the user's deployed agent."""
    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(
                    AgentConfig.user_id == user_id,
                    AgentConfig.deploy_status == "active",
                )
            )
            row = result.first()
    except Exception:
        return None  # agent_configs table may not exist on agent DBs
    if row and row.agent_url and row.agent_api_key:
            ws_url = row.agent_url.replace("https://", "wss://").replace("http://", "ws://")
            if not ws_url.endswith("/"):
                ws_url += "/api/ws/browser"
            else:
                ws_url += "api/ws/browser"
            return (ws_url, row.agent_api_key)
    return None


@router.websocket("/ws/browser")
async def ws_browser_proxy(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    WebSocket proxy: toup.ai browser page <-> platform <-> agent VPS browser.
    """
    await websocket.accept()
    user_id: Optional[str] = None

    if token:
        user_id = await _authenticate_ws(token)

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
    agent_info = None
    for attempt in range(6):
        agent_info = await _get_agent_ws_info(user_id)
        if agent_info:
            break
        if attempt < 5:
            await asyncio.sleep(5)

    if not agent_info:
        await websocket.send_json({"type": "error", "message": "No active agent found. Deploy your agent first."})
        await websocket.close(code=4404, reason="No agent")
        return

    agent_ws_url, agent_api_key = agent_info
    full_url = f"{agent_ws_url}?agent_key={agent_api_key}"

    logger.info("[WS Browser Proxy] User %s -> Agent %s", user_id, agent_ws_url)

    try:
        import websockets
    except ImportError:
        await websocket.send_json({"type": "error", "message": "WebSocket proxy not available"})
        await websocket.close(code=4500)
        return

    # Connect to agent — browser messages can be large (screenshots)
    agent_ws = None
    for attempt in range(3):
        try:
            agent_ws = await asyncio.wait_for(
                websockets.connect(
                    full_url,
                    max_size=20 * 1024 * 1024,  # 20MB for screenshots
                    ping_interval=30,
                    ping_timeout=120,  # Agent may take long to process
                    close_timeout=10,
                ),
                timeout=15.0,
            )
            break
        except asyncio.TimeoutError:
            if attempt == 2:
                await websocket.send_json({"type": "error", "message": "Agent connection timed out"})
                await websocket.close(code=4504)
                return
            await asyncio.sleep(3)
        except Exception as e:
            if attempt == 2:
                await websocket.send_json({"type": "error", "message": f"Cannot reach agent: {e}"})
                await websocket.close(code=4502)
                return
            await asyncio.sleep(3)

    if not agent_ws:
        await websocket.send_json({"type": "error", "message": "Failed to connect to agent"})
        await websocket.close(code=4502)
        return

    logger.info("[WS Browser Proxy] Connected to agent browser for user %s", user_id)

    async def browser_to_agent():
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

    try:
        await asyncio.gather(browser_to_agent(), agent_to_browser())
    except Exception as e:
        logger.info("[WS Browser Proxy] Session ended for user %s: %s", user_id, e)
    finally:
        if agent_ws and not agent_ws.closed:
            await agent_ws.close()
        logger.info("[WS Browser Proxy] Disconnected user %s", user_id)
