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
            print(f"[WS Proxy] JWT decode returned None (token len={len(token) if token else 0})", flush=True)
            return None
        print(f"[WS Proxy] JWT decoded OK: user_id={user_id}", flush=True)
        async with async_session_maker() as db:
            user = await get_user_by_id(db, user_id)
            if user and user.is_active:
                print(f"[WS Proxy] Auth OK: user={user_id}", flush=True)
                return user.id
            print(f"[WS Proxy] User lookup failed: user_id={user_id} found={user is not None} active={user.is_active if user else 'N/A'}", flush=True)
        return None
    except Exception as e:
        print(f"[WS Proxy] Auth exception: {type(e).__name__}: {e}", flush=True)
        return None


async def _get_agent_ws_info(user_id: str) -> Optional[tuple[str, str]]:
    """Return (agent_ws_url, agent_api_key) for the user's deployed agent."""
    async with async_session_maker() as db:
        result = await db.execute(
            select(AgentConfig.agent_url, AgentConfig.agent_api_key, AgentConfig.deploy_status)
            .where(AgentConfig.user_id == user_id)
        )
        row = result.first()
        if not row:
            print(f"[WS Proxy] No AgentConfig for user={user_id}", flush=True)
            return None
        print(f"[WS Proxy] AgentConfig: url={row.agent_url} status={row.deploy_status} has_key={bool(row.agent_api_key)}", flush=True)
        if row.deploy_status != "active":
            print(f"[WS Proxy] Agent not active (status={row.deploy_status})", flush=True)
            return None
        if not row.agent_url or not row.agent_api_key:
            print(f"[WS Proxy] Agent missing url or key", flush=True)
            return None
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
    Retries agent lookup if agent is still deploying/starting.
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

    # Look up agent info — retry a few times if agent is still deploying
    agent_info = None
    for attempt in range(6):  # Up to 30s of retries
        agent_info = await _get_agent_ws_info(user_id)
        if agent_info:
            break
        if attempt < 5:
            await asyncio.sleep(5)
        logger.info("[WS Proxy] Agent not ready for %s (attempt %d/6)", user_id, attempt + 1)

    if not agent_info:
        await websocket.send_json({"type": "error", "message": "No active agent found. Deploy your agent first."})
        await websocket.close(code=4404, reason="No agent")
        return

    agent_ws_url, agent_api_key = agent_info
    full_url = f"{agent_ws_url}?agent_key={agent_api_key}"

    logger.info("[WS Proxy] User %s → Agent %s", user_id, agent_ws_url)

    # Connect to the agent's WebSocket — retry if agent is still starting
    try:
        import websockets
    except ImportError:
        await websocket.send_json({"type": "error", "message": "WebSocket proxy not available"})
        await websocket.close(code=4500)
        return

    agent_ws = None
    for attempt in range(3):
        try:
            agent_ws = await asyncio.wait_for(
                websockets.connect(
                    full_url,
                    max_size=10 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=10,
                ),
                timeout=15.0,
            )
            break
        except asyncio.TimeoutError:
            if attempt == 2:
                await websocket.send_json({"type": "error", "message": "Agent connection timed out"})
                await websocket.close(code=4504, reason="Agent timeout")
                return
            await asyncio.sleep(3)
        except Exception as e:
            if attempt == 2:
                logger.warning("[WS Proxy] Agent connect failed: %s", e)
                await websocket.send_json({"type": "error", "message": f"Cannot reach agent: {e}"})
                await websocket.close(code=4502, reason="Agent unreachable")
                return
            await asyncio.sleep(3)

    if not agent_ws:
        await websocket.send_json({"type": "error", "message": "Failed to connect to agent"})
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
        """Forward messages from agent → browser, enriching build job events."""
        import re as _re

        # Track build jobs seen during this session for injection into 'done'
        _pending_build_jobs: list = []

        try:
            async for raw in agent_ws:
                try:
                    _d = json.loads(raw)
                    _t = _d.get("type", "?")
                    if _t not in ("text_chunk", "pong"):
                        print(f"[WS Proxy] agent→browser: type={_t}", flush=True)

                    # Capture build job info from tool_end events
                    if _t == "tool_end" and _d.get("tool") == "app_builder__build_app":
                        _summary = _d.get("summary", "")
                        _jid_m = _re.search(r"Job ID:\s*([a-f0-9-]+)", _summary, _re.I)
                        _jnm_m = _re.search(r"Building '([^']+)'", _summary)
                        if _jid_m:
                            _pending_build_jobs.append({
                                "job_id": _jid_m.group(1),
                                "job_name": _jnm_m.group(1) if _jnm_m else "App Build",
                            })
                            print(f"[WS Proxy] Captured build job: {_jid_m.group(1)}", flush=True)

                    # Inject build_jobs into 'done' events so frontend can create cards
                    if _t == "done" and _pending_build_jobs:
                        _d["build_jobs"] = list(_pending_build_jobs)
                        _pending_build_jobs.clear()
                        raw = json.dumps(_d)
                        print(f"[WS Proxy] Injected {len(_d['build_jobs'])} build_jobs into done event", flush=True)

                except (json.JSONDecodeError, Exception):
                    pass

                await websocket.send_text(raw)
        except Exception as e:
            print(f"[WS Proxy] agent_to_browser error: {e}", flush=True)
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
        if agent_ws:
            try:
                await agent_ws.close()
            except Exception:
                pass
        logger.info("[WS Proxy] Disconnected user %s", user_id)
