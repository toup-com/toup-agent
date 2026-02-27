"""
Agent Tunnel — Platform-side WebSocket endpoint for terminal agent connections.

When a user runs `toup run` in their terminal, the agent establishes a persistent
WebSocket connection to this endpoint. The platform uses this tunnel to dispatch
voice tool calls to the terminal agent, which executes them with full computer access.

Architecture:
  Terminal Agent ──WS (outbound)──→ Platform (this endpoint)
  Browser ──WS──→ Platform ──(OpenAI Realtime API)──→ OpenAI
                       └──(tool calls via tunnel)──→ Terminal Agent

Protocol:
  Agent → Platform:
    { "type": "pong" }                              — heartbeat response
    { "type": "tool_result", "id": "...", "result": "..." } — tool execution result

  Platform → Agent:
    { "type": "ping" }                              — heartbeat
    { "type": "tool_call", "id": "...", "tool_name": "...", "arguments": {...} }
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agent Tunnel"])

# ── Active tunnel connections (user_id → TunnelConnection) ───────────
_tunnels: dict[str, "TunnelConnection"] = {}

# Pending tool call futures (call_id → asyncio.Future)
_pending_calls: dict[str, asyncio.Future] = {}

TOOL_CALL_TIMEOUT = 120  # seconds — tools like exec/browser can take a while


class TunnelConnection:
    """Tracks an active tunnel WebSocket from a terminal agent."""

    def __init__(self, user_id: str, ws: WebSocket):
        self.user_id = user_id
        self.ws = ws
        self.connected_at = time.time()
        self.last_pong = time.time()

    @property
    def uptime(self) -> float:
        return time.time() - self.connected_at


def get_tunnel(user_id: str) -> Optional[TunnelConnection]:
    """Get the active tunnel for a user, if any."""
    return _tunnels.get(user_id)


def is_agent_connected(user_id: str) -> bool:
    """Check if a user's terminal agent is connected."""
    return user_id in _tunnels


async def send_tool_call(user_id: str, tool_name: str, arguments: dict) -> str:
    """Send a tool call through the tunnel and wait for the result.

    Called by ws_realtime.py when a voice tool call needs to be executed
    on the user's terminal agent.

    Returns the tool result string, or an error message.
    """
    tunnel = _tunnels.get(user_id)
    if not tunnel:
        return "ERROR: Terminal agent not connected. Run `toup run` in your terminal."

    call_id = str(uuid.uuid4())
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    _pending_calls[call_id] = future

    try:
        # Send tool call to agent
        await tunnel.ws.send_json({
            "type": "tool_call",
            "id": call_id,
            "tool_name": tool_name,
            "arguments": arguments,
        })
        logger.info("[TUNNEL] Sent tool_call %s → agent %s", tool_name, user_id[:8])

        # Wait for result with timeout
        result = await asyncio.wait_for(future, timeout=TOOL_CALL_TIMEOUT)
        return result

    except asyncio.TimeoutError:
        logger.warning("[TUNNEL] Tool call %s timed out for %s", tool_name, user_id[:8])
        return f"ERROR: Tool '{tool_name}' timed out after {TOOL_CALL_TIMEOUT}s"
    except WebSocketDisconnect:
        return "ERROR: Terminal agent disconnected during tool execution"
    except Exception as e:
        logger.exception("[TUNNEL] Tool call %s failed", tool_name)
        return f"ERROR: {e}"
    finally:
        _pending_calls.pop(call_id, None)


async def _authenticate_tunnel(token: str) -> Optional[str]:
    """Validate JWT token and return user_id."""
    try:
        from app.api.auth import verify_token
        payload = verify_token(token)
        return payload.get("sub") or payload.get("user_id")
    except Exception:
        return None


@router.websocket("/ws/agent-tunnel")
async def agent_tunnel_ws(
    websocket: WebSocket,
    token: str = Query(None),
):
    """WebSocket endpoint for terminal agent tunnel connections.

    The terminal agent connects here on startup. The platform uses this
    tunnel to dispatch voice tool calls to the agent.
    """
    await websocket.accept()

    # ── Authenticate ──
    user_id = None
    if token:
        user_id = await _authenticate_tunnel(token)

    if not user_id:
        await websocket.send_json({"type": "error", "message": "Authentication failed"})
        await websocket.close(code=4401)
        return

    # ── Register tunnel ──
    old_tunnel = _tunnels.get(user_id)
    if old_tunnel:
        logger.info("[TUNNEL] Replacing existing tunnel for %s", user_id[:8])
        try:
            await old_tunnel.ws.close(code=4000)
        except Exception:
            pass

    tunnel = TunnelConnection(user_id, websocket)
    _tunnels[user_id] = tunnel
    logger.info("[TUNNEL] Agent connected for user %s", user_id[:8])

    await websocket.send_json({"type": "connected", "user_id": user_id})

    # ── Heartbeat task ──
    async def heartbeat():
        while True:
            try:
                await asyncio.sleep(30)
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    heartbeat_task = asyncio.create_task(heartbeat())

    # ── Message loop ──
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "pong":
                tunnel.last_pong = time.time()

            elif msg_type == "tool_result":
                call_id = msg.get("id")
                result = msg.get("result", "")
                future = _pending_calls.get(call_id)
                if future and not future.done():
                    future.set_result(result)
                    logger.info("[TUNNEL] Got tool_result for %s", call_id[:8])

            elif msg_type == "status":
                logger.info("[TUNNEL] Agent status: %s", msg.get("status"))

    except WebSocketDisconnect:
        logger.info("[TUNNEL] Agent disconnected for user %s", user_id[:8])
    except Exception as e:
        logger.exception("[TUNNEL] Tunnel error for %s", user_id[:8])
    finally:
        heartbeat_task.cancel()
        # Clean up tunnel
        if _tunnels.get(user_id) is tunnel:
            del _tunnels[user_id]
        # Cancel any pending tool calls
        for call_id, future in list(_pending_calls.items()):
            if not future.done():
                future.set_exception(
                    ConnectionError("Terminal agent disconnected")
                )


@router.get("/agent/tunnel-status")
async def tunnel_status(user_id: str = Query(None)):
    """Check if a user's terminal agent is connected via tunnel.

    Accepts user_id as query param (for authenticated API calls).
    """
    if not user_id:
        return {"connected": False, "error": "user_id required"}

    tunnel = _tunnels.get(user_id)
    if not tunnel:
        return {"connected": False}

    return {
        "connected": True,
        "uptime_seconds": round(tunnel.uptime, 1),
        "last_pong": round(time.time() - tunnel.last_pong, 1),
    }


@router.get("/agent/tunnel-status/me")
async def tunnel_status_me(
    token: str = Query(None),
):
    """Check if the current user's terminal agent is connected.

    Uses JWT auth (same as other endpoints) — no need to pass user_id.
    """
    user_id = None
    if token:
        user_id = await _authenticate_tunnel(token)

    if not user_id:
        return {"connected": False, "error": "auth required"}

    tunnel = _tunnels.get(user_id)
    if not tunnel:
        return {"connected": False}

    return {
        "connected": True,
        "uptime_seconds": round(tunnel.uptime, 1),
    }
