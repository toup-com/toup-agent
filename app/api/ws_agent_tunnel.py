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
    { "type": "restart" }                            — graceful restart (settings changed)
    { "type": "config_update", "env_content": "..." } — write .env + restart (live settings sync)
"""

import asyncio
from dataclasses import dataclass
import json
import logging
import math
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, Query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agent Tunnel"])

# ── Active tunnel connections (user_id → TunnelConnection) ───────────
_tunnels: dict[str, "TunnelConnection"] = {}

# This state is intentionally process-local. Railway replicas do not share a
# tunnel map, so neither this module nor /agent/tunnel-status may claim a
# global pending-call census.
#
# Pending calls are owned by the exact user AND tunnel instance that received
# them.  A replacement connection for one tenant must not cancel another
# tenant's future (or adopt the retiring connection's potentially-side-effecting
# call).
@dataclass(frozen=True)
class PendingCall:
    user_id: str
    tunnel: "TunnelConnection"
    future: asyncio.Future


_pending_calls: dict[str, PendingCall] = {}

TOOL_CALL_TIMEOUT = 120  # seconds — tools like exec/browser can take a while
HTTP_FORWARD_MAX_TIMEOUT = 30.0
TUNNEL_REPLACE_CLOSE_TIMEOUT = 2.0
TUNNEL_CONTROL_SEND_TIMEOUT = 5.0
TUNNEL_CONFIG_SYNC_TIMEOUT = 30.0


class TunnelDisconnected(ConnectionError):
    """The exact tunnel owning an already-dispatched call disconnected."""


class HTTPForwardOutcomeUnknown(RuntimeError):
    """A mutating HTTP request may have reached the agent; never replay it."""


class TunnelConnection:
    """Tracks an active tunnel WebSocket from a terminal agent."""

    def __init__(self, user_id: str, ws: WebSocket):
        self.user_id = user_id
        self.ws = ws
        self.connected_at = time.time()
        self.last_pong = time.time()
        self.accepting_calls = False

    @property
    def uptime(self) -> float:
        return time.time() - self.connected_at


def _new_pending(user_id: str, tunnel: TunnelConnection) -> tuple[str, PendingCall]:
    """Register one call before its first await, bound to this connection."""
    call_id = str(uuid.uuid4())
    pending = PendingCall(
        user_id=user_id,
        tunnel=tunnel,
        future=asyncio.get_running_loop().create_future(),
    )
    _pending_calls[call_id] = pending
    return call_id, pending


def _discard_pending(call_id: str, pending: PendingCall) -> None:
    """Remove only this registration and settle its future for clean teardown."""
    if _pending_calls.get(call_id) is pending:
        _pending_calls.pop(call_id, None)
    if not pending.future.done():
        pending.future.cancel()
    elif not pending.future.cancelled():
        # Retrieve a disconnect exception even when dispatch/send failed before
        # the caller reached its future await. Results return None here.
        pending.future.exception()


def _resolve_pending(
    tunnel: TunnelConnection, call_id: object, result: object,
) -> bool:
    """Accept a result only from the connection that received the call."""
    if not isinstance(call_id, str):
        return False
    pending = _pending_calls.get(call_id)
    if (
        pending is None
        or pending.tunnel is not tunnel
        or pending.user_id != tunnel.user_id
        or pending.future.done()
    ):
        return False
    pending.future.set_result(result)
    return True


def _fail_pending_for_tunnel(tunnel: TunnelConnection) -> int:
    """Fail only calls dispatched to this exact retiring connection."""
    failed = 0
    for pending in list(_pending_calls.values()):
        if pending.tunnel is tunnel and not pending.future.done():
            pending.future.set_exception(
                TunnelDisconnected("owning agent tunnel disconnected after dispatch")
            )
            failed += 1
    return failed


def _local_pending_count(user_id: str) -> int:
    """Test/diagnostic helper; deliberately not a cross-replica claim."""
    return sum(
        pending.user_id == user_id and not pending.future.done()
        for pending in _pending_calls.values()
    )


async def _install_tunnel(tunnel: TunnelConnection) -> TunnelConnection | None:
    """Publish replacement before closing old so old cleanup cannot erase new."""
    old = _tunnels.get(tunnel.user_id)
    _tunnels[tunnel.user_id] = tunnel
    if old is not None and old is not tunnel:
        old.accepting_calls = False
        try:
            await asyncio.wait_for(
                old.ws.close(code=4000), timeout=TUNNEL_REPLACE_CLOSE_TIMEOUT,
            )
        except asyncio.CancelledError:
            # Publication precedes the bounded close. If this handler is
            # cancelled in that window, do not leave its connection current.
            _retire_tunnel(tunnel)
            raise
        except Exception:
            pass
    return old


def _retire_tunnel(tunnel: TunnelConnection) -> int:
    """Remove only a still-current tunnel, but always retire its own calls."""
    tunnel.accepting_calls = False
    if _tunnels.get(tunnel.user_id) is tunnel:
        del _tunnels[tunnel.user_id]
    return _fail_pending_for_tunnel(tunnel)


def _http_forward_failed_after_dispatch(method: str) -> None:
    """Permit direct fallback only for methods that are safe to replay."""
    if method.upper() in {"GET", "HEAD"}:
        return
    raise HTTPForwardOutcomeUnknown(
        "agent outcome unknown after tunnel dispatch; request was not replayed"
    )


def _reject_nonfinite_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _validated_http_result(method: str, result: object) -> object | None:
    """Return finite JSON only; invalid post-dispatch data is uncertainty."""
    try:
        parsed = (
            json.loads(result, parse_constant=_reject_nonfinite_json_constant)
            if isinstance(result, str)
            else result
        )
        if parsed is None:
            raise ValueError("null tunnel response")
        # Native dict/list results bypass json.loads, so recursively reject
        # NaN/Infinity and other values JSONResponse cannot safely encode.
        json.dumps(parsed, allow_nan=False)
    except (TypeError, ValueError):
        _http_forward_failed_after_dispatch(method)
        return None
    return parsed


async def _send_current_control(
    tunnel: TunnelConnection, payload: dict[str, object],
) -> bool:
    """Send setup control or retire this exact tunnel on transport failure."""
    if _tunnels.get(tunnel.user_id) is not tunnel:
        _retire_tunnel(tunnel)
        return False
    try:
        await asyncio.wait_for(
            tunnel.ws.send_json(payload), timeout=TUNNEL_CONTROL_SEND_TIMEOUT,
        )
    except Exception:
        _retire_tunnel(tunnel)
        return False
    if _tunnels.get(tunnel.user_id) is not tunnel:
        _retire_tunnel(tunnel)
        return False
    return True


def get_tunnel(user_id: str) -> Optional[TunnelConnection]:
    """Get the active tunnel for a user, if any."""
    return _tunnels.get(user_id)


def _get_ready_tunnel(user_id: str) -> Optional[TunnelConnection]:
    tunnel = _tunnels.get(user_id)
    if tunnel is None or not tunnel.accepting_calls:
        return None
    return tunnel


def _mark_tunnel_ready(tunnel: TunnelConnection) -> bool:
    """Admit calls only after setup, and only if still the current tunnel."""
    if _tunnels.get(tunnel.user_id) is not tunnel:
        return False
    tunnel.accepting_calls = True
    return True


def is_agent_connected(user_id: str) -> bool:
    """Check if a user's terminal agent is connected."""
    return _get_ready_tunnel(user_id) is not None


async def send_tool_call(user_id: str, tool_name: str, arguments: dict) -> str:
    """Send a tool call through the tunnel and wait for the result.

    Called by ws_realtime.py when a voice tool call needs to be executed
    on the user's terminal agent.

    Returns the tool result string, or an error message.
    """
    tunnel = _get_ready_tunnel(user_id)
    if not tunnel:
        return "ERROR: Terminal agent not connected. Run `toup run` in your terminal."

    call_id, pending = _new_pending(user_id, tunnel)

    try:
        # The single deadline covers both potentially-backpressured dispatch
        # and the result wait. Entering send makes the outcome uncertain.
        async with asyncio.timeout(float(TOOL_CALL_TIMEOUT)):
            await tunnel.ws.send_json({
                "type": "tool_call",
                "id": call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "user_id": user_id,
            })
            logger.info(
                "[TUNNEL] Sent tool_call %s → agent %s",
                tool_name,
                user_id[:8],
            )
            result = await pending.future
        return result

    except asyncio.TimeoutError:
        logger.warning("[TUNNEL] Tool call %s timed out for %s", tool_name, user_id[:8])
        return (
            f"ERROR: Tool '{tool_name}' outcome is unknown after the finite "
            f"{TOOL_CALL_TIMEOUT}s tunnel timeout; do not retry automatically."
        )
    except (TunnelDisconnected, WebSocketDisconnect):
        return (
            f"ERROR: Tool '{tool_name}' tunnel disconnected after dispatch; "
            "outcome is unknown, so do not retry automatically."
        )
    except Exception as e:
        logger.exception("[TUNNEL] Tool call %s failed", tool_name)
        return (
            f"ERROR: Tool '{tool_name}' dispatch failed with "
            f"{type(e).__name__}; outcome is unknown, so do not retry "
            "automatically."
        )
    finally:
        _discard_pending(call_id, pending)


async def send_http_forward(
    user_id: str, method: str, path: str, headers: dict | None = None, body: dict | None = None,
    timeout: float = 15.0,
) -> dict | None:
    """Forward an HTTP request through the tunnel to the agent.

    Used when the agent is behind NAT and can't be reached directly.
    Returns the parsed JSON response, or None on failure.
    """
    tunnel = _get_ready_tunnel(user_id)
    if not tunnel:
        return None
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= HTTP_FORWARD_MAX_TIMEOUT
    ):
        raise ValueError(
            f"timeout must be finite and in (0, {HTTP_FORWARD_MAX_TIMEOUT}]"
        )

    call_id, pending = _new_pending(user_id, tunnel)

    try:
        # One total deadline covers send backpressure plus response latency.
        # Once send is entered a mutating request must never be replayed.
        async with asyncio.timeout(float(timeout)):
            await tunnel.ws.send_json({
                "type": "http_forward",
                "id": call_id,
                "method": method,
                "path": path,
                "headers": headers or {},
                "body": body,
            })
            result = await pending.future
        return _validated_http_result(method, result)
    except asyncio.TimeoutError:
        logger.warning("[TUNNEL] HTTP forward %s %s timed out for %s", method, path, user_id[:8])
        _http_forward_failed_after_dispatch(method)
        return None
    except (TunnelDisconnected, WebSocketDisconnect):
        _http_forward_failed_after_dispatch(method)
        return None
    except HTTPForwardOutcomeUnknown:
        raise
    except Exception as e:
        logger.warning("[TUNNEL] HTTP forward failed: %s", e)
        _http_forward_failed_after_dispatch(method)
        return None
    finally:
        _discard_pending(call_id, pending)


async def send_restart(user_id: str) -> bool:
    """Send a restart command to the terminal agent via tunnel.

    Called when agent settings are saved on the platform.
    The agent will gracefully restart to pick up new config.
    """
    tunnel = _get_ready_tunnel(user_id)
    if not tunnel:
        logger.info("[TUNNEL] No active tunnel for %s — cannot send restart", user_id[:8])
        return False

    try:
        await tunnel.ws.send_json({"type": "restart"})
        logger.info("[TUNNEL] Sent restart command to agent %s", user_id[:8])
        return True
    except Exception as e:
        logger.warning("[TUNNEL] Failed to send restart to %s: %s", user_id[:8], e)
        return False


async def send_config_update(user_id: str, env_content: str) -> bool:
    """Send updated .env content to the terminal agent via tunnel.

    The agent will write the new .env and restart to pick up changes.
    This is the mechanism that makes Agent Settings a live control panel.
    """
    tunnel = _get_ready_tunnel(user_id)
    if not tunnel:
        logger.info("[TUNNEL] No active tunnel for %s — cannot push config", user_id[:8])
        return False

    try:
        await tunnel.ws.send_json({
            "type": "config_update",
            "env_content": env_content,
        })
        logger.info("[TUNNEL] Sent config_update to agent %s (%d bytes)", user_id[:8], len(env_content))
        return True
    except Exception as e:
        logger.warning("[TUNNEL] Failed to send config_update to %s: %s", user_id[:8], e)
        return False


async def _authenticate_tunnel(token: str) -> Optional[str]:
    """Validate tunnel token and return user_id.

    Accepts two token types:
    1. Connect token (toup_ct_...) — looked up in AgentConfig.connect_token
    2. JWT token — verified via standard auth (fallback)
    """
    # 1. Connect token (from dashboard "Generate Connect Token")
    if token.startswith("toup_ct_"):
        try:
            from sqlalchemy import text
            from app.db.database import engine
            async with engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT user_id FROM agent_configs WHERE connect_token = :token"),
                    {"token": token},
                )
                row = result.first()
                if row:
                    logger.info("[TUNNEL] Token auth OK for user %s", row[0][:8])
                    return row[0]
                else:
                    logger.warning("[TUNNEL] Token not found in DB (len=%d)", len(token))
        except Exception as e:
            logger.exception("[TUNNEL] Token auth DB error: %s", e)
        return None

    # 2. JWT token (fallback — used by tunnel-status/me from frontend)
    try:
        from app.services import decode_access_token
        user_id = decode_access_token(token)
        if user_id:
            return user_id
    except Exception as e:
        logger.warning("[TUNNEL] JWT auth failed: %s", e)
    return None


async def _push_latest_config(tunnel: TunnelConnection) -> bool:
    """Best-effort DB config read; fail closed only on a transport send."""
    try:
        from app.db.database import async_session_maker
        from app.db.models import AgentConfig
        from sqlalchemy import select as _sel

        async with async_session_maker() as db:
            result = await db.execute(
                _sel(AgentConfig).where(AgentConfig.user_id == tunnel.user_id)
            )
            config = result.scalars().first()
            if not config:
                return True

            from app.api.agent_setup import _build_env, _automations_env_flag

            env = _build_env(
                config,
                tunnel.user_id,
                automations_enabled=await _automations_env_flag(tunnel.user_id),
            )
            if not await _send_current_control(
                tunnel, {"type": "config_update", "env_content": env},
            ):
                return False
            logger.info(
                "[TUNNEL] Pushed config sync on connect for %s",
                tunnel.user_id[:8],
            )
    except Exception as error:
        logger.warning(
            "[TUNNEL] Config sync on connect failed for %s: %s",
            tunnel.user_id[:8],
            error,
        )
    return True


@router.websocket("/ws/agent-tunnel")
async def agent_tunnel_ws(
    websocket: WebSocket,
    token: str = Query(None),
):
    """WebSocket endpoint for terminal agent tunnel connections.

    The terminal agent connects here on startup. The platform uses this
    tunnel to dispatch voice tool calls to the agent.
    """
    # ST-2: accept + subprotocol JWT/connect-token extraction. Falls
    # back to ?token= for the bake window.
    from app.api._ws_auth_helpers import (
        accept_with_subprotocol_auth,
        log_deprecated_query_token,
    )
    subprotocol_token = await accept_with_subprotocol_auth(websocket)

    # ── Authenticate ──
    # Order: subprotocol → ?token= (deprecated) → first-frame.
    if not token:
        token = websocket.query_params.get("token")

    user_id = None
    if subprotocol_token:
        user_id = await _authenticate_tunnel(subprotocol_token)
        if user_id:
            token = subprotocol_token

    if not user_id and token:
        log_deprecated_query_token("/api/ws/agent-tunnel")
        logger.info("[TUNNEL] Auth — query param token len=%d",
                    len(token) if token else 0)
        user_id = await _authenticate_tunnel(token)

    # If query param auth failed, wait for auth message from client
    client_disconnected = False
    if not user_id:
        logger.info("[TUNNEL] Query/subprotocol auth failed, waiting for auth message...")
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("type") == "auth" and msg.get("token"):
                token = msg["token"]
                logger.info("[TUNNEL] Got auth message — token len=%d", len(token))
                user_id = await _authenticate_tunnel(token)
        except asyncio.TimeoutError:
            logger.warning("[TUNNEL] No auth message received within 10s")
        except WebSocketDisconnect:
            client_disconnected = True
        except Exception as e:
            logger.warning("[TUNNEL] Error reading auth message: %s", e)

    if not user_id:
        logger.warning(
            "[TUNNEL] Auth failed — token len=%d",
            len(token) if token else 0,
        )
        if client_disconnected:
            return
        from app.api._ws_auth_helpers import safe_send_close_ws
        await safe_send_close_ws(
            websocket, code=4401, message="Authentication failed",
        )
        return

    # ── Register tunnel ──
    tunnel = TunnelConnection(user_id, websocket)
    old_tunnel = _tunnels.get(user_id)
    if old_tunnel is not None:
        logger.info("[TUNNEL] Replacing existing tunnel for %s", user_id[:8])
    installed = False
    heartbeat_task: asyncio.Task | None = None
    try:
        await _install_tunnel(tunnel)
        installed = True
        logger.info("[TUNNEL] Agent connected for user %s", user_id[:8])

        if not await _send_current_control(
            tunnel, {"type": "connected", "user_id": user_id},
        ):
            return
        try:
            config_ready = await asyncio.wait_for(
                _push_latest_config(tunnel), timeout=TUNNEL_CONFIG_SYNC_TIMEOUT,
            )
        except asyncio.TimeoutError:
            _retire_tunnel(tunnel)
            return
        if not config_ready:
            return
        if not _mark_tunnel_ready(tunnel):
            return

        # ── Heartbeat task ──
        async def heartbeat():
            while True:
                try:
                    await asyncio.sleep(15)
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break

        heartbeat_task = asyncio.create_task(heartbeat())

        # ── Message loop ──
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "pong":
                tunnel.last_pong = time.time()

            elif msg_type == "tool_result":
                call_id = msg.get("id")
                result = msg.get("result", "")
                if _resolve_pending(tunnel, call_id, result):
                    logger.info("[TUNNEL] Got tool_result for %s", call_id[:8])

            elif msg_type == "status":
                logger.info("[TUNNEL] Agent status: %s", msg.get("status"))

    except WebSocketDisconnect:
        logger.info("[TUNNEL] Agent disconnected for user %s", user_id[:8])
    except Exception:
        logger.exception("[TUNNEL] Tunnel error for %s", user_id[:8])
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
        # A replacement is published before its predecessor is closed.  The
        # retiring predecessor may fail only calls it actually received and
        # must never erase the new current tunnel or another tenant's work.
        if installed:
            _retire_tunnel(tunnel)


@router.get("/agent/tunnel-status")
async def tunnel_status(
    request: Request,
    user_id: str = Query(None),
    token: str = Query(None),
):
    """Check if the CALLER's own terminal agent is connected via tunnel.

    Authz (re-audit round 6 IDOR fix): the caller is derived from their own
    bearer/token — another tenant's tunnel presence is NEVER disclosed. A
    client-supplied ``user_id`` must equal the authenticated user or nothing is
    returned. (Previously this took ``user_id`` from the query with no auth,
    making it a cross-tenant presence oracle.)
    """
    from app.api._ws_auth_helpers import log_deprecated_http_query_token

    caller = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        header_token = auth_header[7:].strip()
        if header_token:
            caller = await _authenticate_tunnel(header_token)
    if not caller and token:
        caller = await _authenticate_tunnel(token)
        if caller:
            log_deprecated_http_query_token("/api/agent/tunnel-status")

    if not caller:
        return {"connected": False, "error": "authentication required"}
    if user_id and user_id != caller:
        return {"connected": False}  # never disclose another tenant's presence

    tunnel = _get_ready_tunnel(caller)
    if not tunnel:
        return {"connected": False}

    return {
        "connected": True,
        "uptime_seconds": round(tunnel.uptime, 1),
        "last_pong": round(time.time() - tunnel.last_pong, 1),
    }


@router.get("/agent/tunnel-status/me")
async def tunnel_status_me(
    request: Request,
    token: str = Query(None),
):
    """Check if the current user's terminal agent is connected.

    Auth order (Ticket 1 / ST-4a):
      1. Authorization: Bearer <token> header (preferred)
      2. ?token= URL query param (deprecated — bake-window fallback,
         emits [DEPRECATED-HTTP-AUTH] when used so the cutover gate
         can detect zero usage before the legacy path is removed)
    """
    from app.api._ws_auth_helpers import log_deprecated_http_query_token

    user_id = None

    # 1. Authorization header (preferred)
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        header_token = auth_header[7:].strip()
        if header_token:
            user_id = await _authenticate_tunnel(header_token)

    # 2. ?token= URL fallback (deprecated)
    if not user_id and token:
        user_id = await _authenticate_tunnel(token)
        if user_id:
            log_deprecated_http_query_token("/api/agent/tunnel-status/me")

    if not user_id:
        return {"connected": False, "error": "auth required"}

    tunnel = _get_ready_tunnel(user_id)
    if not tunnel:
        return {"connected": False, "user_id": user_id}

    return {
        "connected": True,
        "uptime_seconds": round(tunnel.uptime, 1),
    }


@router.get("/agent/tunnel-debug")
async def tunnel_debug(
    request: Request,
    token: str = Query(None),
):
    """Debug endpoint: test token auth and show tunnel state.

    Auth order (Ticket 1 / ST-4a):
      1. Authorization: Bearer <token> header (preferred)
      2. ?token= URL query param (deprecated — emits [DEPRECATED-HTTP-AUTH])
    """
    from app.api._ws_auth_helpers import log_deprecated_http_query_token

    # Do NOT expose the live tenant user_id set before auth — it was previously
    # returned on the no-token path, an unauthenticated cross-tenant enumeration
    # (re-audit round 6). Populated only after the caller authenticates below.
    result: dict = {}

    user_id = None
    auth_token: Optional[str] = None

    # 1. Authorization header (preferred)
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        header_token = auth_header[7:].strip()
        if header_token:
            user_id = await _authenticate_tunnel(header_token)
            if user_id:
                auth_token = header_token
                result["auth_source"] = "header"

    # 2. ?token= URL fallback (deprecated)
    if not user_id and token:
        user_id = await _authenticate_tunnel(token)
        if user_id:
            auth_token = token
            result["auth_source"] = "query"
            log_deprecated_http_query_token("/api/agent/tunnel-debug")

    if not auth_token:
        result["error"] = "no token provided"
        return result

    result["auth_user_id"] = user_id

    if user_id:
        result["is_connected"] = is_agent_connected(user_id)
        tunnel = _get_ready_tunnel(user_id)
        if tunnel:
            result["tunnel_uptime"] = round(tunnel.uptime, 1)

    # Also check raw DB
    try:
        from sqlalchemy import text
        from app.db.database import engine
        async with engine.begin() as conn:
            # Count total configs with connect tokens
            r = await conn.execute(text("SELECT COUNT(*) FROM agent_configs WHERE connect_token IS NOT NULL"))
            result["configs_with_tokens"] = r.scalar()
            # Check if this specific token exists. Uses the resolved
            # auth_token (header or query) so DB lookup matches whichever
            # path actually authenticated.
            if auth_token and auth_token.startswith("toup_ct_"):
                r2 = await conn.execute(
                    text("SELECT user_id, LENGTH(connect_token) FROM agent_configs WHERE connect_token = :t"),
                    {"t": auth_token},
                )
                row = r2.first()
                result["token_found_in_db"] = row is not None
                if row:
                    result["token_user_id"] = row[0]
                    result["token_length"] = row[1]
    except Exception as e:
        result["db_error"] = str(e)

    return result
