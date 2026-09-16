"""Shared WS auth-path helpers — Ticket 1 / ST-2.

Two responsibilities:

  1. `accept_with_subprotocol_auth(websocket)` — read subprotocols
     pre-accept, extract a `bearer.<token>` if present, accept with
     `toup.auth.v1` echoed back so browsers complete the handshake.
     Returns the extracted token (None if no subprotocol auth supplied).

  2. `safe_send_close_ws(websocket, code, message, reason)` — send the
     standard error frame + close, swallowing the race where the
     client already disconnected. Used in every WS auth-fail path so
     a client that drops between auth-fail and our send doesn't
     surface an ASGI traceback.

Design notes:

- `accept_with_subprotocol_auth` always calls `websocket.accept()`
  (with or without the subprotocol kwarg). The endpoint should not
  call accept itself afterward.
- `safe_send_close_ws` is idempotent against a dead connection; calling
  it after the client has disconnected is a no-op.
- The deprecation log line is the load-bearing string for the bake-
  window metric (`[DEPRECATED-WS-AUTH]`) — `log_deprecated_query_token`
  emits it with a per-endpoint label so future grep can tell which
  endpoint is still seeing legacy traffic.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import WebSocket

from app.api._fault_codes import FAULTS, close_code_for, fault_frame

logger = logging.getLogger(__name__)


SUBPROTOCOL_VERSION = "toup.auth.v1"
SUBPROTOCOL_BEARER_PREFIX = "bearer."


async def accept_with_subprotocol_auth(websocket: WebSocket) -> Optional[str]:
    """Accept the WebSocket, extracting a JWT from the Sec-WebSocket-
    Protocol negotiation if the client supplied one. Returns the token
    or None.

    Safe to call on a connection that supplies no subprotocols — the
    websocket is accepted with no subprotocol echo and the caller
    proceeds with its existing auth fallbacks (?token= / first-frame).
    """
    subprotocols = list(websocket.scope.get("subprotocols") or [])
    token: Optional[str] = None
    selected: Optional[str] = None

    if SUBPROTOCOL_VERSION in subprotocols:
        for sp in subprotocols:
            if sp.startswith(SUBPROTOCOL_BEARER_PREFIX):
                token = sp[len(SUBPROTOCOL_BEARER_PREFIX):]
                break
        if token:
            selected = SUBPROTOCOL_VERSION

    if selected:
        await websocket.accept(subprotocol=selected)
    else:
        await websocket.accept()
    return token


def log_deprecated_query_token(endpoint_label: str) -> None:
    """Emit the load-bearing [DEPRECATED-WS-AUTH] marker. Call only when
    the ?token= fallback was actually used for auth (not just present)."""
    logger.warning(
        "[DEPRECATED-WS-AUTH] %s used ?token= query param for auth — "
        "migrate client to subprotocol header",
        endpoint_label,
    )


def log_deprecated_agent_key_url(endpoint_label: str) -> None:
    """Vault Ticket 1 / ST-3: parallel marker for `agent_key` migration
    from URL query param to `X-Agent-Key` header. Call only when the
    URL fallback was actually used for auth (not just present), so a
    future grep "[DEPRECATED-AGENT-KEY-URL]" reflects live exposure."""
    logger.warning(
        "[DEPRECATED-AGENT-KEY-URL] %s used ?agent_key= query param for auth — "
        "migrate caller to X-Agent-Key request header",
        endpoint_label,
    )


def log_deprecated_http_query_token(endpoint_label: str) -> None:
    """Vault Ticket 1 / ST-4a: parallel marker for HTTP endpoint JWT
    migration from `?token=` URL query param to `Authorization: Bearer`
    header. Distinct from [DEPRECATED-WS-AUTH] (WebSocket subprotocol
    migration) so the two cutover gates are independently observable —
    ST-3 / WS bake clock and ST-4a / HTTP bake clock have separate
    7-day-zero windows. Call only when the URL fallback was actually
    used for auth (not just present)."""
    logger.warning(
        "[DEPRECATED-HTTP-AUTH] %s used ?token= URL param for auth — "
        "migrate caller to Authorization: Bearer header",
        endpoint_label,
    )


async def safe_send_close_ws(
    websocket: WebSocket,
    code: int,
    message: str,
    reason: str = "Unauthorized",
    *,
    fault_code: Optional[str] = None,
    retry_after_ms: Optional[int] = None,
) -> None:
    """Send the {type:error, message:...} frame and close, swallowing
    the WebSocketDisconnect / RuntimeError race that occurs when the
    client has already closed.

    `fault_code` stamps a machine code from `_fault_codes.FAULTS` onto the
    frame. Round 46 incident 3: this frame had no `code` key at all, which
    is why ws_chat_proxy has to sniff the agent's 4001 rejection by its
    English text (ws_chat_proxy.py ~1322) and why the app ended up classing
    a database outage with a regex over prose."""
    frame: dict = {"type": "error", "message": message}
    if fault_code:
        frame["code"] = fault_code
        entry = FAULTS.get(fault_code)
        if entry is not None:
            frame["retryable"] = bool(entry["retryable"])
            after = retry_after_ms if retry_after_ms is not None else entry["retry_after_ms"]
            if after is not None:
                frame["retry_after_ms"] = int(after)
    # `except Exception`, not the two named classes: a fault REPORTER may
    # never be the thing that raises. uvicorn turns a send on a half-dead
    # socket into `uvicorn.protocols.utils.ClientDisconnected(OSError)` and
    # the pinned 0.27.0 lets the raw `websockets.ConnectionClosed` through —
    # neither is a RuntimeError or a WebSocketDisconnect, so both escaped this
    # helper, escaped `ws_chat`'s outer net, and came out of uvicorn as a bare
    # `transport.close()`: close code 1006, which is the incident-3 path this
    # round exists to eliminate.
    try:
        await websocket.send_json(frame)
    except Exception:  # noqa: BLE001 — see above
        pass
    try:
        await websocket.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001
        pass


async def safe_send_fault_ws(
    websocket: WebSocket,
    code: str,
    *,
    retry_after_ms: Optional[int] = None,
    detail: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    """The typed twin of `safe_send_close_ws`: send `fault_frame(code)` then
    close with that fault's close code.

    Both halves matter. The close code is lost on several paths (a browser
    that never surfaces it, a proxy hop that rewrites it — 1006 was
    laundered to 1000 in incident 3); the frame's `code` survives them, and
    both clients already class an error FRAME by code, so an app build that
    predates these numbers still gets an honest classification.

    `detail` is machine-oriented and must never carry user content."""
    frame = fault_frame(code, retry_after_ms=retry_after_ms, detail=detail)
    # Same rule as `safe_send_close_ws`: swallow EVERYTHING. This helper is
    # now the only handler on ws_chat's outer safety net, so anything it lets
    # escape becomes a bare 1006 at the peer.
    try:
        await websocket.send_json(frame)
    except Exception:  # noqa: BLE001
        pass
    try:
        await websocket.close(code=close_code_for(code), reason=reason or code)
    except Exception:  # noqa: BLE001
        pass
