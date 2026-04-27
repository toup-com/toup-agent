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

from fastapi import WebSocket, WebSocketDisconnect

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


async def safe_send_close_ws(
    websocket: WebSocket,
    code: int,
    message: str,
    reason: str = "Unauthorized",
) -> None:
    """Send the {type:error, message:...} frame and close, swallowing
    the WebSocketDisconnect / RuntimeError race that occurs when the
    client has already closed."""
    try:
        await websocket.send_json({"type": "error", "message": message})
    except (WebSocketDisconnect, RuntimeError):
        pass
    try:
        await websocket.close(code=code, reason=reason)
    except (WebSocketDisconnect, RuntimeError):
        pass
