"""
T1g — Agent-side MCP client auth.

The platform's T0c MCPAuthMiddleware resolves user_id from `X-Agent-Key`
on every incoming MCP request. T1f's connector_mcp filter middleware
reads channel from `X-Toup-Channel` to drive the dispatcher's voice/
telegram default-deny on mutating tools. This module is the
agent-side counterpart that injects both headers on every outbound
MCP request via FastMCP's `httpx.Auth` integration point.

How channel reaches the auth class:
  - The agent runner sets `_current_channel` on the tool_executor at
    the start of each turn (`agent_runner.py:305`).
  - When the executor dispatches an MCP call, it sets a ContextVar
    (`_pending_channel`) with the active channel value, then calls
    `mcp_client.call_tool(...)`.
  - FastMCP's transport eventually hands the prepared httpx Request
    to our Auth's `auth_flow`. The Auth reads the ContextVar and
    sets the header on that request.
  - After the call completes, the executor resets the ContextVar.

For tool-list calls (boot-time discovery, `tools/list` polls), no
ContextVar is set — the Auth defaults to `web`. Matches the spec:
"For tool-list calls (which aren't tied to a single tool invocation),
default to `web`."

Design note — why not pass channel via `Client.call_tool(headers=...)`:
FastMCP 2.9 doesn't expose a per-call headers kwarg on the high-level
Client. The httpx.Auth approach works for both list_tools and call_tool
without modifying the client surface, and the ContextVar gives us
per-call scoping without a global mutable.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Generator, Optional

import httpx


# Per-call channel. Set by the executor before mcp_client.call_tool;
# defaults to "web" for boot discovery and any other unscoped call.
_pending_channel: ContextVar[Optional[str]] = ContextVar(
    "agent_mcp_channel", default=None
)


# Header constants — single source of truth, must match
# `app.mcp_auth.HEADER_*` on the platform side. Drift between the two
# would silently break auth resolution.
HEADER_AGENT_KEY = "X-Agent-Key"
HEADER_TOUP_CHANNEL = "X-Toup-Channel"


def set_pending_channel(channel: Optional[str]):
    """Bind a channel for the duration of the next outbound MCP request.
    Returns the ContextVar token so the caller can `reset()` after."""
    return _pending_channel.set((channel or "").strip().lower() or None)


def reset_pending_channel(token) -> None:
    _pending_channel.reset(token)


class AgentMCPAuth(httpx.Auth):
    """Injects X-Agent-Key on every request and X-Toup-Channel from the
    per-call ContextVar (or `web` when unset).

    Stateless apart from the agent_api_key. Safe to share across the
    process — the channel value comes from the ContextVar, not from
    instance state, so there's no cross-call leak.
    """

    # FastMCP auth issues request → reads response → may issue more
    # requests. We don't need the response body or to retry on 401, so
    # the simplest auth_flow that yields once is enough.
    requires_response_body = False
    requires_request_body = False

    def __init__(self, agent_api_key: str):
        self._agent_api_key = agent_api_key

    def auth_flow(
        self, request: httpx.Request,
    ) -> Generator[httpx.Request, httpx.Response, None]:
        if self._agent_api_key:
            request.headers[HEADER_AGENT_KEY] = self._agent_api_key
        channel = _pending_channel.get() or "web"
        request.headers[HEADER_TOUP_CHANNEL] = channel
        yield request
