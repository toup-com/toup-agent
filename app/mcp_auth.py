"""
T0c — Platform MCP transport auth.

The platform mounts a FastMCP server at `/api/mcp` (see platform_main.py).
Tenant agents connect to it as MCP clients. Each tenant has a unique
`agent_api_key` (per Phase 3 multi-tenant Contabo, generated at provision
time from `secrets.token_urlsafe(32)`).

Before this module existed, the MCP server resolved user_id from
`settings.user_id` — a single platform-process global that's empty on
multi-tenant Railway. Tools either failed (ValueError) or, in a worst
case, returned the wrong tenant's data. This module closes that gap.

What this module does:

  1. ASGI middleware (`MCPAuthMiddleware`) wraps the FastMCP app. Every
     incoming HTTP request to /api/mcp/* has its `X-Agent-Key` header
     looked up against `agent_configs.agent_api_key` to resolve a user_id.
     The user_id is then bound to a contextvar for the duration of the
     request.

  2. `get_mcp_user_id()` reads that contextvar — used by every MCP tool
     handler to scope DB queries to the authenticated user.

Modes (settings.mcp_require_x_agent_key):

  * False (default — warn-only). On missing/invalid key, log a structured
    warning with fingerprint (key prefix, path, request_id, client) and
    let the request proceed. Tool handlers then raise ValueError from
    `get_mcp_user_id()`, which FastMCP serialises as a clean error
    response. This lets us soak the auth flow in production without
    breaking any caller that's already exercising the broken pre-auth
    path.

  * True (enforce). On missing/invalid key, return 401 from the transport
    before any tool dispatch. Flip to True after staging shows a clean
    24h with zero warn-only rejections.

What this module deliberately does NOT do:

  * Touch the legacy `settings.agent_api_key` comparison in `auth.py:62`,
    `ws_chat.py:1091`, etc. Those FastAPI endpoints have their own auth
    chain (JWT primary, agent-mode fallback). Unifying X-Agent-Key
    validation across the platform is a separate ticket — the comparisons
    against `settings.agent_api_key` are largely dead code on multi-tenant
    deploys (settings is empty), but cleaning them up risks breaking
    single-tenant deploys and is out of scope for T0c.

  * Cache `agent_api_key → user_id` lookups. Each MCP request hits the DB
    once. `agent_configs.agent_api_key` already has unique-on-user
    constraint via `agent_configs.user_id`'s unique index — cheap lookup.
    Caching adds invalidation complexity (key rotation = T0b) without a
    clear win at v1 traffic levels.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

from sqlalchemy import select

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AgentConfig

logger = logging.getLogger(__name__)


# ─── Per-request context ──────────────────────────────────────────────


_current_user_id: ContextVar[Optional[str]] = ContextVar(
    "mcp_user_id", default=None
)
_current_agent_key_prefix: ContextVar[Optional[str]] = ContextVar(
    "mcp_agent_key_prefix", default=None
)
# T1f: channel + request_id are read by connector tool handlers and
# passed into ConnectorContext / dispatcher.execute. Both have safe
# defaults so non-connector tools (memory_*, entity_*, ...) are
# unaffected by missing headers.
_current_channel: ContextVar[str] = ContextVar(
    "mcp_channel", default="web"
)
_current_request_id: ContextVar[str] = ContextVar(
    "mcp_request_id", default="no-id"
)


# Header names live as constants so the agent-side client and
# platform-side reader cannot drift apart silently.
HEADER_AGENT_KEY = "x-agent-key"
HEADER_TOUP_CHANNEL = "x-toup-channel"
HEADER_REQUEST_ID = "x-request-id"
HEADER_TRACEPARENT = "traceparent"

# Channels the dispatcher knows about. MUST stay a superset of every channel
# name the connector_dispatcher deny-sets reference (_MUTATES_DEFAULT_DENY_CHANNELS
# + _MUTATES_UNATTENDED_DENY_CHANNELS) — otherwise the unattended/autopilot deny is
# silently bypassed because the transport rewrites e.g. "routine" → "web" BEFORE
# the dispatcher's policy check ever sees it (docs/security/audit-2026.md — the
# re-audit found the two sets had drifted, nulling the INJ-1 unattended deny).
# Interactive channels first, then the unattended/automation channels.
_KNOWN_CHANNELS = frozenset({
    # interactive / attended
    "web", "mobile", "voice", "telegram", "whatsapp",
    "app", "api", "agent", "netflix", "chat_intent",
    # automation / unattended (also in the dispatcher deny sets)
    "autopilot", "routine", "trigger", "heartbeat", "cron",
    "agent_task", "email_briefing", "background",
    # internal orchestration turns — pass through so the dispatcher applies its
    # normal policy instead of being fail-closed-clamped (re-audit round 6 found
    # the round-5 clamp over-denied these, breaking their connector calls).
    "subagent", "app_builder",
})
# Fail-CLOSED clamp target for a genuinely-unknown channel: a member of the
# unattended deny set, so an unrecognized value denies mutating connectors
# rather than falling through to permissive "web". (Legit channels are all in
# _KNOWN_CHANNELS above and pass through verbatim.)
_UNKNOWN_CHANNEL_CLAMP = "background"


def get_mcp_user_id() -> str:
    """Return the authenticated MCP request's user_id.

    Raises ValueError when no user is bound — happens during warn-only
    soak when the request had a missing/invalid X-Agent-Key. FastMCP
    surfaces ValueError as a structured tool-call error to the client,
    which is the right failure mode (the caller learns their request
    wasn't authenticated).
    """
    uid = _current_user_id.get()
    if uid is None:
        raise ValueError(
            "MCP request has no authenticated user — X-Agent-Key was "
            "missing or did not match any agent_config row. If the "
            "platform is in warn-only mode, this is the expected error "
            "for unauthenticated callers; flip MCP_REQUIRE_X_AGENT_KEY "
            "to true to fail at the transport layer instead."
        )
    return uid


def try_get_mcp_user_id() -> Optional[str]:
    """Non-raising variant. Used by `tools/list` filtering so an
    unauthenticated `list` returns the global tool set (memory_* etc)
    rather than 500-ing the LLM client. Connector tools are filtered
    out when this returns None — caller scope is intentional."""
    return _current_user_id.get()


def get_mcp_agent_key_prefix() -> Optional[str]:
    """First 8 chars of the X-Agent-Key for the current request, or None.

    Used for log fingerprinting — never the full key.
    """
    return _current_agent_key_prefix.get()


def get_mcp_channel() -> str:
    """Channel that originated this MCP request (`web` default).

    Channel governs the dispatcher's mutating-tool default-deny on
    `voice` / `telegram` (architecture §4.4). Default `web` matches the
    browser-originated case where there's an inline confirmation
    surface.
    """
    return _current_channel.get()


def get_mcp_request_id() -> str:
    """Best-effort request id for trace correlation. Falls back to the
    sentinel `"no-id"` when neither X-Request-Id nor traceparent was
    sent — visible in dispatcher logs so missing wiring is greppable."""
    return _current_request_id.get()


# ─── Lookup helper ────────────────────────────────────────────────────


async def _resolve_agent_key_to_user_id(agent_key: str) -> Optional[str]:
    """Look up `agent_configs.agent_api_key == <agent_key>` → user_id.

    Returns None on miss (key didn't match any tenant). DB error bubbles
    up — middleware caller decides whether to 5xx or fail-soft.
    """
    if not agent_key:
        return None
    async with async_session_maker() as db:
        result = await db.execute(
            select(AgentConfig.user_id).where(
                AgentConfig.agent_api_key == agent_key
            )
        )
        return result.scalar_one_or_none()


# ─── ASGI middleware ──────────────────────────────────────────────────


class MCPAuthMiddleware:
    """Wraps the FastMCP ASGI app to enforce X-Agent-Key auth.

    Pure ASGI — no FastAPI/Starlette dependency injection — because
    FastMCP mounts as a sub-app and FastAPI's dependency chain doesn't
    apply to its routes.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            # Lifespan, websocket, etc — pass through untouched.
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        agent_key = headers.get(HEADER_AGENT_KEY, "")
        path = scope.get("path", "?")
        # Best-effort request id for log correlation.
        request_id = (
            headers.get(HEADER_REQUEST_ID)
            or headers.get(HEADER_TRACEPARENT)
            or "no-id"
        )
        # T1f: Channel header. Default "web" when missing — matches the
        # majority case (browser-originated tool calls). A genuinely-unknown
        # value FAILS CLOSED to a deny-set channel (not "web") so a typo/garbage
        # value can't bypass the mutating-connector default-deny.
        raw_channel = (headers.get(HEADER_TOUP_CHANNEL) or "").strip().lower()
        if raw_channel and raw_channel not in _KNOWN_CHANNELS:
            logger.warning(
                "MCP unknown channel %r → clamped to %r (fail-closed) "
                "(path=%s, request_id=%s)",
                raw_channel, _UNKNOWN_CHANNEL_CLAMP, path, request_id,
            )
            channel = _UNKNOWN_CHANNEL_CLAMP
        elif raw_channel:
            channel = raw_channel
        else:
            channel = "web"
        client = scope.get("client") or ("?", 0)

        user_id: Optional[str] = None
        if agent_key:
            try:
                user_id = await _resolve_agent_key_to_user_id(agent_key)
            except Exception:
                logger.exception(
                    "MCP auth lookup raised — failing closed for this "
                    "request regardless of mode"
                )
                await _send_503(send)
                return

        if not user_id:
            fingerprint = {
                "key_prefix": (agent_key[:8] + "...") if agent_key else "<missing>",
                "path": path,
                "request_id": request_id,
                "client_ip": client[0],
            }
            if settings.mcp_require_x_agent_key:
                logger.warning("MCP auth REJECT (enforce mode): %s", fingerprint)
                await _send_401(send)
                return
            else:
                # Warn-only soak. Log enough fingerprint to identify the
                # call site that needs fixing before we flip to enforce.
                logger.warning(
                    "MCP auth WARN (would-reject in enforce mode): %s",
                    fingerprint,
                )
                # Fall through with no user_id bound. Tool handlers will
                # raise from get_mcp_user_id() — that's the correct
                # failure mode (cleaner than the pre-T0c silent miss).

        # Bind contextvars for the duration of this request.
        token_user = _current_user_id.set(user_id)
        token_prefix = _current_agent_key_prefix.set(
            agent_key[:8] if agent_key else None
        )
        token_channel = _current_channel.set(channel)
        token_request_id = _current_request_id.set(request_id)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_user_id.reset(token_user)
            _current_agent_key_prefix.reset(token_prefix)
            _current_channel.reset(token_channel)
            _current_request_id.reset(token_request_id)


# ─── 401/503 responses ────────────────────────────────────────────────


async def _send_401(send) -> None:
    body = b'{"error":"Missing or invalid X-Agent-Key"}'
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("latin-1")),
            (b"www-authenticate", b'X-Agent-Key realm="toup-platform-mcp"'),
        ],
    })
    await send({"type": "http.response.body", "body": body})


async def _send_503(send) -> None:
    body = b'{"error":"MCP auth lookup failed"}'
    await send({
        "type": "http.response.start",
        "status": 503,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("latin-1")),
        ],
    })
    await send({"type": "http.response.body", "body": body})
