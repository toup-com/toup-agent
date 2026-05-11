"""
T1f — Connector MCP namespace registration.

Bridges the platform-side dispatcher (T1e) to the FastMCP server
mounted at `/api/mcp`. Every active connector tool becomes a callable
MCP tool; the agent's MCP client sees them in `tools/list` and can
invoke them like any other platform tool.

What this module owns:

  - One FastMCP `FunctionTool` per `(connector, manifest_tool)` pair,
    registered at platform boot after `connector_registry.load_all()`.
    Tool name is `<connector_id>__<tool_name>` (already manifest-
    validated globally unique by T1c).
  - `ConnectorToolFilterMiddleware` — FastMCP middleware that filters
    `tools/list` per requesting user. User A sees `stub__echo` only
    when A has an active stub identity in `connector_identities`.
    Cross-tenant isolation lives here.
  - Serialisation of `ConnectorResult` sum-type to FastMCP tool-call
    output. Each variant becomes a structured string the LLM can
    parse on its own (no raw enum names, no Python type leaks).

What this module deliberately does NOT do:

  - Per-call user resolution — that's T0c's `MCPAuthMiddleware`
    binding ContextVars (`get_mcp_user_id`, `get_mcp_channel`,
    `get_mcp_request_id`).
  - Caching of per-user tool lists — every `tools/list` reads
    `vault.list_visible_to_agent(user_id)`. Single indexed query;
    cheap. The agent-side cache (T1h) is the right place for the
    hot path.
  - The dispatcher's audit/refresh/policy logic — all owned by T1e.
    This module is a thin transport adapter.

Production hardening (T5a) lands metrics; until then INFO logs are
the operational signal.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from app.connectors.base import (
    ConnectorOk,
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorScopeMissing,
    ConnectorToolError,
)
from app.db.database import async_session_maker
from app.mcp_auth import (
    get_mcp_channel,
    get_mcp_request_id,
    get_mcp_user_id,
    try_get_mcp_user_id,
)
from app.services import connector_dispatcher as dispatcher
from app.services import connector_vault as vault
from app.services.connector_registry import ConnectorRegistry

logger = logging.getLogger(__name__)


# ─── Result serialisation ────────────────────────────────────────────


def _serialize_result(result: ConnectorResult) -> dict:
    """Convert a `ConnectorResult` to the dict shape FastMCP returns to
    the LLM client.

    Why dict-with-string-fields rather than just `result.content`: the
    LLM client serialises the dict to JSON before handing it to the
    model. Rate-limit/reauth/down/scope-missing variants need a
    machine-greppable kind plus a human-readable line. Keeping the
    error envelope flat (`kind`, `message`, plus variant-specific
    fields) means the model can both react ("retry in 30s") and surface
    a clean note to the user.

    Architecture §2.5 lists these six variants. If any provider returns
    a subclass not handled here, we surface `tool_error` with a clear
    message — better than letting the dispatch path 500.
    """
    if isinstance(result, ConnectorOk):
        # Unredacted content goes to the LLM. Audit-row redaction
        # already happened inside the dispatcher.
        return {"kind": "ok", "content": result.content}

    if isinstance(result, ConnectorRateLimited):
        return {
            "kind": "rate_limited",
            "retry_after_s": result.retry_after_s,
            "message": (
                f"[rate_limited] Provider asked us to wait "
                f"{result.retry_after_s}s before retrying."
            ),
        }

    if isinstance(result, ConnectorReauthRequired):
        return {
            "kind": "reauth_required",
            "reauth_url": result.reauth_url,
            "message": (
                f"[reauth_required] Reconnect at {result.reauth_url} "
                f"and try again."
            ),
        }

    if isinstance(result, ConnectorProviderDown):
        return {
            "kind": "provider_down",
            "provider_status_url": result.provider_status_url,
            "message": (
                "[provider_down] The connector's provider is not "
                "responding. Try again shortly."
            ),
        }

    if isinstance(result, ConnectorScopeMissing):
        return {
            "kind": "scope_missing",
            "required_scope": result.required_scope,
            "message": (
                f"[scope_missing] This tool needs the "
                f"'{result.required_scope}' scope. User must reconnect "
                f"and grant it."
            ),
        }

    if isinstance(result, ConnectorToolError):
        return {
            "kind": "tool_error",
            "retryable": result.retryable,
            "message": f"[tool_error] {result.message}",
        }

    # Unknown subclass — should never happen, but stay well-shaped.
    logger.error(
        "[connector_mcp] unknown ConnectorResult subclass: %s",
        type(result).__name__,
    )
    return {
        "kind": "tool_error",
        "retryable": False,
        "message": (
            f"[tool_error] internal error — unknown result type "
            f"{type(result).__name__!r}"
        ),
    }


# ─── Tool handler factory ────────────────────────────────────────────


def _make_handler(connector_id: str, tool_name: str):
    """Build the per-(connector, tool) async handler that FastMCP
    invokes when the client calls this tool.

    The handler:
      1. Resolves user_id from the auth contextvar (raises if missing).
      2. Reads channel + request_id from contextvars (defaults are
         safe).
      3. Opens a fresh async session — the dispatcher commits twice
         (audit-then-act) so it owns its own transaction boundaries.
      4. Calls `dispatcher.execute(...)`.
      5. Serialises the result variant for the MCP client.

    Tool input arrives as kwargs matching the manifest's input_schema.
    We collect them via `**tool_input` so any schema shape is handled
    without per-tool wiring.
    """

    async def handler(**tool_input) -> dict:
        # 1. Auth — raises ValueError if no X-Agent-Key resolved.
        user_id = get_mcp_user_id()
        # 2. Per-request envelope.
        channel = get_mcp_channel()
        request_id = get_mcp_request_id()
        # 3. Fresh session per tool call (dispatcher owns its own
        #    commit boundaries via _audit_then_commit).
        async with async_session_maker() as db:
            result = await dispatcher.execute(
                db=db,
                user_id=user_id,
                connector_id=connector_id,
                tool_name=tool_name,
                tool_input=dict(tool_input),
                channel=channel,
                agent_request_id=request_id,
            )
        return _serialize_result(result)

    handler.__name__ = f"{connector_id}__{tool_name}"
    handler.__qualname__ = handler.__name__
    return handler


# ─── Registration entry point ────────────────────────────────────────


CONNECTOR_TOOL_TAG = "connector"


def register_connector_tools(
    mcp_server,
    registry: ConnectorRegistry,
    *,
    skill_prefixes: Optional[set[str]] = None,
) -> int:
    """Register one FastMCP tool per (connector, manifest_tool).

    Called at platform boot after `registry.load_all()`. Returns the
    number of tools registered (sum of manifest tool counts across
    all loaded connectors).

    Idempotent against double-call: if a tool with the same name is
    already registered we log a WARN and skip — better than letting
    FastMCP raise during boot. The registry's load-time lint already
    guarantees uniqueness across manifests; this is a second-line
    audit for the (rare) reload case.

    Connector tools are tagged `connector` so the filter middleware
    can identify them in `on_list_tools` without a round-trip to the
    registry.
    """
    # Lazy import to avoid pulling FastMCP into modules that don't
    # need it (tests of pure dispatcher logic, etc).
    from fastmcp.tools import FunctionTool

    skill_prefixes = skill_prefixes or set()
    registered = 0
    for entry in registry.list_all():
        for manifest_tool in entry.manifest.tools:
            tool_full_name = manifest_tool.name
            # Belt-and-braces collision warn — registry already lints
            # this at load. If a tool slips through (e.g. a future
            # registry change) we want a loud signal, not a silent
            # shadow of an existing skill.
            prefix = tool_full_name.split("__", 1)[0]
            if prefix in skill_prefixes:
                logger.warning(
                    "[connector_mcp] tool %r prefix collides with "
                    "skill namespace %r — registering anyway, but "
                    "tool_executor (T1g) will need to disambiguate",
                    tool_full_name, prefix,
                )
            handler = _make_handler(entry.manifest.id, manifest_tool.name)
            tool = FunctionTool(
                name=tool_full_name,
                description=manifest_tool.description,
                parameters=manifest_tool.input_schema,
                tags={CONNECTOR_TOOL_TAG, f"connector:{entry.manifest.id}"},
                fn=handler,
                enabled=True,
            )
            try:
                mcp_server.add_tool(tool)
            except Exception as e:
                # FastMCP's add_tool may raise on duplicate. The
                # registry's load-time lint already guarantees
                # uniqueness across manifests; this branch is for the
                # rare double-register case (test reload, hot-reload).
                logger.warning(
                    "[connector_mcp] add_tool(%r) raised %s — "
                    "tool may already be registered; continuing",
                    tool_full_name, e,
                )
                continue
            registered += 1
            logger.info(
                "[connector_mcp] registered tool %r (connector=%s)",
                tool_full_name, entry.manifest.id,
            )

    logger.info(
        "[connector_mcp] %d connector tool(s) registered across %d connector(s)",
        registered, len(registry.list_all()),
    )
    return registered


# ─── tools/list filter middleware ────────────────────────────────────


try:
    # fastmcp 2.11+ requires middleware classes to inherit from this
    # base — its `__call__` orchestrates per-method dispatch
    # (`on_list_tools`, `on_call_tool`, …). Without it,
    # `FastMCP._build_chain` does `partial(middleware, call_next=...)`
    # which raises `TypeError: the first argument must be callable`
    # because plain instances aren't callable. The error propagates
    # back to the agent client as `McpError("the first argument must
    # be callable")` and every tools/list dies — taking the entire
    # connector tool surface (Gmail etc.) offline.
    from fastmcp.server.middleware import Middleware as _FastMCPMiddleware
except ImportError:
    # Pre-2.11 fastmcp had no formal middleware base. Defensive
    # fallback only; the production pin is 2.11+.
    _FastMCPMiddleware = object  # type: ignore[assignment,misc]


class ConnectorToolFilterMiddleware(_FastMCPMiddleware):
    """FastMCP middleware that filters connector tools per requesting
    user.

    The cross-tenant isolation point for `tools/list`: user A's
    response includes connector tools only for connectors A has
    actively connected. User B with no stub identity does not see
    `stub__echo` in their tool list — even though it's registered
    globally on the FastMCP server.

    Tool execution does NOT need a parallel filter here: the
    dispatcher already returns `ConnectorReauthRequired` when
    `vault.get` finds no active identity. Adding a redundant pre-
    check in `on_call_tool` would just duplicate the no-identity
    branch. The dispatcher's audit row also captures the call attempt,
    which `on_call_tool`'s pre-check would not.

    Warn-only mode (no resolved user_id): pass through the FULL tool
    list including connector tools. The connector tool's handler will
    raise `ValueError` from `get_mcp_user_id()` when invoked, which
    FastMCP surfaces as a structured error. This matches the existing
    memory_*/entity_* warn-only behavior.
    """

    def __init__(self, registry: ConnectorRegistry):
        self._registry = registry

    async def on_list_tools(self, context, call_next):
        # Run the chain first — let FastMCP build the unfiltered list,
        # then trim it based on this request's user.
        # fastmcp 2.11+ contract: call_next returns `list[Tool]`
        # directly. Pre-2.11 wrapped it in a `ListToolsResult` with a
        # `.tools` attribute — accessing `.tools` on the bare list now
        # raises AttributeError (surfaced to the client as McpError
        # "'list' object has no attribute 'tools'").
        tools = await call_next(context)

        user_id = try_get_mcp_user_id()
        if user_id is None:
            # Warn-only soak: leave the list unfiltered. Caller will
            # either invoke and hit the dispatcher's auth error, or
            # not invoke. Either way we're not silently hiding the
            # connector surface.
            return tools

        # Per-request DB read. Cheap (single indexed query); see
        # T1h for the agent-side cache that elides repeat traffic.
        #
        # `list_visible_to_agent` returns identities in `active`,
        # `reauth_required`, AND `provider_down` statuses — wider than
        # the pre-2026-05-12 `list_active` call. The motivation: an
        # identity with an EXPIRED token but no terminal disconnect is
        # still a connector the user knows about. Hiding the tool sent
        # the agent to the `browser` fallback (which fails because
        # Chromium isn't installed in tenant containers); KEEPING the
        # tool lets the dispatcher return ConnectorReauthRequired,
        # which the LLM surfaces as "Please reconnect your Gmail".
        # See `vault.list_visible_to_agent` docstring for the contract.
        async with async_session_maker() as db:
            visible_rows = await vault.list_visible_to_agent(db, user_id)
        # connector_id → (status, read_only). The status drives the
        # mutating-tool gate below: read-only only applies to identities
        # that are actually active (a reauth_required identity isn't
        # "read-only", it's broken — drop *no* tools for it so the
        # dispatcher's reauth error reaches the LLM).
        visible_by_connector = {
            row.connector_id: (
                row.status,
                bool(getattr(row, "read_only", False)),
            )
            for row in visible_rows
        }

        # Filter: keep all non-connector tools (memory_*, entity_*,
        # etc) + connector tools whose owner has a visible identity.
        # Drop mutating connector tools when the user's ACTIVE identity
        # for that connector is read-only. reauth_required / provider_down
        # identities pass through unfiltered — the dispatcher returns
        # the structured error, which the LLM surfaces to the user.
        filtered = []
        for tool in tools:
            owner = self._registry.get_owner(tool.name)
            if owner is None:
                # Not a connector tool — keep.
                filtered.append(tool)
                continue
            if owner not in visible_by_connector:
                # User hasn't connected this connector (or it's
                # `revoked`) — drop. Agent's system prompt covers
                # "tell the user to connect if they ask for an
                # un-connected service".
                continue
            status, read_only = visible_by_connector[owner]
            if status == "active" and read_only:
                # Active read-only identity: drop mutating tools so
                # the LLM doesn't even see them. Read tools pass.
                spec = self._registry.get_tool_spec(tool.name)
                if spec is not None and spec.mutates:
                    continue
            filtered.append(tool)

        return filtered


# ─── Module-level test reset ─────────────────────────────────────────


def deregister_connector_tools_for_tests(mcp_server, registry: ConnectorRegistry) -> None:
    """Remove all connector tools that were registered via
    `register_connector_tools`. Tests only — production never needs
    to deregister at runtime."""
    for entry in registry.list_all():
        for manifest_tool in entry.manifest.tools:
            try:
                mcp_server.remove_tool(manifest_tool.name)
            except Exception:
                # Tool wasn't registered — silently ignore.
                pass
