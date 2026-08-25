"""Platform-side automations RPC — the agent executor's only door to
connectors (Round 26).

Four routes, all authenticated the way trigger provisioning is
(X-Agent-Key must match the tenant's AgentConfig row; the user id rides
X-Agent-User-Id, same header pair as credit_reporter):

  GET  /v1/automations/registry      — capability metadata snapshot
  GET  /v1/automations/connections   — per-user connector status/scopes
  GET  /v1/automations/grant-status  — one grant, authoritative
  POST /v1/automations/dispatch      — run one connector tool call with
                                       channel="automation" (+grant_id)

Why an RPC and not MCP: the automations outbox flushes writes outside
any agent turn, and the grant ref must reach the dispatcher — the one
process holding tokens — without widening the MCP surface. The
dispatcher's step-1.7 gate does the enforcement; this module only
authenticates the tenant and shapes the payload.

Everything 404s when the `automations` flag is off for the user — the
same "the surface does not exist" answer the routines kinds give.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.db.database import async_session_maker
from app.db.models import AgentConfig, AutomationGrant
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/automations", tags=["automations (platform RPC)"])


async def _auth_agent(x_agent_key: Optional[str], user_id: Optional[str]) -> str:
    """Validate the calling tenant; returns the user_id. 401 on any
    mismatch — same contract as triggers_provision._auth_agent."""
    if not x_agent_key or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Agent-Key and X-Agent-User-Id required",
        )
    async with async_session_maker() as db:
        row = (await db.execute(
            select(AgentConfig).where(
                AgentConfig.user_id == user_id,
                AgentConfig.agent_api_key == x_agent_key,
            )
        )).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="agent key / user id mismatch",
        )
    return user_id


async def _flag_or_404(user_id: str) -> None:
    from app.services import feature_flags
    async with async_session_maker() as db:
        enabled = await feature_flags.is_enabled(db, "automations", user_id)
    if not enabled:
        raise HTTPException(status_code=404, detail="Feature not available")


@router.get("/registry")
async def automation_registry(
    x_agent_key: Optional[str] = Header(default=None),
    x_agent_user_id: Optional[str] = Header(default=None),
) -> dict:
    user_id = await _auth_agent(x_agent_key, x_agent_user_id)
    await _flag_or_404(user_id)
    from app.services.connector_registry import get_registry
    return {"connectors": get_registry().automation_registry()}


@router.get("/templates")
async def automation_templates(
    x_agent_key: Optional[str] = Header(default=None),
    x_agent_user_id: Optional[str] = Header(default=None),
) -> dict:
    """The template catalog for the tenant setup agent (Round 28) —
    same serializer as the user-facing route, agent-key auth."""
    user_id = await _auth_agent(x_agent_key, x_agent_user_id)
    await _flag_or_404(user_id)
    from sqlalchemy import select
    from app.db.models.platform_automation import AutomationTemplate
    from app.services.automation_template_catalog import template_payload
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationTemplate)
            .where(AutomationTemplate.enabled.is_(True))
            .order_by(AutomationTemplate.sort_order, AutomationTemplate.name)
        )).scalars().all()
        return {"templates": [template_payload(t) for t in rows]}


@router.get("/connections")
async def automation_connections(
    x_agent_key: Optional[str] = Header(default=None),
    x_agent_user_id: Optional[str] = Header(default=None),
) -> dict:
    user_id = await _auth_agent(x_agent_key, x_agent_user_id)
    await _flag_or_404(user_id)
    from app.services import connector_vault as vault
    out = []
    async with async_session_maker() as db:
        rows = await vault.list_active(db, user_id)
    for ident in rows:
        try:
            scopes = json.loads(ident.scopes_json) if ident.scopes_json else []
        except (ValueError, TypeError):
            scopes = []
        out.append({
            "connector_id": ident.connector_id,
            "status": ident.status,
            "connected": ident.status == "active",
            "scopes": scopes,
            # R28 connector disclosure: the bound account's identity —
            # the Gmail address for Google (backfilled at watch-arm),
            # login for GitHub, None where the provider never told us
            # (Outlook). The setup skill names it; clients may render it.
            "account": ident.provider_account_id or None,
        })
    return {"connections": out}


@router.get("/grant-status")
async def automation_grant_status(
    grant_id: str = Query(..., min_length=1, max_length=36),
    x_agent_key: Optional[str] = Header(default=None),
    x_agent_user_id: Optional[str] = Header(default=None),
) -> dict:
    user_id = await _auth_agent(x_agent_key, x_agent_user_id)
    await _flag_or_404(user_id)
    async with async_session_maker() as db:
        row = (await db.execute(
            select(AutomationGrant)
            .where(AutomationGrant.id == grant_id)
            .where(AutomationGrant.user_id == user_id)
        )).scalar_one_or_none()
    if row is None:
        return {"grant": None}
    return {"grant": _grant_payload(row)}


def _grant_payload(row: AutomationGrant) -> dict:
    try:
        target = json.loads(row.target_json or "{}")
    except (ValueError, TypeError):
        target = {}
    try:
        cadence = json.loads(row.cadence_json) if row.cadence_json else {}
    except (ValueError, TypeError):
        cadence = {}
    return {
        "id": row.id,
        "automation_id": row.automation_id,
        "connector_id": row.connector_id,
        "tool_name": row.tool_name,
        "target": target,
        "cadence": cadence,
        "mode": row.mode,
        "status": row.status,
        "summary": row.summary,
        "created_at": row.created_at.isoformat() + "Z",
        "expires_at": row.expires_at.isoformat() + "Z",
        "decided_at": (row.decided_at.isoformat() + "Z") if row.decided_at else None,
    }


class GrantRequestReq(BaseModel):
    connector_id: str = Field(..., min_length=1, max_length=64)
    tool_name: str = Field(..., min_length=1, max_length=128)
    target: dict = Field(...)          # {kind, id, label}
    cadence: Optional[dict] = None     # {per_day?, per_hour?}
    mode: str = Field(default="confirm", pattern="^(auto|confirm)$")
    summary: str = Field(..., min_length=1, max_length=300)
    preview: Optional[dict] = None
    automation_id: Optional[str] = Field(default=None, max_length=36)


@router.post("/grant-requests")
async def create_grant_request(
    req: GrantRequestReq,
    x_agent_key: Optional[str] = Header(default=None),
    x_agent_user_id: Optional[str] = Header(default=None),
) -> dict:
    """Agent stages a grant REQUEST (status=pending, 1-hour card TTL).
    Only the user can approve it — via the platform-native
    /automations/grant-requests/{id}/approve endpoint, never this RPC.

    The (connector, tool, target) triple is validated against the
    capability metadata HERE, at request time, so a card the user is
    about to be shown can never describe an ungrantable action."""
    user_id = await _auth_agent(x_agent_key, x_agent_user_id)
    await _flag_or_404(user_id)

    from datetime import datetime, timedelta
    from app.db.models import AUTOMATION_GRANT_REQUEST_TTL_S
    from app.services.connector_registry import get_registry

    cap = get_registry().get_automation_capability(req.connector_id)
    if cap is None or req.tool_name not in (cap.scopes_write_by_action or {}):
        raise HTTPException(
            status_code=422,
            detail=f"{req.tool_name!r} is not a grantable write action "
                   f"for {req.connector_id!r}",
        )
    target_id = str((req.target or {}).get("id") or "").strip()
    if not target_id:
        raise HTTPException(
            status_code=422,
            detail="target.id is required — a grant pins ONE target",
        )

    # R29 §5 — the scope truth, checked while the user is still in the
    # setup conversation: a connected identity that never consented to
    # the write's scope (a pre-R29 Outlook connection vs Mail.ReadWrite)
    # gets the stable reconnect-shaped refusal HERE, not a doomed grant
    # card followed by dispatch-time 403s.
    needed_scopes = list(
        (cap.scopes_write_by_action or {}).get(req.tool_name) or []
    )
    if needed_scopes:
        from app.services import connector_vault as vault
        held: Optional[set] = None
        async with async_session_maker() as db:
            for ident in await vault.list_active(db, user_id):
                if ident.connector_id == req.connector_id:
                    try:
                        held = set(json.loads(ident.scopes_json or "[]"))
                    except (ValueError, TypeError):
                        held = set()
                    break
        # A connection with NO recorded scopes predates scope tracking —
        # let dispatch fail honestly rather than refusing everyone.
        if held:
            missing = [s for s in needed_scopes if s not in held]
            if missing:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "scope_missing",
                        "connector_id": req.connector_id,
                        "needed_scope": missing[0],
                        "reconnect": True,
                        "message": (
                            f"The connected {req.connector_id} account "
                            f"hasn't granted the permission this action "
                            f"needs — reconnect it to continue."
                        ),
                    },
                )

    row = AutomationGrant(
        user_id=user_id,
        automation_id=req.automation_id,
        connector_id=req.connector_id,
        tool_name=req.tool_name,
        target_json=json.dumps(req.target, default=str),
        cadence_json=json.dumps(req.cadence, default=str) if req.cadence else None,
        mode=req.mode,
        summary=req.summary,
        preview_json=json.dumps(req.preview, default=str) if req.preview else None,
        status="pending",
        expires_at=datetime.utcnow()
        + timedelta(seconds=AUTOMATION_GRANT_REQUEST_TTL_S),
    )
    async with async_session_maker() as db:
        db.add(row)
        await db.commit()
        return {"grant": _grant_payload(row)}


class GrantBindReq(BaseModel):
    grant_id: str = Field(..., min_length=1, max_length=36)
    automation_id: str = Field(..., min_length=1, max_length=36)


@router.post("/grant-bind")
async def bind_grant_to_automation(
    req: GrantBindReq,
    x_agent_key: Optional[str] = Header(default=None),
    x_agent_user_id: Optional[str] = Header(default=None),
) -> dict:
    """ND-1 (GROUND-TRUTH-R30): stamp `automation_id` onto a grant that
    was staged before its automation existed.

    The skill's setup order is permission card FIRST (step 5), create
    SECOND (step 6) — so /grant-requests, which accepts and stamps
    `automation_id`, receives None every time, `GET /{id}/grants`
    serves `[]`, and revoke→pause can never fire. The agent calls this
    the moment the automation exists (arm verifies each grant and binds
    the orphans).

    Bind-once: NULL → id, idempotent on the same id, 409 on an attempt
    to move a grant to a DIFFERENT automation — a grant the user
    approved for one rule must not silently start covering another."""
    user_id = await _auth_agent(x_agent_key, x_agent_user_id)
    await _flag_or_404(user_id)
    async with async_session_maker() as db:
        row = (await db.execute(
            select(AutomationGrant)
            .where(AutomationGrant.id == req.grant_id)
            .where(AutomationGrant.user_id == user_id)
        )).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="No such grant")
        if row.automation_id and row.automation_id != req.automation_id:
            raise HTTPException(
                status_code=409,
                detail="grant is already bound to a different automation",
            )
        if not row.automation_id:
            row.automation_id = req.automation_id
            await db.commit()
            logger.info("[automations] grant %s bound to automation %s",
                        row.id, req.automation_id)
        return {"grant": _grant_payload(row)}


class DispatchReq(BaseModel):
    connector_id: str = Field(..., min_length=1, max_length=64)
    tool_name: str = Field(..., min_length=1, max_length=128)
    tool_input: dict = Field(default_factory=dict)
    grant_id: Optional[str] = Field(default=None, max_length=36)
    automation_id: Optional[str] = Field(default=None, max_length=36)
    request_id: Optional[str] = Field(default=None, max_length=64)
    # 'e2e' excludes the call from metering — honored ONLY outside
    # production (a live tenant cannot dodge billing with it).
    mode: Optional[str] = Field(default=None, pattern="^(e2e)$")


@router.post("/dispatch")
async def automation_dispatch(
    req: DispatchReq,
    x_agent_key: Optional[str] = Header(default=None),
    x_agent_user_id: Optional[str] = Header(default=None),
) -> dict:
    """One connector call on the automation channel. The dispatcher's
    grant gate does the real enforcement; the one check that lives HERE
    is grant↔automation binding (the dispatcher doesn't know which
    automation is calling, this endpoint does)."""
    user_id = await _auth_agent(x_agent_key, x_agent_user_id)
    await _flag_or_404(user_id)

    if req.grant_id:
        async with async_session_maker() as db:
            grant = (await db.execute(
                select(AutomationGrant)
                .where(AutomationGrant.id == req.grant_id)
                .where(AutomationGrant.user_id == user_id)
            )).scalar_one_or_none()
        if grant is None:
            return {"kind": "tool_error",
                    "message": "grant not found", "retryable": False}
        if (
            grant.automation_id
            and req.automation_id
            and grant.automation_id != req.automation_id
        ):
            return {"kind": "tool_error",
                    "message": "grant belongs to a different automation",
                    "retryable": False}

    from app.config import settings
    from app.services import connector_dispatcher as dispatcher

    exclude_metering = (
        req.mode == "e2e"
        and (settings.environment or "").lower() != "production"
    )
    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db=db,
            user_id=user_id,
            connector_id=req.connector_id,
            tool_name=req.tool_name,
            tool_input=req.tool_input or {},
            channel="automation",
            agent_request_id=req.request_id,
            grant_id=req.grant_id,
            automation_id=req.automation_id,
            exclude_metering=exclude_metering,
        )
    return _serialize_connector_result(result)


def _serialize_connector_result(result: Any) -> dict:
    """The ConnectorResult sum type → the RPC wire envelope.

    Deliberately NOT `connector_mcp._serialize_result`: importing
    connector_mcp pulls fastmcp at module scope, and this RPC must keep
    working when the MCP layer cannot load (the boot already tolerates
    exactly that — "Connector MCP registration skipped"). Kinds match
    the MCP envelope so the agent-side client has one shape to parse.
    """
    from app.connectors.base import (
        ConnectorConfirmationRequired, ConnectorOk, ConnectorProviderDown,
        ConnectorRateLimited, ConnectorReauthRequired, ConnectorScopeMissing,
        ConnectorToolError,
    )
    if isinstance(result, ConnectorOk):
        return {"kind": "ok", "content": result.content}
    if isinstance(result, ConnectorRateLimited):
        return {"kind": "rate_limited",
                "retry_after_s": result.retry_after_s,
                "retryable": True,
                "message": f"rate limited; retry in {result.retry_after_s}s"}
    if isinstance(result, ConnectorReauthRequired):
        return {"kind": "reauth_required", "reauth_url": result.reauth_url,
                "retryable": False,
                "message": f"reconnect at {result.reauth_url}"}
    if isinstance(result, ConnectorProviderDown):
        return {"kind": "provider_down",
                "provider_status_url": result.provider_status_url,
                "retryable": True,
                "message": "provider not responding"}
    if isinstance(result, ConnectorScopeMissing):
        return {"kind": "scope_missing",
                "required_scope": result.required_scope,
                "retryable": False,
                "message": f"missing scope {result.required_scope}"}
    if isinstance(result, ConnectorConfirmationRequired):
        return {"kind": "confirmation_required",
                "action_id": result.action_id,
                "summary": result.summary,
                # R29-C's park card + expiry sweep read these from the
                # envelope (the MCP serializer already sends both) —
                # without them the card is payload-less and expiry
                # falls back to the generic 25h.
                "payload": result.payload or {},
                "expires_at": (result.expires_at.isoformat() + "Z")
                if result.expires_at else None,
                "retryable": False,
                "message": "staged for user confirmation — not executed"}
    if isinstance(result, ConnectorToolError):
        return {"kind": "tool_error", "message": result.message,
                "retryable": bool(result.retryable)}
    return {"kind": "tool_error", "retryable": False,
            "message": f"unhandled result type {type(result).__name__}"}
