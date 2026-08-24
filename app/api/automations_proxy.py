"""Platform-side `/api/automations/*` — native routes + agent proxy.

Two kinds of route share the prefix, declared in this order so the
literal paths win before the `{automation_id}` parameter routes:

  PLATFORM-NATIVE (the data lives in the platform DB):
    GET  /automations/templates                      — Suggested
    GET  /automations/grant-requests/{id}            — card reload
    POST /automations/grant-requests/{id}/approve    — guarded UPDATE
    POST /automations/grant-requests/{id}/reject     — guarded UPDATE

  PROXIED to the tenant agent (routines_proxy pattern — resolve
  agent_url + X-Agent-Key, forward verbatim):
    GET/POST /automations, PATCH/DELETE /automations/{id},
    POST /automations/{id}/(arm|pause|resume|test-run),
    GET /automations/runs, GET/POST /automations/auth-sessions/…,
    POST /automations/outbox/{id}/undo

Everything 404s while the `automations` rollout flag is off for the
user — the surface does not exist during the dark launch.

Grant decisions mirror connector_pending_actions' invariants: the
claim is ONE guarded UPDATE (`WHERE status='pending'`), the action
described by the card comes from the DB row and never the request
body, and after the commit the tenant agent is told best-effort so the
chat card updates in place (`/api/automations/_grant_decided`).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db.database import get_db
from app.db.models import AutomationGrant, AutomationTemplate, User

router = APIRouter(prefix="/automations", tags=["automations (platform)"])
logger = logging.getLogger(__name__)

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length",
}


async def _flag_or_404(db: AsyncSession, user_id: str) -> None:
    from app.services import feature_flags
    if not await feature_flags.is_enabled(db, "automations", user_id):
        raise HTTPException(status_code=404, detail="Feature not available")


# ── Platform-native: templates ───────────────────────────────────────


@router.get("/templates")
async def list_templates(
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _flag_or_404(db, str(current_user.id))
    q = (
        select(AutomationTemplate)
        .where(AutomationTemplate.enabled.is_(True))
        .order_by(AutomationTemplate.sort_order, AutomationTemplate.name)
    )
    if category:
        q = q.where(AutomationTemplate.category == category.strip().lower())
    rows = (await db.execute(q)).scalars().all()
    from app.services.automation_template_catalog import template_payload
    return {"templates": [template_payload(t) for t in rows]}


# ── Platform-native: grant requests ──────────────────────────────────


def _grant_card_payload(row: AutomationGrant) -> dict:
    try:
        target = json.loads(row.target_json or "{}")
    except (ValueError, TypeError):
        target = {}
    try:
        cadence = json.loads(row.cadence_json) if row.cadence_json else {}
    except (ValueError, TypeError):
        cadence = {}
    try:
        preview = json.loads(row.preview_json) if row.preview_json else None
    except (ValueError, TypeError):
        preview = None
    return {
        "id": row.id,
        "automation_id": row.automation_id,
        "connector_id": row.connector_id,
        "action": row.tool_name,
        "action_label": row.tool_name.split("__", 1)[-1].replace("_", " "),
        "target": target,
        "cadence": cadence,
        "mode": row.mode,
        "summary": row.summary,
        "preview": preview,
        "status": row.status,
        "created_at": row.created_at.isoformat() + "Z",
        "expires_at": row.expires_at.isoformat() + "Z",
        "decided_at": (row.decided_at.isoformat() + "Z")
        if row.decided_at else None,
        "decided_via": row.decided_via,
    }


async def _load_grant_or_404(
    db: AsyncSession, grant_id: str, user_id: str,
) -> AutomationGrant:
    # 404 on ownership miss, not 403 — an enumeration oracle over other
    # users' grants is the same leak the pending-actions module closed.
    row = (await db.execute(
        select(AutomationGrant)
        .where(AutomationGrant.id == grant_id)
        .where(AutomationGrant.user_id == user_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="No such grant request")
    return row


async def _notify_agent_grant_decided(
    db: AsyncSession, user_id: str, row: AutomationGrant,
) -> None:
    """Best-effort card update on the tenant. The grant row is the
    record; a sleeping agent must not turn a decision into a 5xx."""
    target = await _get_agent_target(user_id, db)
    if target is None:
        return
    agent_url, agent_key = target
    try:
        from app.services.agent_http import get_agent_http_client
        client = get_agent_http_client()
        await client.post(
            f"{agent_url.rstrip('/')}/api/automations/_grant_decided",
            json={"grant_id": row.id, "status": row.status,
                  "payload": _grant_card_payload(row)},
            headers={"X-Agent-Key": agent_key},
            timeout=5.0,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] grant-decided hook failed for %s: %s",
                       row.id, e)


class DecisionBody(BaseModel):
    decided_via: Optional[str] = None


@router.get("/grant-requests/{grant_id}")
async def get_grant_request(
    grant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    row = await _load_grant_or_404(db, grant_id, uid)
    # Lazy expiry — a card being looked at must not render actionable
    # past its deadline.
    if row.status == "pending" and row.expires_at <= datetime.utcnow():
        await db.execute(
            sa_update(AutomationGrant)
            .where(AutomationGrant.id == row.id)
            .where(AutomationGrant.status == "pending")
            .values(status="expired", decided_at=datetime.utcnow())
        )
        await db.commit()
        row = await _load_grant_or_404(db, grant_id, uid)
    return _grant_card_payload(row)


@router.post("/grant-requests/{grant_id}/approve")
async def approve_grant_request(
    grant_id: str,
    body: DecisionBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    row = await _load_grant_or_404(db, grant_id, uid)
    now = datetime.utcnow()
    if row.status != "pending":
        raise HTTPException(status_code=409,
                            detail=f"This was already {row.status}.")
    if row.expires_at <= now:
        await db.execute(
            sa_update(AutomationGrant)
            .where(AutomationGrant.id == row.id)
            .where(AutomationGrant.status == "pending")
            .values(status="expired", decided_at=now)
        )
        await db.commit()
        raise HTTPException(
            status_code=410,
            detail="This request expired before it was decided. Ask your "
                   "agent to prepare it again.",
        )
    # The claim — one statement, double-tap safe.
    claimed = await db.execute(
        sa_update(AutomationGrant)
        .where(AutomationGrant.id == row.id)
        .where(AutomationGrant.status == "pending")
        .values(
            status="approved",
            decided_at=now,
            decided_via=(body.decided_via or "web")[:32],
        )
    )
    await db.commit()
    if (claimed.rowcount or 0) != 1:
        raise HTTPException(status_code=409, detail="Already decided.")
    row = await _load_grant_or_404(db, grant_id, uid)
    await _notify_agent_grant_decided(db, uid, row)
    return _grant_card_payload(row)


@router.post("/grant-requests/{grant_id}/reject")
async def reject_grant_request(
    grant_id: str,
    body: DecisionBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    row = await _load_grant_or_404(db, grant_id, uid)
    if row.status == "rejected":
        return _grant_card_payload(row)   # double-tap no-op
    if row.status != "pending":
        raise HTTPException(status_code=409,
                            detail=f"This was already {row.status}.")
    await db.execute(
        sa_update(AutomationGrant)
        .where(AutomationGrant.id == row.id)
        .where(AutomationGrant.status == "pending")
        .values(
            status="rejected",
            decided_at=datetime.utcnow(),
            decided_via=(body.decided_via or "web")[:32],
        )
    )
    await db.commit()
    row = await _load_grant_or_404(db, grant_id, uid)
    await _notify_agent_grant_decided(db, uid, row)
    return _grant_card_payload(row)


# ── Revoke (standing grant → dead) ───────────────────────────────────


@router.post("/grant-requests/{grant_id}/revoke")
async def revoke_grant(
    grant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    row = await _load_grant_or_404(db, grant_id, uid)
    if row.status != "approved":
        raise HTTPException(status_code=409,
                            detail=f"Only an approved grant can be "
                                   f"revoked (this one is {row.status}).")
    await db.execute(
        sa_update(AutomationGrant)
        .where(AutomationGrant.id == row.id)
        .where(AutomationGrant.status == "approved")
        .values(status="revoked", revoked_at=datetime.utcnow())
    )
    await db.commit()
    row = await _load_grant_or_404(db, grant_id, uid)
    await _notify_agent_grant_decided(db, uid, row)
    return _grant_card_payload(row)


# ── Agent proxy (routines_proxy pattern) ─────────────────────────────


async def _get_agent_target(
    user_id: str, db: AsyncSession,
) -> Optional[Tuple[str, str]]:
    try:
        from app.db.models import AgentConfig
        result = await db.execute(
            select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
                AgentConfig.user_id == user_id,
                AgentConfig.deploy_status == "active",
            )
        )
        row = result.first()
        if row and row.agent_url and row.agent_api_key:
            return (row.agent_url, row.agent_api_key)
    except Exception as e:  # noqa: BLE001
        logger.warning("automations_proxy: agent target for %s: %s",
                       user_id, e)
    return None


async def _proxy(
    request: Request, sub_path: str, *,
    current_user: User, db: AsyncSession,
) -> Response:
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    target = await _get_agent_target(uid, db)
    if target is None:
        raise HTTPException(status_code=404,
                            detail="No active agent for this user")
    agent_url, agent_api_key = target
    url = f"{agent_url.rstrip('/')}/api/automations{sub_path}"
    headers = {
        "X-Agent-Key": agent_api_key,
        "content-type": request.headers.get("content-type",
                                            "application/json"),
        "accept": "application/json",
    }
    body = await request.body()
    from app.services.agent_http import get_agent_http_client
    try:
        client = get_agent_http_client()
        resp = await client.request(
            request.method.upper(), url,
            params=dict(request.query_params),
            headers=headers,
            content=body if body else None,
            timeout=30.0,
        )
    except httpx.RequestError as e:
        logger.warning("automations_proxy %s %s failed: %s",
                       request.method, url, e)
        raise HTTPException(status_code=502, detail="Agent unreachable")
    out_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    return Response(
        content=resp.content, status_code=resp.status_code,
        headers=out_headers,
        media_type=resp.headers.get("content-type"),
    )


@router.get("/runs")
async def proxy_runs(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "/runs", current_user=current_user, db=db)


@router.get("/auth-sessions/{session_id}")
async def proxy_get_auth_session(
    session_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/auth-sessions/{session_id}",
                        current_user=current_user, db=db)


@router.post("/auth-sessions/{session_id}/reject")
async def proxy_reject_auth_session(
    session_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/auth-sessions/{session_id}/reject",
                        current_user=current_user, db=db)


@router.post("/outbox/{outbox_id}/undo")
async def proxy_undo(
    outbox_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/outbox/{outbox_id}/undo",
                        current_user=current_user, db=db)


@router.get("")
async def proxy_list(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "", current_user=current_user, db=db)


@router.post("")
async def proxy_create(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "", current_user=current_user, db=db)


@router.patch("/{automation_id}")
async def proxy_update(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}",
                        current_user=current_user, db=db)


@router.delete("/{automation_id}")
async def proxy_delete(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/arm")
async def proxy_arm(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/arm",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/pause")
async def proxy_pause(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/pause",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/resume")
async def proxy_resume(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/resume",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/test-run")
async def proxy_test_run(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/test-run",
                        current_user=current_user, db=db)
