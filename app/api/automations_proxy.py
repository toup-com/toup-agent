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
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db.database import get_db
from app.db.models import AutomationGrant, AutomationTemplate, User

router = APIRouter(prefix="/automations", tags=["automations (platform)"])
logger = logging.getLogger(__name__)

# Below the mobile client's 15 s per-attempt budget by a clear margin, so the
# platform is always the side that answers first on a slow tenant.
_READ_TIMEOUT_S = 12.0
# A mutation is not abandoned early: see `_proxy`.
_WRITE_TIMEOUT_S = 30.0

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


async def _revoke_grant_row(
    db: AsyncSession, uid: str, row: AutomationGrant,
) -> AutomationGrant:
    """The one revoke transition (guarded), shared by the flat and the
    R29 nested route. The agent hook it fires also pauses a dependent
    armed automation (`paused_reason="grant_revoked"`) — the dispatcher
    already fails closed, this makes the STATE honest."""
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
    row = await _load_grant_or_404(db, row.id, uid)
    await _notify_agent_grant_decided(db, uid, row)
    return row


@router.post("/grant-requests/{grant_id}/revoke")
async def revoke_grant(
    grant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    row = await _load_grant_or_404(db, grant_id, uid)
    row = await _revoke_grant_row(db, uid, row)
    return _grant_card_payload(row)


# ── Grants on the Overview (Round 29, platform-native) ───────────────


@router.get("/{automation_id}/grants")
async def list_automation_grants(
    automation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Every live grant backing one automation — approved and pending
    both (the client may filter); revoked/expired history stays off
    the Overview."""
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    rows = (await db.execute(
        select(AutomationGrant)
        .where(AutomationGrant.user_id == uid)
        .where(AutomationGrant.automation_id == automation_id)
        .where(AutomationGrant.status.in_(("approved", "pending")))
        .order_by(AutomationGrant.created_at.desc())
    )).scalars().all()
    grants = []
    for row in rows:
        payload = _grant_card_payload(row)
        payload["granted_at"] = payload.get("decided_at")
        grants.append(payload)
    return {"grants": grants}


@router.post("/{automation_id}/grants/{grant_id}/revoke")
async def revoke_automation_grant(
    automation_id: str, grant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    row = await _load_grant_or_404(db, grant_id, uid)
    if row.automation_id != automation_id:
        raise HTTPException(status_code=404, detail="No such grant request")
    row = await _revoke_grant_row(db, uid, row)
    return _grant_card_payload(row)


class ModePatchBody(BaseModel):
    mode: str


@router.patch("/{automation_id}/mode")
async def patch_automation_mode(
    automation_id: str, request: Request, body: ModePatchBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Flip auto/confirm for one automation — BOTH halves
    (CONTRACTS-R29.md §3.3): the spec via the agent first, then this
    automation's approved grants. User-JWT only; the grant mode is
    consent and the authenticated Overview is the consent surface. No
    agent RPC can reach the grant half."""
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    if body.mode not in ("auto", "confirm"):
        raise HTTPException(status_code=422,
                            detail="mode must be auto or confirm")
    resp = await _proxy(request, f"/{automation_id}/mode",
                        current_user=current_user, db=db)
    if resp.status_code >= 300:
        return resp
    await db.execute(
        sa_update(AutomationGrant)
        .where(AutomationGrant.user_id == uid)
        .where(AutomationGrant.automation_id == automation_id)
        .where(AutomationGrant.status == "approved")
        .values(mode=body.mode, mode_changed_at=datetime.utcnow())
    )
    await db.commit()
    return resp


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


# The agent's own gate answers `404 Feature not available` when ITS
# `automations_enabled` is off — the same status and the same words the
# PLATFORM gate uses to mean "not for you". Once the platform gate has already
# passed, those two mean opposite things:
#
#   platform 404  the feature is not on for this account
#   agent    404  it IS on, and this tenant's container has not caught up
#
# The second is temporary and self-healing (the bridge's reconciler upgrades
# assigned pool slots on its own). Passing it through as a 404 told the app the
# feature was off for a user it had just been switched on for — and because the
# suggestion routes are platform-native and answer 200, that user saw a full
# sheet of suggestions and an error the moment they pressed Set up.
_AGENT_DARK_DETAIL = "Feature not available"


def _translate_agent_dark(resp) -> Optional[JSONResponse]:
    """503 for a tenant whose engine has not caught up, never 404."""
    if resp.status_code != 404:
        return None
    try:
        if (resp.json() or {}).get("detail") != _AGENT_DARK_DETAIL:
            return None
    except Exception:  # noqa: BLE001 — not JSON ⇒ not this case
        return None
    return JSONResponse(
        status_code=503,
        content={"detail": "agent_starting"},
        headers={"Retry-After": "120"},
    )


async def _proxy(
    request: Request, sub_path: str, *,
    current_user: User, db: AsyncSession,
) -> Response:
    uid = str(current_user.id)
    await _flag_or_404(db, uid)
    target = await _get_agent_target(uid, db)
    if target is None:
        # WHY THIS IS NOT A 404. The client's rule is "404/405 = the feature is
        # not on this backend, render absence" — so this answer painted "No
        # automations yet" over a full account. But `_get_agent_target`
        # requires `deploy_status == 'active'`, and a container that is
        # redeploying, provisioning or errored is none of those while the
        # user's automations sit intact inside it. `_flag_or_404` above is the
        # one honest 404 here: the feature really is off for this user.
        #
        # A user who has never set an agent up is the remaining case, and 503
        # is right for them too — the platform is mid-provisioning, and the
        # onboarding surface, not this screen, is what tells them so.
        from app.db.models import AgentConfig as _AC
        try:
            row = (await db.execute(
                select(_AC.deploy_status).where(_AC.user_id == uid)
            )).first()
        except Exception:
            row = None
        state = (row[0] if row else None) or "none"
        logger.info("[automations_proxy] no active agent for %s (deploy_status=%s)",
                    uid[:8], state)
        raise HTTPException(
            status_code=503,
            detail="Your agent is still starting up. Nothing has changed — try again in a moment.",
            headers={"Retry-After": "5", "X-Toup-Reason": "agent_provisioning"},
        )
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
    method = request.method.upper()
    # THE TIMEOUT HIERARCHY, which was inverted. The mobile client aborts every
    # request at 15 s (`api.ts DEFAULT_TIMEOUT_MS`); this proxy waited 30 s. So
    # for the whole 15–30 s band the phone had already given up and drawn
    # "The server didn't answer" while the platform was still patiently holding
    # the connection — and when the tenant finally answered, that answer went
    # nowhere. Measured 2026-08-31: `GET /api/automations/summary -> 200
    # 22677ms`, a correct response the user never saw.
    #
    # A READ now fails fast and honestly INSIDE the client's budget, so the
    # client's retry (which it now has) is what recovers the blip. A WRITE
    # keeps the long budget: abandoning a mutation mid-flight does not undo it,
    # it only makes the outcome unknown to both sides.
    read_only = method in ("GET", "HEAD")
    timeout_s = _READ_TIMEOUT_S if read_only else _WRITE_TIMEOUT_S
    try:
        client = get_agent_http_client()
        resp = await client.request(
            method, url,
            params=dict(request.query_params),
            headers=headers,
            content=body if body else None,
            timeout=timeout_s,
        )
    except httpx.TimeoutException as e:
        # 504, not 502: "answered too slowly" and "did not answer at all" are
        # different operational facts and the trail could not tell them apart —
        # `str()` of an httpx timeout is the empty string, so the old
        # "failed: %s" logged nothing at all after the colon.
        logger.warning("automations_proxy %s %s timed out after %.0fs: %r",
                       method, url, timeout_s, e)
        raise HTTPException(
            status_code=504,
            detail="Your agent is taking longer than usual. Nothing has changed — try again.",
            headers={"Retry-After": "3", "X-Toup-Reason": "agent_slow"},
        )
    except httpx.RequestError as e:
        logger.warning("automations_proxy %s %s failed: %r",
                       method, url, e)
        raise HTTPException(
            status_code=502,
            detail="Agent unreachable",
            headers={"Retry-After": "3", "X-Toup-Reason": "agent_unreachable"},
        )
    dark = _translate_agent_dark(resp)
    if dark is not None:
        logger.warning(
            "[automations_proxy] tenant engine dark for %s on %s — container "
            "predates the launch image", uid[:8], sub_path,
        )
        return dark
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


@router.get("/{automation_id}/thread")
async def proxy_thread(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/thread",
                        current_user=current_user, db=db)


@router.get("/{automation_id}/memory")
async def proxy_memory(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/memory",
                        current_user=current_user, db=db)


# ── Round 29 pass-throughs (seen, facts, schedule, membership) ───────


@router.post("/{automation_id}/seen")
async def proxy_seen(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/seen",
                        current_user=current_user, db=db)


@router.get("/{automation_id}/memory/facts")
async def proxy_list_facts(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/memory/facts",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/memory/facts")
async def proxy_add_fact(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/memory/facts",
                        current_user=current_user, db=db)


@router.patch("/{automation_id}/memory/facts/{fact_id}")
async def proxy_update_fact(
    automation_id: str, fact_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/memory/facts/{fact_id}",
                        current_user=current_user, db=db)


@router.delete("/{automation_id}/memory/facts/{fact_id}")
async def proxy_delete_fact(
    automation_id: str, fact_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/memory/facts/{fact_id}",
                        current_user=current_user, db=db)


@router.patch("/{automation_id}/schedule")
async def proxy_schedule(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/schedule",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/connectors/{connector_id}")
async def proxy_add_connector(
    automation_id: str, connector_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request,
                        f"/{automation_id}/connectors/{connector_id}",
                        current_user=current_user, db=db)


@router.delete("/{automation_id}/connectors/{connector_id}")
async def proxy_remove_connector(
    automation_id: str, connector_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request,
                        f"/{automation_id}/connectors/{connector_id}",
                        current_user=current_user, db=db)


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


# ── R30 pass-throughs (summary, threads, stop/resume, workflow) ──────
# Appended literals never collide with the R26 parametrized routes:
# every new literal here differs in segment count from the
# {automation_id} family, so declaration order stays safe.


@router.get("/summary")
async def proxy_summary(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "/summary",
                        current_user=current_user, db=db)


@router.get("/{automation_id}/runs")
async def proxy_nested_runs(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/runs",
                        current_user=current_user, db=db)


@router.post("/runs/{run_id}/stop")
async def proxy_stop_run(
    run_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/runs/{run_id}/stop",
                        current_user=current_user, db=db)


@router.post("/runs/{run_id}/resume")
async def proxy_resume_run(
    run_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/runs/{run_id}/resume",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/run-now")
async def proxy_run_now(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/run-now",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/thread/messages")
async def proxy_thread_message(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/thread/messages",
                        current_user=current_user, db=db)


@router.get("/{automation_id}/workflow")
async def proxy_workflow(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/workflow",
                        current_user=current_user, db=db)


@router.put("/{automation_id}/workflow/schedule")
async def proxy_workflow_schedule(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/workflow/schedule",
                        current_user=current_user, db=db)


@router.put("/{automation_id}/workflow/steps")
async def proxy_workflow_steps(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/workflow/steps",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/workflow/rules")
async def proxy_workflow_rule_add(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/workflow/rules",
                        current_user=current_user, db=db)


@router.put("/{automation_id}/workflow/rules/{rule_id}")
async def proxy_workflow_rule_edit(
    automation_id: str, rule_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(
        request, f"/{automation_id}/workflow/rules/{rule_id}",
        current_user=current_user, db=db,
    )


@router.delete("/{automation_id}/workflow/rules/{rule_id}")
async def proxy_workflow_rule_delete(
    automation_id: str, rule_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(
        request, f"/{automation_id}/workflow/rules/{rule_id}",
        current_user=current_user, db=db,
    )


@router.put("/{automation_id}/workflow/accounts/{account_id}/permissions")
async def proxy_workflow_permissions(
    automation_id: str, account_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(
        request,
        f"/{automation_id}/workflow/accounts/{account_id}/permissions",
        current_user=current_user, db=db,
    )


@router.post("/{automation_id}/workflow/accounts")
async def proxy_workflow_account_add(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/workflow/accounts",
                        current_user=current_user, db=db)


@router.delete("/{automation_id}/workflow/accounts/{account_id}")
async def proxy_workflow_account_remove(
    automation_id: str, account_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(
        request, f"/{automation_id}/workflow/accounts/{account_id}",
        current_user=current_user, db=db,
    )


@router.get("/{automation_id}/workflow/accounts/{account_id}/contents")
async def proxy_account_contents(
    automation_id: str, account_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """R38 — what is inside the account the user just tapped. The agent
    side holds its own deadline well inside this forwarder's 30 s, so a
    slow provider comes back as a named reason rather than a 502."""
    return await _proxy(
        request,
        f"/{automation_id}/workflow/accounts/{account_id}/contents",
        current_user=current_user, db=db,
    )


@router.post("/{automation_id}/workflow/accounts/{account_id}/focus")
async def proxy_account_focus_add(
    automation_id: str, account_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(
        request, f"/{automation_id}/workflow/accounts/{account_id}/focus",
        current_user=current_user, db=db,
    )


@router.delete("/{automation_id}/workflow/accounts/{account_id}/focus")
async def proxy_account_focus_remove(
    automation_id: str, account_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(
        request, f"/{automation_id}/workflow/accounts/{account_id}/focus",
        current_user=current_user, db=db,
    )


@router.post("/{automation_id}/workflow/ask")
async def proxy_workflow_ask(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/workflow/ask",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/workflow/undo")
async def proxy_workflow_undo(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{automation_id}/workflow/undo",
                        current_user=current_user, db=db)


@router.post("/from-template")
async def proxy_from_template(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "/from-template",
                        current_user=current_user, db=db)


@router.post("/describe")
async def proxy_describe(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "/describe",
                        current_user=current_user, db=db)


# ── R30 §4.6 — the served catalog (PLATFORM-NATIVE: templates and
# identity states are platform rows; no agent hop) ───────────────────


@router.get("/catalog")
async def catalog(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _flag_or_404(db, str(current_user.id))
    from sqlalchemy import select as _select
    from app.db.models import ConnectorIdentity
    from app.services.automation_template_catalog import template_payload

    templates = list((await db.execute(
        _select(AutomationTemplate)
        .where(AutomationTemplate.enabled.is_(True))
        .order_by(AutomationTemplate.sort_order,
                  AutomationTemplate.name)
    )).scalars())
    identities = {
        r.connector_id: r
        for r in (await db.execute(
            _select(ConnectorIdentity)
            .where(ConnectorIdentity.user_id == current_user.id)
        )).scalars()
    }

    def _meta(t) -> str:
        try:
            connectors = json.loads(t.connectors_json or "[]")
        except (ValueError, TypeError):
            connectors = []
        active = [c for c in connectors
                  if (identities.get(c) is not None
                      and identities[c].status == "active")]
        missing = [c for c in connectors if c not in active]
        from app.services.automation_verbs import display_name
        try:
            spec = json.loads(t.spec_json or "{}")
        except (ValueError, TypeError):
            spec = {}
        has_writes = any(
            isinstance(s, dict) and s.get("grant_target") is not None
            for s in (spec.get("steps") or [])
        ) or "{{grant.target.id}}" in (t.spec_json or "")
        if not missing:
            if len(connectors) == 1:
                name = display_name(connectors[0]) or connectors[0]
                return f"{name} · connected"
            if not has_writes:
                return "Reads only"
            return f"{len(connectors)} connected"
        name = display_name(missing[0]) or missing[0]
        return f"{name} on setup"

    cards = []
    for t in templates:
        payload = template_payload(t)
        try:
            connectors = json.loads(t.connectors_json or "[]")
        except (ValueError, TypeError):
            connectors = []
        cards.append({
            "id": t.slug,
            "cats": [t.category] if t.category else [],
            "title": t.name,
            "when": payload.get("cadence_human") or "",
            "desc": t.description or "",
            "icons": connectors,
            "meta": _meta(t),
        })
    # unused_count is authoritative on /summary (the agent knows which
    # templates this user already set up; the platform does not) —
    # served null here so nothing renders a wrong badge from this route.
    return {"cards": cards, "unused_count": None}


# ── R30 §4.7 — accounts card/reconnect proxy (own prefix) ────────────

accounts_proxy_router = APIRouter(prefix="/accounts", tags=["accounts"])


async def _proxy_accounts(
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
    url = f"{agent_url.rstrip('/')}/api/accounts{sub_path}"
    from app.services.agent_http import get_agent_http_client
    # R31: forward the BODY.
    #
    # This never passed `content=`, which was harmless while the two
    # accounts routes were a GET and a bodyless POST — and became a
    # silent data loss the moment `reconnect` grew `add_scopes` and
    # `probe` grew `force`. A dropped body does not error: the agent
    # sees an empty request and answers plausibly, so `Grant access`
    # would have run an ordinary reconnect and told the user it was
    # done. `_proxy` (the automations sibling) has always relayed it.
    body = await request.body()
    headers = {"X-Agent-Key": agent_api_key, "accept": "application/json"}
    ctype = request.headers.get("content-type")
    if ctype:
        headers["content-type"] = ctype
    try:
        client = get_agent_http_client()
        resp = await client.request(
            request.method.upper(), url,
            params=dict(request.query_params),
            content=body or None,
            headers=headers,
            timeout=30.0,
        )
    except httpx.RequestError as e:
        logger.warning("accounts_proxy %s %s failed: %s",
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


@accounts_proxy_router.get("/{account_id}/card")
async def proxy_account_card(
    account_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy_accounts(request, f"/{account_id}/card",
                                 current_user=current_user, db=db)


@accounts_proxy_router.post("/{account_id}/reconnect")
async def proxy_account_reconnect(
    account_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy_accounts(request, f"/{account_id}/reconnect",
                                 current_user=current_user, db=db)


@accounts_proxy_router.post("/{account_id}/probe")
async def proxy_account_probe(
    account_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """R31 §4.4 — ask the vendor now. `force` bypasses the ten-minute
    cache and is what the two user-initiated paths send."""
    return await _proxy_accounts(request, f"/{account_id}/probe",
                                 current_user=current_user, db=db)


@router.post("/{automation_id}/workflow/commit")
async def proxy_workflow_commit(
    automation_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """R31 §4.6 — the workflow's one commit."""
    return await _proxy(request, f"/{automation_id}/workflow/commit",
                        current_user=current_user, db=db)


@router.post("/{automation_id}/runs/{run_id}/resume-source")
async def proxy_resume_source(
    automation_id: str, run_id: str, request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """R31 §4.2a — the `Try again` button on a needs-you card."""
    return await _proxy(
        request, f"/{automation_id}/runs/{run_id}/resume-source",
        current_user=current_user, db=db)


@router.post("/purge-junk-facts")
async def proxy_purge_junk_facts(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Round 33, item 6 — purge the curator's own failure reports."""
    return await _proxy(request, "/purge-junk-facts",
                        current_user=current_user, db=db)


@router.post("/backfill")
async def proxy_backfill(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """R31-18 — rules + thread-fact scope back-fill."""
    return await _proxy(request, "/backfill",
                        current_user=current_user, db=db)


@router.post("/cleanup-day-chat")
async def proxy_cleanup_day_chat(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """R31 §4.1 — move the leaked rows out of the day chat."""
    return await _proxy(request, "/cleanup-day-chat",
                        current_user=current_user, db=db)


@router.post("/migrate-routines")
async def proxy_migrate_routines(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "/migrate-routines",
                        current_user=current_user, db=db)


@router.post("/migrate-routines/repair")
async def proxy_migrate_routines_repair(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "/migrate-routines/repair",
                        current_user=current_user, db=db)


@router.get("/migrate-routines/report")
async def proxy_migrate_routines_report(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "/migrate-routines/report",
                        current_user=current_user, db=db)
