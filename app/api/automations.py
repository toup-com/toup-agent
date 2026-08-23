"""Agent-side automations HTTP surface (Round 26).

Mounted by agent_main behind the X-Agent-Key middleware (same trust
model as /api/routines: one user per container, `settings.user_id` is
the owner, the platform proxy forwards with the key). Every route 404s
while `settings.automations_enabled` is off — the surface does not
exist on a dark tenant, mirroring the routines kind gate.

The two `_hook` routes are platform→agent callbacks (OAuth completed,
grant decided). They ride the same middleware; the platform calls them
with the tenant's X-Agent-Key exactly like trigger dispatch.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, update as sa_update

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import Automation, AutomationAuthSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/automations", tags=["automations"])


def _user_id() -> str:
    return settings.user_id


def _flag_or_404() -> None:
    if not getattr(settings, "automations_enabled", False):
        raise HTTPException(status_code=404, detail="Feature not available")


# ── CRUD + lifecycle ─────────────────────────────────────────────────


class SpecBody(BaseModel):
    spec: dict = Field(...)
    template_slug: Optional[str] = Field(default=None, max_length=64)


@router.get("")
async def list_automations():
    _flag_or_404()
    from app.agent.automations.service import list_automations as _list
    async with async_session_maker() as db:
        return {"automations": await _list(db, _user_id())}


@router.post("")
async def create_automation(body: SpecBody):
    _flag_or_404()
    from app.agent.automations.service import create_automation as _create
    from app.agent.automations.spec import SpecError
    try:
        async with async_session_maker() as db:
            automation, _ = await _create(
                db, user_id=_user_id(), spec=body.spec,
                template_slug=body.template_slug,
            )
            from app.agent.automations.service import automation_payload
            return {"automation": automation_payload(automation)}
    except SpecError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})


@router.patch("/{automation_id}")
async def update_automation(automation_id: str, body: SpecBody):
    _flag_or_404()
    from app.agent.automations.service import (
        AutomationNotFound, automation_payload,
        update_automation as _update,
    )
    from app.agent.automations.spec import SpecError
    try:
        async with async_session_maker() as db:
            automation, _ = await _update(
                db, automation_id=automation_id, user_id=_user_id(),
                spec=body.spec,
            )
            return {"automation": automation_payload(automation)}
    except AutomationNotFound:
        raise HTTPException(status_code=404, detail="No such automation")
    except SpecError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})


@router.delete("/{automation_id}")
async def delete_automation(automation_id: str):
    _flag_or_404()
    from app.agent.automations.service import (
        AutomationNotFound, delete_automation as _delete,
    )
    try:
        async with async_session_maker() as db:
            await _delete(db, automation_id=automation_id,
                          user_id=_user_id())
    except AutomationNotFound:
        raise HTTPException(status_code=404, detail="No such automation")
    return {"deleted": True}


async def _lifecycle(automation_id: str, verb: str) -> dict:
    _flag_or_404()
    from app.agent.automations import service
    from app.agent.automations.compiler import CompileError
    from app.agent.automations.service import (
        AutomationNotFound, automation_payload,
    )
    fn = {
        "arm": service.arm_automation,
        "pause": service.pause_automation,
        "resume": service.resume_automation,
    }[verb]
    try:
        async with async_session_maker() as db:
            automation = await fn(
                db, automation_id=automation_id, user_id=_user_id(),
            )
            return {"automation": automation_payload(automation)}
    except AutomationNotFound:
        raise HTTPException(status_code=404, detail="No such automation")
    except CompileError as e:
        raise HTTPException(status_code=409,
                            detail={"code": e.code, "message": str(e)})


@router.post("/{automation_id}/arm")
async def arm(automation_id: str):
    return await _lifecycle(automation_id, "arm")


@router.post("/{automation_id}/pause")
async def pause(automation_id: str):
    return await _lifecycle(automation_id, "pause")


@router.post("/{automation_id}/resume")
async def resume(automation_id: str):
    return await _lifecycle(automation_id, "resume")


@router.post("/{automation_id}/test-run")
async def test_run(automation_id: str):
    _flag_or_404()
    from app.agent.automations.service import (
        AutomationNotFound, test_run as _test,
    )
    try:
        async with async_session_maker() as db:
            return await _test(db, automation_id=automation_id,
                               user_id=_user_id())
    except AutomationNotFound:
        raise HTTPException(status_code=404, detail="No such automation")


@router.get("/runs")
async def list_runs(
    automation_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    _flag_or_404()
    from app.agent.automations.service import list_runs as _runs
    async with async_session_maker() as db:
        return {"runs": await _runs(
            db, _user_id(), automation_id=automation_id, limit=limit,
        )}


# ── Outbox undo ──────────────────────────────────────────────────────


@router.post("/outbox/{outbox_id}/undo")
async def undo_outbox(outbox_id: str):
    _flag_or_404()
    from app.agent.automations.outbox import undo_row
    async with async_session_maker() as db:
        ok = await undo_row(db, outbox_id, _user_id())
    if not ok:
        raise HTTPException(
            status_code=409,
            detail="Too late — this write already went out (or was "
                   "already cancelled).",
        )
    return {"undone": True}


# ── Connector-card auth sessions ─────────────────────────────────────


def _session_payload(s: AutomationAuthSession, *, name: str = "",
                     icon: Optional[str] = None) -> dict:
    from app.agent.automations.cards import connector_card_payload
    try:
        scopes = json.loads(s.scopes_json) if s.scopes_json else []
    except (ValueError, TypeError):
        scopes = []
    return connector_card_payload(
        s, name=name or s.connector_id, icon=icon, scopes=scopes,
    )


async def _expire_lazily(db, s: AutomationAuthSession) -> None:
    if s.status in ("offered", "connecting") \
            and s.expires_at <= datetime.utcnow():
        s.status = "expired"
        s.decided_at = datetime.utcnow()
        await db.commit()


@router.get("/auth-sessions/{session_id}")
async def get_auth_session(session_id: str):
    _flag_or_404()
    async with async_session_maker() as db:
        s = (await db.execute(
            select(AutomationAuthSession)
            .where(AutomationAuthSession.id == session_id)
            .where(AutomationAuthSession.user_id == _user_id())
        )).scalar_one_or_none()
        if s is None:
            raise HTTPException(status_code=404, detail="No such session")
        await _expire_lazily(db, s)
        return _session_payload(s)


@router.post("/auth-sessions/{session_id}/reject")
async def reject_auth_session(session_id: str):
    _flag_or_404()
    async with async_session_maker() as db:
        res = await db.execute(
            sa_update(AutomationAuthSession)
            .where(AutomationAuthSession.id == session_id)
            .where(AutomationAuthSession.user_id == _user_id())
            .where(AutomationAuthSession.status.in_(
                ("offered", "connecting", "failed"),
            ))
            .values(status="rejected", decided_at=datetime.utcnow())
        )
        await db.commit()
        s = (await db.execute(
            select(AutomationAuthSession)
            .where(AutomationAuthSession.id == session_id)
            .where(AutomationAuthSession.user_id == _user_id())
        )).scalar_one_or_none()
        if s is None:
            raise HTTPException(status_code=404, detail="No such session")
        if (res.rowcount or 0) == 1:
            await _refresh_card(db, s)
        return _session_payload(s)


async def _refresh_card(db, s: AutomationAuthSession) -> None:
    from app.agent.automations import cards
    payload = _session_payload(s)
    await cards.update_card_message(
        db, message_id=s.message_id,
        metadata_key=cards.CONNECTOR_CARD_KEY, payload=payload,
    )
    await cards.broadcast_card(
        s.user_id, cards.CONNECTOR_CARD_KEY, payload,
    )


# ── Platform → agent hooks ───────────────────────────────────────────


class ConnectorHook(BaseModel):
    connector_id: str = Field(..., min_length=1, max_length=64)
    ok: bool = True
    error: Optional[str] = Field(default=None, max_length=300)


@router.post("/_connector_connected")
async def connector_connected_hook(body: ConnectorHook):
    """OAuth callback landed on the platform for this tenant. Resolve
    every open connector-card session for that connector and update the
    cards in place. Idempotent; unknown connector is a no-op 200 (the
    platform fires this on EVERY connect, most of which have no open
    card)."""
    _flag_or_404()
    updated = 0
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationAuthSession)
            .where(AutomationAuthSession.user_id == _user_id())
            .where(AutomationAuthSession.connector_id == body.connector_id)
            .where(AutomationAuthSession.status.in_(
                ("offered", "connecting"),
            ))
        )).scalars().all()
        for s in rows:
            await _expire_lazily(db, s)
            if s.status not in ("offered", "connecting"):
                continue
            if body.ok:
                s.status = "connected"
            elif s.retry_used:
                s.status = "failed"
                s.decided_at = datetime.utcnow()
            else:
                s.status = "failed"
                s.retry_used = True
            s.decided_at = datetime.utcnow()
            await db.commit()
            await _refresh_card(db, s)
            updated += 1
    # Fresh scopes may have arrived — the registry cache must not serve
    # the pre-connect view for the next 5 minutes.
    from app.agent.automations.registry import invalidate_cache
    invalidate_cache()
    return {"updated": updated}


class GrantHook(BaseModel):
    grant_id: str = Field(..., min_length=1, max_length=36)
    status: str = Field(..., max_length=16)
    payload: dict = Field(default_factory=dict)


@router.post("/_grant_decided")
async def grant_decided_hook(body: GrantHook):
    """The user decided a grant request on the platform. Update the
    grant card message in place + re-broadcast. The card row is found
    by its grant id inside metadata_json — decisions are rare enough
    that the scan (bounded to automation-source messages) is fine."""
    _flag_or_404()
    from app.agent.automations import cards
    from app.db.models import Message

    async with async_session_maker() as db:
        msg = (await db.execute(
            select(Message)
            .where(Message.source == "automation")
            .where(Message.metadata_json.like(f"%{body.grant_id}%"))
            .order_by(Message.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if msg is not None:
            try:
                meta = json.loads(msg.metadata_json or "{}")
            except (ValueError, TypeError):
                meta = {}
            card = meta.get(cards.GRANT_CARD_KEY) or {}
            card.update(body.payload or {})
            card["status"] = body.status
            meta[cards.GRANT_CARD_KEY] = card
            msg.metadata_json = json.dumps(meta, default=str)
            await db.commit()
            await cards.broadcast_card(
                _user_id(), cards.GRANT_CARD_KEY, card,
            )
    return {"ok": True}
