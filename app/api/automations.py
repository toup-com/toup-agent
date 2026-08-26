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
async def delete_automation(automation_id: str,
                            undo: int = Query(default=0, ge=0, le=1)):
    """R30 §4.6/§4.8: `?undo=1` hard-deletes (allowed only until the
    first run starts); the plain delete is SOFT — schedule disarmed,
    thread archived 30 days, drafts untouched."""
    _flag_or_404()
    from app.agent.automations.service import (
        AutomationNotFound, MembershipError, delete_automation as _delete,
    )
    try:
        async with async_session_maker() as db:
            await _delete(db, automation_id=automation_id,
                          user_id=_user_id(), undo=bool(undo))
    except AutomationNotFound:
        raise HTTPException(status_code=404, detail="No such automation")
    except MembershipError as e:
        raise HTTPException(status_code=409, detail={"code": e.code})
    return {"deleted": True, "undo": bool(undo)}


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


# ── Session thread + working memory (Round 28) ──────────────────────


async def _owned_automation_or_404(db, automation_id: str) -> "Automation":
    a = (await db.execute(
        select(Automation).where(
            Automation.id == automation_id,
            Automation.user_id == _user_id(),
        )
    )).scalar_one_or_none()
    if a is None:
        raise HTTPException(status_code=404, detail="No such automation")
    return a


@router.get("/{automation_id}/thread")
async def automation_thread(
    automation_id: str,
    limit: int = Query(default=100, ge=1, le=200),
    before: Optional[str] = Query(default=None, max_length=40),
):
    """The automation's session thread. `session_id` is TODAY's
    conversation row, minted lazily on this GET so a composer can
    always bind to it; `messages` spill over previous days (newest
    `limit`, returned oldest-first) and use the exact
    /api/sessions/{id}/messages serialization — cards and run markers
    hydrate through the same pipeline.
    """
    _flag_or_404()
    from app.agent.automations.session import (
        list_session_conversation_ids, resolve_session_conversation,
    )
    from app.api.message_cards import attach_run_to_cards, load_build_jobs
    from app.api.sessions import _message_to_response
    from app.db.models import Message

    async with async_session_maker() as db:
        a = await _owned_automation_or_404(db, automation_id)
        conv, _day = await resolve_session_conversation(
            db, user_id=_user_id(), automation_id=automation_id,
            title=a.name,
        )
        await db.commit()
        conv_ids = await list_session_conversation_ids(
            db, user_id=_user_id(), automation_id=automation_id,
        )
        if conv.id not in conv_ids:
            conv_ids.append(conv.id)
        rows = (await db.execute(
            select(Message)
            .where(Message.conversation_id.in_(conv_ids))
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )).scalars().all()
        messages = list(reversed(rows))
        build_jobs = await load_build_jobs(db, messages)
        from app.agent.reply_quote import (
            resolve_reply_targets_for_serialization,
        )
        reply_targets = await resolve_reply_targets_for_serialization(
            db, messages,
        )
        channels = {cid: "automation" for cid in conv_ids}

        # R30 §4.10: the first-class thread rides the same route —
        # {thread_id, turns, has_more, tz} beside the legacy keys until
        # B flips (CONTRACTS-R30 §9 retires the legacy pair).
        from app.agent.automations import ledger as _ledger
        thread = await _ledger.ensure_thread(
            db, user_id=_user_id(), automation_id=automation_id,
        )
        # Direct-call tests bypass FastAPI's Query resolution — a bare
        # default is a Query object, never a turn id.
        _before = before if isinstance(before, str) else None
        _limit = limit if isinstance(limit, int) else 100
        turns, has_more = await _ledger.list_turns(
            db, thread_id=thread.id, before=_before, limit=_limit,
        )
        from app.agent._user_tz_cache import get_cached_user_tz
        return {
            "session_id": conv.id,
            # R29: `session_id` is TODAY's conversation row, not the
            # automation id the deep links use — serve both explicitly.
            "automation_id": automation_id,
            "messages": attach_run_to_cards([
                _message_to_response(m, build_jobs, reply_targets, channels)
                for m in messages
            ]),
            "thread_id": thread.id,
            "turns": turns,
            "has_more": has_more,
            "tz": get_cached_user_tz(_user_id()),
        }


@router.get("/{automation_id}/memory")
async def automation_memory(automation_id: str):
    """R30 §4.5 / §3.10 — this automation's scoped view of the platform
    memory: five categories, always present, with evidence and dates.
    The R28 engine-state body is RETIRED from this route (D-07: the raw
    ISO row must never reach a UI again); the state row itself stays
    internal to the executor. Legacy keys are served null so the
    pre-rebuild app renders nothing rather than crashing."""
    _flag_or_404()
    async with async_session_maker() as db:
        await _owned_automation_or_404(db, automation_id)
        from app.services.memory_v2_service import list_facts_for_scope
        payload = await list_facts_for_scope(
            db, user_id=_user_id(), scope=automation_id,
        )
        from app.agent._user_tz_cache import get_cached_user_tz
        payload["tz"] = get_cached_user_tz(_user_id())
        payload["content"] = None
        payload["metadata"] = {}
        return payload


# ── Seen, curated facts, schedule/mode, membership (Round 29) ────────


@router.post("/{automation_id}/seen")
async def mark_seen(automation_id: str):
    """CAS-style read receipt for the last outcome — B calls it when
    the session screen opens (CONTRACTS-R29.md §2)."""
    _flag_or_404()
    from app.agent.automations.service import (
        AutomationNotFound, mark_outcome_seen,
    )
    try:
        async with async_session_maker() as db:
            await mark_outcome_seen(
                db, automation_id=automation_id, user_id=_user_id(),
            )
    except AutomationNotFound:
        raise HTTPException(status_code=404, detail="No such automation")
    return {"seen": True}


class FactBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=400)
    category: str = Field(..., min_length=2, max_length=32)


class FactPatchBody(BaseModel):
    text: Optional[str] = Field(default=None, min_length=1, max_length=400)
    category: Optional[str] = Field(default=None, min_length=2, max_length=32)


@router.get("/{automation_id}/memory/facts")
async def list_memory_facts(automation_id: str):
    """The Memory tab's curated facts. An empty ledger is 200 with an
    empty list — 404 means the automation doesn't exist, never "no
    facts yet" (absence of facts is not absence of the feature)."""
    _flag_or_404()
    from app.agent.automations import facts
    async with async_session_maker() as db:
        await _owned_automation_or_404(db, automation_id)
        return await facts.list_facts(
            db, user_id=_user_id(), automation_id=automation_id,
        )


@router.post("/{automation_id}/memory/facts")
async def add_memory_fact(automation_id: str, body: FactBody):
    _flag_or_404()
    from app.agent.automations import facts
    async with async_session_maker() as db:
        await _owned_automation_or_404(db, automation_id)
        fact = await facts.add_fact(
            db, user_id=_user_id(), automation_id=automation_id,
            text=body.text, category=body.category,
        )
    if fact is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "bad_fact",
                    "message": "category must be a lowercase slug and "
                               "the text non-empty (duplicates are "
                               "refused)"},
        )
    return {"fact": fact}


@router.patch("/{automation_id}/memory/facts/{fact_id}")
async def update_memory_fact(
    automation_id: str, fact_id: str, body: FactPatchBody,
):
    _flag_or_404()
    from app.agent.automations import facts
    async with async_session_maker() as db:
        await _owned_automation_or_404(db, automation_id)
        fact = await facts.update_fact(
            db, user_id=_user_id(), automation_id=automation_id,
            fact_id=fact_id, text=body.text, category=body.category,
        )
    if fact is None:
        raise HTTPException(status_code=404, detail="No such fact")
    return {"fact": fact}


@router.delete("/{automation_id}/memory/facts/{fact_id}")
async def delete_memory_fact(automation_id: str, fact_id: str):
    _flag_or_404()
    from app.agent.automations import facts
    async with async_session_maker() as db:
        await _owned_automation_or_404(db, automation_id)
        ok = await facts.delete_fact(
            db, user_id=_user_id(), automation_id=automation_id,
            fact_id=fact_id,
        )
    if not ok:
        raise HTTPException(status_code=404, detail="No such fact")
    return {"deleted": True}


class ScheduleBody(BaseModel):
    cron_local: Optional[str] = Field(default=None, max_length=64)
    at: Optional[str] = Field(default=None, max_length=16)
    every_s: Optional[int] = Field(default=None, ge=1)


class ModeBody(BaseModel):
    mode: str = Field(..., pattern="^(auto|confirm)$")


async def _spec_edit(fn, automation_id: str, **kwargs) -> dict:
    """Shared plumbing for the focused spec edits: MembershipError →
    409 with its stable code, SpecError → 422 like PATCH."""
    from app.agent.automations.service import (
        AutomationNotFound, MembershipError, automation_payload,
    )
    from app.agent.automations.spec import SpecError
    try:
        async with async_session_maker() as db:
            automation, _ = await fn(
                db, automation_id=automation_id, user_id=_user_id(),
                **kwargs,
            )
            return {"automation": automation_payload(automation)}
    except AutomationNotFound:
        raise HTTPException(status_code=404, detail="No such automation")
    except MembershipError as e:
        raise HTTPException(status_code=409,
                            detail={"code": e.code, "message": str(e)})
    except SpecError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})


@router.patch("/{automation_id}/schedule")
async def patch_schedule(automation_id: str, body: ScheduleBody):
    _flag_or_404()
    from app.agent.automations.service import set_schedule
    schedule = {
        k: v for k, v in (
            ("cron_local", body.cron_local), ("at", body.at),
            ("every_s", body.every_s),
        ) if v is not None
    }
    return await _spec_edit(set_schedule, automation_id, schedule=schedule)


@router.patch("/{automation_id}/mode")
async def patch_mode(automation_id: str, body: ModeBody):
    _flag_or_404()
    from app.agent.automations.service import set_mode
    return await _spec_edit(set_mode, automation_id, mode=body.mode)


@router.post("/{automation_id}/connectors/{connector_id}")
async def add_connector_membership(automation_id: str, connector_id: str):
    _flag_or_404()
    from app.agent.automations.service import add_connector
    return await _spec_edit(add_connector, automation_id,
                            connector_id=connector_id)


@router.delete("/{automation_id}/connectors/{connector_id}")
async def remove_connector_membership(automation_id: str, connector_id: str):
    _flag_or_404()
    from app.agent.automations.service import remove_connector
    return await _spec_edit(remove_connector, automation_id,
                            connector_id=connector_id)


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
    # R30 §4.7: ONE connector.state frame + auto-resume of everything
    # blocked on this account (checkpointed runs resumed, reauth-paused
    # automations re-armed, RECONNECTED notes appended). Best-effort —
    # the hook's own updates above never wait on it.
    try:
        from app.agent.automations import connector_state
        async with async_session_maker() as db:
            if body.ok:
                await connector_state.on_connector_connected(
                    db, user_id=_user_id(), connector_id=body.connector_id,
                )
            else:
                await connector_state.on_connector_expired(
                    db, user_id=_user_id(), connector_id=body.connector_id,
                )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] connector state hook skipped: %s", e)
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

    # R29 §3.1: a revoked grant pauses its armed automation — the
    # dispatcher already fails closed, this makes the STATE honest
    # (feeds the `attention: grant_revoked` pill). Best-effort in its
    # own session; the platform's grant row is the record either way.
    if body.status == "revoked":
        automation_id = (body.payload or {}).get("automation_id")
        if automation_id:
            try:
                from app.agent.automations.service import pause_automation
                async with async_session_maker() as db:
                    a = (await db.execute(
                        select(Automation)
                        .where(Automation.id == automation_id)
                        .where(Automation.user_id == _user_id())
                    )).scalar_one_or_none()
                    if a is not None and a.status == "armed":
                        await pause_automation(
                            db, automation_id=automation_id,
                            user_id=_user_id(), reason="grant_revoked",
                        )
            except Exception as e:  # noqa: BLE001 — state stays honest
                logger.warning(
                    "[automations] revoke-pause failed for %s: %s",
                    automation_id, e,
                )
    return {"ok": True}


# ── R30 (CONTRACTS-R30) — summary, threads, stop/resume, workflow ────


@router.get("/summary")
async def automations_summary():
    """§4.1 — the home cards / sidebar / menu-header shape. Served
    beside the legacy list (changing the legacy `status` vocabulary in
    place would break the shipped app; the list retires with B's flip)."""
    _flag_or_404()
    from app.agent.automations.summary import summary_payload
    async with async_session_maker() as db:
        return await summary_payload(db, user_id=_user_id())


@router.get("/{automation_id}/runs")
async def nested_runs(automation_id: str,
                      limit: int = Query(default=50, ge=1, le=200)):
    """Nested alias for the flat /runs (GROUND-TRUTH route-shape gap)."""
    _flag_or_404()
    from app.agent.automations.service import list_runs
    async with async_session_maker() as db:
        return {"runs": await list_runs(
            db, _user_id(), automation_id=automation_id, limit=limit,
        )}


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str):
    """§4.3 — takes effect at the next step boundary; no write may
    start after it. The executor terminalizes and writes the stop note
    with the honest count; a run past its boundaries just refuses."""
    _flag_or_404()
    from app.db.models import BuildJob
    from app.agent.automations.run_v3 import request_stop
    async with async_session_maker() as db:
        job = await db.get(BuildJob, run_id)
        if job is None or job.job_type != "automation_run" \
                or job.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Run not found")
        stamped = await request_stop(db, run_id)
        if not stamped:
            raise HTTPException(status_code=409, detail={
                "code": "not_running",
                "sentence": "That run already finished.",
            })
        return {"stopping": True, "run_id": run_id}


@router.post("/runs/{run_id}/resume")
async def resume_run_route(run_id: str):
    """§4.3 — continue a stopped run from its checkpoint."""
    _flag_or_404()
    from app.db.models import BuildJob
    from app.agent.automations.run_v3 import resume_run
    async with async_session_maker() as db:
        job = await db.get(BuildJob, run_id)
        if job is None or job.job_type != "automation_run" \
                or job.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Run not found")
        result = await resume_run(db, job_id=run_id)
        if not result.get("resumed"):
            raise HTTPException(status_code=409, detail={
                "code": result.get("error") or "not_stopped",
                "sentence": "Only a stopped run can be resumed.",
            })
        return result


@router.post("/{automation_id}/run-now")
async def run_now(automation_id: str):
    """§4.3 — manual fire (kind run_now); honors cadence counters,
    dedupe namespaces and the grant gate; refused while in flight."""
    _flag_or_404()
    import uuid as _uuid
    from sqlalchemy import select as _select
    from app.db.models import BuildJob
    from app.agent.automations.service import (
        _load_owned, AutomationNotFound, parse_spec_live,
    )
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        live = (await db.execute(
            _select(BuildJob)
            .where(BuildJob.source_id == automation_id)
            .where(BuildJob.job_type == "automation_run")
            .where(BuildJob.status.in_(("queued", "running")))
            .limit(1)
        )).scalar_one_or_none()
        if live is not None:
            total = int(live.progress_total or 0)
            step = int(live.progress_step or 0)
            raise HTTPException(status_code=409, detail={
                "code": "already_running",
                "sentence": f"Already running — step {max(step, 1)} of "
                            f"{max(total, 1)}.",
            })
        vspec = await parse_spec_live(automation)
        from app.agent.automations.spec_v2 import ValidatedSpecV2
        if not isinstance(vspec, ValidatedSpecV2):
            raise HTTPException(status_code=409, detail={
                "code": "v1_not_supported",
                "sentence": "This automation predates run-now.",
            })
        from app.agent.automations.executor_v2 import run_schedule_fire_v2
        source = vspec.schedule_source() or (
            vspec.sources[0] if vspec.sources else None
        )
        if source is None:
            raise HTTPException(status_code=409, detail={
                "code": "no_source", "sentence": "Nothing to fire.",
            })
        status = await run_schedule_fire_v2(
            db, automation, vspec, source,
            fire_key=f"manual:{_uuid.uuid4()}", run_kind="run_now",
        )
        return {"fired": True, "status": status}


class ThreadMessageBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@router.post("/{automation_id}/thread/messages")
async def post_thread_message(automation_id: str, body: ThreadMessageBody):
    """§4.10 — persist the user turn on the thread. The conversational
    reply rides the existing WS chat (session-resolved to this
    automation); this REST leg is the durable record + deep-link
    anchor, so the thread stays complete even when the WS is down."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        thread = await ledger.ensure_thread(
            db, user_id=_user_id(), automation_id=automation.id,
        )
        turn = await ledger.append_turn(
            db, user_id=_user_id(), thread=thread, run_id=None,
            kind="user", payload={"text": body.text},
        )
        return {"turn": turn, "thread_id": thread.id}


@router.get("/{automation_id}/workflow")
async def get_workflow(automation_id: str):
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import workflow_payload
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        return await workflow_payload(
            db, automation=automation, user_id=_user_id(),
        )


def _workflow_409(e) -> HTTPException:
    return HTTPException(status_code=409, detail={
        "code": e.code, "sentence": e.sentence, **(e.extra or {}),
    })


class PresetBody(BaseModel):
    preset_id: str = Field(..., max_length=32)


@router.put("/{automation_id}/workflow/schedule")
async def put_workflow_schedule(automation_id: str, body: PresetBody):
    _flag_or_404()
    from app.agent.automations.service import (
        _load_owned, AutomationNotFound, MembershipError,
    )
    from app.agent.automations.workflow import (
        WorkflowError, set_schedule_preset,
    )
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await set_schedule_preset(
                db, automation=automation, user_id=_user_id(),
                preset_id=body.preset_id,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)
        except MembershipError as e:
            raise HTTPException(status_code=409, detail={"code": e.code})


class StepsBody(BaseModel):
    steps: list[dict] = Field(..., min_length=1, max_length=8)


@router.put("/{automation_id}/workflow/steps", status_code=202)
async def put_workflow_steps(automation_id: str, body: StepsBody):
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, set_steps
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await set_steps(
                db, automation=automation, user_id=_user_id(),
                steps=body.steps,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)


class RuleBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=300)


@router.post("/{automation_id}/workflow/rules")
async def post_workflow_rule(automation_id: str, body: RuleBody):
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, add_rule
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await add_rule(db, automation=automation, text=body.text)
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)


@router.put("/{automation_id}/workflow/rules/{rule_id}")
async def put_workflow_rule(automation_id: str, rule_id: str,
                            body: RuleBody):
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, update_rule
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await update_rule(
                db, automation=automation, rule_id=rule_id, text=body.text,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise HTTPException(status_code=404 if e.code == "not_found"
                                else 409,
                                detail={"code": e.code,
                                        "sentence": e.sentence})


@router.delete("/{automation_id}/workflow/rules/{rule_id}")
async def delete_workflow_rule(automation_id: str, rule_id: str):
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import delete_rule
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await delete_rule(
                db, automation=automation, rule_id=rule_id,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")


class PermissionsBody(BaseModel):
    can: list[str] = Field(default_factory=list, max_length=64)
    cant: list[str] = Field(default_factory=list, max_length=64)


@router.put("/{automation_id}/workflow/accounts/{account_id}/permissions")
async def put_account_permissions(automation_id: str, account_id: str,
                                  body: PermissionsBody):
    """§4.4 — the green ✓ commit. 409 hard_rail / last_read /
    needs_consent are the app's three refusal branches."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import (
        WorkflowError, save_permissions,
    )
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await save_permissions(
                db, automation=automation, user_id=_user_id(),
                account_id=account_id, can_ids=body.can, cant_ids=body.cant,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)


class AccountBody(BaseModel):
    account_id: str = Field(..., max_length=64)


@router.post("/{automation_id}/workflow/accounts")
async def post_workflow_account(automation_id: str, body: AccountBody):
    """§4.4 — membership add, READ permissions only; a missing or
    expired account returns the consent URL first."""
    _flag_or_404()
    from app.agent.automations.service import (
        AutomationNotFound, MembershipError, add_connector,
    )
    from app.agent.automations import registry as _reg
    async with async_session_maker() as db:
        state = (await _reg.fetch_connection_state(_user_id())).get(
            body.account_id) or {}
        if not state.get("connected") or state.get("status") != "active":
            raise HTTPException(status_code=409, detail={
                "code": "needs_consent",
                "connector_id": body.account_id,
                "consent_url": f"/api/oauth/connect/{body.account_id}",
            })
        try:
            automation, _spec = await add_connector(
                db, automation_id=automation_id, user_id=_user_id(),
                connector_id=body.account_id,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except MembershipError as e:
            raise HTTPException(status_code=409, detail={"code": e.code})
        from app.agent.automations.workflow import (
            workflow_payload, _edited_note,
        )
        await _edited_note(db, automation)
        return await workflow_payload(
            db, automation=automation, user_id=_user_id(),
        )


@router.delete("/{automation_id}/workflow/accounts/{account_id}")
async def delete_workflow_account(automation_id: str, account_id: str):
    _flag_or_404()
    from app.agent.automations.service import (
        AutomationNotFound, MembershipError, remove_connector,
    )
    async with async_session_maker() as db:
        try:
            automation, _spec = await remove_connector(
                db, automation_id=automation_id, user_id=_user_id(),
                connector_id=account_id,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except MembershipError as e:
            raise HTTPException(status_code=409, detail={"code": e.code})
        from app.agent.automations.workflow import (
            workflow_payload, _edited_note,
        )
        await _edited_note(db, automation)
        return await workflow_payload(
            db, automation=automation, user_id=_user_id(),
        )


class AskBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


@router.post("/{automation_id}/workflow/ask")
async def post_workflow_ask(automation_id: str, body: AskBody):
    """§4.4 — the composer: classify → apply the safe changes →
    {applied, needs} + the thread record."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, composer_ask
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await composer_ask(
                db, automation=automation, user_id=_user_id(),
                text=body.text,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)


class UndoBody(BaseModel):
    undo_token: str = Field(..., max_length=64)


@router.post("/{automation_id}/workflow/undo")
async def post_workflow_undo(automation_id: str, body: UndoBody):
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, composer_undo
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await composer_undo(
                db, automation=automation, user_id=_user_id(),
                token=body.undo_token,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)


# ── R30 §4.6 — from-template, describe ───────────────────────────────


class FromTemplateBody(BaseModel):
    template_id: str = Field(..., max_length=64)


@router.post("/from-template")
async def from_template(body: FromTemplateBody):
    """Create from a catalog card: spec v2 + mode from the template,
    variables defaulted, write steps grant-gated (template_mode drafts
    cannot ARM until consent lands — arm verifies fail-closed). Armed
    immediately when every account is connected and nothing needs a
    grant; otherwise `unarmed` with the consent conversation in the
    setup thread."""
    _flag_or_404()
    from app.agent.automations import registry as _reg
    from app.agent.automations.service import create_automation, arm_automation
    from app.agent.automations.spec import SpecError
    from app.agent.automations.compiler import CompileError

    templates = await _reg.fetch_templates(_user_id())
    template = next(
        (t for t in templates
         if t.get("id") == body.template_id
         or t.get("slug") == body.template_id),
        None,
    )
    if template is None:
        raise HTTPException(status_code=404, detail="No such template")
    spec = dict(template.get("spec") or {})
    variables = dict(spec.get("variables") or {})
    for v in template.get("variables") or []:
        name = v.get("name")
        if name and not variables.get(name):
            variables[name] = v.get("default") or v.get("example") or ""
    spec["variables"] = variables

    async with async_session_maker() as db:
        try:
            automation, vspec = await create_automation(
                db, user_id=_user_id(), spec=spec,
                template_slug=template.get("slug"),
                domain=template.get("category"),
                template_mode=True,
            )
        except SpecError as e:
            raise HTTPException(status_code=422,
                                detail={"errors": e.errors})
        # Arm only when nothing is missing: every member account
        # connected AND no write step without a grant.
        connections = await _reg.fetch_connection_state(_user_id())
        from app.agent.automations.workflow import _member_connectors
        raw = json.loads(automation.spec_json or "{}")
        members = _member_connectors(raw)
        all_connected = all(
            (connections.get(cid) or {}).get("connected")
            and (connections.get(cid) or {}).get("status") == "active"
            for cid in members
        )
        has_ungranted_write = any(
            s.get("tool") and not s.get("grant_id")
            and s.get("grant_target") is not None
            for s in (raw.get("steps") or [])
            if isinstance(s, dict)
        ) or any("{{grant.target.id}}" in json.dumps(s.get("params") or {})
                 for s in (raw.get("steps") or []) if isinstance(s, dict))
        armed = False
        if all_connected and not has_ungranted_write:
            try:
                await arm_automation(
                    db, automation_id=automation.id, user_id=_user_id(),
                )
                armed = True
            except (CompileError, Exception):  # noqa: BLE001
                armed = False

        # Seed the setup thread (§5.3 — C's script, honest fallback).
        from app.agent.automations import ledger as _ledger
        thread = await _ledger.ensure_thread(
            db, user_id=_user_id(), automation_id=automation.id,
        )
        await _ledger.append_turn(
            db, user_id=_user_id(), thread=thread, run_id=None,
            kind="note", payload={"stamp": "added",
                                  "at": datetime.utcnow().isoformat() + "Z"},
        )
        try:
            from app.agent.automations.setup_script import setup_turns
            from app.agent.automations.workflow import (
                mode_of, schedule_block,
            )
            mode, _label = mode_of(automation, raw)
            sched = schedule_block(automation, raw)
            # The close reads best with the REAL next-run label
            # ("tomorrow 8:00" beats "weekdays at 8:00").
            try:
                from app.agent.automations.summary import (
                    _next_run_at, _tz, _when_label,
                )
                first_run = _when_label(
                    await _next_run_at(db, automation.id), _tz(_user_id()),
                )
            except Exception:  # noqa: BLE001
                first_run = sched.get("sentence") or "soon"
            drafts = setup_turns(mode, _label, first_run, [])
            for d in drafts or []:
                kind = d.get("kind")
                if kind in ("agent", "think"):
                    await _ledger.append_turn(
                        db, user_id=_user_id(), thread=thread, run_id=None,
                        kind=kind, payload={"text": d.get("text") or ""},
                    )
                elif kind == "tool":
                    await _ledger.append_turn(
                        db, user_id=_user_id(), thread=thread, run_id=None,
                        kind="tool", payload={
                            "account_id": d.get("account_id") or
                            (members[0] if members else ""),
                            "tool_kind": "read",
                            "action": d.get("action")
                            or "Checked what I can do",
                            "detail": d.get("detail") or "",
                            "ok": True, "ms": 0,
                            "steps": d.get("steps") or [],
                            "items": [], "write_ids": [], "rest": "",
                        },
                    )
        except ImportError:
            await _ledger.append_turn(
                db, user_id=_user_id(), thread=thread, run_id=None,
                kind="agent", payload={"text": (
                    "Here is what I will be able to do — read, and tell "
                    "you. Nothing runs until everything it needs is "
                    "connected."
                )},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[automations] setup script skipped: %s", e)

        from app.agent.automations.service import automation_payload
        payload = automation_payload(automation)
        payload["armed"] = armed
        return {"automation": payload, "thread_id": thread.id}


class DescribeBody(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)


@router.post("/describe")
async def describe(body: DescribeBody):
    """§4.6 — "every Friday, summarise my week": compile a plan from
    one sentence. The compile itself is C's seam; without it the app
    falls back to the setup conversation in chat (503, never a
    fabricated spec)."""
    _flag_or_404()
    try:
        from app.agent.automations.describe_compile import (
            DescribeError, compile_describe,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail={
            "code": "compiler_unavailable",
            "sentence": "Describe it to me in chat and I will set it up "
                        "there.",
        })
    async with async_session_maker() as db:
        try:
            return await compile_describe(
                db, user_id=_user_id(), text=body.text,
            )
        except DescribeError as e:
            raise HTTPException(
                status_code=503 if e.code == "compiler_unavailable" else 422,
                detail={"code": e.code, "sentence": e.sentence},
            )


# ── R30 §4.7 — the connector sheet's card (accounts router) ──────────

accounts_router = APIRouter(prefix="/accounts", tags=["accounts"])


@accounts_router.get("/{account_id}/card")
async def account_card(account_id: str,
                       automation_id: Optional[str] = Query(default=None)):
    """The §3.7 sheet: IT CAN / IT CANNOT / Last use. With
    `automation_id` the lists read the per-automation permissions (the
    same source as the workflow's captions and badges); without it,
    the connector's granted surface. `cant` always ends with the hard
    rails in the connector's own words."""
    _flag_or_404()
    from app.agent.automations import permissions as _perms
    from app.agent.automations import registry as _reg
    from app.services.automation_verbs import display_name

    async with async_session_maker() as db:
        connections = await _reg.fetch_connection_state(_user_id())
        conn = connections.get(account_id) or {}
        status = conn.get("status") or ""
        connected = bool(conn.get("connected"))
        state = "connected" if connected and status == "active" else (
            "expired" if status in ("reauth_required", "provider_down")
            else ("connected" if connected else "missing")
        )
        cat = _perms.catalog_for(account_id)
        if automation_id:
            from app.agent.automations.service import (
                _load_owned, AutomationNotFound,
            )
            try:
                automation = await _load_owned(
                    db, automation_id, _user_id(),
                )
            except AutomationNotFound:
                raise HTTPException(status_code=404, detail="Not found")
            resolved = await _perms.resolve(
                db, automation=automation, account_id=account_id,
            )
            can = [p["label"] for p in resolved["can"]]
            cant = [p["label"] for p in resolved["cant"]]
        else:
            can = [p["label"] for p in cat["reads"]]
            if conn.get("scopes"):
                can += [p["label"] for p in cat["writes"]]
                cant = [p["label"] for p in cat["rails"]]
            else:
                cant = [p["label"] for p in cat["writes"]] \
                    + [p["label"] for p in cat["rails"]]

        # Last use from the newest tool turn on ANY of this user's
        # threads for that account (the ledger is the truth).
        last_use = {"sentence": "No runs yet", "at": None}
        from app.db.models import AutomationTurn as _Turn
        rows = (await db.execute(
            select(_Turn).where(_Turn.kind == "tool")
            .order_by(_Turn.created_at.desc()).limit(60)
        )).scalars().all()
        for r in rows:
            try:
                body = json.loads(r.payload_json)
            except (ValueError, TypeError):
                continue
            if body.get("account_id") != account_id:
                continue
            action = body.get("action") or "Used it"
            detail = body.get("detail") or ""
            last_use = {
                "sentence": (f"{action} · {detail}" if detail
                             else action)[:120],
                "at": r.created_at.isoformat() + "Z",
            }
            break

        return {
            "account_id": account_id,
            "connector_id": account_id,
            "name": display_name(account_id) or account_id,
            "account_label": conn.get("account") or "",
            "state": state,
            "can": can,
            "cant": cant,
            "last_use": last_use,
        }


@accounts_router.post("/{account_id}/reconnect")
async def account_reconnect(account_id: str):
    """§4.7: the consent URL for the in-app flow. The callback emits
    ONE connector.state frame and auto-resumes everything blocked on
    this account (the _connector_connected hook)."""
    _flag_or_404()
    return {
        "consent_url": f"/api/oauth/connect/{account_id}?return_to=app",
        "account_id": account_id,
    }


# ── R30 §4.11a — the routine-migration trigger (ND-6) ────────────────


class MigrateBody(BaseModel):
    # ND-12: intent is SELECTED, never inferred. Without ids only the
    # structurally-safe set (kind == email_briefing) migrates; anything
    # else must be named, because the migrated spec reads Gmail and a
    # keyword scan of a prompt once rewrote a motivational-quote routine
    # into "check Gmail".
    routine_ids: Optional[list[str]] = Field(default=None, max_length=50)


@router.post("/migrate-routines")
async def migrate_routines(body: Optional[MigrateBody] = None):
    """Run the §4.11a migration for THIS tenant. Idempotent (the
    `migrated_to` stamp no-ops a second call), so it is safe to drive
    repeatedly; the live pass calls it explicitly so the before/after
    states are captured rather than raced by a boot hook."""
    _flag_or_404()
    from app.agent.automations.routine_migration import (
        migrate_email_briefings,
    )
    async with async_session_maker() as db:
        return await migrate_email_briefings(
            db, user_id=_user_id(),
            routine_ids=(body.routine_ids if body else None),
        )


@router.post("/migrate-routines/repair")
async def migrate_routines_repair(body: Optional[MigrateBody] = None):
    """Undo migrations that should never have been made (ND-12): the
    automation is deleted and the routine restored to the state
    recorded at migration time.

    Without `routine_ids` this is a DRY RUN — it returns the plan and
    changes nothing, because "would today's rules produce this pair?"
    would also undo every correct, explicitly-selected migration.
    Refuses any automation that has RUN or is ARMED even when named."""
    _flag_or_404()
    from app.agent.automations.routine_migration import (
        repair_mismigrations,
    )
    async with async_session_maker() as db:
        return await repair_mismigrations(
            db, user_id=_user_id(),
            routine_ids=(body.routine_ids if body else None),
        )


@router.get("/migrate-routines/report")
async def migrate_routines_report():
    _flag_or_404()
    from app.agent.automations.routine_migration import migration_report
    async with async_session_maker() as db:
        return await migration_report(db, user_id=_user_id())
