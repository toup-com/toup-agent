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
    """DEV ONLY (R31-04).

    A test run mints a real `BuildJob`, posts a run notification and
    fires a `mission_started` push for an automation the user may never
    have armed. It exists for the harness, and until R31 it was
    reachable three ways: this route, its proxy twin, and — worst — the
    `automations__test_run` TOOL, which the skill's own build order made
    step 7. That is how "Run all of them again" was answered by a
    staged synthetic fire reporting `TEST RUN STAGED` and a status of
    `paused`.

    The tool is gone from the model's array. The route stays for
    `make e2e-automations`, behind the same flag, and 404s in
    production so it cannot be curl'd into a user's thread.
    """
    _flag_or_404()
    if not getattr(settings, "automations_dev_tools", False):
        raise HTTPException(status_code=404, detail="Feature not available")
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
                # R31 §4.4: `body.error` was accepted here and never
                # forwarded, so "reauth_required" (a dead credential)
                # and "provider_down" (a vendor having a bad minute)
                # produced an identical `expired` frame — and only one
                # of them should move the account off `connected`.
                await connector_state.on_connector_expired(
                    db, user_id=_user_id(), connector_id=body.connector_id,
                    error=getattr(body, "error", "") or "",
                )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] connector state hook skipped: %s", e)
    return {"updated": updated}


def _connector_of_grant(automation, grant_id: str) -> Optional[str]:
    """Which connector does this grant back? Read from the spec, not
    from the hook payload — the payload is the caller's word for it."""
    try:
        raw = json.loads(automation.spec_json or "{}")
    except (ValueError, TypeError):
        return None
    if raw.get("version") != 2:
        action = raw.get("action") or {}
        if action.get("grant_id") == grant_id:
            return action.get("connector_id")
        return None
    for step in raw.get("steps") or []:
        if isinstance(step, dict) and step.get("grant_id") == grant_id:
            return step.get("connector_id")
    return None


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
                    # R30 AUDIT-4: pausing made the RUN state honest but
                    # left the PERMISSION state lying — the account sheet
                    # kept showing the write in IT CAN after the platform
                    # had taken the grant away. Demote it here, at the
                    # event, rather than re-deriving on every read.
                    if a is not None:
                        from app.agent.automations import permissions
                        cid = _connector_of_grant(a, body.grant_id) or (
                            (body.payload or {}).get("connector_id"))
                        if cid:
                            await permissions.revoke_writes(
                                db, automation=a, connector_id=cid,
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
        # CONTRACTS-R31 §4.2: `already_running` is refused for a RUNNING
        # run and nothing else.
        #
        # The old predicate was `status IN (queued, running)` on the job
        # row, and the menu's own disable was `run_in_flight is not
        # None` — which the summary populated for `stopped_by_user` too.
        # So a run the user had STOPPED disabled `Run it now` with
        # `Already running — step 0 of 5`: a refusal, naming a step
        # count of zero, for a run that had already ended. Both sides
        # now read one predicate.
        #
        # `waiting_on_user` is refused too, but never as "already
        # running": a parked run is waiting for a decision, and firing a
        # second one beside it double-posts the moment the card is
        # approved. It gets its own code and its own true sentence.
        live = (await db.execute(
            _select(BuildJob)
            .where(BuildJob.source_id == automation_id)
            .where(BuildJob.job_type == "automation_run")
            .where(BuildJob.status.in_(("queued", "running",
                                        "waiting_on_user", "paused")))
            .order_by(BuildJob.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if live is not None:
            from app.agent.automations import ledger as _ledger
            v3 = _ledger.run_v3_status(live)
            if v3 == "waiting_on_user":
                # Not "already running" — a parked run is waiting for a
                # DECISION, and firing a second one beside it
                # double-posts the moment the card is approved.
                raise HTTPException(status_code=409, detail={
                    "code": "waiting_on_you",
                    "sentence": "It is waiting for you to approve a "
                                "change. Decide that first.",
                })
            if v3 == "running":
                total = int(live.progress_total or 0)
                step = int(live.progress_step or 0)
                # R31-30 / §4.4 string table: "Already running" is
                # retired as a SENTENCE (it was shown for a run the user
                # had STOPPED — "Already running — step 0 of 5" — which
                # is the one thing it was not). The CODE keeps its name;
                # the form is `run_now_disabled_sub` in
                # fixtures/automations/reason-strings.json.
                raise HTTPException(status_code=409, detail={
                    "code": "already_running",
                    "sentence": f"It is running now — step "
                                f"{max(step, 1)} of {max(total, 1)}.",
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


class ResumeSourceBody(BaseModel):
    account_id: str = Field(..., max_length=64)


@router.post("/{automation_id}/runs/{run_id}/resume-source")
async def resume_source_route(automation_id: str, run_id: str,
                              body: ResumeSourceBody):
    """§4.4's `retry` fix — re-run ONE account's step and merge it.

    This is what the `Try again` button on a `needs_you` card calls,
    and what the E-1 line's button calls. It is deliberately not
    `run-now`: the other accounts already read successfully, and
    starting over both wastes their work and produces a different
    brief than the one the user is looking at.
    """
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations import executor_v2
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        out = await executor_v2.resume_source(
            db, automation=automation, job_id=run_id,
            account_id=body.account_id,
        )
        if not out.get("resumed"):
            raise HTTPException(status_code=409, detail={
                "code": out.get("reason") or "not_resumable",
                "sentence": "That one cannot be picked up now — run it "
                            "again instead.",
            })
        return out


class ThreadMessageBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    # Replay protection, the same guarantee the chat socket has. Without
    # it a retried POST after a dropped connection is a second user turn
    # AND a second agent run.
    client_msg_id: Optional[str] = Field(default=None, max_length=64)


@router.post("/{automation_id}/thread/messages", status_code=202)
async def post_thread_message(automation_id: str, body: ThreadMessageBody):
    """§4.1 — the thread's own turn: persist, then ANSWER, in the thread.

    Supersedes R30 §4.10's "the conversational reply rides the existing
    WS chat (session-resolved to this automation)". That sentence is
    what made an automation's conversation a day-chat conversation:
    `ws_chat` stamps every user message with today's `day_chat_id`
    before it has looked at `session_id` at all, so a thread question
    was a main-chat row by construction and its reply was an ordinary
    main-chat turn. The founder's 11:17 answer about "everything in all
    channels" is in his main chat for that reason, and `Memory updated ·
    5 facts` is underneath it.

    A question that needs new reading becomes a `question` run in this
    same thread (§4.9) and its id comes back on `run_id`.
    """
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations import ledger, thread_agent
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        thread = await ledger.ensure_thread(
            db, user_id=_user_id(), automation_id=automation.id,
        )

        if body.client_msg_id:
            existing = await _thread_turn_by_client_id(
                db, thread_id=thread.id, client_msg_id=body.client_msg_id,
            )
            if existing is not None:
                # A replay. Return the turn we already have and start no
                # second run — the expensive half of this route is the
                # agent turn, not the insert.
                return {"turn": existing, "thread_id": thread.id,
                        "replayed": True}

        turn = await ledger.append_turn(
            db, user_id=_user_id(), thread=thread, run_id=None,
            kind="user",
            payload={"text": body.text,
                     "client_msg_id": body.client_msg_id},
        )

        run_id = None
        try:
            if thread_agent.needs_fresh_read(body.text):
                run_id = await thread_agent.open_question_run(
                    db, automation=automation, thread=thread,
                    user_text=body.text,
                )
            if run_id is None:
                await thread_agent.answer_in_thread(
                    db, automation=automation, thread=thread,
                    user_text=body.text,
                )
        except Exception as e:  # noqa: BLE001
            # The user's turn is already durable; an answer that failed
            # must say so in the thread rather than leave the live state
            # spinning (R31-17's silence).
            logger.warning("[automations] thread answer failed: %s", e)
            try:
                await ledger.append_turn(
                    db, user_id=_user_id(), thread=thread, run_id=None,
                    kind="agent",
                    payload={"text": (
                        "Something went wrong answering that. Ask me "
                        "again and I will try once more."
                    )},
                )
            except Exception:  # noqa: BLE001
                pass
        return {"turn": turn, "thread_id": thread.id, "run_id": run_id}


async def _thread_turn_by_client_id(
    db, *, thread_id: str, client_msg_id: str,
):
    """The already-persisted turn for this client id, if any."""
    from app.db.models import AutomationTurn
    rows = (await db.execute(
        select(AutomationTurn)
        .where(AutomationTurn.thread_id == thread_id)
        .where(AutomationTurn.kind == "user")
        .order_by(AutomationTurn.seq.desc())
        .limit(40)
    )).scalars().all()
    for row in rows:
        try:
            if json.loads(row.payload_json).get("client_msg_id") \
                    == client_msg_id:
                from app.agent.automations.ledger import _serialize_row
                return _serialize_row(row)
        except (ValueError, TypeError):
            continue
    return None


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


class CustomSchedule(BaseModel):
    """§4.7's `Custom…` row — what the time wheel and the weekday chips
    send. `days` is ISO (1 = Monday, 7 = Sunday); `date` makes it a
    one-time automation and is mutually exclusive with `days`."""
    time: str = Field(..., max_length=5)             # "HH:MM", 24h
    days: list[int] = Field(default_factory=list, max_length=7)
    date: Optional[str] = Field(default=None, max_length=10)
    tz: Optional[str] = Field(default=None, max_length=64)


class PresetBody(BaseModel):
    # Either a preset or a custom body. `preset_id` stays optional
    # rather than being replaced, so the four canvas presets keep the
    # exact wire shape B already ships.
    preset_id: Optional[str] = Field(default=None, max_length=32)
    custom: Optional[CustomSchedule] = None


@router.put("/{automation_id}/workflow/schedule")
async def put_workflow_schedule(automation_id: str, body: PresetBody):
    _flag_or_404()
    from app.agent.automations.service import (
        _load_owned, AutomationNotFound, MembershipError,
    )
    from app.agent.automations.workflow import (
        WorkflowError, set_schedule_preset, set_schedule_custom,
    )
    if not body.preset_id and not body.custom:
        raise HTTPException(status_code=422, detail={
            "code": "no_schedule", "sentence": "Pick a time first.",
        })
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            if body.custom is not None:
                return await set_schedule_custom(
                    db, automation=automation, user_id=_user_id(),
                    custom=body.custom.model_dump(),
                )
            return await set_schedule_preset(
                db, automation=automation, user_id=_user_id(),
                preset_id=body.preset_id or "",
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)
        except MembershipError as e:
            raise HTTPException(status_code=409, detail={"code": e.code})


class RuleEdit(BaseModel):
    id: str = Field(..., max_length=64)
    text: str = Field(..., max_length=300)


class RulesDraft(BaseModel):
    add: list[str] = Field(default_factory=list, max_length=32)
    remove: list[str] = Field(default_factory=list, max_length=32)
    edit: list[RuleEdit] = Field(default_factory=list, max_length=32)


class AccountsDraft(BaseModel):
    add: list[str] = Field(default_factory=list, max_length=16)
    remove: list[str] = Field(default_factory=list, max_length=16)


class PermissionDraft(BaseModel):
    account_id: str = Field(..., max_length=64)
    can: list[str] = Field(default_factory=list, max_length=64)
    cant: list[str] = Field(default_factory=list, max_length=64)


class CommitBody(BaseModel):
    workflow_rev: Optional[int] = None
    schedule: Optional[PresetBody] = None
    permissions: Optional[list[PermissionDraft]] = None
    steps: Optional[list[dict]] = Field(default=None, max_length=8)
    rules: Optional[RulesDraft] = None
    accounts: Optional[AccountsDraft] = None


@router.post("/{automation_id}/workflow/commit")
async def post_workflow_commit(automation_id: str, body: CommitBody):
    """§4.6 — the workflow's green ✓: every draft, one transaction.

    Supersedes R30 §4.4's "never one big PUT" for THIS path only; the
    per-sheet routes stay for `/workflow/ask` and the web. The canvas
    holds local drafts and commits them together, so a refusal leaves
    the workflow exactly as the user last saw it rather than
    half-applied — and one edit costs one round trip on a tenant that
    can boot dark for the better part of a minute.

    `409 stale` re-bases rather than refusing outright: the drafts were
    made against an older revision, so the app re-layers them over the
    workflow returned here and the user sees nothing unless a draft's
    target is gone.
    """
    _flag_or_404()
    from app.agent.automations.service import (
        _load_owned, AutomationNotFound, MembershipError,
    )
    from app.agent.automations.workflow import (
        WorkflowError, commit_workflow, set_steps,
    )
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            out = await commit_workflow(
                db, automation=automation, user_id=_user_id(),
                workflow_rev=body.workflow_rev,
                schedule=body.schedule.model_dump() if body.schedule else None,
                permissions=[p.model_dump() for p in body.permissions]
                if body.permissions else None,
                steps=body.steps,
                rules=body.rules.model_dump() if body.rules else None,
                accounts=body.accounts.model_dump() if body.accounts else None,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            # A refusal applies NOTHING — the nested transaction rolled
            # back before this reached the caller.
            raise HTTPException(status_code=409, detail={
                "code": "refused", "item": e.code, "sentence": e.sentence,
                **(e.extra or {}),
            })
        except MembershipError as e:
            raise HTTPException(status_code=409, detail={
                "code": "refused", "item": e.code,
                "sentence": "That change was refused.",
            })
        if out.get("code") == "stale":
            raise HTTPException(status_code=409, detail=out)

        # Steps land AFTER the transaction, through C's recompiler: it
        # is an LLM call that can take seconds and can legitimately
        # refuse one step, and neither of those belongs inside a lock
        # holding the user's schedule and permissions.
        if body.steps:
            try:
                step_out = await set_steps(
                    db, automation=automation, user_id=_user_id(),
                    steps=body.steps,
                )
                out["steps"] = step_out
                out.pop("pending", None)
            except WorkflowError as e:
                out["steps_refused"] = {
                    "code": e.code, "sentence": e.sentence,
                }
        from app.agent.automations.workflow import workflow_payload
        out["summary"] = None
        try:
            from app.agent.automations.summary import summary_payload
            full = await summary_payload(db, user_id=_user_id())
            for row in (full or {}).get("automations") or []:
                if row.get("id") == automation_id:
                    out["summary"] = row
                    break
        except Exception:  # noqa: BLE001 — the commit already applied
            pass
        return out


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
    from app.agent.automations.workflow import WorkflowError, delete_rule
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await delete_rule(
                db, automation=automation, rule_id=rule_id,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            # `delete_rule` learned to raise `not_found` rather than
            # answer 200 for an edit that never happened — but this route
            # caught only AutomationNotFound, so the honest refusal
            # surfaced as a 500 with no body the app can read. The PUT
            # sibling above already maps it; this now matches.
            raise HTTPException(status_code=404 if e.code == "not_found"
                                else 409,
                                detail={"code": e.code,
                                        "sentence": e.sentence})


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
                    _next_run_at, _tz_name, _when_label, _zone,
                )
                # R31-26: the tz resolver gained a DB fallback and split
                # in two — the cache-then-row lookup is async now,
                # because a summary served by a cold worker had no tz at
                # all and rendered every stamp in UTC. A setup thread's
                # first-run label is exactly the kind of line that gets
                # read once, at the moment the automation is created.
                first_run = _when_label(
                    await _next_run_at(db, automation.id),
                    _zone(await _tz_name(db, _user_id())),
                )
            except Exception:  # noqa: BLE001
                first_run = sched.get("sentence") or "soon"
            # R31-22 / §5.3: the capability check's lines. These were
            # `[]`, so the one turn whose whole job is to say what this
            # automation will and will not be able to do said nothing —
            # on every automation ever created from a template.
            scope_lines: list = []
            try:
                from app.agent.automations import permissions as _perms
                from app.agent.automations.setup_script import (
                    scope_lines_from,
                )
                from app.services import automation_verbs as _v
                for cid in members:
                    scope_lines += scope_lines_from(
                        await _perms.resolve(
                            db, automation=automation, account_id=cid,
                        ),
                        connector_name=(_v.display_name(cid) or cid)
                        if len(members) > 1 else "",
                    )
            except Exception:  # noqa: BLE001 — the turn degrades, the
                # thread does not: an empty capability list is worse
                # than a short one, but neither is worth losing setup.
                scope_lines = []
            drafts = setup_turns(mode, _label, first_run, scope_lines)
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
            # R31-22: the sheet shows this; it never sends the user to
            # the main chat to set an automation up.
            "sentence": "I could not set that up just now. Try again in "
                        "a moment.",
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


async def account_last_use(
    db, *, user_id: str, account_id: str,
    automation_id: Optional[str] = None,
) -> dict:
    """The §4.7 "Last use" line — newest `tool` turn for that account.

    AUDIT-9: this scanned the 60 newest tool turns across EVERY thread
    with no user or automation scope, so the sheet opened inside an
    automation reported a neighbour's activity as this one's — and when
    a busier automation filled those 60 rows, an account that had just
    run read "No runs yet". A route-inline loop also could not be
    tested; it lives here now for both reasons.
    """
    from app.db.models import AutomationTurn as _Turn
    from app.db.models import AutomationThread as _Thread

    q = (
        select(_Turn)
        .join(_Thread, _Thread.id == _Turn.thread_id)
        .where(_Turn.kind == "tool")
        .where(_Thread.user_id == user_id)
    )
    if automation_id:
        q = q.where(_Thread.automation_id == automation_id)
    rows = (await db.execute(
        q.order_by(_Turn.created_at.desc()).limit(200)
    )).scalars().all()
    for r in rows:
        try:
            body = json.loads(r.payload_json)
        except (ValueError, TypeError):
            continue
        if body.get("account_id") != account_id:
            continue
        # R31-25 at the READ boundary — and THIS is the call that put
        # `{need_count}` on the founder's Jira card. These fields are
        # stored verbatim by the turn that wrote them, so making the
        # renderer total cannot reach a row an older build persisted.
        from app.services.automation_verbs import drop_unfilled
        action = drop_unfilled(body.get("action") or "") or "Used it"
        detail = drop_unfilled(body.get("detail") or "")
        return {
            "sentence": (f"{action} · {detail}" if detail else action)[:120],
            "at": r.created_at.isoformat() + "Z",
        }
    return {"sentence": "No runs yet", "at": None}


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
        from app.agent.automations import account_health as _health
        connections = await _reg.fetch_connection_state(_user_id())
        conn = connections.get(account_id) or {}
        status = conn.get("status") or ""
        connected = bool(conn.get("connected"))
        # R31-13 / §4.4: ONE derivation, and the last REAL USE outranks
        # the identity's opinion of the credential. This route used to
        # collapse four server states into `connected|expired|missing`
        # from `conn` alone — which is how the Connectors page read
        # `Connected · 10` while this same account's sheet, two taps
        # away, read `Last use · Could not connect · access expired`.
        health = await _health.state_for(
            db, user_id=_user_id(), account_id=account_id,
            identity_status=status or ("active" if connected else None),
        )
        state = health["account_state"]
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

        last_use = await account_last_use(
            db, user_id=_user_id(), account_id=account_id,
            automation_id=automation_id,
        )

        name = _health.display_of(account_id, display_name(account_id) or "")
        reason = health["reason_code"]
        return {
            "account_id": account_id,
            "connector_id": account_id,
            "name": name,
            "account_label": conn.get("account") or "",
            # The R30 keys, unchanged, so B's card keeps rendering.
            "state": state,
            "can": can,
            "cant": cant,
            "last_use": last_use,
            # R31 §4.4 — additive. Without these the card could show
            # that something was wrong and not what, or what to press.
            "account_state": state,
            "reason_code": reason,
            "fix": health["fix"],
            "fix_label": _health.fix_button(health["fix"], account_id, name),
            "sentence": _health.sentence_for(
                account_state=state, reason_code=reason,
                connector_id=account_id, name=name,
                surface="sheet_subtitle",
            ),
            "checked_at": health["checked_at"],
        }


class ReconnectBody(BaseModel):
    # §4.4's `grant` fix: reconnect asking for the ONE scope that was
    # missing, rather than the whole optional set. A user who is told
    # "it needs more access than you gave it" and then sees a consent
    # screen listing everything has been asked a different question.
    add_scopes: list[str] = Field(default_factory=list, max_length=16)


@accounts_router.post("/{account_id}/reconnect")
async def account_reconnect(account_id: str,
                            body: Optional[ReconnectBody] = None):
    """§4.7/§4.4: the consent URL for the in-app flow.

    The callback emits ONE `connector.state` frame and resumes what was
    blocked on this account (the `_connector_connected` hook). With
    `add_scopes` the round trip asks for exactly those scopes on top of
    what is already granted — the `grant` fix, as distinct from
    `reconnect`, which the two were indistinguishable from until R31
    because `scope_missing` aliased onto `access_expired`.
    """
    _flag_or_404()
    url = f"/api/oauth/connect/{account_id}?return_to=app"
    scopes = [s for s in ((body.add_scopes if body else []) or []) if s]
    if scopes:
        from urllib.parse import quote
        url += "&add_scopes=" + quote(",".join(scopes), safe="")
    return {
        "consent_url": url,
        "account_id": account_id,
        "add_scopes": scopes,
    }


class ProbeBody(BaseModel):
    force: bool = False


@accounts_router.post("/{account_id}/probe")
async def account_probe(account_id: str,
                        body: Optional[ProbeBody] = None):
    """§4.4 — ask the vendor, now, and tell every surface the answer.

    The reason this exists is the GitHub org-approval case. An owner
    approves Toup in GitHub's own UI; nothing about that reaches us,
    and the account sits at `Waiting for the organisation` until the
    next scheduled run happens to try again. The card's `I approved it`
    calls this, the thread calls it on open, and A's scheduler polls it
    every ten minutes while a `needs_you` stands.

    Cached ten minutes by default because these are vendor calls under
    vendor rate limits; `force` bypasses the cache and is what the two
    user-initiated paths send. Emits `connector.state` with the same
    three fields it returns, so a client that is not the one who asked
    still repaints.
    """
    _flag_or_404()
    from app.agent.automations import account_health as _health
    from app.agent.automations import registry as _reg
    force = bool(body.force if body else False)
    async with async_session_maker() as db:
        out = await _health.probe(
            db, user_id=_user_id(), account_id=account_id, force=force,
        )
        return out


class CleanupBody(BaseModel):
    # Dry run by default. A migration that MOVES a user's history should
    # have to be asked twice — the routine-migration repair route sets
    # the same precedent, and for the same reason.
    apply: bool = False


@router.post("/backfill")
async def backfill_route(body: Optional[CleanupBody] = None):
    """R31-18 — the two back-fills: rules, and thread-fact scope.

    `rules_json` had three writers and all three were the user typing
    into the Rules sheet, so an automation whose description says "post
    ONE line, no thread" and whose steps say it again opened its
    Workflow reading `LINES IT WILL NOT CROSS 0`.

    And `curator_v2` files a thread-learned fact as `global` unless the
    model literally says "automation", while the Memory tab reads
    `scope == automation_id` exactly — so an automation that has run
    for days showed five memory groups at `0 things`.

    Dry run unless `apply` is true. Idempotent.
    """
    _flag_or_404()
    from app.agent.automations.backfill_r31 import run_all
    apply_it = bool(body.apply if body else False)
    async with async_session_maker() as db:
        return await run_all(db, user_id=_user_id(), dry_run=not apply_it)


@router.post("/cleanup-day-chat")
async def cleanup_day_chat_route(body: Optional[CleanupBody] = None):
    """§4.1's clean-up — move the leaked rows out of the day chat.

    R31 stops the writers; this moves what they already wrote, so a user
    who opens 26 August after the fix does not still see the defect.
    Identified by PRODUCER (`Message.source` / the job row), never by
    title. Idempotent. Dry run unless `apply` is true.
    """
    _flag_or_404()
    from app.agent.automations.daychat_cleanup import cleanup_day_chat
    apply_it = bool(body.apply if body else False)
    async with async_session_maker() as db:
        return await cleanup_day_chat(
            db, user_id=_user_id(), dry_run=not apply_it,
        )


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
