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
                # R38: a spec replacement is an EDIT — divider + frame.
                edited_note=True,
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
    except service.MissingSettings as e:
        # Ordered BEFORE CompileError — it is one, and this refusal
        # owes the user a question rather than a code. A 409 alone
        # would leave the app to invent a sentence for a state only the
        # spec knows, so the question lands in the thread, which is
        # where it is answered.
        async with async_session_maker() as db:
            automation = await _owned_automation_or_404(db, automation_id)
            asked = await _ask_in_thread(db, automation, str(e))
        raise HTTPException(status_code=409, detail={
            "code": e.code, "message": str(e), "sentence": str(e),
            "missing": e.missing, "refusal_turn": asked,
        })
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
    """A rehearsal: the reads run against live data, the writes are
    rendered and reported, nothing is staged and nothing is sent.

    R31-04 recorded why the old implementation was DEV ONLY — it minted
    a real `BuildJob`, posted a run notification, and its "staged"
    write was swept and sent by `outbox.flush_loop` like any other. R38
    replaced the implementation (`service.rehearse`) rather than the
    gate: there is no outbox row now, so there is nothing a loop can
    send. The route keeps its dev flag anyway — the harness is what it
    was built for, and `automations__test_run` is the surface a user
    reaches — so its blast radius is unchanged while its behaviour got
    strictly safer.
    """
    _flag_or_404()
    if not getattr(settings, "automations_dev_tools", False):
        # NOT `404 Feature not available`. That is the exact body
        # `_flag_or_404` uses, and `automations_proxy._translate_agent_dark`
        # turns any 404 carrying it into `503 agent_starting` — so a
        # rehearsal that is merely switched off reached every remote
        # caller as "your agent is still booting". Two shut doors, one
        # sentence, opposite fixes. 403 with a code says which.
        raise HTTPException(status_code=403, detail={
            "code": "rehearsal_disabled",
            "sentence": (
                "Rehearsals are switched off on this tenant. Nothing was "
                "run. Validate the spec by saving it as a draft instead."
            ),
        })
    from app.agent.automations.service import AutomationNotFound, rehearse
    try:
        async with async_session_maker() as db:
            return await rehearse(db, automation_id=automation_id,
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


#: A thread page is at most 200 turns and a run writes once or twice,
#: so this bound is never reached in practice — it is here so a single
#: pathological run can never make the thread response unbounded.
_THREAD_WRITES_CAP = 500


async def _writes_for_turns(
    db, automation_id: str, turns: list[dict],
) -> list[dict]:
    """The write ledger for the runs this page's turns belong to.

    The app resolves each tool turn's `write_ids` against this list to
    say what a run actually sent (`jobSheetSubtitle`). Served empty,
    every group took the "Nothing was sent or changed" branch — under a
    title that read "Posted in #all-toup", about the same run, in the
    same sheet. Scoped to the page rather than to the thread: a
    year-old automation's whole write history is neither wanted here
    nor cheap.

    Field list and ordering are the §4.8 run projection's, so a client
    reading `writes` from a run and from a thread reads one shape.
    """
    from app.db.models import AutomationWrite

    run_ids = sorted({
        str(t.get("run_id")) for t in turns if t.get("run_id")
    })
    if not run_ids:
        return []
    rows = list((await db.execute(
        select(AutomationWrite)
        .where(AutomationWrite.automation_id == automation_id)
        .where(AutomationWrite.run_id.in_(run_ids))
        # Newest first for the CAP (the page's newest turns are the ones
        # a sheet is opened from), re-sorted below to the ascending
        # order the projection serves.
        .order_by(AutomationWrite.created_at.desc())
        .limit(_THREAD_WRITES_CAP)
    )).scalars())
    rows.reverse()
    return [
        {
            "id": w.id, "account_id": w.account_id, "what": w.what,
            "target": w.target, "audience": w.audience,
            "reversible": w.reversible, "undo_ref": w.undo_ref,
        }
        for w in rows
    ]


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
            "writes": await _writes_for_turns(db, automation_id, turns),
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

    # R37: an APPROVED grant finishes what `automations__set_destination`
    # started. The pin stamped grant_id + grant_target onto the step and
    # left the automation a draft; approval is the moment every arm
    # check can finally pass — and before this branch existed, nothing
    # ran it: the user approved the card and the automation sat in
    # draft until they happened to say something in the thread. Best
    # effort, its own session — the grant row is the record either way.
    if body.status == "approved":
        automation_id = (body.payload or {}).get("automation_id")
        if automation_id:
            try:
                from app.agent.automations.service import arm_automation
                from app.agent.automations import ledger as _ledger
                async with async_session_maker() as db:
                    a = (await db.execute(
                        select(Automation)
                        .where(Automation.id == automation_id)
                        .where(Automation.user_id == _user_id())
                        .where(Automation.deleted_at.is_(None))
                    )).scalar_one_or_none()
                    if a is not None and a.status != "armed":
                        from app.agent.automations.service import (
                            MissingSettings,
                        )
                        try:
                            await arm_automation(
                                db, automation_id=automation_id,
                                user_id=_user_id(),
                            )
                        except MissingSettings as ms:
                            # R42 (founder 16): approving the permission
                            # card armed the automation whatever else was
                            # still unanswered, and this was the arm that
                            # reproduced it — the setup questions were
                            # never asked again, so every weekday the run
                            # read GitHub with an empty owner. The grant
                            # is real and stays; what is missing is named.
                            await _ask_in_thread(db, a, str(ms))
                        else:
                            thread = await _ledger.ensure_thread(
                                db, user_id=_user_id(),
                                automation_id=automation_id,
                            )
                            await _ledger.append_turn(
                                db, user_id=_user_id(), thread=thread,
                                run_id=None, kind="agent",
                                payload={"text": (
                                    "You approved it — the permission is "
                                    "in place and this automation is "
                                    "armed. Say run it to watch the "
                                    "first one."
                                )},
                            )
            except Exception as e:  # noqa: BLE001 — approval stands;
                # the next thread turn can still arm by hand.
                logger.info(
                    "[automations] post-approval arm skipped for %s: %s",
                    automation_id, e,
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


async def _ask_in_thread(db, automation, sentence: str) -> bool:
    """Say ONE sentence to the user, as an agent turn in the
    automation's own thread. Returns whether it landed.

    R38 — a run-now refusal is a THREAD TURN, not server silence.
    rec1 f020–f030: "Run it now" answered 409 and wrote nothing, so the
    app alerted AND posted its own local bubble — the same sentence
    twice, beside a phantom run card. The refusal now lands ONCE, as an
    agent turn in the automation's thread (broadcast like any other),
    and the 409 detail carries `refusal_turn: true` so the client knows
    the account already exists and posts nothing of its own.

    R42 gave it a second caller: an arm refused for an unanswered
    setup question asks for the answer here, because the thread is
    where the question is answered — the user replies and the thread
    agent writes the value back through `automations__update`. That is
    the whole mechanism; there is no second one.

    Deduped against the thread's last agent turn: a second press of the
    same dead button re-raises the same 409 but does not stack a second
    identical bubble. Best-effort — a thread that cannot be written
    still gets its honest 409, just without the flag.
    """
    try:
        from sqlalchemy import select as _select
        from app.agent.automations import ledger as _ledger
        from app.db.models import AutomationTurn
        thread = await _ledger.ensure_thread(
            db, user_id=automation.user_id, automation_id=automation.id,
        )
        last = (await db.execute(
            _select(AutomationTurn)
            .where(AutomationTurn.thread_id == thread.id)
            .where(AutomationTurn.kind == "agent")
            .order_by(AutomationTurn.seq.desc())
            .limit(1)
        )).scalar_one_or_none()
        if last is not None:
            try:
                if (json.loads(last.payload_json) or {}).get("text") \
                        == sentence:
                    return True
            except (ValueError, TypeError):
                pass
        await _ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=None,
            kind="agent", payload={"text": sentence},
        )
        return True
    except Exception as e:  # noqa: BLE001 — the caller's answer stands
        logger.warning("[automations] thread turn skipped "
                       "automation=%s: %s", automation.id, e)
        return False


async def _refuse_run_now(db, automation, *, code: str,
                          sentence: str) -> None:
    """Append the refusal turn, then raise the 409 with the flag."""
    detail: dict = {"code": code, "sentence": sentence}
    if await _ask_in_thread(db, automation, sentence):
        detail["refusal_turn"] = True
    raise HTTPException(status_code=409, detail=detail)


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
                await _refuse_run_now(
                    db, automation, code="waiting_on_you",
                    sentence="It is waiting for you to approve a "
                             "change. Decide that first.",
                )
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
        from app.agent.automations.service import MissingSettings
        from app.agent.automations.spec import SpecError as _SpecError
        try:
            vspec = await parse_spec_live(automation)
        except MissingSettings as ms:
            # A run cannot be honest about a setting it does not have:
            # `render_value` would put "" where the owner, the repo or
            # the chat id belongs, the provider would refuse, and the
            # thread would blame the account. Name the setting instead
            # — the same sentence the setup thread opened with, in the
            # place it is answered.
            await _refuse_run_now(
                db, automation, code="needs_setup", sentence=str(ms),
            )
        except _SpecError as se:
            # A spec that no longer parses is a SETUP state, not a
            # server error. Answering 500 here is how the founder's Run
            # button died silently: the app's non-409 path asked the
            # summary what happened, the summary said nothing, and the
            # honest 409 sentence below was never reached.
            # R38: no "in its thread" — the sentence LANDS in the
            # thread now, and pointing at the room you are standing in
            # reads absurd (rec1 f020).
            try:
                await _refuse_run_now(
                    db, automation, code="needs_setup",
                    sentence=(
                        "It is not finished being set up — finish the "
                        "questions and I will run it."
                    ),
                )
            except HTTPException as he:
                raise he from se
        from app.agent.automations.spec_v2 import ValidatedSpecV2
        if not isinstance(vspec, ValidatedSpecV2):
            raise HTTPException(status_code=409, detail={
                "code": "v1_not_supported",
                "sentence": "This automation predates run-now.",
            })
        source = vspec.schedule_source() or (
            vspec.sources[0] if vspec.sources else None
        )
        if source is None:
            raise HTTPException(status_code=409, detail={
                "code": "no_source", "sentence": "Nothing to fire.",
            })
        # R36-2a: an unpinned write cannot produce what the automation
        # promises. Firing anyway sent gmail__create_draft into the
        # dispatcher's grant gate wearing a read's clothes, and the
        # thread reported "Could not reach Gmail" about a permission
        # that was simply never asked for. Refuse with the true need.
        # R39: the predicate is workflow.run_blockers — the ONE shared
        # with the thread agent's grounding and the setup copy, so no
        # surface can promise a run this gate refuses.
        from app.services import automation_verbs as _verbs
        from app.agent.automations import workflow as _wf
        blockers = _wf.run_blockers(_wf._spec_raw(automation))
        if blockers:
            await _refuse_run_now(
                db, automation, code="needs_setup",
                sentence=(f"It is not finished being set up — "
                          f"{blockers[0]['sentence']}."),
            )
        for st in vspec.write_steps:
            # R37: a pinned step whose grant the user has not yet
            # approved used to pass this gate and fire — the reads ran
            # and the write met the dispatcher's fail-closed refusal,
            # so "run it now" mid-approval produced a NEEDS YOU card
            # about a permission the user was already looking at.
            # Refuse with the true state instead. Unreachable platform
            # fires anyway — the dispatcher stays the enforcement.
            from app.agent.automations.registry import fetch_grant
            try:
                grant = await fetch_grant(_user_id(), st.grant_id)
            except Exception:  # noqa: BLE001
                grant = None
            if grant is not None and grant.get("status") != "approved":
                clause = _verbs._WRITE_CLAUSES.get(st.tool) or "write"
                if grant.get("status") == "pending":
                    sentence = (
                        f"It is waiting on your permission to {clause} "
                        f"— approve the permission card first and it "
                        f"runs on its own."
                    )
                else:
                    # Expired, denied, revoked — the card is gone; the
                    # actionable move is a fresh ask, not a hunt for it.
                    sentence = (
                        f"The permission it needs to {clause} was never "
                        f"granted — tell me where that should go and I "
                        f"will ask again."
                    )
                await _refuse_run_now(
                    db, automation, code="needs_setup",
                    sentence=sentence,
                )

    # R36-2: the fire is DETACHED. A run is minutes of reads plus two
    # narration phases; holding the HTTP response open for all of it
    # taught the app's 15 s client to declare "Nothing ran" about a run
    # the server was mid-way through. The route now answers the only
    # question it was asked — did it start — and the run reports itself
    # through the ledger and the activity frames like any other.
    fire_key = f"manual:{_uuid.uuid4()}"
    _detach_run_now(automation_id, _user_id(), source.id, fire_key)
    return {"fired": True, "status": "started"}


# Strong refs: a bare create_task can be garbage-collected mid-run.
_RUN_NOW_TASKS: set = set()


def _detach_run_now(automation_id: str, user_id: str, source_id: str,
                    fire_key: str) -> None:
    import asyncio as _asyncio

    async def _go() -> None:
        from app.agent.automations.service import (
            _load_owned, AutomationNotFound, parse_spec_live,
        )
        from app.agent.automations.executor_v2 import run_schedule_fire_v2
        # R39: the route already answered {"fired": true} and the app has
        # painted its optimistic STARTED — so a death IN HERE, before the
        # run's own ledger exists, used to be perfect silence (the log
        # line is the operator's, not the user's). Anything that stops
        # the fire without a run record now says so in the thread.
        failure: Optional[str] = None
        try:
            async with async_session_maker() as db:
                try:
                    automation = await _load_owned(db, automation_id, user_id)
                except AutomationNotFound:
                    return
                vspec = await parse_spec_live(automation)
                source = next(
                    (s for s in getattr(vspec, "sources", ()) or ()
                     if s.id == source_id),
                    None,
                ) or (vspec.sources[0] if getattr(vspec, "sources", None)
                      else None)
                if source is None:
                    failure = ("It could not start — its trigger is "
                               "missing. Open the workflow and check it.")
                else:
                    status = await run_schedule_fire_v2(
                        db, automation, vspec, source,
                        fire_key=fire_key, run_kind="run_now",
                    )
                    logger.info("[automations] run-now detached finished "
                                "automation=%s status=%s",
                                automation_id, status)
                    if status == "drained":
                        failure = ("It could not start just now — the "
                                   "platform is mid-update. Try again in "
                                   "a minute.")
        except Exception:  # noqa: BLE001 — the run's own record is the
            # user-facing account once a job exists; this line plus the
            # turn below cover the window before it does.
            logger.exception("[automations] run-now detached crashed "
                             "automation=%s", automation_id)
            failure = ("It could not start just now. Nothing ran — "
                       "try it again.")
        if failure:
            try:
                async with async_session_maker() as db:
                    automation = await _load_owned(db, automation_id,
                                                   user_id)
                    await _ask_in_thread(db, automation, failure)
            except Exception:  # noqa: BLE001
                logger.exception("[automations] run-now failure turn "
                                 "skipped automation=%s", automation_id)

    task = _asyncio.create_task(_go())
    _RUN_NOW_TASKS.add(task)
    task.add_done_callback(_RUN_NOW_TASKS.discard)


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
                # agent turn, not the insert. R37: unless the first
                # attempt's answer never happened. The app's Try again
                # now re-sends the SAME client_msg_id precisely so a
                # retry cannot mint a duplicate user turn — which made
                # this branch the place where a lost answer went to die:
                # the turn came back `replayed` and nothing was ever
                # scheduled to answer it (the original answer task lives
                # in process memory, so a restart or a crash between the
                # 202 and the agent turn orphaned the question for good).
                from app.db.models import AutomationTurn
                answered = (await db.execute(
                    select(AutomationTurn.id)
                    .where(AutomationTurn.thread_id == thread.id)
                    .where(AutomationTurn.kind == "agent")
                    .where(AutomationTurn.seq > int(existing.get("seq") or 0))
                    .limit(1)
                )).scalar_one_or_none()
                if answered is None and thread.id not in _ANSWERING_THREADS:
                    _schedule_thread_answer(
                        automation.id, thread.id, body.text,
                    )
                return {"turn": existing, "thread_id": thread.id,
                        "replayed": True,
                        "client_msg_id": body.client_msg_id}

        turn = await ledger.append_turn(
            db, user_id=_user_id(), thread=thread, run_id=None,
            kind="user",
            payload={"text": body.text,
                     "client_msg_id": body.client_msg_id},
        )
        thread_id = thread.id
        automation_id_str = automation.id

    # ── 202 means 202 (round 33, item 7) ──────────────────────────────
    # This route answered the WHOLE agent turn before replying, while
    # `append_turn` had already broadcast the user's own sentence on its
    # way in. So the broadcast always beat the response by the length of
    # a model call, the client had no way to recognise its own turn
    # coming back, and every sentence appeared twice until the answer
    # landed. The answer now runs on its own task with its own session,
    # which is what the `status_code=202` on this route has always
    # claimed and what CONTRACTS-R31 §4.1 documents. The reply reaches
    # the client the way it always did — `automation.activity` →
    # `automation.turn.delta` → `automation.turn` frames — so nothing
    # downstream depended on holding the connection open.
    _schedule_thread_answer(automation_id_str, thread_id, body.text)
    return {"turn": turn, "thread_id": thread_id, "run_id": None,
            # Echoed at the top level as well as inside `turn`, so a
            # client can match its optimistic row without reaching into
            # the turn body.
            "client_msg_id": body.client_msg_id}


# Strong references to in-flight thread answers: `asyncio.create_task`
# keeps only a WEAK one, so without this the loop can collect an answer
# mid-flight and the thread simply never replies.
_PENDING_THREAD_ANSWERS: set = set()

# Threads with an answer task LIVE right now. The replay branch's whole
# licence to re-schedule is "the first answer is LOST" — a restart, a
# crash between the 202 and the agent turn. An answer still RUNNING is
# not lost, and re-scheduling beside it ran two full agent turns for
# one question: two reply bubbles, doubled tool side effects (two
# permission cards from one 'post it to #general'). In-process on
# purpose: after the restart this set is empty, which is exactly the
# case the retry exists for.
_ANSWERING_THREADS: set = set()


def _schedule_thread_answer(
    automation_id: str, thread_id: str, user_text: str,
) -> None:
    """Answer a thread turn off the request's hot path, on its own session."""
    import asyncio
    try:
        task = asyncio.create_task(
            _answer_thread_turn(automation_id, thread_id, user_text),
        )
    except RuntimeError:
        logger.warning(
            "[automations] no event loop to answer thread turn for %s",
            automation_id,
        )
        return
    _PENDING_THREAD_ANSWERS.add(task)
    _ANSWERING_THREADS.add(thread_id)
    def _done(t, _tid=thread_id):
        _PENDING_THREAD_ANSWERS.discard(t)
        _ANSWERING_THREADS.discard(_tid)
    task.add_done_callback(_done)


async def _answer_thread_turn(
    automation_id: str, thread_id: str, user_text: str,
) -> None:
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations import ledger, thread_agent
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
        except AutomationNotFound:
            return
        thread = await ledger.ensure_thread(
            db, user_id=_user_id(), automation_id=automation.id,
        )
        if thread.id != thread_id:
            # The thread was recreated under us; answering into a
            # different one would put the reply where nobody is looking.
            logger.warning(
                "[automations] thread moved while answering (%s → %s)",
                thread_id, thread.id,
            )
        try:
            await thread_agent.answer_in_thread(
                db, automation=automation, thread=thread,
                user_text=user_text,
            )
        except Exception as e:  # noqa: BLE001
            # The user's turn is already durable; an answer that failed
            # must say so in the thread rather than leave the live state
            # spinning (R31-17's silence).
            logger.warning("[automations] thread answer failed: %s", e)
            # Retire the live surface, or the thread shows the agent-state
            # ladder forever behind an answer that is never coming.
            try:
                await ledger.emit_activity(
                    _user_id(), automation_id=automation.id,
                    thread_id=thread.id, run_id=None, phase="done",
                )
            except Exception:  # noqa: BLE001
                pass
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


# R43 §3: a name the catalogue does not hold is a MALFORMED request
# (400) — the app hard-codes those nine strings, so it can only send an
# unknown one by being out of date. A name the catalogue does hold but
# this account cannot use is a 409 about state, and the app draws that
# row with its reason instead of retrying. Collapsing the two into one
# status hid which of the two had happened.
_BAD_REQUEST_CODES = frozenset({
    "unknown_channel", "unknown_format", "unknown_cadence", "unknown_source",
})


def _workflow_error(e) -> HTTPException:
    if e.code in _BAD_REQUEST_CODES:
        return HTTPException(status_code=400, detail={
            "code": e.code, "sentence": e.sentence, **(e.extra or {}),
        })
    return _workflow_409(e)


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
            # R39: the real reason, not a shrug — "That change was
            # refused." hid `no_schedule` behind copy that blamed
            # nobody, while the sheet had just offered a clock for an
            # event trigger (founder P12).
            raise HTTPException(status_code=409, detail={
                "code": "refused", "item": e.code,
                "sentence": {
                    "no_schedule": "This one starts on its own trigger "
                                   "— there is no clock to set.",
                    "bad_schedule": "That time did not make sense — "
                                    "pick it again.",
                    "already_member": "That account is already on it.",
                }.get(e.code, "That change was refused."),
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


# ── R38 §contents — what is inside the account you just tapped ──────

@router.get("/{automation_id}/workflow/accounts/{account_id}/contents")
async def get_account_contents(automation_id: str, account_id: str):
    """The node's own material: recent mail, per-channel messages, the
    tickets due, the open pull requests.

    Always 200 with `ok` — an unreachable agent, a dead credential and
    an account that genuinely holds nothing are three different answers
    and the app must be able to tell them apart. An HTTP error would
    collapse the first two into the app's generic failure banner and
    lose the sentence that says which.
    """
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations import contents as _contents
    from app.agent.automations import registry as _reg
    from app.agent.automations.workflow import (
        _spec_raw, account_sources_of, focus_of,
    )
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        raw = _spec_raw(automation)
        pins = focus_of(raw).get(account_id) or []
        # R43 §5.1 — which places are PICKED, so each group can say
        # whether its checkbox is on. Without it every group serves
        # `selected: false` and the popup's ticks are all empty however
        # many sources the user has chosen.
        picked = account_sources_of(raw).get(account_id)
    # Outside the session on purpose: the readers below are N provider
    # calls at up to 60 s each, and holding a database connection across
    # them is the same pool-exhaustion mistake `save_permissions`
    # documents.
    connection = (await _reg.fetch_connection_state(_user_id())).get(
        account_id)
    return await _contents.account_contents(
        _user_id(), connector_id=account_id, focus=pins,
        connection=connection, sources=picked,
    )


# ── R38 §focus — where this account starts every run ────────────────

class FocusBody(BaseModel):
    kind: str = Field(..., max_length=24)
    id: str = Field(..., min_length=1, max_length=200)
    label: Optional[str] = Field(default=None, max_length=120)
    # R39 — the user's instruction for this place; re-posting the same
    # (kind, id) with a new note updates it.
    note: Optional[str] = Field(default=None, max_length=280)


@router.post("/{automation_id}/workflow/accounts/{account_id}/focus")
async def post_account_focus(automation_id: str, account_id: str,
                             body: FocusBody):
    """Pin one place under an account. Writes the EDITED note like any
    other workflow edit — a pin changes what the automation does."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, add_focus
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await add_focus(
                db, automation=automation, user_id=_user_id(),
                account_id=account_id, kind=body.kind,
                target_id=body.id, label=body.label or "",
                # None ≠ "" here: absent means "no note intent", empty
                # means "clear it" — a bare "+" must never clear.
                note=body.note,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)


@router.delete("/{automation_id}/workflow/accounts/{account_id}/focus")
async def delete_account_focus(
    automation_id: str, account_id: str,
    kind: str = Query(..., max_length=24),
    id: str = Query(..., min_length=1, max_length=200),
):
    """Unpin one place. `kind` + `id` because an id alone is not unique
    across kinds — a Slack channel id and a thread ts can collide in
    shape, and removing the wrong one is silent."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, remove_focus
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await remove_focus(
                db, automation=automation, user_id=_user_id(),
                account_id=account_id, kind=kind, target_id=id,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)


# ── R42 §5.2 / §5.3 — narrow it, and tell me the moment ─────────────

class FiltersBody(BaseModel):
    """The whole set the chips drew, not a toggle — two quick taps
    cannot interleave into a state neither of them meant."""
    filters: list[str] = Field(default_factory=list, max_length=8)


@router.put("/{automation_id}/workflow/accounts/{account_id}/filters")
async def put_account_filters(automation_id: str, account_id: str,
                              body: FiltersBody):
    """The account's read filters. Same EDITED note as every other
    workflow edit: a filter changes what the automation reads."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, set_filters
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await set_filters(
                db, automation=automation, user_id=_user_id(),
                connector_id=account_id, filters=body.filters,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)


class TriggersBody(BaseModel):
    triggers: list[str] = Field(default_factory=list, max_length=4)


@router.put("/{automation_id}/workflow/accounts/{account_id}/triggers")
async def put_account_triggers(automation_id: str, account_id: str,
                               body: TriggersBody):
    """The account's instant triggers — real `trigger.sources` lanes,
    so this goes through the spec validator and the compiler like any
    other trigger change."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, set_triggers
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await set_triggers(
                db, automation=automation, user_id=_user_id(),
                connector_id=account_id, triggers=body.triggers,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_409(e)


# ── R43 §3 — delivery, the link, an account's sources, its ping ──────

class DeliveryBody(BaseModel):
    """Partial: only the keys the sheet sent are written. `channels` is
    the whole set the rows drew, never a toggle — same reason
    `FiltersBody` is."""
    channels: Optional[list[str]] = Field(default=None, max_length=9)
    format: Optional[str] = Field(default=None, max_length=32)
    cadence: Optional[str] = Field(default=None, max_length=32)


@router.put("/{automation_id}/workflow/delivery")
async def put_workflow_delivery(automation_id: str, body: DeliveryBody):
    """Where the brief reaches you. Refuses an unavailable channel
    (409) rather than storing a delivery that silently never
    happens."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import (
        UNSET, WorkflowError, set_delivery,
    )
    sent = body.model_fields_set
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await set_delivery(
                db, automation=automation, user_id=_user_id(),
                # An absent key and an explicit null are different
                # requests — "leave the channels alone" versus "send it
                # nowhere" — and pydantic cannot tell them apart in the
                # value, so the field set is what decides.
                channels=body.channels if "channels" in sent else UNSET,
                format_id=body.format if "format" in sent else UNSET,
                cadence=body.cadence if "cadence" in sent else UNSET,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_error(e)


class LinkBody(BaseModel):
    channel: str = Field(..., min_length=1, max_length=32)


@router.post("/{automation_id}/workflow/delivery/link")
async def post_workflow_delivery_link(automation_id: str, body: LinkBody):
    """Start linking WhatsApp or Telegram. It never SELECTS the
    channel — the app calls PUT /delivery once the link took, so a link
    that did not complete cannot leave a delivery pointing nowhere."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, link_channel
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await link_channel(
                db, automation=automation, user_id=_user_id(),
                channel=body.channel,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_error(e)


class SourcesBody(BaseModel):
    """The whole set the checkbox rows drew. The cap matches
    `spec_v2.MAX_ACCOUNT_SOURCES`; the writer quotes the number back."""
    sources: list[str] = Field(default_factory=list, max_length=10)


@router.put("/{automation_id}/workflow/accounts/{account_id}/sources")
async def put_account_sources(automation_id: str, account_id: str,
                              body: SourcesBody):
    """Which objects inside the account the agent may open. Same EDITED
    note as every other workflow edit: this changes what it reads."""
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import WorkflowError, set_sources
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await set_sources(
                db, automation=automation, user_id=_user_id(),
                connector_id=account_id, sources=body.sources,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_error(e)


class PingBody(BaseModel):
    """§8 — this connector's instant pings. An explicit null CLEARS the
    override (the automation's delivery is used again); an absent key
    leaves that half alone."""
    channel: Optional[str] = Field(default=None, max_length=32)
    format: Optional[str] = Field(default=None, max_length=32)


@router.put("/{automation_id}/workflow/accounts/{account_id}/ping")
async def put_account_ping(automation_id: str, account_id: str,
                           body: PingBody):
    _flag_or_404()
    from app.agent.automations.service import _load_owned, AutomationNotFound
    from app.agent.automations.workflow import UNSET, WorkflowError, set_ping
    sent = body.model_fields_set
    async with async_session_maker() as db:
        try:
            automation = await _load_owned(db, automation_id, _user_id())
            return await set_ping(
                db, automation=automation, user_id=_user_id(),
                connector_id=account_id,
                channel=body.channel if "channel" in sent else UNSET,
                format_id=body.format if "format" in sent else UNSET,
            )
        except AutomationNotFound:
            raise HTTPException(status_code=404, detail="Not found")
        except WorkflowError as e:
            raise _workflow_error(e)


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
    # ── An EXAMPLE is not a default (round 33, item 8) ─────────────────
    # `example` is the placeholder shown in the setup form ("toup-com",
    # "toup-platform"). Substituting it when the user has not answered
    # meant every adopted Morning work brief polled TOUP'S OWN repo and
    # an empty Teams chat id — reads that fail, are swallowed by the
    # step's on_error, and are then published as "No open repo issues."
    # and "No new Teams messages." as if they were facts about the
    # user's work. Only a `default` is an answer; a variable with
    # neither stays unset, and the compile below refuses to arm an
    # automation whose required params are empty.
    unanswered: list[str] = []
    declared_vars: set[str] = set()
    for v in template.get("variables") or []:
        name = v.get("name")
        if not name:
            continue
        declared_vars.add(str(name))
        if variables.get(name):
            continue
        if v.get("default"):
            variables[name] = v["default"]
        else:
            unanswered.append(str(v.get("label") or name))
    # R36-1: two seam bugs made 19 of 26 templates 422 on "Set up".
    # `variables` is a v2-only key — stamping it on a v1 spec fails
    # `unknown_field` for EVERY v1 template; and an unanswered required
    # variable ({{var.github_owner}} with no default) must reach the
    # validator as DECLARED (that is exactly what `template_vars` is
    # for) or every template that needs an answer fails
    # `unknown_variable` before the setup thread can ask for it.
    if spec.get("version") == 2:
        spec["variables"] = variables
    if spec.get("description") is None and template.get("description"):
        # The catalog card's sentence is the automation's own task
        # statement — without it the narrator knows nothing but a name.
        spec["description"] = template.get("description")

    # R38 — the build history. The recorder times the REAL segments of
    # this build; `build_ledger.record` derives every word from the
    # finished automation. Opened here so `total_ms` covers the whole
    # creation, not just the part after the spec was persisted.
    from app.agent.automations import build_ledger as _build
    rec = _build.BuildRecorder("template")

    async with async_session_maker() as db:
        try:
            with rec.phase("trigger"):
                automation, vspec = await create_automation(
                    db, user_id=_user_id(), spec=spec,
                    template_slug=template.get("slug"),
                    domain=template.get("category"),
                    template_mode=True,
                    template_vars=declared_vars,
                )
        except SpecError as e:
            # Never a bare status code on a phone screen: the app shows
            # `sentence` and files `errors` where a developer looks.
            raise HTTPException(status_code=422, detail={
                "errors": e.errors,
                "sentence": "That template is broken on our side — "
                            "nothing was created. It is ours to fix, "
                            "not yours.",
            })
        # Arm only when nothing is missing: every member account
        # connected AND no write step without a grant.
        with rec.phase("output"):
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
        with rec.phase("agent"):
            if all_connected and not has_ungranted_write and not unanswered:
                try:
                    await arm_automation(
                        db, automation_id=automation.id, user_id=_user_id(),
                    )
                    armed = True
                except (CompileError, Exception):  # noqa: BLE001
                    armed = False

        # R38 — one measured segment per account. This is also the ONE
        # permission resolution of the build: the setup script's
        # capability check reads it below instead of calling again, so
        # the history's account phase and the thread's capability turn
        # can never describe two different permission sets.
        from app.agent.automations import permissions as _perms
        resolved_perms: dict = {}
        for cid in members:
            with rec.phase(f"account:{cid}"):
                try:
                    resolved_perms[cid] = await _perms.resolve(
                        db, automation=automation, account_id=cid,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[automations] permission resolve failed for "
                        "%s: %s", cid, e,
                    )

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
            # R35: one check PER member. Flattening every account's
            # lines into one turn stamped `members[0]` opened a
            # six-account brief with "Checked 1 account" and a lone
            # Jira chip.
            # R38 — rec1 f007–f011: every account's row said "posts"
            # while Gmail and Outlook were read-only. The verb is per
            # ACCOUNT: only the one whose step writes wears the
            # write-mode label.
            from app.agent.automations.setup_script import (
                writer_connectors,
            )
            _writer_cids = writer_connectors(raw)
            account_scopes: list = []
            try:
                from app.agent.automations.setup_script import (
                    scope_lines_from,
                )
                for cid in members:
                    # No `· Gmail` suffix per line: each account's turn
                    # carries its own chip, so the name would repeat
                    # what the row already shows.
                    account_scopes.append({
                        "account_id": cid,
                        "writes": cid in _writer_cids,
                        "steps": scope_lines_from(
                            resolved_perms[cid] if cid in resolved_perms
                            else await _perms.resolve(
                                db, automation=automation, account_id=cid,
                            ),
                        ),
                    })
            except Exception:  # noqa: BLE001 — the turn degrades, the
                # thread does not: an empty capability list is worse
                # than a short one, but neither is worth losing setup.
                account_scopes = [
                    {"account_id": cid, "writes": cid in _writer_cids,
                     "steps": []}
                    for cid in members
                ]
            # What it still needs from the user, named, before the
            # capability list — an automation that cannot run without an
            # answer must say which answer, in the thread where the
            # conversation is (round 33, item 8).
            if unanswered:
                # R42: one grammar for this question, shared with the
                # arm and fire refusals — the sentence a user meets at
                # creation and the one they meet three days later when
                # the same answer is still missing must be the same
                # sentence, or the second reads as a new problem.
                from app.agent.automations.service import (
                    missing_settings_sentence,
                )
                await _ledger.append_turn(
                    db, user_id=_user_id(), thread=thread, run_id=None,
                    kind="agent", payload={"text": missing_settings_sentence(
                        [{"label": label} for label in unanswered],
                    )},
                )
            # R36-2a: the promised grant conversation, actually seeded.
            # A template whose write is grant-gated used to open with
            # "reads only — I cannot change anything" and then NOTHING
            # asked for the pin, so the draft the card advertises could
            # never exist and run-now walked into the grant gate.
            if has_ungranted_write:
                try:
                    from app.services.automation_verbs import (
                        _WRITE_CLAUSES,
                    )
                    clauses = [
                        _WRITE_CLAUSES.get(s.get("tool") or "")
                        for s in (raw.get("steps") or [])
                        if isinstance(s, dict)
                        and _WRITE_CLAUSES.get(s.get("tool") or "")
                        and not s.get("grant_id")
                    ]
                    clause = clauses[0] if clauses else "write"
                    await _ledger.append_turn(
                        db, user_id=_user_id(), thread=thread, run_id=None,
                        kind="agent", payload={"text": (
                            f"One thing before it can {clause}: tell me "
                            f"here where that should go, and I will pin "
                            f"it and ask for your permission."
                        )},
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[automations] grant ask skipped: %s", e,
                    )
            from app.agent.automations.workflow import (
                _spec_raw as _sr, run_blockers as _rb,
            )
            try:
                _blocked = bool(_rb(_sr(automation)))
            except Exception:  # noqa: BLE001
                _blocked = False
            drafts = setup_turns(mode, _label, first_run,
                                 accounts=account_scopes,
                                 blocked=_blocked)
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

        await _build.record(db, automation=automation, recorder=rec)

        from app.agent.automations.service import automation_payload
        payload = automation_payload(automation)
        payload["armed"] = armed
        # R39 (founder P20): creation emitted NO summary frame, and this
        # response is the spec payload, not the §4.1 row — so the app's
        # optimistic card kept its empty meta ("" where "First run soon ·
        # 1 account · asks first" belongs) until the next full summary
        # load. emit_updated serves the real row the way every other
        # mutation already does.
        try:
            await _ledger.emit_updated(
                db, _user_id(), automation_id=automation.id,
            )
        except Exception as e:  # noqa: BLE001 — the create stands
            logger.warning("[automations] create emit_updated skipped: %s", e)
        return {"automation": payload, "thread_id": thread.id,
                "build_history": _build.read(automation)}


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


@router.post("/purge-junk-facts")
async def purge_junk_facts_route(body: Optional[CleanupBody] = None):
    """Round 33, item 6 — remove the curator's own failure reports.

    Curator v2 filed what the AGENT could not reach, and the state of
    tickets in other people's systems, as durable facts about the USER.
    The write gate refuses that class now; this removes what it already
    wrote, and projects the removal into the brain so the agent stops
    repeating it. Dry run unless `apply` is true; idempotent.

    The fleet-wide half is `database._alter_statements`, which every
    agent runs at boot — this is the immediate, reportable one.
    """
    _flag_or_404()
    from app.agent.automations.junk_facts import purge
    apply_it = bool(body.apply if body else False)
    async with async_session_maker() as db:
        return await purge(db, user_id=_user_id(), apply=apply_it)


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
