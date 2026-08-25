"""Run lifecycle v3 — open, stop, terminal; the notification pipeline.

CONTRACTS-R30 §4.3/§4.10. This module is the glue between the v2
executor (mechanical steps) and the v3 ledger (typed turns, threads,
notifications):

  - `open_run` — thread ensure, config stamp, the STARTED note, the
    once-per-run notification (in-chat card + live-activity start),
    the first progress frame.
  - `stop_requested` / `handle_stop` — the §4.3 stop semantics: a stop
    takes effect at the next step boundary; no write may start after
    it; the stop note carries the HONEST writes count from the write
    ledger, never the agent's account of events.
  - `on_terminal` — called from `_finalize_job`'s exactly-once gate:
    flips the run's head note (started → ran/tried), runs the
    completeness invariant + episodes (`ledger.close_ledger`), updates
    the notification (body via C's template seam), emits the terminal
    progress frame.

The notification `body` is written by C's template module
(`app.agent.automations.notification_templates.notification_body`);
this module carries a safe fallback so a missing template never breaks
a terminal. The same string reaches the in-chat card, the push banner
and the live activity — byte-identical (§4.10).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select, update as sa_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Automation, AutomationNotification, AutomationThread, AutomationTurn,
    AutomationWrite, BuildJob,
)
from . import ledger

logger = logging.getLogger(__name__)

# Live-activity mission id namespace (§4.10; B's Stop App Intent keys
# off this exact prefix).
AUTORUN_MISSION_PREFIX = "autorun:"


def _mission_id(run_id: str) -> str:
    return f"{AUTORUN_MISSION_PREFIX}{run_id}"


# ------------------------------------------------------------------ open

async def open_run(
    db: AsyncSession, *, automation: Automation, job: BuildJob,
    kind: str = "scheduled", total_steps: int = 0,
) -> Optional[AutomationThread]:
    """Open the v3 record for a freshly minted run. Best-effort by
    contract — a ledger failure never blocks the run itself."""
    try:
        thread = await ledger.ensure_thread(
            db, user_id=automation.user_id, automation_id=automation.id,
        )
        from .executor_v2 import merge_job_config
        await merge_job_config(
            db, job.id, run_kind=kind, thread_id=thread.id,
        )
        await ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job.id,
            kind="note",
            payload={
                "stamp": "started",
                "at": (job.created_at or datetime.utcnow()).isoformat() + "Z",
            },
        )
        if kind in ("scheduled", "run_now"):
            await _notify_start(
                db, automation=automation, job=job, thread_id=thread.id,
                total_steps=total_steps,
            )
        await ledger.emit_progress(
            automation.user_id, run_id=job.id, automation_id=automation.id,
            step=0, total=total_steps, sentence="Starting",
            fraction=0.0, status="running",
        )
        return thread
    except Exception as e:  # noqa: BLE001
        logger.warning("[run_v3] open_run skipped job=%s: %s", job.id[:8], e)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None


# ------------------------------------------------------------------ stop

async def stop_requested(db: AsyncSession, job_id: str) -> bool:
    """Fresh read of the stop flag — checked at EVERY step boundary and
    by the outbox before executing a staged write (second line)."""
    row = (
        await db.execute(
            select(BuildJob.stop_requested_at).where(BuildJob.id == job_id)
        )
    ).scalar_one_or_none()
    return row is not None


async def request_stop(db: AsyncSession, job_id: str) -> bool:
    """CAS the stop stamp onto a live run. True iff this call stamped."""
    result = await db.execute(
        sa_update(BuildJob)
        .where(BuildJob.id == job_id)
        .where(BuildJob.status.in_(("queued", "running")))
        .where(BuildJob.stop_requested_at.is_(None))
        .values(stop_requested_at=datetime.utcnow())
    )
    await db.commit()
    return result.rowcount == 1


async def writes_count(db: AsyncSession, run_id: str) -> int:
    from sqlalchemy import func
    return (
        await db.execute(
            select(func.count()).select_from(AutomationWrite).where(
                AutomationWrite.run_id == run_id,
            )
        )
    ).scalar_one()


async def handle_stop(
    db: AsyncSession, *, automation: Automation, job: BuildJob,
    step_index: int,
) -> None:
    """Terminalize a stopped run (§4.3): status cancelled / outcome
    stopped, checkpoint set, the stop note with the honest count."""
    from .executor import _finalize_job
    n_writes = await writes_count(db, job.id)
    job_row = await db.get(BuildJob, job.id)
    if job_row is not None:
        job_row.checkpoint_json = json.dumps({"step_index": int(step_index)})
        await db.commit()
    await _finalize_job(
        db, job.id, status="cancelled", outcome="stopped",
        user_message=(
            "You stopped it. Nothing was sent."
            if n_writes == 0 else
            f"You stopped it. {n_writes} change"
            + ("" if n_writes == 1 else "s") + " already made."
        ),
    )
    thread = await ledger.thread_for(db, automation.id)
    if thread is not None:
        await ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job.id,
            kind="note",
            payload={
                "stamp": "stopped",
                "at": datetime.utcnow().isoformat() + "Z",
                "writes_count": n_writes,
            },
        )


# -------------------------------------------------------------- terminal

async def on_terminal(db: AsyncSession, job_id: str) -> None:
    """Post-terminal v3 closing — called from `_finalize_job` AFTER its
    guarded UPDATE won (rowcount == 1). Best-effort throughout; a v3
    failure never un-finalizes a run."""
    try:
        job = await db.get(BuildJob, job_id)
        if job is None or not job.source_id:
            return
        cfg = ledger._cfg_of(job)
        if not cfg.get("thread_id"):
            return  # pre-v3 run — nothing to close
        automation = await db.get(Automation, job.source_id)
        if automation is None:
            return
        v3 = ledger.run_v3_status(job)

        await _flip_head_note(db, job=job, automation=automation, v3=v3)
        if v3 == "skipped":
            await _append_skip_note(db, job=job, automation=automation)
        await ledger.close_ledger(
            db, user_id=automation.user_id, job=job, automation=automation,
        )
        await _notify_terminal(db, automation=automation, job=job, v3=v3)
        total = int(job.progress_total or 0)
        await ledger.emit_progress(
            automation.user_id, run_id=job.id, automation_id=automation.id,
            step=int(job.progress_step or total), total=total,
            sentence="Done" if v3 in ("completed", "partial") else "Stopped",
            fraction=1.0, status=v3,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[run_v3] on_terminal skipped job=%s: %s",
                       job_id[:8], e)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def _flip_head_note(
    db: AsyncSession, *, job: BuildJob, automation: Automation, v3: str,
) -> None:
    """started → ran (completed/partial) / tried (failed). Stopped and
    superseded keep their stop note; waiting keeps STARTED."""
    target = {"completed": "ran", "partial": "ran", "failed": "tried",
              "skipped": "ran"}.get(v3)
    if target is None:
        return
    rows = list(
        (await db.execute(
            select(AutomationTurn)
            .where(AutomationTurn.run_id == job.id)
            .where(AutomationTurn.kind == "note")
            .order_by(AutomationTurn.seq.asc())
        )).scalars()
    )
    if not rows:
        return
    head = rows[0]
    try:
        body = json.loads(head.payload_json)
    except (ValueError, TypeError):
        return
    if body.get("stamp") != "started":
        return
    body["stamp"] = target
    head.payload_json = json.dumps(body, default=str)
    await db.commit()
    await ledger._broadcast(automation.user_id, {
        "type": "automation.turn",
        "thread_id": head.thread_id,
        "run_id": job.id,
        "turn": ledger._serialize_row(head),
    })


async def _append_skip_note(
    db: AsyncSession, *, job: BuildJob, automation: Automation,
) -> None:
    thread = await ledger.thread_for(db, automation.id)
    if thread is None:
        return
    await ledger.append_turn(
        db, user_id=automation.user_id, thread=thread, run_id=job.id,
        kind="note",
        payload={"stamp": "skipped",
                 "at": datetime.utcnow().isoformat() + "Z"},
    )
    # §4.9: the honest agent line beside the SKIPPED note.
    await ledger.append_turn(
        db, user_id=automation.user_id, thread=thread, run_id=job.id,
        kind="agent",
        payload={"text": ("Skipped — you did not approve it in time. "
                          "Nothing was changed.")},
    )


# --------------------------------------------------------- notifications

def _run_summary(
    db_writes: int, *, job: BuildJob, automation: Automation, v3: str,
    needs_count: int = 0, failed_connector_name: Optional[str] = None,
    vocabulary: Optional[str] = None,
) -> dict:
    """The exact fields C's notification templates consume (§5.7)."""
    return {
        "run_kind": ledger.run_kind_of(job),
        "status": v3,
        "vocabulary": vocabulary,
        "needs_count": needs_count,
        "writes_count": db_writes,
        "failed_connector_name": failed_connector_name,
        "automation_name": automation.name,
    }


def _notification_body(kind: str, summary: dict) -> str:
    """C's seam, with an A-owned fallback. Never a finding — the count
    of things that need the user + the invitation to open the run."""
    try:
        from .notification_templates import notification_body
        body = notification_body(kind, summary)
        if isinstance(body, str) and body.strip():
            return body
    except Exception:  # noqa: BLE001 — template module is C's, optional
        pass
    n = int(summary.get("needs_count") or 0)
    status = summary.get("status") or ""
    if kind == "automation_setup":
        return "Continue setting it up ›"
    if kind == "automation_needs_you" or status == "waiting_on_user":
        return "One thing is waiting on you — open the run to decide."
    if status in ("failed",):
        return ("It could not finish. Open the run and I will show you "
                "what stopped it.")
    if n == 1:
        return ("It ran on time. One thing needs you today — open the run "
                "and I will walk you through it there.")
    if n > 1:
        return (f"It ran on time. {n} things need you today — open the run "
                "and I will walk you through them there.")
    return "It ran on time. Nothing needs you — the full run is inside."


async def _accounts_payload(automation: Automation) -> list[dict]:
    try:
        spec = json.loads(automation.spec_json or "{}")
    except (ValueError, TypeError):
        spec = {}
    from app.services.automation_verbs import display_name
    ids: list[str] = []
    if spec.get("version") == 2:
        for s in (spec.get("trigger") or {}).get("sources") or []:
            cid = s.get("connector_id")
            if cid and cid not in ids:
                ids.append(cid)
        for s in spec.get("steps") or []:
            cid = s.get("connector_id")
            if cid and cid not in ids:
                ids.append(cid)
    elif automation.connector_id:
        ids.append(automation.connector_id)
    return [
        {"account_id": cid, "connector_id": cid,
         "name": display_name(cid) or cid, "account_label": ""}
        for cid in ids
    ]


async def _mint_notification(
    db: AsyncSession, *, automation: Automation, job: BuildJob,
    kind: str, thread_id: Optional[str], status: str,
    sentence: Optional[str] = None, fraction: Optional[float] = None,
    body: Optional[str] = None, turn_id: Optional[str] = None,
) -> Optional[AutomationNotification]:
    # Snapshot scalars FIRST: the dedupe rollback below expires every
    # object in this session, and touching an expired attribute inside
    # the query build is a sync lazy-load (the MissingGreenlet class).
    run_id = str(job.id)
    user_id, automation_id = automation.user_id, automation.id
    title = automation.name[:200]
    accounts_json = json.dumps(await _accounts_payload(automation))
    row = AutomationNotification(
        user_id=user_id, automation_id=automation_id,
        run_id=run_id, thread_id=thread_id, turn_id=turn_id, kind=kind,
        title=title,
        accounts_json=accounts_json,
        sentence=(sentence or "")[:300] or None,
        fraction=int(round((fraction or 0.0) * 100)),
        status=status, body=(body or "")[:500] or None,
    )
    db.add(row)
    try:
        await db.flush()
        await db.commit()
        return row
    except IntegrityError:
        await db.rollback()
        return (
            await db.execute(
                select(AutomationNotification).where(
                    AutomationNotification.run_id == run_id,
                    AutomationNotification.kind == kind,
                )
            )
        ).scalar_one_or_none()


def _notification_payload(row: AutomationNotification) -> dict:
    try:
        accounts = json.loads(row.accounts_json or "[]")
    except (ValueError, TypeError):
        accounts = []
    return {
        "id": row.id, "kind": row.kind, "automation_id": row.automation_id,
        "run_id": row.run_id, "thread_id": row.thread_id,
        "turn_id": row.turn_id, "title": row.title, "accounts": accounts,
        "sentence": row.sentence,
        "fraction": (row.fraction or 0) / 100.0,
        "status": row.status, "body": row.body,
    }


async def _write_chat_card(
    db: AsyncSession, *, automation: Automation,
    row: AutomationNotification,
) -> None:
    """The in-chat notification card — ONE day-chat message per
    notification, metadata key `automation_notification`, updated in
    place on status flips (§4.10). The ONLY automation presence in the
    main chat (D-05)."""
    from .cards import write_card_message, update_card_message, broadcast_card
    payload = _notification_payload(row)
    if row.message_id:
        await update_card_message(
            db, message_id=row.message_id,
            metadata_key="automation_notification", payload=payload,
        )
    else:
        msg_id, _day = await write_card_message(
            db, user_id=automation.user_id,
            content="",
            metadata_key="automation_notification", payload=payload,
            title=automation.name[:80],
        )
        if msg_id:
            row.message_id = msg_id
            await db.commit()
            payload = _notification_payload(row)
    await broadcast_card(
        automation.user_id, "automation_notification",
        {**payload, "message_id": row.message_id},
    )


async def _notify_start(
    db: AsyncSession, *, automation: Automation, job: BuildJob,
    thread_id: str, total_steps: int,
) -> None:
    """Run start: mint the notification, post the in-chat card, start
    the live activity (`mission_started`, mission_id autorun:{run_id})."""
    row = await _mint_notification(
        db, automation=automation, job=job, kind="automation_run",
        thread_id=thread_id, status="running",
        sentence="Starting", fraction=0.0,
    )
    if row is None:
        return
    await _write_chat_card(db, automation=automation, row=row)
    try:
        from app.services.agent_notify_client import notify
        await notify(
            event_kind="mission_started",
            title=automation.name[:200],
            body="Working now",
            priority="default",
            dedup_key=f"autorun:{job.id}:start",
            data={
                "kind": "automation", "route": "automation",
                "mission_id": _mission_id(job.id),
                "automation_id": automation.id, "run_id": job.id,
                "thread_id": thread_id,
                "steps_total": total_steps,
                "no_agent_fallback": True, "silent": True,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[run_v3] LA start skipped: %s", e)


async def notify_progress(
    db: AsyncSession, *, automation: Automation, job: BuildJob,
    step: int, total: int, sentence: str, fraction: float,
) -> None:
    """Mirror a §4.3 progress frame into the live-activity lane
    (event_kind `progress` is LA-only by dispatcher invariant)."""
    try:
        from app.services.agent_notify_client import notify
        await notify(
            event_kind="progress",
            title=automation.name[:200],
            body=sentence[:200],
            priority="low",
            data={
                "kind": "automation", "route": "automation",
                "mission_id": _mission_id(job.id),
                "automation_id": automation.id, "run_id": job.id,
                "step_name": sentence[:80],
                "steps_done": step, "steps_total": total,
                "progress": int(round(fraction * 100)),
                "update_only": True, "no_agent_fallback": True,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[run_v3] LA progress skipped: %s", e)


async def _notify_terminal(
    db: AsyncSession, *, automation: Automation, job: BuildJob, v3: str,
) -> None:
    """Terminal: fill the run notification's body (same string on card,
    push and LA), flip the card, push completed-only, end the LA."""
    cfg = ledger._cfg_of(job)
    thread_id = cfg.get("thread_id")
    n_writes = await writes_count(db, job.id)
    needs, vocab = await _needs_count(db, job)
    summary = _run_summary(
        n_writes, job=job, automation=automation, v3=v3,
        needs_count=needs, vocabulary=vocab,
        failed_connector_name=await _failed_connector(db, job),
    )
    kind = "automation_needs_you" if v3 == "waiting_on_user" \
        else "automation_run"
    body = _notification_body(kind, summary)

    row = (
        await db.execute(
            select(AutomationNotification).where(
                AutomationNotification.run_id == job.id,
                AutomationNotification.kind == "automation_run",
            )
        )
    ).scalar_one_or_none()
    if row is None:
        row = await _mint_notification(
            db, automation=automation, job=job, kind="automation_run",
            thread_id=thread_id, status=v3, body=body,
        )
        if row is None:
            return
    row.status = v3
    row.body = body[:500]
    row.fraction = 100
    row.sentence = None
    await db.commit()
    await _write_chat_card(db, automation=automation, row=row)

    if v3 == "waiting_on_user":
        needs_row = await _mint_notification(
            db, automation=automation, job=job,
            kind="automation_needs_you", thread_id=thread_id,
            status=v3, body=body,
        )
        if needs_row is not None and needs_row.body != body:
            needs_row.body = body
            await db.commit()

    # Push + LA end — completed-only for the OS banner (the R28
    # invariant), the LA end for every terminal so no card goes stale.
    try:
        from app.services.agent_notify_client import notify
        event_kind = "mission_completed" if v3 in (
            "completed", "partial",
        ) else "mission_failed"
        push_worthy = v3 in ("completed", "partial", "waiting_on_user")
        if v3 == "waiting_on_user":
            event_kind = "needs_approval"
        await notify(
            event_kind=event_kind,
            title=automation.name[:200],
            body=body[:500],
            priority="default",
            dedup_key=f"autorun:{job.id}:done",
            data={
                "kind": "automation", "route": "automation",
                "mission_id": _mission_id(job.id),
                "automation_id": automation.id, "run_id": job.id,
                "thread_id": thread_id or "",
                "progress": 100,
                "no_agent_fallback": True,
                **({} if push_worthy else {"silent": True}),
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[run_v3] terminal notify skipped: %s", e)


async def _needs_count(
    db: AsyncSession, job: BuildJob,
) -> tuple[int, Optional[str]]:
    """Tier-1/2 row count from the run's result turn (brief), or the
    pending waiting turns (changes/confirm)."""
    turns = await ledger.run_turns(db, run_id=job.id)
    vocab = None
    needs = 0
    for t in turns:
        if t["kind"] == "result":
            vocab = t.get("vocabulary")
            for g in (t.get("groups") or [])[:2]:
                needs += len(g.get("rows") or [])
        elif t["kind"] in ("draft", "waiting"):
            needs += 1
    return needs, vocab


async def _failed_connector(
    db: AsyncSession, job: BuildJob,
) -> Optional[str]:
    from app.services.automation_verbs import display_name
    turns = await ledger.run_turns(db, run_id=job.id)
    for t in turns:
        if t["kind"] == "tool" and not t.get("ok", True):
            return display_name(t.get("account_id"))
    return None


# ----------------------------------------------------- setup notification

async def notify_setup(
    db: AsyncSession, *, automation: Automation, thread_id: str,
) -> None:
    """The `automation_setup` card — posted ONCE when a setup request
    arrives through the main chat (§4.10). `run_id` slot carries the
    automation id (there is no run yet); dedupe holds per automation."""
    job_like = type("J", (), {"id": f"setup:{automation.id}"})()
    row = await _mint_notification(
        db, automation=automation, job=job_like, kind="automation_setup",
        thread_id=thread_id, status="setup",
        body=_notification_body("automation_setup", {}),
        sentence=f"Setting up: {automation.name}"[:300],
    )
    if row is not None:
        await _write_chat_card(db, automation=automation, row=row)


# ----------------------------------------------------------------- resume

async def resume_run(db: AsyncSession, *, job_id: str) -> dict:
    """Resume a stopped run from its checkpoint (§4.3).

    Reopens the SAME job row (the wire keeps one run id), clears the
    stop flag, re-runs the pipeline with `resume=True` — reads
    re-execute (the honest answer when the dedupe window moved),
    surviving staged writes are reused, and the stop note stays in the
    thread as part of the record. Returns {"resumed": bool, "status"}.
    """
    from sqlalchemy import update as sa_update
    job = await db.get(BuildJob, job_id)
    if job is None:
        return {"resumed": False, "error": "not_found"}
    if ledger.run_v3_status(job) != "stopped_by_user":
        return {"resumed": False, "error": "not_stopped",
                "status": ledger.run_v3_status(job)}
    automation = await db.get(Automation, job.source_id or "")
    if automation is None:
        return {"resumed": False, "error": "automation_gone"}

    result = await db.execute(
        sa_update(BuildJob)
        .where(BuildJob.id == job_id)
        .where(BuildJob.status == "cancelled")
        .values(status="running", outcome=None, completed_at=None,
                stop_requested_at=None)
    )
    await db.commit()
    if (result.rowcount or 0) != 1:
        return {"resumed": False, "error": "race_lost"}

    checkpoint = ledger.checkpoint_of(job) or {"step_index": 0}
    thread = await ledger.thread_for(db, automation.id)
    if thread is not None:
        k = int(checkpoint.get("step_index") or 0)
        await ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job.id,
            kind="agent",
            payload={"text": (
                f"Picking up from step {k + 1}. Anything that changed "
                "while it was stopped gets re-read, not skipped."
            )},
        )
    await ledger.emit_progress(
        automation.user_id, run_id=job.id, automation_id=automation.id,
        step=int(checkpoint.get("step_index") or 0),
        total=int(job.progress_total or 0),
        sentence="Picking up where it stopped",
        fraction=0.0, status="running",
    )

    import asyncio
    import json as _json
    from app.db.models import AUTOMATION_RUN_CAP_S, AutomationEvent
    from .spec_v2 import ValidatedSpecV2
    from . import executor_v2 as _ex

    try:
        from .service import parse_spec_live
        vspec = await parse_spec_live(automation)
        if not isinstance(vspec, ValidatedSpecV2):
            return {"resumed": False, "error": "v1_not_supported"}
    except Exception as e:  # noqa: BLE001
        return {"resumed": False, "error": f"spec: {e}"}

    cfg = ledger._cfg_of(job)
    payload: dict = {}
    source = None
    event_id = cfg.get("automation_event_id")
    if event_id:
        ev = await db.get(AutomationEvent, event_id)
        if ev is not None and ev.payload_json:
            try:
                payload = _json.loads(ev.payload_json)
            except (ValueError, TypeError):
                payload = {}
            source = vspec.source_by_id(str(payload.get("_source") or "")) \
                or (vspec.sources[0] if vspec.sources else None)
    if source is None:
        source = vspec.schedule_source() or (
            vspec.sources[0] if vspec.sources else None
        )
    idem = job.idempotency_key or f"resume:{job.id}"

    try:
        status = await asyncio.wait_for(
            _ex._run_steps(
                db, automation, vspec, job.id, payload, source,
                idem_prefix=idem, resume=True,
            ),
            timeout=AUTOMATION_RUN_CAP_S,
        )
    except asyncio.TimeoutError:
        from .executor import _finalize_job
        await _finalize_job(
            db, job.id, status="failed", outcome="run_cap",
            error_class="timeout",
            user_message="The resumed run exceeded the 3-minute cap and "
                         "was stopped.",
        )
        return {"resumed": True, "status": "failed"}
    return {"resumed": True, "status": status}


async def supersede_stopped_run(
    db: AsyncSession, *, automation_id: str,
) -> None:
    """A stopped run is superseded when the next scheduled fire arrives
    (§4.3): outcome flips stopped→superseded, no new turn (its stop
    note was already written)."""
    from sqlalchemy import update as sa_update
    await db.execute(
        sa_update(BuildJob)
        .where(BuildJob.source_id == automation_id)
        .where(BuildJob.job_type == "automation_run")
        .where(BuildJob.status == "cancelled")
        .where(BuildJob.outcome == "stopped")
        .values(outcome="superseded")
    )
    await db.commit()
