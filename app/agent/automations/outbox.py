"""Write-outbox flush — the ONLY code that sends an automation write.

Every row waits out its undo window (`execute_after`), is claimed with
a single guarded UPDATE (staged→executing — the double-fire defence,
same idiom as connector_pending_actions), and goes to the provider via
the platform's grant-gated dispatch RPC. Retryable failures back off
(10/30/90s, the routines ladder) and hard-fail after 3 attempts.

Two entry points:
  - `flush_row_when_due(db, id)` — inline from the run, sub-second
    after the window closes (the happy path).
  - `flush_loop()` — the background guarantee: sweeps rows the inline
    path lost to a restart, and retries backoffs. Started by
    agent_main only when `settings.automations_enabled` is true.

Undo: `undo_row` flips staged→undone with the same guarded UPDATE —
once a row is claimed, undo loses, which is exactly what a 6-second
window means.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, update as sa_update

from app.db.database import async_session_maker
from app.db.models import Automation, AutomationOutbox, BuildJob
from . import registry as reg
from .draft_card import DRAFT_TOOLS as _DRAFT_TOOLS

logger = logging.getLogger(__name__)

_RETRY_DELAYS_S = (10, 30, 90)
_MAX_ATTEMPTS = 3
_LOOP_INTERVAL_S = 5.0
_BOOT_DELAY_S = 20.0


async def _claim(db, outbox_id: str) -> bool:
    """staged→executing, one statement. rowcount!=1 ⇒ someone else
    (another flush, an undo, a cancel) owns the row."""
    res = await db.execute(
        sa_update(AutomationOutbox)
        .where(AutomationOutbox.id == outbox_id)
        .where(AutomationOutbox.status == "staged")
        .where(AutomationOutbox.execute_after <= datetime.utcnow())
        .values(status="executing")
    )
    await db.commit()
    return (res.rowcount or 0) == 1


async def flush_row_when_due(db, outbox_id: str) -> Optional[str]:
    """Inline flush: sleep out the remainder of the undo window, then
    claim and send. Returns the terminal outbox status, or None when
    the claim was lost."""
    row = await db.get(AutomationOutbox, outbox_id)
    if row is None:
        return None
    delay = (row.execute_after - datetime.utcnow()).total_seconds()
    if delay > 0:
        await asyncio.sleep(delay)
    if not await _claim(db, outbox_id):
        return None
    return await _execute_claimed(db, outbox_id)


async def _execute_claimed(db, outbox_id: str) -> str:
    """The send. The row is OURS (status=executing)."""
    row = await db.get(AutomationOutbox, outbox_id)
    if row is None:
        return "failed"

    # R30 §4.3 second line of defence: no write may start after a stop.
    # The executor checks at step boundaries; a staged row that outlived
    # a stop (undo-window sleep, backoff retry, loop sweep) is refused
    # HERE, where the send actually happens.
    if row.job_id:
        try:
            from .run_v3 import stop_requested
            if await stop_requested(db, row.job_id):
                row.status = "cancelled"
                row.last_error = "run stopped by the user before the write"
                await db.commit()
                logger.info("[automations] outbox %s refused: run %s stopped",
                            row.id, row.job_id)
                # ND-7a: the refusal must also TERMINALIZE the run — a
                # cancelled row with a still-running job was the live
                # wedge (stop during the write step never resolved back
                # to the executor). handle_stop's guarded finalize makes
                # this exactly-once even when the executor also lands.
                try:
                    from .run_v3 import handle_stop
                    from app.agent import job_steps as _js
                    automation = await db.get(Automation, row.automation_id)
                    job = await db.get(BuildJob, row.job_id)
                    if automation is not None and job is not None:
                        done = sum(
                            1 for st in _js.parse_steps(job.steps_json)
                            if st.get("status") in ("done", "completed")
                        )
                        await handle_stop(db, automation=automation,
                                          job=job, step_index=done)
                except Exception as e2:  # noqa: BLE001 — sweep backstops
                    logger.warning(
                        "[automations] stop terminalize failed run=%s: %s",
                        row.job_id, e2,
                    )
                return row.status
        except Exception as e:  # noqa: BLE001
            # D's design-review note (R30): when the stop CHECK itself
            # errors we cannot know the user's intent — for an
            # unattended writer the safe direction is retry-later,
            # never send-anyway. Return the claim and back off.
            logger.warning("[automations] outbox %s stop check errored — "
                           "deferring the send: %s", row.id, e)
            try:
                await db.rollback()
            except Exception:  # noqa: BLE001
                pass
            row2 = await db.get(AutomationOutbox, outbox_id)
            if row2 is not None:
                row2.status = "staged"
                row2.next_attempt_at = datetime.utcnow() + timedelta(
                    seconds=_RETRY_DELAYS_S[0])
                row2.execute_after = row2.next_attempt_at
                await db.commit()
                return row2.status
            return "failed"

    try:
        payload = json.loads(row.payload_json)
    except (ValueError, TypeError):
        payload = {}

    row.attempts = (row.attempts or 0) + 1
    await db.commit()

    result = await reg.dispatch_via_platform(
        row.user_id,
        connector_id=row.connector_id,
        tool_name=row.tool_name,
        tool_input=payload,
        grant_id=row.grant_id,
        automation_id=row.automation_id,
        request_id=f"outbox:{row.id}",
    )
    kind = result.get("kind")
    now = datetime.utcnow()

    if kind == "ok":
        row.status = "executed"
        row.executed_at = now
        row.result_json = json.dumps(result, default=str)[:8000]
        # R30 §4.8: the honest write ledger — appended in the SAME
        # transaction as the executed flip, from the display form
        # snapshotted at staging. The job-sheet grammar, the `changes`
        # vocabulary and stop notes read ONLY these rows.
        write_row = _write_ledger_row(row)
        if write_row is not None:
            db.add(write_row)
        await db.commit()
        await _mark_write_step(db, row, ok=True)
        await _finalize_run(db, row, status="completed", outcome="sent")
        # R31-43: a write that landed is not a clean run if a source was
        # lost getting there. `steps_partial` is the same flag the
        # aggregate finalizer reads to report `partial`; reading it here
        # is what stops a partial run clearing the 3-strike streak it
        # should be building.
        _job = await db.get(BuildJob, row.job_id) if row.job_id else None
        _clean = not bool(((_job.config_json or {}) if _job else {})
                          .get("steps_partial"))
        await _record_health(db, row.automation_id, ok=True, error=None,
                             clean=_clean)
        await _append_write_turn(db, row, write_row)
        if row.tool_name in _DRAFT_TOOLS:
            # The proactive-draft surface (R29): a session card that
            # names the draft and tells the "nothing was sent" truth.
            from .draft_card import write_draft_card
            await write_draft_card(db, row, result)
        return row.status

    if kind == "confirmation_required":
        # Confirm-mode grant: the dispatcher staged the per-fire card;
        # the pending action owns the send from here. The row is
        # terminal — result_json says who owns it — and the run parks
        # exactly the way an elevated chat tool parks.
        action_id = result.get("action_id")
        row.status = "executed"
        row.executed_at = now
        row.result_json = json.dumps(result, default=str)[:8000]
        await db.commit()
        await _park_run_on_card(db, row, result)
        await _notify_needs_approval(row, action_id)
        return row.status

    retryable = bool(result.get("retryable")) or kind in (
        "rate_limited", "provider_down",
    )
    if retryable and row.attempts < _MAX_ATTEMPTS:
        delay = _RETRY_DELAYS_S[min(row.attempts - 1, len(_RETRY_DELAYS_S) - 1)]
        row.status = "staged"
        row.next_attempt_at = now + timedelta(seconds=delay)
        row.execute_after = row.next_attempt_at
        row.last_error = str(result.get("message") or kind)[:2000]
        await db.commit()
        logger.info("[automations] outbox retry %s attempt=%s in %ss",
                    row.id, row.attempts, delay)
        return row.status

    row.status = "failed"
    row.last_error = str(result.get("message") or kind)[:2000]
    row.result_json = json.dumps(result, default=str)[:8000]
    await db.commit()
    await _mark_write_step(db, row, ok=False)
    await _finalize_run(
        db, row, status="failed", outcome="write_failed",
        error_class="tool_error",
        user_message=(str(result.get("message") or "The write failed."))[:300],
    )
    await _record_health(db, row.automation_id, ok=False, error=row.last_error)
    return row.status


async def undo_row(db, outbox_id: str, user_id: str) -> bool:
    """User hit undo inside the window. Guarded UPDATE; a claimed row
    cannot be undone."""
    res = await db.execute(
        sa_update(AutomationOutbox)
        .where(AutomationOutbox.id == outbox_id)
        .where(AutomationOutbox.user_id == user_id)
        .where(AutomationOutbox.status == "staged")
        .values(status="undone")
    )
    await db.commit()
    if (res.rowcount or 0) != 1:
        return False
    row = await db.get(AutomationOutbox, outbox_id)
    if row is not None:
        await _finalize_run(db, row, status="cancelled", outcome="undone")
    return True


async def flush_loop() -> None:
    """Background guarantee — restart recovery + retry backoffs."""
    await asyncio.sleep(_BOOT_DELAY_S)
    logger.info("[automations] outbox flush loop started")
    while True:
        try:
            async with async_session_maker() as db:
                due = (await db.execute(
                    select(AutomationOutbox.id)
                    .where(AutomationOutbox.status == "staged")
                    .where(AutomationOutbox.execute_after <= datetime.utcnow())
                    .limit(20)
                )).scalars().all()
                for oid in due:
                    if await _claim(db, oid):
                        await _execute_claimed(db, oid)
                # Executing rows older than the run cap lost their
                # flusher to a restart mid-send. The provider MAY have
                # received the call — mark lost, never blind-retry a
                # write (the idempotency key already stopped a re-stage;
                # re-SENDING is what must not happen).
                stale_cutoff = datetime.utcnow() - timedelta(minutes=10)
                stale = (await db.execute(
                    select(AutomationOutbox)
                    .where(AutomationOutbox.status == "executing")
                    .where(AutomationOutbox.execute_after <= stale_cutoff)
                )).scalars().all()
                for row in stale:
                    row.status = "failed"
                    row.last_error = "lost mid-send (agent restart); not retried"
                    await db.commit()
                    await _finalize_run(
                        db, row, status="failed", outcome="lost",
                        error_class="interrupted",
                        user_message="This write was interrupted mid-send "
                                     "and was not retried.",
                    )
                    await _record_health(db, row.automation_id, ok=False,
                                         error=row.last_error)
        except Exception as e:  # noqa: BLE001 — loop must survive anything
            logger.warning("[automations] flush loop error: %s", e)
        await asyncio.sleep(_LOOP_INTERVAL_S)


# ── Run-ledger + health plumbing (thin wrappers over executor's) ─────


async def _finalize_run(db, row: AutomationOutbox, *, status: str,
                        outcome: str, error_class: Optional[str] = None,
                        user_message: Optional[str] = None) -> None:
    if not row.job_id:
        return
    from sqlalchemy import select as sa_select
    from .executor import _finalize_job

    siblings = (await db.execute(
        sa_select(AutomationOutbox)
        .where(AutomationOutbox.job_id == row.job_id)
    )).scalars().all()
    if len(siblings) <= 1:
        # v1 (and single-write v2): the row IS the run — pass the
        # caller's exact terminal through, byte-identical to Round 26.
        # The one v2 nuance: a run whose read steps were skipped
        # reports `partial`, not `sent` (v1 never sets the flag).
        if status == "completed":
            job = await db.get(BuildJob, row.job_id)
            if job is not None and (job.config_json or {}).get("steps_partial"):
                outcome = "partial"
        await _finalize_job(db, row.job_id, status=status, outcome=outcome,
                            error_class=error_class,
                            user_message=user_message)
        return

    # Round 28, multi-write runs: the job closes only when EVERY
    # sibling row is terminal, and the terminal aggregates.
    job = await db.get(BuildJob, row.job_id)
    if job is not None and job.status == "waiting_on_user":
        # A confirm-mode card parked the run — the pending-action
        # resolution path owns the close.
        return
    if any(s.status in ("staged", "executing") for s in siblings):
        return
    failed = [s for s in siblings if s.status == "failed"]
    undone = [s for s in siblings if s.status in ("undone", "cancelled")]
    if failed:
        await _finalize_job(
            db, row.job_id, status="failed", outcome="write_failed",
            error_class="tool_error",
            user_message=(failed[0].last_error or "A write failed.")[:300],
        )
    elif len(undone) == len(siblings):
        await _finalize_job(db, row.job_id, status="cancelled",
                            outcome="undone")
    else:
        partial = bool(((job.config_json or {}) if job else {})
                       .get("steps_partial"))
        await _finalize_job(db, row.job_id, status="completed",
                            outcome="partial" if partial else "sent")


async def _record_health(db, automation_id: str, *, ok: bool,
                         error: Optional[str],
                         clean: Optional[bool] = None) -> None:
    """Thin shim onto the executor's recorder.

    It forwards `clean` explicitly rather than `**kwargs`: a shim that
    silently drops an argument its caller passed is how a fix lands in
    the source and never reaches the behaviour, and this one sits
    between the write path and the 3-strike streak (R31-43).
    """
    from .executor import _record_health as _rh
    await _rh(db, automation_id, ok=ok, error=error, clean=clean)


async def _park_run_on_card(db, row: AutomationOutbox,
                            result: dict) -> None:
    """waiting_on_user + config.pending_action_id — the EXISTING
    resolve-pending-action hop closes the run when the user decides.

    R29: the park now stamps `error_class` (the class the reaper and
    the confirm sweep match on — R28 omitted it, which made the park
    invisible to every TTL backstop), a user_message that says what
    "Waiting on you" means, and the session's pending-action card so
    the park is visible where the automation lives.
    """
    from app.agent.job_status import ERR_AWAITING_CONFIRMATION
    from . import confirm

    if not row.job_id:
        return
    job = await db.get(BuildJob, row.job_id)
    if job is None:
        return

    action_id = result.get("action_id")
    expires_at = result.get("expires_at")
    card_msg_id: Optional[str] = None
    automation = await db.get(Automation, row.automation_id)
    if automation is not None and action_id:
        card = confirm.pending_card_payload(
            action_id=str(action_id),
            connector_id=row.connector_id,
            tool_name=row.tool_name,
            summary=str(result.get("summary") or ""),
            payload=result.get("payload")
            if isinstance(result.get("payload"), dict) else None,
            expires_at=str(expires_at) if expires_at else None,
            automation_id=row.automation_id,
            job_id=row.job_id,
        )
        card_msg_id = await confirm.write_pending_card(
            db, automation=automation, job_id=row.job_id, card=card,
        )

    cfg = dict(job.config_json or {})
    if action_id:
        cfg["pending_action_id"] = action_id
    if expires_at:
        cfg["pending_action_expires_at"] = str(expires_at)
    if card_msg_id:
        cfg["pending_card_message_id"] = card_msg_id
    job.config_json = cfg
    job.status = "waiting_on_user"
    job.error_class = ERR_AWAITING_CONFIRMATION
    job.user_message = (
        "Waiting for your approval — nothing is sent until you confirm."
    )
    job.completed_at = None
    await db.commit()

    # R30 §4.10 (AUDIT-12): tell the notification pipeline. The park
    # writes `job.status` straight onto the row — no finalize gate runs,
    # so without this call the run that is waiting for the user is the
    # one run that never tells them. AFTER the commit, for the R28-D
    # reason: a reader inside the transaction sees the old row.
    from . import run_v3 as _run_v3
    await _run_v3.on_parked(db, job_id=row.job_id)

    # R30 §4.9: the park is a `waiting` turn in the thread — the
    # WAITING ON YOU card with Approve / Not now. Best-effort; the
    # R29 pending card above stays the resolve surface.
    if action_id:
        try:
            from . import ledger as _ledger
            # `ensure_thread`, not `thread_for`. R30 wrote this turn
            # only if a thread already existed, because the day-chat
            # pending card was the guaranteed surface and this was the
            # extra one. R31 retired that card, so this IS the surface —
            # and a park with no card at all is the run that is
            # literally waiting for the user being the one run that does
            # not tell them (AUDIT-12's shape, in a new place).
            thread = await _ledger.ensure_thread(
                db, user_id=row.user_id, automation_id=row.automation_id,
            )
            if thread is not None:
                await _ledger.append_turn(
                    db, user_id=row.user_id, thread=thread,
                    run_id=row.job_id, kind="waiting",
                    payload={
                        "pending_action_id": str(action_id),
                        "text": "Nothing happens until you approve.",
                        "expires_at": str(expires_at) if expires_at
                        else None,
                    },
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("[automations] waiting turn skipped: %s", e)


async def _notify_needs_approval(row: AutomationOutbox,
                                 action_id: Optional[str]) -> None:
    try:
        from app.services.agent_notify_client import notify
        auto_name = row.automation_id[:8]
        try:
            async with async_session_maker() as db:
                a = await db.get(Automation, row.automation_id)
                if a is not None:
                    auto_name = a.name
        except Exception:  # noqa: BLE001
            pass
        await notify(
            event_kind="needs_approval",
            title=f"{auto_name} wants to run — review it",
            body="An automation staged an action that needs your approval.",
            data={"pending_action_id": action_id,
                  "automation_id": row.automation_id},
            dedup_key=f"automation:{row.automation_id}:approval",
        )
    except Exception as e:  # noqa: BLE001 — notify is best-effort
        logger.warning("[automations] approval notify failed: %s", e)


# ── R30: the write ledger + the write tool turn ─────────────────────


def _display_of(row: AutomationOutbox) -> dict:
    """The staged display form, with a total fallback so a pre-R30 row
    (no display_json) still ledgers honestly."""
    try:
        d = json.loads(row.display_json) if row.display_json else {}
    except (ValueError, TypeError):
        d = {}
    if not isinstance(d, dict):
        d = {}
    if not d.get("what"):
        from app.services.automation_verbs import turn_action
        d["what"] = turn_action(
            row.connector_id, row.tool_name, kind="write", ok=True,
        )["action"]
    d.setdefault("target", None)
    d.setdefault("audience",
                 "you" if row.tool_name in _DRAFT_TOOLS else "others")
    d.setdefault("reversible", row.tool_name in _DRAFT_TOOLS)
    return d


def _write_ledger_row(row: AutomationOutbox):
    """Build the AutomationWrite row for an about-to-be-executed write.
    Returns None only if the model import itself fails (boot order)."""
    try:
        from app.db.models import AutomationWrite
        d = _display_of(row)
        return AutomationWrite(
            user_id=row.user_id,
            automation_id=row.automation_id,
            run_id=row.job_id or "",
            account_id=row.connector_id,
            what=str(d["what"])[:200],
            target=(str(d["target"])[:200] if d.get("target") else None),
            audience=d["audience"] if d["audience"] in ("you", "others")
            else "others",
            reversible=bool(d.get("reversible")),
            undo_ref=row.id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] write ledger row skipped: %s", e)
        return None


async def _append_write_turn(db, row: AutomationOutbox, write_row) -> None:
    """The v3 write tool turn (+ the draft turn for draft tools) into
    the automation's thread. Best-effort: the ledger row above is the
    durable record; the turn is its display."""
    try:
        from . import ledger
        from .run_v3 import notify_progress  # noqa: F401 — module load check
        # `ensure_thread` for the same reason as the park above: the
        # day-chat card that used to be the guaranteed surface is gone,
        # so a write's record in the thread cannot be conditional on a
        # thread happening to exist already.
        thread = await ledger.ensure_thread(
            db, user_id=row.user_id, automation_id=row.automation_id,
        )
        if thread is None:
            return
        d = _display_of(row)
        from app.services.automation_verbs import turn_action
        act = turn_action(
            row.connector_id, row.tool_name, kind="write", ok=True,
            target=d.get("target"), audience=d["audience"],
        )
        await ledger.append_turn(
            db, user_id=row.user_id, thread=thread, run_id=row.job_id,
            kind="tool",
            payload={
                "account_id": row.connector_id, "tool_kind": "write",
                "action": act["action"], "detail": act["detail"],
                "ok": True, "ms": 0, "steps": [], "items": [],
                "write_ids": [write_row.id] if write_row is not None else [],
                "rest": "",
            },
        )
        if row.tool_name in _DRAFT_TOOLS:
            try:
                payload = json.loads(row.payload_json)
            except (ValueError, TypeError):
                payload = {}
            body = str(payload.get("body") or "").strip()
            if body:
                await ledger.append_turn(
                    db, user_id=row.user_id, thread=thread,
                    run_id=row.job_id, kind="draft",
                    payload={
                        "text": body[:2000],
                        "target": {"account_id": row.connector_id,
                                   "ref": None},
                        "sent_at": None,
                    },
                )
    except Exception as e:  # noqa: BLE001
        logger.debug("[automations] write turn skipped: %s", e)


async def _mark_write_step(db, row: AutomationOutbox, *, ok: bool) -> None:
    """ND-4: the write step's verb flips to its DONE form only when the
    write executed — a refused/failed write wears the failed state, not
    "Posted to Slack". Keyed by display_json.step_id, mutated directly
    in steps_json (the shared substrate)."""
    try:
        if not row.job_id:
            return
        d = _display_of(row)
        step_id = d.get("step_id")
        if not step_id:
            return
        from datetime import datetime as _dt
        from app.agent import job_steps
        job = await db.get(BuildJob, row.job_id)
        if job is None:
            return
        steps = job_steps.parse_steps(job.steps_json)
        now = _dt.utcnow()
        touched = False
        for s in steps:
            if s.get("id") == step_id:
                s["status"] = "done" if ok else "failed"
                s.setdefault("started_at", now.isoformat())
                s["completed_at"] = now.isoformat()
                touched = True
                break
        if touched:
            job.steps_json = job_steps.dump_steps(steps)
            await db.commit()
    except Exception as e:  # noqa: BLE001 — a label flip never blocks a send
        logger.debug("[automations] write-step mark skipped: %s", e)
