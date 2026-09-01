"""Write-outbox flush — the ONLY code that sends an automation write.

Every row waits out its undo window (`execute_after`), is claimed with
a single guarded UPDATE (staged→executing — the double-fire defence,
same idiom as connector_pending_actions), and goes to the provider via
the platform's grant-gated dispatch RPC. Retryable failures back off
(10/30/90s, the routines ladder) and hard-fail after 3 attempts.

Two entry points, and BOTH send on a session the outbox owns:
  - `flush_row_when_due(db, id)` — inline from the run, sub-second
    after the window closes (the happy path). `db` is the run's
    session and is used only to re-read what the send changed.
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
from sqlalchemy.orm import aliased
from sqlalchemy.orm.util import identity_key

from app.db.database import async_session_maker
from app.db.models import Automation, AutomationOutbox, BuildJob
from . import registry as reg
from .draft_card import DRAFT_TOOLS as _DRAFT_TOOLS

logger = logging.getLogger(__name__)

_RETRY_DELAYS_S = (10, 30, 90)
_MAX_ATTEMPTS = 3
_LOOP_INTERVAL_S = 5.0
_BOOT_DELAY_S = 20.0
# How long an `executed` row may sit under a still-running job before
# the loop treats its terminal as LOST. Comfortably past the 30s
# `statement_timeout` (db/database.py), so a finalize merely blocked on
# a slow statement is not called missing — and still six times faster
# than the 360s stuck-run reaper, which is the only thing that used to
# close such a run and closed it as "lost" over a post that landed.
_LOST_TERMINAL_AFTER_S = 60


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
    claim and send — on the outbox's OWN session, never the caller's.
    Returns the terminal outbox status, or None when the claim was lost.

    R42 made this call inline from the run (`executor_v2._run_steps`),
    so `db` here is the RUN's session and the run goes on using its
    `automation`, `job` and `thread` instances after we return. The send
    contains best-effort helpers that must roll back on a failed
    statement (`_reopen`), and a ROLLBACK expires every instance the
    session holds whatever `expire_on_commit` says — so a repair inside
    the outbox would turn the run's next `job.status` into lazy IO an
    async session cannot perform, i.e. a `MissingGreenlet` raised out of
    the run, caused by the write's own error handling. A separate
    session makes that structurally impossible instead of leaving it to
    review. `_claim` is a guarded UPDATE, so exactly-once is the
    database's guarantee and does not care whose session asks.

    Two obligations come with the split. The caller MUST have COMMITTED
    before calling — all three call sites commit in the same breath as
    they stage — and that is two requirements in one. A row still
    pending in the caller's transaction is invisible to every other
    session, so we would flush nothing at all; and P14 was this send's
    UPDATE of `build_jobs` blocking behind a progress stamp the run had
    flushed and not committed, until the 30s `statement_timeout`
    cancelled it with the post already landed. What fixed P14 is the
    ORDER — the write goes before the narration, so the run's session is
    idle here — not the session being shared, so it survives the split.
    Do not move a flush ahead of a commit.

    And the send changes rows the caller has already loaded, so
    `_resync_caller` re-reads them: SQLAlchemy would otherwise keep
    answering the run's `db.get(BuildJob, …)` from its identity map with
    the pre-send copy, and the resume path reads exactly that row's
    `status` to decide whether the write it just repaired may flip the
    run back to `sent`.
    """
    async with async_session_maker() as own:
        row = await own.get(AutomationOutbox, outbox_id)
        if row is None:
            logger.warning("[automations] outbox %s not visible to the "
                           "flush — was it staged without a commit?",
                           outbox_id)
            return None
        delay = (row.execute_after - datetime.utcnow()).total_seconds()
        job_id = row.job_id
    # The window is seconds and a session that has run a statement holds
    # a pooled connection until it ends — so the sleep happens between
    # two short sessions, not inside one.
    if delay > 0:
        await asyncio.sleep(delay)
    status: Optional[str] = None
    async with async_session_maker() as own:
        if await _claim(own, outbox_id):
            status = await _execute_claimed(own, outbox_id)
    await _resync_caller(db, outbox_id, job_id)
    return status


async def _resync_caller(db, outbox_id: str, job_id: Optional[str]) -> None:
    """Re-read, on the CALLER's session, the two rows the flush just
    changed on another one.

    Only rows the caller already holds: `async_session_maker` sets
    `expire_on_commit=False`, so its identity map answers from the copy
    it loaded before the send, and a plain `db.get` would hand back that
    stale row rather than notice. A row the caller never loaded needs
    nothing — reading it here would only be a SELECT for no one.
    """
    for model, pk in ((AutomationOutbox, outbox_id), (BuildJob, job_id)):
        if not pk:
            continue
        inst = db.identity_map.get(identity_key(model, pk))
        if inst is None:
            continue
        try:
            await db.refresh(inst)
        except Exception as e:  # noqa: BLE001 — the write already went
            # out and the run is already terminal; raising here would
            # unwind a landed send. We do NOT roll back: expiring the
            # run's instances is the exact failure this split exists to
            # prevent, and the caller's own handler owns its session.
            logger.warning("[automations] post-flush reload of %s %s "
                           "failed — the run may read a stale row: %s",
                           model.__name__, pk, e)


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

    import time as _time
    _t0 = _time.monotonic()
    result = await reg.dispatch_via_platform(
        row.user_id,
        connector_id=row.connector_id,
        tool_name=row.tool_name,
        tool_input=payload,
        grant_id=row.grant_id,
        automation_id=row.automation_id,
        request_id=f"outbox:{row.id}",
    )
    _ms = int((_time.monotonic() - _t0) * 1000)
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
        # R42 (P14-4): the send's own record comes BEFORE the terminal —
        # the order the failure branch below has always used. The client
        # latches a run card's content the instant `run_in_flight` goes
        # null, and the terminal is what nulls it (`_finalize_job` →
        # `run_v3.on_terminal` → `ledger.emit_run_finished`), so a turn
        # appended after it lands on a card that can no longer show it:
        # the landed card never said what it posted or where.
        await _append_write_turn(db, row, write_row, ms=_ms)
        if row.tool_name in _DRAFT_TOOLS:
            # The proactive-draft surface (R29): a session card that
            # names the draft and tells the "nothing was sent" truth.
            from .draft_card import write_draft_card
            await write_draft_card(db, row, result)
            # It rolls back its own failure, and a rollback expires every
            # instance the session holds — so the terminal below would
            # read `row` through lazy IO an async session cannot do.
            # `get` is free when nothing expired it.
            fresh = await db.get(AutomationOutbox, outbox_id)
            if fresh is not None:
                row = fresh
        await _finalize_run_safe(db, row, status="completed", outcome="sent")
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
    # R35: a delivery that hard-failed used to vanish — no tool turn,
    # no needs-you card, absent from `accounts_failed` — so the run
    # said "Could not reach GitHub and Teams" while the one step the
    # automation exists FOR had also failed, silently. The founder's
    # Slack card on the fleet came from the deleted question-run path;
    # this is the honest replacement: the turn (with the real call
    # under it), the card with its button, and membership in the
    # failed list so `resume_source` can pick it up.
    await _append_failed_write_turn(
        db, row, ms=_ms,
        reason_kind=str(kind or ""), message=str(result.get("message") or ""),
    )
    await _finalize_run_safe(
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
                for row in await _lost_terminals(db):
                    # R42 (P14-5): the send LANDED and its terminal did
                    # not. The row is committed `executed`, so the staged
                    # query never claims it again and the stale sweep
                    # above only looks at `executing` — nothing closed
                    # the run until the 360s stuck-run reaper called a
                    # real Slack post "lost". `_finalize_run_safe` is the
                    # in-process repair; this is the one for the process
                    # that died between the send and the terminal.
                    logger.warning(
                        "[automations] outbox %s executed but run %s was "
                        "never finalized — closing it now",
                        row.id, row.job_id,
                    )
                    await _finalize_run_safe(db, row, status="completed",
                                             outcome="sent")
        except Exception as e:  # noqa: BLE001 — loop must survive anything
            logger.warning("[automations] flush loop error: %s", e)
        await asyncio.sleep(_LOOP_INTERVAL_S)


async def _lost_terminals(db) -> list[AutomationOutbox]:
    """Executed rows whose run nobody ever closed.

    Three conditions, and each one is load-bearing. The job is still
    `running` — a confirm-mode park (`waiting_on_user`) is owned by the
    pending-action resolution, and a run already terminal needs nothing.
    No sibling is still `staged`/`executing` — a multi-write run legally
    holds an executed row for the length of another row's undo window
    and retry ladder, and warning about that every loop is noise, not a
    signal. And the row is past `_LOST_TERMINAL_AFTER_S`, so a finalize
    merely blocked on a slow statement finishes on its own first.
    """
    sib = aliased(AutomationOutbox)
    rows = (await db.execute(
        select(AutomationOutbox)
        .join(BuildJob, BuildJob.id == AutomationOutbox.job_id)
        .where(AutomationOutbox.status == "executed")
        .where(AutomationOutbox.executed_at <= datetime.utcnow() - timedelta(
            seconds=_LOST_TERMINAL_AFTER_S))
        .where(BuildJob.status == "running")
        .where(~select(sib.id)
               .where(sib.job_id == AutomationOutbox.job_id)
               .where(sib.status.in_(("staged", "executing")))
               .exists())
        .limit(20)
    )).scalars().all()
    # A confirm-mode row is `executed` too — it means "the dispatcher
    # staged the approval card", not "sent". Announcing `sent` for one
    # would tell the user about a send that has not happened; if its
    # park failed, the confirm sweep and the reaper still own it.
    return [r for r in rows if _result_kind(r) != "confirmation_required"]


def _result_kind(row: AutomationOutbox) -> str:
    try:
        return str((json.loads(row.result_json or "{}") or {}).get("kind")
                   or "")
    except (ValueError, TypeError, AttributeError):
        return ""


async def _reopen(db, row: Optional[AutomationOutbox] = None) -> None:
    """Undo a best-effort helper's failed statement, and hand the caller
    back a row it can still read.

    Two halves, and the second is not optional. A helper that swallows a
    DB error must not leave the session in a failed transaction: the
    next statement — the terminal — then raises `InFailedSQLTransaction`
    out of a path whose row is already committed `executed`, and nothing
    ever closes the run (`statement_timeout` is 30s, db/database.py, so
    this is reachable from any slow statement, not only from a bug).
    But a ROLLBACK expires every instance the session holds, whatever
    `expire_on_commit` says — and this whole module reads its row across
    commits because `async_session_maker` sets that False. Without the
    reload the caller's next `row.user_id` is lazy IO an async session
    cannot perform, and the repair for the swallowed error becomes a
    `MissingGreenlet` standing where it used to be.
    """
    try:
        await db.rollback()
    except Exception as e:  # noqa: BLE001 — the session is being abandoned
        logger.debug("[automations] rollback after a best-effort DB "
                     "failure did not take: %s", e)
        return
    if row is None:
        return
    try:
        await db.refresh(row)
    except Exception as e:  # noqa: BLE001 — the caller's own handler owns
        # what happens next; an unreadable row raises there, not here.
        logger.debug("[automations] outbox row reload after rollback "
                     "failed: %s", e)


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


async def _finalize_run_safe(db, row: AutomationOutbox, *, status: str,
                             outcome: str,
                             error_class: Optional[str] = None,
                             user_message: Optional[str] = None) -> None:
    """`_finalize_run`, on a session a best-effort helper may have
    poisoned (R42, P14-5).

    By the time this is called the outbox row is committed terminal, so
    the staged-row query will never claim it again: an exception here
    unwinds out of `_execute_claimed`, is caught by `flush_loop`'s
    blanket handler, and leaves a write that ALREADY WENT OUT attached
    to a run no surface ever closes. One rollback re-opens the session
    and the retry costs milliseconds — `_finalize_job`'s guarded UPDATE
    only moves a non-terminal row, so a second attempt is free even when
    the first got further than it looked.

    The arguments are spelled out rather than forwarded as `**kwargs`
    for the same reason `_record_health` below spells its own out: a
    wrapper that silently drops what its caller passed is how a fix
    lands in the source and never reaches the behaviour.
    """
    # Read before the first attempt: a failure in there can expire `row`,
    # and the log line that reports it must not be the thing that raises.
    outbox_id, job_id = row.id, row.job_id
    try:
        await _finalize_run(db, row, status=status, outcome=outcome,
                            error_class=error_class,
                            user_message=user_message)
        return
    except Exception as e:  # noqa: BLE001 — retried on a clean session
        logger.warning("[automations] finalize failed outbox=%s run=%s — "
                       "retrying on a clean session: %s",
                       outbox_id, job_id, e)
    await _reopen(db, row)
    try:
        await _finalize_run(db, row, status=status, outcome=outcome,
                            error_class=error_class,
                            user_message=user_message)
    except Exception as e:  # noqa: BLE001 — `_lost_terminals` is the
        # backstop: it re-finalizes an executed row whose job is still
        # running, one loop interval later.
        logger.error("[automations] finalize retry failed outbox=%s run=%s "
                     "— leaving it to the lost-terminal sweep: %s",
                     outbox_id, job_id, e)
        # WITH the row: `_execute_claimed` reads it on the very next
        # line either way (`row.job_id` for the health flag, or
        # `row.automation_id`/`row.last_error` for the failure record),
        # and a rollback with nothing reloaded is the lazy-IO trap
        # `_reopen` exists to close, left open on the one path that
        # reaches it.
        await _reopen(db, row)


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
            await _reopen(db, row)


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


async def _append_write_turn(db, row: AutomationOutbox, write_row,
                             *, ms: int = 0) -> None:
    """The v3 write tool turn (+ the draft turn for draft tools) into
    the automation's thread. Best-effort: the ledger row above is the
    durable record; the turn is its display."""
    try:
        from . import ledger
        from .run_v3 import notify_progress  # noqa: F401 — module load check
        from .executor_v2 import _action_record
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
                "ok": True, "ms": max(int(ms), 0), "steps": [], "items": [],
                "actions": [_action_record(
                    row.tool_name, ok=True, ms=ms,
                    summary=act["detail"] or None,
                )],
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
        await _reopen(db, row)


async def _append_failed_write_turn(
    db, row: AutomationOutbox, *, ms: int,
    reason_kind: str, message: str,
) -> None:
    """R35: the failed delivery's honest record — the ok=False tool
    turn (which is what puts the account into `close_ledger`'s
    `accounts_failed`, and with it inside `resume_source`'s reach), the
    needs-you card with its button, and the health projection update.
    Best-effort like its success twin: the outbox row is the durable
    record; these are its display."""
    try:
        from . import account_health as _ah, ledger
        from .executor_v2 import _action_record, merge_job_config
        from app.services import automation_verbs as _verbs
        thread = await ledger.ensure_thread(
            db, user_id=row.user_id, automation_id=row.automation_id,
        )
        if thread is None:
            return
        code = _ah.classify(reason_kind, message)
        state, fix = _ah.state_for_reason(code)
        name = _verbs.display_name(row.connector_id) or "the account"
        act = _verbs.failure_action(row.connector_id, reason_kind or None)
        line = _ah.sentence_for(
            account_state=state, reason_code=code,
            connector_id=row.connector_id or "",
            name=_ah.display_of(row.connector_id or ""),
        ) or act["detail"] or f"Could not post to {name}."
        await ledger.append_turn(
            db, user_id=row.user_id, thread=thread, run_id=row.job_id,
            kind="tool",
            payload={
                "account_id": row.connector_id, "tool_kind": "write",
                "action": act["action"], "detail": act["detail"],
                "ok": False, "ms": max(int(ms), 0),
                "steps": [
                    {"text": f"Asked {name} to take the post", "ok": True},
                    {"text": (act["detail"] or "It did not answer")
                     .capitalize(), "ok": False},
                ],
                "actions": [_action_record(
                    row.tool_name, ok=False, ms=ms,
                    summary=(message or reason_kind or None),
                )],
                "items": [], "write_ids": [], "rest": "",
                "line": line, "tone": "warning",
                **({"fix": fix} if fix else {}),
                "reason_code": code,
            },
        )
        await ledger.append_turn(
            db, user_id=row.user_id, thread=thread, run_id=row.job_id,
            kind="needs_you",
            payload=_ah.needs_you_payload(
                account_id=row.connector_id,
                connector_id=row.connector_id,
                name=name,
                reason_code=code or "unknown_error",
            ),
        )
        if row.job_id:
            job = await db.get(BuildJob, row.job_id)
            cfg = (job.config_json or {}) if job is not None else {}
            failed = list(cfg.get("accounts_failed") or [])
            if row.connector_id not in failed:
                failed.append(row.connector_id)
            sources = list(cfg.get("failed_sources") or [])
            if not any(f.get("account_id") == row.connector_id
                       for f in sources):
                sources.append({
                    "account_id": row.connector_id,
                    "reason_code": code,
                    "step_id": (_display_of(row).get("step_id") or ""),
                    "at": datetime.utcnow().isoformat() + "Z",
                    "message": (message or reason_kind or "")[:300],
                })
            await merge_job_config(
                db, row.job_id,
                accounts_failed=failed, failed_sources=sources,
            )
        await _ah.record_use(
            db, user_id=row.user_id, account_id=row.connector_id,
            ok=False, reason_code=reason_kind or "", message=message or "",
        )
    except Exception as e:  # noqa: BLE001 — display beside a durable row
        logger.warning("[automations] failed-write turn skipped: %s", e)
        await _reopen(db, row)


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
        await _reopen(db, row)
