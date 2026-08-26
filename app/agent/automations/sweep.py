"""Reconciler sweeps (Round 26 — Phase 6).

Called from job_reconciler's 60-second loop (one hook, gated on the
flag) rather than owning another timer. Four responsibilities:

  1. Stuck runs — an `automation_run` older than 2× the run cap that is
     still running lost its executor (restart mid-run). Terminal
     `failed/lost`, never blind-retried: the outbox idempotency gate
     stops a re-stage, and re-SENDING a write is the one thing that
     must not happen (the flush loop applies the same rule to
     `executing` outbox rows).
  2. Stale bindings — an ARMED automation whose primitive row went
     missing or disabled out-of-band gets its bindings rebuilt and
     re-enabled. Catch-up is naturally capped: polls re-observe at most
     one page and the event dedupe gate collapses everything already
     seen.
  3. Auto-pause — consecutive_failures >= threshold flips the
     automation to `error`, deactivates bindings, and posts exactly ONE
     chat notice with a fix chip (`error_notice_at` is the dedupe) plus
     one push notify.
  4. Expired confirm-parks (R29) — runs parked on a confirmation card
     whose TTL passed without the platform's resolve hop landing close
     as outcome "skipped" (confirm.sweep_expired_confirm_parks).
  5. Token refresh / webhook renewal — deliberately NOT here: token
     refresh is lazy+coalesced in the dispatcher at call time, and the
     Gmail watch already rides the platform's 6-hour refresh cron. A
     second refresher would race the coalescing locks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import select, update as sa_update

from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationBinding, BuildJob, Routine, Trigger,
    AUTOMATION_AUTO_PAUSE_FAILURES, AUTOMATION_RUN_CAP_S,
)

logger = logging.getLogger(__name__)

_STUCK_AFTER = timedelta(seconds=AUTOMATION_RUN_CAP_S * 2)


async def sweep_automations() -> dict:
    """One pass; every leg isolated so a failure in one cannot starve
    the others. Returns counters for the reconcile log line."""
    stats = {"stuck_runs": 0, "stale_bindings": 0, "auto_paused": 0,
             "expired_confirms": 0}
    try:
        # ND-7d ORDER MATTERS: a user-stopped run must be claimed by the
        # wedged-stop leg BEFORE the stuck-run reaper can see it — the
        # reaper's terminal is `failed/lost` with a "Fix this" chip, i.e.
        # the product telling the user it broke when they pressed Stop
        # (live: run 5f3f57bf reaped at start+363s). The reaper ALSO
        # excludes stop-requested rows, so this is belt and braces.
        stats["wedged_stops"] = await _sweep_wedged_stops()
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] wedged-stop sweep failed: %s", e)
    try:
        stats["stuck_runs"] = await _sweep_stuck_runs()
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] stuck-run sweep failed: %s", e)
    try:
        stats["stale_bindings"] = await _sweep_stale_bindings()
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] stale-binding sweep failed: %s", e)
    try:
        stats["auto_paused"] = await _sweep_auto_pause()
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] auto-pause sweep failed: %s", e)
    try:
        # R29: confirm-parks whose card expired and whose platform
        # resolve hop never landed — close as outcome "skipped".
        from .confirm import sweep_expired_confirm_parks
        stats["expired_confirms"] = await sweep_expired_confirm_parks()
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] confirm-park sweep failed: %s", e)
    try:
        # R30 §4.8: soft-deleted automations past the 30-day window are
        # purged for real — thread, turns, facts, engine namespace.
        async with async_session_maker() as db:
            stats["purged"] = await sweep_purge_soft_deleted(db)
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] purge sweep failed: %s", e)
    if any(stats.values()):
        logger.info("[automations] sweep %s", stats)
    return stats


SOFT_DELETE_RETENTION_DAYS = 30


_WEDGED_STOP_AFTER = timedelta(seconds=120)


async def _sweep_wedged_stops() -> int:
    """ND-7d: runs whose stop was requested over two minutes ago and
    which are STILL running never got their terminal from the executor
    (the stop landed mid-write, or the run task died). Terminalize each
    through run_v3.handle_stop so the SHAPE is honest — stopped_by_user
    + checkpoint + the stop note with the real writes count, and NO
    "Fix this" chip (the chip is minted only for status=failed). This
    leg runs BEFORE the stuck-run reaper, which is what produced the
    lying `failed/lost` terminal on the live run 5f3f57bf."""
    from app.agent import job_steps as _js
    from .run_v3 import handle_stop

    cutoff = datetime.utcnow() - _WEDGED_STOP_AFTER
    n = 0
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(BuildJob)
            .where(BuildJob.job_type == "automation_run")
            .where(BuildJob.status.in_(("queued", "running")))
            .where(BuildJob.stop_requested_at.isnot(None))
            .where(BuildJob.stop_requested_at < cutoff)
        )).scalars().all()
        for job in rows:
            try:
                automation = await db.get(Automation, job.source_id or "")
                if automation is None:
                    continue
                done = sum(
                    1 for st in _js.parse_steps(job.steps_json)
                    if st.get("status") in ("done", "completed")
                )
                await handle_stop(db, automation=automation, job=job,
                                  step_index=done)
                n += 1
                logger.info("[automations] wedged stop terminalised "
                            "run=%s", job.id)
            except Exception as e:  # noqa: BLE001 — next tick retries
                await db.rollback()
                logger.warning("[automations] wedged-stop failed run=%s: "
                               "%s", job.id, e)
    return n



async def sweep_purge_soft_deleted(db) -> int:
    """Hard-delete every automation whose soft delete is older than the
    retention window (§4.8: "memory kept for 30 days then purged")."""
    cutoff = datetime.utcnow() - timedelta(days=SOFT_DELETE_RETENTION_DAYS)
    rows = (await db.execute(
        select(Automation).where(
            Automation.deleted_at.isnot(None),
            Automation.deleted_at < cutoff,
        )
    )).scalars().all()
    from .service import _hard_delete
    purged = 0
    for a in rows:
        try:
            await _hard_delete(db, a, a.user_id)
            purged += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("[automations] purge failed id=%s: %s",
                           a.id, e)
            await db.rollback()
    return purged


async def _sweep_stuck_runs() -> int:
    """ND-10: this used to be ONE bulk UPDATE, which is why a reaped run
    left the card lying. `_stamp_last_outcome` has exactly one caller —
    `_finalize_job` — so a raw UPDATE terminalised the row while the
    automation kept advertising an older, rosier `last_outcome` (live:
    the founder's brief still read "Posted to Slack — some sources were
    unavailable" hours after run 5f3f57bf was reaped). It also skipped
    the outcome notification and, in R30, the v3 ledger close.

    Every reaper terminal now goes through the SAME gated finalize the
    live path uses, so stamp + notify + ledger close stay coupled by
    construction. `_finalize_job`'s guarded UPDATE keeps it exactly-once
    even if the executor lands at the same moment.
    """
    from .executor import _finalize_job

    cutoff = datetime.utcnow() - _STUCK_AFTER
    n = 0
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(BuildJob)
            .where(BuildJob.job_type == "automation_run")
            .where(BuildJob.status.in_(("queued", "running")))
            .where(BuildJob.created_at < cutoff)
            # ND-7d: a run the user STOPPED is not a stuck run. Reaping
            # it as `failed/lost` mints a "Fix this" chip offering to
            # diagnose a failure that never happened — the write rail
            # stayed honest (zero sends) but the narrative lied. Those
            # rows belong to `_sweep_wedged_stops`, which terminalises
            # them as stopped_by_user with the real writes count.
            .where(BuildJob.stop_requested_at.is_(None))
        )).scalars().all()
        for job in rows:
            try:
                await _finalize_job(
                    db, job.id, status="failed", outcome="lost",
                    error_class="interrupted",
                    user_message="This run was interrupted and did not "
                                 "finish.",
                )
                n += 1
            except Exception as e:  # noqa: BLE001 — next tick retries
                await db.rollback()
                logger.warning("[automations] stuck-run finalize failed "
                               "run=%s: %s", job.id, e)
    if n:
        logger.info("[automations] terminalised %d stuck runs", n)
    return n


async def _sweep_stale_bindings() -> int:
    """Armed automations whose primitive row vanished or was disabled
    out-of-band. Rebuild from the spec (the spec is the truth) and
    re-enable."""
    fixed = 0
    async with async_session_maker() as db:
        armed = (await db.execute(
            select(Automation).where(Automation.status == "armed")
        )).scalars().all()
        for automation in armed:
            bindings = (await db.execute(
                select(AutomationBinding)
                .where(AutomationBinding.automation_id == automation.id)
            )).scalars().all()
            stale = not bindings
            for b in bindings:
                target = None
                if b.kind == "routine":
                    target = await db.get(Routine, b.target_id)
                elif b.kind == "trigger":
                    target = await db.get(Trigger, b.target_id)
                if target is None or not getattr(target, "enabled", False):
                    stale = True
            if not stale:
                continue
            try:
                from .service import _parse_spec
                from . import compiler
                vspec = _parse_spec(automation)
                await compiler.compile_bindings(db, automation, vspec)
                routine_ids = await compiler.set_bindings_active(
                    db, automation, True,
                )
                await db.commit()
                await compiler.nudge_routines(routine_ids)
                fixed += 1
                logger.info("[automations] reset stale bindings for %s",
                            automation.id)
            except Exception as e:  # noqa: BLE001 — leave for next sweep
                await db.rollback()
                logger.warning("[automations] binding reset failed %s: %s",
                               automation.id, e)
    return fixed


async def _sweep_auto_pause() -> int:
    paused = 0
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Automation)
            .where(Automation.status == "armed")
            .where(Automation.consecutive_failures
                   >= AUTOMATION_AUTO_PAUSE_FAILURES)
        )).scalars().all()
        for automation in rows:
            from . import compiler
            routine_ids: list[str] = []
            try:
                routine_ids = await compiler.set_bindings_active(
                    db, automation, False,
                )
            except compiler.CompileError:
                pass  # binding already gone — the pause still stands
            automation.status = "error"
            automation.paused_reason = "auto_failures"
            already_noticed = automation.error_notice_at is not None
            automation.error_notice_at = datetime.utcnow()
            await db.commit()
            await compiler.nudge_routines(routine_ids)
            paused += 1
            if not already_noticed:
                await _post_error_notice(db, automation)
    return paused


def _fix_chip_for(automation: Automation) -> dict:
    """The notice's one-tap repair path: `prompt` is sent as a user
    turn into this automation's session (CONTRACTS-R29 §1). The verbs
    module composes it; this fallback keeps the notice whole when the
    module predates the rebase or ever raises."""
    try:
        from app.services.automation_verbs import fix_chip
        chip = fix_chip(automation.name, "auto_paused",
                        automation.last_error)
        if isinstance(chip, dict) and chip.get("label") and chip.get("prompt"):
            return {"label": str(chip["label"]), "prompt": str(chip["prompt"])}
    except Exception:  # noqa: BLE001 — the chip must always exist
        pass
    return {
        "label": "Help me fix it",
        "prompt": (
            f'My automation "{automation.name}" was paused after repeated '
            f"failures. Diagnose what went wrong, help me fix it, and "
            f"turn it back on."
        ),
    }


async def _post_error_notice(db, automation: Automation) -> None:
    """Exactly ONE chat notice + one push per error episode
    (`error_notice_at` dedupes; arming again clears it)."""
    chip = _fix_chip_for(automation)
    # R30 copy contract: no emoji, no markdown markup, and the raw
    # provider error never reaches the UI unformatted (D-03) — it stays
    # on the row for diagnosis; the fix chip carries the honest ask.
    # C's template module owns the sentence; the fallback keeps the
    # notice whole when the module predates the merge.
    #
    # R31: this called `auto_pause_body(name, 3)` against a ZERO-arg
    # function. The bare except below swallowed the TypeError, so A's
    # fallback shipped every time and C's sentence — written precisely
    # to replace a live string wearing an emoji and markdown bold — has
    # never reached a user. An arity mismatch that a broad except turns
    # into "the other branch always wins" is invisible to every test
    # that only checks a notice exists.
    text = None
    try:
        from .notification_templates import auto_pause_body
        text = auto_pause_body()
    except Exception:  # noqa: BLE001 — the notice must always exist
        text = None
    if not text:
        text = (
            f"{automation.name} was paused after "
            f"{AUTOMATION_AUTO_PAUSE_FAILURES} failed runs in a row. "
            f"Nothing more will run until you resume it — I can help "
            f"you fix it."
        )
    try:
        # CONTRACTS-R31 §4.1: the notice lands in the automation's own
        # THREAD, as a turn. It used to be a day-chat Message plus a
        # `type: "message"` chat frame — an automation speaking in the
        # main chat's own voice, which is exactly what the isolation
        # rule forbids. The push below is untouched: a paused
        # automation must still reach the user when the app is closed.
        from . import ledger as _ledger
        thread = await _ledger.ensure_thread(
            db, user_id=automation.user_id, automation_id=automation.id,
        )
        await _ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=None,
            kind="agent", payload={"text": text},
        )
        del chip  # the thread's own fix affordances replace the chip
    except Exception as e:  # noqa: BLE001 — the notice is best-effort
        logger.warning("[automations] error notice write failed %s: %s",
                       automation.id, e)
    try:
        from app.services.agent_notify_client import notify
        await notify(
            event_kind="mission_failed",
            title=f"{automation.name} was paused",
            body="It failed 3 times in a row. Open Activity to fix it.",
            data={"automation_id": automation.id},
            dedup_key=f"automation:{automation.id}:auto_pause",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] error notify failed %s: %s",
                       automation.id, e)
