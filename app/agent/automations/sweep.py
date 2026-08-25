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
    cutoff = datetime.utcnow() - _STUCK_AFTER
    async with async_session_maker() as db:
        res = await db.execute(
            sa_update(BuildJob)
            .where(BuildJob.job_type == "automation_run")
            .where(BuildJob.status.in_(("queued", "running")))
            .where(BuildJob.created_at < cutoff)
            .values(
                status="failed",
                outcome="lost",
                error_class="interrupted",
                user_message="This run was interrupted and did not finish.",
                completed_at=datetime.utcnow(),
            )
        )
        await db.commit()
        n = res.rowcount or 0
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
    text = None
    try:
        from .notification_templates import auto_pause_body
        text = auto_pause_body(
            automation.name, AUTOMATION_AUTO_PAUSE_FAILURES,
        )
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
        # R28: the notice lands in the automation's own session thread,
        # not the shared routine thread — same exactly-once semantics,
        # new address. The broadcast frame is unchanged (type "message",
        # no channel key) so both clients keep hearing it live.
        from app.agent.automations.session import write_session_message
        from app.agent.routines.message_writer import (
            broadcast_routine_message,
        )
        message_id, day_chat_id = await write_session_message(
            db,
            user_id=automation.user_id,
            automation_id=automation.id,
            content=text,
            metadata={"fix_chip": chip},
            title=automation.name,
        )
        if not message_id:
            raise RuntimeError("session write returned no message id")
        try:
            await broadcast_routine_message(
                automation.user_id,
                message_id=message_id,
                day_chat_id=day_chat_id,
                source="automation",
                content=text,
                routine_name=automation.name,
                extra={"fix_chip": chip,
                       "automation_id": automation.id},
            )
        except Exception:  # noqa: BLE001 — broadcast is a courtesy
            pass
    except Exception as e:  # noqa: BLE001 — chat write is best-effort
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
