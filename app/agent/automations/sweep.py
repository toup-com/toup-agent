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
  4. Token refresh / webhook renewal — deliberately NOT here: token
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
    stats = {"stuck_runs": 0, "stale_bindings": 0, "auto_paused": 0}
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
    if any(stats.values()):
        logger.info("[automations] sweep %s", stats)
    return stats


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
                await compiler.set_bindings_active(db, automation, True)
                await db.commit()
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
            try:
                await compiler.set_bindings_active(db, automation, False)
            except compiler.CompileError:
                pass  # binding already gone — the pause still stands
            automation.status = "error"
            automation.paused_reason = "auto_failures"
            already_noticed = automation.error_notice_at is not None
            automation.error_notice_at = datetime.utcnow()
            await db.commit()
            paused += 1
            if not already_noticed:
                await _post_error_notice(db, automation)
    return paused


async def _post_error_notice(db, automation: Automation) -> None:
    """Exactly ONE chat notice + one push per error episode
    (`error_notice_at` dedupes; arming again clears it)."""
    text = (
        f"⚠️ **{automation.name}** was paused after "
        f"{AUTOMATION_AUTO_PAUSE_FAILURES} failed runs in a row.\n\n"
        f"Last error: {(automation.last_error or 'unknown')[:200]}\n\n"
        f"[[navigate:/activity]]"
    )
    try:
        from app.agent.routines.message_writer import (
            broadcast_routine_message, write_routine_message,
        )
        message_id, day_chat_id = await write_routine_message(
            db,
            user_id=automation.user_id,
            content=text,
            source="automation",
            title=automation.name,
        )
        try:
            await broadcast_routine_message(
                automation.user_id,
                message_id=message_id,
                day_chat_id=day_chat_id,
                source="automation",
                content=text,
                routine_name=automation.name,
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
