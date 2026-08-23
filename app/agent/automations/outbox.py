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
        await db.commit()
        await _finalize_run(db, row, status="completed", outcome="sent")
        await _record_health(db, row.automation_id, ok=True, error=None)
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
        await _park_run_on_card(db, row, action_id)
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
    from .executor import _finalize_job
    await _finalize_job(db, row.job_id, status=status, outcome=outcome,
                        error_class=error_class, user_message=user_message)


async def _record_health(db, automation_id: str, *, ok: bool,
                         error: Optional[str]) -> None:
    from .executor import _record_health as _rh
    await _rh(db, automation_id, ok=ok, error=error)


async def _park_run_on_card(db, row: AutomationOutbox,
                            action_id: Optional[str]) -> None:
    """waiting_on_user + config.pending_action_id — the EXISTING
    resolve-pending-action hop then closes the run when the user
    decides, and the reaper's card-park sweep is the TTL backstop."""
    if not row.job_id:
        return
    job = await db.get(BuildJob, row.job_id)
    if job is None:
        return
    cfg = dict(job.config_json or {})
    if action_id:
        cfg["pending_action_id"] = action_id
    job.config_json = cfg
    job.status = "waiting_on_user"
    job.completed_at = None
    await db.commit()


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
