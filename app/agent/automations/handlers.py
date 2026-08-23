"""The engine's hooks into the two firing primitives.

Routine side: two RoutineHandler implementations —
`automation_poll` (hidden system routine) and `automation_schedule`
(user-visible). Registered by `register_automation_handlers()`, which
`app/agent/routines/__init__.py` calls ONLY when
`settings.automations_enabled` is true, so a dark tenant's handler
registry is byte-identical to today's.

Trigger side: `handle_push_events` — the `run_automation` action branch
in the email handler calls it with the fetched, filter-passed batch.

Both funnel into executor.py. The handler contract mirrors autopilot's:
expected failures come back as a RoutineResult, never an exception —
the runner's retry ladder is for transport blips, not for re-running a
poll that already ingested half its events (the dedupe gate makes that
safe anyway, but the health ledger reads cleaner without ghost
attempts).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.routines.base_handler import RoutineResult
from app.db.models import Automation
from . import executor
from .service import find_by_binding_target, parse_spec_live

logger = logging.getLogger(__name__)


async def _automation_for_routine(
    db: AsyncSession, routine: Any,
) -> Automation | None:
    """Routine → its automation, preferring the config back-pointer and
    falling back to the binding table (stale-binding tolerant)."""
    automation_id = (routine.config_json or {}).get("automation_id")
    if automation_id:
        a = await db.get(Automation, automation_id)
        if a is not None:
            return a
    return await find_by_binding_target(db, routine.id)


def _not_runnable(automation: Automation | None) -> RoutineResult | None:
    if automation is None:
        return RoutineResult(
            status="failed", error_class="orphan_binding",
            error_detail="No automation owns this routine — the sweep "
                         "will disable it.",
        )
    if automation.status != "armed":
        # A disabled binding should never fire, but a reconcile race
        # can slip one tick through — a clean no-op, not a failure.
        return RoutineResult(status="success", outcome="success_empty")
    return None


class AutomationPollHandler:
    kind = "automation_poll"

    async def execute(self, routine, run, db: AsyncSession) -> RoutineResult:
        automation = await _automation_for_routine(db, routine)
        early = _not_runnable(automation)
        if early is not None:
            return early
        try:
            vspec = await parse_spec_live(automation)
            stats = await executor.poll_and_run(db, automation, vspec)
        except Exception as e:  # noqa: BLE001 — poll transport/shape errors
            logger.warning("[automations] poll failed automation=%s: %s",
                           automation.id, e)
            await executor._record_health(
                db, automation.id, ok=False, error=str(e)[:500],
            )
            return RoutineResult(
                status="failed", error_class="poll_failed",
                error_detail=str(e)[:300],
            )
        if stats["failed"]:
            return RoutineResult(
                status="partial" if stats["ran"] else "failed",
                outcome="partial" if stats["ran"] else "failure",
                error_class="run_failed",
                error_detail=f"{stats['failed']} of {stats['fresh']} runs failed",
                metrics=stats,
            )
        return RoutineResult(
            status="success",
            outcome="success" if stats["ran"] else "success_empty",
            metrics=stats,
        )


class AutomationScheduleHandler:
    kind = "automation_schedule"

    async def execute(self, routine, run, db: AsyncSession) -> RoutineResult:
        automation = await _automation_for_routine(db, routine)
        early = _not_runnable(automation)
        if early is not None:
            return early
        fire_instant = getattr(run, "fire_instant", None) \
            or getattr(run, "created_at", None)
        fire_key = (
            fire_instant.strftime("%Y%m%dT%H%M%S")
            if fire_instant is not None else "manual"
        )
        try:
            vspec = await parse_spec_live(automation)
            status = await executor.run_schedule_fire(
                db, automation, vspec, fire_key,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[automations] schedule fire failed %s: %s",
                           automation.id, e)
            await executor._record_health(
                db, automation.id, ok=False, error=str(e)[:500],
            )
            return RoutineResult(
                status="failed", error_class="fire_failed",
                error_detail=str(e)[:300],
            )
        if status == "failed":
            return RoutineResult(
                status="failed", error_class="run_failed",
                error_detail=automation.last_error or "run failed",
            )
        return RoutineResult(status="success", outcome="success")


def register_automation_handlers() -> None:
    """Idempotent; called from routines/__init__ behind the flag."""
    from app.agent.routines.registry import register_handler
    register_handler(AutomationPollHandler())
    register_handler(AutomationScheduleHandler())


# ── Push (trigger) side ──────────────────────────────────────────────


async def handle_push_events(
    trigger, kept_emails: list, db: AsyncSession,
) -> dict[str, str]:
    """The `run_automation` trigger action. `kept_emails` are the
    email handler's fetched, filter-passed `_FetchedEmail`s. Returns
    per-event statuses keyed by the trigger-side event id."""
    from app.config import settings
    out: dict[str, str] = {}
    if not getattr(settings, "automations_enabled", False):
        # A stray row on a flag-off tenant: logged no-op, never a fire.
        logger.info("[automations] run_automation trigger %s skipped: "
                    "flag off", trigger.id)
        return {getattr(fe, "event_id", "?"): "skipped_filter"
                for fe in kept_emails}

    automation_id = (trigger.config_json or {}).get("automation_id")
    automation = await db.get(Automation, automation_id) if automation_id else None
    if automation is None or automation.status != "armed":
        logger.info("[automations] run_automation trigger %s: automation "
                    "%s not armed", trigger.id, automation_id)
        return {getattr(fe, "event_id", "?"): "skipped_filter"
                for fe in kept_emails}

    vspec = await parse_spec_live(automation)
    items = []
    for fe in kept_emails:
        headers = fe.headers or {}
        items.append({
            "gmail_message_id": fe.gmail_id,
            "message_id": fe.gmail_id,
            "subject": headers.get("subject") or headers.get("Subject") or "",
            "from": headers.get("from") or headers.get("From") or "",
            "snippet": fe.snippet or "",
            "_trigger_event_id": fe.event_id,
        })

    by_key = {i["gmail_message_id"]: i["_trigger_event_id"] for i in items}
    fresh = await executor.ingest_items(db, automation, vspec, items)
    fresh_keys = {e.dedupe_key for e in fresh}
    for gmail_id, ev_id in by_key.items():
        if gmail_id not in fresh_keys:
            out[ev_id] = "coalesced"

    for event in fresh:
        status = await executor.run_event(db, automation, vspec, event)
        ev_id = by_key.get(event.dedupe_key, event.id)
        out[ev_id] = "success" if status == "run" else (
            "skipped_filter" if status == "skipped_filter" else "failed"
        )
    return out
