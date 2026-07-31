"""Stalled-job reaper — no zombie progress bars.

Inline chat jobs (create_job / update_job) are advanced by the agent's
conversation turn. When a turn ends with the job still 'running' and
nothing owns it afterwards (no Autopilot mission, no sub-agent), the
row sits at its last percentage forever and every progress surface
lies — the in-chat job card, the floating capsule, and the phone's
Live Activity. This sweep fails such jobs honestly and tells the
phone, so a dead bar becomes an explicit "didn't finish".

A job is stalled when status ∈ (queued, running) AND its last sign of
life — newest ``job_events.ts``, else ``created_at`` — is older than
``STALE_AFTER``. Exemptions:
- ``paused_at`` set: token-limit pauses legitimately sleep for hours.
- Sub-agent jobs get their own hard timeout + boot orphan sweep
  (subagent_orchestrator); this net only catches them after a
  mid-flight crash that survived a restart.

Registered in agent_main next to the notify-outbox loop (agent-side
only — build_jobs is a tenant table).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, select

logger = logging.getLogger(__name__)

STALE_AFTER = timedelta(minutes=30)
SWEEP_INTERVAL_S = 600
_BOOT_DELAY_S = 120


async def sweep_stalled_jobs(now: Optional[datetime] = None) -> int:
    """Fail every stalled job; returns how many were reaped."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, JobEvent

    from app.agent.job_status import STATUS_CANCELLED, turn_interrupted

    now = now or datetime.utcnow()
    cutoff = now - STALE_AFTER
    stale_minutes = int(STALE_AFTER.total_seconds() // 60)

    async with async_session_maker() as db:
        result = await db.execute(
            select(BuildJob).where(
                BuildJob.status.in_(("queued", "running")),
                BuildJob.created_at < cutoff,
                BuildJob.paused_at.is_(None),
            )
        )
        jobs = list(result.scalars().all())
        if not jobs:
            return 0

        last_event_ts = dict(
            (
                await db.execute(
                    select(JobEvent.job_id, func.max(JobEvent.ts))
                    .where(JobEvent.job_id.in_([j.id for j in jobs]))
                    .group_by(JobEvent.job_id)
                )
            ).all()
        )

        reaped: list[tuple[str, str, str]] = []
        for job in jobs:
            last_alive = last_event_ts.get(job.id) or job.created_at
            if last_alive and last_alive >= cutoff:
                continue
            # `cancelled`, not `failed`. The dominant cause of a stall is a
            # turn that went away mid-flight — the voice relay cancels its SSE
            # generator the moment the caller hangs up — and the agent has
            # usually already delivered the answer. Calling that "Failed" is
            # the exact lie this pipeline exists to remove; the work stopped,
            # it did not break.
            verdict = turn_interrupted()
            job.status = STATUS_CANCELLED
            job.error_class = verdict.error_class
            job.user_message = verdict.user_message
            # Kept verbatim for operators; never serialized to a client.
            job.technical_detail = (
                f"Stalled: no progress for {stale_minutes} minutes — "
                "reaped so progress surfaces stay honest."
            )
            job.completed_at = now
            reaped.append((job.id, job.title or "", job.user_id))
        if not reaped:
            return 0
        await db.commit()

    for job_id, title, user_id in reaped:
        logger.warning(
            "[job_reaper] failed stalled job %s (%s)", job_id[:8], title[:60]
        )
        # In-app surfaces: a status-only frame makes clients refetch
        # (shapeless-step contract in the mobile JobsContext).
        try:
            from app.api.ws_chat import broadcast_to_user

            await broadcast_to_user(user_id, {
                "type": "job_update",
                "job_id": job_id,
                "name": title,
                "status": STATUS_CANCELLED,
            })
        except Exception:  # noqa: BLE001 — reaping must never crash
            pass
        # Phone surface: end the Live Activity card honestly.
        try:
            from app.agent.subagent_orchestrator import _notify_job_event

            await _notify_job_event(
                job_id=job_id,
                label=title,
                # `mission_failed` is the only terminal lane besides
                # `mission_completed` (KNOWN_NOTIFY_KINDS is a closed enum
                # validated at ingest), and ONLY a terminal notification closes
                # a Live Activity card — a DB write does nothing to it. So the
                # kind stays; the copy stops shouting failure at work the agent
                # very likely already delivered.
                kind="mission_failed",
                title=f"Stopped: {(title or 'background task')[:150]}",
                body=(
                    "This stopped before it finished. Ask me to pick it up "
                    "again."
                ),
                dismiss_after_s=900,
                dedup_suffix="stalled",
                urgent=False,
            )
        except Exception:  # noqa: BLE001
            pass
    return len(reaped)


async def stalled_jobs_sweep_loop() -> None:
    """Forever loop for agent_main. Boot delay lets init_db / pool bind
    restore settle before the first tenant-DB query."""
    await asyncio.sleep(_BOOT_DELAY_S)
    while True:
        try:
            n = await sweep_stalled_jobs()
            if n:
                logger.info("[job_reaper] reaped %d stalled job(s)", n)
        except Exception as e:  # noqa: BLE001 — pre-bind lobby / DB hiccup
            logger.warning("[job_reaper] sweep failed: %s", e)
        await asyncio.sleep(SWEEP_INTERVAL_S)
