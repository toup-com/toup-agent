"""Recovery of jobs orphaned by an agent crash / restart / blue-green roll.

Extracted from ``agent_main.py``'s lifespan so it can be unit-tested: this
is boot-critical logic that decides, for every in-flight job, whether the
user sees a failure. It ran untested inline for months.

Why this exists
---------------
The original sweep flipped EVERY ``running`` *and* ``queued`` row to
``failed`` with the literal message "Agent restarted during execution".
Audited 2026-07-29 against the founder's live tenant: that string was
**7 of 11** user-visible failures (64%) — the single largest source of
"Failed" jobs in production, and not one of them was a real failure. Agent
containers rebuild on every push touching ``backend/**``, so it fired
roughly once per working day.

The contract now (docs/audits/mission-control-design.md §1.4):

``queued``   LEAVE ALONE. It never started; there is nothing to recover.
             The old code killed these too, which was pure loss.
``running``  RE-QUEUE where a drain loop will actually pick the row back
             up; otherwise terminalise as INTERRUPTED — visible, honest
             and retryable — rather than stranding it.
parked /
terminal     untouched.

Two things make the ``source_kind`` split load-bearing rather than
cosmetic:

1. Only ``TriggerRunner`` polls for queued rows (``_fetch_queued_ids``
   filters ``source_kind == 'trigger'``). Every other dispatch path is
   in-process and push-based, so re-queueing a routine / subagent /
   agent_task row parks it in ``queued`` FOREVER.
2. A Live Activity card is closed ONLY by a terminal notification — a DB
   write never touches it. So a stranded row also leaves its lock screen /
   Dynamic Island card lingering on stale progress for hours. That is
   strictly worse than the bug being fixed.

Blue-green safety: the incoming container boots bound to the SAME tenant
database while the outgoing one is still draining live jobs, so
re-queueing a *fresh* ``running`` row would double-execute it. Only rows
older than ``REQUEUE_GRACE`` are touched; anything newer belongs to the
still-draining sibling and is left for it to finish (or for
``job_reaper``'s stalled-job sweep to catch later).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import select

from app.agent.job_status import (
    DISPOSITION_REQUEUE,
    ERR_INFRA_UNRECOVERABLE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    restart_verdict,
)

logger = logging.getLogger(__name__)

#: Rows younger than this are assumed live in a still-draining sibling
#: container and are left completely alone.
REQUEUE_GRACE = timedelta(minutes=15)

#: After this many restart re-queues a job is presumed to be what is
#: killing the container. Stop looping and surface it honestly.
MAX_RESTART_REQUEUES = 3

#: Key inside ``BuildJob.config_json`` holding the re-queue counter.
#: Stored there rather than in ``attempt`` because ``attempt`` is the
#: routine-retry counter and conflating the two would corrupt both.
REQUEUE_COUNT_KEY = "restart_requeues"


@dataclass
class RecoveryOutcome:
    """What the sweep did, so the caller can notify and log accurately."""

    requeued: int = 0
    #: (job_id, title, user_id) — each MUST get a terminal notification so
    #: its Live Activity card is closed.
    interrupted: list[tuple[str, str, str]] = field(default_factory=list)
    gave_up: list[tuple[str, str, str]] = field(default_factory=list)
    skipped_in_grace: int = 0
    #: App builds handed to ``build_watchdog.settle_build`` (round 27).
    #: Deliberately NOT in ``interrupted``: that list drives the caller's
    #: generic terminal notifier, and the settle has already sent this row's
    #: own frame and Live Activity push with the build's real step counts.
    settled_builds: list[str] = field(default_factory=list)

    @property
    def touched(self) -> int:
        return (self.requeued + len(self.interrupted) + len(self.gave_up)
                + len(self.settled_builds))


def _fail_row(job: Any, *, error_class: str, user_message: str,
              technical_detail: str, now: datetime) -> None:
    job.status = STATUS_FAILED
    job.error_class = error_class
    job.user_message = user_message
    job.technical_detail = technical_detail
    job.completed_at = now


def _rewind_running_steps(job: Any) -> None:
    """Roll half-finished steps back to pending.

    Without this a re-queued job reports a phantom in-flight step forever
    — the founder's job stuck reading "2/5" for 79 days.
    """
    try:
        steps = json.loads(job.steps_json) if job.steps_json else []
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(steps, list):
        return
    changed = False
    for s in steps:
        if isinstance(s, dict) and s.get("status") == "running":
            s["status"] = "pending"
            s.pop("started_at", None)
            changed = True
    if changed:
        job.steps_json = json.dumps(steps)


async def recover_orphaned_jobs(
    session_maker: Any,
    *,
    now: Optional[datetime] = None,
    grace: timedelta = REQUEUE_GRACE,
    max_requeues: int = MAX_RESTART_REQUEUES,
) -> RecoveryOutcome:
    """Reconcile in-flight jobs after a restart. Returns what changed.

    Commits at most once. Never raises on a single bad row — a malformed
    job must not stop the container from booting.
    """
    from app.db.models import App as AppModel, BuildJob

    now = now or datetime.utcnow()
    out = RecoveryOutcome()

    builds: list[str] = []
    async with session_maker() as db:
        # `queued` is deliberately NOT in this filter.
        rows = (await db.execute(
            select(BuildJob).where(BuildJob.status == STATUS_RUNNING)
        )).scalars().all()

        for job in rows:
            try:
                if job.created_at and (now - job.created_at) < grace:
                    out.skipped_in_grace += 1
                    continue

                # ── App builds take the build-shaped close ──────────
                # Round 27. `_fail_row` writes a status and never touches
                # `steps_json`, so a build orphaned by a restart came back
                # with `look` still spinning — and both clients render a
                # build card off its STEPS. `settle_build` resolves the
                # rows, marks the publish that never happened, and
                # broadcasts a terminal frame that carries them. Run after
                # this session commits, because it opens its own.
                if (getattr(job, "job_type", None) == "auto_builder"
                        and getattr(job, "app_id", None)):
                    builds.append(job.id)
                    continue

                source_kind = getattr(job, "source_kind", None)
                verdict = restart_verdict(source_kind)

                if verdict.disposition != DISPOSITION_REQUEUE:
                    _fail_row(
                        job,
                        error_class=verdict.error_class,
                        user_message=verdict.user_message or "",
                        technical_detail=(
                            f"orphaned by agent restart; source_kind="
                            f"{source_kind!r} has no queued-row drain"
                        ),
                        now=now,
                    )
                    out.interrupted.append((job.id, job.title or "", job.user_id))
                    await _mark_app_errored(db, AppModel, job)
                    continue

                cfg = dict(job.config_json or {})
                tries = int(cfg.get(REQUEUE_COUNT_KEY) or 0)

                if tries >= max_requeues:
                    _fail_row(
                        job,
                        error_class=ERR_INFRA_UNRECOVERABLE,
                        user_message=(
                            "This task couldn't be completed after several "
                            "attempts. We've been notified."
                        ),
                        technical_detail=(
                            f"orphaned by restart {tries}x; exceeded "
                            f"max_requeues={max_requeues}"
                        ),
                        now=now,
                    )
                    out.gave_up.append((job.id, job.title or "", job.user_id))
                    await _mark_app_errored(db, AppModel, job)
                    logger.warning(
                        "[job_recovery] %s exceeded restart re-queue cap — failing",
                        job.id[:8],
                    )
                    continue

                # Re-queue. Invisible by design: no error fields, no
                # completed_at, and (see module docstring) no notification.
                cfg[REQUEUE_COUNT_KEY] = tries + 1
                job.config_json = cfg
                job.status = STATUS_QUEUED
                job.error_message = None
                job.error_class = None
                job.user_message = None
                job.completed_at = None
                _rewind_running_steps(job)

                out.requeued += 1
                logger.info(
                    "[job_recovery] re-queued %s (%s) attempt %d/%d",
                    job.id[:8], (job.title or "")[:40], tries + 1, max_requeues,
                )
            except Exception:  # noqa: BLE001
                # One unreadable row must never block a container boot.
                logger.exception(
                    "[job_recovery] skipped malformed job %s", getattr(job, "id", "?"),
                )

        if out.touched:
            await db.commit()

    for job_id in builds:
        try:
            from app.agent.build_watchdog import settle_build
            if await settle_build(job_id, now=now, reason="restart"):
                out.settled_builds.append(job_id)
        except Exception:  # noqa: BLE001 — a boot must never fail on a sweep
            logger.exception("[job_recovery] build settle failed for %s", job_id)

    return out


async def _mark_app_errored(db: Any, app_model: Any, job: Any) -> None:
    """A build that died mid-flight leaves its app stuck on 'building'."""
    if not getattr(job, "app_id", None):
        return
    app = await db.get(app_model, job.app_id)
    if app is not None and app.status == "building":
        app.status = "error"
