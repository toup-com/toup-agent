"""Activity feed — server-side query over ``job_events`` JOIN ``build_jobs``.

PR 6 of the unified-jobs arc. Replaces the dashboard's client-side
flatten of ``BuildJob.steps_json`` (the source of the "everything is
Nokia Snake Arcade" attribution bug, see
``docs/architecture/jobs-investigation-2026-05-18.md`` §1.1).

The endpoint:

  ``GET /apps/jobs/events?limit=50&before=<iso-ts>``

returns a forward-paged list of recent ``job_events`` rows for the
current user, JOINed to ``build_jobs`` for attribution. The
frontend renders each event as one row in the activity feed; the
``job_title`` field is what shows on the right side ("Nokia Snake
Arcade" today, but now correctly attributed per-source).

Read paths:
  - Auto Builder phases (PR 5)         → kind='phase_started'/'phase_completed'
  - JobLogger info/tool/edit/error
    (PR 3)                              → kind='info'/'tool_call'/'error'
  - Future: output_posted (PR 8)        → kind='output_posted'

The endpoint is read-only and best-effort: a missing job_events
row simply yields no event; a missing build_jobs row yields a
NULL job_title (rare race — orphaned events from a deleted job).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import BuildJob, JobEvent


logger = logging.getLogger(__name__)
router = APIRouter()


def _get_user_id() -> str:
    """Mirror of ``apps.py._get_user_id`` — single-tenant agent
    mode resolves the caller from ``settings.user_id``."""
    from app.config import settings
    return settings.user_id


class JobEventResponse(BaseModel):
    """One activity-feed entry. Frontend renders these flat
    (no nesting). All fields except ``ts``/``kind``/``user_id``
    are nullable to tolerate row variations across event kinds
    (e.g. an ``error`` event has no ``status``)."""

    id: str
    job_id: str
    job_title: Optional[str] = None
    job_type: Optional[str] = None
    source_kind: Optional[str] = None
    ts: str  # ISO 8601
    kind: str
    label: Optional[str] = None
    status: Optional[str] = None
    level: str = "info"


class JobEventsPage(BaseModel):
    events: list[JobEventResponse]
    # When non-NULL, the caller can pass it as ``?before=`` to
    # fetch the next page back in time. NULL when this page is
    # the oldest available.
    next_before: Optional[str] = None


@router.get("/events", response_model=JobEventsPage)
async def list_job_events(
    limit: int = Query(50, ge=1, le=200),
    before: Optional[str] = Query(
        None,
        description="ISO timestamp; return events with ts strictly older than this. "
                    "Use for forward pagination — pass the previous page's next_before.",
    ),
) -> JobEventsPage:
    """Server-side activity feed.

    JOINs ``job_events`` to ``build_jobs`` so attribution
    (``job_title``, ``job_type``, ``source_kind``) lands per-event
    instead of being computed in the browser. Filtered to the
    container's owning user. Ordered by ``ts DESC`` (most recent
    first). Capped at ``limit`` rows.

    ``before`` is exclusive (``ts < before``) so the natural
    pagination contract is: caller fetches a page, takes the
    oldest event's ``ts``, passes it back as ``before`` to get
    the next page. ``next_before`` in the response is the page's
    oldest ``ts`` (or NULL when this page returned fewer than
    ``limit`` rows — i.e. it's the last page).
    """
    user_id = _get_user_id()

    before_dt: Optional[datetime] = None
    if before:
        try:
            # Tolerate both "...Z" and "...+00:00" suffixes.
            before_dt = datetime.fromisoformat(before.replace("Z", "+00:00"))
            # Stored ts in DB is naive UTC (mirrors the rest of
            # the codebase's datetime convention). Strip tzinfo
            # so the comparison doesn't error on mixed tz.
            if before_dt.tzinfo is not None:
                from datetime import timezone as _tz
                before_dt = before_dt.astimezone(_tz.utc).replace(tzinfo=None)
        except ValueError:
            logger.warning("[jobs_events] invalid before= value: %r", before)
            before_dt = None

    async with async_session_maker() as db:
        # Fetch limit+1 rows so we can tell "this is a full page and
        # there's more" apart from "this is a full page and we just
        # happened to land exactly on the boundary". The +1 row is
        # trimmed before returning; its existence sets next_before.
        stmt = (
            select(JobEvent, BuildJob)
            .outerjoin(BuildJob, JobEvent.job_id == BuildJob.id)
            .where(JobEvent.user_id == user_id)
            .order_by(JobEvent.ts.desc())
            .limit(limit + 1)
        )
        if before_dt is not None:
            stmt = stmt.where(JobEvent.ts < before_dt)
        rows = (await db.execute(stmt)).all()

    has_more = len(rows) > limit
    rows = rows[:limit]

    events: list[JobEventResponse] = []
    for ev, job in rows:
        events.append(JobEventResponse(
            id=ev.id,
            job_id=ev.job_id,
            job_title=getattr(job, "title", None) if job else None,
            job_type=getattr(job, "job_type", None) if job else None,
            source_kind=getattr(job, "source_kind", None) if job else None,
            ts=(ev.ts.isoformat() if isinstance(ev.ts, datetime) else str(ev.ts)),
            kind=ev.kind,
            label=ev.label,
            status=ev.status,
            level=ev.level or "info",
        ))

    next_before: Optional[str] = events[-1].ts if (has_more and events) else None
    return JobEventsPage(events=events, next_before=next_before)
