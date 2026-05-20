"""Pagination + input-validation edges for ``GET /apps/jobs/events``.

The base happy-path test (``test_jobs_events_endpoint.py``) pins:
attribution shape, ``before`` round-trip, user scoping. Missing
coverage:

  1. Boundary: ``len(rows) == limit`` exactly — ``next_before``
     must be NULL (we just consumed the last page). The endpoint's
     limit+1 fetch trick disambiguates this; pin it.
  2. ``limit=1`` honors strict ts-DESC ordering across many rows.
  3. Empty feed: no rows at all. ``next_before`` is NULL, events
     is ``[]``.
  4. Invalid ``before`` format (not ISO 8601) is tolerated — the
     endpoint logs a warning and treats it as if absent, rather
     than returning a 4xx (so a malformed cookie / stale URL
     doesn't crash the dashboard).
  5. Same-ts events ordered by ``ts DESC`` are insertion-stable
     enough for the frontend to not de-dupe falsely. (Same-second
     events can collide; we just need the page to be deterministic
     in some order.)
  6. The ``before`` filter is STRICT inequality (``<``), not
     ``<=``. Passing a previous page's last-event ts must NOT
     re-include that same event in the next page.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-edges")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000edg01")


USER_ID = "00000000-0000-0000-0000-0000000edg01"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    from app.db.database import rebind_database
    from app.config import settings

    await rebind_database(settings.DATABASE_URL)

    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (JobEvent, BuildJob, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)


@pytest_asyncio.fixture
async def _seed_user_and_job():
    """Seed User + one BuildJob row. Events are added per-test."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, User

    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"edges-{USER_ID[:8]}@example.com",
                hashed_password="x", timezone="UTC",
            ))
        db.add(BuildJob(
            id=job_id, user_id=USER_ID, title="Edges test",
            prompt="", job_type="agent_task", status="running",
            source_kind="agent_task", source_id=job_id,
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return {"user_id": USER_ID, "job_id": job_id}


async def _seed_events(job_id: str, count: int, base_ts: datetime) -> list[str]:
    """Insert ``count`` events with monotonically increasing ts
    (1 second apart, so the order is unambiguous). Returns the list
    of event IDs in the order they were created (oldest → newest)."""
    from app.db.database import async_session_maker
    from app.db.models import JobEvent

    ids: list[str] = []
    async with async_session_maker() as db:
        for i in range(count):
            ev_id = str(uuid.uuid4())
            ids.append(ev_id)
            db.add(JobEvent(
                id=ev_id,
                job_id=job_id,
                user_id=USER_ID,
                ts=base_ts + timedelta(seconds=i),
                kind="info",
                label=f"event-{i}",
                level="info",
            ))
        await db.commit()
    return ids


async def _call(limit: int = 50, before: str | None = None) -> dict:
    from app.api.jobs_events import list_job_events
    page = await list_job_events(limit=limit, before=before)
    return page.model_dump()


# ──────────────────────────────────────────────────────────────────────
# 1. Exact-limit boundary: next_before is NULL when we just consumed
#    the last page.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_next_before_null_when_total_equals_limit(_seed_user_and_job):
    """Seed exactly 5 events, fetch with limit=5. The page contains
    all 5; ``next_before`` MUST be NULL (no more pages). The endpoint
    uses ``limit+1`` internally to disambiguate this — regressing to
    a plain ``limit`` query would return next_before=<ts of oldest>
    and the frontend would infinite-paginate against an empty next
    page."""
    base_ts = datetime(2026, 5, 19, 12, 0, 0)
    await _seed_events(_seed_user_and_job["job_id"], 5, base_ts)

    page = await _call(limit=5)
    assert len(page["events"]) == 5
    assert page["next_before"] is None, (
        "len(rows) == limit but no more rows behind — next_before "
        "must be NULL. Regression would infinite-paginate."
    )


@pytest.mark.asyncio
async def test_next_before_set_when_more_rows_behind(_seed_user_and_job):
    """Seed 7 events, fetch with limit=5. Page has 5 events;
    next_before is set to the oldest event's ts (so the next page
    fetches strictly older)."""
    base_ts = datetime(2026, 5, 19, 12, 0, 0)
    await _seed_events(_seed_user_and_job["job_id"], 7, base_ts)

    page = await _call(limit=5)
    assert len(page["events"]) == 5
    assert page["next_before"] is not None
    # The oldest event in the page is the 3rd event (events ordered
    # DESC: 6, 5, 4, 3, 2). next_before == ts of event-2.
    assert page["events"][-1]["ts"] == page["next_before"]


# ──────────────────────────────────────────────────────────────────────
# 2. before= filter is strict (<), not <= — no double-fetch of boundary
#    event.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_before_is_strict_inequality(_seed_user_and_job):
    """Fetch page 1, then pass page 1's ``next_before`` as the
    ``before`` for page 2. Page 2 MUST NOT re-include the event
    whose ts equals ``next_before`` — that would double-render it
    on the dashboard."""
    base_ts = datetime(2026, 5, 19, 12, 0, 0)
    await _seed_events(_seed_user_and_job["job_id"], 10, base_ts)

    page1 = await _call(limit=5)
    boundary_ts = page1["next_before"]
    page1_ids = {ev["id"] for ev in page1["events"]}

    page2 = await _call(limit=5, before=boundary_ts)
    page2_ids = {ev["id"] for ev in page2["events"]}
    overlap = page1_ids & page2_ids
    assert not overlap, (
        f"before= must be strict inequality; got {len(overlap)} "
        f"duplicate event(s) across pages: {overlap}"
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Empty feed.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_feed_returns_empty_page(_seed_user_and_job):
    """No events at all. Page must return ``events=[], next_before=None``
    without raising."""
    page = await _call(limit=50)
    assert page["events"] == []
    assert page["next_before"] is None


# ──────────────────────────────────────────────────────────────────────
# 4. Invalid ``before`` — tolerated, NOT a 4xx.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_malformed_before_falls_back_to_no_filter(_seed_user_and_job):
    """A stale URL with ``?before=banana`` shouldn't break the
    dashboard. The endpoint logs a warning and treats it as if
    absent (same response as the no-filter call). Pin so a future
    "fail closed" change doesn't break in-flight users."""
    base_ts = datetime(2026, 5, 19, 12, 0, 0)
    await _seed_events(_seed_user_and_job["job_id"], 3, base_ts)

    page_clean = await _call(limit=50)
    page_malformed = await _call(limit=50, before="not-an-iso-timestamp")
    assert len(page_malformed["events"]) == len(page_clean["events"])


@pytest.mark.asyncio
async def test_before_tolerates_z_and_offset_suffixes(_seed_user_and_job):
    """``datetime.fromisoformat`` only learned ``Z`` parsing in
    3.11. The endpoint replaces ``Z`` with ``+00:00`` before
    parsing so the dashboard can round-trip what the response
    serialised. Pin both forms."""
    base_ts = datetime(2026, 5, 19, 12, 0, 0)
    await _seed_events(_seed_user_and_job["job_id"], 3, base_ts)

    # base_ts + 1s in two forms
    z_form = (base_ts + timedelta(seconds=1)).isoformat() + "Z"
    offset_form = (base_ts + timedelta(seconds=1)).isoformat() + "+00:00"

    page_z = await _call(limit=50, before=z_form)
    page_offset = await _call(limit=50, before=offset_form)
    # Both should filter to events strictly older than base_ts+1s,
    # so just event-0 (at base_ts).
    assert len(page_z["events"]) == 1
    assert len(page_offset["events"]) == 1


# ──────────────────────────────────────────────────────────────────────
# 5. ts DESC ordering with single-row pages walks the full history.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_limit_1_walk_iterates_in_strict_ts_desc(_seed_user_and_job):
    """Hammer the smallest-page case: limit=1 should walk every row
    in strict ts-DESC order across n pages. Verifies the (limit+1
    sentinel, before strict-<) interaction holds at the smallest
    page size."""
    base_ts = datetime(2026, 5, 19, 12, 0, 0)
    seeded_ids = await _seed_events(_seed_user_and_job["job_id"], 5, base_ts)
    expected_walk = list(reversed(seeded_ids))  # newest → oldest

    walked: list[str] = []
    before: str | None = None
    while True:
        page = await _call(limit=1, before=before)
        if not page["events"]:
            break
        walked.append(page["events"][0]["id"])
        if page["next_before"] is None:
            break
        before = page["next_before"]
        # Defensive cap so a regression doesn't hang the test.
        if len(walked) > 10:
            break

    assert walked == expected_walk, (
        f"limit=1 walk should iterate every row in ts-DESC order; "
        f"got {walked}, expected {expected_walk}"
    )


# ──────────────────────────────────────────────────────────────────────
# 6. limit=200 (the max) still works.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_limit_200_works(_seed_user_and_job):
    """The FastAPI Query bound is ``le=200``. A page of exactly 200
    must work (not be off-by-one rejected). Seed 250 events, fetch
    200, assert ``next_before`` is set (50 more behind)."""
    base_ts = datetime(2026, 5, 19, 12, 0, 0)
    await _seed_events(_seed_user_and_job["job_id"], 250, base_ts)

    page = await _call(limit=200)
    assert len(page["events"]) == 200
    assert page["next_before"] is not None, (
        "limit=200 with 250 rows behind must set next_before"
    )
