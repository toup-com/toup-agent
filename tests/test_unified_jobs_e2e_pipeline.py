"""End-to-end pipeline tests for the unified-jobs arc.

The per-PR tests cover each layer in isolation (JobRunner.create_job,
JobLogger, status mirrors, the endpoint). What was missing: a single
test that exercises the *full* path — intake call →
``JobRunner.create_job`` →  ``JobLogger`` entries → ``job_events``
rows → ``/api/apps/jobs/events`` response — and asserts that
attribution survives every hop.

What we pin here:

  1. Each ``source_kind`` (trigger / routine / agent_task /
     auto_builder) drives the same pipeline and produces a feed
     entry with the right ``job_title``, ``job_type``,
     ``source_kind`` triple. No cross-contamination.

  2. The "no more Nokia Snake Arcade everywhere" regression
     check at integration scale: 4 different jobs of 4 different
     source_kinds, each producing 1+ events, all visible in the
     feed under distinct attribution.

  3. ``JobLogger.write`` events appear in chronological order
     in the feed (ts DESC) and carry their ``kind`` / ``label`` /
     ``level`` exactly as written.

  4. Per-user scoping holds at the endpoint level: a second
     user's events are excluded even though they share the same
     in-memory DB.

These are *integration* tests — they call the real intake helpers
(``JobRunner.create_job``, the real ``JobLogger``), the real
endpoint function, and assert against real DB state. Nothing is
mocked.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-e2e-unified-jobs")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000e2e01")


USER_ID = "00000000-0000-0000-0000-0000000e2e01"
OTHER_USER_ID = "00000000-0000-0000-0000-0000000e2e02"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Per-test fresh DB. We rebind the engine to the current event
    loop (same StaticPool + aiosqlite pattern as the other arc
    tests) and create only the tables the pipeline touches."""
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
async def _seed_user():
    """Seed the test user. Other users are referenced by uuid string
    only — SQLite doesn't enforce FKs, and seeding extra Users hits
    the User.is_canary partial-UNIQUE that SQLite can't represent."""
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"e2e-{USER_ID[:8]}@example.com",
                hashed_password="x",
                timezone="UTC",
            ))
            await db.commit()
    return USER_ID


# ──────────────────────────────────────────────────────────────────────
# Helpers — drive the real intake + logger + feed.
# ──────────────────────────────────────────────────────────────────────


async def _create_job_for_source(
    source_kind: str,
    *,
    title: str,
    source_id: str,
    idempotency_key: str | None = None,
    user_id: str = USER_ID,
):
    """Drive ``JobRunner.create_job`` with the same kwargs the real
    intake paths use. Returns the freshly-created BuildJob row id."""
    from app.agent.job_runner import JobRunner, TaskSpec

    runner = JobRunner()
    job_type = {
        "trigger": "trigger_run",
        "routine": "routine_run",
        "agent_task": "agent_task",
        "auto_builder": "auto_builder",
    }[source_kind]
    spec = TaskSpec(
        user_id=user_id,
        channel="web",
        source_kind=source_kind,
        source_id=source_id,
        prompt=f"pipeline-test prompt for {source_kind}",
        config_json=None,
        conversation_id=None,
    )
    # JobRunner.create_job is kwarg-only after self.
    job = await runner.create_job(
        job_type=job_type,
        spec=spec,
        title=title,
        idempotency_key=idempotency_key,
    )
    return job


async def _log_via_real_logger(
    job_id: str,
    *,
    level: str = "info",
    message: str = "step",
    user_id: str = USER_ID,
) -> None:
    """Drive the real ``JobLogger._log`` dual-write so we know
    ``job_events`` is populated by production code, not by a test
    shortcut. JobLogger maps level → event kind:
       info        → info
       tool / edit → tool_call
       error       → error
    """
    from app.agent.job_logger import JobLogger
    job_logger = JobLogger(job_id=job_id, user_id=user_id)
    await job_logger._log(level=level, message=message)


async def _call_feed(limit: int = 50, before: str | None = None) -> dict:
    """Call the real endpoint route function directly — same pattern
    as test_jobs_events_endpoint.py. No HTTP/ASGI ceremony."""
    from app.api.jobs_events import list_job_events
    page = await list_job_events(limit=limit, before=before)
    return page.model_dump()


# ──────────────────────────────────────────────────────────────────────
# 1. Each source_kind drives the pipeline end-to-end.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", [
    "trigger", "routine", "agent_task", "auto_builder",
])
async def test_pipeline_e2e_for_each_source_kind(_seed_user, source_kind):
    """For every intake surface: create_job + log + feed = one
    event with the right attribution. This is the smoke test for
    the whole unified-jobs arc — if any of these breaks, something
    along the pipeline regressed."""
    title = f"E2E {source_kind} job"
    source_id = str(uuid.uuid4())

    job = await _create_job_for_source(
        source_kind, title=title, source_id=source_id,
    )
    await _log_via_real_logger(job.id, level="info", message="started")

    feed = await _call_feed(limit=10)
    assert len(feed["events"]) == 1, (
        f"expected exactly 1 event in feed for {source_kind}; "
        f"got {len(feed['events'])}"
    )
    ev = feed["events"][0]
    assert ev["job_id"] == job.id
    assert ev["job_title"] == title
    assert ev["source_kind"] == source_kind
    assert ev["kind"] == "info"
    assert ev["label"] == "started"


# ──────────────────────────────────────────────────────────────────────
# 2. Cross-source feed: 4 source_kinds × 1 job each = 4 distinct rows
#    with no attribution cross-contamination.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cross_source_feed_no_attribution_cross_contamination(_seed_user):
    """The pre-arc bug: every event in the feed got attributed to
    "Nokia Snake Arcade" (the most recent BuildJob.title) because
    the dashboard flattened all of build_jobs.steps_json into one
    list. Post-arc: each event carries its own job_id linkage and
    the feed JOIN attributes per-row.

    Drive 4 jobs of 4 source_kinds, write 2 events on each (so the
    feed has 8 rows), assert every event maps to the right job."""
    sources = [
        ("trigger", "Gmail watcher"),
        ("routine", "Morning briefing"),
        ("agent_task", "Slack summary task"),
        ("auto_builder", "Nokia Snake Arcade"),
    ]
    job_ids: dict[str, str] = {}
    for kind, title in sources:
        job = await _create_job_for_source(
            kind, title=title, source_id=str(uuid.uuid4()),
        )
        job_ids[kind] = job.id
        await _log_via_real_logger(job.id, level="info", message="start")
        await _log_via_real_logger(job.id, level="info", message="finish")

    feed = await _call_feed(limit=50)
    assert len(feed["events"]) == 8, (
        f"4 jobs × 2 events each = 8; got {len(feed['events'])}"
    )

    # Group events by source_kind and assert each maps to the right
    # title — the regression check.
    by_source: dict[str, set[str]] = {}
    for ev in feed["events"]:
        by_source.setdefault(ev["source_kind"], set()).add(ev["job_title"])

    for kind, title in sources:
        assert kind in by_source, f"feed missing source_kind={kind}"
        assert by_source[kind] == {title}, (
            f"source_kind={kind} should map ONLY to {title!r}; "
            f"got {by_source[kind]}. This is the Nokia Snake Arcade "
            "attribution bug regressing."
        )


# ──────────────────────────────────────────────────────────────────────
# 3. Chronological ordering — feed is ts DESC across event hops.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_events_in_feed_are_ts_desc(_seed_user):
    """Activity-feed UX contract: most-recent event first. Drive
    three events on one job with deliberate gaps and assert the
    feed reorders them DESC."""
    job = await _create_job_for_source(
        "agent_task", title="Sequencing", source_id=str(uuid.uuid4()),
    )

    # JobLogger stamps ts at write time; tiny sleeps create the
    # ordering. We use real DB writes so the test exercises the
    # ts column's default behaviour, not a manual timestamp.
    import asyncio
    await _log_via_real_logger(job.id, level="info", message="first")
    await asyncio.sleep(0.01)
    await _log_via_real_logger(job.id, level="info", message="second")
    await asyncio.sleep(0.01)
    await _log_via_real_logger(job.id, level="info", message="third")

    feed = await _call_feed(limit=10)
    labels = [ev["label"] for ev in feed["events"]]
    assert labels == ["third", "second", "first"], (
        f"feed must order events ts DESC; got {labels}"
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Per-user scoping at integration level.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_other_users_events_excluded_from_feed(_seed_user):
    """Single-tenant agent: ``settings.user_id`` resolves the
    caller. An event written under a different user_id must NOT
    appear in the feed even if it shares the same DB."""
    # Mine — should appear.
    job_mine = await _create_job_for_source(
        "trigger", title="Mine", source_id=str(uuid.uuid4()),
    )
    await _log_via_real_logger(job_mine.id, level="info", message="mine")

    # Other user's — should NOT appear.
    job_other = await _create_job_for_source(
        "trigger", title="Other",
        source_id=str(uuid.uuid4()),
        user_id=OTHER_USER_ID,
    )
    await _log_via_real_logger(
        job_other.id, level="info", message="other",
        user_id=OTHER_USER_ID,
    )

    feed = await _call_feed(limit=50)
    titles = {ev["job_title"] for ev in feed["events"]}
    assert "Mine" in titles
    assert "Other" not in titles, (
        "events for OTHER_USER_ID leaked into the current user's feed"
    )


# ──────────────────────────────────────────────────────────────────────
# 5. JobLogger kind variants surface intact.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_joblogger_kind_variants_surface_in_feed(_seed_user):
    """JobLogger normalises its caller-side ``kind`` to one of
    {info, tool_call, error}. Verify each lands in the feed with
    the right ``level`` (info | error) and ``kind`` (info | tool_call
    | error). A regression here would crash the frontend filter
    pills."""
    job = await _create_job_for_source(
        "agent_task", title="Kind variants", source_id=str(uuid.uuid4()),
    )
    await _log_via_real_logger(job.id, level="info", message="info_label")
    await _log_via_real_logger(job.id, level="tool", message="tool_label")
    await _log_via_real_logger(job.id, level="edit", message="edit_label")
    await _log_via_real_logger(job.id, level="error", message="error_label")

    feed = await _call_feed(limit=10)
    by_label = {ev["label"]: ev for ev in feed["events"]}

    assert by_label["info_label"]["kind"] == "info"
    assert by_label["info_label"]["level"] == "info"

    assert by_label["tool_label"]["kind"] == "tool_call"
    assert by_label["edit_label"]["kind"] == "tool_call", (
        "JobLogger maps 'edit' → 'tool_call' the same as 'tool'"
    )

    assert by_label["error_label"]["kind"] == "error"
    assert by_label["error_label"]["level"] == "error", (
        "error kind must promote level=error so frontend filter "
        "pills work"
    )
