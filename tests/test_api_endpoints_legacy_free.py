"""API endpoints are off the legacy tables — pin for PR #51.

The 9-PR unified-jobs arc (PRs #31→#42) plus the cutover arc
(PRs #45→#50) moved every read and write off ``trigger_events`` and
``routine_runs``. PR #50 verified the three runner/intake files
(``app/agent/triggers/runner.py``, ``app/agent/routines/runner.py``,
``app/api/triggers_inbound.py``) were clean. PR #51 cuts the
remaining API endpoints — the LIST + FORCE-RUN paths in
``app/api/triggers.py`` and ``app/api/routines.py`` — off the same
tables so PR #52 (mig 050) can finally drop them.

What this file pins:

  1. ``GET /api/triggers/<id>/events`` returns events sourced from
     ``build_jobs`` (no TriggerEvent rows seeded — the response
     must still be non-empty).
  2. ``POST /api/triggers/<id>/test`` mints a ``build_jobs`` row via
     ``JobRunner.create_job`` (no TriggerEvent insert).
  3. ``GET /api/routines/<id>/runs`` returns runs sourced from
     ``build_jobs`` (no RoutineRun rows seeded).
  4. ``POST /api/routines/<id>/run`` mints a ``build_jobs`` row
     via ``JobRunner.create_job`` through ``_fire``.
  5. Source-grep regression pin: every file under ``backend/app/api/``
     is free of code-level references to ``TriggerEvent`` or
     ``RoutineRun`` (docstrings/comments are fine — they document
     the migration). Same triple-quote state machine as
     ``test_no_legacy_table_references.py``.
"""
from __future__ import annotations

import os
import re
import uuid
from datetime import date as date_cls, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-api-endpoints-legacy-free")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000abe51")


USER_ID = "00000000-0000-0000-0000-0000000abe51"


# ── Schema setup ──────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Override conftest's autouse fixture. Build ONLY the tables these
    tests touch — no TriggerEvent / RoutineRun creation needed, which
    is exactly the point of the cutover. Same pattern as
    ``test_panels_recent_jobs.py``.
    """
    from app.config import settings
    from app.db.database import rebind_database

    await rebind_database(settings.DATABASE_URL)

    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, Routine, Trigger, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent, Trigger, Routine):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (JobEvent, BuildJob, Trigger, Routine, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)


@pytest_asyncio.fixture
async def _seed_user(monkeypatch):
    # Pin ``settings.user_id`` to this file's USER_ID. The big batched
    # pytest invocation in ``.github/workflows/test-backend.yml`` runs
    # several test files together, and whichever file imports first
    # wins the ``os.environ.setdefault("USER_ID", ...)`` race —
    # ``settings`` is a module-level singleton instantiated once via
    # ``@lru_cache``, so by the time this file's fixtures run
    # ``settings.user_id`` is some sibling file's id, not ours. The
    # owner-scoped lookup in ``app/api/triggers.py`` and
    # ``app/api/routines.py`` (``row.user_id != _user_id()``) then
    # 404s. Pinning per-test keeps each file's USER_ID local without
    # touching the production lookup contract.
    from app.config import settings
    monkeypatch.setattr(settings, "user_id", USER_ID)
    # email_briefing routines are gated by this flag (default False); the
    # routines API ``_kind_enabled_or_404`` raises a 404 "Feature not
    # available" before the BuildJob lookup runs. Tests in this file use
    # ``kind="email_briefing"`` because the cutover code-paths are
    # kind-agnostic but the seed routine has to be a real kind.
    monkeypatch.setattr(settings, "routines_email_briefing_enabled", True)

    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"legacy-free-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
            await db.commit()


# ── Helpers ───────────────────────────────────────────────────────────


async def _make_trigger(name: str = "Inbox") -> str:
    from app.db.database import async_session_maker
    from app.db.models import Trigger

    tid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Trigger(
            id=tid,
            user_id=USER_ID,
            kind="email_received",
            action="summarize_and_post",
            name=name,
            enabled=True,
            filter_json={},
            config_json={},
            provider_state_json={},
            last_status="never_fired",
            fire_count=0,
        ))
        await db.commit()
    return tid


async def _make_routine(name: str = "Morning briefing") -> str:
    from app.db.database import async_session_maker
    from app.db.models import Routine

    rid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Routine(
            id=rid,
            user_id=USER_ID,
            kind="email_briefing",
            name=name,
            enabled=True,
            schedule_cron_local="0 7 * * *",
            schedule_kind="cron",
            config_json={},
            last_state_json={},
            last_status="never_run",
        ))
        await db.commit()
    return rid


async def _add_trigger_job(
    trigger_id: str,
    *,
    title: str,
    status: str = "completed",
    outcome: str | None = "success",
    error_message: str | None = None,
    summary_message_id: str | None = None,
    created_at_offset_minutes: int = 0,
) -> str:
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    jid = str(uuid.uuid4())
    now = datetime.utcnow() + timedelta(minutes=created_at_offset_minutes)
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=jid,
            user_id=USER_ID,
            title=title,
            prompt="",
            job_type="trigger_run",
            status=status,
            outcome=outcome,
            error_message=error_message,
            summary_message_id=summary_message_id,
            source_kind="trigger",
            source_id=trigger_id,
            # idempotency_key is the legacy event_dedupe_id for trigger fires.
            idempotency_key=f"msg-{jid[:8]}",
            created_at=now,
            completed_at=now if status in ("completed", "failed") else None,
        ))
        await db.commit()
    return jid


async def _add_routine_job(
    routine_id: str,
    *,
    local_date: date_cls,
    status: str = "completed",
    outcome: str | None = "success",
    emails_fetched: int = 3,
    attempt: int = 1,
) -> str:
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    jid = str(uuid.uuid4())
    now = datetime.utcnow()
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=jid,
            user_id=USER_ID,
            title=f"Routine fire: email_briefing {local_date}",
            prompt="",
            job_type="routine_run",
            status=status,
            outcome=outcome,
            source_kind="routine",
            source_id=routine_id,
            # idempotency_key is the local date ISO string for routine fires.
            idempotency_key=local_date.isoformat(),
            emails_fetched=emails_fetched,
            attempt=attempt,
            created_at=now,
            completed_at=now if status in ("completed", "failed") else None,
        ))
        await db.commit()
    return jid


# ──────────────────────────────────────────────────────────────────────
# 1. GET /api/triggers/<id>/events — sourced from BuildJob
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_events_returns_jobs_only(_seed_user):
    """``GET /api/triggers/<id>/events`` must return BuildJob rows
    projected into the legacy TriggerEventResponse shape — without any
    TriggerEvent rows in the DB.

    Seeds two BuildJobs (one success, one failure) and asserts both
    surface, the (status, outcome) collapse maps back to the legacy
    enum, and the error_detail carries through from
    BuildJob.error_message.
    """
    from app.api.triggers import list_events

    tid = await _make_trigger(name="Watch inbox")
    j1 = await _add_trigger_job(
        tid, title="Trigger fire OK",
        status="completed", outcome="success",
        summary_message_id="msg-out-1",
        created_at_offset_minutes=2,
    )
    j2 = await _add_trigger_job(
        tid, title="Trigger fire boom",
        status="failed", outcome=None,
        error_message="handler_crash: ValueError(boom)",
        created_at_offset_minutes=1,
    )

    events = await list_events(tid, limit=50, offset=0)
    assert len(events) == 2, (
        f"expected 2 event rows from BuildJob, got {len(events)}"
    )
    by_id = {e.id: e for e in events}

    ok = by_id[j1]
    assert ok.trigger_id == tid
    assert ok.status == "success", (
        f"completed+success on BuildJob must collapse back to legacy "
        f"status='success', got {ok.status!r}"
    )
    assert ok.summary_message_id == "msg-out-1"

    boom = by_id[j2]
    assert boom.status == "failed"
    assert boom.error_detail == "handler_crash: ValueError(boom)"
    # error_class has no separate column on BuildJob; the legacy field
    # is preserved as None (PR #51 intentional — see _job_to_event_response).
    assert boom.error_class is None


@pytest.mark.asyncio
async def test_list_events_unknown_trigger_returns_404(_seed_user):
    """Unknown trigger_id → 404. Same auth gate as before; just the
    backing query changed."""
    from fastapi import HTTPException
    from app.api.triggers import list_events

    with pytest.raises(HTTPException) as ei:
        await list_events(str(uuid.uuid4()), limit=50, offset=0)
    assert ei.value.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# 2. POST /api/triggers/<id>/test — mints BuildJob via JobRunner
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_test_trigger_creates_build_job_not_trigger_event(_seed_user):
    """``POST /api/triggers/<id>/test`` must create one ``build_jobs``
    row (not a TriggerEvent). The row carries ``source_kind='trigger'``,
    ``source_id=trigger_id``, ``idempotency_key=test:<uuid>``, and
    ``job_type='trigger_run'``."""
    from sqlalchemy import select
    from app.api.triggers import test_trigger
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    tid = await _make_trigger(name="Test fire")
    resp = await test_trigger(tid)

    # Response shape is the legacy TriggerEventResponse, but the source
    # row is a BuildJob. event_dedupe_id is the idempotency_key.
    assert resp.trigger_id == tid
    assert resp.event_dedupe_id.startswith("test:"), (
        f"test fire should use 'test:' prefix, got {resp.event_dedupe_id!r}"
    )

    # Verify the row landed in build_jobs.
    async with async_session_maker() as db:
        jobs = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == tid)
        )).scalars().all()
    assert len(jobs) == 1, (
        f"expected exactly 1 BuildJob row, got {len(jobs)}"
    )
    job = jobs[0]
    assert job.source_kind == "trigger"
    assert job.job_type == "trigger_run"
    assert job.idempotency_key == resp.event_dedupe_id


# ──────────────────────────────────────────────────────────────────────
# 3. GET /api/routines/<id>/runs — sourced from BuildJob
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_runs_returns_jobs_only(_seed_user):
    """``GET /api/routines/<id>/runs`` must return BuildJob rows
    projected into the legacy RoutineRunResponse shape — without any
    RoutineRun rows in the DB. The richer terminal column set
    (emails_fetched, attempt, etc.) reads back through the response."""
    from app.api.routines import list_runs

    rid = await _make_routine(name="Briefing")
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    await _add_routine_job(
        rid, local_date=today,
        status="completed", outcome="success",
        emails_fetched=5, attempt=1,
    )
    await _add_routine_job(
        rid, local_date=yesterday,
        status="failed", outcome=None,
        emails_fetched=0, attempt=2,
    )

    runs = await list_runs(rid, limit=30, offset=0)
    assert len(runs) == 2

    # Status enum collapses back from (status, outcome) → legacy.
    statuses = {r.status for r in runs}
    assert statuses == {"success", "failed"}, (
        f"expected legacy status set, got {statuses}"
    )

    # Terminal columns (mig 052) round-trip through the response.
    success_run = next(r for r in runs if r.status == "success")
    assert success_run.emails_fetched == 5
    assert success_run.attempt == 1

    # scheduled_for_local_date is parsed back from the idempotency_key.
    dates_in_resp = {r.scheduled_for_local_date for r in runs}
    assert dates_in_resp == {today, yesterday}


@pytest.mark.asyncio
async def test_list_runs_unknown_routine_returns_404(_seed_user):
    from fastapi import HTTPException
    from app.api.routines import list_runs

    with pytest.raises(HTTPException) as ei:
        await list_runs(str(uuid.uuid4()), limit=30, offset=0)
    assert ei.value.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# 4. POST /api/routines/<id>/run — mints BuildJob via JobRunner / _fire
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_force_run_returns_409_when_today_already_succeeded(_seed_user):
    """The pre-fire idempotency check reads BuildJob (not RoutineRun).
    Seed a completed/success Job for today, then assert force_run
    returns 409 — not because we wrote to routine_runs but because the
    BuildJob lookup finds the row.

    We avoid touching the actual runner (which would require an MCP
    stack) by relying on the 409 short-circuit BEFORE _fire is
    invoked."""
    from fastapi import HTTPException
    from app.api.routines import force_run

    rid = await _make_routine(name="Briefing")
    today = datetime.utcnow().date()
    await _add_routine_job(
        rid, local_date=today,
        status="completed", outcome="success",
    )

    with pytest.raises(HTTPException) as ei:
        await force_run(rid)
    assert ei.value.status_code == 409
    # The legacy status enum surfaces in the detail for the UI message.
    detail = ei.value.detail
    assert detail.get("status") == "success", (
        f"expected legacy status='success' in 409 detail, got {detail}"
    )


@pytest.mark.asyncio
async def test_force_run_deletes_failed_prior_then_fires(_seed_user, monkeypatch):
    """A failed prior run for today is dropped (BuildJob delete, NOT
    RoutineRun delete), then ``_runner._fire`` is invoked. We patch
    ``_runner`` with a stub so we can pin the call without a real
    APScheduler / MCP stack."""
    from sqlalchemy import select
    from app.api import routines as routines_api
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    rid = await _make_routine(name="Briefing")
    today = datetime.utcnow().date()
    # Failed prior run — recoverable, so force_run drops it.
    await _add_routine_job(
        rid, local_date=today,
        status="failed", outcome=None,
    )

    fired_for: list[str] = []

    class _StubRunner:
        async def _fire(self, routine_id: str) -> None:
            fired_for.append(routine_id)
            # Simulate _fire claiming the BuildJob row.
            await _add_routine_job(
                rid, local_date=today,
                status="running", outcome=None,
            )

    monkeypatch.setattr(routines_api, "_runner", _StubRunner())

    resp = await routines_api.force_run(rid)
    # _fire was invoked exactly once on the right routine.
    assert fired_for == [rid]
    # Response is the BuildJob-backed shape; status from the running row.
    assert resp.status == "running"

    # The failed prior row was deleted; only the running row remains.
    async with async_session_maker() as db:
        jobs = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == rid)
        )).scalars().all()
    assert len(jobs) == 1
    assert jobs[0].status == "running"


# ──────────────────────────────────────────────────────────────────────
# 5. Source-grep regression pin for the API tree
# ──────────────────────────────────────────────────────────────────────
#
# Copy of the triple-quote state machine in
# ``test_no_legacy_table_references.py`` — extended to scan every file
# in ``app/api/`` rather than the three hand-listed runner files. After
# this PR ships, PR #52 drops the legacy tables; a stray re-add under
# app/api/ would crash at runtime.


_BANNED_SYMBOLS = ("TriggerEvent", "RoutineRun")


def _code_only_lines(text: str) -> list[tuple[int, str]]:
    """Yield (1-based lineno, source) for lines NOT inside a triple-
    quoted block and NOT pure comments. Inline-comment tails are
    stripped — only the code prefix on each line is returned. Same
    heuristic as ``test_no_legacy_table_references._code_only_lines``;
    duplicated here to keep the two pins independent (a regression
    in the shared helper shouldn't quietly disable both pins)."""
    out: list[tuple[int, str]] = []
    in_block: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw
        while line:
            if in_block is None:
                tq_pos = -1
                tq_kind: str | None = None
                for q in ('"""', "'''"):
                    p = line.find(q)
                    if p >= 0 and (tq_pos < 0 or p < tq_pos):
                        tq_pos = p
                        tq_kind = q
                hash_pos = line.find("#")
                if tq_pos >= 0 and (hash_pos < 0 or tq_pos < hash_pos):
                    prefix = line[:tq_pos]
                    if prefix.strip():
                        out.append((lineno, prefix))
                    rest = line[tq_pos + 3:]
                    close_pos = rest.find(tq_kind)  # type: ignore[arg-type]
                    if close_pos >= 0:
                        line = rest[close_pos + 3:]
                        continue
                    in_block = tq_kind
                    line = ""
                else:
                    if hash_pos >= 0:
                        line = line[:hash_pos]
                    if line.strip():
                        out.append((lineno, line))
                    line = ""
            else:
                close_pos = line.find(in_block)
                if close_pos >= 0:
                    in_block = None
                    line = line[close_pos + 3:]
                else:
                    line = ""
    return out


def _find_banned_references(file_path: Path) -> list[tuple[int, str, str]]:
    text = file_path.read_text()
    code_lines = _code_only_lines(text)
    out: list[tuple[int, str, str]] = []
    # Word-boundary so ``TriggerEventResponse`` (the Pydantic class
    # name on the API surface, deliberately kept for wire-compat) does
    # NOT match the banned ``TriggerEvent`` ORM symbol. Same for
    # ``RoutineRunResponse`` and ``RoutineRunner``. The legacy ORM
    # classes have no trailing suffix.
    suffix_pattern = re.compile(r"\w")
    for symbol in _BANNED_SYMBOLS:
        word = re.compile(rf"\b{re.escape(symbol)}\b")
        for lineno, code in code_lines:
            for match in word.finditer(code):
                end = match.end()
                # Word boundary is supposed to handle this, but
                # ``\b`` doesn't distinguish ``TriggerEvent`` from
                # ``TriggerEventResponse`` because there's no
                # non-word char between them. Manually check: the
                # next char must NOT be a word char.
                if end < len(code) and suffix_pattern.match(code[end]):
                    continue
                out.append((lineno, symbol, code.rstrip()))
    return out


def test_app_api_tree_has_no_legacy_orm_references():
    """Every Python file under ``backend/app/api/`` must contain zero
    code-level references to the legacy ``TriggerEvent`` or
    ``RoutineRun`` ORM classes. PR #51 cutover — the runner-side pin
    is in ``test_no_legacy_table_references.py``; this one covers the
    API tree.

    ``TriggerEventResponse``, ``TriggerRecentJob``,
    ``RoutineRunResponse``, ``RoutineRecentJob``, and ``RoutineRunner``
    are NOT banned — those are response models / class names that
    share a prefix with the legacy ORM symbols. The word-boundary
    suffix check rejects identifiers extended by a word char.
    """
    api_dir = Path(os.getcwd()) / "app" / "api"
    assert api_dir.is_dir(), f"api dir not found at {api_dir}"

    failures: list[str] = []
    for py in sorted(api_dir.rglob("*.py")):
        hits = _find_banned_references(py)
        if hits:
            failures.append(
                f"{py.relative_to(api_dir.parent.parent)}: " + "; ".join(
                    f"L{ln} {sym} → {code.strip()[:80]}"
                    for (ln, sym, code) in hits
                )
            )

    assert not failures, (
        "Banned ORM references found in app/api/ after PR #51 cutover:\n"
        + "\n".join("  " + f for f in failures)
    )
