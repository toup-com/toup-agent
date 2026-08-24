"""Round 27 — a build card cannot outlive its build.

The recorded defect, verbatim: **"Build: Habit Garden · In progress · 4/7
steps · 57%"**, sitting in a chat for hours. That number is not decoration,
it is the shape of the failure and every test here is built on it — four
rows `done` (`create`, `verify`, and the `review`/`edit` pair one repair
pass appended), `look` still `running`, `logo` and `present` still
`pending`, and **no failed row anywhere**. The build died inside the
looks-right loop.

Three mechanisms each declined to resolve it and each for its own reason,
so there are three pins:

* **Scope.** `reconcile_delivered_turn_jobs` selects
  `job_type == 'agent_task'`. A build is `auto_builder`, so the minute-loop
  watchdog whose entire purpose is killing zombie progress bars had the
  build lane out of scope. Every caller of it now sweeps builds too.

* **Shape.** `job_reaper` writes a status and never touches `steps_json`,
  and broadcasts a status-ONLY frame. **Both** build cards are rendered off
  their STEPS — the phone's `JobProgressCard` derives `hasFailed`,
  `isComplete` and the "N/M steps" line from the step list and never reads
  `job.status` at all — so a terminal status under live steps is a card
  that goes on saying "In progress". `test_the_settled_card_reads_terminal_
  to_a_steps_only_client` is the client's own state machine, transcribed.

* **Word.** The settle used to write `cancelled`, which is precisely the
  value neither client treats as terminal: the phone's `JobsContext` drops
  a card on `completed`/`failed` only, so the row stayed in the live map
  forever *and* polling stopped. A build that did not publish now closes
  `failed` — a fact about what happened, not an invented diagnosis.

* **The other branch.** A build that HAD published and then died still ends
  with `skipped` rows (a `logo` that blew its budget), so `steps.every(done)`
  is false and the phone would read "In progress · 5/7 steps" over a live
  app. Its completeness test is `appReady || steps.every(done)`, so the
  settle replays `app_ready` — `test_a_published_build_still_announces_the_
  app` is the pin, and the fixture deliberately leaves `logo` unfinished.

Round 23's actual invariant is re-asserted here rather than relaxed: **no
phase may be invented done by a watchdog**, unreported rows are `skipped`
with their honest words, and a row that was ever done keeps its result.
The single row whose meaning changed is `present` — the phase whose absence
IS the failure.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

os.environ.setdefault("AGENT_API_KEY", "test-key-r27-zombie-builds")

USER_ID = "00000000-0000-0000-0000-00000000ff27"


@pytest_asyncio.fixture(autouse=True)
async def _tables():
    """Only the tables these paths touch — the same pattern as
    test_r23_step_semantics / test_r24_steps_registry, which is what keeps
    this file runnable on a box where the full conftest init is broken."""
    from app.db.database import engine
    from app.db.models import App, BuildJob, JobEvent, User

    async with engine.begin() as conn:
        for model_cls in (User, App, BuildJob, JobEvent):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (JobEvent, BuildJob, App, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)
    await engine.dispose()


# ── The recorded card, exactly ────────────────────────────────────────

def habit_garden_steps() -> list:
    """4 done of 7, no failed row, `look` spinning. The recorded shape.

    Order is the order `emit_step` produces: `review`/`edit` are not in the
    up-front plan and are inserted after everything that has started, so a
    repair pass leaves them between `look` and the untouched tail.
    """
    def row(t, status, was_done=False):
        s = {"id": str(uuid.uuid4()), "type": t, "status": status,
             "label": f"{t}-{status}", "rev": 1}
        if was_done:
            s["was_done"] = True
        return s

    return [
        row("create", "done", was_done=True),
        row("verify", "done", was_done=True),
        row("look", "running"),          # died here, in the looks-right loop
        row("review", "done", was_done=True),
        row("edit", "done", was_done=True),
        row("logo", "pending"),
        row("present", "pending"),
    ]


#: `app_id=None` must mean "no app", not "give me a fresh one".
_AUTO = object()


async def _seed_build(*, steps=None, status="running", age_minutes=90,
                      job_type="auto_builder", app_id=_AUTO,
                      last_event_minutes=None) -> str:
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, JobEvent

    job_id = str(uuid.uuid4())
    created = datetime.utcnow() - timedelta(minutes=age_minutes)
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=USER_ID, title="Build: Habit Garden",
            prompt="Single-file HTML app: Habit Garden",
            job_type=job_type, status=status,
            app_id=str(uuid.uuid4()) if app_id is _AUTO else app_id,
            conversation_id="conv-r27",
            steps_json=json.dumps(
                habit_garden_steps() if steps is None else steps),
            created_at=created,
        ))
        if last_event_minutes is not None:
            db.add(JobEvent(
                job_id=job_id, user_id=USER_ID, kind="phase_started",
                level="info", status="running", label="Looking at the app",
                ts=datetime.utcnow() - timedelta(minutes=last_event_minutes),
            ))
        await db.commit()
    return job_id


async def _row(job_id: str):
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        return json.loads(job.steps_json), job.status, job


def _by_type(steps) -> dict:
    return {s["type"]: s for s in steps}


def _mute_surfaces(monkeypatch):
    """Silence the two best-effort announcers. They are exercised on their
    own in `test_the_terminal_frame_carries_the_steps`; everywhere else a
    missing ws_chat / notify stack must not be what makes a settle look
    like it worked."""
    import app.agent.build_watchdog as bw

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(bw, "announce_settled", _noop)


# ── 1. Scope: the sweep can see the build lane at all ─────────────────

@pytest.mark.asyncio
async def test_the_reconciler_sweeps_app_builds(monkeypatch):
    """The scope bug, from the entry point the reaper and boot both use.

    `reconcile_delivered_turn_jobs` filtered `job_type == 'agent_task'`, so
    this row — the recorded card — was never a candidate for the watchdog
    that runs every 60 seconds.
    """
    from app.agent.job_reconciler import reconcile_delivered_turn_jobs

    _mute_surfaces(monkeypatch)
    job_id = await _seed_build()
    await reconcile_delivered_turn_jobs()

    _steps, status, _job = await _row(job_id)
    assert status == "failed", "the build lane is still out of the sweep's scope"


@pytest.mark.asyncio
async def test_a_fresh_build_is_left_alone(monkeypatch):
    """The window is a real threshold, not a formality: a build that emitted
    a phase two minutes ago is WORKING, and killing it would be worse than
    the zombie."""
    from app.agent.build_watchdog import sweep_stuck_builds

    _mute_surfaces(monkeypatch)
    fresh = await _seed_build(age_minutes=90, last_event_minutes=2)
    assert await sweep_stuck_builds() == 0
    _steps, status, _ = await _row(fresh)
    assert status == "running"


@pytest.mark.asyncio
async def test_a_paused_build_is_left_alone(monkeypatch):
    """The token-limit pause legitimately sleeps for hours with a
    checkpoint. `paused_at` is the one exemption every sweep honours."""
    from app.agent.build_watchdog import sweep_stuck_builds
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    _mute_surfaces(monkeypatch)
    job_id = await _seed_build()
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        job.paused_at = datetime.utcnow()
        await db.commit()

    assert await sweep_stuck_builds() == 0
    _steps, status, _ = await _row(job_id)
    assert status == "running"


# ── 2. Shape: what a settled build's rows look like ───────────────────

@pytest.mark.asyncio
async def test_the_settle_marks_the_publish_that_never_happened(monkeypatch):
    from app.agent.build_watchdog import settle_build

    _mute_surfaces(monkeypatch)
    job_id = await _seed_build()
    settled = await settle_build(job_id, reason="watchdog")
    assert settled is not None and settled.status == "failed"
    assert settled.published is False

    steps, status, job = await _row(job_id)
    rows = _by_type(steps)
    assert status == "failed"
    assert rows["present"]["status"] == "failed"
    assert rows["present"]["label"] == "Couldn't publish the app"
    assert job.completed_at is not None
    assert job.user_message, "a failed build must carry copy a client may render"


@pytest.mark.asyncio
async def test_no_phase_is_invented_done_by_a_watchdog(monkeypatch):
    """Round 23's invariant, unchanged. `look` never reported back and
    `logo` never started; neither becomes a tick."""
    from app.agent.build_watchdog import settle_build

    _mute_surfaces(monkeypatch)
    job_id = await _seed_build()
    await settle_build(job_id, reason="watchdog")

    steps, _status, _ = await _row(job_id)
    rows = _by_type(steps)
    assert rows["look"]["status"] == "skipped"
    assert rows["logo"]["status"] == "skipped"
    assert rows["look"].get("detail"), "a skipped phase says why"


@pytest.mark.asyncio
async def test_the_settle_never_decreases_the_done_count(monkeypatch):
    """Round 24/25's invariant. The card read 4 of 7 the moment before it
    died and it must not close reading 3."""
    from app.agent.build_watchdog import settle_build
    from app.agent.skills.builtins.app_html.steps import step_counts

    _mute_surfaces(monkeypatch)
    before, _ = step_counts(habit_garden_steps())
    assert before == 4, "the fixture is not the recorded card any more"

    job_id = await _seed_build()
    settled = await settle_build(job_id, reason="watchdog")
    assert settled.done >= before


@pytest.mark.asyncio
async def test_a_published_build_settles_completed_not_failed(monkeypatch):
    """Robust to partial completion. The publish reported done and then the
    process died before `finish_job` — the app IS live, and closing it
    `failed` would be the mirror-image lie of the one this round removes."""
    from app.agent.build_watchdog import settle_build

    _mute_surfaces(monkeypatch)
    steps = habit_garden_steps()
    for s in steps:
        if s["type"] in ("look", "logo", "present"):
            s["status"] = "done"
            s["was_done"] = True
    job_id = await _seed_build(steps=steps)

    settled = await settle_build(job_id, reason="watchdog")
    assert settled.published is True
    assert settled.status == "completed"
    assert settled.percent == 100
    _steps, status, _ = await _row(job_id)
    assert status == "completed"


@pytest.mark.asyncio
async def test_a_failed_publish_keeps_its_own_diagnosis(monkeypatch):
    """A publish that reported its OWN failure keeps that row verbatim —
    overwriting it would replace a real reason with a generic one."""
    from app.agent.build_watchdog import settle_build

    _mute_surfaces(monkeypatch)
    steps = habit_garden_steps()
    rows = _by_type(steps)
    rows["present"]["status"] = "failed"
    rows["present"]["detail"] = "the workspace refused the write"
    job_id = await _seed_build(steps=steps)

    await settle_build(job_id, reason="watchdog")
    after, status, _ = await _row(job_id)
    assert status == "failed"
    assert _by_type(after)["present"]["detail"] == "the workspace refused the write"


# ── 3. Word: what the CLIENTS do with the settled row ─────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("published", [False, True])
async def test_the_settled_card_reads_terminal_to_a_steps_only_client(
    monkeypatch, published,
):
    """The phone card's state machine, transcribed from
    ``src/shared/JobProgressCard.tsx``::

        hasFailed  = steps.some(s => s.status === 'failed')
        isComplete = steps.every(s => s.status === 'done')
        label      = hasFailed  ? "Didn't finish"
                   : isComplete ? "App built"
                   :              `In progress · ${done}/${total} steps`

    It never reads `job.status`. So "the row is terminal" is not the
    property that matters — "the STEPS are terminal" is, and it is the one
    every close path was failing to provide. Both outcomes must land on a
    word; neither may fall through to "In progress".
    """
    from app.agent.build_watchdog import settle_build

    _mute_surfaces(monkeypatch)
    steps = habit_garden_steps()
    if published:
        for s in steps:
            if s["type"] in ("look", "present"):
                s["status"] = "done"
                s["was_done"] = True
    job_id = await _seed_build(steps=steps)
    await settle_build(job_id, reason="watchdog")

    after, _status, _ = await _row(job_id)
    has_failed = any(s["status"] == "failed" for s in after)
    is_complete = bool(after) and all(s["status"] == "done" for s in after)
    assert has_failed or is_complete or published, (
        "the settled card still reads 'In progress' on a steps-only client"
    )
    assert has_failed is not published


def _capture_frames():
    """Swap `app.api.ws_chat` for a recorder and hand back the list."""
    import sys
    from types import SimpleNamespace

    frames: list = []

    async def _bcast(_user_id, payload):
        frames.append(payload)

    sys.modules["app.api.ws_chat"] = SimpleNamespace(broadcast_to_user=_bcast)
    return frames


@pytest.mark.asyncio
async def test_a_published_build_still_announces_the_app(monkeypatch):
    """The other branch of the same zombie, and the reason the parametrised
    test above lets `published` off the steps check.

    A real build legitimately ends with rows that are `skipped` — a `logo`
    that blew its 45s budget, a `look` the renderer could not run. The phone
    card's `isComplete` is `appReady || steps.every(done)`, so a build that
    HAD published and then died before `finish_job` would settle `completed`
    in the database and still read "In progress · 5/7 steps" on the phone.

    `app_ready` (behind `app_artifact`) is what the happy path sends after
    `finish_job` and what puts a client into "App built" directly. The
    settle replays it, and the App row stops saying `building`.
    """
    import sys

    import app.agent.build_watchdog as bw
    from app.db.database import async_session_maker
    from app.db.models import App as AppModel

    steps = habit_garden_steps()
    for s in steps:
        if s["type"] in ("look", "present"):
            s["status"] = "done"
            s["was_done"] = True
    # `logo` stays PENDING — it becomes `skipped`, which is what breaks
    # `steps.every(done)` and is the whole point of this test.
    app_id = str(uuid.uuid4())
    job_id = await _seed_build(steps=steps, app_id=app_id)
    async with async_session_maker() as db:
        db.add(AppModel(id=app_id, user_id=USER_ID, name="Habit Garden",
                        slug="habit-garden", status="building",
                        app_dir="/tmp/apps/habit-garden"))
        await db.commit()

    frames = _capture_frames()
    try:
        settled = await bw.settle_build(job_id, reason="watchdog")
    finally:
        sys.modules.pop("app.api.ws_chat", None)

    assert settled.status == "completed" and settled.published
    after, _status, _ = await _row(job_id)
    assert _by_type(after)["logo"]["status"] == "skipped", \
        "the fixture no longer exercises the skipped-row case"
    assert not any(s["status"] == "done" for s in after
                   if s["type"] == "logo")

    kinds = [f.get("type") for f in frames]
    assert "app_ready" in kinds, (
        "a published build that settles without app_ready reads "
        "'In progress' on the phone forever"
    )
    ready = next(f for f in frames if f.get("type") == "app_ready")
    assert ready["app_id"] == app_id and ready["slug"] == "habit-garden"

    async with async_session_maker() as db:
        assert (await db.get(AppModel, app_id)).status == "ready"


@pytest.mark.asyncio
async def test_the_settle_survives_an_absent_apps_table(monkeypatch):
    """`apps` is AGENT_ONLY, and the settle must not depend on it.

    Caught by CI, not by me: the first cut read the App row INSIDE the
    settle's own transaction, so under `RUN_MODE=platform` — where
    `init_db()` does not build AGENT_ONLY tables — `no such table: apps`
    landed on the settle's path, the blanket `except` swallowed it, and
    `settle_build` returned None. The card stayed exactly as stuck as
    before, with a log line politely saying so. Two suites went red in the
    platform sweep and green in the agent one, which is the tell.

    A cosmetic row on a second table may not veto the close.
    """
    import app.agent.build_watchdog as bw
    from app.db.database import engine
    from app.db.models import App as AppModel

    _mute_surfaces(monkeypatch)
    job_id = await _seed_build()
    async with engine.begin() as conn:
        await conn.run_sync(AppModel.__table__.drop, checkfirst=True)

    settled = await bw.settle_build(job_id, reason="watchdog")
    assert settled is not None and settled.status == "failed"
    assert settled.slug is None
    _steps, status, _ = await _row(job_id)
    assert status == "failed"


@pytest.mark.asyncio
async def test_an_unpublished_build_marks_its_app_errored(monkeypatch):
    """The mirror: an app left on `building` by a dead build is a spinner
    one surface over, in the library rather than the chat."""
    import app.agent.build_watchdog as bw
    from app.db.database import async_session_maker
    from app.db.models import App as AppModel

    _mute_surfaces(monkeypatch)
    app_id = str(uuid.uuid4())
    job_id = await _seed_build(app_id=app_id)
    async with async_session_maker() as db:
        db.add(AppModel(id=app_id, user_id=USER_ID, name="Habit Garden",
                        slug="habit-garden-2", status="building",
                        app_dir="/tmp/apps/habit-garden-2"))
        await db.commit()

    await bw.settle_build(job_id, reason="watchdog")
    async with async_session_maker() as db:
        assert (await db.get(AppModel, app_id)).status == "error"


@pytest.mark.asyncio
async def test_a_build_never_settles_to_cancelled(monkeypatch):
    """`cancelled` is the one word neither client acts on: the phone's
    `JobsContext` drops a card from the live map on `completed`/`failed`
    only, so a cancelled build stayed pinned in the chat AND stopped being
    polled. It is gone from the build lane — check every close path, not
    just the watchdog's."""
    from app.agent.build_watchdog import settle_build
    from app.agent.job_reconciler import close_job_completed
    from app.db.database import async_session_maker

    _mute_surfaces(monkeypatch)
    for reason in ("watchdog", "reaper", "turn_interrupted", "restart"):
        job_id = await _seed_build()
        await settle_build(job_id, reason=reason)
        _s, status, _ = await _row(job_id)
        assert status != "cancelled", reason

    # …and through the shared turn-end finalizer, which routes builds here.
    job_id = await _seed_build()
    async with async_session_maker() as db:
        closed = await close_job_completed(
            db, job_id, user_id=USER_ID, now=datetime.utcnow(),
            reason="turn_end",
        )
        await db.commit()
    assert closed is None, "a build must never come back as a ClosedJob"
    _s, status, _ = await _row(job_id)
    assert status == "failed"


@pytest.mark.asyncio
async def test_the_terminal_frame_carries_the_steps():
    """The stall reaper's frame is `{job_id, name, status}` and nothing
    else. A client that renders a build off its steps learns nothing from
    that — which is why the reaper has been "resolving" these rows for
    weeks with the card unchanged. The terminal frame carries the list."""
    import app.agent.build_watchdog as bw

    frames: list = []

    class _FakeWs:
        @staticmethod
        async def broadcast_to_user(user_id, payload):
            frames.append(payload)

    import sys
    from types import SimpleNamespace
    sys.modules["app.api.ws_chat"] = SimpleNamespace(
        broadcast_to_user=_FakeWs.broadcast_to_user,
    )
    try:
        job_id = await _seed_build()
        await bw.settle_build(job_id, reason="watchdog")
    finally:
        sys.modules.pop("app.api.ws_chat", None)

    assert frames, "no terminal frame reached the transcript"
    frame = frames[-1]
    assert frame["type"] == "job_update"
    assert frame["status"] == "failed"
    assert isinstance(frame.get("steps"), list) and frame["steps"], \
        "a status-only terminal frame is what the reaper already sent"
    assert frame["total_steps"] and frame["completed_steps"] is not None
    assert frame.get("percent") is not None


# ── 4. The assertion that says this is impossible ─────────────────────

@pytest.mark.asyncio
async def test_the_zombie_assertion_can_fire(monkeypatch, caplog):
    """A check that cannot fail proves nothing. The stale row is seeded and
    the sweep deliberately NOT run, so the assertion has something to find
    and logs it at ERROR."""
    import logging

    from app.agent.build_watchdog import assert_no_zombie_cards

    job_id = await _seed_build()
    with caplog.at_level(logging.ERROR, logger="app.agent.build_watchdog"):
        offenders = await assert_no_zombie_cards()
    assert job_id in offenders
    assert any("ZOMBIE CARD" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_the_zombie_assertion_is_silent_after_the_sweep(monkeypatch, caplog):
    """…and finds nothing once the sweep has run, which is the state the
    fleet is supposed to be in at every tick."""
    import logging

    from app.agent.build_watchdog import (
        assert_no_zombie_cards, sweep_stuck_builds,
    )

    _mute_surfaces(monkeypatch)
    await _seed_build()
    await _seed_build()
    assert await sweep_stuck_builds() == 2
    with caplog.at_level(logging.ERROR, logger="app.agent.build_watchdog"):
        assert await assert_no_zombie_cards() == []
    assert not [r for r in caplog.records if "ZOMBIE CARD" in r.getMessage()]


# ── 5. The other lanes are untouched ──────────────────────────────────

@pytest.mark.asyncio
async def test_a_non_build_job_is_not_settled_here(monkeypatch):
    """`settle_build` owns ONE lane. An `agent_task` — narration for work
    the model did inline — has its own rule (a delivered answer) and must
    not be failed by the build watchdog."""
    from app.agent.build_watchdog import settle_build, sweep_stuck_builds

    _mute_surfaces(monkeypatch)
    task = await _seed_build(job_type="agent_task")
    assert await settle_build(task, reason="watchdog") is None
    assert await sweep_stuck_builds() == 0
    _s, status, _ = await _row(task)
    assert status == "running"


@pytest.mark.asyncio
async def test_an_auto_builder_row_with_no_app_is_not_a_build(monkeypatch):
    """`auto_builder` predates this pipeline. A row with no `app_id` drives
    no app card, so it stays the generic reaper's."""
    from app.agent.build_watchdog import is_build_row, settle_build
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    _mute_surfaces(monkeypatch)
    job_id = await _seed_build(app_id=None)
    async with async_session_maker() as db:
        assert is_build_row(await db.get(BuildJob, job_id)) is False
    assert await settle_build(job_id, reason="watchdog") is None


@pytest.mark.asyncio
async def test_the_stall_reaper_hands_builds_over_instead_of_cancelling(
    monkeypatch,
):
    """The reaper's own net, at 30 minutes. Before this round it wrote
    `cancelled` and left `steps_json` alone — a terminal row under a live
    card."""
    from app.agent.job_reaper import sweep_stalled_jobs

    _mute_surfaces(monkeypatch)
    job_id = await _seed_build(age_minutes=90)
    await sweep_stalled_jobs()

    steps, status, _ = await _row(job_id)
    assert status == "failed"
    assert _by_type(steps)["present"]["status"] == "failed"
    assert not any(s["status"] in ("running", "pending") for s in steps), \
        "the reaper left live rows under a terminal status again"


# ── 6. The crash path: a terminal event without waiting for a sweep ───

def _skill_and_ctx(monkeypatch, job_id: str):
    from app.agent.skills.base import SkillContext
    from app.agent.skills.builtins.app_html import steps as steps_mod
    from app.agent.skills.builtins.app_html.skill import AppHtmlSkill

    async def _jid(*_a, **_k):
        return job_id

    monkeypatch.setattr(steps_mod, "job_id_for_slug", _jid)
    _mute_surfaces(monkeypatch)
    return AppHtmlSkill(), SkillContext(
        workspace="/tmp", user_id=USER_ID, session_id="s-r27",
    )


@pytest.mark.asyncio
async def test_a_crash_inside_present_settles_the_build_at_once(monkeypatch):
    """`execute_tool` catches `AppStoreError` / `ShellRefusal` / `OSError` and
    nothing else, so a raise out of the visual review, the icon draw or the
    publish unwound the turn with the phase it was in still `running` — no
    terminal event, and the card frozen at whatever it last showed.

    The seeded row is ONE MINUTE old: the watchdog would not touch it for
    another fourteen, so a pass here can only come from the crash path
    itself.
    """
    from app.agent.skills.builtins.app_html.skill import AppHtmlSkill

    job_id = await _seed_build(age_minutes=1, last_event_minutes=0)
    skill, ctx = _skill_and_ctx(monkeypatch, job_id)

    async def _boom(*_a, **_k):
        raise RuntimeError("the visual review died mid-call")

    monkeypatch.setattr(AppHtmlSkill, "_present_build", _boom)
    with pytest.raises(RuntimeError):
        await skill._present({"slug": "habit-garden"}, ctx)

    steps, status, _ = await _row(job_id)
    assert status == "failed"
    assert _by_type(steps)["present"]["status"] == "failed"
    assert not any(s["status"] in ("running", "pending") for s in steps)


@pytest.mark.asyncio
async def test_a_cancelled_present_still_settles(monkeypatch):
    """Cancellation is how a turn normally dies — the SSE generator cancels
    the task the moment the client disconnects. The settle is DETACHED
    there, because awaiting inside a cancelled task raises at the await and
    the cleanup would be skipped by the very condition that makes it
    necessary."""
    from app.agent.skills.builtins.app_html.skill import AppHtmlSkill

    job_id = await _seed_build(age_minutes=1, last_event_minutes=0)
    skill, ctx = _skill_and_ctx(monkeypatch, job_id)

    async def _cancelled(*_a, **_k):
        raise asyncio.CancelledError()

    monkeypatch.setattr(AppHtmlSkill, "_present_build", _cancelled)
    with pytest.raises(asyncio.CancelledError):
        await skill._present({"slug": "habit-garden"}, ctx)

    for _ in range(60):           # let the detached task run
        await asyncio.sleep(0.01)
        _s, status, _ = await _row(job_id)
        if status == "failed":
            break
    _s, status, _ = await _row(job_id)
    assert status == "failed"


@pytest.mark.asyncio
async def test_a_gate_refusal_does_not_settle_the_build(monkeypatch):
    """Anti-vacuity, and the single most important non-regression in this
    round. The publish gate refusing is the DESIGNED loop: the model gets a
    repair list and comes back. If the wrapper settled on that too, the
    looks-right loop would be dead on arrival — every refusal would close
    the card and the repair would have nowhere to land."""
    from app.agent.skills.builtins.app_html.skill import AppHtmlSkill
    from app.agent.skills.builtins.app_html.store import AppStoreError

    job_id = await _seed_build(age_minutes=1, last_event_minutes=0)
    skill, ctx = _skill_and_ctx(monkeypatch, job_id)

    async def _refuse(*_a, **_k):
        raise AppStoreError("the app is not working yet, so it was not published")

    monkeypatch.setattr(AppHtmlSkill, "_present_build", _refuse)
    with pytest.raises(AppStoreError):
        await skill._present({"slug": "habit-garden"}, ctx)

    steps, status, _ = await _row(job_id)
    assert status == "running", "a refusal must leave the build alive"
    assert _by_type(steps)["look"]["status"] == "running"


@pytest.mark.asyncio
async def test_an_agent_task_keeps_the_models_own_step_words():
    """Found on the way. `_settle_build_steps` was called on EVERY row the
    interrupted-turn sweep closed, and `settle_steps` rewrites each
    unfinished row's label from `app_html._PHASE_WORDS`. A `create_job` step
    carries the model's own words and no `type`, so `phase_label('',
    'skipped')` fell through to the pipeline's generic string and an
    interrupted research task came back with its steps relabelled "Working
    on your app"."""
    from app.agent.agent_runner import _settle_build_steps
    from app.db.database import async_session_maker

    steps = [
        {"label": "Read the three papers", "status": "done"},
        {"label": "Compare the funding numbers", "status": "running"},
    ]
    job_id = await _seed_build(steps=steps, job_type="agent_task")
    async with async_session_maker() as db:
        await _settle_build_steps(db, job_id)
        await db.commit()

    after, _status, _ = await _row(job_id)
    assert [s["label"] for s in after] == [
        "Read the three papers", "Compare the funding numbers",
    ]
    assert after[1]["status"] == "running", \
        "an agent_task's rows are not this helper's to resolve"


@pytest.mark.asyncio
async def test_the_settle_is_idempotent(monkeypatch):
    """Five callers, one row. The second one through must be a no-op, not a
    second close with a different timestamp."""
    from app.agent.build_watchdog import settle_build

    _mute_surfaces(monkeypatch)
    job_id = await _seed_build()
    first = await settle_build(job_id, reason="watchdog")
    assert first is not None
    assert await settle_build(job_id, reason="reaper") is None
    _s, status, _ = await _row(job_id)
    assert status == "failed"
