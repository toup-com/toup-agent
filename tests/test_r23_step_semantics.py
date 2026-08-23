"""Round 23 — the build card's arithmetic is honest, and stays honest.

Three recorded defects, one root each, all pinned here:

* **"2/7 steps" fell back to "1/7", a green check re-spun.** `emit_step`
  keyed rows by TYPE and reused one `verify` row for the create-time syntax
  check, `bash_app` and the publish gate — re-running the phase took a
  finished tick back. A phase re-entered after `done` now APPENDS a fresh
  occurrence; the done-count is monotonic by construction.

* **7 planned rows became 4 at completion.** `finish_job` dropped every row
  still pending/running. It now marks them `skipped` with their own words —
  a planned phase may be shown as skipped, never vanish.

* **A turn-end watchdog force-greened an unpublished build.** The shared
  `close_job_completed` marks every remaining step done — right for a task
  whose answer WAS the work, and "Published the app ✓" for an app that never
  published. Build jobs are settled honestly instead (`cancelled`/`failed`),
  and `finish_job` stays the ONE way a build completes.

Plus the two round-23 additions to the gate's instruments: keyboard-hint
text is a publish-refusing finding, and the publish stores the start-screen
snapshot the card serves as its preview.
"""
from __future__ import annotations

import json
import os
import uuid

import pytest
import pytest_asyncio

os.environ.setdefault("AGENT_API_KEY", "test-key-r23-step-semantics")

USER_ID = "00000000-0000-0000-0000-00000000ff23"


@pytest_asyncio.fixture(autouse=True)
async def _tables():
    """Only the tables these paths touch — same pattern as
    test_auto_builder_phase_events, which is what keeps this file runnable
    on a box where the full conftest init is broken."""
    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (JobEvent, BuildJob, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)
    await engine.dispose()


@pytest.fixture
def apps_dir(tmp_path, monkeypatch):
    root = tmp_path / "apps"
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(root))
    return root


async def _seed_job(steps=None) -> str:
    from app.agent.skills.builtins.app_html import steps as steps_mod
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=USER_ID, title="Build: Test App",
            prompt="test", job_type="auto_builder", status="running",
            app_id=str(uuid.uuid4()),
            steps_json=json.dumps(steps or steps_mod.initial_steps()),
        ))
        await db.commit()
    return job_id


async def _steps_of(job_id: str):
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        return json.loads(job.steps_json), job.status


def _done_count(steps) -> int:
    return sum(1 for s in steps if s.get("status") == "done")


# ── The plan ──────────────────────────────────────────────────────────

def test_the_plan_is_the_five_walked_phases():
    """`review`/`edit` are phases a clean first build never enters; putting
    them in the up-front plan GUARANTEED two rows that vanish at the end."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    types = [s["type"] for s in steps_mod.initial_steps()]
    assert types == ["create", "verify", "look", "logo", "present"]
    assert all(s["status"] == "pending" for s in steps_mod.initial_steps())


# ── Occurrence semantics / monotonic progress ─────────────────────────

@pytest.mark.asyncio
async def test_a_rerun_check_retries_the_same_row():
    """Round N correction: a re-run is a RETRY of the SAME row.

    Round 23 appended a second row here, and the 22:05 recording showed
    where that leads — a duplicate "Checking the app" under the completed
    one, the plan growing 5 → 6 mid-run, and the header's percentage
    regressing 40% → 33% because the denominator grew. The re-entered
    check now flips the SAME row (stable id) back to running: the
    denominator never moves, and the momentary done-count dip is the
    clients' high-water latch's problem, not the list's.
    """
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()

    # The walk the recording showed: create, syntax verify done, then the
    # publish gate re-entering verify.
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="create", status="running")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="create", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="done",
                              detail="the code checks out")
    steps, _ = await _steps_of(job_id)
    first_id = next(s["id"] for s in steps if s["type"] == "verify")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="running")
    steps, _ = await _steps_of(job_id)

    # ONE verify row, the SAME row, visibly re-running.
    verify_rows = [s for s in steps if s["type"] == "verify"]
    assert len(verify_rows) == 1, "a re-run must never append a second row"
    assert verify_rows[0]["id"] == first_id
    assert verify_rows[0]["status"] == "running"
    # The plan never grew: the denominator is stable by construction.
    types = [s["type"] for s in steps]
    assert types == ["create", "verify", "look", "logo", "present"]
    # …and when it lands again, the tick returns on the same row.
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="done",
                              detail="opened it — no errors")
    steps, _ = await _steps_of(job_id)
    verify_rows = [s for s in steps if s["type"] == "verify"]
    assert len(verify_rows) == 1
    assert verify_rows[0]["status"] == "done"
    assert verify_rows[0]["id"] == first_id
    # Every write bumps `rev` — what lets the clients tell this retry
    # (higher rev) from a stale poll (older rev) and show the re-check.
    assert verify_rows[0].get("rev", 0) >= 3


@pytest.mark.asyncio
async def test_a_failed_phase_retries_in_place():
    """failed → running is the retry/fixing state — the ONE in-place
    re-entry, so a repaired problem does not leave a permanent ✗."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="running")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="failed",
                              detail="found 2 problems to fix")
    _, status = await _steps_of(job_id)
    assert status == "failed"  # honest while broken
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="running")
    steps, status = await _steps_of(job_id)
    verify_rows = [s for s in steps if s["type"] == "verify"]
    assert len(verify_rows) == 1, "a retry must not append a second row"
    assert verify_rows[0]["status"] == "running"
    assert status == "running", "a retried build is running, not failed"


@pytest.mark.asyncio
async def test_edit_rows_land_in_work_order():
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="create", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="edit", status="running",
                              detail="bigger D-pad")
    steps, _ = await _steps_of(job_id)
    types = [s["type"] for s in steps]
    # After the finished work, before the untouched plan.
    assert types == ["create", "edit", "verify", "look", "logo", "present"]


@pytest.mark.asyncio
async def test_a_check_loop_cannot_grow_the_card_at_all():
    """Retry-in-place makes the old occurrence cap unnecessary: a model
    stuck re-checking forever cycles ONE row, and the card never grows."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    for _ in range(12):
        await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                                  step_type="verify", status="done")
        await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                                  step_type="verify", status="running")
    steps, _ = await _steps_of(job_id)
    assert len([s for s in steps if s["type"] == "verify"]) == 1
    assert len(steps) == len(steps_mod.PLANNED_TYPES)


@pytest.mark.asyncio
async def test_nothing_spins_above_an_advancing_row():
    """The ordering net: an earlier row abandoned `running` is resolved
    skipped the moment a later row advances — the recorded card had
    "Looking at the app" spinning while icon and publish completed
    beneath it."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="create", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="look", status="running")
    # The pipeline bug: logo advances while look never resolved.
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="logo", status="running")
    steps, _ = await _steps_of(job_id)
    look = next(s for s in steps if s["type"] == "look")
    assert look["status"] == "skipped"
    assert look.get("detail"), "an abandoned row carries a reason"
    # And the counts exclude it from both numbers.
    done, total = steps_mod.step_counts(steps)
    assert (done, total) == (2, 4)


@pytest.mark.asyncio
async def test_a_skip_emitted_by_the_skill_lands_with_its_reason():
    """The look phase resolves skipped AT THE MOMENT it cannot run —
    never left spinning for finish_job to settle at build end."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="look", status="running")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="look", status="skipped",
                              detail="no browser on this host")
    steps, status = await _steps_of(job_id)
    look = next(s for s in steps if s["type"] == "look")
    assert look["status"] == "skipped"
    assert look["detail"] == "no browser on this host"
    assert look["label"] == "Couldn't look at the app here"
    assert status == "running", "a skip is not a failure"


# ── finish_job: skip, never drop ──────────────────────────────────────

@pytest.mark.asyncio
async def test_finish_skips_unreported_phases_instead_of_dropping_them():
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    plan_len = len(steps_mod.initial_steps())
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="create", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="done")
    # `look` reported running and never came back (a look that couldn't
    # run) — the exact row the old code deleted.
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="look", status="running")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="present", status="done")
    final = await steps_mod.finish_job(USER_ID, job_id)
    steps, status = await _steps_of(job_id)

    assert final == "completed" and status == "completed"
    assert len(steps) == plan_len, "no row may vanish at completion"
    by_type = {s["type"]: s for s in steps}
    assert by_type["look"]["status"] == "skipped"
    assert by_type["look"]["label"] == "Couldn't look at the app here"
    assert by_type["logo"]["status"] == "skipped"
    assert by_type["create"]["status"] == "done"


@pytest.mark.asyncio
async def test_a_failed_step_at_finish_fails_the_build():
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="create", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="failed")
    final = await steps_mod.finish_job(USER_ID, job_id)
    assert final == "failed"


# ── The watchdog settle: a build is never completed by anyone else ────

@pytest.mark.asyncio
async def test_an_unpublished_build_is_settled_cancelled_not_completed():
    from datetime import datetime

    from app.agent.job_reconciler import close_job_completed
    from app.db.database import async_session_maker

    job_id = await _seed_job()
    async with async_session_maker() as db:
        closed = await close_job_completed(
            db, job_id, user_id=USER_ID, now=datetime.utcnow(),
            reason="turn_end",
        )
        await db.commit()
    assert closed is None, "a build must never come back as a ClosedJob"
    steps, status = await _steps_of(job_id)
    assert status == "cancelled"
    assert all(s["status"] != "done" for s in steps), \
        "no phase may be invented done by a watchdog"
    assert all(s["status"] == "skipped" for s in steps)


@pytest.mark.asyncio
async def test_an_abandoned_failed_build_settles_failed():
    from datetime import datetime

    from app.agent.job_reconciler import close_job_completed
    from app.agent.skills.builtins.app_html import steps as steps_mod
    from app.db.database import async_session_maker

    job_id = await _seed_job()
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="create", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="failed")
    # emit_step marked the job failed; the settle only touches running
    # rows, so put it back the way an in-flight repair would have.
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="running")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="failed")
    async with async_session_maker() as db:
        from app.db.models import BuildJob
        job = await db.get(BuildJob, job_id)
        job.status = "running"  # the shape at turn end mid-repair
        await db.commit()
    async with async_session_maker() as db:
        closed = await close_job_completed(
            db, job_id, user_id=USER_ID, now=datetime.utcnow(),
            reason="turn_end",
        )
        await db.commit()
    assert closed is None
    steps, status = await _steps_of(job_id)
    assert status == "failed"
    assert any(s["status"] == "failed" for s in steps)


# ── Keyboard hints are findings ───────────────────────────────────────

def test_keyboard_instructions_are_refused():
    from app.agent.skills.builtins.app_html import verify as verify_mod

    for text in ("ARROWS / WASD · SPACE PAUSES",
                 "Use the arrow keys to steer",
                 "Press SPACE to begin",
                 "press any key"):
        findings = verify_mod.keyboard_hint_findings(text)
        assert findings, text
        assert "keyboard" in findings[0].message


def test_an_archery_game_may_count_its_arrows():
    from app.agent.skills.builtins.app_html import verify as verify_mod

    for text in ("Arrows left: 5", "Score 120 · Best 300",
                 "Swipe on the display to steer", ""):
        assert verify_mod.keyboard_hint_findings(text) == [], text


# ── The preview follows the icon's contract ───────────────────────────

def test_preview_roundtrip_and_cleanup(apps_dir):
    from app.agent.skills.builtins.app_html import store

    store.ensure_root()
    store.write_app("snake", "Snake", "<!doctype html><html><head>"
                    "<meta charset='utf-8'><title>s</title></head>"
                    "<body><main>snake app body for the store test, long "
                    "enough to clear the stub gate. " + "x" * 400 +
                    "</main></body></html>")
    png = b"\x89PNG\r\n\x1a\n" + b"fake-bytes" * 10

    assert store.preview_etag("snake") == ""
    assert store.write_preview("snake", png) is True
    assert store.read_preview("snake") == png
    etag = store.preview_etag("snake")
    assert etag and len(etag) == 32

    # Oversized is refused, not stored.
    assert store.write_preview("snake", b"x" * (store.MAX_PREVIEW_BYTES + 1)) is False

    # Slugs are reusable: a deleted app takes its face with it.
    store.delete_app("snake")
    assert store.read_preview("snake") is None


def test_artifact_payload_carries_the_preview_etag(apps_dir):
    from app.agent.skills.builtins.app_html import steps as steps_mod, store

    store.ensure_root()
    store.write_app("mines", "Mines", "<!doctype html><html><head>"
                    "<meta charset='utf-8'><title>m</title></head>"
                    "<body><main>minesweeper body for the payload test. "
                    + "y" * 400 + "</main></body></html>")
    payload = steps_mod.artifact_payload("mines")
    assert payload["has_preview"] is False and payload["preview_etag"] == ""

    store.write_preview("mines", b"\x89PNG\r\n\x1a\nabc")
    payload = steps_mod.artifact_payload("mines")
    assert payload["has_preview"] is True
    assert payload["preview_etag"] == store.preview_etag("mines")
