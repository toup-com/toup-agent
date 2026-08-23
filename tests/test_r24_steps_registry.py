"""Round 24 — the build job is visible to the turn's finalizers, the final
artifact frame carries the mark, and no close un-counts finished work.

Three pins:

* **`ensure_job` registers every id it returns.** The system prompt forbids
  the model calling `create_job` for a build, so the pipeline mints the job
  itself — and an unregistered build job is invisible to both turn-end
  finalizers (agent_runner's happy-path close and
  `_sweep_unclosed_created_jobs`); only the 30-minute reaper would close it,
  which is the recorded 1:26 AM zombie card.

* **The `app_artifact` frame follows the logo step and carries the icon
  inline.** `announce_ready` is the ONLY producer of that frame and the
  skill calls it after `ensure_icon`, so the card's final state has the
  drawn mark and the client never rests on a monogram. Pinned as ORDER —
  the emission already exists and already inlines the mark (capped).

* **A close never decreases the done count.** Retry-in-place takes a done
  row back through `running`, so a close that resolves running → skipped
  un-counted finished work (the recorded 2/5 → 1/5 at death). Rows now
  carry `was_done`; `finish_job` restores such a row to done, and
  `step_counts` keeps counting it while it re-runs.
"""
from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace

import pytest
import pytest_asyncio

os.environ.setdefault("AGENT_API_KEY", "test-key-r24-steps-registry")

USER_ID = "00000000-0000-0000-0000-00000000ff24"


@pytest_asyncio.fixture(autouse=True)
async def _tables():
    """Only the tables these paths touch — same pattern as
    test_r23_step_semantics, which is what keeps this file runnable on a
    box where the full conftest init is broken."""
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


def _fresh_registry() -> list:
    from app.agent import tool_executor as te

    reg: list = []
    te._CREATED_JOB_IDS_CTX.set(reg)
    return reg


async def _none(*_a, **_k):
    return None


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


# ── A: the minted build job joins the created-jobs registry ───────────

@pytest.mark.asyncio
async def test_a_minted_build_job_is_registered_for_the_finalizers(monkeypatch):
    """The zombie shape: `ensure_job` mints via JobRunner, the model never
    called `create_job`, so nothing appended the id — both turn-end
    finalizers saw an empty registry and the abnormally-ended build sat
    `running` until the 30-minute reaper."""
    import app.agent.job_runner as job_runner_mod
    from app.agent.skills.builtins.app_html import steps as steps_mod
    from app.agent.tool_executor import created_job_ids

    _fresh_registry()
    monkeypatch.setattr(steps_mod, "_existing_job_for_app", _none)
    monkeypatch.setattr(steps_mod, "_adopt_turn_job", _none)

    minted: dict = {}

    class _FakeRunner:
        async def create_job(self, **kwargs):
            minted.update(kwargs)
            return SimpleNamespace(id="job-mint-r24")

    monkeypatch.setattr(job_runner_mod, "JobRunner", _FakeRunner)

    job_id = await steps_mod.ensure_job(USER_ID, "snake", "Snake")
    assert job_id == "job-mint-r24"
    assert "job-mint-r24" in created_job_ids()
    assert minted.get("status") == "running"

    # ensure_job runs once per tool call — five times per build. One id,
    # one close.
    await steps_mod.ensure_job(USER_ID, "snake", "Snake")
    assert list(created_job_ids()).count("job-mint-r24") == 1


@pytest.mark.asyncio
async def test_a_revived_prior_turn_job_is_registered_too(monkeypatch):
    """An edit three turns later reuses the existing job and may revive it
    to `running` — that path must be finalizer-visible for the same
    reason the mint is."""
    from app.agent.skills.builtins.app_html import steps as steps_mod
    from app.agent.tool_executor import created_job_ids

    _fresh_registry()

    async def _existing(_app_id):
        return "job-prior-r24"

    monkeypatch.setattr(steps_mod, "_existing_job_for_app", _existing)

    assert await steps_mod.ensure_job(USER_ID, "snake", "Snake") == "job-prior-r24"
    assert "job-prior-r24" in created_job_ids()


@pytest.mark.asyncio
async def test_an_adopted_job_is_not_registered_twice(monkeypatch):
    """The adopted job came from the `create_job` tool this turn, so it is
    ALREADY in the registry — a second append would hand the finalizers
    the same id twice."""
    from app.agent.skills.builtins.app_html import steps as steps_mod
    from app.agent.tool_executor import created_job_ids

    reg = _fresh_registry()
    reg.append("job-adopt-r24")  # the tool path's own append
    monkeypatch.setattr(steps_mod, "_existing_job_for_app", _none)

    async def _adopt(_user_id, _app_id, _title):
        return "job-adopt-r24"

    monkeypatch.setattr(steps_mod, "_adopt_turn_job", _adopt)

    assert await steps_mod.ensure_job(USER_ID, "snake", "Snake") == "job-adopt-r24"
    assert list(created_job_ids()) == ["job-adopt-r24"]


# ── B: the artifact frame follows the logo step, mark aboard ──────────

def test_the_artifact_frame_is_emitted_after_the_logo_step():
    """ORDER pin, not a new emission: `announce_ready` is the only producer
    of the `app_artifact` frame and the skill calls it after `ensure_icon`
    ran — so the one frame the client receives already describes the drawn
    mark, never the monogram the logo step replaced."""
    from app.agent.skills.builtins.app_html import skill as skill_mod

    src = open(skill_mod.__file__, encoding="utf-8").read()
    icon_at = src.rindex("ensure_icon(")
    announce_at = src.index("announce_ready(")
    assert icon_at < announce_at, \
        "the artifact frame must be emitted after the icon is drawn"
    # And there is exactly one call site — a second, pre-logo emission
    # would put a markless card back on the wire.
    assert src.count("announce_ready(") == 1


@pytest.mark.asyncio
async def test_announce_ready_inlines_the_icon_on_the_artifact_frame(
        apps_dir, monkeypatch):
    from app.agent.skills.builtins.app_html import (
        logo as logo_mod, steps as steps_mod, store,
    )

    store.ensure_root()
    store.write_app("snake", "Snake", "<!doctype html><html><head>"
                    "<meta charset='utf-8'><title>s</title></head>"
                    "<body><main>snake body for the announce test. "
                    + "x" * 400 + "</main></body></html>")
    svg = "<svg xmlns='http://www.w3.org/2000/svg'><circle r='4'/></svg>"
    icon_file = logo_mod.icon_path("snake")
    os.makedirs(os.path.dirname(icon_file), exist_ok=True)
    with open(icon_file, "w", encoding="utf-8") as fh:
        fh.write(svg)

    frames = []

    async def _capture(_user_id, payload):
        frames.append(payload)

    monkeypatch.setattr(steps_mod, "_broadcast", _capture)
    await steps_mod.announce_ready(
        user_id=USER_ID, job_id="j1", app_id="a1", title="Snake",
        slug="snake",
    )

    assert [f["type"] for f in frames] == ["app_artifact", "app_ready"], \
        "app_artifact first, so the registry holds the new revision when " \
        "app_ready draws the card"
    art = frames[0]["artifact"]
    assert art["has_icon"] is True
    assert art["icon_svg"] == svg
    assert len(art["icon_etag"]) == 32


@pytest.mark.asyncio
async def test_an_oversized_icon_rides_the_etag_not_the_frame(
        apps_dir, monkeypatch):
    """Past the cap the mark stays behind its route — the client still
    learns it CHANGED (`icon_etag`) and fetches once."""
    from app.agent.skills.builtins.app_html import (
        logo as logo_mod, steps as steps_mod, store,
    )

    store.ensure_root()
    store.write_app("mines", "Mines", "<!doctype html><html><head>"
                    "<meta charset='utf-8'><title>m</title></head>"
                    "<body><main>mines body for the cap test. "
                    + "y" * 400 + "</main></body></html>")
    big = "<svg>" + "z" * (steps_mod.MAX_INLINE_ICON_BYTES + 1) + "</svg>"
    icon_file = logo_mod.icon_path("mines")
    os.makedirs(os.path.dirname(icon_file), exist_ok=True)
    with open(icon_file, "w", encoding="utf-8") as fh:
        fh.write(big)

    frames = []

    async def _capture(_user_id, payload):
        frames.append(payload)

    monkeypatch.setattr(steps_mod, "_broadcast", _capture)
    await steps_mod.announce_ready(
        user_id=USER_ID, job_id="j2", app_id="a2", title="Mines",
        slug="mines",
    )

    art = frames[0]["artifact"]
    assert "icon_svg" not in art
    assert art["has_icon"] is True and len(art["icon_etag"]) == 32


# ── C: no close un-counts finished work ───────────────────────────────

@pytest.mark.asyncio
async def test_finish_restores_a_mid_retry_row_instead_of_skipping_it():
    """The 2/5 → 1/5 shape: verify completed, the publish gate re-entered
    it (done → running), and the build closed before the re-check
    reported. The row's last reported result stands — done, not skipped."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="create", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="running")
    steps, _ = await _steps_of(job_id)
    done_before, _ = steps_mod.step_counts(steps)
    assert done_before == 2, "a retry in flight must not dip the count"

    final = await steps_mod.finish_job(USER_ID, job_id)
    steps, status = await _steps_of(job_id)
    assert final == "completed" and status == "completed"
    verify = next(s for s in steps if s["type"] == "verify")
    assert verify["status"] == "done"
    assert verify["label"] == "Checked the app"
    done_after, _ = steps_mod.step_counts(steps)
    assert done_after >= done_before, "a close must never un-count done work"


@pytest.mark.asyncio
async def test_a_failed_retry_is_not_restored_done():
    """`was_done` records "ever done", not "ever touched": a row whose last
    verdict was FAILED and whose repair never reported back closes
    skipped — restoring done would invent a pass."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    # done FIRST: the pass the retry is re-litigating. The failed verdict
    # must supersede it, or finish_job would restore a check that failed.
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="failed")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="running")
    await steps_mod.finish_job(USER_ID, job_id)
    steps, _ = await _steps_of(job_id)
    verify = next(s for s in steps if s["type"] == "verify")
    assert verify["status"] == "skipped"


@pytest.mark.asyncio
async def test_a_reported_skip_supersedes_an_old_done():
    """A rebuild reuses the row. If its check genuinely could not run this
    time, the skill's own skip is a verdict — the ever-done marker from
    the previous build must not keep counting it."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    job_id = await _seed_job()
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="look", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="look", status="running")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="look", status="skipped",
                              detail="no browser on this host")
    steps, _ = await _steps_of(job_id)
    look = next(s for s in steps if s["type"] == "look")
    assert look["status"] == "skipped" and not look.get("was_done")
    done, total = steps_mod.step_counts(steps)
    assert (done, total) == (0, 4), "a reported skip leaves BOTH numbers"


def test_a_build_stranded_mid_retry_never_uncounts_done():
    """The cancellation sweep closes the JOB without rewriting its steps,
    so a mid-retry row is persisted `running` forever. The one arithmetic
    still counts it: it was done, and the death of the re-check does not
    unfinish it."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    steps = [
        {"type": "create", "status": "done", "was_done": True},
        {"type": "verify", "status": "running", "was_done": True},
        {"type": "look", "status": "pending"},
        {"type": "logo", "status": "pending"},
        {"type": "present", "status": "pending"},
    ]
    assert steps_mod.step_counts(steps) == (2, 5)


@pytest.mark.asyncio
async def test_the_settle_path_keeps_every_done_row_done():
    """The abnormal close (`_settle_unpublished_build`, reached through the
    shared `close_job_completed`) flips only unreported rows to skipped —
    a row holding its verdict keeps it, so the done count at death is the
    done count the user last saw."""
    from datetime import datetime

    from app.agent.job_reconciler import close_job_completed
    from app.agent.skills.builtins.app_html import steps as steps_mod
    from app.db.database import async_session_maker

    job_id = await _seed_job()
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="create", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="verify", status="done")
    await steps_mod.emit_step(user_id=USER_ID, job_id=job_id,
                              step_type="look", status="running")
    async with async_session_maker() as db:
        closed = await close_job_completed(
            db, job_id, user_id=USER_ID, now=datetime.utcnow(),
            reason="turn_end",
        )
        await db.commit()
    assert closed is None, "a build must never come back as a ClosedJob"
    steps, status = await _steps_of(job_id)
    # Round 27 changed the word from `cancelled` to `failed` — a build that
    # stopped without publishing did not publish, and `cancelled` was the
    # one status neither client treats as terminal. What this test pins is
    # the done count, and it is unchanged.
    assert status == "failed"
    by_type = {s["type"]: s for s in steps}
    assert by_type["create"]["status"] == "done"
    assert by_type["verify"]["status"] == "done"
    assert by_type["look"]["status"] == "skipped"
    assert sum(1 for s in steps if s["status"] == "done") == 2, \
        "the settle must not decrease what was already done"
