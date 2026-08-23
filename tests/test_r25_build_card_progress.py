"""Round 25 — the build card never goes backwards, and a fix round is not a failure.

Two defects, one card, and they were visible in the same recording.

**The bar ran backwards, three times in one build.** The percentage was never
computed on the server: the wire carried ``(completed_steps, total_steps)`` and
each of the three clients divided, so each re-derived the same number and each
broke it in its own way. The only monotonicity guarantee anywhere was a
per-mount ``useRef`` in ONE client, which latched the BAR and not the "N/M
steps" text printed beside it. Driving the pipeline's own designed path —
write, check, publish (refused), read, edit, publish — produced:

    0/5 → 1/5 (20%) → 2/5 (40%) → 1/5 (20%) → 1/6 (17%) → 2/6 (33%)
        → 2/7 (29%) → 3/7 (43%) → 4/7 (57%)

40 → 20, 20 → 17, 33 → 29. The first dip is the largest and the least
defensible: it fired at the exact moment the gate started *helping*.

**And the whole card went red while it was being helped.** A refused check set
``BuildJob.status = "failed"``, and not for one frame — ``any_failed`` re-scans
the entire step list on EVERY subsequent emit, so the refused row held the job
at ``failed`` through the model's read, through its edit and through the
write-back, until the next ``present_app``. The user watched a red "Couldn't
build · Try again" pill for the whole repair, for a build that went on to
succeed.

Both are the same missing distinction: nothing on the wire could tell "the
check found something and I am fixing it" from "this build is dead". These
tests drive the REAL ``emit_step`` against a REAL ``BuildJob`` row and assert
on the frames a client would actually receive.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.agent.skills.builtins.app_html import steps as steps_mod
from app.db.database import async_session_maker
from app.db.models import BuildJob


@pytest.fixture
def frames(monkeypatch) -> List[Dict[str, Any]]:
    """Every ``job_update`` a client would have received, in order."""
    captured: List[Dict[str, Any]] = []

    async def _capture(user_id: str, payload: Dict[str, Any]) -> None:
        captured.append(payload)

    monkeypatch.setattr(steps_mod, "_broadcast", _capture)
    return captured


async def _job(user_id: str, slug: str = "snake", title: str = "Snake") -> str:
    jid = await steps_mod.ensure_job(user_id, slug, title)
    assert jid, "the pipeline could not open a job"
    return jid


async def _steps_of(job_id: str) -> List[Dict[str, Any]]:
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        return json.loads(job.steps_json) if job and job.steps_json else []


async def _status_of(job_id: str) -> str:
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        return job.status if job else ""


async def _walk_the_designed_path(user_id: str, job_id: str) -> None:
    """The exact sequence the recording produced: a build whose publish gate
    refuses once, and which is then repaired and published.

    This is not an unusual path. ``view_app_file`` before every edit is
    mandated by the skill's own prompt, and the gate refusing at least once is
    what ``GATE_MAX_REFUSALS`` budgets for.
    """
    E = steps_mod.emit_step
    await E(user_id=user_id, job_id=job_id, step_type="create", status="running")
    await E(user_id=user_id, job_id=job_id, step_type="create", status="done")
    # The shallow syntax check on write passes.
    await E(user_id=user_id, job_id=job_id, step_type="verify", status="done",
            recoverable=True)
    # present_app: the deep gate re-runs verify and REFUSES.
    await E(user_id=user_id, job_id=job_id, step_type="verify", status="running")
    await E(user_id=user_id, job_id=job_id, step_type="verify", status="failed",
            detail="1 problem", recoverable=True)
    # The model reads the file back and edits it — both append a NEW row, so
    # the denominator grows here (5 → 6 → 7).
    await E(user_id=user_id, job_id=job_id, step_type="review", status="running")
    await E(user_id=user_id, job_id=job_id, step_type="review", status="done")
    await E(user_id=user_id, job_id=job_id, step_type="edit", status="running")
    await E(user_id=user_id, job_id=job_id, step_type="edit", status="done",
            recoverable=True)
    # present_app again: this time the gate passes.
    await E(user_id=user_id, job_id=job_id, step_type="verify", status="running")
    await E(user_id=user_id, job_id=job_id, step_type="verify", status="done",
            recoverable=True)


# ── The headline ──────────────────────────────────────────────────────

async def test_the_percentage_never_goes_backwards_on_the_designed_path(
    test_user_id, frames,
):
    job_id = await _job(test_user_id)
    await _walk_the_designed_path(test_user_id, job_id)

    percents = [f["percent"] for f in frames]
    assert percents, "no frames were broadcast"
    regressions = [
        (a, b) for a, b in zip(percents, percents[1:]) if b < a
    ]
    assert not regressions, (
        f"the bar ran backwards {len(regressions)} time(s): {regressions}\n"
        f"full sequence: {percents}"
    )


async def test_the_done_count_never_goes_backwards_either(test_user_id, frames):
    """The percentage is a ratio, so it can be held steady while the numbers
    under it still regress — and the "N/M steps" text is printed from those
    numbers, not from the percentage. Both have to hold."""
    job_id = await _job(test_user_id)
    await _walk_the_designed_path(test_user_id, job_id)

    done = [f["completed_steps"] for f in frames]
    regressions = [(a, b) for a, b in zip(done, done[1:]) if b < a]
    assert not regressions, (
        f"the done-count regressed: {regressions}\nfull sequence: {done}"
    )


async def test_a_refused_check_does_not_un_count_the_work_it_already_did(
    test_user_id, frames,
):
    """The single largest dip in the recording, isolated.

    ``verify`` passes (2/5, 40%), the publish gate re-runs it and refuses. The
    row is the SAME row — retry-in-place — so the phase it represents really
    did complete once. Popping ``was_done`` on the refusal is what took the
    card to 1/5 / 20% at the exact moment the pipeline began repairing itself.
    """
    job_id = await _job(test_user_id)
    E = steps_mod.emit_step
    await E(user_id=test_user_id, job_id=job_id, step_type="create", status="running")
    await E(user_id=test_user_id, job_id=job_id, step_type="create", status="done")
    await E(user_id=test_user_id, job_id=job_id, step_type="verify", status="done")
    peak = frames[-1]["completed_steps"]

    await E(user_id=test_user_id, job_id=job_id, step_type="verify", status="running")
    await E(user_id=test_user_id, job_id=job_id, step_type="verify",
            status="failed", detail="1 problem", recoverable=True)

    assert frames[-1]["completed_steps"] >= peak, (
        "a recoverable refusal un-counted a phase that had already completed"
    )


# ── A fix round is not a failure ──────────────────────────────────────

async def test_a_recoverable_failure_leaves_the_job_running(test_user_id, frames):
    job_id = await _job(test_user_id)
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="verify",
        status="failed", detail="1 problem", recoverable=True,
    )
    assert await _status_of(job_id) == "running", (
        "a gate refusal flipped the whole job to failed — this is what painted "
        "the card red for the entire repair window"
    )
    assert frames[-1]["status"] == "running"


async def test_the_job_stays_running_for_the_WHOLE_repair_window(
    test_user_id, frames,
):
    """``any_failed`` re-scans every row on every emit, so the regression this
    guards is not a single frame — it is every frame until the next
    ``present_app``. Assert on the read and the edit that follow."""
    job_id = await _job(test_user_id)
    E = steps_mod.emit_step
    await E(user_id=test_user_id, job_id=job_id, step_type="verify",
            status="failed", detail="1 problem", recoverable=True)
    for step_type in ("review", "edit"):
        for status in ("running", "done"):
            await E(user_id=test_user_id, job_id=job_id,
                    step_type=step_type, status=status)
            assert frames[-1]["status"] == "running", (
                f"the card was still red at {step_type}/{status}"
            )


async def test_a_terminal_failure_still_goes_red(test_user_id, frames):
    """The other half. Red has to keep meaning something."""
    job_id = await _job(test_user_id)
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create",
        status="failed", detail="disk full",
    )
    assert await _status_of(job_id) == "failed"
    assert frames[-1]["status"] == "failed"
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.user_message, "a failed build must say what happened"


async def test_the_header_says_fixing_it_while_repairing(test_user_id, frames):
    job_id = await _job(test_user_id)
    E = steps_mod.emit_step
    await E(user_id=test_user_id, job_id=job_id, step_type="verify",
            status="failed", detail="1 problem", recoverable=True)
    await E(user_id=test_user_id, job_id=job_id, step_type="review",
            status="running")
    assert frames[-1]["step"] == steps_mod.FIXING_LABEL, (
        "during a repair the header showed the label of whichever row was "
        "touched last ('Reading the app'), which describes the mechanism "
        "rather than the state"
    )


async def test_a_build_that_dies_mid_repair_resolves_terminal(test_user_id, frames):
    """``recoverable`` is only true while there is a build running to recover
    it. At close the loop has stopped, so the badge must come off — otherwise a
    dead build wears a friendly in-progress row for ever."""
    job_id = await _job(test_user_id)
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="verify",
        status="failed", detail="1 problem", recoverable=True,
    )
    final = await steps_mod.finish_job(test_user_id, job_id)

    assert final == "failed"
    rows = await _steps_of(job_id)
    assert not any(s.get("recoverable") for s in rows), (
        "a closed build still carried the mid-repair badge"
    )


# ── The arithmetic itself ─────────────────────────────────────────────

def test_percent_is_the_honest_quotient():
    assert steps_mod.percent_for(0, 5) == 0
    assert steps_mod.percent_for(1, 5) == 20
    assert steps_mod.percent_for(4, 7) == 57
    assert steps_mod.percent_for(5, 5) == 100
    assert steps_mod.percent_for(3, 0) == 0, "an empty plan is not 100% done"


def test_plan_growth_alone_cannot_lower_the_percentage():
    """``review``/``edit`` are appended at the moment they happen (round 23
    removed them from the up-front plan on purpose), so the denominator RISES
    mid-build. 1/5 is 20%, 1/6 is 17%."""
    plan = [{"type": t, "status": "done" if t == "create" else "pending"}
            for t in steps_mod.PLANNED_TYPES]
    done, total, before, cfg = steps_mod.progress(plan, {})
    assert (done, total, before) == (1, 5, 20)

    grown = plan + [{"type": "review", "status": "pending"}]
    done, total, after, cfg = steps_mod.progress(grown, cfg)
    assert (done, total) == (1, 6)
    assert after >= before, f"plan growth dropped the bar {before}% → {after}%"


def test_the_high_water_mark_survives_a_restart():
    """It is persisted in ``config_json``, not latched in a client — the whole
    point. A reconnect mid-build must not restart the bar lower than the user
    last saw it."""
    _, _, first, cfg = steps_mod.progress(
        [{"type": "a", "status": "done"}, {"type": "b", "status": "done"}], {},
    )
    assert first == 100
    # A later read sees a row that regressed (a forced skip, say).
    _, _, second, _ = steps_mod.progress(
        [{"type": "a", "status": "done"}, {"type": "b", "status": "pending"}], cfg,
    )
    assert second == 100, "the persisted high-water mark was not honoured"


def test_a_forced_skip_does_not_strand_a_completed_phase(test_user_id):
    """``step_counts`` excludes a skipped row from BOTH numbers before it ever
    reaches the ``was_done`` test, so a row force-skipped by the ordering net
    while still carrying the marker was a phase counted in neither."""
    rows = [
        {"type": "create", "status": "skipped", "was_done": True},
        {"type": "verify", "status": "done"},
    ]
    done, total = steps_mod.step_counts(rows)
    assert (done, total) == (1, 1)
    assert done <= total, "more work done than there is work to do"


async def test_every_frame_carries_the_steps_it_describes(test_user_id, frames):
    """``job_update`` never carried the rows, so a client that had not seen
    every earlier frame — one that reconnected, or hydrated from history — had
    a header and a bar with nothing to put under them."""
    job_id = await _job(test_user_id)
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create", status="running",
    )
    frame = frames[-1]
    assert frame["steps"], "the frame described a build but carried no rows"
    assert {s["type"] for s in frame["steps"]} >= set(steps_mod.PLANNED_TYPES)
    assert frame["completed_steps"] <= frame["total_steps"]


async def test_a_completed_build_reads_one_hundred_percent(test_user_id, frames):
    job_id = await _job(test_user_id)
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create", status="done",
    )
    await steps_mod.finish_job(test_user_id, job_id)
    last = frames[-1]
    assert last["status"] == "completed"
    assert last["percent"] == 100, (
        "a finished card showed a bar that was not full"
    )
