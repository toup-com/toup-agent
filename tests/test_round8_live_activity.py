"""Round 8 — the phone card must agree with the app.

Two platform-side gaps behind "Live Activity says 1/3, the app says 2/3":

* The app starts a job's card LOCALLY when it backgrounds mid-job, named
  after the raw job id ("the platform's mission id for this job"), and
  reports its token. The platform keyed chat-job pushes ``chatjob:<chat>``
  and adopted only ``chatturn:``/``voice:`` reports — so the local card was
  invisible to it, froze at whatever the app last painted, and the platform
  went on updating a card of its own. Now: job cards adopt, and a job push
  also reaches a card registered under the job id.
* The widget's ContentState v2 (``phase`` / ``stepLabel`` / ``jobKind``)
  was never sent by the platform; the widget inferred a phase from the bar
  and the subtitle. Now every job/turn row carries ``phase``.
* A reordered/retried OLDER job progress row was half-applied (bar
  clamped, stale step line + n/m written over a newer state). Now it is
  dropped for that device.

Platform sweep (RUN_MODE=platform): users / devices / notification_queue /
live_activities are platform tables.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from app.config import settings
from app.db.models import (
    LA_ENDED, LA_STARTED, LiveActivity, LiveActivityDevice,
    NotificationQueue, User, NQ_QUEUED,
)
from app.services import apns_push
from app.services import live_activity_service as las
from app.services import notification_dispatcher as nd


# ── content-state contract ──────────────────────────────────────────────

def test_extra_state_accepts_the_v2_keys_and_drops_an_unknown_phase():
    out = apns_push._extra_state({
        "phase": "running", "stepLabel": "**Compare** evidence", "jobKind": "compare",
        "stepName": "Compare evidence", "jobType": "compare",
        "stepsDone": 2, "stepsTotal": 3, "percent": 67,
    })
    assert out["phase"] == "running" and out["jobKind"] == "compare"
    assert out["stepLabel"] == "Compare evidence"          # markdown stripped
    assert out["stepName"] == "Compare evidence" and out["jobType"] == "compare"
    assert apns_push._extra_state({"phase": "exploding"}) == {}
    assert apns_push._extra_state({"phase": 3}) == {}


def _row(kind: str, data: dict) -> NotificationQueue:
    return NotificationQueue(
        id=str(uuid.uuid4()), user_id="u", source="agent", event_kind=kind,
        title="t", body="b", priority="low", idempotency_key="k",
        status=NQ_QUEUED, created_at=datetime.utcnow(), data_json=data,
    )


def test_row_extra_state_carries_phase_step_label_and_kind():
    job = {"mission_id": "chatjob:s1", "kind": "job", "job_id": "j1",
           "job_type": "verify", "step_name": "Read the page", "steps_done": 1,
           "steps_total": 3, "progress": 33}
    st = las._row_extra_state(_row("progress", job))
    assert st["phase"] == "running" and st["stepLabel"] == "Read the page"
    assert st["jobKind"] == "verify" and st["stepName"] == "Read the page"
    assert las._row_extra_state(_row("mission_started", job))["phase"] == "starting"
    assert las._row_extra_state(_row("mission_completed", job))["phase"] == "completed"
    assert las._row_extra_state(_row("mission_failed", job))["phase"] == "failed"
    assert las._row_extra_state(_row("needs_approval", job))["phase"] == "needs_you"
    # the producer's explicit word wins; junk falls back to the kind
    assert las._row_extra_state(_row("progress", dict(job, phase="completed")))["phase"] == "completed"
    assert las._row_extra_state(_row("progress", dict(job, phase="nope")))["phase"] == "running"
    # a chat-turn row gets one too; a reminder row does not
    assert las._row_extra_state(_row("mission_completed", {
        "mission_id": "chatturn:abc", "kind": "chat_turn"}))["phase"] == "completed"
    assert "phase" not in las._row_extra_state(_row("mission_completed", {
        "mission_id": "reminder:r1", "kind": "reminder"}))


def test_job_id_alias_only_for_job_rows_that_address_a_different_mission():
    assert las._job_id_of(_row("progress", {"mission_id": "chatjob:s1", "kind": "job", "job_id": "j1"})) == "j1"
    assert las._job_id_of(_row("progress", {"mission_id": "j1", "kind": "job", "job_id": "j1"})) is None
    assert las._job_id_of(_row("progress", {"mission_id": "chatturn:x", "kind": "chat_turn"})) is None
    assert las._job_id_of(_row("progress", {"mission_id": "chatjob:s1", "kind": "job"})) is None


# ── LA lane — behavioural (sqlite queue, captured APNs) ─────────────────

async def _mk_user() -> str:
    from app.db import async_session_maker
    from app.services.auth_service import get_password_hash
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"r8-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="R8",
            timezone="America/Toronto",
            notification_preferences={
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
            },
        ))
        await db.commit()
    return user_id


async def _mk_device(user_id: str) -> str:
    from app.db import async_session_maker
    device_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(LiveActivityDevice(
            id=device_id, user_id=user_id,
            push_to_start_token=uuid.uuid4().hex + uuid.uuid4().hex,
            apns_environment="development", created_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        ))
        await db.commit()
    return device_id


async def _mk_local_card(user_id: str, device_id: str, mission_id: str, *, started_at=None) -> str:
    """What the app's token report leaves behind for a locally-started
    card: a STARTED row under the card's name with a per-activity token."""
    from app.db import async_session_maker
    la_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(LiveActivity(
            id=la_id, user_id=user_id, mission_id=mission_id, device_id=device_id,
            activity_push_token="a" * 64, apns_environment="development",
            status=LA_STARTED, started_at=started_at or datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ))
        await db.commit()
    return la_id


async def _enqueue(user_id: str, **overrides) -> str:
    from app.db import async_session_maker
    row_id = str(uuid.uuid4())
    fields = dict(
        id=row_id, user_id=user_id, source="agent",
        event_kind="progress", title="Working on: T", body="Reading a page…",
        priority="low", idempotency_key=f"idem-{row_id}",
        status=NQ_QUEUED, created_at=datetime.utcnow(),
        data_json={"mission_id": "chatjob:s1", "mission_title": "T", "progress": 40},
    )
    fields.update(overrides)
    async with async_session_maker() as db:
        db.add(NotificationQueue(**fields))
        await db.commit()
    return row_id


async def _dispatch(row_id: str, now: datetime = None) -> str:
    from app.db import async_session_maker
    now = now or datetime.utcnow()
    async with async_session_maker() as db:
        claimed = await nd._claim_batch(db, now)
        assert row_id in claimed, "row not claimable"
        return await nd._dispatch_row(db, row_id, now)


def _patch_apns(monkeypatch, sent: list, status: int = 200, reason: str = ""):
    async def fake_send(token, payload, *, environment="development", priority=10):
        sent.append({"token": token, "payload": payload,
                     "environment": environment, "priority": priority})
        return status, reason

    monkeypatch.setattr(las.apns_push, "send_live_activity", fake_send)
    monkeypatch.setattr(settings, "apns_key_b64", "eA==")
    monkeypatch.setattr(settings, "apns_key_id", "KEY123")
    monkeypatch.setattr(settings, "apns_team_id", "TEAM123")


async def _la_rows(user_id: str) -> list:
    from app.db import async_session_maker
    async with async_session_maker() as db:
        return list((await db.execute(
            select(LiveActivity).where(LiveActivity.user_id == user_id)
        )).scalars().all())


def _job_data(job_id: str, chat: str = "s1", **kw) -> dict:
    d = {
        "mission_id": f"chatjob:{chat}", "mission_title": "Verify the release",
        "kind": "job", "job_id": job_id, "route": "chat", "urgent": True,
        "chat_id": chat, "message_id": "m1", "job_type": "verify",
        "step_name": "Searching…", "steps_done": 0, "steps_total": 3,
    }
    d.update(kw)
    return d


@pytest.mark.asyncio
async def test_job_pushes_reach_the_card_the_app_started_under_the_job_id(monkeypatch):
    """The app-local card (named by job id) receives the chatjob:<chat>
    progress and the terminal push — over its own token — and the card
    ends. Before: no_active_activity for the local card, and a second
    platform card restarted beside it."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_device(user_id)
    job_id = str(uuid.uuid4())
    await _mk_local_card(user_id, device_id, job_id)

    prog = await _enqueue(user_id, data_json=_job_data(
        job_id, progress=67, step_name="Compare evidence", steps_done=2, phase="running"))
    assert (await _dispatch(prog)).startswith("suppressed:progress_in_app_only")
    assert len(sent) == 1 and sent[0]["token"] == "a" * 64, sent
    cs = sent[0]["payload"]["aps"]["content-state"]
    assert cs["stepsDone"] == 2 and cs["percent"] == 67 and cs["phase"] == "running"
    assert cs["stepLabel"] == "Compare evidence" and cs["jobKind"] == "verify"
    # no platform card was restarted beside it
    rows = await _la_rows(user_id)
    assert [r.mission_id for r in rows] == [job_id]

    sent.clear()
    done = await _enqueue(user_id, event_kind="mission_completed", priority="default",
                          title="✅ Done: Verify", body="Kling wins",
                          data_json=_job_data(job_id, progress=100, step_name="Done",
                                              steps_done=3, preview="Kling wins",
                                              phase="completed", dismiss_after_s=900))
    await _dispatch(done)
    assert sent and all(s["token"] == "a" * 64 for s in sent)
    upd = sent[0]["payload"]["aps"]
    assert upd["event"] in ("update", "end")
    cs = upd["content-state"]
    assert cs["phase"] == "completed" and cs["stepsDone"] == 3 and cs["preview"] == "Kling wins"
    assert (await _la_rows(user_id))[0].status == LA_ENDED


@pytest.mark.asyncio
async def test_a_start_for_a_job_whose_local_card_is_up_does_not_stack_a_second_card(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_device(user_id)
    job_id = str(uuid.uuid4())
    await _mk_local_card(user_id, device_id, job_id, started_at=datetime.utcnow() + timedelta(seconds=1))
    start = await _enqueue(user_id, event_kind="mission_started", priority="default",
                           title="🛠 Working on: Verify",
                           data_json=_job_data(job_id, refresh_if_started=True,
                                               timer_end_ms=1_900_000_000_000))
    res = await _dispatch(start)
    assert not any(s["payload"]["aps"].get("event") == "start" for s in sent), sent
    assert res.startswith("suppressed") or res == "sent"
    assert [r.mission_id for r in await _la_rows(user_id)] == [job_id]


@pytest.mark.asyncio
async def test_a_stale_job_progress_row_is_dropped_whole(monkeypatch):
    """2/3 landed; a reordered 1/3 arrives later. The bar never went
    backwards before either — but the step line and n/m did. Now the whole
    row is skipped for that device."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_device(user_id)
    job_id = str(uuid.uuid4())
    await _mk_local_card(user_id, device_id, f"chatjob:s1")
    newer = await _enqueue(user_id, data_json=_job_data(
        job_id, progress=67, step_name="Write recommendation", steps_done=2))
    await _dispatch(newer)
    assert sent[-1]["payload"]["aps"]["content-state"]["stepsDone"] == 2
    n = len(sent)
    older = await _enqueue(user_id, data_json=_job_data(
        job_id, progress=33, step_name="Compare evidence", steps_done=1))
    await _dispatch(older)
    assert len(sent) == n, "the stale row must not reach the device"
    from app.db import async_session_maker
    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, older)
    assert row.channels_json["live_activity"]["devices"][device_id]["reason"] == "stale_progress"
    # equal progress with a fresher subtitle still goes through
    same = await _enqueue(user_id, data_json=_job_data(
        job_id, progress=67, step_name="Write recommendation", steps_done=2))
    await _dispatch(same)
    assert len(sent) == n + 1


@pytest.mark.asyncio
async def test_a_missions_lower_estimate_still_updates_the_subtitle(monkeypatch):
    """The drop is JOB rows only — an autopilot mission's percent is its own
    estimate and may honestly fall; its subtitle must keep refreshing."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_device(user_id)
    await _mk_local_card(user_id, device_id, "mission-1")
    hi = await _enqueue(user_id, data_json={"mission_id": "mission-1", "mission_title": "M", "progress": 45})
    await _dispatch(hi)
    lo = await _enqueue(user_id, body="Drafting…", data_json={"mission_id": "mission-1", "mission_title": "M", "progress": 30})
    await _dispatch(lo)
    assert len(sent) == 2
    cs = sent[-1]["payload"]["aps"]["content-state"]
    assert cs["progress"] == 0.45 and cs["subtitle"] == "Drafting…"


# ── token report: job cards adopt ───────────────────────────────────────

@pytest.mark.asyncio
async def test_token_report_adopts_a_locally_started_job_card(monkeypatch):
    from app.api import live_activity_devices as lad
    from app.db import async_session_maker

    user_id = await _mk_user()
    device_id = await _mk_device(user_id)
    job_id = str(uuid.uuid4())

    class _User:  # what get_current_user yields
        id = user_id

    async with async_session_maker() as db:
        out = await lad.report_activity_token(
            lad.ActivityTokenReport(mission_id=job_id, activity_push_token="b" * 64),
            current_user=_User(), db=db,
        )
    assert out.get("adopted") is True, out
    rows = await _la_rows(user_id)
    assert len(rows) == 1 and rows[0].mission_id == job_id
    assert rows[0].status == LA_STARTED and rows[0].activity_push_token == "b" * 64
    assert rows[0].device_id == device_id

    # chatjob:<chat> (the updated app contract) adopts too; a stranger
    # prefix still does not.
    async with async_session_maker() as db:
        out2 = await lad.report_activity_token(
            lad.ActivityTokenReport(mission_id="chatjob:s9", activity_push_token="c" * 64),
            current_user=_User(), db=db,
        )
        out3 = await lad.report_activity_token(
            lad.ActivityTokenReport(mission_id="mystery:1", activity_push_token="d" * 64),
            current_user=_User(), db=db,
        )
    assert out2.get("adopted") is True
    assert out3 == {"ok": True, "updated": 0}


def test_looks_like_job_id():
    from app.api.live_activity_devices import _looks_like_job_id
    assert _looks_like_job_id(str(uuid.uuid4()))
    assert not _looks_like_job_id("chatjob:abc")
    assert not _looks_like_job_id("reminder:" + str(uuid.uuid4()))
    assert not _looks_like_job_id("ExpoLiveActivity")
