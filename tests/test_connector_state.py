# agent-mode
"""Connector state frames + §4.7 auto-resume (Round 30, R30-A).

RUN_MODE=agent (automations / automation_threads / automation_turns /
build_jobs are AGENT_ONLY — init_db() does not create them under
platform, so the platform lane fails with "no such table" as a
mis-invocation, not a defect). Listed in COVERAGE_DEBT.txt with
`# agent-mode` so the CI agent sweep runs it.

Proves, against real rows:
  - the `connector.state` frame shape: `account_id == connector_id`
    verbatim, state, reconnected_at ISO, and NO `channel` key
    (broadcast_to_user monkeypatched and captured);
  - `on_connector_connected` resumes a checkpointed stopped run
    (cancelled + outcome "stopped") through `run_v3.resume_run`
    (monkeypatched to record the call) and appends the RECONNECTED
    note (`stamp: reconnected`) to the automation's thread with the
    stopped run's id;
  - a paused `connector_reauth` automation with no stopped run is
    re-armed through `service.arm_automation` (monkeypatched); one
    with no thread is re-armed but skipped silently from `noted`;
    an automation that never references the connector is untouched;
  - `on_connector_expired` emits the `expired` frame and pauses
    NOTHING — the run-level failure path owns pausing;
  - R39: it does ANNOUNCE — exactly one deduped `needs_you` turn on
    each armed automation's thread (a Friday-evening expiry used to
    surface nothing until Monday's run), and a repeat expiry while
    that card is still the thread's most recent needs_you stacks no
    duplicate.
"""

import json
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select

from app.agent.automations import connector_state, ledger
from app.db.database import async_session_maker
from app.db.models import Automation, AutomationTurn, BuildJob, User


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Connector State"))
        await db.commit()
    return uid


async def _mk_automation(
    uid: str,
    *,
    connector_id: str = "gmail",
    status: str = "armed",
    paused_reason=None,
) -> Automation:
    """A minimal automation row. The spec is hand-rolled JSON — this
    file drives connector_state's reference check, not the compiler."""
    spec = {
        "name": "Mail brief",
        "trigger": {"mode": "push", "connector_id": connector_id,
                    "event": "email_received"},
    }
    async with async_session_maker() as db:
        a = Automation(
            user_id=uid, name="Mail brief", status=status,
            paused_reason=paused_reason,
            spec_json=json.dumps(spec, sort_keys=True),
            trigger_mode="push", connector_id=connector_id,
        )
        db.add(a)
        await db.commit()
        await db.refresh(a)
        return a


async def _mk_stopped_run(uid: str, automation_id: str) -> BuildJob:
    """The §4.3 stop terminal: cancelled + outcome stopped + checkpoint."""
    async with async_session_maker() as db:
        job = BuildJob(
            user_id=uid, title="Mail brief run", prompt="",
            job_type="automation_run", status="cancelled",
            outcome="stopped", source_id=automation_id,
            checkpoint_json=json.dumps({"step_index": 1}),
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job


def _capture_frames(monkeypatch) -> list:
    frames: list = []

    async def _fake_broadcast(user_id, frame):
        frames.append((user_id, frame))

    monkeypatch.setattr("app.api.ws_chat.broadcast_to_user", _fake_broadcast)
    return frames


@pytest.mark.asyncio
async def test_state_frame_shape_and_no_channel_key(monkeypatch):
    frames = _capture_frames(monkeypatch)
    at = datetime(2026, 8, 25, 12, 0, 0)
    await connector_state.emit_state_frame(
        "u-1", connector_id="gmail", state="connected", reconnected_at=at,
    )
    assert len(frames) == 1
    uid, frame = frames[0]
    assert uid == "u-1"
    assert frame == {
        "type": "connector.state",
        "account_id": "gmail",       # == connector_id verbatim (§1)
        "connector_id": "gmail",
        "state": "connected",
        "reconnected_at": "2026-08-25T12:00:00Z",
        # R31 §4.4 — additive, and both EMPTY on the reconnect leg: this
        # account is simply working, so there is nothing to name and
        # nothing to offer.
        "reason_code": "",
        "fix": "",
    }
    assert "channel" not in frame


@pytest.mark.asyncio
async def test_a_healthy_account_offers_no_fix_and_a_transient_one_offers_retry(
    monkeypatch,
):
    """§4.4. `buttonLabel` renders ANY non-empty `fix`, so a blanket
    default of `retry` for every `connected` frame drew "Try again" on an
    account that had just started working — R31-13's class, inverted.

    Only the three transient reason codes keep the state `connected` AND
    earn a remedy.
    """
    for reason, expected in (
        ("", ""),                       # the reconnect leg — healthy
        ("rate_limited", "retry"),
        ("vendor_down", "retry"),
        ("timeout", "retry"),
        # A credential problem may not present as `connected` at all, but
        # if a caller ever does, it must not be offered a bare retry.
        ("token_expired", ""),
    ):
        frames = _capture_frames(monkeypatch)
        await connector_state.emit_state_frame(
            "u-1", connector_id="gmail", state="connected", reason_code=reason,
        )
        assert frames[0][1]["fix"] == expected, (reason, frames[0][1]["fix"])

    # An account that is NOT connected still defaults to reconnect.
    frames = _capture_frames(monkeypatch)
    await connector_state.emit_state_frame(
        "u-1", connector_id="gmail", state="expired",
    )
    assert frames[0][1]["fix"] == "reconnect"

    # An explicit fix always wins over the default.
    frames = _capture_frames(monkeypatch)
    await connector_state.emit_state_frame(
        "u-1", connector_id="gmail", state="scope_missing", fix="grant",
    )
    assert frames[0][1]["fix"] == "grant"


@pytest.mark.asyncio
async def test_connected_resumes_stopped_run_and_appends_note(monkeypatch):
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    a = await _mk_automation(uid, connector_id="gmail")
    job = await _mk_stopped_run(uid, a.id)
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(db, user_id=uid, automation_id=a.id)
        await db.commit()
        thread_id = thread.id

    resumed_calls = []

    async def _fake_resume(db, *, job_id):
        resumed_calls.append(job_id)
        return {"resumed": True, "status": "completed"}

    monkeypatch.setattr(
        "app.agent.automations.run_v3.resume_run", _fake_resume,
    )

    async with async_session_maker() as db:
        result = await connector_state.on_connector_connected(
            db, user_id=uid, connector_id="gmail",
        )

    assert resumed_calls == [job.id]
    assert result["resumed"] == [job.id]
    assert result["noted"] == [a.id]
    assert result["rearmed"] == []

    # The connected frame went out first, before any recovery work.
    assert frames[0][1]["type"] == "connector.state"
    assert frames[0][1]["state"] == "connected"
    assert frames[0][1]["reconnected_at"] is not None

    # The RECONNECTED note is on the thread, linked to the stopped run.
    async with async_session_maker() as db:
        turns = (await db.execute(
            select(AutomationTurn)
            .where(AutomationTurn.thread_id == thread_id)
            .order_by(AutomationTurn.seq)
        )).scalars().all()
        notes = [t for t in turns if t.kind == "note"]
        assert len(notes) == 1
        body = json.loads(notes[0].payload_json)
        assert body["stamp"] == "reconnected"
        assert body["at"]  # ISO stamp present
        assert notes[0].run_id == job.id


@pytest.mark.asyncio
async def test_connected_rearms_paused_reauth_and_skips_strangers(monkeypatch):
    _capture_frames(monkeypatch)
    uid = await _mk_user()
    # Blocked on gmail, has a thread → noted + re-armed.
    a_threaded = await _mk_automation(
        uid, connector_id="gmail", status="paused",
        paused_reason="connector_reauth",
    )
    # Blocked on gmail, NO thread → re-armed, silently absent from noted.
    a_bare = await _mk_automation(
        uid, connector_id="gmail", status="error",
        paused_reason="connector_reauth",
    )
    # Paused on a DIFFERENT connector → untouched.
    a_other = await _mk_automation(
        uid, connector_id="jira", status="paused",
        paused_reason="connector_reauth",
    )
    async with async_session_maker() as db:
        await ledger.ensure_thread(db, user_id=uid, automation_id=a_threaded.id)
        await db.commit()

    armed_calls = []

    async def _fake_arm(db, *, automation_id, user_id):
        armed_calls.append(automation_id)
        return None

    monkeypatch.setattr(
        "app.agent.automations.service.arm_automation", _fake_arm,
    )

    async with async_session_maker() as db:
        result = await connector_state.on_connector_connected(
            db, user_id=uid, connector_id="gmail",
        )

    assert sorted(armed_calls) == sorted([a_threaded.id, a_bare.id])
    assert sorted(result["rearmed"]) == sorted([a_threaded.id, a_bare.id])
    assert result["noted"] == [a_threaded.id]
    assert result["resumed"] == []
    assert a_other.id not in result["rearmed"]
    assert a_other.id not in result["noted"]


@pytest.mark.asyncio
async def test_expired_emits_frame_and_pauses_nothing(monkeypatch):
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    a = await _mk_automation(uid, connector_id="gmail", status="armed")

    async with async_session_maker() as db:
        await connector_state.on_connector_expired(
            db, user_id=uid, connector_id="gmail",
        )

    assert len(frames) == 1
    frame = frames[0][1]
    assert frame["type"] == "connector.state"
    assert frame["state"] == "expired"
    assert frame["reconnected_at"] is None
    assert "channel" not in frame

    # The run-level failure path owns pausing — the armed row is intact.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert row.status == "armed"
        assert row.paused_reason is None


@pytest.mark.asyncio
async def test_expired_announces_once_on_the_armed_thread(monkeypatch):
    """R39: `paused_reason="connector_reauth"` has NO writer, so the
    "run-level failure path" the expiry leg deferred to only speaks when
    a run fires — a Friday-evening expiry surfaced nothing until Monday
    8:00. The hook now appends ONE `needs_you` turn per armed automation
    on that connector, broadcast like any other turn, and a repeat
    expiry while that card is still the thread's most recent needs_you
    stacks no duplicate."""
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    a = await _mk_automation(uid, connector_id="gmail", status="armed")
    # A paused sibling and an armed stranger both stay silent.
    a_paused = await _mk_automation(uid, connector_id="gmail",
                                    status="paused")
    a_other = await _mk_automation(uid, connector_id="jira", status="armed")
    threads = {}
    async with async_session_maker() as db:
        for row in (a, a_paused, a_other):
            t = await ledger.ensure_thread(
                db, user_id=uid, automation_id=row.id)
            threads[row.id] = t.id
        await db.commit()

    async with async_session_maker() as db:
        await connector_state.on_connector_expired(
            db, user_id=uid, connector_id="gmail", error="reauth_required",
        )

    async def _turns_of(thread_id):
        async with async_session_maker() as db:
            return (await db.execute(
                select(AutomationTurn)
                .where(AutomationTurn.thread_id == thread_id)
                .order_by(AutomationTurn.seq)
            )).scalars().all()

    turns = await _turns_of(threads[a.id])
    assert [t.kind for t in turns] == ["needs_you"]
    body = json.loads(turns[0].payload_json)
    assert body["account_id"] == "gmail"
    assert body["connector_id"] == "gmail"
    assert body["fix"] == "reconnect"          # a credential problem
    assert body["name"] and body["sentence"]   # named, in words
    assert turns[0].run_id is None             # no run fired — that is the point
    for silent in (a_paused.id, a_other.id):
        assert await _turns_of(threads[silent]) == []

    # Broadcast rode the same wire as every turn, attributed.
    turn_frames = [f for _, f in frames if f["type"] == "automation.turn"]
    assert len(turn_frames) == 1
    assert turn_frames[0]["automation_id"] == a.id
    assert turn_frames[0]["turn"]["kind"] == "needs_you"

    # A second expiry event while the card is still current: no stack.
    async with async_session_maker() as db:
        await connector_state.on_connector_expired(
            db, user_id=uid, connector_id="gmail", error="reauth_required",
        )
    assert len(await _turns_of(threads[a.id])) == 1

    # A TRANSIENT flip announces nothing — the account stays `connected`
    # and a reconnect card for a vendor's bad minute is the §4.4
    # inversion.
    async with async_session_maker() as db:
        await connector_state.on_connector_expired(
            db, user_id=uid, connector_id="jira", error="provider_down",
        )
    assert await _turns_of(threads[a_other.id]) == []
