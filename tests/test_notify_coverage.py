"""Feature 2 — nothing silently misses the phone.

Contracts under test:
  - evaluate_policy: cap_exempt/urgent bypass the daily cap; the
    autopilot_push pref only suppresses mission-kind rows (data.kind
    chat_turn/reminder/routine/job flows through); quiet hours still
    defer non-urgent rows.
  - ws_chat: the 'Answer ready' push is unconditional (the
    done-delivered + single-WS gate is gone) and cap-exempt.
  - RoutineRunner._post_terminal: every non-autopilot fire notifies —
    mission_completed on success (except reminders, which self-notify
    at fire time in ReminderHandler), mission_failed on
    failed/skipped_reauth.
  - JobRunner._mark_failed: runner-internal job deaths notify once.
  - Dispatcher fallback: data.no_agent_fallback suppresses the agent
    telegram/whatsapp re-delivery (producers that already fanned out).
  - All new payloads pass the platform ingest validator (unknown kinds
    or non-scalar data would poison the outbox with permanent rejects).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.db.models import NotificationQueue
from app.services import notification_dispatcher as nd


# ── notify() spy (same shape as test_autopilot_engine) ────────────


@pytest.fixture
def notify_calls(monkeypatch):
    calls = []

    async def fake_notify(**kwargs):
        calls.append(kwargs)
        return "nid"

    import app.services.agent_notify_client as anc
    monkeypatch.setattr(anc, "notify", fake_notify)
    return calls


# ── evaluate_policy ───────────────────────────────────────────────


def _prefs(**overrides):
    from app.api.account import _merged_prefs
    base = _merged_prefs(None)
    base.update(overrides)
    return base


_DAYTIME = datetime(2026, 7, 17, 15, 0)  # outside default quiet hours


def test_cap_exempt_bypasses_daily_cap():
    row = NotificationQueue(
        event_kind="mission_completed", priority="high",
        data_json={"kind": "chat_turn", "cap_exempt": True},
    )
    decision, _ = nd.evaluate_policy(
        row, _prefs(daily_push_cap=3), "UTC", _DAYTIME, 99, False,
    )
    assert decision == "send"


def test_urgent_bypasses_daily_cap():
    row = NotificationQueue(
        event_kind="mission_completed", priority="high",
        data_json={"kind": "reminder", "urgent": True},
    )
    decision, _ = nd.evaluate_policy(
        row, _prefs(daily_push_cap=3), "UTC", _DAYTIME, 99, False,
    )
    assert decision == "send"


def test_daily_cap_still_applies_without_exemptions():
    row = NotificationQueue(event_kind="mission_completed", priority="default")
    decision, reason = nd.evaluate_policy(
        row, _prefs(daily_push_cap=3), "UTC", _DAYTIME, 3, False,
    )
    assert (decision, reason) == ("suppress", "daily_cap")


@pytest.mark.parametrize("data_kind", ["chat_turn", "reminder", "routine", "job"])
def test_autopilot_pref_never_suppresses_other_producers(data_kind):
    row = NotificationQueue(
        event_kind="mission_completed", priority="high",
        data_json={"kind": data_kind, "urgent": True},
    )
    decision, _ = nd.evaluate_policy(
        row, _prefs(autopilot_push=False), "UTC", _DAYTIME, 0, False,
    )
    assert decision == "send"


@pytest.mark.parametrize("data_json", [None, {"kind": "mission"}])
def test_autopilot_pref_still_suppresses_mission_rows(data_json):
    row = NotificationQueue(
        event_kind="mission_completed", priority="default", data_json=data_json,
    )
    decision, reason = nd.evaluate_policy(
        row, _prefs(autopilot_push=False), "UTC", _DAYTIME, 0, False,
    )
    assert (decision, reason) == ("suppress", "autopilot_push_disabled")


def test_quiet_hours_still_defer_non_urgent_and_urgent_bypasses():
    now_utc = datetime(2026, 7, 8, 3, 0)  # 23:00 EDT previous day
    prefs = _prefs(quiet_hours={"enabled": True, "start": "22:00", "end": "08:00"})

    row = NotificationQueue(
        event_kind="mission_completed", priority="default",
        data_json={"kind": "routine"},
    )
    decision, until = nd.evaluate_policy(row, prefs, "America/Toronto", now_utc, 0, False)
    assert decision == "defer" and until is not None

    urgent = NotificationQueue(
        event_kind="mission_completed", priority="high",
        data_json={"kind": "reminder", "urgent": True},
    )
    decision, _ = nd.evaluate_policy(urgent, prefs, "America/Toronto", now_utc, 0, False)
    assert decision == "send"


# ── ws_chat: unconditional 'Answer ready' ─────────────────────────


def test_answer_push_gate_removed_and_cap_exempt():
    import inspect
    from app.api import ws_chat

    src = inspect.getsource(ws_chat)
    # The presence gate is gone — every turn's answer notifies.
    assert "if not _done_delivered and len(_user_ws_queues" not in src
    # Spam brakes on the payload.
    assert '"cap_exempt": True' in src
    assert 'dedup_key=f"{_turn_mission_id}:completed"' in src


# ── RoutineRunner._post_terminal — the one funnel ─────────────────


def _mk_runner(monkeypatch):
    from app.agent.routines.runner import RoutineRunner

    runner = RoutineRunner()

    async def no_nudge(routine, run_id, **kw):
        return None, {}

    async def no_finalize(run_id, **kw):
        return None

    async def no_sync(routine_id):
        return None

    monkeypatch.setattr(runner, "_write_nudge", no_nudge)
    monkeypatch.setattr(runner, "_finalize_run", no_finalize)
    monkeypatch.setattr(runner, "_sync_next_run", no_sync)
    return runner


def _routine(kind: str, **kw):
    return SimpleNamespace(
        id=str(uuid.uuid4()), user_id=str(uuid.uuid4()), kind=kind,
        name=kw.pop("name", "Morning briefing"),
        auto_disable_after_fire=False, config_json=None, **kw,
    )


def _result(status: str, **kw):
    from app.agent.routines.base_handler import RoutineResult
    return RoutineResult(status=status, **kw)


@pytest.mark.asyncio
async def test_post_terminal_success_notifies_routine_kinds(monkeypatch, notify_calls):
    runner = _mk_runner(monkeypatch)
    routine = _routine("email_briefing")
    run_id = str(uuid.uuid4())

    await runner._post_terminal(routine, run_id, _result("success", emails_fetched=3))

    assert len(notify_calls) == 1
    call = notify_calls[0]
    assert call["event_kind"] == "mission_completed"
    assert call["priority"] == "high"
    assert call["title"].startswith("✅")
    assert call["dedup_key"] == f"{routine.id}:fired:{run_id}"
    data = call["data"]
    assert data["mission_id"] == f"routine:{routine.id}"
    assert data["kind"] == "routine"
    assert data["urgent"] is True
    assert data["cap_exempt"] is True
    assert data["no_agent_fallback"] is True


@pytest.mark.asyncio
async def test_post_terminal_reminder_success_is_silent_here(monkeypatch, notify_calls):
    """Reminder SUCCESS self-notifies at fire time inside ReminderHandler
    (its payload flips the countdown card) — a second row here would
    double-push."""
    runner = _mk_runner(monkeypatch)
    routine = _routine("reminder", name="Stretch")

    await runner._post_terminal(routine, str(uuid.uuid4()), _result("success"))
    assert notify_calls == []


@pytest.mark.asyncio
async def test_post_terminal_failure_notifies_with_reminder_mission_id(
    monkeypatch, notify_calls,
):
    runner = _mk_runner(monkeypatch)
    routine = _routine("reminder", name="Stretch")

    await runner._post_terminal(
        routine, str(uuid.uuid4()),
        _result("failed", error_class="Boom", error_detail="exploded"),
    )
    assert len(notify_calls) == 1
    call = notify_calls[0]
    assert call["event_kind"] == "mission_failed"
    assert "didn't run" in call["title"]
    # Same mission_id namespace as the countdown card so a failure still
    # closes it out.
    assert call["data"]["mission_id"] == f"reminder:{routine.id}"
    assert call["data"]["urgent"] is False
    assert call["data"]["no_agent_fallback"] is True
    assert call["dedup_key"].startswith(f"{routine.id}:failed:")


@pytest.mark.asyncio
async def test_post_terminal_skipped_reauth_notifies(monkeypatch, notify_calls):
    runner = _mk_runner(monkeypatch)
    routine = _routine("email_briefing")

    await runner._post_terminal(
        routine, str(uuid.uuid4()),
        _result("skipped_reauth", error_class="reauth", error_detail="expired"),
    )
    assert [c["event_kind"] for c in notify_calls] == ["mission_failed"]


@pytest.mark.asyncio
async def test_post_terminal_autopilot_never_double_notifies(monkeypatch, notify_calls):
    runner = _mk_runner(monkeypatch)
    routine = _routine("autopilot")

    await runner._post_terminal(routine, str(uuid.uuid4()), _result("success"))
    await runner._post_terminal(
        routine, str(uuid.uuid4()), _result("failed", error_class="X"),
    )
    assert notify_calls == []


@pytest.mark.asyncio
async def test_post_terminal_notify_failure_never_raises(monkeypatch):
    async def broken_notify(**kwargs):
        raise RuntimeError("outbox down")

    import app.services.agent_notify_client as anc
    monkeypatch.setattr(anc, "notify", broken_notify)

    runner = _mk_runner(monkeypatch)
    routine = _routine("email_briefing")
    # Must not raise — notify plumbing never fails a delivered run.
    await runner._post_terminal(routine, str(uuid.uuid4()), _result("success"))


# ── JobRunner._mark_failed ────────────────────────────────────────


async def _mk_job(status: str = "running") -> str:
    from app.db import async_session_maker
    from app.db.models import BuildJob, User
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"jr-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="JR",
        ))
        db.add(BuildJob(
            id=job_id, user_id=user_id, title="Fetch the data",
            prompt="fetch", job_type="agent_task", status=status,
        ))
        await db.commit()
    return job_id


@pytest.mark.asyncio
async def test_mark_failed_notifies_phone(notify_calls):
    from app.agent.job_runner import JobRunner

    job_id = await _mk_job()
    await JobRunner()._mark_failed(job_id, "handler crashed")

    assert len(notify_calls) == 1
    call = notify_calls[0]
    assert call["event_kind"] == "mission_failed"
    assert call["dedup_key"] == f"{job_id}:failed"
    assert call["data"]["mission_id"] == job_id
    assert call["data"]["kind"] == "job"
    assert "Fetch the data" in call["title"]


@pytest.mark.asyncio
async def test_mark_failed_skips_notify_when_already_terminal(notify_calls):
    from app.agent.job_runner import JobRunner

    job_id = await _mk_job(status="completed")
    await JobRunner()._mark_failed(job_id, "late failure report")
    assert notify_calls == []


# ── Dispatcher fallback: no_agent_fallback suppression ────────────


async def _mk_nq_user(**prefs) -> str:
    from app.db import async_session_maker
    from app.db.models import User
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"nc-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="NC",
            timezone="UTC",
            notification_preferences={
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
                **prefs,
            },
        ))
        await db.commit()
    return user_id


async def _enqueue_row(user_id: str, data_json) -> str:
    from app.db import async_session_maker

    row_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(NotificationQueue(
            id=row_id, user_id=user_id, source="agent",
            event_kind="mission_completed", title="⏰ Stretch — now",
            body="Time to stretch", priority="high",
            idempotency_key=f"idem-{row_id}", status="queued",
            created_at=datetime.utcnow(), data_json=data_json,
        ))
        await db.commit()
    return row_id


async def _dispatch(row_id: str) -> str:
    from app.db import async_session_maker

    now = datetime.utcnow()
    async with async_session_maker() as db:
        claimed = await nd._claim_batch(db, now)
        assert row_id in claimed
        return await nd._dispatch_row(db, row_id, now)


@pytest.mark.asyncio
async def test_no_agent_fallback_flag_skips_channel_redelivery(monkeypatch):
    fallback_calls = []

    async def spy_fallback(db, row):
        fallback_calls.append(row.id)
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", spy_fallback)

    user_id = await _mk_nq_user()
    # No mission_id → LA lane declines; no devices → no push lane; the
    # ONLY remaining surface is the agent fallback, which the producer
    # opted out of (it already broadcast to telegram/whatsapp itself).
    flagged = await _enqueue_row(
        user_id, {"kind": "reminder", "urgent": True, "no_agent_fallback": True},
    )
    await _dispatch(flagged)
    assert fallback_calls == []

    unflagged = await _enqueue_row(user_id, {"kind": "reminder", "urgent": True})
    await _dispatch(unflagged)
    assert fallback_calls == [unflagged]


# ── Ingest contract pins for every new payload ────────────────────


def _ingest(**kw):
    from app.api.agent_notify import AgentNotifyRequest

    return AgentNotifyRequest(
        user_id="u" * 36, idempotency_key=f"row-{uuid.uuid4().hex[:12]}", **kw,
    )


def test_new_producer_payloads_pass_ingest_validation():
    rid = str(uuid.uuid4())

    routine_fired = _ingest(
        event_kind="mission_completed", title="✅ Morning briefing",
        data={"mission_id": f"routine:{rid}", "mission_title": "Morning briefing",
              "kind": "routine", "route": "chat", "urgent": True,
              "cap_exempt": True, "no_agent_fallback": True,
              "progress": 100, "dismiss_after_s": 3600},
        priority="high", dedup_key=f"{rid}:fired:run-1",
    )
    assert routine_fired.data["cap_exempt"] is True

    routine_failed = _ingest(
        event_kind="mission_failed", title="⚠️ Morning briefing didn't run",
        data={"mission_id": f"routine:{rid}", "mission_title": "Morning briefing",
              "kind": "routine", "route": "chat", "urgent": False,
              "no_agent_fallback": True, "dismiss_after_s": 900},
        priority="default", dedup_key=f"{rid}:failed:2026-07-17",
    )
    assert routine_failed.event_kind == "mission_failed"

    job_failed = _ingest(
        event_kind="mission_failed", title="⚠️ Fetch the data failed",
        body="RuntimeError('x')",
        data={"route": "mission-control", "mission_id": rid,
              "mission_title": "Fetch the data", "kind": "job",
              "urgent": True, "dismiss_after_s": 900},
        priority="high", dedup_key=f"{rid}:failed",
    )
    assert job_failed.data["kind"] == "job"
