# agent-mode: automation_notifications/threads/turns are AGENT_ONLY.
"""The notification pipeline — CONTRACTS-R30 §4.10 proof (R30-A).

One notification per run (deduped by run_id+kind), ONE day-chat card
message updated in place, the same `body` on the card and the push,
LA mission ids under the `autorun:` namespace with deep-link data,
`question`/`setup` runs never notifying, and the five-serializer
parity for the `automation_notification` card key.
"""

import json
import uuid

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationNotification, BuildJob, Message, User,
)

from tests.test_run_ledger_v3 import (  # noqa: F401 — shared fixtures
    REGISTRY_V2, _ISSUES, _OK, _fire, _mk_automation_v2, _mk_user,
    _one_run, _v2_spec,
)


@pytest.fixture()
def _notify_spy(monkeypatch):
    calls = []

    async def _notify(**kwargs):
        calls.append(kwargs)
        return "outbox-row"

    monkeypatch.setattr(
        "app.services.agent_notify_client.notify", _notify,
    )
    return calls


@pytest.mark.asyncio
async def test_one_notification_per_run_same_body_card_and_push(
    monkeypatch, _notify_spy,
):
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec)
    job = await _one_run(a.id)

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationNotification).where(
                AutomationNotification.run_id == job.id,
            )
        )).scalars().all()
        run_rows = [r for r in rows if r.kind == "automation_run"]
        assert len(run_rows) == 1
        row = run_rows[0]
        assert row.status == "completed"
        assert row.body, "the terminal fills the body"
        assert row.thread_id
        assert row.message_id, "the in-chat card message exists"

        # ONE card message, updated in place — never a second row.
        msgs = (await db.execute(
            select(Message).where(
                Message.metadata_json.like("%automation_notification%"),
            )
        )).scalars().all()
        card_msgs = [
            m for m in msgs
            if json.loads(m.metadata_json or "{}")
            .get("automation_notification", {}).get("run_id") == job.id
        ]
        assert len(card_msgs) == 1
        card = json.loads(card_msgs[0].metadata_json)[
            "automation_notification"]
        assert card["status"] == "completed"
        assert card["body"] == row.body
        assert card["id"] == row.id

    # Push: LA start at run start (silent), terminal with the SAME body.
    started = [c for c in _notify_spy
               if c.get("event_kind") == "mission_started"]
    done = [c for c in _notify_spy
            if c.get("event_kind") in ("mission_completed",
                                       "mission_failed")]
    assert len(started) == 1 and len(done) == 1
    assert started[0]["data"]["mission_id"] == f"autorun:{job.id}"
    assert done[0]["body"] == row.body
    data = done[0]["data"]
    assert data["automation_id"] == a.id
    assert data["run_id"] == job.id
    assert data["thread_id"] == row.thread_id
    assert data["route"] == "automation"


@pytest.mark.asyncio
async def test_question_and_setup_runs_never_notify(monkeypatch,
                                                    _notify_spy):
    from app.agent.automations import run_v3
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        for kind in ("question", "setup"):
            job = BuildJob(
                id=str(uuid.uuid4()), user_id=uid, title="q", prompt="",
                job_type="automation_run", status="running",
                source_kind="automation", source_id=a.id,
            )
            db.add(job)
            await db.commit()
            await run_v3.open_run(
                db, automation=a2, job=job, kind=kind, total_steps=1,
            )
        rows = (await db.execute(
            select(AutomationNotification).where(
                AutomationNotification.automation_id == a.id,
            )
        )).scalars().all()
        assert rows == []
    assert _notify_spy == []


@pytest.mark.asyncio
async def test_notification_dedupe_is_run_and_kind():
    from app.agent.automations import run_v3
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        job = BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="r", prompt="",
            job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
        )
        db.add(job)
        await db.commit()
        first = await run_v3._mint_notification(
            db, automation=a2, job=job, kind="automation_run",
            thread_id=None, status="running",
        )
        second = await run_v3._mint_notification(
            db, automation=a2, job=job, kind="automation_run",
            thread_id=None, status="running",
        )
        assert first is not None and second is not None
        assert first.id == second.id


def test_card_key_rides_all_five_serializers():
    """The allowlist trap (R29-A): a card key missing anywhere vanishes
    on a fallback path. Source-probe all five sites."""
    from app.schemas import ChatMessageResponse
    assert "automation_notification" in ChatMessageResponse.model_fields

    import inspect
    from app.api import day_chats, messages_recover, sessions
    for mod, fn in ((sessions, "_message_to_response"),
                    (messages_recover, "messages_since"),
                    (day_chats, "get_day_chat_messages")):
        src = inspect.getsource(getattr(mod, fn))
        count = src.count('"automation_notification"')
        expected = 2 if mod is day_chats else 1
        assert count >= expected, (
            f"{mod.__name__}.{fn} misses automation_notification "
            f"({count} < {expected})"
        )


def test_notification_body_fallback_never_a_finding():
    from app.agent.automations.run_v3 import _notification_body
    body = _notification_body("automation_run", {
        "run_kind": "scheduled", "status": "completed",
        "needs_count": 1, "writes_count": 0,
    })
    assert "open the run" in body
    assert "TP-" not in body
    body2 = _notification_body("automation_needs_you", {"status": "waiting_on_user"})
    assert "waiting" in body2.lower()
