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
from datetime import datetime

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


@pytest.mark.asyncio
async def test_notification_body_is_total_over_the_v3_statuses():
    """AUDIT-11: every status `run_v3_status` can return has a sentence.

    The table knew `waiting_on_user` and `failed`; the other six fell
    through to the closing line, so a run the user stopped, a run a
    newer one superseded and a run that found nothing to do all pushed
    "It ran on time. Nothing needs you" — the notification asserting the
    opposite of what happened.
    """
    from app.agent.automations.run_v3 import _notification_body

    v3_statuses = ("running", "waiting_on_user", "completed", "partial",
                   "superseded", "stopped_by_user", "skipped", "failed")
    bodies = {}
    for status in v3_statuses:
        body = _notification_body("automation_run", {
            "run_kind": "scheduled", "status": status,
            "needs_count": 0, "writes_count": 0,
        })
        assert body and body.strip(), f"{status}: no body"
        bodies[status] = body

    # The three that used to lie must not claim a clean on-time run.
    for status in ("stopped_by_user", "superseded", "skipped"):
        assert "ran on time" not in bodies[status], (
            f"{status} claims it ran on time: {bodies[status]!r}")
        assert "Nothing needs you" not in bodies[status]
    assert "stopped it" in bodies["stopped_by_user"]
    assert "newer run" in bodies["superseded"]
    assert "nothing to do" in bodies["skipped"]
    # A partial run does not claim a whole one.
    assert "ran on time" not in bodies["partial"]
    # Distinct statuses get distinct sentences — a table that collapses
    # them is the defect in a different shape.
    assert len(set(bodies.values())) >= 6


@pytest.mark.asyncio
async def test_confirm_park_reaches_the_notification_pipeline(
    monkeypatch, _notify_spy,
):
    """AUDIT-12: the park is a state §4.10 must hear.

    `_park_run_on_card` writes `job.status = "waiting_on_user"` straight
    onto the row — no finalize gate runs. Without the post-commit hop
    the run that is waiting for the user was the one run that never told
    them: no `automation_needs_you` row, no card, no push.
    """
    from app.agent.automations.outbox import _park_run_on_card
    from app.db.models import AutomationOutbox

    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec)
    job = await _one_run(a.id)
    _notify_spy.clear()

    async with async_session_maker() as db:
        row = AutomationOutbox(
            id=str(uuid.uuid4()), user_id=uid, automation_id=a.id,
            job_id=job.id, connector_id="slack",
            tool_name="slack__send_message",
            payload_json=json.dumps({}), status="staged",
            idempotency_key=str(uuid.uuid4()),
            execute_after=datetime.utcnow(),
        )
        db.add(row)
        await db.commit()
        await _park_run_on_card(db, row, {
            "action_id": "act-1",
            "summary": "Post the digest to #platform",
        })

    async with async_session_maker() as db:
        parked = await db.get(BuildJob, job.id)
        assert parked.status == "waiting_on_user"
        rows = (await db.execute(
            select(AutomationNotification).where(
                AutomationNotification.run_id == job.id,
            )
        )).scalars().all()
        kinds = {r.kind for r in rows}
        assert "automation_needs_you" in kinds, (
            "the park minted no needs-you notification")
        run_row = next(r for r in rows if r.kind == "automation_run")
        assert run_row.status == "waiting_on_user"
        # A park is not a finished run: it must not claim 100%.
        assert run_row.fraction < 100
        assert "waiting" in (run_row.body or "").lower()

        # The card message the app renders.
        msgs = (await db.execute(
            select(Message).where(Message.source == "automation")
        )).scalars().all()
        assert any("automation_notification" in (m.metadata_json or "")
                   for m in msgs), "no chat card for the park"

    # ...and the push went out under its OWN dedup key, so the later
    # completion notice is not suppressed as a duplicate.
    park_pushes = [c for c in _notify_spy
                   if str(c.get("dedup_key", "")).endswith(":park")]
    assert park_pushes, f"no park push in {[c.get('dedup_key') for c in _notify_spy]}"
    assert park_pushes[0]["event_kind"] == "needs_approval"
