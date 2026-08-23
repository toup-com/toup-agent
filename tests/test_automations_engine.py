"""Automations engine — agent-lane rails (Round 26).

RUN_MODE=agent (automations/bindings/events/outbox/routines/triggers
are AGENT_ONLY). Listed in COVERAGE_DEBT.txt with `# agent-mode` so the
CI agent sweep runs it.

Proves, against the real tables:
  - compile: poll spec → hidden system routine + binding, disabled
  - compile: push spec → trigger row with the system-managed action
  - arm/pause flip primitives and binding.active together
  - event intake: UNIQUE (automation_id, dedupe_key) collapses replays
  - outbox: claim is single-winner; undo loses to a claim; undo wins
    inside the window
  - sweep: 3 consecutive failures → error + bindings off + ONE notice
  - flag off: routine kinds are dark (runner gate + list exclusion)
"""

import json
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationBinding, AutomationEvent, AutomationOutbox,
    Routine, Trigger, User,
)
from app.agent.automations import compiler
from app.agent.automations.spec import validate_spec
from app.agent.automations import executor

REGISTRY = {
    "jira": {
        "connector_id": "jira", "push": False, "poll": True, "floor_s": 300,
        "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {"jira__add_comment": ["w"]},
        "target_param_by_action": {"jira__add_comment": "issue_key"},
        "events": [{
            "key": "issue_created", "description": "",
            "source_tool": "jira__search_issues",
            "poll_args": {}, "items_path": "issues",
            "dedupe_field": "key",
            "fields": {"key": "key", "summary": "summary"},
        }],
    },
    "gmail": {
        "connector_id": "gmail", "push": True, "poll": False, "floor_s": 300,
        "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {}, "target_param_by_action": {},
        "events": [{
            "key": "email_received", "description": "",
            "dedupe_field": "gmail_message_id",
            "fields": {"message_id": "gmail_message_id"},
        }],
    },
    "slack": {
        "connector_id": "slack", "push": False, "poll": False, "floor_s": 300,
        "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {"slack__send_message": ["w"]},
        "target_param_by_action": {"slack__send_message": "channel"},
        "events": [],
    },
}


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Automations"))
        await db.commit()
    return uid


def _poll_spec():
    return validate_spec({
        "name": "Jira watch",
        "trigger": {"mode": "poll", "connector_id": "jira",
                    "event": "issue_created", "poll_interval_s": 300},
        "action": {"connector_id": "slack", "tool": "slack__send_message",
                   "params_template": {"channel": "{{grant.target.id}}",
                                       "text": "{{event.summary}}"},
                   "grant_id": "g-1"},
        "dedupe_key": "event.key",
        "mode": "auto",
    }, REGISTRY)


async def _mk_automation(uid: str, vspec) -> Automation:
    async with async_session_maker() as db:
        a = Automation(
            user_id=uid, name=vspec.name, status="draft",
            spec_json=json.dumps(vspec.raw, sort_keys=True),
            trigger_mode=vspec.trigger_mode,
            connector_id=vspec.trigger_connector_id,
        )
        db.add(a)
        await db.flush()
        await compiler.compile_bindings(db, a, vspec)
        await db.commit()
        return a


@pytest.mark.asyncio
async def test_poll_compile_creates_hidden_disabled_routine():
    uid = await _mk_user()
    a = await _mk_automation(uid, _poll_spec())
    async with async_session_maker() as db:
        b = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == a.id)
        )).scalar_one()
        assert b.kind == "routine"
        assert b.active is False
        routine = await db.get(Routine, b.target_id)
        assert routine.kind == "automation_poll"
        assert routine.enabled is False
        assert routine.schedule_kind == "every"
        assert routine.schedule_interval_seconds == 300
        assert (routine.config_json or {}).get("automation_id") == a.id


@pytest.mark.asyncio
async def test_push_compile_creates_system_trigger():
    uid = await _mk_user()
    vspec = validate_spec({
        "name": "Mail watch",
        "trigger": {"mode": "push", "connector_id": "gmail",
                    "event": "email_received"},
        "action": {"connector_id": "slack", "tool": "slack__send_message",
                   "params_template": {"channel": "{{grant.target.id}}",
                                       "text": "mail"},
                   "grant_id": "g-1"},
        "dedupe_key": "event.message_id",
        "mode": "auto",
    }, REGISTRY)
    a = await _mk_automation(uid, vspec)
    async with async_session_maker() as db:
        b = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == a.id)
        )).scalar_one()
        assert b.kind == "trigger"
        trigger = await db.get(Trigger, b.target_id)
        assert trigger.kind == "email_received"
        assert trigger.action == "run_automation"
        assert trigger.enabled is False
        assert (trigger.config_json or {}).get("automation_id") == a.id


@pytest.mark.asyncio
async def test_arm_and_pause_flip_primitives_and_bindings():
    uid = await _mk_user()
    a = await _mk_automation(uid, _poll_spec())
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        await compiler.set_bindings_active(db, a2, True)
        await db.commit()
        b = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == a.id)
        )).scalar_one()
        assert b.active is True
        assert (await db.get(Routine, b.target_id)).enabled is True

        await compiler.set_bindings_active(db, a2, False)
        await db.commit()
        await db.refresh(b)
        assert b.active is False
        assert (await db.get(Routine, b.target_id)).enabled is False


@pytest.mark.asyncio
async def test_event_dedupe_collapses_replays():
    uid = await _mk_user()
    a = await _mk_automation(uid, _poll_spec())
    vspec = _poll_spec()
    items = [{"key": "ENG-1", "summary": "one"},
             {"key": "ENG-2", "summary": "two"}]
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        fresh1 = await executor.ingest_items(db, a2, vspec, items)
        assert {e.dedupe_key for e in fresh1} == {"ENG-1", "ENG-2"}
        # replay + one new
        fresh2 = await executor.ingest_items(
            db, a2, vspec, items + [{"key": "ENG-3", "summary": "three"}],
        )
        assert {e.dedupe_key for e in fresh2} == {"ENG-3"}
        total = (await db.execute(
            select(AutomationEvent)
            .where(AutomationEvent.automation_id == a.id)
        )).scalars().all()
        assert len(total) == 3


@pytest.mark.asyncio
async def test_outbox_claim_single_winner_and_undo():
    uid = await _mk_user()
    a = await _mk_automation(uid, _poll_spec())
    from app.agent.automations.outbox import _claim, undo_row

    async def _stage(execute_in_s: float) -> str:
        async with async_session_maker() as db:
            row = AutomationOutbox(
                user_id=uid, automation_id=a.id,
                connector_id="slack", tool_name="slack__send_message",
                payload_json="{}", grant_id="g-1",
                idempotency_key=f"t:{uuid.uuid4()}",
                execute_after=datetime.utcnow()
                + timedelta(seconds=execute_in_s),
            )
            db.add(row)
            await db.commit()
            return row.id

    # Past the window: first claim wins, second loses.
    oid = await _stage(-1)
    async with async_session_maker() as db:
        assert await _claim(db, oid) is True
    async with async_session_maker() as db:
        assert await _claim(db, oid) is False

    # A claimed row cannot be undone.
    async with async_session_maker() as db:
        assert await undo_row(db, oid, uid) is False

    # Inside the window: undo wins, then the claim loses.
    oid2 = await _stage(30)
    async with async_session_maker() as db:
        assert await undo_row(db, oid2, uid) is True
    async with async_session_maker() as db:
        assert await _claim(db, oid2) is False
        row = await db.get(AutomationOutbox, oid2)
        assert row.status == "undone"


@pytest.mark.asyncio
async def test_auto_pause_after_three_failures_posts_one_notice():
    uid = await _mk_user()
    a = await _mk_automation(uid, _poll_spec())
    from app.agent.automations.sweep import _sweep_auto_pause
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        await compiler.set_bindings_active(db, a2, True)
        a2.status = "armed"
        a2.consecutive_failures = 3
        a2.last_error = "boom"
        await db.commit()

    n = await _sweep_auto_pause()
    assert n == 1
    async with async_session_maker() as db:
        a3 = await db.get(Automation, a.id)
        assert a3.status == "error"
        assert a3.paused_reason == "auto_failures"
        first_notice_at = a3.error_notice_at
        assert first_notice_at is not None
        b = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == a.id)
        )).scalar_one()
        assert b.active is False
        # still >= threshold and still error → the second sweep pass
        # must NOT post again (error_notice_at dedupe); status is no
        # longer 'armed' so it isn't even selected.
    n2 = await _sweep_auto_pause()
    assert n2 == 0


@pytest.mark.asyncio
async def test_flag_off_keeps_routine_kinds_dark():
    from app.agent.routines.runner import RoutineRunner
    from app.config import settings
    assert getattr(settings, "automations_enabled", False) is False
    assert RoutineRunner._kind_enabled("automation_poll", settings) is False
    assert RoutineRunner._kind_enabled("automation_schedule", settings) is False


@pytest.mark.asyncio
async def test_run_event_filter_skip_records_status():
    uid = await _mk_user()
    vspec = validate_spec({
        "name": "Filtered",
        "trigger": {"mode": "poll", "connector_id": "jira",
                    "event": "issue_created", "poll_interval_s": 300,
                    "filter": {"summary": ["urgent"]}},
        "action": {"connector_id": "slack", "tool": "slack__send_message",
                   "params_template": {"channel": "C1", "text": "x"},
                   "grant_id": "g-1"},
        "dedupe_key": "event.key",
        "mode": "auto",
    }, REGISTRY)
    a = await _mk_automation(uid, vspec)
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        fresh = await executor.ingest_items(
            db, a2, vspec, [{"key": "ENG-9", "summary": "routine chore"}],
        )
        assert len(fresh) == 1
        status = await executor.run_event(db, a2, vspec, fresh[0])
        assert status == "skipped_filter"
        row = await db.get(AutomationEvent, fresh[0].id)
        assert row.status == "skipped_filter"
        assert row.job_id is None
