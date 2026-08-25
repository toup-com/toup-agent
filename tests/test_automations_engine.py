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


# ═════════════════════════════════════════════════════════════════════
# Round 28 — spec v2: multi-source, steps, memory, fast-lane
# ═════════════════════════════════════════════════════════════════════

REGISTRY_V2 = {
    "jira": REGISTRY["jira"],
    "gmail": REGISTRY["gmail"],
    "slack": REGISTRY["slack"],
    "teams": {
        "connector_id": "teams", "push": False, "poll": True, "floor_s": 300,
        "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {"teams__send_chat_message": ["w"]},
        "target_param_by_action": {"teams__send_chat_message": "chat_id"},
        "events": [{
            "key": "chat_message_received", "description": "",
            "source_tool": "teams__read_chat_messages",
            "poll_args": {"max_results": 25},
            "params_required": ["chat_id"],
            "items_path": "messages", "dedupe_field": "id",
            "fields": {"id": "id", "body": "body"},
        }],
    },
}


def _v2_spec(**over):
    spec = {
        "version": 2,
        "name": "Brief v2",
        "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
            {"id": "tickets", "mode": "poll", "connector_id": "jira",
             "event": "issue_created", "poll_interval_s": 600,
             "dedupe_key": "event.key"},
        ]},
        "steps": [
            {"id": "issues", "connector_id": "jira",
             "tool": "jira__search_issues",
             "params": {"jql": "x"},
             "collect": {"items_path": "issues",
                         "fields": {"key": "key", "summary": "summary"},
                         "format": "• {{item.key}} {{item.summary}}",
                         "empty_text": "none"},
             "on_error": "skip"},
            {"id": "post", "connector_id": "slack",
             "tool": "slack__send_message",
             "params": {"channel": "{{grant.target.id}}",
                        "text": "[{{steps.issues.count}}] "
                                "{{steps.issues.text}} m:{{memory.last_outcome}}"},
             "grant_id": "g-1",
             "grant_target": {"kind": "channel", "id": "C-PIN"}},
        ],
    }
    spec.update(over)
    return validate_spec(spec, REGISTRY_V2)


async def _mk_automation_v2(uid: str, vspec):
    async with async_session_maker() as db:
        a = Automation(
            user_id=uid, name=vspec.name, status="armed",
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
async def test_v2_compile_one_binding_per_source():
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == a.id)
        )).scalars().all()
        assert len(rows) == 2
        by_source = {}
        for b in rows:
            routine = await db.get(Routine, b.target_id)
            sid = (routine.config_json or {}).get("source_id")
            by_source[sid] = routine
        assert by_source["sched"].kind == "automation_schedule"
        assert by_source["tickets"].kind == "automation_poll"
        assert by_source["tickets"].schedule_interval_seconds == 600
        assert (by_source["tickets"].config_json or {}).get(
            "automation_id") == a.id


@pytest.mark.asyncio
async def test_v2_ingest_namespaces_dedupe_per_source():
    from app.agent.automations import executor_v2
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    source = vspec.source_by_id("tickets")
    items = [{"key": "ENG-1", "summary": "one"}]
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        fresh1 = await executor_v2.ingest_items_v2(db, a2, source, items)
        assert [e.dedupe_key for e in fresh1] == ["tickets:ENG-1"]
        payload = json.loads(fresh1[0].payload_json)
        assert payload["_source"] == "tickets"
        # Same provider key through a DIFFERENT source id lands fresh —
        # lanes never collide.
        source_b = type(source)(
            id="other", mode=source.mode, connector_id=source.connector_id,
            event=source.event, params=source.params,
            poll_interval_s=source.poll_interval_s, schedule=None,
            filter_rules={}, dedupe_key_field=source.dedupe_key_field,
            event_spec=source.event_spec,
        )
        fresh2 = await executor_v2.ingest_items_v2(db, a2, source_b, items)
        assert [e.dedupe_key for e in fresh2] == ["other:ENG-1"]
        # Replay through the original source is still collapsed.
        assert await executor_v2.ingest_items_v2(db, a2, source, items) == []


def _fake_dispatch(responses):
    """dispatch_via_platform stub: tool_name → result dict (or a
    callable). Records calls."""
    calls = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        grant_id=None, automation_id=None, request_id=None,
                        timeout_s=60.0):
        calls.append({"tool": tool_name, "input": tool_input,
                      "grant_id": grant_id})
        resp = responses.get(tool_name)
        if callable(resp):
            resp = resp(tool_input)
        return resp or {"kind": "ok", "content": "{}"}

    _dispatch.calls = calls
    return _dispatch


@pytest.mark.asyncio
async def test_v2_run_reads_collect_then_stage_and_send(monkeypatch):
    """The full v2 pipeline on real rows: read step collected into the
    write's template, outbox key w0, job steps evaluate→issues→post→
    record, memory written after the run."""
    from app.agent.automations import executor_v2, memory as engine_memory
    from app.db.models import BuildJob

    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    source = vspec.source_by_id("tickets")

    dispatch = _fake_dispatch({
        "jira__search_issues": {
            "kind": "ok",
            "content": json.dumps({"issues": [
                {"key": "ENG-7", "summary": "fix the gate"},
            ]}),
        },
        "slack__send_message": {"kind": "ok", "content": "{}"},
    })
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", dispatch)
    monkeypatch.setattr(executor_v2, "AUTOMATION_OUTBOX_UNDO_WINDOW_S", 0)

    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        fresh = await executor_v2.ingest_items_v2(
            db, a2, source, [{"key": "ENG-7", "summary": "fix the gate"}])
        status = await executor_v2.run_event_v2(db, a2, vspec, source, fresh[0])
        assert status == "run"

        rows = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
        )).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.idempotency_key.endswith(":w0")
        assert row.status == "executed"
        sent = json.loads(row.payload_json)
        assert sent["channel"] == "C-PIN"
        assert "ENG-7 fix the gate" in sent["text"]
        assert sent["text"].startswith("[1]")

        job = await db.get(BuildJob, row.job_id)
        assert job.status == "completed" and job.outcome == "sent"
        ids = [s["id"] for s in json.loads(job.steps_json)]
        assert ids == ["evaluate", "issues", "post", "record"]

        # Memory: the namespace row exists and carries the run.
        mem = await engine_memory.read_context(db, a2)
        assert mem.get("last_outcome") == "sent"
        assert json.loads(mem["last_counts"]) == {"issues": 1}

    # Second run renders {{memory.last_outcome}} from the first.
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        fresh = await executor_v2.ingest_items_v2(
            db, a2, source, [{"key": "ENG-8", "summary": "two"}])
        await executor_v2.run_event_v2(db, a2, vspec, source, fresh[0])
        rows = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
            .order_by(AutomationOutbox.created_at)
        )).scalars().all()
        assert "m:sent" in json.loads(rows[-1].payload_json)["text"]


@pytest.mark.asyncio
async def test_v2_on_error_skip_continues_and_reports_partial(monkeypatch):
    from app.agent.automations import executor_v2
    from app.db.models import BuildJob

    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    source = vspec.source_by_id("tickets")

    dispatch = _fake_dispatch({
        "jira__search_issues": {"kind": "tool_error", "retryable": False,
                                "message": "provider melted"},
        "slack__send_message": {"kind": "ok", "content": "{}"},
    })
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", dispatch)
    monkeypatch.setattr(executor_v2, "AUTOMATION_OUTBOX_UNDO_WINDOW_S", 0)

    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        fresh = await executor_v2.ingest_items_v2(
            db, a2, source, [{"key": "ENG-9", "summary": "x"}])
        status = await executor_v2.run_event_v2(db, a2, vspec, source, fresh[0])
        assert status == "run"
        row = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
        )).scalar_one()
        sent = json.loads(row.payload_json)
        # The skipped read contributed its empty_text and count 0.
        assert sent["text"].startswith("[0] none")
        job = await db.get(BuildJob, row.job_id)
        assert job.status == "completed"
        assert job.outcome == "partial"


@pytest.mark.asyncio
async def test_v2_on_error_fail_fails_the_run_and_stages_nothing(monkeypatch):
    from app.agent.automations import executor_v2
    from app.db.models import BuildJob

    uid = await _mk_user()
    raw = _v2_spec().raw
    raw["steps"][0]["on_error"] = "fail"
    vspec = validate_spec(raw, REGISTRY_V2)
    a = await _mk_automation_v2(uid, vspec)
    source = vspec.source_by_id("tickets")

    dispatch = _fake_dispatch({
        "jira__search_issues": {"kind": "tool_error", "retryable": False,
                                "message": "nope"},
    })
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", dispatch)

    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        fresh = await executor_v2.ingest_items_v2(
            db, a2, source, [{"key": "ENG-1", "summary": "x"}])
        status = await executor_v2.run_event_v2(db, a2, vspec, source, fresh[0])
        assert status == "failed"
        rows = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
        )).scalars().all()
        assert rows == []
        a3 = await db.get(Automation, a.id)
        assert a3.consecutive_failures == 1
        job = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == a.id)
        )).scalars().first()
        assert job.status == "failed" and job.outcome == "step_failed"


@pytest.mark.asyncio
async def test_v2_multi_write_aggregate_finalize(monkeypatch):
    """Two write steps → two outbox rows (w0, w1); the job closes only
    when BOTH are terminal, and one failure fails the aggregate."""
    from app.agent.automations import executor_v2
    from app.db.models import BuildJob

    uid = await _mk_user()
    raw = _v2_spec().raw
    raw["steps"].append({
        "id": "chat", "connector_id": "teams",
        "tool": "teams__send_chat_message",
        "params": {"chat_id": "{{grant.target.id}}", "message": "hi"},
        "grant_id": "g-2",
        "grant_target": {"kind": "chat", "id": "T-PIN"},
    })
    vspec = validate_spec(raw, REGISTRY_V2)
    a = await _mk_automation_v2(uid, vspec)
    source = vspec.source_by_id("tickets")

    dispatch = _fake_dispatch({
        "jira__search_issues": {"kind": "ok",
                                "content": json.dumps({"issues": []})},
        "slack__send_message": {"kind": "ok", "content": "{}"},
        "teams__send_chat_message": {"kind": "tool_error",
                                     "retryable": False,
                                     "message": "chat gone"},
    })
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", dispatch)
    monkeypatch.setattr(executor_v2, "AUTOMATION_OUTBOX_UNDO_WINDOW_S", 0)

    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        fresh = await executor_v2.ingest_items_v2(
            db, a2, source, [{"key": "ENG-2", "summary": "x"}])
        await executor_v2.run_event_v2(db, a2, vspec, source, fresh[0])
        rows = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
            .order_by(AutomationOutbox.idempotency_key)
        )).scalars().all()
        assert [r.idempotency_key.rsplit(":", 1)[-1] for r in rows] == \
            ["w0", "w1"]
        assert rows[0].status == "executed"
        assert rows[1].status == "failed"
        job = await db.get(BuildJob, rows[0].job_id)
        assert job.status == "failed" and job.outcome == "write_failed"


@pytest.mark.asyncio
async def test_v2_delete_removes_memory_namespace(monkeypatch):
    from app.agent.automations import executor_v2, memory as engine_memory
    from app.agent.automations import service as svc

    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await engine_memory.write_after_run(
        user_id=uid, automation_id=a.id, automation_name=a.name,
        outcome="sent", counts={"issues": 3},
    )
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        assert (await engine_memory.read_context(db, a2)) != {}
        await svc.delete_automation(db, automation_id=a.id, user_id=uid)
    async with async_session_maker() as db:
        class _Shim:
            id = a.id
            user_id = uid
        assert await engine_memory.read_context(db, _Shim()) == {}


@pytest.mark.asyncio
async def test_fast_lane_compiles_second_intervals_off_production(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "automations_dev_fast_lane", True,
                        raising=False)
    monkeypatch.setattr(settings, "environment", "development")
    uid = await _mk_user()
    raw = _v2_spec().raw
    raw["trigger"]["sources"][1]["poll_interval_s"] = 5
    vspec = validate_spec(raw, REGISTRY_V2)
    a = await _mk_automation_v2(uid, vspec)
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == a.id)
        )).scalars().all()
        intervals = set()
        for b in rows:
            routine = await db.get(Routine, b.target_id)
            if routine.kind == "automation_poll":
                intervals.add(routine.schedule_interval_seconds)
        assert intervals == {5}


@pytest.mark.asyncio
async def test_fast_lane_interval_builds_a_runner_trigger(monkeypatch):
    """R28-D: the runner's 60-s hygiene floor marked every fast-lane
    automation routine invalid_schedule — armed, compiled, and never
    fired. Sub-60 intervals must build a trigger ONLY for automation
    kinds, only at/above the 5-s fast-lane floor, and only while the
    fast lane is active outside production."""
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    from app.agent.routines.runner import _build_trigger_for_routine
    from app.config import settings

    tz = ZoneInfo("UTC")

    def routine(kind, interval):
        return SimpleNamespace(kind=kind, schedule_kind="every",
                               schedule_interval_seconds=interval,
                               id="r-fast-lane")

    monkeypatch.setattr(settings, "automations_dev_fast_lane", True,
                        raising=False)
    monkeypatch.setattr(settings, "environment", "development")
    for kind in ("automation_poll", "automation_schedule"):
        trig, tag = _build_trigger_for_routine(routine(kind, 5), tz)
        assert trig is not None and tag == "every", kind
    # A 5-s reminder keeps the 60-s hygiene floor.
    assert _build_trigger_for_routine(routine("reminder", 5), tz) \
        == (None, "invalid")
    # Below the fast-lane floor stays invalid even for automations.
    assert _build_trigger_for_routine(routine("automation_poll", 3), tz) \
        == (None, "invalid")
    # Fast lane off → automation kinds keep the 60-s floor.
    monkeypatch.setattr(settings, "automations_dev_fast_lane", False,
                        raising=False)
    assert _build_trigger_for_routine(routine("automation_poll", 5), tz) \
        == (None, "invalid")
    # Production refuses the lane even with the env var set.
    monkeypatch.setattr(settings, "automations_dev_fast_lane", True,
                        raising=False)
    monkeypatch.setattr(settings, "environment", "production")
    assert _build_trigger_for_routine(routine("automation_poll", 5), tz) \
        == (None, "invalid")
    # At/above 60 s nothing changed for anyone.
    monkeypatch.setattr(settings, "environment", "development")
    trig, tag = _build_trigger_for_routine(routine("reminder", 300), tz)
    assert trig is not None and tag == "every"


@pytest.mark.asyncio
async def test_arm_nudges_the_runner_after_the_commit(monkeypatch):
    """R28-D: `reload_routine` re-reads the row in its OWN session, so
    a nudge fired before the arm transaction commits sees the OLD
    disabled row and unregisters the routine the arm just enabled —
    armed automations then only scheduled at the 10-minute reconcile.
    Pin the contract: when the runner is nudged, a fresh session must
    already see status='armed' and the routine enabled."""
    from app.agent.automations import service as svc
    from app.api import routines as routines_api

    uid = await _mk_user()
    vspec = _poll_spec()
    a = await _mk_automation(uid, vspec)

    async def _fake_verify(automation, _vspec):
        return {"target": {"kind": "channel", "id": "C1", "label": "#x"}}

    async def _fake_parse(_automation):
        return vspec

    monkeypatch.setattr(compiler, "verify_grant_for_arm", _fake_verify)
    monkeypatch.setattr(svc, "parse_spec_live", _fake_parse)

    seen: list[tuple[str, str, bool]] = []

    class _Runner:
        async def reload_routine(self, rid):
            async with async_session_maker() as db:
                r = await db.get(Routine, rid)
                auto = await db.get(Automation, a.id)
            seen.append((rid, auto.status, bool(r and r.enabled)))

    monkeypatch.setattr(routines_api, "_runner", _Runner(), raising=False)

    async with async_session_maker() as db:
        await svc.arm_automation(db, automation_id=a.id, user_id=uid)

    assert seen, "arm never nudged the runner"
    assert all(status == "armed" and enabled
               for _, status, enabled in seen), seen


def test_schedule_fires_are_keyed_per_instant_not_per_day():
    """R28-D: `_fire_idempotency_key` keyed automation_schedule by
    local DATE alone, so the second `every_s` fire of the day hit the
    (source_id, idempotency_key) UNIQUE and silently exited — an
    "every 2 hours" automation became "once a day". Multi-fire kinds
    key by fire instant; reminders keep the one-per-day contract."""
    from datetime import date, datetime as dt

    from app.agent.routines.runner import RoutineRunner

    d = date(2026, 8, 24)
    t1 = dt(2026, 8, 24, 10, 0, 0)
    t2 = dt(2026, 8, 24, 10, 0, 5)
    key = RoutineRunner._fire_idempotency_key
    for kind in ("automation_schedule", "automation_poll", "autopilot"):
        assert key(kind, d, t1) != key(kind, d, t2), kind
        # retries of the SAME fire share the key
        assert key(kind, d, t1) == key(kind, d, t1), kind
    assert key("reminder", d, t1) == key("reminder", d, t2) == str(d)


# ── Round 29: last outcome, unseen, payload shape, membership ────────


@pytest.mark.asyncio
async def test_finalize_stamps_last_outcome_before_notify(monkeypatch):
    """CONTRACTS-R29.md §2 order pin: when the exactly-once notify
    fires, the automation row ALREADY carries the sentence — and the
    R28 wrote_count=0 bug stays dead (real count reaches the push)."""
    from app.db.models import BuildJob

    uid = await _mk_user()
    a = await _mk_automation(uid, _poll_spec())
    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=uid, title="Run", prompt="(automation)",
            job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
        ))
        await db.commit()

    seen = {}

    async def fake_notify(**kw):
        async with async_session_maker() as db:
            row = await db.get(Automation, a.id)
            seen["stamped_at_notify"] = row.last_outcome
            seen["text_at_notify"] = row.last_outcome_text
        seen["wrote_count"] = kw.get("wrote_count")
        return True

    monkeypatch.setattr(
        "app.agent.automations.notify.notify_run_outcome", fake_notify)

    async with async_session_maker() as db:
        await executor._finalize_job(
            db, job_id, status="completed", outcome="sent",
        )

    assert seen["stamped_at_notify"] == "sent"
    assert seen["text_at_notify"] == "Posted to Slack."
    assert seen["wrote_count"] == 1

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert row.last_outcome_at is not None
        assert row.outcome_seen_at is None


@pytest.mark.asyncio
async def test_unseen_flips_with_the_seen_cas_and_the_next_stamp():
    from app.agent.automations.service import (
        automation_payload, mark_outcome_seen,
    )
    from app.db.models import BuildJob

    uid = await _mk_user()
    a = await _mk_automation(uid, _poll_spec())

    async def _terminal(outcome):
        jid = str(uuid.uuid4())
        async with async_session_maker() as db:
            db.add(BuildJob(
                id=jid, user_id=uid, title="Run", prompt="(automation)",
                job_type="automation_run", status="running",
                source_kind="automation", source_id=a.id,
            ))
            await db.commit()
            await executor._finalize_job(
                db, jid, status="completed", outcome=outcome,
            )

    await _terminal("sent")
    async with async_session_maker() as db:
        p = automation_payload(await db.get(Automation, a.id))
    assert p["unseen"] is True
    assert p["last_outcome"]["tone"] == "ok"
    assert p["last_outcome"]["sentence"] == "Posted to Slack."

    async with async_session_maker() as db:
        await mark_outcome_seen(db, automation_id=a.id, user_id=uid)
        p = automation_payload(await db.get(Automation, a.id))
    assert p["unseen"] is False

    await _terminal("sent")
    async with async_session_maker() as db:
        p = automation_payload(await db.get(Automation, a.id))
    assert p["unseen"] is True, "a fresh terminal outcome re-arms unseen"


@pytest.mark.asyncio
async def test_payload_carries_the_r29_shape():
    """B's canvas contract: connectors[], schedule_human, rule_text,
    mode (already served, pinned so it cannot regress), attention."""
    from app.agent.automations.service import automation_payload

    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    async with async_session_maker() as db:
        p = automation_payload(await db.get(Automation, a.id))
    assert p["connectors"] == ["jira", "slack"]
    assert p["schedule_human"] == "weekdays 8:00"
    assert p["rule_text"] == (
        "Every weekday at 8:00, check Jira and post to Slack."
    )
    assert p["mode"] == "auto"
    assert p["attention"] is None
    assert p["unseen"] is False and p["last_outcome"] is None
    assert "domain" in p, "the v2 branch used to drop the column"
    for s in p["steps"]:
        assert "grant_target" not in s


@pytest.mark.asyncio
async def test_attention_states_are_explicit():
    from app.agent.automations.service import (
        automation_payload, pause_automation,
    )

    uid = await _mk_user()
    a = await _mk_automation(uid, _poll_spec())
    async with async_session_maker() as db:
        await pause_automation(
            db, automation_id=a.id, user_id=uid, reason="grant_revoked",
        )
        p = automation_payload(await db.get(Automation, a.id))
    assert p["attention"] == "grant_revoked"

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        row.status = "error"
        row.paused_reason = "auto_failures"
        await db.commit()
        p = automation_payload(await db.get(Automation, a.id))
    assert p["attention"] == "auto_paused"


@pytest.mark.asyncio
async def test_steps_json_is_humanized_at_mint_with_brands():
    """The substrate rule (§1): labels are human at MINT — every
    downstream surface (runs API, job cards, web) inherits; no raw
    step id or tool name is ever stored."""
    from app.agent.automations.executor import _new_steps
    from app.agent.automations.executor_v2 import _new_steps_v2

    v1 = json.loads(_new_steps(_poll_spec()))
    assert [s["label"] for s in v1] == [
        "Checking triggers", "Composing", "Posting to Slack", "Wrapping up",
    ]
    assert [s["brand"] for s in v1] == [None, None, "slack", None]

    v2 = json.loads(_new_steps_v2(_v2_spec()))
    by_id = {s["id"]: s for s in v2}
    assert by_id["issues"]["label"] == "Checking Jira"
    assert by_id["issues"]["brand"] == "jira"
    assert by_id["post"]["label"] == "Posting to Slack"
    assert by_id["evaluate"]["brand"] is None
    for s in v1 + v2:
        assert "__" not in s["label"]


@pytest.mark.asyncio
async def test_runs_api_serves_verbs_with_counts_and_fix_chips(monkeypatch):
    """One render path for every era: the collected count reaches the
    done-form verb, a failed run grows its fix chip, and no served
    step string carries a raw name."""
    from app.agent.automations import executor_v2
    from app.agent.automations.service import list_runs
    from app.db.models import BuildJob

    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)

    dispatch = _fake_dispatch({
        "jira__search_issues": {
            "kind": "ok",
            "content": json.dumps({"issues": [
                {"key": "ENG-1", "summary": "a"},
                {"key": "ENG-2", "summary": "b"},
            ]}),
        },
        "slack__send_message": {"kind": "ok", "content": "{}"},
    })
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", dispatch)
    monkeypatch.setattr(executor_v2, "AUTOMATION_OUTBOX_UNDO_WINDOW_S", 0)

    async with async_session_maker() as db:
        jid = str(uuid.uuid4())
        db.add(BuildJob(
            id=jid, user_id=uid, title="Run", prompt="(automation)",
            job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
            steps_json=executor_v2._new_steps_v2(vspec),
        ))
        await db.commit()
        await executor_v2._run_steps(
            db, await db.get(Automation, a.id), vspec, jid, {}, None,
            idem_prefix=f"t:{jid[:8]}",
        )

    async with async_session_maker() as db:
        runs = await list_runs(db, uid)
    run = next(r for r in runs if r["id"] == jid)
    by_id = {s["id"]: s for s in run["steps"]}
    assert by_id["issues"]["verb"] == "Read 2 Jira issues"
    assert by_id["issues"]["brand"] == "jira"
    assert by_id["post"]["brand"] == "slack"
    assert run["fix"] is None
    for s in run["steps"]:
        assert "__" not in (s["verb"] or "")
        assert "__" not in (s["label"] or "")

    # A failed run grows the chip.
    async with async_session_maker() as db:
        fid = str(uuid.uuid4())
        db.add(BuildJob(
            id=fid, user_id=uid, title="Run", prompt="(automation)",
            job_type="automation_run", status="failed",
            outcome="step_failed", source_kind="automation", source_id=a.id,
        ))
        await db.commit()
        runs = await list_runs(db, uid)
    failed = next(r for r in runs if r["id"] == fid)
    assert failed["fix"]["label"] == "Fix this"
    assert "Brief v2" in failed["fix"]["prompt"]


@pytest.mark.asyncio
async def test_schedule_mode_and_membership_edits(monkeypatch):
    """§3.2/§3.3 — the focused edits ride the full update path
    (revalidate, recompile); refusals carry stable codes."""
    from app.agent.automations import service
    from app.db.models import Routine

    async def _registry(user_id, *, force=False):
        return REGISTRY_V2

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _registry)

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    async with async_session_maker() as db:
        automation, vspec = await service.set_schedule(
            db, automation_id=a.id, user_id=uid,
            schedule={"cron_local": "30 7 * * *"},
        )
        raw = json.loads(automation.spec_json)
        sched = next(s for s in raw["trigger"]["sources"]
                     if s["mode"] == "schedule")
        assert sched["schedule"] == {"cron_local": "30 7 * * *"}
        b_rows = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == a.id)
        )).scalars().all()
        routine_ids = [b.target_id for b in b_rows if b.kind == "routine"]
        crons = [
            r.schedule_cron_local
            for r in (await db.execute(
                select(Routine).where(Routine.id.in_(routine_ids))
            )).scalars().all()
            if r.kind == "automation_schedule"
        ]
        assert crons == ["30 7 * * *"], "recompile carries the new cron"

        with pytest.raises(service.MembershipError) as e:
            await service.set_schedule(
                db, automation_id=a.id, user_id=uid,
                schedule={"cron_local": "0 8 * * *", "every_s": 60},
            )
        assert e.value.code == "bad_schedule"

        automation, _ = await service.set_mode(
            db, automation_id=a.id, user_id=uid, mode="confirm",
        )
        assert json.loads(automation.spec_json)["mode"] == "confirm"

        # Membership: the write connector is load-bearing.
        with pytest.raises(service.MembershipError) as e:
            await service.remove_connector(
                db, automation_id=a.id, user_id=uid, connector_id="slack",
            )
        assert e.value.code == "connector_required"
        # Removing the only poll source's connector still leaves the
        # schedule source — allowed, and the payload loses the brand.
        automation, _ = await service.remove_connector(
            db, automation_id=a.id, user_id=uid, connector_id="jira",
        )
        assert "jira" not in service.automation_payload(automation)[
            "connectors"]
        with pytest.raises(service.MembershipError) as e:
            await service.remove_connector(
                db, automation_id=a.id, user_id=uid, connector_id="jira",
            )
        assert e.value.code == "not_member"

        # Re-add from the template skeleton.
        async def _templates(user_id):
            return [{
                "slug": "brief-v2",
                "spec": _v2_spec().raw,
            }]
        monkeypatch.setattr(
            "app.agent.automations.registry.fetch_templates", _templates)
        row = await db.get(Automation, a.id)
        row.template_slug = "brief-v2"
        await db.commit()
        automation, _ = await service.add_connector(
            db, automation_id=a.id, user_id=uid, connector_id="jira",
        )
        assert "jira" in service.automation_payload(automation)["connectors"]
        with pytest.raises(service.MembershipError) as e:
            await service.add_connector(
                db, automation_id=a.id, user_id=uid, connector_id="jira",
            )
        assert e.value.code == "already_member"


@pytest.mark.asyncio
async def test_grant_revoked_hook_pauses_the_armed_dependent(monkeypatch):
    """§3.1: the platform's revoke reaches the tenant through the
    existing _grant_decided hook, which now pauses the armed dependent
    (`grant_revoked`) — the dispatcher already fails closed; this makes
    the STATE honest for the attention pill."""
    from app.config import settings
    from app.api.automations import GrantHook, grant_decided_hook
    from app.agent.automations.service import automation_payload

    uid = await _mk_user()
    a = await _mk_automation(uid, _poll_spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        row.status = "armed"
        await db.commit()

    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)

    await grant_decided_hook(GrantHook(
        grant_id=str(uuid.uuid4()), status="revoked",
        payload={"automation_id": a.id},
    ))
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert row.status == "error"
        assert row.paused_reason == "grant_revoked"
        assert automation_payload(row)["attention"] == "grant_revoked"

    # A non-revoke decision leaves the automation alone.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        row.status = "armed"
        row.paused_reason = None
        await db.commit()
    await grant_decided_hook(GrantHook(
        grant_id=str(uuid.uuid4()), status="approved",
        payload={"automation_id": a.id},
    ))
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert row.status == "armed"
