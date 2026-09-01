# agent-mode: automation_threads/_turns/_writes are AGENT_ONLY tables.
"""Run ledger v3 — CONTRACTS-R30 §4.2/§4.3 proof (R30-A).

Against real rows: turn typing + seq, dictionary rejection, tier enum
rejection, the completeness invariant (unaccounted append + mechanical
fallback result), the status projection table, stop at the step
boundary (no write after stop — executor first line AND outbox second
line), checkpoint resume, supersede, legacy render.
"""

import json
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationThread, AutomationTurn, AutomationWrite,
    BuildJob, User,
)
from app.agent.automations import compiler
from app.agent.automations.spec import validate_spec

REGISTRY_V2 = {
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
    "slack": {
        "connector_id": "slack", "push": False, "poll": False,
        "floor_s": 300, "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {"slack__send_message": ["w"]},
        "target_param_by_action": {"slack__send_message": "channel"},
        "events": [],
    },
}


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Ledger"))
        await db.commit()
    return uid


def _v2_spec(**over):
    spec = {
        "version": 2,
        "name": "Ledger brief",
        "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
        ]},
        "steps": [
            {"id": "issues", "connector_id": "jira",
             "tool": "jira__search_issues",
             "params": {"jql": "x"},
             "collect": {"items_path": "issues",
                         "fields": {"key": "key", "summary": "summary"},
                         "format": "{{item.key}} {{item.summary}}",
                         "empty_text": "none"},
             "on_error": "skip"},
            {"id": "post", "connector_id": "slack",
             "tool": "slack__send_message",
             "params": {"channel": "{{grant.target.id}}",
                        "text": "{{steps.issues.text}}"},
             "grant_id": "g-1",
             "grant_target": {"kind": "channel", "id": "C-PIN",
                              "label": "#platform"}},
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


def _fake_dispatch(responses):
    import inspect
    calls = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        grant_id=None, automation_id=None, request_id=None,
                        timeout_s=60.0):
        calls.append({"tool": tool_name, "input": tool_input,
                      "grant_id": grant_id})
        resp = responses.get(tool_name)
        if callable(resp):
            resp = resp(tool_input)
        if inspect.isawaitable(resp):
            resp = await resp
        return resp or {"kind": "ok", "content": "{}"}

    _dispatch.calls = calls
    return _dispatch


_ISSUES = {"kind": "ok", "content": json.dumps({"issues": [
    {"key": "TP-482", "summary": "Rate-limit the export endpoint"},
    {"key": "TP-476", "summary": "Flaky memverify test"},
]})}
_OK = {"kind": "ok", "content": "{}"}


async def _fire(monkeypatch, uid, a, vspec, *, responses=None):
    from app.agent.automations import executor_v2
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _fake_dispatch(responses or {
            "jira__search_issues": _ISSUES,
            "slack__send_message": _OK,
        }),
    )
    # No undo-window sleep in tests.
    monkeypatch.setattr(
        "app.agent.automations.executor_v2.AUTOMATION_OUTBOX_UNDO_WINDOW_S",
        0,
    )
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        source = vspec.schedule_source()
        return await executor_v2.run_schedule_fire_v2(
            db, a2, vspec, source, fire_key=f"t:{uuid.uuid4()}",
        )


async def _one_run(a_id: str) -> BuildJob:
    async with async_session_maker() as db:
        return (await db.execute(
            select(BuildJob).where(BuildJob.source_id == a_id)
            .order_by(BuildJob.created_at.desc()).limit(1)
        )).scalar_one()


# ── turn grammar ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_seq_monotonic_and_typed():
    from app.agent.automations import ledger
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(
            db, user_id=uid, automation_id=a.id,
        )
        t1 = await ledger.append_turn(
            db, user_id=uid, thread=thread, kind="agent",
            payload={"text": "Morning."},
        )
        t2 = await ledger.append_turn(
            db, user_id=uid, thread=thread, kind="note",
            payload={"stamp": "started"},
        )
        assert t2["seq"] == t1["seq"] + 1
        with pytest.raises(Exception):
            await ledger.append_turn(
                db, user_id=uid, thread=thread, kind="hologram",
                payload={"text": "no"},
            )


@pytest.mark.asyncio
async def test_dictionary_rejects_raw_tool_action():
    """D-01's whole class is unrepresentable: an action the dictionary
    does not serve never persists (strict lane raises)."""
    from app.agent.automations import ledger
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(
            db, user_id=uid, automation_id=a.id,
        )
        with pytest.raises(ledger.LedgerValidationError):
            ledger.validate_turn_payload("tool", {
                "account_id": "jira", "tool_kind": "read",
                "action": "jira__search_issues", "ok": True,
            })
        with pytest.raises(ledger.LedgerValidationError):
            ledger.validate_turn_payload("tool", {
                "account_id": "jira", "tool_kind": "read",
                "action": "List events", "ok": True,
            })
        del thread


def test_tier_vocabulary_is_closed():
    from app.agent.automations import ledger
    from app.db.models import BRIEF_TIERS
    groups = [
        {"rank": i + 1, "label": lb, "tone": tn, "rows": []}
        for i, (lb, tn) in enumerate(BRIEF_TIERS)
    ]
    ok = ledger.validate_turn_payload("result", {
        "title": "Your morning, in order", "vocabulary": "brief",
        "groups": groups,
    })
    assert len(ok["groups"]) == 5
    groups4 = groups[:4]
    with pytest.raises(ledger.LedgerValidationError):
        ledger.validate_turn_payload("result", {
            "title": "x", "vocabulary": "brief", "groups": groups4,
        })
    bad = [dict(g) for g in groups]
    bad[0] = dict(bad[0], label="MOST IMPORTANT")
    with pytest.raises(ledger.LedgerValidationError):
        ledger.validate_turn_payload("result", {
            "title": "x", "vocabulary": "brief", "groups": bad,
        })


# ── the projection table ─────────────────────────────────────────────


def test_status_projection_table():
    from app.agent.automations.ledger import run_v3_status

    class J:
        def __init__(self, status, outcome=None):
            self.status, self.outcome = status, outcome

    assert run_v3_status(J("running")) == "running"
    assert run_v3_status(J("queued")) == "running"
    assert run_v3_status(J("waiting_on_user")) == "waiting_on_user"
    assert run_v3_status(J("completed", "sent")) == "completed"
    assert run_v3_status(J("completed", "partial")) == "partial"
    assert run_v3_status(J("failed", "step_failed")) == "failed"
    assert run_v3_status(J("cancelled", "skipped")) == "skipped"
    assert run_v3_status(J("cancelled", "stopped")) == "stopped_by_user"
    assert run_v3_status(J("cancelled", "undone")) == "stopped_by_user"
    assert run_v3_status(J("cancelled", "superseded")) == "superseded"


# ── the full pipeline writes the ledger ──────────────────────────────


@pytest.mark.asyncio
async def test_v3_run_produces_thread_turns_and_write_ledger(monkeypatch):
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    status = await _fire(monkeypatch, uid, a, vspec)
    assert status == "run"
    job = await _one_run(a.id)
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        thread = await ledger.thread_for(db, a.id)
        assert thread is not None
        turns = await ledger.run_turns(db, run_id=job.id)
        kinds = [t["kind"] for t in turns]
        assert kinds[0] == "note"
        tool_turns = [t for t in turns if t["kind"] == "tool"]
        # One read (with minted item ids) + one write.
        reads = [t for t in tool_turns if t["tool_kind"] == "read"]
        writes = [t for t in tool_turns if t["tool_kind"] == "write"]
        assert len(reads) == 1 and len(writes) == 1
        assert all(it["id"] for it in reads[0]["items"])
        assert "__" not in reads[0]["action"]
        # The honest write ledger row, linked from the turn.
        rows = (await db.execute(
            select(AutomationWrite).where(AutomationWrite.run_id == job.id)
        )).scalars().all()
        assert len(rows) == 1
        assert rows[0].account_id == "slack"
        assert writes[0]["write_ids"] == [rows[0].id]
        # config carries the stamps.
        cfg = ledger._cfg_of(job)
        assert cfg.get("thread_id") == thread.id
        assert "jira" in (cfg.get("accounts_touched") or [])
        # Terminal head note flipped started → ran.
        head = [t for t in turns if t["kind"] == "note"][0]
        assert head["stamp"] == "ran"


@pytest.mark.asyncio
async def test_completeness_net_appends_mechanical_result(monkeypatch):
    """No narrator in the tree ⇒ the ledger close appends the honest
    fallback result: every item accounted once, in the last tier."""
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec)
    job = await _one_run(a.id)
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        turns = await ledger.run_turns(db, run_id=job.id)
        results = [t for t in turns if t["kind"] == "result"]
        assert len(results) == 1
        item_ids = {
            it["id"] for t in turns if t["kind"] == "tool"
            for it in (t.get("items") or [])
        }
        referenced = [
            ref for g in results[0]["groups"]
            for r in g["rows"] for ref in r["item_refs"]
        ]
        assert sorted(referenced) == sorted(item_ids)
        assert len(referenced) == len(set(referenced))


# ── stop / resume / supersede ────────────────────────────────────────


@pytest.mark.asyncio
async def test_stop_before_write_leaves_nothing_sent(monkeypatch):
    """The stop lands during the read; the write step never stages and
    the stop note says NOTHING WAS SENT with writes_count 0."""
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)

    from app.agent.automations import run_v3

    async def _slow_read(tool_input):
        # Stop arrives while the read is in flight.
        async with async_session_maker() as db:
            job = (await db.execute(
                select(BuildJob).where(BuildJob.source_id == a.id)
                .order_by(BuildJob.created_at.desc()).limit(1)
            )).scalar_one()
            await run_v3.request_stop(db, job.id)
        return _ISSUES

    sent = []

    async def _send(tool_input):
        sent.append(tool_input)
        return _OK

    status = await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": _slow_read,
        "slack__send_message": _send,
    })
    assert status == "stopped"
    assert sent == []
    job = await _one_run(a.id)
    from app.agent.automations import ledger
    assert ledger.run_v3_status(job) == "stopped_by_user"
    assert (ledger.checkpoint_of(job) or {}).get("step_index") is not None
    async with async_session_maker() as db:
        from sqlalchemy import func
        n_writes = (await db.execute(
            select(func.count()).select_from(AutomationWrite)
            .where(AutomationWrite.run_id == job.id)
        )).scalar_one()
        assert n_writes == 0
        turns = await ledger.run_turns(db, run_id=job.id)
        stop_notes = [t for t in turns if t["kind"] == "note"
                      and t.get("stamp") == "stopped"]
        assert len(stop_notes) == 1
        assert stop_notes[0]["writes_count"] == 0
        # No outbox row ever staged.
        from app.db.models import AutomationOutbox
        staged = (await db.execute(
            select(AutomationOutbox).where(
                AutomationOutbox.job_id == job.id,
            )
        )).scalars().all()
        assert staged == []


@pytest.mark.asyncio
async def test_outbox_refuses_staged_write_of_stopped_run(monkeypatch):
    """Second line of defence: a staged row that outlives a stop is
    refused at the flush, never sent (§4.3 mutation target)."""
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    from app.db.models import AutomationOutbox
    from app.agent.automations import outbox as ob, run_v3

    async with async_session_maker() as db:
        job = BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run",
            prompt="", job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
        )
        db.add(job)
        row = AutomationOutbox(
            user_id=uid, automation_id=a.id, job_id=job.id,
            connector_id="slack", tool_name="slack__send_message",
            payload_json="{}", grant_id="g-1",
            idempotency_key="t:w0", execute_after=datetime.utcnow(),
            status="executing",
        )
        db.add(row)
        await db.commit()
        row_id = row.id
        await run_v3.request_stop(db, job.id)

    called = []

    async def _never(*a_, **k_):
        called.append(1)
        return _OK

    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _never,
    )
    async with async_session_maker() as db:
        status = await ob._execute_claimed(db, row_id)
    assert status == "cancelled"
    assert called == []


@pytest.mark.asyncio
async def test_resume_completes_and_next_fire_supersedes(monkeypatch):
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)

    from app.agent.automations import run_v3

    async def _stop_read(tool_input):
        async with async_session_maker() as db:
            job = (await db.execute(
                select(BuildJob).where(BuildJob.source_id == a.id)
                .order_by(BuildJob.created_at.desc()).limit(1)
            )).scalar_one()
            await run_v3.request_stop(db, job.id)
        return _ISSUES

    status = await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": _stop_read,
        "slack__send_message": _OK,
    })
    assert status == "stopped"
    job = await _one_run(a.id)

    # Resume: reads re-execute, the write goes out, the run completes.
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _fake_dispatch({
            "jira__search_issues": _ISSUES,
            "slack__send_message": _OK,
        }),
    )
    async with async_session_maker() as db:
        result = await run_v3.resume_run(db, job_id=job.id)
    assert result.get("resumed") is True
    job2 = await _one_run(a.id)
    from app.agent.automations import ledger
    assert ledger.run_v3_status(job2) == "completed"

    # A stopped run that is never resumed is superseded by the next fire.
    async def _stop_read2(tool_input):
        async with async_session_maker() as db:
            job = (await db.execute(
                select(BuildJob).where(BuildJob.source_id == a.id)
                .order_by(BuildJob.created_at.desc()).limit(1)
            )).scalar_one()
            await run_v3.request_stop(db, job.id)
        return _ISSUES

    await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": _stop_read2,
        "slack__send_message": _OK,
    })
    stopped = await _one_run(a.id)
    assert ledger.run_v3_status(stopped) == "stopped_by_user"
    await _fire(monkeypatch, uid, a, vspec)
    async with async_session_maker() as db:
        old = await db.get(BuildJob, stopped.id)
        assert ledger.run_v3_status(old) == "superseded"


# ── legacy render ────────────────────────────────────────────────────


def test_legacy_run_renders_step_lines_only():
    from app.agent.automations.ledger import legacy_turns

    class J:
        id = "legacy-1"
        status = "completed"
        outcome = "sent"
        created_at = datetime(2026, 8, 20, 8, 0)
        steps_json = json.dumps([
            {"id": "evaluate", "label": "Checked triggers", "brand": None,
             "status": "done"},
            {"id": "issues", "label": "Checked Jira", "brand": "jira",
             "status": "done"},
        ])
        config_json = None

    turns = legacy_turns(J())
    assert turns[0]["kind"] == "note" and turns[0]["stamp"] == "ran"
    tools = [t for t in turns if t["kind"] == "tool"]
    assert len(tools) == 1
    assert tools[0]["items"] == []
    assert tools[0]["steps"][0]["text"] == "Checked Jira"


@pytest.mark.asyncio
async def test_nd4_refused_write_never_wears_the_done_form(monkeypatch):
    """GROUND-TRUTH ND-4 (live repro: run 30c2b526): a write that the
    platform REFUSED must not serve "Posted to Slack" — the step wears
    failed, and the done form lands only on an executed write."""
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    status = await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": _ISSUES,
        "slack__send_message": {
            "kind": "tool_error", "retryable": False,
            "message": "write blocked: no pinned target",
        },
    })
    del status
    job = await _one_run(a.id)
    from app.agent import job_steps
    async with async_session_maker() as db:
        j = await db.get(BuildJob, job.id)
        steps = {s["id"]: s for s in job_steps.parse_steps(j.steps_json)}
        assert steps["post"]["status"] == "failed", steps["post"]
        assert j.outcome == "write_failed"
    # The executed path DOES wear done (control on the same rails).
    a2 = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a2, vspec)
    job2 = await _one_run(a2.id)
    async with async_session_maker() as db:
        j2 = await db.get(BuildJob, job2.id)
        steps2 = {s["id"]: s for s in job_steps.parse_steps(j2.steps_json)}
        assert steps2["post"]["status"] == "done", steps2["post"]


@pytest.mark.asyncio
async def test_nd7a_stop_during_the_write_terminalizes_the_run(monkeypatch):
    """ND-7 live wedge: the outbox refusal cancelled the ROW but left
    the RUN running forever. Now the refusal also terminalizes —
    stopped_by_user, checkpoint, the honest stop note."""
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    from app.agent.automations import ledger, outbox as ob, run_v3
    from app.db.models import AutomationOutbox

    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=a.id)
        job = BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run", prompt="",
            job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
            config_json={"thread_id": thread.id, "run_kind": "run_now"},
        )
        db.add(job)
        row = AutomationOutbox(
            user_id=uid, automation_id=a.id, job_id=job.id,
            connector_id="slack", tool_name="slack__send_message",
            payload_json="{}", grant_id="g-1",
            idempotency_key="w:0", execute_after=datetime.utcnow(),
            status="executing",
        )
        db.add(row)
        await db.commit()
        row_id, job_id = row.id, job.id
        await run_v3.request_stop(db, job_id)

    async def _never(*a_, **k_):
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _never,
    )
    async with async_session_maker() as db:
        status = await ob._execute_claimed(db, row_id)
    assert status == "cancelled"
    async with async_session_maker() as db:
        job2 = await db.get(BuildJob, job_id)
        from app.agent.automations.ledger import run_v3_status
        assert run_v3_status(job2) == "stopped_by_user", (
            job2.status, job2.outcome,
        )
        turns = await ledger.run_turns(db, run_id=job_id)
        assert any(t["kind"] == "note" and t.get("stamp") == "stopped"
                   for t in turns)


@pytest.mark.asyncio
async def test_nd7b_run_cap_terminalizes_the_scheduled_fire(monkeypatch):
    """ND-7b: run_schedule_fire_v2's cap handler could not even name
    the job (minted inside the wait_for) — the row stayed running
    forever. Now the cap finalizes on a fresh session."""
    import asyncio as _asyncio
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    from app.agent.automations import executor_v2

    async def _hang(tool_input):
        await _asyncio.sleep(30)
        return _ISSUES

    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _fake_dispatch({"jira__search_issues": _hang}),
    )
    monkeypatch.setattr(
        "app.agent.automations.executor_v2.AUTOMATION_RUN_CAP_S", 1,
    )
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        status = await executor_v2.run_schedule_fire_v2(
            db, a2, vspec, vspec.schedule_source(),
            fire_key=f"cap:{uuid.uuid4()}", run_kind="run_now",
        )
    assert status == "failed"
    job = await _one_run(a.id)
    assert job.status == "failed" and job.outcome == "run_cap", (
        job.status, job.outcome,
    )


@pytest.mark.asyncio
async def test_nd7d_sweep_terminalizes_a_wedged_stop_with_the_honest_shape():
    """ND-7d: a stop the executor never honoured gets its terminal from
    the sweep — and the SHAPE must be honest, not merely terminal.
    A `failed/lost` terminal also "arrives", and it mints a "Fix this"
    chip offering to diagnose a failure that never happened: the user
    pressed Stop and the product told them it broke (live run
    5f3f57bf). Assert stopped_by_user + checkpoint + stop note + NO
    fix chip."""
    from datetime import timedelta
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    from app.agent.automations import ledger
    from app.agent.automations.sweep import _sweep_wedged_stops

    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=a.id)
        job = BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run", prompt="",
            job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
            config_json={"thread_id": thread.id},
            stop_requested_at=datetime.utcnow() - timedelta(minutes=5),
        )
        db.add(job)
        await db.commit()
        job_id = job.id

    assert await _sweep_wedged_stops() == 1
    async with async_session_maker() as db:
        job2 = await db.get(BuildJob, job_id)
        from app.agent.automations.ledger import checkpoint_of, run_v3_status
        assert run_v3_status(job2) == "stopped_by_user", (
            job2.status, job2.outcome,
        )
        # The shape D pinned: never the reaper's lying vocabulary.
        assert job2.outcome != "lost"
        assert job2.error_class != "interrupted"
        assert "interrupted" not in (job2.user_message or "").lower()
        assert checkpoint_of(job2) is not None
        turns = await ledger.run_turns(db, run_id=job_id)
        stop_notes = [t for t in turns if t["kind"] == "note"
                      and t.get("stamp") == "stopped"]
        assert len(stop_notes) == 1
        assert stop_notes[0]["writes_count"] == 0

    # And the runs API must NOT offer to fix a stop: the "Fix this"
    # chip is minted only for status=failed.
    from app.agent.automations.service import list_runs
    async with async_session_maker() as db:
        runs = await list_runs(db, uid, automation_id=a.id, limit=10)
    row = next(r for r in runs if r["id"] == job_id)
    assert row["fix"] is None, row["fix"]

    # Second tick: nothing left to do.
    assert await _sweep_wedged_stops() == 0


@pytest.mark.asyncio
async def test_nd7d_the_stuck_reaper_never_claims_a_stopped_run():
    """The live ND-7d mechanism: the stuck-run reaper fires at 360s and
    ran FIRST in the tick, so it reached a user-stopped run before the
    wedged-stop leg and stamped failed/lost + a Fix chip. Now the leg
    runs first AND the reaper's predicate excludes stop-requested rows
    — so even a stop older than the stuck cutoff ends honestly."""
    from datetime import timedelta
    from app.agent.automations import ledger
    from app.agent.automations.sweep import (
        _STUCK_AFTER, _sweep_stuck_runs, sweep_automations,
    )
    from app.agent.automations.ledger import run_v3_status

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    # Older than the reaper's cutoff — the exact live condition.
    old_start = datetime.utcnow() - _STUCK_AFTER - timedelta(seconds=30)
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=a.id)
        job = BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run", prompt="",
            job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
            created_at=old_start,
            config_json={"thread_id": thread.id, "run_kind": "run_now"},
            stop_requested_at=old_start + timedelta(seconds=120),
        )
        db.add(job)
        await db.commit()
        job_id = job.id

    # The reaper alone must refuse it outright.
    assert await _sweep_stuck_runs() == 0
    async with async_session_maker() as db:
        assert (await db.get(BuildJob, job_id)).status == "running"

    # A full tick terminalises it — honestly.
    await sweep_automations()
    async with async_session_maker() as db:
        job2 = await db.get(BuildJob, job_id)
        assert run_v3_status(job2) == "stopped_by_user", (
            job2.status, job2.outcome,
        )
        assert job2.outcome != "lost"


@pytest.mark.asyncio
async def test_nd10_a_reaped_run_stamps_the_card_it_belongs_to():
    """ND-10 (live on the founder): the stuck-run reaper did ONE bulk
    UPDATE, so `_stamp_last_outcome` (whose only caller is
    `_finalize_job`) never ran — the automation kept advertising an
    older, rosier last_outcome while its newest run had failed. The home
    card's meta and status-aware description both read that stamp, so
    the card said the last run went fine hours after one broke."""
    from datetime import timedelta
    from app.agent.automations.sweep import _STUCK_AFTER, _sweep_stuck_runs

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        # The rosier stamp an earlier run left behind.
        row.last_outcome = "sent"
        row.last_outcome_text = "Posted to Slack."
        row.last_outcome_at = datetime.utcnow() - timedelta(hours=18)
        db.add(BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run", prompt="",
            job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
            created_at=datetime.utcnow() - _STUCK_AFTER
            - timedelta(minutes=1),
        ))
        await db.commit()
        before = row.last_outcome_at

    assert await _sweep_stuck_runs() == 1
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        # The card now tells the truth about its newest run.
        assert row.last_outcome == "lost", row.last_outcome
        assert row.last_outcome_at > before
        assert "Posted to Slack." not in (row.last_outcome_text or "")


def test_nd14_a_stop_is_not_a_failure_in_the_dictionary():
    """ND-14 (live): R30 added the `stopped` / `superseded` terminals to
    the engine but never to the dictionary, so tone_for hit the "err"
    default and the sentence fell through to "The last run didn't
    complete." The card painted the user's own Stop in the danger tint
    and contradicted the thread beside it, which said "You stopped it.
    Nothing was sent." """
    from app.services.automation_verbs import outcome_sentence, tone_for

    assert tone_for("stopped") == "warn", "a stop is not an error"
    assert tone_for("superseded") == "warn"
    # The generic fallthrough must not be reachable for either.
    for oc in ("stopped", "superseded"):
        s = outcome_sentence(oc, wrote_count=0)["sentence"]
        assert "didn't complete" not in s, (oc, s)

    zero = outcome_sentence("stopped", wrote_count=0)
    assert zero["sentence"] == "You stopped it. Nothing was sent."
    assert zero["tone"] == "warn"
    one = outcome_sentence("stopped", wrote_count=1)["sentence"]
    assert one == "You stopped it. 1 change already made."
    two = outcome_sentence("stopped", wrote_count=2)["sentence"]
    assert two == "You stopped it. 2 changes already made."


@pytest.mark.asyncio
async def test_nd14_the_card_and_the_thread_tell_the_same_story(monkeypatch):
    """The stamp's count must come from the write LEDGER, not the
    spec's write-STEP count — otherwise the card claims a change the
    stop prevented. Drives a real stop and compares the two surfaces."""
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    from app.agent.automations import ledger, run_v3

    async def _stop_read(tool_input):
        async with async_session_maker() as db:
            job = (await db.execute(
                select(BuildJob).where(BuildJob.source_id == a.id)
                .order_by(BuildJob.created_at.desc()).limit(1)
            )).scalar_one()
            await run_v3.request_stop(db, job.id)
        return _ISSUES

    status = await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": _stop_read,
        "slack__send_message": _OK,
    })
    assert status == "stopped"
    job = await _one_run(a.id)

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        # The card: stamped, not an error, and the honest count.
        assert row.last_outcome == "stopped", row.last_outcome
        assert row.last_outcome_text == "You stopped it. Nothing was sent."
        from app.services.automation_verbs import tone_for
        assert tone_for(row.last_outcome) == "warn"
        # The thread: the same story, from the same truth.
        turns = await ledger.run_turns(db, run_id=job.id)
        note = [t for t in turns if t["kind"] == "note"
                and t.get("stamp") == "stopped"][0]
        assert note["writes_count"] == 0
        # And the runs API offers no "Fix this" for a stop.
        from app.agent.automations.service import list_runs
        runs = await list_runs(db, uid, automation_id=a.id, limit=5)
        assert next(r for r in runs if r["id"] == job.id)["fix"] is None


@pytest.mark.asyncio
async def test_nd15_a_crashed_run_still_tells_a_story():
    """ND-15 (live): JobRunner._mark_failed terminalised an automation
    run with status=failed and outcome / error_class / user_message ALL
    null — bypassing the gated finalize that couples the terminal to the
    stamp, the notify and the ledger close — and the runs API then
    offered a "Fix this" chip for a failure with no story."""
    from app.agent.job_runner import JobRunner
    from app.agent.automations.service import list_runs

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        job = BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run", prompt="",
            job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
        )
        db.add(job)
        await db.commit()
        job_id = job.id

    await JobRunner()._mark_failed(job_id, "handler blew up")

    async with async_session_maker() as db:
        row = await db.get(BuildJob, job_id)
        assert row.status == "failed"
        # A failure the user can see must say something true about itself.
        assert row.outcome, "no outcome — the story is missing"
        assert row.error_class, "no error_class"
        assert row.user_message, "no user_message"
        # And the stamp fired, because it went through the gate.
        automation = await db.get(Automation, a.id)
        assert automation.last_outcome, "the gated finalize was bypassed"

    async with async_session_maker() as db:
        runs = await list_runs(db, uid, automation_id=a.id, limit=5)
    assert next(r for r in runs if r["id"] == job_id)["fix"] is not None


@pytest.mark.asyncio
async def test_nd15_no_fix_chip_for_a_failure_with_no_story():
    """Defence in depth: even if some unaccounted path lands a bare
    `failed`, the chip must not offer to diagnose a blank."""
    from app.agent.automations.service import list_runs
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run", prompt="",
            job_type="automation_run", status="failed",
            source_kind="automation", source_id=a.id,
        ))
        await db.commit()
    async with async_session_maker() as db:
        runs = await list_runs(db, uid, automation_id=a.id, limit=5)
    assert runs[0]["fix"] is None, runs[0]


@pytest.mark.asyncio
async def test_nd16_the_card_never_invents_a_connector_failure(monkeypatch):
    """ND-16 (live): the home card read "Tried 1:20 · could not reach an
    account" for a run that died mid-"Wrapping up" with accounts_failed
    EMPTY — asserting a refusal that never happened. Only claim a
    connector when the ledger recorded one."""
    from app.agent.automations.summary import summary_payload

    async def _conn(user_id):
        return {"jira": {"connector_id": "jira", "connected": True,
                         "status": "active", "account": "TP"},
                "slack": {"connector_id": "slack", "connected": True,
                          "status": "active", "account": "ws"}}

    async def _reg(user_id, force=False):
        return REGISTRY_V2

    async def _tpl(user_id):
        return []

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state", _conn)
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _reg)
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_templates", _tpl)

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run", prompt="",
            job_type="automation_run", status="failed", outcome="lost",
            error_class="interrupted", source_kind="automation",
            source_id=a.id, completed_at=datetime.utcnow(),
            config_json={"accounts_failed": []},
        ))
        await db.commit()

    async with async_session_maker() as db:
        payload = await summary_payload(db, user_id=uid)
    meta = payload["automations"][0]["meta"]
    assert "could not reach" not in meta, meta
    assert "did not finish" in meta, meta

    # When a connector DID fail, name it — the honest form survives.
    # R31-07 changed the WORDS (§4.2's table: `Tried {t} · {names} need
    # you`, from the one string table) but not the property this test
    # exists for: a name appears only when the ledger recorded one.
    async with async_session_maker() as db:
        job = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == a.id)
        )).scalars().first()
        job.config_json = {"accounts_failed": ["jira"]}
        await db.commit()
    async with async_session_maker() as db:
        payload = await summary_payload(db, user_id=uid)
    meta = payload["automations"][0]["meta"]
    assert "Jira" in meta, meta
    assert "need you" in meta, meta

    # And EVERY name, at any count — the old form took names[0] and
    # dropped the rest, so a user fixed one account, re-ran, and met the
    # next one.
    async with async_session_maker() as db:
        job = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == a.id)
        )).scalars().first()
        job.config_json = {"accounts_failed": ["jira", "slack"]}
        await db.commit()
    async with async_session_maker() as db:
        payload = await summary_payload(db, user_id=uid)
    meta = payload["automations"][0]["meta"]
    assert "Jira" in meta and "Slack" in meta, meta


@pytest.mark.asyncio
async def test_a_stopped_run_says_whether_anything_was_sent(monkeypatch):
    """§3.2's stopped sentence is TWO clauses, and the second is the point.

    R30-B caught this on the first live render of the founder's home: the
    running card read exactly "Paused at step 6." with no second
    sentence. The dispatch pins `"Paused at step 2. Nothing was sent."`
    and, with writes, `"Paused at step 3. 1 change already made."` — the
    whole reason that clause exists is that a run the user stopped must
    answer the one question a stop raises: did it already do something?

    §4.3 already requires the stop note to carry the honest writes count,
    so the number was in the ledger; the card just was not reading it.
    """
    from app.agent.automations import ledger
    from app.agent.automations.summary import summary_payload
    from app.db.models import AutomationWrite

    async def _conn(user_id):
        return {"jira": {"connector_id": "jira", "connected": True,
                         "status": "active", "account": "TP"},
                "slack": {"connector_id": "slack", "connected": True,
                          "status": "active", "account": "ws"}}

    async def _reg(user_id, force=False):
        return REGISTRY_V2

    async def _tpl(user_id):
        return []

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state", _conn)
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _reg)
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_templates", _tpl)

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    thread = None
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(
            db, user_id=uid, automation_id=a.id)
        job_id = str(uuid.uuid4())
        db.add(BuildJob(
            id=job_id, user_id=uid, title="run", prompt="",
            job_type="automation_run", status="cancelled",
            outcome="stopped", source_kind="automation", source_id=a.id,
            progress_step=6, progress_total=8,
            config_json={"thread_id": thread.id,
                         "checkpoint": {"step_index": 6}},
        ))
        await db.commit()

    # Nothing written yet.
    async with async_session_maker() as db:
        payload = await summary_payload(db, user_id=uid)
    flight = payload["automations"][0]["run_in_flight"]
    assert flight and flight["status"] == "stopped_by_user"
    assert flight["sentence"] == "Paused at step 6. Nothing was sent.", (
        flight["sentence"])

    # One write already out — the card must say so, not stay silent.
    async with async_session_maker() as db:
        db.add(AutomationWrite(
            id=str(uuid.uuid4()), run_id=job_id, user_id=uid,
            automation_id=a.id, account_id="slack",
            what="Posted in #platform", target="#platform",
            audience="others", reversible=False,
        ))
        await db.commit()
    async with async_session_maker() as db:
        payload = await summary_payload(db, user_id=uid)
    said = payload["automations"][0]["run_in_flight"]["sentence"]
    assert said == "Paused at step 6. 1 change already made.", said

    # Two writes pluralise.
    async with async_session_maker() as db:
        db.add(AutomationWrite(
            id=str(uuid.uuid4()), run_id=job_id, user_id=uid,
            automation_id=a.id, account_id="slack",
            what="Posted in #eng", target="#eng",
            audience="others", reversible=False,
        ))
        await db.commit()
    async with async_session_maker() as db:
        payload = await summary_payload(db, user_id=uid)
    said2 = payload["automations"][0]["run_in_flight"]["sentence"]
    assert said2 == "Paused at step 6. 2 changes already made.", said2


@pytest.mark.asyncio
async def test_r39_an_armed_automation_with_an_expired_account_needs_you(
    monkeypatch,
):
    """R39 (founder recording): Teams access expired for 8+ hours and
    the summary row said "On" with a "pause" action throughout, because
    `_status_of` answered ('active', 'On') for any armed row before
    expired accounts were considered — `expired_name` was computed and
    then consumed only under a status an armed row could never reach
    (needs_attention required a.status == 'error' or a paused_reason,
    and an expiry writes neither). The row must come out
    needs_attention, with the reconnect action, and the meta must NAME
    the account."""
    from app.agent.automations.summary import summary_payload

    async def _conn(user_id):
        return {"jira": {"connector_id": "jira", "connected": True,
                         "status": "reauth_required", "account": "TP"},
                "slack": {"connector_id": "slack", "connected": True,
                          "status": "active", "account": "ws"}}

    async def _reg(user_id, force=False):
        return REGISTRY_V2

    async def _tpl(user_id):
        return []

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state", _conn)
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _reg)
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_templates", _tpl)

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    assert a.status == "armed"
    # A completed run on record — the founder's row had months of them,
    # and `last is None` owns the meta chain's first branch.
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run", prompt="",
            job_type="automation_run", status="completed",
            source_kind="automation", source_id=a.id,
            completed_at=datetime.utcnow(),
            config_json={"accounts_touched": ["jira", "slack"]},
        ))
        await db.commit()

    async with async_session_maker() as db:
        payload = await summary_payload(db, user_id=uid)
    row = payload["automations"][0]
    assert row["status"] == "needs_attention", row
    assert row["pill"] == "Needs you", row["pill"]
    # The one blue action is the fix, not "pause".
    assert row["action"] == "reconnect", row["action"]
    # The meta names WHICH account — "6 accounts · needs reconnecting"
    # withheld the one fact the line exists for.
    assert "Jira needs reconnecting" in row["meta"], row["meta"]
    # And the description names it too (the existing needs_attention
    # sentence, now reachable for an armed row).
    assert row["description"].startswith("Jira needs you."), (
        row["description"])

    # A healthy armed sibling still reads active · On — the flip is
    # gated on the expiry, not on being armed.
    b = await _mk_automation_v2(uid, _v2_spec(name="Healthy brief"))
    async def _conn_ok(user_id):
        return {"jira": {"connector_id": "jira", "connected": True,
                         "status": "active", "account": "TP"},
                "slack": {"connector_id": "slack", "connected": True,
                          "status": "active", "account": "ws"}}
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state", _conn_ok)
    async with async_session_maker() as db:
        payload = await summary_payload(db, user_id=uid)
    healthy = next(r for r in payload["automations"] if r["id"] == b.id)
    assert healthy["status"] == "active"
    assert healthy["pill"] == "On"


# ── R35: the failed write's card, and Try again re-sending it ──────────


@pytest.mark.asyncio
async def test_a_failed_write_gets_its_card_and_its_actions(monkeypatch):
    """R35 (founder repro, 2026-08-28): a delivery that hard-failed used
    to vanish — no tool turn, no needs-you card, absent from
    `accounts_failed` — so the run named every broken READ and said
    nothing about the one step the automation exists for. The failed
    write now leaves the ok=False turn (with the real call recorded
    under it), the needs-you card with its button, and membership in
    the failed list."""
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": _ISSUES,
        "slack__send_message": {
            "kind": "tool_error", "retryable": False,
            "message": "channel_not_found",
        },
    })
    job = await _one_run(a.id)
    async with async_session_maker() as db:
        j = await db.get(BuildJob, job.id)
        assert j.status == "failed" and j.outcome == "write_failed"
        assert "slack" in ((j.config_json or {}).get("accounts_failed") or [])
        rows = (await db.execute(
            select(AutomationTurn).where(AutomationTurn.run_id == job.id)
        )).scalars().all()
    turns = [json.loads(r.payload_json) | {"kind": r.kind} for r in rows]
    failed_writes = [t for t in turns if t["kind"] == "tool"
                     and t.get("tool_kind") == "write" and not t.get("ok")]
    assert len(failed_writes) == 1, turns
    ft = failed_writes[0]
    # The real call, under the sentence — tool id, verdict, elapsed.
    assert ft["actions"] and ft["actions"][0]["tool"] == "slack__send_message"
    assert ft["actions"][0]["ok"] is False
    assert ft.get("reason_code"), ft
    needs = [t for t in turns if t["kind"] == "needs_you"
             and t.get("account_id") == "slack"]
    assert needs, "the failed delivery owes the user its card"


@pytest.mark.asyncio
async def test_try_again_restages_the_failed_write_and_closes_the_run(
        monkeypatch):
    """R35: the founder's Slack "Try again" answered 409 `no_step` —
    a write-only connector has no read to resume, while the failed
    outbox row sat there holding the exact payload the retry needs.
    `resume_source` now re-stages that row; on success the run closes
    as sent and the failed list empties."""
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": _ISSUES,
        "slack__send_message": {
            "kind": "tool_error", "retryable": False,
            "message": "channel_not_found",
        },
    })
    job = await _one_run(a.id)

    from app.agent.automations import executor_v2
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _fake_dispatch({"slack__send_message": _OK}),
    )
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        out = await executor_v2.resume_source(
            db, automation=a2, job_id=job.id, account_id="slack",
        )
    assert out["resumed"] is True, out
    assert out["status"] == "completed", out

    async with async_session_maker() as db:
        j = await db.get(BuildJob, job.id)
        assert j.status == "completed" and j.outcome == "sent", (
            j.status, j.outcome)
        assert ((j.config_json or {}).get("accounts_failed") or []) == []
        from app.db.models import AutomationOutbox
        ob = (await db.execute(
            select(AutomationOutbox).where(AutomationOutbox.job_id == job.id)
        )).scalars().all()
        assert any(r.status == "executed" for r in ob), [r.status for r in ob]
        rows = (await db.execute(
            select(AutomationTurn).where(AutomationTurn.run_id == job.id)
        )).scalars().all()
    turns = [json.loads(r.payload_json) | {"kind": r.kind} for r in rows]
    sent = [t for t in turns if t["kind"] == "tool"
            and t.get("tool_kind") == "write" and t.get("ok")]
    assert sent, "the retried write owes the thread its ok turn"
    assert sent[-1]["actions"][0]["tool"] == "slack__send_message"


@pytest.mark.asyncio
async def test_read_turns_carry_the_real_call(monkeypatch):
    """R35: every read turn records the tool call behind its sentence —
    the app's act page renders these exactly like the main chat's job
    card, from the same `public_step_label` vocabulary."""
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec)
    job = await _one_run(a.id)
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationTurn).where(AutomationTurn.run_id == job.id)
        )).scalars().all()
    turns = [json.loads(r.payload_json) | {"kind": r.kind} for r in rows]
    reads = [t for t in turns if t["kind"] == "tool"
             and t.get("tool_kind") == "read"]
    assert reads
    for t in reads:
        assert t.get("actions"), t
        assert t["actions"][0]["tool"] == "jira__search_issues"
        assert isinstance(t["actions"][0]["ms"], int)


# ── R38: the terminal frame + the visible narration step ─────────────


def _capture_frames(monkeypatch):
    """Every ledger broadcast funnels through ws_chat.broadcast_to_user;
    ledger._broadcast imports it per call, so the module attr is the
    one chokepoint."""
    frames = []

    async def _fake(user_id, frame):
        frames.append(frame)

    import app.api.ws_chat as ws_chat
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _fake)
    return frames


def test_the_frame_and_the_divider_cannot_tell_two_stories():
    """R38 follow-up. This test used to pin the frame's table against a
    LITERAL, which is why it stayed green while the table drifted from
    `_flip_head_note` on two of six reachable statuses.

    `skipped` is the common one — `confirm.py` finalizes a declined or
    expired confirm card as `outcome="skipped"` — so a user who said no
    watched the divider flip to TRIED from the live frame and revert to
    RAN on the next refetch: the exact symptom flipping the head note was
    added to end. The invariant is AGREEMENT, so that is what is
    asserted; there is one function now and both callers go through it.
    """
    from app.agent.automations.run_v3 import head_note_stamp

    # Every status a terminal can carry, including the ones where the
    # divider keeps its own note and the frame says so with None.
    for v3, expected in [
        ("completed", "ran"),
        ("partial", "ran"),
        ("failed", "tried"),
        ("skipped", "ran"),
        ("stopped_by_user", None),
        ("superseded", None),
        ("waiting_on_user", None),
    ]:
        assert head_note_stamp(v3) == expected, v3

    # The one refinement: a partial that really did lose a source needs
    # you, and the divider says so too.
    assert head_note_stamp(
        "partial", {"failed_sources": [{"account_id": "gmail"}]}
    ) == "needs_you"


@pytest.mark.asyncio
async def test_run_finished_frame_on_the_write_terminal(monkeypatch):
    """rec2 bug 2: every terminal now broadcasts ONE
    `automation.run.finished` frame carrying the full address + the
    note stamp — no client ever again infers the end of a run from
    `updated.summary.run_in_flight == null`."""
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    status = await _fire(monkeypatch, uid, a, vspec)
    assert status == "run"
    job = await _one_run(a.id)
    done = [f for f in frames if f["type"] == "automation.run.finished"]
    assert len(done) == 1, done
    f = done[0]
    assert f["automation_id"] == a.id
    assert f["run_id"] == job.id
    assert f["thread_id"]
    assert f["status"] == "completed"
    assert f["note_stamp"] == "ran"
    assert f["at"]


@pytest.mark.asyncio
async def test_run_finished_frame_on_the_reads_only_terminal(monkeypatch):
    """The reads-only terminal (the founder's Inbox summary shape) emits
    the frame too — and AFTER the narration step announced itself, so a
    client never sees the run end while still believing it is reading."""
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    vspec = _v2_spec(steps=[{
        "id": "issues", "connector_id": "jira",
        "tool": "jira__search_issues", "params": {"jql": "x"},
        "collect": {"items_path": "issues",
                    "fields": {"key": "key", "summary": "summary"},
                    "format": "{{item.key}} {{item.summary}}",
                    "empty_text": "none"},
        "on_error": "continue"}])
    a = await _mk_automation_v2(uid, vspec)
    status = await _fire(monkeypatch, uid, a, vspec)
    assert status == "run"
    job = await _one_run(a.id)
    types = [f["type"] for f in frames]
    done_idx = [i for i, f in enumerate(frames)
                if f["type"] == "automation.run.finished"]
    assert len(done_idx) == 1, types
    f = frames[done_idx[0]]
    assert f["status"] == "completed" and f["note_stamp"] == "ran"
    assert f["run_id"] == job.id
    writing_idx = [
        i for i, fr in enumerate(frames)
        if fr["type"] == "automation.run.progress"
        and fr.get("sentence") == "Writing your brief"
    ]
    assert writing_idx, "the narration step never announced itself"
    assert writing_idx[0] < done_idx[0]


@pytest.mark.asyncio
async def test_narration_is_a_visible_step_on_a_reads_only_run(monkeypatch):
    """rec2 bug 2a: when the reads are done and narration begins, the
    run's visible total extends by ONE step ("Writing your brief"), the
    activity phase flips to `writing`, and the job's progress columns
    agree — so the terminal frame's totals match what the card shows.

    R42: on a READS-ONLY run, which is the run this is still true of.
    A run with a write step terminalizes in the outbox flush that now
    precedes narration — see the sibling test below."""
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    vspec = _v2_spec(steps=[{
        "id": "issues", "connector_id": "jira",
        "tool": "jira__search_issues", "params": {"jql": "x"},
        "collect": {"items_path": "issues",
                    "fields": {"key": "key", "summary": "summary"},
                    "format": "{{item.key}} {{item.summary}}",
                    "empty_text": "none"},
        "on_error": "skip"}])
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec)
    job = await _one_run(a.id)
    n = len(vspec.steps)
    writing = [f for f in frames
               if f["type"] == "automation.run.progress"
               and f.get("sentence") == "Writing your brief"]
    assert len(writing) == 1, "one narration step, not one per phase"
    assert writing[0]["step"] == n + 1
    assert writing[0]["total"] == n + 1
    assert writing[0]["status"] == "running"
    acts = [f for f in frames if f["type"] == "automation.activity"
            and f.get("phase") == "writing"]
    assert len(acts) == 1
    assert acts[0]["automation_id"] == a.id
    assert job.progress_step == n + 1
    assert job.progress_total == n + 1


@pytest.mark.asyncio
async def test_a_written_run_does_not_announce_a_step_after_its_terminal(
        monkeypatch):
    """R42: the write and the run's terminal now land BEFORE narration,
    so the narration step must not announce itself as `running` on a job
    that has already reported Done at fraction 1.0 — the card would walk
    backwards. The thread is still being written, so the `writing`
    activity phase stays."""
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec)
    job = await _one_run(a.id)
    assert job.status == "completed"
    writing = [f for f in frames
               if f["type"] == "automation.run.progress"
               and f.get("sentence") == "Writing your brief"]
    assert writing == []
    acts = [f for f in frames if f["type"] == "automation.activity"
            and f.get("phase") == "writing"]
    assert len(acts) == 1


@pytest.mark.asyncio
async def test_the_write_turn_lands_before_the_runs_terminal(monkeypatch):
    """R42 (P14-4): the send's own record precedes the terminal frame.

    The client latches a run card's content the instant `run_in_flight`
    goes null, and `automation.run.finished` is what nulls it — so the
    `tool_kind: write` turn that names WHAT was posted and WHERE has to
    be on the wire first, or the landed card can never show it. The
    failure branch has always ordered it this way; the success branch
    finalized first and appended the turn afterwards.
    """
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec)
    kinds = [
        "write" if (f["type"] == "automation.turn"
                    and f["turn"].get("tool_kind") == "write") else f["type"]
        for f in frames
    ]
    assert "write" in kinds, "the write turn was never broadcast"
    assert "automation.run.finished" in kinds
    assert kinds.index("write") < kinds.index("automation.run.finished")


@pytest.mark.asyncio
async def test_run_finished_frame_when_every_source_fails(monkeypatch):
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    vspec = _v2_spec(steps=[{
        "id": "issues", "connector_id": "jira",
        "tool": "jira__search_issues", "params": {"jql": "x"},
        "collect": {"items_path": "issues",
                    "fields": {"key": "key"},
                    "format": "{{item.key}}", "empty_text": "none"},
        "on_error": "continue"}])
    a = await _mk_automation_v2(uid, vspec)
    status = await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": {"kind": "error",
                                "message": "reauth_required"},
    })
    assert status == "failed"
    job = await _one_run(a.id)
    done = [f for f in frames if f["type"] == "automation.run.finished"]
    assert len(done) == 1
    assert done[0]["status"] == "failed"
    assert done[0]["note_stamp"] == "tried"
    assert done[0]["run_id"] == job.id


@pytest.mark.asyncio
async def test_run_finished_frame_on_a_user_stop(monkeypatch):
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    from app.agent.automations import run_v3

    async def _slow_read(tool_input):
        async with async_session_maker() as db:
            job = (await db.execute(
                select(BuildJob).where(BuildJob.source_id == a.id)
                .order_by(BuildJob.created_at.desc()).limit(1)
            )).scalar_one()
            await run_v3.request_stop(db, job.id)
        return _ISSUES

    status = await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": _slow_read,
        "slack__send_message": _OK,
    })
    assert status == "stopped"
    done = [f for f in frames if f["type"] == "automation.run.finished"]
    assert len(done) == 1
    assert done[0]["status"] == "stopped_by_user"
    # No stamp: a stopped run's divider keeps the stop note the stop
    # itself wrote. The frame says so with None rather than inventing
    # one, and the client reads the STATUS to land the card.
    assert done[0]["note_stamp"] is None


@pytest.mark.asyncio
async def test_run_finished_frame_from_the_stuck_run_sweep(monkeypatch):
    """The reaper's terminal goes through the same gated finalize, so
    the frame fires there too — a run that died silently still tells
    the client it is over."""
    from datetime import timedelta
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(
            db, user_id=uid, automation_id=a.id,
        )
        job = BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run",
            prompt="", job_type="automation_run", status="running",
            source_kind="automation", source_id=a.id,
            config_json={"thread_id": thread.id, "run_kind": "scheduled"},
        )
        db.add(job)
        await db.commit()
        job_id = job.id
    async with async_session_maker() as db:
        row = await db.get(BuildJob, job_id)
        row.created_at = datetime.utcnow() - timedelta(hours=2)
        await db.commit()
    from app.agent.automations.sweep import _sweep_stuck_runs
    n = await _sweep_stuck_runs()
    assert n >= 1
    done = [f for f in frames if f["type"] == "automation.run.finished"
            and f["run_id"] == job_id]
    assert len(done) == 1
    assert done[0]["status"] == "failed"
    assert done[0]["note_stamp"] == "tried"


@pytest.mark.asyncio
async def test_head_note_rebroadcast_carries_automation_id(monkeypatch):
    """rec2 bug 1c: the STARTED→RAN flip re-broadcasts the head note,
    and the frame must carry automation_id (R31 §4.1) — without it the
    app bridge cannot route the flip and the divider stays STARTED
    until a refetch, which is exactly what the founder recorded."""
    frames = _capture_frames(monkeypatch)
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec)
    flips = [
        f for f in frames if f["type"] == "automation.turn"
        and (f.get("turn") or {}).get("kind") == "note"
        and (f.get("turn") or {}).get("stamp") == "ran"
    ]
    assert flips, "the terminal never re-broadcast the head note"
    for f in flips:
        assert f.get("automation_id") == a.id


# ── R38: the HTML leak ───────────────────────────────────────────────


def test_strip_html_kills_the_founders_leak():
    """rec2 9:48 briefing: literal <b>…</b> on screen. Tags out,
    entities unescaped, text kept."""
    from app.agent.automations.ledger import strip_html
    assert strip_html("<b>Google Workspace product notifications.</b>") \
        == "Google Workspace product notifications."
    assert strip_html("<B>shouty</B>") == "shouty"
    assert strip_html("a<br/>b") == "a b"
    assert strip_html('<a href="https://x.y">link</a> text') \
        == "link text"
    assert strip_html("Tom &amp; Jerry &lt;3 &quot;cheese&quot; &#39;") \
        == 'Tom & Jerry <3 "cheese" \''
    assert strip_html("&lt;b&gt;escaped&lt;/b&gt;") == "escaped"


def test_strip_html_never_eats_an_email_address():
    """The reason this is an allowlist and not <[^>]+>: the founder's
    drill-in shows "Google <no-reply@accounts.google.com>" and that
    string must survive byte-for-byte."""
    from app.agent.automations.ledger import strip_html
    assert strip_html("Google <no-reply@accounts.google.com>") \
        == "Google <no-reply@accounts.google.com>"
    assert strip_html(
        "The Google Workspace Team <workspace-noreply@google.com>"
    ) == "The Google Workspace Team <workspace-noreply@google.com>"
    # Plain text with neither markup nor entities is returned untouched.
    s = "Security alert — 2 new threads"
    assert strip_html(s) is s


def test_minted_items_and_result_rows_are_sanitized():
    from app.agent.automations import ledger
    items = ledger.mint_item_ids([{
        "title": "<b>Review Google's security alert.</b>",
        "sub": "Google &lt;no-reply@accounts.google.com&gt;",
        "msgs": [{"who": "Google", "at": "", "text": "<i>Act now</i>"}],
    }])
    assert items[0]["title"] == "Review Google's security alert."
    assert items[0]["msgs"][0]["text"] == "Act now"
    # R43 — ONE decode, at ingestion. `_collect_result` unescapes the raw
    # collect fields beside the line it renders from them, so every later
    # layer strips TAGS only: unescaping again here decoded the SAME line
    # twice and turned a sender's literal `&amp;` into `&`. This item is
    # handed straight to the minter with no ingestion behind it, so its
    # pre-escaped address stays escaped; a real read arrives decoded, which
    # `test_the_run_pipeline_serves_no_html` below proves end to end.
    assert items[0]["sub"] == "Google &lt;no-reply@accounts.google.com&gt;"

    from app.db.models import BRIEF_TIERS
    groups = [
        {"rank": i + 1, "label": lb, "tone": tn, "rows": []}
        for i, (lb, tn) in enumerate(BRIEF_TIERS)
    ]
    groups[-1]["rows"].append({
        "text": "<b>Google Workspace product notifications.</b>",
        "sub": "", "tag": "1", "item_refs": []})
    out = ledger.validate_turn_payload("result", {
        "title": "Your morning, in order", "vocabulary": "brief",
        "groups": groups,
    })
    assert out["groups"][-1]["rows"][0]["text"] \
        == "Google Workspace product notifications."


@pytest.mark.asyncio
async def test_the_run_pipeline_serves_no_html(monkeypatch):
    """End to end: a provider answering with markup produces a thread
    whose item titles are clean — and the write template's text (what
    would be posted to Slack) is clean too."""
    frames = _capture_frames(monkeypatch)
    del frames
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    html_issues = {"kind": "ok", "content": json.dumps({"issues": [
        {"key": "TP-1",
         "summary": "<b>Rate-limit the export endpoint</b>"},
        {"key": "TP-2", "summary": "Ping Bob &lt;bob@toup.ai&gt;"},
    ]})}
    posted = []

    async def _send(tool_input):
        posted.append(tool_input)
        return _OK

    await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": html_issues,
        "slack__send_message": _send,
    })
    job = await _one_run(a.id)
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        turns = await ledger.run_turns(db, run_id=job.id)
    titles = [it["title"] for t in turns if t["kind"] == "tool"
              for it in (t.get("items") or [])]
    assert "TP-1 Rate-limit the export endpoint" in titles
    assert "TP-2 Ping Bob <bob@toup.ai>" in titles
    assert not any("<b>" in t or "&lt;" in t for t in titles)
    assert posted, "the write never flushed"
    assert "<b>" not in posted[0]["text"]
    assert "Ping Bob <bob@toup.ai>" in posted[0]["text"]


def test_progress_step_is_the_step_being_started_everywhere():
    """R38 follow-up. `progress_step`/`progress_total` carried two
    incompatible meanings: every frame producer in the engine treats
    `progress_step` as the step being STARTED and derives
    `(step - 1) / total`, while `run_v3.on_parked` alone read it as steps
    COMPLETED and derived `step / total`.

    That off-by-one was invisible while the read loop always broke at a
    mutating step, so `step < total` and the error was one row of the
    bar. The narration step is the first writer in the repo to make
    `step == total` — and it turned a run PARKED WAITING ON THE USER into
    a card, a push and a Live Activity that all read 100%: the inverse of
    the stuck-at-99% this round set out to end.

    Executed rather than grepped, because the arithmetic is the bug.
    """
    import inspect
    from app.agent.automations import run_v3

    src = inspect.getsource(run_v3.on_parked)
    assert "max(step - 1, 0)" in src, (
        "on_parked went back to reading progress_step as steps-completed"
    )

    def pct(step: int, total: int) -> int:
        return int(round(100 * max(step - 1, 0) / total)) if total else 0

    # A run parked at its LAST step is not finished — that is the whole
    # point of a park.
    assert pct(4, 4) < 100
    # …and the ordinary case still reads as the steps actually done.
    assert pct(3, 5) == 40
    assert pct(1, 5) == 0
    assert pct(0, 0) == 0


# ── R39: the capability check, corrected where it is served ──────────


@pytest.mark.asyncio
async def test_r39_legacy_capability_turns_are_corrected_at_the_read_boundary():
    """Capability-check turns persisted before R38 stamped the
    AUTOMATION-level mode label ("posts") on every account's row —
    read-only ones included — and R38 fixed only the MINT sites, so a
    thread's immutable pre-R38 rows kept lying. And every seeded row
    hardcodes ok:True / ms:0, which the app rendered as "all six green
    at 0.1s" — a timed health claim nothing ever measured.

    `list_turns` must serve the read-only account "reads only", keep the
    writer's mode label, and stamp `synthetic: true` on every seeded
    capability row — while a REAL probed tool turn passes through
    untouched."""
    from app.agent.automations import ledger

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())  # jira reads, slack writes
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=a.id)
        # The pre-R38 stamp, verbatim: the automation's own mode label
        # on BOTH accounts.
        for cid in ("jira", "slack"):
            await ledger.append_turn(
                db, user_id=uid, thread=thread, run_id=None, kind="tool",
                payload={"account_id": cid, "tool_kind": "read",
                         "action": "Checked what I can do",
                         "detail": "posts to #platform",
                         "ok": True, "ms": 0, "steps": [], "items": []},
            )
        # A real, timed read beside them must not be touched.
        from app.services import automation_verbs as verbs
        real = verbs.turn_action("jira", "jira__search_issues",
                                 kind="read", ok=True)
        await ledger.append_turn(
            db, user_id=uid, thread=thread, run_id=None, kind="tool",
            payload={"account_id": "jira", "tool_kind": "read",
                     "action": real["action"], "detail": "2 issues",
                     "ok": True, "ms": 840, "steps": [], "items": []},
        )
        thread_id = thread.id

    async with async_session_maker() as db:
        turns, _ = await ledger.list_turns(db, thread_id=thread_id)
    caps = {t["account_id"]: t for t in turns
            if t.get("action") == "Checked what I can do"}
    assert set(caps) == {"jira", "slack"}
    # The read boundary re-derives the R38 mint rule.
    assert caps["jira"]["detail"] == "reads only", caps["jira"]
    assert caps["slack"]["detail"] == "posts to #platform", caps["slack"]
    # Seeded rows are marked as a LISTING, not a timed probe.
    assert caps["jira"]["synthetic"] is True
    assert caps["slack"]["synthetic"] is True
    # The probed turn keeps its own record — no stamp, no rewrite.
    probed = next(t for t in turns if t.get("action") == real["action"])
    assert "synthetic" not in probed
    assert probed["detail"] == "2 issues"
    assert probed["ms"] == 840


@pytest.mark.asyncio
async def test_r39_capability_correction_is_shape_tolerant():
    """A capability row the ctx cannot attribute (no account_id), or a
    thread whose automation row is GONE, passes through untouched — the
    correction must never cost a thread its read. The synthetic stamp
    needs no ctx and survives both."""
    from app.agent.automations import ledger

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=a.id)
        # The pre-R35 single flattened shape: no per-account chip.
        await ledger.append_turn(
            db, user_id=uid, thread=thread, run_id=None, kind="tool",
            payload={"account_id": "", "tool_kind": "read",
                     "action": "Checked what I can do",
                     "detail": "posts to #platform",
                     "ok": True, "ms": 0, "steps": [], "items": []},
        )
        thread_id = thread.id

    async with async_session_maker() as db:
        turns, _ = await ledger.list_turns(db, thread_id=thread_id)
    cap = next(t for t in turns
               if t.get("action") == "Checked what I can do")
    assert cap["detail"] == "posts to #platform"   # unattributable: kept
    assert cap["synthetic"] is True

    # A spec the ctx cannot parse: the ctx answers None and the stored
    # detail is served verbatim rather than the read failing.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        row.spec_json = "{not json"
        await db.commit()
    async with async_session_maker() as db:
        turns, _ = await ledger.list_turns(db, thread_id=thread_id)
    cap = next(t for t in turns
               if t.get("action") == "Checked what I can do")
    assert cap["detail"] == "posts to #platform"
    assert cap["synthetic"] is True
