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
