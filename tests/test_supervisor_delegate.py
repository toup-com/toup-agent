"""Supervisor + executor sub-agents — the `delegate` tool (voice round
2026-10-05, D-P6).

Lane: RUN_MODE=agent (build_jobs / job_events / users are AGENT_ONLY).

The supervisor is the user's own turn: it plans 2–6 steps, runs each on a
SUPERVISED child, JOINS on them inside the turn and hands the results back so
the model writes the merged answer itself. What is pinned here, each against
the real orchestrator + the real rows (only `agent_runner.run`, the chat
broadcast and the phone notify are doubles):

* plan → workers: independent steps start together (≤ max_workers), a step
  with ``depends_on`` starts only when its upstreams are done and receives
  their results in its task text;
* a supervised child announces NOTHING of its own — no day-chat message, no
  Live Activity start/terminal — and stores its result + sources on its row;
* its frames carry the §2.6 worker keys and never a ``channel`` key;
* a failed worker is retried ONCE as a NEW row (attempt 2, alternate model);
  a step whose upstream finally failed is ``skipped``;
* the plan deadline cancels a hung worker; a cancelled supervisor cancels its
  children (rows + tasks);
* the gate (image allowlist, per-tenant override, attended channels) and the
  60 s override cache; the role-model resolver fails CLOSED;
* `_tool_delegate` end-to-end: opens (or adopts) the supervisor job, emits
  ``supervisor.started`` through the SSE sink, returns every step's result.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from tests.test_subagent_orchestrator import _drain_background_tasks, _seed_user


@dataclass
class _Resp:
    text: str
    tokens_input: int = 10
    tokens_output: int = 5
    model: str = "gpt-test"


class _Runner:
    """`agent_runner.run` double. `script(task_text, call_no)` returns the
    reply text, raises, or awaits (a hung worker)."""

    def __init__(self, script):
        self.script = script
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kw):
        self.calls.append(kw)
        ev = kw.get("on_tool_event")
        if ev is not None:
            await ev({"phase": "start", "call_id": f"c{len(self.calls)}",
                      "name": "web_search", "input": {"query": "q"},
                      "started_ms": 1})
            await ev({"phase": "end", "call_id": f"c{len(self.calls)}",
                      "name": "web_search", "input": {"query": "q"},
                      "result": "ok", "elapsed_ms": 5,
                      "domains": ["example.com"],
                      "urls": ["https://www.example.com/a"]})
        out = self.script(kw["user_message"], len(self.calls))
        if asyncio.iscoroutine(out):
            out = await out
        return _Resp(text=out)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import app.agent.lanes as lanes
    import app.agent.subagent_orchestrator as orch
    import app.agent.subagent_message_writer as mw
    from app.agent import supervisor_gate
    from app.config import settings

    lanes._lane_manager = None
    supervisor_gate.reset_for_test()
    monkeypatch.setattr(settings, "subagent_spawning_enabled", True)
    calls: dict[str, list] = {"broadcast": [], "notify": [], "write": []}

    async def _bcast(user_id, event):
        calls["broadcast"].append(dict(event))

    async def _notify(**kw):
        calls["notify"].append(kw)

    async def _write(db, **kw):
        calls["write"].append(kw)
        return ("m", "d")

    import app.api.ws_chat as ws_chat
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)
    monkeypatch.setattr(orch, "_notify_job_event", _notify)
    monkeypatch.setattr(mw, "write_subagent_message", _write)
    yield calls
    lanes._lane_manager = None
    supervisor_gate.reset_for_test()


async def _user() -> str:
    from app.db.database import async_session_maker
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed_user(db, uid)
    return uid


async def _supervisor_job(uid: str, plan: list[dict]) -> str:
    from app.agent.job_runner import JobRunner, TaskSpec
    from app.agent.job_steps import dump_steps
    from app.agent.subagent_orchestrator import supervisor_plan_steps
    job = await JobRunner().create_job(
        job_type="agent_task",
        spec=TaskSpec(user_id=uid, channel="agent_task", source_kind="manual",
                      config_json={"job_type": "research", "role": "supervisor"}),
        title="Compare three things", prompt="p", status="running",
        model="m", layer=0,
        steps_json=dump_steps(supervisor_plan_steps(plan, "Compare three things")),
    )
    return job.id


async def _rows(parent: str):
    from sqlalchemy import select
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    async with async_session_maker() as db:
        return (await db.execute(
            select(BuildJob).where(BuildJob.parent_job_id == parent)
            .order_by(BuildJob.created_at)
        )).scalars().all()


async def _row(job_id: str):
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    async with async_session_maker() as db:
        return await db.get(BuildJob, job_id)


def _plan(n=3, deps=None):
    deps = deps or {}
    return [{"label": f"Step {i}", "task": f"TASK-{i}",
             "depends_on": deps.get(i, [])} for i in range(n)]


class _SerialMaker:
    """`async_session_maker` with one session at a time.

    The test DB is in-memory sqlite on a StaticPool: every session shares ONE
    connection, so two concurrent workers' sessions interleave and one's close
    (ROLLBACK) discards the other's uncommitted UPDATE — a test-harness
    artifact (production sessions have their own connections), recorded in
    app/db/database.py. Serialising the sessions keeps the workers concurrent
    while their writes stay whole."""

    def __init__(self):
        self.lock = asyncio.Lock()

    def __call__(self):
        return _SerialSession(self.lock)


class _SerialSession:
    def __init__(self, lock):
        self.lock = lock

    async def __aenter__(self):
        from app.db.database import async_session_maker
        await self.lock.acquire()
        self.cm = async_session_maker()
        return await self.cm.__aenter__()

    async def __aexit__(self, *exc):
        try:
            return await self.cm.__aexit__(*exc)
        finally:
            self.lock.release()


async def _run(uid, sup, plan, runner, **kw):
    from app.agent.subagent_orchestrator import run_supervised_plan
    args = dict(user_id=uid, supervisor_job_id=sup, title="Compare three things",
                plan=plan, agent_runner=runner, channel="web",
                executor_model="gpt-exec", alternate_model="gpt-default",
                max_workers=3, deadline_s=30, worker_timeout_s=20,
                session_maker=_SerialMaker())
    args.update(kw)
    return await run_supervised_plan(**args)


# ── plan → workers → join ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_runs_workers_and_hands_results_to_dependents(_env):
    uid = await _user()
    plan = _plan(3, deps={2: [0, 1]})
    sup = await _supervisor_job(uid, plan)
    starts: list[str] = []

    def script(task, n):
        starts.append(task)
        if "TASK-0" in task:
            return "RESULT-ZERO"
        if "TASK-1" in task:
            return "RESULT-ONE"
        return "MERGED"

    runner = _Runner(script)
    results = await _run(uid, sup, plan, runner)
    await _drain_background_tasks()

    assert [r["outcome"] for r in results] == ["success"] * 3
    assert results[0]["result"] == "RESULT-ZERO"
    assert results[0]["sources"][0]["domain"] == "example.com"
    # The dependent ran LAST and carried both upstream results.
    dependent = next(t for t in starts if "TASK-2" in t)
    assert starts.index(dependent) == 2
    assert "RESULT-ZERO" in dependent and "RESULT-ONE" in dependent

    rows = await _rows(sup)
    assert len(rows) == 3
    for r in rows:
        assert r.job_type == "subagent" and r.status == "completed"
        assert r.config_json["role"] == "executor"
        assert r.config_json["supervised"] is True
        assert r.config_json["result"]
        assert r.config_json["worker_model"] == "gpt-exec"
    steps = json.loads((await _row(sup)).steps_json)
    assert [s["status"] for s in steps[:3]] == ["done", "done", "done"]
    assert steps[3]["status"] == "running"      # Check and merge — the supervisor's
    assert steps[4]["status"] == "pending"      # Write the answer — the turn's


@pytest.mark.asyncio
async def test_supervised_worker_announces_nothing_of_its_own(_env):
    uid = await _user()
    plan = _plan(2)
    sup = await _supervisor_job(uid, plan)
    await _run(uid, sup, plan, _Runner(lambda t, n: "ok"))
    await _drain_background_tasks()
    assert _env["write"] == []                       # no day-chat message
    kinds = [c["kind"] for c in _env["notify"]]
    assert "mission_started" not in kinds            # no card of its own
    assert "mission_completed" not in kinds
    assert "mission_failed" not in kinds


@pytest.mark.asyncio
async def test_worker_frames_carry_the_worker_keys_and_no_channel(_env):
    uid = await _user()
    plan = _plan(2)
    sup = await _supervisor_job(uid, plan)
    sink_events: list[dict] = []

    async def sink(ev):
        sink_events.append(dict(ev))

    await _run(uid, sup, plan, _Runner(lambda t, n: "ok"), sink=sink)
    await _drain_background_tasks()

    worker = [e for e in _env["broadcast"] if e.get("parent_job_id") == sup]
    assert worker, "no worker job_update reached the chat socket"
    for e in worker:
        assert e["type"] == "job_update"
        assert e["role"] == "executor"
        assert isinstance(e["worker_index"], int)
        assert e["color"].startswith("#")
        assert e["plan_step_index"] in (0, 1)
        assert "channel" not in e
    assert any(e.get("status") == "completed" and e.get("result_preview") for e in worker)
    sup_frames = [e for e in _env["broadcast"] if e.get("job_id") == sup]
    assert sup_frames and sup_frames[-1]["role"] == "supervisor"
    assert sup_frames[-1]["workers_total"] == 2
    assert sup_frames[-1]["workers_done"] == 2
    assert all("channel" not in e for e in sup_frames)
    # The SSE: the same frames as job.update, plus each worker's actions on
    # the SUPERVISOR's job, keyed to the worker.
    assert any(e.get("type") == "job.update" and e.get("role") == "supervisor" for e in sink_events)
    tool = [e for e in sink_events if e.get("phase") == "start"]
    assert tool and tool[0]["job_id"] == sup and tool[0]["worker_job_id"] != sup
    assert tool[0]["parent_job_id"] == sup and isinstance(tool[0]["worker_index"], int)


# ── retry, skip, deadline, cancel ─────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failed_worker_is_retried_once_as_a_new_row_on_the_alternate_model(_env):
    uid = await _user()
    plan = _plan(2)
    sup = await _supervisor_job(uid, plan)

    def script(task, n):
        if "TASK-0" in task and not any("TASK-0" in c["user_message"] for c in runner.calls[:n - 1]):
            raise RuntimeError("provider down")
        return "ok"

    runner = _Runner(script)
    results = await _run(uid, sup, plan, runner)
    await _drain_background_tasks()

    assert results[0]["outcome"] == "success"
    step0 = [r for r in await _rows(sup) if r.config_json.get("plan_step_index") == 0]
    assert len(step0) == 2
    first, second = sorted(step0, key=lambda r: r.config_json["attempt"])
    assert first.status == "failed" and second.status == "completed"
    assert second.config_json["attempt"] == 2
    assert first.idempotency_key != second.idempotency_key
    assert second.config_json["worker_model"] == "gpt-default"
    assert first.config_json["worker_index"] != second.config_json["worker_index"]


@pytest.mark.asyncio
async def test_a_step_whose_upstream_finally_failed_is_skipped(_env):
    uid = await _user()
    plan = _plan(2, deps={1: [0]})
    sup = await _supervisor_job(uid, plan)

    def script(task, n):
        if "TASK-0" in task:
            raise RuntimeError("always")
        return "never"

    runner = _Runner(script)
    results = await _run(uid, sup, plan, runner)
    await _drain_background_tasks()
    assert results[0]["outcome"] == "failed"
    assert results[1]["outcome"] == "skipped"
    assert not any("TASK-1" in c["user_message"] for c in runner.calls)
    steps = json.loads((await _row(sup)).steps_json)
    assert steps[0]["status"] == "failed" and steps[1]["status"] == "skipped"


@pytest.mark.asyncio
async def test_the_plan_deadline_stops_a_hung_worker(_env, monkeypatch):
    """A worker's own timeout only bounds its RUN; one stuck before it (here:
    waiting for a lane slot) is stopped by the plan deadline, not left behind."""
    import app.agent.lanes as lanes
    uid = await _user()
    plan = _plan(2)
    sup = await _supervisor_job(uid, plan)
    never = asyncio.Event()

    class _Lanes:
        def __init__(self):
            self.n = 0

        async def acquire(self, *a, **k):
            self.n += 1
            if self.n == 2:
                await never.wait()
            return object()

        async def release(self, *a, **k):
            return None

    fake = _Lanes()
    monkeypatch.setattr(lanes, "get_lane_manager", lambda: fake)
    results = await _run(uid, sup, plan, _Runner(lambda t, n: "ok"), deadline_s=6)
    await _drain_background_tasks(timeout=5)      # nothing may be left running
    assert sorted(r["outcome"] for r in results) == ["success", "timeout"]


@pytest.mark.asyncio
async def test_cancelling_the_supervisor_cancels_its_children(_env):
    uid = await _user()
    plan = _plan(2)
    sup = await _supervisor_job(uid, plan)
    never = asyncio.Event()
    runner = _Runner(lambda t, n: never.wait())
    task = asyncio.ensure_future(_run(uid, sup, plan, runner, worker_timeout_s=120))
    for _ in range(200):
        if len(runner.calls) == 2:
            break
        await asyncio.sleep(0.05)
    assert len(runner.calls) == 2
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _drain_background_tasks(timeout=5)      # the children were cancelled
    rows = await _rows(sup)
    assert rows and all(r.status == "cancelled" for r in rows)


# ── gate, overrides, models ───────────────────────────────────────────


def test_gate_is_the_allowlist_unless_the_tenant_override_says_otherwise(monkeypatch):
    from app.agent import supervisor_gate as g
    from app.config import settings

    monkeypatch.setattr(settings, "supervisor_enabled", False)
    monkeypatch.setattr(settings, "user_id", "u-1")
    monkeypatch.setattr(settings, "supervisor_user_allowlist", "u-0, u-1")
    assert g.supervisor_enabled_for(settings) is True
    assert g.supervisor_enabled_for(settings, {"enabled": False}) is False
    monkeypatch.setattr(settings, "supervisor_user_allowlist", "u-0")
    assert g.supervisor_enabled_for(settings) is False
    assert g.supervisor_enabled_for(settings, {"enabled": True}) is True
    monkeypatch.setattr(settings, "supervisor_enabled", True)
    assert g.supervisor_enabled_for(settings) is True


def test_delegate_is_hidden_off_attended_surfaces_and_when_the_gate_is_off(monkeypatch):
    from app.agent import supervisor_gate as g
    from app.config import settings

    monkeypatch.setattr(settings, "supervisor_enabled", True)
    for ch in ("web", "app", "voice", "telegram"):
        assert g.supervisor_disabled_tools(ch, "u-1") == frozenset()
    for ch in ("subagent", "trigger", "automation_thread", "autopilot", None):
        assert g.supervisor_disabled_tools(ch, "u-1") == frozenset({"delegate"})
    monkeypatch.setattr(settings, "supervisor_enabled", False)
    monkeypatch.setattr(settings, "supervisor_user_allowlist", "")
    assert g.supervisor_disabled_tools("web", "u-1") == frozenset({"delegate"})
    g._cache["u-1"] = (0.0, {"enabled": True})          # a stale admin flip still applies
    assert g.supervisor_disabled_tools("web", "u-1") == frozenset()


@pytest.mark.asyncio
async def test_overrides_are_fetched_at_most_once_a_minute(monkeypatch):
    from app.agent import supervisor_gate as g

    n = {"calls": 0}

    async def fake_fetch(uid):
        n["calls"] += 1
        return {"supervisor_enabled": True, "executor_model": "gpt-5.6-luna",
                "supervisor_max_workers": 2, "unknown_key": 1}

    monkeypatch.setattr(g, "_fetch_flags", fake_fetch)
    a = await g.get_overrides("u-9")
    b = await g.get_overrides("u-9")
    assert n["calls"] == 1
    assert a == b == {"enabled": True, "executor_model": "gpt-5.6-luna", "max_workers": 2}


@pytest.mark.asyncio
async def test_a_failed_refresh_keeps_the_admin_kill_switch(monkeypatch):
    """TW-7. The operator switched the supervisor off for this tenant; a later
    refresh that cannot reach the platform must not erase that to ``{}`` —
    which falls back to the image allowlist and shows `delegate` again. An
    ANSWER replaces the override (an empty one included); a never-fetched
    tenant still gets ``{}``."""
    from app.agent import supervisor_gate as g
    from app.config import settings
    monkeypatch.setattr(settings, "supervisor_enabled", False)
    monkeypatch.setattr(settings, "supervisor_user_allowlist", "u1")
    monkeypatch.setattr(settings, "user_id", "u1")
    answers: list = [{"supervisor_enabled": False}, None, {}]

    async def fake_fetch(uid):
        return answers.pop(0)

    monkeypatch.setattr(g, "_fetch_flags", fake_fetch)
    assert await g.get_overrides("u1", force=True) == {"enabled": False}
    assert g.supervisor_disabled_tools("voice", "u1") == frozenset({"delegate"})
    stamp = g._cache["u1"][0]
    assert await g.get_overrides("u1", force=True) == {"enabled": False}   # blip
    assert g._cache["u1"][0] >= stamp
    assert g.supervisor_disabled_tools("voice", "u1") == frozenset({"delegate"})
    assert await g.get_overrides("u1", force=True) == {}                   # answered
    assert g.supervisor_disabled_tools("voice", "u1") == frozenset()

    async def down(uid):
        return None
    monkeypatch.setattr(g, "_fetch_flags", down)
    assert await g.get_overrides("never-seen", force=True) == {}


def test_role_models_fail_closed_to_the_default(monkeypatch):
    from app.config import settings
    from app.services import model_resolver as mr

    monkeypatch.setattr(settings, "agent_model", "gpt-6-sol")
    monkeypatch.setattr(settings, "subagent_model", None)
    monkeypatch.setattr(settings, "anthropic_enabled", False)
    assert mr.subagent_model({"executor_model": "gpt-5.6-luna"}) == "gpt-5.6-luna"
    assert mr.subagent_model({"executor_model": "typo-model"}) == "gpt-6-sol"
    assert mr.subagent_model({"executor_model": "claude-opus-4-7"}) == "gpt-6-sol"
    assert mr.subagent_model({"executor_model": "gpt-5.5-pro"}) == "gpt-6-sol"
    monkeypatch.setattr(settings, "subagent_model", "o3-mini")
    assert mr.subagent_model({}) == "o3-mini"
    assert mr.supervisor_model({"supervisor_model": "nonsense"}) == "gpt-6-sol"
    # TW-6: a routable PREFIX is not a chat model. A typo and every non-chat
    # model that starts with "gpt" fall back, for both roles.
    monkeypatch.setattr(settings, "subagent_model", None)
    for bad in ("gpt-5.5-mni", "gpt-image-1", "gpt-4o-transcribe", "gpt-realtime",
                "o3-nonsense", "gpt-realtime-2.1"):
        assert mr.subagent_model({"executor_model": bad}) == "gpt-6-sol", bad
        assert mr.supervisor_model({"supervisor_model": bad}) == "gpt-6-sol", bad
        assert mr.supervisor_model({"supervisor_model": bad}, fallback_default=False) is None, bad


def test_a_worker_is_never_a_supervisor():
    from app.agent.prompt_profile import SUBAGENT_DISABLED_TOOLS
    assert "delegate" in SUBAGENT_DISABLED_TOOLS


# ── the tool, end to end ──────────────────────────────────────────────


def _executor(runner):
    from app.agent import tool_executor as te_mod
    te = te_mod.ToolExecutor.__new__(te_mod.ToolExecutor)
    te.agent_runner = runner
    te.subagent_manager = None
    return te, te_mod


@pytest.mark.asyncio
async def test_delegate_refuses_when_the_gate_is_off(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "supervisor_enabled", False)
    monkeypatch.setattr(settings, "supervisor_user_allowlist", "")
    te, te_mod = _executor(_Runner(lambda t, n: "ok"))
    te_mod._USER_ID_CTX.set("u-1")
    te_mod._CHANNEL_CTX.set("web")
    from app.agent import supervisor_gate as g
    g._cache["u-1"] = (10**12, {})
    out = await te._tool_delegate({"title": "T", "steps": _plan(2)})
    assert str(out).startswith("ERROR: delegate is not available")


@pytest.mark.asyncio
async def test_delegate_validates_the_plan(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "supervisor_enabled", True)
    te, te_mod = _executor(_Runner(lambda t, n: "ok"))
    te_mod._USER_ID_CTX.set("u-1")
    te_mod._CHANNEL_CTX.set("web")
    from app.agent import supervisor_gate as g
    g._cache["u-1"] = (10**12, {})
    one = await te._tool_delegate({"title": "T", "steps": _plan(1)})
    assert "2 to 6" in str(one)
    fwd = await te._tool_delegate({"title": "T", "steps": _plan(2, deps={0: [1]})})
    assert "EARLIER" in str(fwd)


@pytest.mark.asyncio
async def test_delegate_opens_the_supervisor_job_and_returns_every_result(monkeypatch, _env):
    from app.config import settings
    from app.agent.subagent_orchestrator import SUPERVISOR_EVENT_SINK
    monkeypatch.setattr(settings, "supervisor_enabled", True)
    uid = await _user()
    te, te_mod = _executor(_Runner(lambda t, n: "R:" + t.split("\n")[-1][-6:]))
    te_mod._USER_ID_CTX.set(uid)
    te_mod._CHANNEL_CTX.set("voice")
    te_mod._JOB_ID_CTX.set(None)
    from app.agent import supervisor_gate as g
    g._cache[uid] = (10**12, {"max_workers": 1})
    sink_events: list[dict] = []

    async def sink(ev):
        sink_events.append(dict(ev))

    token = SUPERVISOR_EVENT_SINK.set(sink)
    try:
        out = await te._tool_delegate({"title": "Compare three phones",
                                       "steps": _plan(2)})
    finally:
        SUPERVISOR_EVENT_SINK.reset(token)
    await _drain_background_tasks()
    payload = json.loads(str(out))
    sup = payload["job_id"]
    assert [s["outcome"] for s in payload["steps"]] == ["success", "success"]
    assert "verify" in payload["instruction"]
    row = await _row(sup)
    assert row.config_json["role"] == "supervisor"
    assert [s["label"] for s in json.loads(row.steps_json)][-2:] == ["Check and merge", "Write the answer"]
    assert sup in te.peek_created_job_ids()      # closed by the turn finalizer
    started = [e for e in sink_events if e.get("type") == "supervisor.started"]
    assert started == [{"type": "supervisor.started", "job_id": sup,
                        "workers_total": 2, "deadline_s": 240}]
    assert len(await _rows(sup)) == 2


@pytest.mark.asyncio
async def test_delegate_adopts_the_turns_own_job(monkeypatch, _env):
    from app.config import settings
    monkeypatch.setattr(settings, "supervisor_enabled", True)
    uid = await _user()
    turn_job = await _supervisor_job(uid, _plan(2))
    te, te_mod = _executor(_Runner(lambda t, n: "ok"))
    te_mod._USER_ID_CTX.set(uid)
    te_mod._CHANNEL_CTX.set("web")
    te_mod._JOB_ID_CTX.set(turn_job)
    from app.agent import supervisor_gate as g
    g._cache[uid] = (10**12, {"max_workers": 1})   # one writer at a time: see _SerialMaker
    out = await te._tool_delegate({"title": "T", "steps": _plan(3)})
    await _drain_background_tasks()
    assert json.loads(str(out))["job_id"] == turn_job
    assert len(json.loads((await _row(turn_job)).steps_json)) == 5


@pytest.mark.asyncio
async def test_the_turns_runner_card_carries_the_supervisor_plan_to_its_close(monkeypatch, _env):
    """Integration (lanes PE + PF): on a voice delegation the turn's job is the
    runner's TurnJob, which keeps writing its row — progress, the terminal
    frame, the close. Unless it FOLLOWS the supervisor's plan it rewrites its
    template ("Search the web", …) over the plan and closes the workers'
    steps as skipped. `delegate` is not work of its own on that card."""
    from app.config import settings
    from app.db.database import async_session_maker
    from app.db.models import Conversation
    import app.agent.voice_jobs as vj
    from app.services import agent_notify_client as anc
    monkeypatch.setattr(settings, "supervisor_enabled", True)
    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)
    assert vj.is_work_tool("delegate") is False
    uid = await _user()
    conv = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Conversation(id=conv, user_id=uid, channel="voice"))
        await db.commit()
    tj = vj.TurnJob(user_id=uid, conversation_id=conv,
                    request_text="compare three phones with sources",
                    delegation_id="dlg-sup")
    vj.set_current_voice_job(tj)
    tj.declare(vj.template_plan("compare three phones", "web"))
    await _drain_background_tasks()
    te, te_mod = _executor(_Runner(lambda t, n: "R:" + t.split("\n")[-1][-6:]))
    te_mod._USER_ID_CTX.set(uid)
    te_mod._CHANNEL_CTX.set("voice")
    te_mod._JOB_ID_CTX.set(tj.job_id)
    from app.agent import supervisor_gate as g
    g._cache[uid] = (10**12, {"max_workers": 1})
    try:
        out = await te._tool_delegate({"title": "Compare three phones", "steps": _plan(2)})
        await _drain_background_tasks()
        assert json.loads(str(out))["job_id"] == tj.job_id
        # No begin_answer: the close alone must still treat the plan's last
        # row as the answer step, never skip it.
        tj.seal(final_text="done")
        await _drain_background_tasks()
    finally:
        vj.set_current_voice_job(None)
    row = await _row(tj.job_id)
    steps = json.loads(row.steps_json)
    assert row.status == "completed"
    assert [(s["label"], s["status"]) for s in steps] == [
        ("Step 0", "done"), ("Step 1", "done"),
        ("Check and merge", "done"), ("Write the answer", "done"),
    ], steps
    assert (row.config_json or {}).get("role") == "supervisor"


@pytest.mark.asyncio
async def test_adopting_the_runner_card_hands_it_the_plan_before_its_next_write(monkeypatch, _env):
    """The adoption itself (before any worker reports): the card's own next
    write — the `note_config` that records the role — must carry the plan,
    or it lands the template over the row the supervisor just wrote."""
    from app.config import settings
    import app.agent.voice_jobs as vj
    monkeypatch.setattr(settings, "supervisor_enabled", True)
    uid = await _user()
    tj = vj.TurnJob(user_id=uid, conversation_id=None,
                    request_text="compare three phones with sources",
                    delegation_id="dlg-sup2")
    vj.set_current_voice_job(tj)
    tj.declare(vj.template_plan("compare three phones", "web"))
    await _drain_background_tasks()
    te, te_mod = _executor(_Runner(lambda t, n: "ok"))
    te_mod._JOB_ID_CTX.set(tj.job_id)
    try:
        jid, _ = await te._open_supervisor_job(
            user_id=uid, title="Compare three phones",
            plan=[{"label": "Step 0"}, {"label": "Step 1"}],
            chat_id=None, asst_message_id=None)
        await _drain_background_tasks()
    finally:
        vj.set_current_voice_job(None)
    assert jid == tj.job_id
    labels = [s["label"] for s in json.loads((await _row(jid)).steps_json)]
    assert labels == ["Step 0", "Step 1", "Check and merge", "Write the answer"], labels


@pytest.mark.asyncio
async def test_a_plan_adopted_after_a_search_keeps_the_search_and_files_workers_by_row(
        monkeypatch, _env):
    """TW-3. The turn searched before it delegated: the card keeps that row
    ahead of the plan — on the row the adoption writes, not only on the
    card's own later writes — and each worker's actions are filed under the
    card's row for its plan step (row = offset + plan index), never by raw
    plan position, which would put worker 0's searches under the turn's own
    "Search the web"."""
    import app.agent.voice_jobs as vj
    from app.agent.step_tracker import StepTracker
    from app.config import settings
    monkeypatch.setattr(settings, "supervisor_enabled", True)
    held: list = []
    real_spawn = vj._spawn_bg
    monkeypatch.setattr(vj, "_spawn_bg", lambda coro, **kw: held.append(coro))
    uid = await _user()
    tracker = StepTracker()
    tj = vj.TurnJob(user_id=uid, conversation_id=None,
                    request_text="plan my trip to japan in detail",
                    delegation_id="dlg-off")
    vj.set_current_voice_job(tj)
    tj.plan([{"name": "web_search"}], tracker)
    for c in held:
        await c                      # the card's own row insert
    held.clear()
    te, te_mod = _executor(_Runner(lambda t, n: "ok"))
    te_mod._JOB_ID_CTX.set(tj.job_id)
    plan = _plan(2)
    try:
        jid, _ = await te._open_supervisor_job(
            user_id=uid, title="Plan the trip", plan=plan,
            chat_id=None, asst_message_id=None)
        # The adoption's own write, with the card's writers still held.
        labels = [s["label"] for s in json.loads((await _row(jid)).steps_json)]
        assert labels == ["Search the web", "Step 0", "Step 1",
                          "Check and merge", "Write the answer"], labels
        assert (tracker.steps_total, tracker.active) == (5, 1)
        # A work tool in the SAME round as `delegate` grows the card at its
        # tail; the plan's rows must still be found at offset + i.
        tj.plan([{"name": "web_fetch"}], tracker)
        monkeypatch.setattr(vj, "_spawn_bg", real_spawn)
        for c in held:
            c.close()
        sink_events: list[dict] = []

        async def sink(ev):
            sink_events.append(dict(ev))

        await _run(uid, jid, plan, _Runner(lambda t, n: "ok"), sink=sink,
                   max_workers=1)
        await _drain_background_tasks()
    finally:
        vj.set_current_voice_job(None)
    starts = [e for e in sink_events if e.get("phase") == "start"]
    assert sorted((e["step_index"], e["step_name"]) for e in starts) == [
        (1, "Step 0"), (2, "Step 1")], starts
    rows = [(s["label"], s["status"]) for s in json.loads((await _row(jid)).steps_json)]
    assert rows[:3] == [("Search the web", "done"), ("Step 0", "done"),
                        ("Step 1", "done")], rows
    assert rows[3] == ("Check and merge", "running"), rows


# ── projections ───────────────────────────────────────────────────────


def test_job_response_and_card_project_the_worker_identity():
    from types import SimpleNamespace
    from app.api.apps import _supervisor_fields
    from app.api.message_cards import job_card_fields

    child = SimpleNamespace(
        status="completed", app_id=None, title="Step 0", steps_json="[]",
        parent_job_id="sup-1", outcome="success", error_class=None,
        user_message=None,
        config_json={"role": "executor", "worker_index": 2, "plan_step_index": 0,
                     "worker_label": "Step 0", "worker_model": "gpt-x",
                     "result_preview": "done", "sources": [{"domain": "a.com"}]},
    )
    f = _supervisor_fields(child)
    assert f["role"] == "executor" and f["worker_index"] == 2
    assert f["label"] == "Step 0" and f["plan_step_index"] == 0
    assert f["result_preview"] == "done" and f["sources"] == [{"domain": "a.com"}]
    assert _supervisor_fields(SimpleNamespace(config_json={"job_type": "x"})) == {}
    msg = SimpleNamespace(role="job", content=json.dumps({"job_id": "c-1"}))
    card = job_card_fields(msg, {"c-1": child})
    assert card["parent_job_id"] == "sup-1" and card["job_role"] == "executor"
    # Spread into the message payload, the card must not rename the message.
    assert "role" not in card and {"role": "job", **card}["role"] == "job"


def test_the_voice_stream_passes_the_worker_keys():
    from app.api.api_v1 import _vs_worker_fields
    assert _vs_worker_fields({"worker_job_id": "w", "parent_job_id": "p",
                              "worker_index": "1", "label": "L", "model": "m",
                              "other": 1}) == {
        "worker_job_id": "w", "parent_job_id": "p", "worker_index": 1,
        "label": "L", "model": "m"}
    assert _vs_worker_fields({"name": "web_search"}) == {}


@pytest.mark.asyncio
async def test_the_voice_stream_carries_the_supervisor_frames(monkeypatch):
    """BEHAVIOURAL, through the real route: whatever the turn does on the
    sink comes out of the SSE — `supervisor.started` and a supervisor
    `job.update` exempt from the per-turn budget, a worker's action as a
    `tool.start` keyed to the worker."""
    from app.api import api_v1
    from app.agent.subagent_orchestrator import SUPERVISOR_EVENT_SINK
    from app.config import settings

    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", "secret-agent-key")
    monkeypatch.setattr(settings, "user_id", "owner-1")
    monkeypatch.setattr(settings, "security_leak_filter", False)
    monkeypatch.setattr(api_v1, "_VS_MAX_EVENTS", 0)   # only exempt frames survive

    class _Done:
        text, session_id, model = "answer", "s", "m"
        tokens_input = tokens_output = tokens_total = processing_time_ms = 1
        tool_calls: list = []
        persisted: dict = {}

    class _SupRunner:
        async def run(self, **kw):
            sink = SUPERVISOR_EVENT_SINK.get()
            assert sink is not None, "the route bound no supervisor sink"
            await sink({"type": "supervisor.started", "job_id": "sup",
                        "workers_total": 2, "deadline_s": 240})
            await sink({"type": "job.update", "job_id": "sup", "role": "supervisor",
                        "steps": [], "workers_total": 2, "workers_done": 0})
            return _Done()

    class _Req:
        headers = {"X-Agent-Key": "secret-agent-key"}

    monkeypatch.setattr(api_v1, "_agent_runner", _SupRunner())
    resp = await api_v1.internal_agent_turn_stream(
        api_v1.ChatRequest(message="compare three phones", save=False), _Req())
    frames = []
    async for chunk in resp.body_iterator:
        for line in str(chunk).splitlines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    kinds = [f["type"] for f in frames]
    assert "supervisor.started" in kinds and "job.update" in kinds and "done" in kinds


@pytest.mark.asyncio
async def test_a_workers_action_reaches_the_stream_with_its_worker_keys(monkeypatch):
    from app.api import api_v1
    from app.agent.subagent_orchestrator import SUPERVISOR_EVENT_SINK
    from app.config import settings

    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", "secret-agent-key")
    monkeypatch.setattr(settings, "user_id", "owner-1")
    monkeypatch.setattr(settings, "security_leak_filter", False)

    class _Done:
        text, session_id, model = "answer", "s", "m"
        tokens_input = tokens_output = tokens_total = processing_time_ms = 1
        tool_calls: list = []
        persisted: dict = {}

    class _WorkerRunner:
        async def run(self, **kw):
            await SUPERVISOR_EVENT_SINK.get()({
                "phase": "start", "call_id": "w1", "name": "web_search",
                "input": {"query": "phones"}, "started_ms": 5, "job_id": "sup",
                "step_index": 1, "worker_job_id": "wk-1", "parent_job_id": "sup",
                "worker_index": 3, "label": "Research B", "model": "gpt-exec"})
            return _Done()

    class _Req:
        headers = {"X-Agent-Key": "secret-agent-key"}

    monkeypatch.setattr(api_v1, "_agent_runner", _WorkerRunner())
    resp = await api_v1.internal_agent_turn_stream(
        api_v1.ChatRequest(message="compare three phones", save=False), _Req())
    frames = []
    async for chunk in resp.body_iterator:
        for line in str(chunk).splitlines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    start = next(f for f in frames if f["type"] == "tool.start")
    assert start["job_id"] == "sup" and start["step_index"] == 1
    assert (start["worker_job_id"], start["parent_job_id"], start["worker_index"],
            start["label"], start["model"]) == ("wk-1", "sup", 3, "Research B", "gpt-exec")


# ── F2: a failed worker step stays failed at a completed close ────────


@pytest.mark.asyncio
async def test_a_failed_worker_step_stays_failed_when_the_supervisor_closes(_env):
    """A supervisor's rows are its workers. The supervisor answering anyway
    completes the CARD; it does not make the worker that failed a success.
    `finish_all_steps` used to green every unresolved row, `failed` included,
    so a card whose second helper died read four ticks. A non-supervisor card
    keeps the old contract (a failed row there was recovered in-turn)."""
    from datetime import datetime
    from app.agent.job_reconciler import close_job_completed
    from app.agent.job_steps import dump_steps
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    uid = await _user()
    plain = await _supervisor_job(uid, _plan(2))
    sup = await _supervisor_job(uid, _plan(2))
    async with async_session_maker() as db:
        for jid, role in ((sup, "supervisor"), (plain, None)):
            row = await db.get(BuildJob, jid)
            steps = json.loads(row.steps_json)
            steps[0]["status"] = "done"
            steps[1]["status"] = "failed"
            row.steps_json = dump_steps(steps)
            row.config_json = {"job_type": "research", **({"role": role} if role else {})}
        await db.commit()
    for jid in (sup, plain):
        async with async_session_maker() as db:
            assert await close_job_completed(db, jid, user_id=uid, now=datetime.utcnow())
            await db.commit()
    got = {jid: [s["status"] for s in json.loads((await _row(jid)).steps_json)]
           for jid in (sup, plain)}
    assert got[sup] == ["done", "failed", "done", "done"], got[sup]
    assert got[plain] == ["done", "done", "done", "done"], got[plain]


@pytest.mark.asyncio
async def test_the_runner_cards_terminal_frame_keeps_a_failed_worker_failed(monkeypatch, _env):
    """Same rule on the delegation stream: the TurnJob's own `completed` frame
    is built in `seal`, before the row closes, and the phone draws it."""
    import app.agent.voice_jobs as vj
    from app.db.database import async_session_maker
    from app.db.models import Conversation
    from app.services import agent_notify_client as anc
    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)
    uid = await _user()
    conv = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Conversation(id=conv, user_id=uid, channel="voice"))
        await db.commit()
    frames: list = []
    tj = vj.TurnJob(user_id=uid, conversation_id=conv, request_text="compare",
                    delegation_id="dlg-f2", frame_sink=frames.append)
    tj.declare(vj.template_plan("compare three phones", "web"))
    await _drain_background_tasks()
    tj.follow_steps([
        {"id": "w0", "label": "Step 0", "status": "done"},
        {"id": "w1", "label": "Step 1", "status": "failed"},
        {"id": "m", "label": "Check and merge", "status": "running"},
        {"id": "a", "label": "Write the answer", "status": "pending"},
    ])
    tj.note_config(role="supervisor", workers_total=2)
    tj.seal(final_text="done")
    await _drain_background_tasks()
    done = [f for f in frames if f.get("status") == "completed"]
    assert done, frames
    assert [s["status"] for s in done[-1]["steps"]] == ["done", "failed", "done", "done"]


# ── Voice round 2026-10-06 (D-T2): a heavy request can reach the supervisor ──

MULTI_PART = [
    "compare the three best noise-cancelling headphones for me, with sources",
    "Compare the three best X for me, with sources",
    "compare iPhone 17 and Pixel 10 cameras",
    "what are the top 3 laptops for students right now",
    "pros and cons of electric cars for a family",
    "سه تا از بهترین هدفون‌ها رو با منبع مقایسه کن",
    "بهترین سه تا رستوران تورنتو رو برام پیدا کن",
    "Who won the last F1 race? And who won the UFC fight on Saturday?",
    "check the weather in Toronto and also find me a sushi place nearby",
    "قیمت بیت کوین امروز چنده؟ دلار به یورو چنده؟",
    "iPhone 17 vs Pixel 10, which is better for photos",
]
SINGLE_ASK = [
    "یواف‌سی چی", "UFC?", "UFC? Miami?",
    "برام یه چیز دیگه هم چک کنی؟ مسابقات اینتر میامی",
    "can you check one more thing, Miami FC",
    "آهنگ از دریک بذار", "stop the music", "what's the best pizza in Toronto",
    "who is the best player in the NBA", "search it", "چی شد", "فورمولوان",
    "Formula one race this weekend, where is it?", "should I buy or lease a car",
    "compare it", "compare my calendar this week with next week",
    "مسابقه اف‌وان این هفته کجاست؟",
    # Phase D (correctness-tenant-4): everyday single questions the join and
    # compare grammar read as heavy — each was offered the 3-worker
    # supervisor as a REQUIRED first choice.
    "how much does Disney plus cost per month", "Lakers vs Celtics score last night",
    "who won the Barcelona versus Madrid game",
    "what happened when he died and then came back to life",
    "what is the score? who scored?", "remind me to call mom and then text dad tomorrow",
    "best 2 for 1 deals near me", "play the top 5 songs",
]


@pytest.mark.parametrize("said", MULTI_PART)
def test_a_multi_part_request_is_recognised(said):
    from app.agent.voice_jobs import multi_part_request

    assert multi_part_request(said), said


@pytest.mark.parametrize("said", SINGLE_ASK)
def test_a_single_lookup_is_never_multi_part(said):
    """The call's own UFC-then-Miami pair, its media asks and its fillers: the
    supervisor is for heavy work, and a false positive hands a one-search
    question to three workers."""
    from app.agent.voice_jobs import multi_part_request

    assert not multi_part_request(said), said


def test_a_question_intent_gains_delegate_only_through_the_helper():
    from app.agent.query_intent import (
        INTENT_FULL, classify_query_intent, with_supervisor_tool,
    )

    q = classify_query_intent("compare the three best headphones for me")
    assert "delegate" not in q.tool_names
    assert "delegate" in with_supervisor_tool(q).tool_names
    assert with_supervisor_tool(INTENT_FULL) is INTENT_FULL


def test_every_attended_turn_logs_the_supervisor_gate(monkeypatch, caplog):
    """One line per attended turn with the effective models and limits — the
    2026-10-06 trail could not say whether a turn even SAW `delegate`."""
    import logging
    from app.agent import agent_runner as ar
    from app.agent import supervisor_gate as g
    from app.config import settings

    monkeypatch.setattr(settings, "supervisor_max_workers", 3)
    monkeypatch.setitem(g._cache, "u-trace", (10**12, {
        "enabled": True, "executor_model": "gpt-5.5-mini", "max_workers": 2}))
    caplog.set_level(logging.INFO, logger=ar.logger.name)
    ar._log_supervisor_gate("voice", "u-trace", False, True)
    ar._log_supervisor_gate("trigger", "u-trace", True, False)
    lines = [r.getMessage() for r in caplog.records if "[SUPERVISOR] gate" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "on=1 src=override multi_part=1 channel=voice" in lines[0]
    assert "max_workers=2" in lines[0]
    assert "executor_model=" in lines[0] and "supervisor_model=" in lines[0]
