"""Round 8 — a job must not outlive the turn that delivered its answer.

Production, 2026-08-19 (founder's tenant, job 6dfd5833): the answer rendered
in chat at 03:01:45, the job card sat on "In progress · 2/3 · 67%" for hours.
Its outbox showed mission_started → ONE progress push (2/3) → nothing. The
turn-end finalizer never ran for it.

Root cause: Round 4's follow-up (a9980513) moved job bookkeeping into
``asyncio.gather(_run_bookkeeping(), batch)`` so create_job/update_job overlap
the web batch. ``gather`` wraps the coroutine in a Task, and a Task runs in a
COPY of the context — so ``_CREATED_JOB_IDS_CTX.set(...)`` inside
``_tool_create_job`` landed in the Task's copy and vanished when it finished.
The finalizer (and the cancellation sweep) read ``()`` and closed nothing.
Every job created in the Round-4 prescribed shape (create_job IN THE SAME
RESPONSE as the first step's searches) was left running; the canary tenant
shows the flip at 01:36Z, the minute that image rolled.

The Round-4 end-to-end test could not see it: it stubbed ``tools.execute``
wholesale, so the real ``_tool_create_job`` — and its ContextVar write — never
ran. The turn below runs the REAL bookkeeping tools against the test DB and
fakes only the web tools.

These tests need RUN_MODE=agent (build_jobs / job_events / the outbox are
AGENT_ONLY tables) — see COVERAGE_DEBT.txt.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta

import pytest


# ── Harness ─────────────────────────────────────────────────────────────

class _ScriptedLLM:
    """Round 1: create_job + two web_searches in ONE response (the Round-4
    contract shape). Round 2: update_job(current_step=1) + a web_fetch — one
    tick that closes TWO steps, exactly what the founder's model did. Round 3:
    the answer. ``jobs`` is the shared dict the executor wrapper fills with the
    real job id, because the scripted update_job must name a REAL row."""

    def __init__(self, jobs: dict, *, second_round_step: int = 1):
        self.calls = 0
        self.jobs = jobs
        self.second_round_step = second_round_step

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent as E
        self.calls += 1
        if self.calls == 1:
            batch = [
                ("c1", "create_job", {
                    "title": "Find the best AI video model",
                    "description": "Check releases and comparisons",
                    "steps": ["Research current models", "Compare evidence", "Write recommendation"],
                }),
                ("s1", "web_search", {"query": "x"}),
                ("s2", "web_search", {"query": "y"}),
            ]
        elif self.calls == 2:
            batch = [
                ("u1", "update_job", {"job_id": self.jobs["id"], "current_step": self.second_round_step}),
                ("f1", "web_fetch", {"url": "https://docs.example.com/x"}),
            ]
        else:
            yield E(type="text", text="**Kling 2.1** is my pick today.")
            yield E(type="message_end", stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 2})
            return
        for cid, name, _ in batch:
            yield E(type="tool_use_start", tool_name=name, tool_id=cid)
        for cid, name, inp in batch:
            yield E(type="tool_use_end", tool_name=name, tool_id=cid, tool_input=inp)
        yield E(type="message_end", stop_reason="tool_use",
                usage={"input_tokens": 10, "output_tokens": 5})


async def _make_user() -> str:
    from app.db import async_session_maker, User
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"{user_id[:8]}@test.local",
                    hashed_password="x" * 60, name="R8"))
        await db.commit()
    return user_id


async def _run_turn(monkeypatch, tmp_path, *, second_round_step: int = 1,
                    channel: str = "web"):
    """One end-to-end turn with the REAL create_job/update_job. Returns
    (response, job_id, broadcast frames, background tasks awaited)."""
    import app.agent.agent_runner as ar
    from app.agent.tool_executor import ToolExecutor
    from app.services import agent_notify_client as anc

    async def _no_history(self, db, session_id, max_messages: int = 50, client_tz=None):
        return []

    async def _fixed_prompt(self, *a, **kw):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", _no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", _fixed_prompt)
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)
    # Deterministic outbox: no opportunistic flush task.
    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)

    # Background work the finalizer schedules (card pushes) — keep the
    # coroutines and await them after the turn, so the assertions can see
    # what they wrote instead of closing them unrun.
    bg: list = []

    def _keep(coro, **kw):
        t = asyncio.get_event_loop().create_task(coro)
        bg.append(t)
        return t

    monkeypatch.setattr(ar, "_spawn_background", _keep)
    monkeypatch.setattr(ar, "_spawn_bg", _keep)

    frames: list = []

    async def _capture(user_id, event, **kw):
        frames.append(dict(event))
        return 0

    import app.api.ws_chat as ws
    monkeypatch.setattr(ws, "broadcast_to_user", _capture)

    tools = ToolExecutor(workspace=str(tmp_path))
    real_execute = tools.execute
    jobs: dict = {}

    async def fake_execute(name, inp):
        if name in ("create_job", "update_job"):
            out = await real_execute(name, inp)
            if name == "create_job":
                try:
                    jobs["id"] = json.loads(out)["job_id"]
                except Exception:  # noqa: BLE001
                    pass
            return out
        if name == "web_search":
            await asyncio.sleep(0.05)
            return f"1. Result for {inp['query']}\n   https://{inp['query']}.example.org/page\n"
        if name == "web_fetch":
            await asyncio.sleep(0.02)
            return "# Docs\nbody text"
        return "?"

    monkeypatch.setattr(tools, "execute", fake_execute)

    runner = ar.AgentRunner(llm_service=_ScriptedLLM(jobs, second_round_step=second_round_step),
                            tool_executor=tools)  # type: ignore[arg-type]
    user_id = await _make_user()
    session_id = str(uuid.uuid4())
    resp = await runner.run(
        user_message="find the best AI video model", user_id=user_id,
        session_id=session_id, channel=channel,
        disable_post_processing=True, model_override="gpt-5.5-mini",
    )
    if bg:
        await asyncio.gather(*bg, return_exceptions=True)
    return resp, jobs.get("id"), frames, user_id, session_id


async def _job_row(job_id: str):
    from app.db import async_session_maker
    from app.db.models import BuildJob
    async with async_session_maker() as db:
        return await db.get(BuildJob, job_id)


async def _outbox_rows(job_id: str) -> list:
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import AgentNotifyOutbox
    async with async_session_maker() as db:
        rows = (await db.execute(select(AgentNotifyOutbox).order_by(AgentNotifyOutbox.created_at))).scalars().all()
    return [r for r in rows if (r.data_json or {}).get("job_id") == job_id]


# ── THE regression: bookkeeping in the batch, job still closes ───────────

@pytest.mark.asyncio
async def test_job_created_alongside_the_web_batch_is_completed_at_turn_end(monkeypatch, tmp_path):
    """The founder's turn shape. create_job runs concurrently with the
    searches; the answer is delivered; the row MUST be completed, its steps
    all done, the in-app surfaces told, and the phone card ended."""
    resp, job_id, frames, user_id, session_id = await _run_turn(monkeypatch, tmp_path)
    assert job_id, "the real create_job must have run"
    assert "Kling" in resp.text

    row = await _job_row(job_id)
    assert row is not None
    assert row.status == "completed", (
        f"job left {row.status!r} after its answer was delivered — the "
        "finalizer never saw the id (ContextVar write lost in gather)")
    assert row.completed_at is not None
    steps = json.loads(row.steps_json)
    assert [s["status"] for s in steps] == ["done", "done", "done"], steps
    # Real timing on every step: a window that opened and closed.
    for s in steps:
        assert s.get("started_at") and s.get("completed_at"), s
        assert isinstance(s.get("duration_ms"), int) and s["duration_ms"] >= 0, s
    # The answer message this job belongs to.
    assert row.summary_message_id == resp.asst_message_id

    # In-app surfaces (web card is WS-driven only): a completion frame with the
    # full counts, so the card can hit 100% without a refetch.
    done_frames = [f for f in frames if f.get("type") == "job_update"
                   and f.get("job_id") == job_id and f.get("status") == "completed"]
    assert done_frames, f"no completion job_update frame; frames={frames}"
    f = done_frames[-1]
    assert f["completed_steps"] == 3 and f["total_steps"] == 3
    # the runner mints the conversation when the given id has no row —
    # compare against what it actually used
    assert f.get("chat_id") == resp.session_id and f.get("message_id") == resp.asst_message_id
    assert row.conversation_id == resp.session_id

    # Phone: the terminal card push, with the answer preview and n/n steps.
    pushes = await _outbox_rows(job_id)
    kinds = [p.event_kind for p in pushes]
    assert kinds[0] == "mission_started"
    assert "mission_completed" in kinds, kinds
    done_push = [p for p in pushes if p.event_kind == "mission_completed"][-1]
    assert done_push.data_json["steps_done"] == 3 and done_push.data_json["steps_total"] == 3
    assert done_push.data_json.get("preview", "").startswith("Kling 2.1")
    assert done_push.data_json.get("phase") == "completed"


@pytest.mark.asyncio
async def test_one_tick_closing_two_steps_gives_no_zero_ms_step(monkeypatch, tmp_path):
    """update_job(current_step=1) straight from step 0 closes two steps at
    once. The client used to read the skipped step's window as 0ms. Server
    timing: a step closed without ever running shares the window it was
    closed in — never zero, never unknown."""
    resp, job_id, frames, *_ = await _run_turn(monkeypatch, tmp_path, second_round_step=1)
    row = await _job_row(job_id)
    steps = json.loads(row.steps_json)
    assert all(s["status"] == "done" for s in steps)
    # step 0 ran from create to the tick; step 1 was closed by the same tick
    # without ever being the running step → shares step 0's window.
    assert steps[1]["duration_ms"] > 0
    assert steps[1]["started_at"] == steps[0]["started_at"]
    assert steps[1]["completed_at"] == steps[0]["completed_at"]
    # step 2 ran from the tick to turn end (the finalizer closes it).
    assert steps[2]["started_at"] == steps[1]["completed_at"]
    assert steps[2]["duration_ms"] > 0
    # the update_job event frame told the clients the real counts
    upd = [f for f in frames if f.get("type") == "job_update" and f.get("job_id") == job_id
           and f.get("status") == "running"]
    assert upd and upd[-1]["completed_steps"] == 2


@pytest.mark.asyncio
async def test_registry_survives_a_task_boundary(tmp_path, test_user_id):
    """The mechanism, in isolation: create_job inside a gather-spawned Task
    must be visible to the parent's finalizer. This is the exact call shape
    the runner uses; if it regresses, everything above regresses with it."""
    from app.agent.tool_executor import ToolExecutor
    te = ToolExecutor(workspace=str(tmp_path))
    te.set_user_id(test_user_id)
    te.set_session_id(f"sess-{uuid.uuid4()}", "msg-1")

    created: dict = {}

    async def bookkeeping():
        out = await te._tool_create_job({"title": "T", "steps": ["a", "b"]})
        created["id"] = json.loads(out)["job_id"]
        # peek from INSIDE the task: this turn's job
        assert created["id"] in te.peek_created_job_ids()

    async def batch():
        await asyncio.sleep(0.01)

    await asyncio.gather(bookkeeping(), batch())
    assert te.take_created_job_ids() == (created["id"],), (
        "the parent context must see the id the Task recorded")
    assert te.take_created_job_ids() == ()


@pytest.mark.asyncio
async def test_concurrent_turns_still_do_not_share_the_registry(tmp_path, test_user_id):
    """The isolation invariant that motivated the ContextVar must survive the
    fix: two turns on the shared executor each see only their own job."""
    from app.agent.tool_executor import ToolExecutor
    te = ToolExecutor(workspace=str(tmp_path))
    te.set_user_id(test_user_id)
    seen: dict = {}

    async def turn(tag: str):
        te.set_session_id(f"sess-{tag}")

        async def bk():
            out = await te._tool_create_job({"title": tag, "steps": ["a"]})
            seen[tag + ":id"] = json.loads(out)["job_id"]

        await asyncio.gather(bk(), asyncio.sleep(0.03))
        seen[tag] = te.take_created_job_ids()

    await asyncio.gather(turn("A"), turn("B"))
    assert seen["A"] == (seen["A:id"],)
    assert seen["B"] == (seen["B:id"],)


# ── The step rules (pure) ────────────────────────────────────────────────

def _steps(n=3):
    return [{"id": f"s{i}", "type": f"step_{i}", "label": f"L{i}", "status": "pending"} for i in range(n)]


def test_step_rules_open_advance_finish_and_shared_windows():
    from app.agent.job_steps import advance_steps, counts, finish_all_steps, open_first_step
    t0 = datetime(2026, 8, 19, 3, 1, 33)
    st = open_first_step(_steps(), t0)
    assert st[0]["status"] == "running" and st[0]["started_at"].startswith("2026-08-19T03:01:33")
    # one tick closes two steps: 0 gets its real window, 1 shares it, 2 opens
    t1 = t0 + timedelta(seconds=5)
    advance_steps(st, 1, t1)
    assert [s["status"] for s in st] == ["done", "done", "running"]
    assert st[0]["duration_ms"] == 5000 and st[1]["duration_ms"] == 5000
    assert st[1]["started_at"] == st[0]["started_at"]
    assert st[2]["started_at"].startswith("2026-08-19T03:01:38")
    assert counts(st) == (2, 3)
    # the finalizer closes the rest at delivery
    t2 = t1 + timedelta(seconds=7)
    finish_all_steps(st, t2)
    assert [s["status"] for s in st] == ["done"] * 3
    assert st[2]["duration_ms"] == 7000 and st[2]["durationMs"] == 7000
    # idempotent — a second close changes nothing
    snap = json.dumps(st)
    finish_all_steps(st, t2 + timedelta(seconds=30))
    advance_steps(st, 0, t2 + timedelta(seconds=30))
    assert json.dumps(st) == snap


def test_step_rules_never_move_backwards_and_tolerate_junk():
    from app.agent.job_steps import advance_steps, open_first_step, parse_steps
    t0 = datetime(2026, 8, 19, 3, 0, 0)
    st = open_first_step(_steps(), t0)
    advance_steps(st, 1, t0 + timedelta(seconds=1))
    # a later, LOWER tick must not reopen a done step
    advance_steps(st, 0, t0 + timedelta(seconds=2))
    assert [s["status"] for s in st] == ["done", "done", "running"]
    assert parse_steps("not json") == [] and parse_steps(None) == []
    assert advance_steps([], 0, t0) == []


def test_untouched_job_finishes_with_one_shared_window():
    """The model never called update_job: every step closes at delivery,
    sharing the [create, delivery] window (never 0ms, never unknown)."""
    from app.agent.job_steps import finish_all_steps, open_first_step
    t0 = datetime(2026, 8, 19, 3, 0, 0)
    st = open_first_step(_steps(2), t0)
    finish_all_steps(st, t0 + timedelta(seconds=9))
    assert all(s["status"] == "done" and s["duration_ms"] == 9000 for s in st)


# ── The reconciler ───────────────────────────────────────────────────────

async def _seed_job(user_id: str, *, conversation_id, created_at, asst_message_id=None,
                    status="running", handed_off=False, source_kind="manual") -> str:
    from app.db import async_session_maker
    from app.db.models import BuildJob
    from app.agent.job_steps import dump_steps, open_first_step
    job_id = str(uuid.uuid4())
    cfg = {"job_type": "search"}
    if asst_message_id:
        cfg["asst_message_id"] = asst_message_id
    if handed_off:
        cfg["handed_off"] = True
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=user_id, title="Find X", prompt="find x",
            job_type="agent_task", status=status, layer=0,
            steps_json=dump_steps(open_first_step(_steps(), created_at)),
            source_kind=source_kind, conversation_id=conversation_id,
            config_json=cfg, created_at=created_at,
        ))
        await db.commit()
    return job_id


async def _seed_conversation(user_id: str) -> str:
    from app.db import async_session_maker
    from app.db.models import Conversation
    cid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Conversation(id=cid, user_id=user_id, title="t"))
        await db.commit()
    return cid


async def _seed_assistant_message(conversation_id: str, created_at, msg_id=None, content="Kling wins") -> str:
    from app.db import async_session_maker
    from app.db.models import Message
    mid = msg_id or str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Message(id=mid, conversation_id=conversation_id, role="assistant",
                       content=content, created_at=created_at))
        await db.commit()
    return mid


@pytest.mark.asyncio
async def test_reconciler_completes_a_delivered_job_and_leaves_the_rest(monkeypatch, test_user_id):
    """The watchdog rule: running + its answer message exists ⇒ completed
    at the message's time, with the frame and the terminal push. Not
    delivered / handed off / dashboard / too young ⇒ untouched."""
    from app.agent import job_reconciler as jr
    from app.services import agent_notify_client as anc
    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)
    frames: list = []

    async def _capture(user_id, event, **kw):
        frames.append(dict(event))
        return 0

    import app.api.ws_chat as ws
    monkeypatch.setattr(ws, "broadcast_to_user", _capture)

    now = datetime.utcnow()
    conv = await _seed_conversation(test_user_id)
    created = now - timedelta(minutes=5)
    delivered_at = created + timedelta(seconds=12)
    mid = str(uuid.uuid4())
    # 1. delivered (precise proof: its own answer id)
    delivered = await _seed_job(test_user_id, conversation_id=conv, created_at=created, asst_message_id=mid)
    await _seed_assistant_message(conv, delivered_at, msg_id=mid, content="**Kling** wins")
    # 2. not delivered (answer id points at no row)
    pending = await _seed_job(test_user_id, conversation_id=conv, created_at=created, asst_message_id=str(uuid.uuid4()))
    # 3. handed off to a mission
    handed = await _seed_job(test_user_id, conversation_id=conv, created_at=created, asst_message_id=mid, handed_off=True)
    # 4. dashboard job — no conversation
    dash = await _seed_job(test_user_id, conversation_id=None, created_at=created)
    # 5. too young — the finalizer's window
    young = await _seed_job(test_user_id, conversation_id=conv, created_at=now - timedelta(seconds=5), asst_message_id=mid)
    # 6. legacy row (no answer id) old enough, with a later assistant message
    legacy_conv = await _seed_conversation(test_user_id)
    legacy = await _seed_job(test_user_id, conversation_id=legacy_conv, created_at=now - timedelta(minutes=10))
    await _seed_assistant_message(legacy_conv, now - timedelta(minutes=9))
    # 7. legacy row whose only assistant message predates it (a previous turn)
    legacy2_conv = await _seed_conversation(test_user_id)
    await _seed_assistant_message(legacy2_conv, now - timedelta(minutes=20))
    legacy2 = await _seed_job(test_user_id, conversation_id=legacy2_conv, created_at=now - timedelta(minutes=10))

    n = await jr.reconcile_delivered_turn_jobs(now)
    assert n == 2, n

    d = await _job_row(delivered)
    assert d.status == "completed"
    assert d.completed_at == delivered_at.replace(microsecond=(delivered_at.microsecond))
    assert d.summary_message_id == mid
    assert d.config_json.get("reconciled_reason") == "answer_delivered"
    steps = json.loads(d.steps_json)
    assert all(s["status"] == "done" and s["duration_ms"] > 0 for s in steps)
    assert (await _job_row(legacy)).status == "completed"
    for jid in (pending, handed, dash, young, legacy2):
        assert (await _job_row(jid)).status == "running", jid

    done = [f for f in frames if f.get("job_id") == delivered and f.get("status") == "completed"]
    assert done and done[0]["completed_steps"] == 3 and done[0]["message_id"] == mid
    pushes = await _outbox_rows(delivered)
    assert [p.event_kind for p in pushes] == ["mission_completed"]
    assert pushes[0].data_json["preview"].startswith("Kling wins")
    assert pushes[0].data_json["phase"] == "completed"
    # job_events heartbeat/timeline row
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import JobEvent
    async with async_session_maker() as db:
        evs = (await db.execute(select(JobEvent).where(JobEvent.job_id == delivered))).scalars().all()
    assert any(e.status == "completed" for e in evs)

    # idempotent
    assert await jr.reconcile_delivered_turn_jobs(now) == 0


@pytest.mark.asyncio
async def test_reaper_completes_a_delivered_stall_instead_of_cancelling_it(monkeypatch, test_user_id):
    """The 30-minute reaper used to call a finalizer-less job 'stopped
    before it finished'. It now reconciles first."""
    from app.agent import job_reaper as reaper
    from app.services import agent_notify_client as anc
    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)

    async def _noop(*a, **k):
        return 0

    import app.api.ws_chat as ws
    monkeypatch.setattr(ws, "broadcast_to_user", _noop)
    now = datetime.utcnow()
    conv = await _seed_conversation(test_user_id)
    created = now - timedelta(minutes=45)
    mid = str(uuid.uuid4())
    delivered = await _seed_job(test_user_id, conversation_id=conv, created_at=created, asst_message_id=mid)
    await _seed_assistant_message(conv, created + timedelta(seconds=20), msg_id=mid)
    truly_stalled = await _seed_job(test_user_id, conversation_id=conv, created_at=created, asst_message_id=str(uuid.uuid4()))

    await reaper.sweep_stalled_jobs(now)
    assert (await _job_row(delivered)).status == "completed"
    assert (await _job_row(truly_stalled)).status == "cancelled"


# ── The interim beacon can only move forward ─────────────────────────────

@pytest.mark.asyncio
async def test_beacon_ignores_a_stale_provisional_step(monkeypatch):
    from app.agent.turn_progress import TurnProgressEmitter
    sent: list = []

    async def fake_notify(**kw):
        sent.append(kw)
        return "row"

    import app.services.agent_notify_client as anc
    monkeypatch.setattr(anc, "notify", fake_notify)
    em = TurnProgressEmitter(mission_id="chatturn:x", mission_title="T", chat_id="c1")
    em.min_interval_s = 0
    await em.on_step_change({"kind": "step_change", "job_id": "J", "step_index": 2,
                             "step_name": "Write", "steps_total": 3, "job_type": "compare"})
    # the batch's tool_start arrives with the snapshot taken BEFORE update_job
    await em.on_tool_start("web_fetch", meta={"job_id": "J", "step_index": 1,
                                              "step_name": "Compare", "steps_total": 3})
    assert em.step_index == 2 and em.step_name == "Write"
    assert sent[-1]["data"]["steps_done"] == 2 and sent[-1]["data"]["step_name"] == "Write"
    # a NEW job resets
    await em.on_tool_start("web_search", meta={"job_id": "K", "step_index": 0,
                                               "step_name": "Search", "steps_total": 2})
    assert em.step_index == 0 and em.job_id == "K"


def test_legacy_rows_without_stamps_get_the_jobs_own_window_not_zero():
    """Rows written before this image carry no step stamps. Closing them at
    delivery must give the running step [job.created_at, delivered] — not a
    zero-width window."""
    from app.agent.job_steps import advance_steps, finish_all_steps
    created = datetime(2026, 8, 19, 3, 1, 33)
    delivered = created + timedelta(seconds=12)
    legacy = [{"id": "a", "type": "step_0", "label": "Research", "status": "done"},
              {"id": "b", "type": "step_1", "label": "Compare", "status": "done"},
              {"id": "c", "type": "step_2", "label": "Write", "status": "running"}]
    finish_all_steps(legacy, delivered, fallback_start=created)
    assert legacy[2]["duration_ms"] == 12000
    assert "duration_ms" not in legacy[0]          # done steps are left as they were
    legacy2 = [{"id": "a", "type": "step_0", "label": "Research", "status": "pending"},
               {"id": "b", "type": "step_1", "label": "Write", "status": "pending"}]
    advance_steps(legacy2, 0, created + timedelta(seconds=4), fallback_start=created)
    assert legacy2[0]["duration_ms"] == 4000 and legacy2[1]["status"] == "running"
