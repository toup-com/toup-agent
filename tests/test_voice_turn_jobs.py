"""Round 13 — a voice turn that does work must leave a JOB behind.

Before this, a spoken "find the strongest image-generation model" produced
four searches, two page reads, a correct answer, and — in the thread — an
action summary: "Searched the web · 4× · 3.5s". The same sentence typed into
chat produced a card titled "Find the strongest image-generation model" with
its steps and a completion. Same agent, same tools, same thread.

The reason was never the persistence: Round 12 already carries `tool_events`
through the relay. It was that the card is minted by the `create_job` TOOL and
voice does not have that tool — removed on purpose
(`prompt_profile.VOICE_DISABLED_TOOLS`), and it stays removed. So the RUNNER
opens the card now (`app/agent/voice_jobs.py`).

These tests drive the REAL runner through a scripted voice turn and assert on
what the surfaces actually received: the BuildJob row, its title, its steps,
the `job_update` frames, the phone outbox, the step attribution stamped on the
persisted tool records, and the row in the thread. They also pin the two
things that must NOT change — a chat turn is untouched, and a voice turn that
only chats mints nothing.

Needs RUN_MODE=agent: build_jobs / job_events / agent_notify_outbox /
messages / conversations are AGENT_ONLY. See COVERAGE_DEBT.txt.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest


# ── Harness ─────────────────────────────────────────────────────────────

class _ScriptedLLM:
    """A real voice research turn: two searches, then a fetch, then the
    spoken answer. No create_job anywhere — voice cannot call it."""

    def __init__(self, rounds=None):
        self.calls = 0
        self._rounds = rounds if rounds is not None else [
            [("s1", "web_search", {"query": "strongest image model"}),
             ("s2", "web_search", {"query": "image model benchmarks"})],
            [("f1", "web_fetch", {"url": "https://example.org/bench"})],
        ]

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent as E
        self.calls += 1
        if self.calls > len(self._rounds):
            yield E(type="text", text="Right now it's Imagen 4 Ultra.")
            yield E(type="message_end", stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 2})
            return
        batch = self._rounds[self.calls - 1]
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
                    hashed_password="x" * 60, name="R13"))
        await db.commit()
    return user_id


async def _run_turn(
    monkeypatch, tmp_path, *,
    said="hey can you please find the strongest image-generation model",
    channel="voice", rounds=None, enabled=True, cancel_after=None,
):
    """One end-to-end turn. Returns a dict of everything the surfaces saw."""
    import app.agent.agent_runner as ar
    import app.agent.voice_jobs as vj
    from app.agent.tool_executor import ToolExecutor
    from app.services import agent_notify_client as anc

    async def _no_history(self, db, session_id, max_messages: int = 50, client_tz=None):
        return []

    async def _fixed_prompt(self, *a, **kw):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", _no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", _fixed_prompt)
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)
    monkeypatch.setattr(ar.settings, "voice_turn_jobs", enabled, raising=False)
    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)

    # The card's writes are deliberately OFF the turn's critical path. Keep
    # the tasks so the assertions can see what they wrote instead of racing
    # them — the point of the design is that the turn does not await these,
    # and a test that could not tell the difference would prove nothing.
    bg: list = []

    def _keep(coro, **kw):
        t = asyncio.get_event_loop().create_task(coro)
        bg.append(t)
        return t

    monkeypatch.setattr(ar, "_spawn_bg", _keep)
    monkeypatch.setattr(vj, "_spawn_bg", _keep)

    frames: list = []

    async def _capture(user_id, event, **kw):
        frames.append(dict(event))
        return 0

    import app.api.ws_chat as ws
    monkeypatch.setattr(ws, "broadcast_to_user", _capture)

    # The sink api_v1's SSE route hands the runner on a voice turn.
    tool_events: list = []

    async def _on_tool_event(ev):
        tool_events.append(dict(ev))

    tools = ToolExecutor(workspace=str(tmp_path))

    async def fake_execute(name, inp):
        if name == "web_search":
            return f"1. Result for {inp['query']}\n   https://example.org/page\n"
        if name == "web_fetch":
            return "# Bench\nImagen 4 Ultra leads."
        if name == "memory_search":
            return "no memories"
        return "?"

    monkeypatch.setattr(tools, "execute", fake_execute)

    runner = ar.AgentRunner(llm_service=_ScriptedLLM(rounds), tool_executor=tools)  # type: ignore[arg-type]
    user_id = await _make_user()
    session_id = str(uuid.uuid4())
    kwargs = dict(
        user_message=said, user_id=user_id, session_id=session_id,
        channel=channel, model_override="gpt-5.5-mini",
        on_tool_event=_on_tool_event,
        # Exactly what api_v1's agent-turn passes for voice: the relay
        # persists the spoken pair itself.
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True,
    )
    if cancel_after is not None:
        task = asyncio.get_event_loop().create_task(runner.run(**kwargs))
        await asyncio.sleep(cancel_after)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        resp = None
    else:
        resp = await runner.run(**kwargs)
    if bg:
        await asyncio.gather(*bg, return_exceptions=True)
        # The close is scheduled by the tasks above (seal fires inside the
        # turn, its coroutine lands in `bg`); drain anything they added.
        if len(bg) > 0:
            await asyncio.gather(*bg, return_exceptions=True)
    return {
        "resp": resp, "frames": frames, "user_id": user_id,
        "session_id": session_id, "tools": tools, "tool_events": tool_events,
    }


async def _jobs_for(user_id: str) -> list:
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import BuildJob
    async with async_session_maker() as db:
        return list((await db.execute(
            select(BuildJob).where(BuildJob.user_id == user_id)
        )).scalars().all())


async def _outbox_kinds(job_id: str) -> list:
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import AgentNotifyOutbox
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AgentNotifyOutbox).order_by(AgentNotifyOutbox.created_at)
        )).scalars().all()
    return [r.event_kind for r in rows if (r.data_json or {}).get("job_id") == job_id]


# ── THE regression ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_voice_research_turn_leaves_a_real_job(monkeypatch, tmp_path):
    """The whole point. A spoken request that searches and reads must end as
    a completed BuildJob titled from what the user asked for — not as a bare
    row of action pills."""
    out = await _run_turn(monkeypatch, tmp_path)
    jobs = await _jobs_for(out["user_id"])
    assert len(jobs) == 1, "a voice work turn must open exactly one job"
    job = jobs[0]

    # Titled from the request, with the spoken scaffolding off.
    assert job.title == "Find the strongest image-generation model"
    # An ordinary agent-authored job — the same row Mission Control, the
    # jobs API, the reaper and the reconciler already understand.
    assert job.job_type == "agent_task"
    assert job.source_kind == "manual"
    # The RESOLVED conversation, which is what the runner works in — a
    # session id the tenant has never seen is minted fresh by
    # _get_or_create_session, and the card belongs to the row that exists.
    assert job.conversation_id == out["resp"].session_id
    assert (job.config_json or {}).get("voice_turn") is True
    assert (job.config_json or {}).get("job_type") == "search"
    assert job.status == "completed"
    assert job.completed_at is not None


@pytest.mark.asyncio
async def test_the_steps_are_the_work_the_turn_actually_did(monkeypatch, tmp_path):
    """Steps come from the tools, collapsed by kind — four searches are one
    step, not four — and the answer round has a step of its own."""
    out = await _run_turn(monkeypatch, tmp_path)
    job = (await _jobs_for(out["user_id"]))[0]
    steps = json.loads(job.steps_json)
    assert [s["label"] for s in steps] == [
        "Search the web", "Read the sources", "Put the answer together",
    ]
    assert all(s["status"] == "done" for s in steps), steps
    # Round 8 timing rules, from the same module `update_job` writes through.
    assert all(s.get("durationMs") is not None for s in steps), steps
    # And a REAL window on the step the turn actually spent time in. The
    # search step opens when the card does; closing it with `now` on both
    # edges is the fabricated 0ms Round 8 exists to remove.
    assert steps[0]["started_at"] < steps[0]["completed_at"], steps[0]
    assert steps[0]["durationMs"] > 0, steps[0]


@pytest.mark.asyncio
async def test_every_action_names_the_step_it_served(monkeypatch, tmp_path):
    """The runner stamps `job_id`/`step_index`/`step_name` on every tool
    event; the clients bucket actions under steps from exactly those keys.
    Without them the run view puts everything under step 1 and prints "No
    actions recorded" for the rest."""
    out = await _run_turn(monkeypatch, tmp_path)
    job = (await _jobs_for(out["user_id"]))[0]
    ends = [e for e in out["tool_events"] if e.get("phase") == "end"]
    assert len(ends) == 3, [e.get("name") for e in ends]
    assert {e["job_id"] for e in ends} == {job.id}
    by_tool = {e["name"]: e for e in ends}
    assert by_tool["web_search"]["step_name"] == "Search the web"
    assert by_tool["web_fetch"]["step_name"] == "Read the sources"
    assert by_tool["web_search"]["step_index"] == 0
    assert by_tool["web_fetch"]["step_index"] == 1
    assert by_tool["web_fetch"]["steps_total"] == 3


@pytest.mark.asyncio
async def test_a_batched_round_is_attributed_to_its_own_step(monkeypatch, tmp_path):
    """The runner pre-executes ≥2 parallel-safe calls concurrently and stamps
    every frame in that batch from ONE `event_fields()` snapshot taken before
    it runs. So the round's step has to be set at plan time — a step set in
    the per-call loop is set after those frames have already gone out, and
    two fetches would land under "Search the web"."""
    out = await _run_turn(
        monkeypatch, tmp_path,
        rounds=[
            [("s1", "web_search", {"query": "a"}), ("s2", "web_search", {"query": "b"})],
            [("f1", "web_fetch", {"url": "https://a.example/1"}),
             ("f2", "web_fetch", {"url": "https://b.example/2"})],
        ],
    )
    ends = [e for e in out["tool_events"] if e.get("phase") == "end"]
    fetches = [e for e in ends if e["name"] == "web_fetch"]
    assert len(fetches) == 2
    assert {e["step_name"] for e in fetches} == {"Read the sources"}
    assert {e["step_index"] for e in fetches} == {1}


@pytest.mark.asyncio
async def test_the_attribution_survives_the_voice_wire(monkeypatch, tmp_path):
    """Three hops, and it used to be dropped at the first two: the runner's
    payload → api_v1's SSE frame → the relay's persisted record. A chat turn
    writes these keys straight onto its tool record, so the voice record has
    to arrive carrying them or the same action renders under no step."""
    from app.api.api_v1 import _vs_step_fields
    from app.api.ws_realtime import _InnerToolRelay

    class _Sock:
        def __init__(self):
            self.frames = []

        async def send_json(self, frame):
            self.frames.append(frame)

    out = await _run_turn(monkeypatch, tmp_path)
    job = (await _jobs_for(out["user_id"]))[0]
    sink: list = []
    relay = _InnerToolRelay(_Sock(), "outer-1", sink=sink)
    for ev in out["tool_events"]:
        # What api_v1 puts on the wire for this payload.
        attr = _vs_step_fields(ev)
        assert attr, "the SSE frame must carry the attribution"
        if ev.get("phase") == "start":
            await relay.on_event({"type": "tool.start", "call_id": ev["call_id"],
                                  "name": ev["name"], "args": {}, **attr})
        else:
            await relay.on_event({"type": "tool.end", "call_id": ev["call_id"],
                                  "name": ev["name"], "ok": True,
                                  "elapsed_ms": 1200, **attr})
    assert sink, "the run must be recorded"
    assert all(r.get("job_id") == job.id for r in sink), sink
    assert [r["step_name"] for r in sink] == [
        "Search the web", "Search the web", "Read the sources",
    ]


@pytest.mark.asyncio
async def test_the_persisted_record_survives_the_readers_filter(monkeypatch, tmp_path):
    """`sessions._clean_tool_events` is an allow-list — a key it does not
    name is dropped on the way into the database. Attribution that the wire
    carries and the writer discards is attribution the thread never sees."""
    from app.api.sessions import _clean_tool_events
    out = await _run_turn(monkeypatch, tmp_path)
    job = (await _jobs_for(out["user_id"]))[0]
    end = next(e for e in out["tool_events"] if e.get("phase") == "end")
    rec = {"tool": end["name"], "started_at_ms": 1, "completed_at_ms": 2,
           "job_id": end["job_id"], "step_index": end["step_index"],
           "step_name": end["step_name"], "steps_total": end["steps_total"]}
    kept = _clean_tool_events([rec])[0]
    assert kept["job_id"] == job.id
    assert kept["step_name"] == end["step_name"]
    assert kept["steps_total"] == end["steps_total"]


@pytest.mark.asyncio
async def test_the_card_reaches_the_in_app_surface(monkeypatch, tmp_path):
    """`job_update` is the frame the web card is driven by and the app
    refetches on — the same frame `create_job`/`update_job` emit."""
    out = await _run_turn(monkeypatch, tmp_path)
    job = (await _jobs_for(out["user_id"]))[0]
    searches = [f for f in out["frames"] if f.get("type") == "job_update"]
    assert searches, "the card must reach the in-app surface"
    assert {f["job_id"] for f in searches} == {job.id}
    assert searches[0]["name"] == job.title
    assert searches[0]["total_steps"] >= 2
    assert searches[-1]["status"] == "completed"
    assert searches[-1]["completed_steps"] == searches[-1]["total_steps"]


@pytest.mark.asyncio
async def test_the_phone_card_opens_and_ends(monkeypatch, tmp_path):
    """mission_started at the first tool, a terminal completion at the end —
    the same lane, kinds and dedup suffixes a chat job uses. `progress` is
    the Live-Activity-only kind; anything outside KNOWN_NOTIFY_KINDS would
    422 at the notify route."""
    from app.db.models import KNOWN_NOTIFY_KINDS
    out = await _run_turn(monkeypatch, tmp_path)
    job = (await _jobs_for(out["user_id"]))[0]
    kinds = await _outbox_kinds(job.id)
    assert kinds, "the phone must be told about the card"
    assert kinds[0] == "mission_started"
    assert "progress" in kinds, (
        "the card must advance as the turn discovers its next step, not jump "
        f"from Starting… to Done — got {kinds}"
    )
    assert kinds[-1] == "mission_completed"
    # A kind outside the closed enum 422s at the notify route, so the card
    # would silently never reach the phone.
    assert set(kinds) <= KNOWN_NOTIFY_KINDS


@pytest.mark.asyncio
async def test_the_card_has_a_row_in_the_thread(monkeypatch, tmp_path):
    """A voice turn touches neither ws_chat writer, so without its own row
    the card would exist everywhere except the thread the user was told to
    look at. Same `job-<id>` key the ws_chat writers use, so a connected
    chat socket cannot produce a second one."""
    from app.db import async_session_maker
    from app.db.models import Message
    out = await _run_turn(monkeypatch, tmp_path)
    job = (await _jobs_for(out["user_id"]))[0]
    async with async_session_maker() as db:
        row = await db.get(Message, f"job-{job.id}")
    assert row is not None
    assert row.role == "job"
    assert row.conversation_id == out["resp"].session_id
    assert json.loads(row.content) == {"job_id": job.id, "job_name": job.title}


@pytest.mark.asyncio
async def test_the_reloaded_card_shows_the_steps_it_finished(monkeypatch, tmp_path):
    """The history serializer counted `status == "completed"` on STEPS — a
    value nothing has ever written (`job_steps`, `_tool_create_job`, `apps.py`
    and the app-builder skill all write `done`; `completed` is the JOB's
    status). Every card ever loaded from history therefore re-rendered a
    finished job as "0/N steps". Chat's cards had it too; the voice card this
    round adds would have shipped straight into it."""
    from app.api.sessions import _message_to_response
    from app.db import async_session_maker
    from app.db.models import BuildJob, Message

    out = await _run_turn(monkeypatch, tmp_path)
    job = (await _jobs_for(out["user_id"]))[0]
    async with async_session_maker() as db:
        row = await db.get(Message, f"job-{job.id}")
        bj = await db.get(BuildJob, job.id)
        resp = _message_to_response(row, {job.id: bj})
    assert resp.job_total_steps == 3
    assert resp.job_completed_steps == 3, (
        "a completed job must not re-render as 0/3 the moment the thread "
        "is reloaded"
    )


# ── What must NOT happen ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_voice_turn_that_only_chats_mints_nothing(monkeypatch, tmp_path):
    """Noise on this surface is the failure the voice path already had once
    (2026-08-01: a card per re-ask, no answers). A turn that reads memory and
    answers is a conversation, not a job."""
    out = await _run_turn(
        monkeypatch, tmp_path, said="what did I say about the offsite",
        rounds=[[("m1", "memory_search", {"query": "offsite"})]],
    )
    assert await _jobs_for(out["user_id"]) == []
    assert [f for f in out["frames"] if f.get("type") == "job_update"] == []


@pytest.mark.asyncio
async def test_a_chat_turn_is_untouched(monkeypatch, tmp_path):
    """The runner opens cards for VOICE only — chat's are the model's, and a
    second one beside them would be a duplicate."""
    out = await _run_turn(monkeypatch, tmp_path, channel="web")
    assert await _jobs_for(out["user_id"]) == []


@pytest.mark.asyncio
async def test_the_flag_turns_it_off(monkeypatch, tmp_path):
    out = await _run_turn(monkeypatch, tmp_path, enabled=False)
    assert await _jobs_for(out["user_id"]) == []
    assert out["resp"].text  # the turn itself is unaffected


@pytest.mark.asyncio
async def test_the_turn_does_not_wait_for_the_card(monkeypatch, tmp_path):
    """The constraint: nothing here may delay TTS. Proven by construction —
    the turn returns with every card write still unstarted, because the test
    holds the spawned coroutines and only drains them afterwards."""
    import app.agent.agent_runner as ar
    import app.agent.voice_jobs as vj
    from app.agent.tool_executor import ToolExecutor
    from app.services import agent_notify_client as anc

    async def _no_history(self, db, session_id, max_messages: int = 50, client_tz=None):
        return []

    async def _fixed_prompt(self, *a, **kw):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", _no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", _fixed_prompt)
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)
    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)

    held: list = []
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kw: held.append(coro))
    monkeypatch.setattr(vj, "_spawn_bg", lambda coro, **kw: held.append(coro))

    tools = ToolExecutor(workspace=str(tmp_path))

    async def fake_execute(name, inp):
        return "1. x\n   https://example.org/p\n"

    monkeypatch.setattr(tools, "execute", fake_execute)
    runner = ar.AgentRunner(llm_service=_ScriptedLLM(), tool_executor=tools)  # type: ignore[arg-type]
    user_id = await _make_user()
    resp = await runner.run(
        user_message="find the strongest image-generation model",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="voice",
        model_override="gpt-5.5-mini", save_user_message=False,
        save_assistant_message=False, disable_post_processing=True,
    )
    assert resp.text, "the answer is ready"
    assert await _jobs_for(user_id) == [], (
        "the turn returned before ANY card write ran — that is the design"
    )
    assert held, "and the writes were scheduled, not skipped"
    for coro in held:
        coro.close()


@pytest.mark.asyncio
async def test_a_hung_up_turn_still_closes_its_card(monkeypatch, tmp_path):
    """Cancellation is how a voice call normally ends — the SSE generator
    cancels the turn task the moment the caller disconnects. A card left
    running would sit frozen until the 30-minute reaper closed it with a
    false "Didn't finish" over work the agent already spoke aloud."""
    import app.agent.voice_jobs as vj

    job = vj.VoiceTurnJob(
        user_id=await _make_user(), conversation_id=str(uuid.uuid4()),
        request_text="find the strongest image-generation model",
    )
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    # The sweep runs from run()'s finally, synchronously, on a cancelled task.
    vj.sweep_current_voice_job()
    assert vj.current_voice_job() is None
    await asyncio.sleep(0.2)
    rows = await _jobs_for(job._user_id)  # noqa: SLF001 — the row is the point
    assert len(rows) == 1
    assert rows[0].status == "completed", (
        "an interrupted voice turn closes COMPLETED, never failed — the "
        "caller hanging up is not the work failing"
    )
