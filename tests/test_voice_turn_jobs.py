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
import time
import json
import uuid

import pytest


# ── Harness ─────────────────────────────────────────────────────────────

class _ScriptedLLM:
    """A real voice research turn: two searches, then a fetch, then the
    spoken answer. No create_job anywhere — voice cannot call it."""

    def __init__(self, rounds=None):
        self.calls = 0
        self.tool_choices: list = []
        self.messages: list = []
        self.efforts: list = []        # `reasoning_effort` per call (T5)
        self.systems: list = []
        self.tools_sent: list = []     # was a tools array on the wire (T7)
        self._rounds = rounds if rounds is not None else [
            [("s1", "web_search", {"query": "strongest image model"}),
             ("s2", "web_search", {"query": "image model benchmarks"})],
            [("f1", "web_fetch", {"url": "https://example.org/bench"})],
        ]

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent as E
        self.calls += 1
        self.tool_choices.append(kwargs.get("tool_choice"))
        self.messages.append(list(kwargs.get("messages") or []))
        self.efforts.append(kwargs.get("reasoning_effort"))
        self.systems.append(kwargs.get("system"))
        self.tools_sent.append(kwargs.get("tools") is not None)
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
    delegation_id=None, display_request=None, require_web=False,
    extra_tools=None, extra_kwargs=None, llm_cls=None, speculative=False,
    settings_over=None,
):
    """One end-to-end turn. Returns a dict of everything the surfaces saw.

    ``extra_tools`` maps a tool name to ``async fn(tools, inp)``, run as that
    tool's execution (the real bookkeeping a test wants inside the turn)."""
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
    # Voice round 2026-10-06 (D-T3): the speculative first search is NEW
    # behaviour, pinned OFF for every turn this harness ran before it existed
    # (the D-P3 forced-web tests assert the forced round it releases); the
    # tests written for it opt in with `speculative=True`.
    monkeypatch.setattr(ar.settings, "voice_speculative_search_enabled", speculative,
                        raising=False)
    # Voice round 2026-10-07 (T5/T6/T7): new voice-delegation behaviour, pinned
    # OFF for every turn written before it (the speculative precedent above);
    # the tests written for it opt in through `settings_over`.
    monkeypatch.setattr(ar.settings, "voice_research_max_searches", 0, raising=False)
    monkeypatch.setattr(ar.settings, "voice_delegation_reasoning_effort", "", raising=False)
    monkeypatch.setattr(ar.settings, "voice_delegation_string_tool_choice", False,
                        raising=False)
    for _k, _v in (settings_over or {}).items():
        monkeypatch.setattr(ar.settings, _k, _v, raising=False)
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
        # A real tool takes time. Step windows are stamped on the turn's own
        # clock now, so an instant fake would measure a true 0 ms window.
        await asyncio.sleep(0.005)
        if extra_tools and name in extra_tools:
            return await extra_tools[name](tools, inp)
        if name == "web_search":
            return f"1. Result for {inp['query']}\n   https://example.org/page\n"
        if name == "web_fetch":
            return "# Bench\nImagen 4 Ultra leads."
        if name == "memory_search":
            return "no memories"
        if name == "create_job":
            return '{"job_id": "model-job-1", "title": "Compare", "steps": 2}'
        return "?"

    monkeypatch.setattr(tools, "execute", fake_execute)

    llm = (llm_cls or _ScriptedLLM)(rounds)
    runner = ar.AgentRunner(llm_service=llm, tool_executor=tools)  # type: ignore[arg-type]
    # What api_v1's SSE route hands the runner as `job_frame_sink`.
    sse: list = []
    sse_t: list = []        # perf_counter of each `sse` frame, same order
    step_events: list = []

    async def _on_step_event(ev):
        step_events.append(dict(ev))

    user_id = await _make_user()
    session_id = str(uuid.uuid4())
    kwargs = dict(
        user_message=said, user_id=user_id, session_id=session_id,
        channel=channel, model_override="gpt-5.5-mini",
        on_tool_event=_on_tool_event, on_step_event=_on_step_event,
        # Exactly what api_v1's agent-turn passes for voice: the relay
        # persists the spoken pair itself.
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True,
    )
    if delegation_id is not None:
        def _sink(f):
            sse.append(dict(f))
            sse_t.append(time.perf_counter())

        kwargs.update(voice_delegation_id=delegation_id,
                      display_request=display_request or said,
                      job_frame_sink=_sink)
    if require_web:
        kwargs["require_web"] = True
    if extra_kwargs:
        kwargs.update(extra_kwargs)
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
        "sse": sse, "sse_t": sse_t, "llm": llm, "step_events": step_events,
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
    # Round 8 timing rules, from the same module `update_job` writes through —
    # amended in round 33. A step reports a duration it MEASURED, or none.
    # Sharing the enclosing window between every step closed in it is what
    # printed a job's whole elapsed time on each of its three steps ("16.2s ·
    # 16.2s · 16.2s" on a founder's card), so a step the turn passed through
    # without ever running now says nothing rather than borrowing its
    # neighbour's number. At least one step measured a real window, and every
    # duration present is a positive one.
    measured = [s for s in steps if s.get("durationMs") is not None]
    assert measured, steps
    assert all(s["durationMs"] > 0 for s in measured), measured
    assert all(s.get("completed_at") for s in steps), steps
    # And a REAL window on the step the turn actually spent time in. The
    # search step opens when the card does; closing it with `now` on both
    # edges is the fabricated 0ms Round 8 exists to remove.
    assert steps[0]["started_at"] < steps[0]["completed_at"], steps[0]
    assert steps[0]["durationMs"] > 0, steps[0]


# ── A Live delegation declares its plan at t0 (voice round 2026-10-05) ──

_PLAN_EN = ["Search the web", "Read the sources", "Put the answer together"]


@pytest.mark.asyncio
async def test_a_delegation_declares_its_plan_before_any_tool_runs(monkeypatch, tmp_path):
    """D-P5. The card used to open at the FIRST WORK TOOL — after phase 1 and
    the first model round, i.e. seconds into the call, and only on the chat
    socket. A delegation's card now exists at t0 with the plan the backend
    declared, and the delegation stream carries it before the first tool."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-1",
        said="search for the strongest image-generation model",
    )
    sse = out["sse"]
    assert sse, "the card must ride the delegation stream"
    first = sse[0]
    assert first["type"] == "job.update"
    assert first["status"] == "running"
    assert first["delegation_id"] == "dlg-1"
    assert [s["label"] for s in first["steps"]] == _PLAN_EN
    assert {s["status"] for s in first["steps"]} == {"pending"}, (
        "a declared step only runs when a tool lands on it"
    )
    # Before the first tool: the runner emits nothing earlier than the plan.
    assert out["tool_events"][0]["job_id"] == first["job_id"]
    job = (await _jobs_for(out["user_id"]))[0]
    assert job.id == first["job_id"]
    assert (job.config_json or {}).get("plan_source") == "template"
    assert (job.config_json or {}).get("voice_delegation_id") == "dlg-1"
    assert job.status == "completed"
    steps = json.loads(job.steps_json)
    assert [s["status"] for s in steps] == ["done", "done", "done"]
    # The terminal frame closes the card on the stream too, with its sources.
    last = sse[-1]
    assert last["status"] == "completed"
    assert [s["status"] for s in last["steps"]] == ["done", "done", "done"]
    assert last["sources"] and last["sources"][0]["domain"] == "example.org"
    assert (job.config_json or {}).get("sources") == last["sources"]


@pytest.mark.asyncio
async def test_the_cards_first_frame_follows_its_row(monkeypatch, tmp_path):
    """W3. The phone creates the job from the first `job_update` and its poll
    reaps any job the account listing does not hold — so a frame sent before
    the row commits can be deleted and redrawn as the derived card. The row
    commits first, then the frame; the turn awaits that before phase 1, and
    the wait costs the insert, not more."""
    from app.agent.job_runner import JobRunner
    commits: list = []
    real = JobRunner.create_job

    entered: list = []

    async def create_job(self, *a, **kw):
        await asyncio.sleep(0.05)          # a loaded DB
        entered.append(time.perf_counter())
        job = await real(self, *a, **kw)
        commits.append((kw.get("job_id"), time.perf_counter()))
        return job

    monkeypatch.setattr(JobRunner, "create_job", create_job)
    llm_calls: list = []
    real_stream = _ScriptedLLM.create_message_stream

    def stream(self, **kw):
        llm_calls.append(time.perf_counter())
        return real_stream(self, **kw)

    monkeypatch.setattr(_ScriptedLLM, "create_message_stream", stream)
    t0 = time.perf_counter()
    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-w3",
                          said="search for the strongest image-generation model")
    first = out["sse"][0]
    assert first["type"] == "job.update" and first["status"] == "running"
    jid, committed = next(c for c in commits if c[0] == first["job_id"])
    assert committed <= out["sse_t"][0], "the frame went out before its row"
    # Every frame the turn sent is about a job the account listing holds.
    assert {f["job_id"] for f in out["sse"] if f.get("type") == "job.update"} == {jid}
    assert committed <= llm_calls[0], "phase 1 / the first model round began before the row"
    # What the wait adds to phase 1 is the insert itself (the 50 ms above is
    # this test's own simulated load, outside the measurement).
    insert_s = committed - entered[0]
    after_s = out["sse_t"][0] - committed
    assert insert_s < 0.15, f"the insert took {insert_s * 1000:.0f} ms"
    assert after_s < 0.15, f"first frame {after_s * 1000:.0f} ms after the commit"
    print(f"W3 insert_ms={insert_s * 1000:.1f} first_frame_after_commit_ms={after_s * 1000:.1f}")


@pytest.mark.asyncio
async def test_no_running_frame_goes_out_before_the_row(monkeypatch):
    """W3, the card itself: steps that move before the insert commits send
    nothing; the first frame carries them when the row exists."""
    import app.agent.voice_jobs as vj
    held: list = []
    monkeypatch.setattr(vj, "_spawn_bg", lambda coro, **kw: held.append(coro))
    frames: list = []
    uid = await _make_user()
    tj = vj.TurnJob(user_id=uid, conversation_id=None,
                    request_text="search for the strongest image model",
                    delegation_id="dlg-w3u", frame_sink=frames.append)
    tj.declare(["Search the web", "Read the sources"])
    tj.plan([{"name": "web_search"}])
    assert frames == []
    await held.pop(0)                      # `_open`: the row commits
    assert [f["status"] for f in frames] == ["running"]
    assert [s["status"] for s in frames[0]["steps"]][0] == "running"
    for c in held:
        c.close()


@pytest.mark.asyncio
async def test_a_planned_step_the_turn_never_reached_is_skipped_never_done(
        monkeypatch, tmp_path):
    """The template plans [search, read, answer]; this turn only searched.
    `finish_all_steps` would paint "Read the sources ✓" over a turn that read
    nothing — the overclaim the cancelled close already refuses."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-2",
        said="search for the strongest image-generation model",
        rounds=[[("s1", "web_search", {"query": "a"})]],
    )
    job = (await _jobs_for(out["user_id"]))[0]
    steps = json.loads(job.steps_json)
    assert [(s["label"], s["status"]) for s in steps] == [
        ("Search the web", "done"), ("Read the sources", "skipped"),
        ("Put the answer together", "done"),
    ]
    assert not steps[1].get("started_at")
    assert out["sse"][-1]["steps"][1]["status"] == "skipped"


@pytest.mark.asyncio
async def test_an_unplanned_tool_kind_is_inserted_before_the_answer(monkeypatch, tmp_path):
    """Honest growth: the plan did not foresee a page read on a generic
    request, so the generic step takes the kind it turned out to be, and a
    second unforeseen kind is inserted — never hidden, never after the
    answer."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-3",
        said="please make my morning plan a bit nicer and tidy up the notes for tomorrow",
        rounds=[[("e1", "exec", {"command": "ls"})],
                [("w1", "write_file", {"path": "a", "content": "b"})]],
    )
    first = out["sse"][0]
    assert [s["label"] for s in first["steps"]] == ["Work on it", "Put the answer together"]
    job = (await _jobs_for(out["user_id"]))[0]
    steps = json.loads(job.steps_json)
    assert [s["label"] for s in steps] == [
        "Run the command", "Write the files", "Put the answer together",
    ]
    assert [s["status"] for s in steps] == ["done", "done", "done"]


@pytest.mark.asyncio
async def test_the_answer_step_runs_from_the_closing_round(monkeypatch, tmp_path):
    """The answer step is the round with no tools: it starts when that round's
    thinking starts, and the search step closes then — before the answer
    step's own window, never inside it."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-4",
        said="search for the strongest image-generation model",
        rounds=[[("s1", "web_search", {"query": "a"})]],
    )
    running = [f for f in out["sse"] if f["status"] == "running"]
    answering = [f for f in running if f["steps"][-1]["status"] == "running"]
    assert answering, [[s["status"] for s in f["steps"]] for f in running]
    steps = answering[0]["steps"]
    assert steps[0]["status"] == "done"
    assert steps[0]["completed_at"] <= steps[-1]["started_at"]


@pytest.mark.asyncio
async def test_a_delegation_that_only_chats_keeps_its_card(monkeypatch, tmp_path):
    """The other half of the old 'only chats mints nothing' rule: a DELEGATED
    request was put on a card the moment it was delegated, so withdrawing it
    would make a card vanish from the call. It closes with what did not run
    skipped."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-5",
        said="what did I say about the offsite",
        rounds=[[("m1", "memory_search", {"query": "offsite"})]],
    )
    jobs = await _jobs_for(out["user_id"])
    assert len(jobs) == 1
    steps = json.loads(jobs[0].steps_json)
    assert [s["status"] for s in steps][:-1] == ["skipped"] * (len(steps) - 1)
    assert steps[-1]["status"] == "done"


# ── An explicit search ask is answered from the web (D-P3) ──────────────

@pytest.mark.asyncio
async def test_require_web_restricts_the_first_round_to_the_web_tools(monkeypatch, tmp_path):
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-6", require_web=True,
        said="what is the best book for raising money",
    )
    tc = out["llm"].tool_choices[0]
    assert isinstance(tc, dict), tc
    names = {t["function"]["name"] for t in tc["allowed_tools"]["tools"]}
    assert "web_search" in names and names <= {
        "web_search", "web_fetch", "extension_search", "extension_research"}
    assert tc["allowed_tools"]["mode"] == "required"
    # Only the first round: later rounds are not restricted to the web.
    assert out["llm"].tool_choices[1] != tc


@pytest.mark.asyncio
async def test_an_explicit_search_that_answered_from_memory_searches_once_more(
        monkeypatch, tmp_path):
    """«خب سرچ کن برام» answered from the model's own head is the 2026-10-05
    call. If the turn would end without a web tool, ONE search round is
    re-entered, restricted to the web tools — and only one."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-7",
        said="خب سرچ کن برام بهترین کتاب برای جذب سرمایه", rounds=[],
    )
    llm = out["llm"]
    assert llm.calls == 2, "exactly one re-entered round"
    assert isinstance(llm.tool_choices[1], dict), llm.tool_choices
    assert "explicitly asked you to search" in json.dumps(llm.messages[1], ensure_ascii=False)
    assert out["resp"].text


@pytest.mark.asyncio
async def test_a_delegations_tools_are_gated_on_the_callers_words(
        monkeypatch, tmp_path, caplog):
    """The relay hands the runner a SCAFFOLDED wrapper (prior accepted
    results, English framing); the caller's words are `display_request`.
    Classifying the wrapper gated the first round's tools on text the caller
    never said."""
    import logging
    caplog.set_level(logging.INFO, logger="app.agent.agent_runner")
    await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-9",
        said=("Live-session context from earlier accepted delegations follows. "
              "Treat it as prior conversation.\nCurrent caller request: "
              "the caller wants a laptop for school, they mentioned a budget and "
              "a preference for long battery life and a light machine overall"),
        display_request="google the best laptop for students",
    )
    lines = [r.getMessage() for r in caplog.records if "[PERF] query_intent" in r.getMessage()]
    assert lines and "category=web" in lines[0], lines


@pytest.mark.parametrize("said,expected", [
    ("برام یه چیزی رو سرچ کن ببین", True),
    ("خب سرچ کن برام", True),
    ("سرچ کن بر", True),
    ("چرا سرچ نمی کنی", True),
    ("search for the best laptop", True),
    ("can you look it up", True),
    ("google the opening hours", True),
    ("find out who won", True),
    ("check online whether it is open", True),
    ("اینو از کجا سرچ کردی", False),
    ("where did you search", False),
    ("search my email for the invoice", False),
    ("ایمیل‌هام رو سرچ کن", False),
    ("what is the capital of France", False),
    ("", False),
])
def test_the_explicit_search_grammar(said, expected):
    from app.agent.voice_jobs import explicit_search_requested
    assert explicit_search_requested(said) is expected


def test_the_template_plan_is_chosen_from_the_request():
    from app.agent.voice_jobs import template_plan
    assert template_plan("search the web for flights", "full") == _PLAN_EN
    assert template_plan("anything", "full", require_web=True) == _PLAN_EN
    assert template_plan("what's new in my gmail", "full") == [
        "Check Gmail", "Put the answer together"]
    assert template_plan("refactor the parser", "code") == [
        "Read the files", "Work on it", "Put the answer together"]
    assert template_plan("make it nicer", "full") == [
        "Work on it", "Put the answer together"]
    assert template_plan("سرچ کن", "full", rtl=True) == [
        "جست‌وجو در وب", "خواندن منابع", "جمع‌بندی پاسخ"]
    for plan in (template_plan(t, c) for t, c in (
            ("a", "web"), ("b", "code"), ("c", "full"), ("d gmail", "full"))):
        assert 2 <= len(plan) <= 4
    # Voice round 2026-10-07 (card 2): a QUESTION the classifier reads as
    # `full` gets the search plan — «انجام کار» was its whole card.
    card2 = "اوکی. بعد به نظرت همین توی آیسآکادمیک چجوریه"
    assert template_plan(card2, "full", rtl=True) == [
        "جست‌وجو در وب", "خواندن منابع", "جمع‌بندی پاسخ"]
    assert template_plan("و استاد ال‌ال‌ام یوافتی کی بود", "full", rtl=True)[0] == "جست‌وجو در وب"
    assert template_plan("is it open on sundays?", "full") == _PLAN_EN
    assert template_plan("make it nicer.", "full") == [
        "Work on it", "Put the answer together"]
    # Phase D (search-play-title-4): …but never a question about the caller's
    # OWN data — that card opened on «جست‌وجو در وب» over a Gmail read.
    for owned in ("ایمیلای امروزم چیه؟", "جلسه فردام کیه؟", "قرار بعدیم کجاست",
                  "فایل قراردادم کجاست؟", "یادداشتم درباره سفر چیه"):
        plan = template_plan(owned, "full", rtl=True)
        assert "جست‌وجو در وب" not in plan and plan[-1] == "جمع‌بندی پاسخ", (owned, plan)


#: The corpus the relay seeds its card from: every card of the 2026-10-07 call
#: plus the shapes the plan grammar distinguishes.
_PLAN_CORPUS = [
    "search the web for flights", "anything", "what's new in my gmail",
    "refactor the parser", "make it nicer", "سرچ کن",
    "یه چیزی برام سرچ کن. ببین آیاس جنرال. خب؟ اه ریدینگ برای هج شدن میتونی چند تا غلط بزنی",
    "اوکی. بعد به نظرت همین توی آیسآکادمیک چجوریه", "کیه",
    "حالا یه دنه آهنگ دریک واسم پخش کن", "استاپش کن", "play some drake songs",
    "who won the last formula one race", "check my calendar for tomorrow",
    "تقویممو چک کن", "what is the capital of France?", "how are you",
    "اسم صبحش آهنگ", "remind me at 5",
]


def test_the_template_plan_is_the_labels_of_its_keys():
    """DESIGN-R3 D-R4(6): the relay seeds the card from `template_plan_keys`
    and the tenant declares `template_plan` — ONE decision, so for every
    request, intent, script and `require_web` the labels ARE the keys
    rendered, and the keys are a plan the TurnJob would declare verbatim."""
    from app.agent.query_intent import classify_query_intent
    from app.agent.voice_jobs import (
        PLAN_KINDS, plan_step_label, template_plan, template_plan_keys,
        valid_planned_steps,
    )
    for text in _PLAN_CORPUS:
        intent = classify_query_intent(text)
        for rw in (False, True):
            for rtl in (False, True):
                keys = template_plan_keys(text, intent, rtl, require_web=rw)
                assert keys[-1] == "answer" and set(keys) <= PLAN_KINDS, (text, keys)
                assert template_plan(text, intent, rtl, require_web=rw) == [
                    plan_step_label(k, rtl=rtl, display_request=text) for k in keys
                ], (text, rw, rtl)
                assert valid_planned_steps(keys, text) == keys, (text, keys)
    assert template_plan_keys("make it nicer", "full") == ["other", "answer"]
    assert template_plan_keys("make it nicer", "full", require_web=True) == [
        "web", "fetch", "answer"]


def test_a_body_plan_is_declared_only_when_every_kind_is_known():
    from app.agent.voice_jobs import valid_planned_steps
    assert valid_planned_steps(["web", "fetch", "answer"]) == ["web", "fetch", "answer"]
    assert valid_planned_steps(["media", "answer"]) == ["media", "answer"]
    for bad in (None, "web", [], ["answer"], ["web", "fetch"], ["answer", "web"],
                ["web", "answer", "answer"], ["web", "bogus", "answer"],
                ["web", 3, "answer"], ["other"] * 6 + ["answer"]):
        assert valid_planned_steps(bad) is None, bad
    # A connector kind needs the brand the request names.
    assert valid_planned_steps(["connector_read", "answer"], "anything new?") is None
    assert valid_planned_steps(["connector_read", "answer"], "check my gmail") == [
        "connector_read", "answer"]


@pytest.mark.asyncio
async def test_a_delegation_declares_the_relays_planned_steps(monkeypatch, tmp_path):
    """DESIGN-R3 D-R4(6): the relay drew `planned_steps` on the card at t0;
    the TurnJob declares exactly those, so the card does not change shape when
    the tenant's first frame lands. An unknown kind falls back to today's plan."""
    # "rename the file" is `code` here: today's plan would be files + work.
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-planned", said="rename the file",
        rounds=[], extra_kwargs={"planned_steps": ["web", "fetch", "answer"]},
    )
    first = out["sse"][0]
    assert [s["label"] for s in first["steps"]] == _PLAN_EN, first["steps"]
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-planned-bad", said="rename the file",
        rounds=[], extra_kwargs={"planned_steps": ["web", "teleport", "answer"]},
    )
    assert [s["label"] for s in out["sse"][0]["steps"]] == [
        "Read the files", "Work on it", "Put the answer together"]


@pytest.mark.asyncio
async def test_the_model_jobs_frames_carry_the_plan_too(monkeypatch, tmp_path):
    """Chat parity (D-P5 §2.6): `create_job`/`update_job` frames carry the
    step list, so neither client waits on a REST read to draw the plan."""
    import app.api.ws_chat as ws
    from app.agent.tool_executor import ToolExecutor
    from app.services import agent_notify_client as anc

    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)
    frames: list = []

    async def _capture(user_id, event, **kw):
        frames.append(dict(event))
        return 0

    monkeypatch.setattr(ws, "broadcast_to_user", _capture)
    tools = ToolExecutor(workspace=str(tmp_path))
    tools.set_user_id(await _make_user())
    made = json.loads(await tools.execute("create_job", {
        "title": "Compare laptops", "steps": ["Search", "Compare", "Write"]}))
    await tools.execute("update_job", {"job_id": made["job_id"], "current_step": 0})
    jf = [f for f in frames if f.get("type") == "job_update"]
    assert len(jf) == 2, frames
    assert [(s["label"], s["status"]) for s in jf[0]["steps"]] == [
        ("Search", "running"), ("Compare", "pending"), ("Write", "pending")]
    assert [s["status"] for s in jf[1]["steps"]] == ["done", "running", "pending"]
    assert "durationMs" not in jf[1]["steps"][0]


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
    # Round 16: the marker gained `job_type` (the clients' card
    # discriminator) and is written by the ONE shared writer,
    # `message_cards.job_marker_content`. Assert the fields, not the exact
    # object — the marker is allowed to grow, and every reader blanks it
    # before a client sees it either way.
    _marker = json.loads(row.content)
    assert _marker["job_id"] == job.id
    assert _marker["job_name"] == job.title
    assert set(_marker) <= {"job_id", "job_name", "job_type"}
    # …and it is NOT ascii-escaped: a Persian title must stay Persian in
    # the column, so the one reader that ever fails open fails legibly.
    assert "\\u" not in row.content


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
async def test_a_chat_turn_that_works_without_create_job_gets_the_runner_card(
        monkeypatch, tmp_path):
    """D-P4: one producer for both channels. A chat turn that searched used to
    get a card only if the model remembered `create_job` (and, for an English
    "research …", a second, empty one from ws_chat's retired regex intake)."""
    out = await _run_turn(monkeypatch, tmp_path, channel="web")
    jobs = await _jobs_for(out["user_id"])
    assert len(jobs) == 1
    assert (jobs[0].config_json or {}).get("turn_job") is True
    assert "voice_turn" not in (jobs[0].config_json or {})
    frames = [f for f in out["frames"] if f.get("type") == "job_update"]
    assert frames and all("steps" in f for f in frames), frames
    assert frames[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_a_chat_turn_whose_model_opens_a_job_gets_no_second_card(
        monkeypatch, tmp_path):
    """The model's `create_job` is the turn's card; the runner never adds one
    beside it."""
    out = await _run_turn(
        monkeypatch, tmp_path, channel="web",
        rounds=[[("c1", "create_job", {"title": "Compare", "steps": ["a", "b"]}),
                 ("s1", "web_search", {"query": "a"})]],
    )
    assert await _jobs_for(out["user_id"]) == []


@pytest.mark.asyncio
async def test_a_model_job_after_a_memory_round_still_gets_no_second_card(
        monkeypatch, tmp_path):
    """The runner's chat card is created lazily and only OPENED on a work
    round; a `create_job` arriving in that round is the model's card, and the
    runner's unopened one must stand down rather than open beside it."""
    out = await _run_turn(
        monkeypatch, tmp_path, channel="web",
        rounds=[[("m1", "memory_search", {"query": "x"})],
                [("c1", "create_job", {"title": "Compare", "steps": ["a", "b"]}),
                 ("s1", "web_search", {"query": "a"})]],
    )
    assert await _jobs_for(out["user_id"]) == []


async def _real_create_job(tools, inp):
    return await tools._tool_create_job(inp)


@pytest.mark.asyncio
async def test_a_model_job_after_the_card_opened_becomes_that_card(monkeypatch, tmp_path):
    """TW-4. Round 1 searches with no `create_job`, so the runner's chat card
    opens; round 2 the model calls the REAL `create_job`. One card: the
    model's plan becomes the card's remaining steps (the search that already
    ran stays), the model is handed the card's own id, and every action of
    the turn names that one job."""
    out = await _run_turn(
        monkeypatch, tmp_path, channel="web",
        rounds=[[("s1", "web_search", {"query": "image models"})],
                [("c1", "create_job", {"title": "Compare the models",
                                       "steps": ["Compare", "Write the answer"]}),
                 ("f1", "web_fetch", {"url": "https://example.org/x"})]],
        extra_tools={"create_job": _real_create_job},
    )
    jobs = await _jobs_for(out["user_id"])
    assert len(jobs) == 1, [(j.title, j.config_json) for j in jobs]
    job = jobs[0]
    assert (job.config_json or {}).get("turn_job") is True
    assert job.status == "completed"
    labels = [s["label"] for s in json.loads(job.steps_json)]
    assert labels[0] == "Search the web" and labels[-2:] == ["Compare", "Write the answer"], labels
    ends = {e["name"]: e for e in out["tool_events"] if e.get("phase") == "end"}
    assert json.loads(ends["create_job"]["result"])["job_id"] == job.id
    c = ends["create_job"]
    assert (c["step_index"], c["step_name"], c["steps_total"]) == (2, "Compare", 4), c
    assert {e.get("job_id") for e in out["tool_events"] if e.get("phase") == "end"} == {job.id}
    assert {f.get("job_id") for f in out["frames"] if f.get("type") == "job_update"} == {job.id}


@pytest.mark.asyncio
async def test_the_models_update_job_on_the_card_counts_from_its_own_plan(monkeypatch, tmp_path):
    """TW-4. Handed the card's id, the model numbers steps from ITS plan;
    `update_job(current_step=0)` starts the plan's second row, never a row of
    the work that ran before it, and the card stays the row's one writer."""
    import app.agent.voice_jobs as vj
    from app.agent.step_tracker import StepTracker
    from app.agent.tool_executor import ToolExecutor, _USER_ID_CTX
    monkeypatch.setattr(vj, "_spawn_bg", lambda coro, **kw: coro.close())
    uid = await _make_user()
    _USER_ID_CTX.set(uid)
    tools = ToolExecutor(workspace=str(tmp_path))
    tracker = StepTracker()
    tj = vj.TurnJob(user_id=uid, conversation_id=None, request_text="compare the models",
                    channel="web")
    vj.set_current_voice_job(tj)
    try:
        tj.plan([{"name": "web_search"}], tracker)
        out = json.loads(str(await tools._tool_create_job(
            {"title": "Compare", "steps": ["Read", "Compare", "Write the answer"]})))
        assert out["job_id"] == tj.job_id and out["steps"] == 4
        assert tj.step_labels() == ["Search the web", "Read", "Compare", "Write the answer"]
        assert tj.progress()[2] == 1
        upd = json.loads(str(await tools._tool_update_job(
            {"job_id": tj.job_id, "current_step": 0})))
        assert (upd["ok"], upd["current_step"], upd["total_steps"]) == (True, 1, 4)
        assert tj.progress()[2] == 2 and tracker.active == 2
    finally:
        vj.set_current_voice_job(None)
    assert await _jobs_for(uid) == []    # the card's own insert was held


@pytest.mark.asyncio
async def test_a_chat_delegate_before_any_work_is_the_turns_one_card(monkeypatch, tmp_path):
    """TW-5. `delegate` is not a work tool, so a chat round holding only it
    leaves the runner's card unopened — and the supervisor used to open a
    job of its own, after which the first verification search opened the
    runner card beside it. The supervisor now opens THROUGH that card: one
    job for the turn, carrying the plan, the role and the later search."""
    out = await _run_turn(
        monkeypatch, tmp_path, channel="web",
        said="plan a five day trip to japan with prices",
        rounds=[[("d1", "delegate", {"title": "Plan", "steps": []})],
                [("s1", "web_search", {"query": "verify a price"})]],
        extra_tools={"delegate": _delegate_through_the_card},
    )
    jobs = await _jobs_for(out["user_id"])
    assert len(jobs) == 1, [(j.title, j.config_json) for j in jobs]
    job = jobs[0]
    cfg = job.config_json or {}
    assert cfg.get("turn_job") is True and cfg.get("role") == "supervisor", cfg
    labels = [s["label"] for s in json.loads(job.steps_json)]
    assert labels[:3] == ["Helper 0", "Helper 1", "Check and merge"], labels
    assert "Search the web" in labels and labels[-1] == "Write the answer", labels
    ends = [e for e in out["tool_events"] if e.get("phase") == "end"]
    assert {e.get("job_id") for e in ends} == {job.id}, ends
    assert {f.get("job_id") for f in out["frames"] if f.get("type") == "job_update"} == {job.id}


@pytest.mark.asyncio
async def test_the_runner_card_is_the_turns_job_for_what_it_spawns(monkeypatch, tmp_path):
    """`tool_executor._JOB_ID_CTX` names the job a `spawn`/`delegate` from this
    turn lands under. With no `create_job` on voice, the runner's card is that
    job — set the moment it opens."""
    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-8",
                          said="search for the strongest image-generation model")
    job = (await _jobs_for(out["user_id"]))[0]
    assert out["tools"]._current_job_id == job.id


async def _delegate_through_the_card(tools, inp):
    """`delegate` without its workers: the real adoption of the turn's card
    (`_open_supervisor_job`) and the real progress write the plan makes when
    every helper has finished (`_update_supervisor`, final)."""
    from datetime import datetime as _dt
    from app.agent.subagent_orchestrator import _update_supervisor
    from app.db.database import async_session_maker
    plan = [{"label": "Helper 0", "task": "a"}, {"label": "Helper 1", "task": "b"}]
    jid, _ = await tools._open_supervisor_job(
        user_id=tools._current_user_id, title="Plan the trip", plan=plan,
        chat_id=None, asst_message_id=None)
    t = _dt.utcnow()
    state = [{"i": i, "label": p["label"], "status": "done", "started_at": t,
              "completed_at": t} for i, p in enumerate(plan)]
    await _update_supervisor(
        user_id=tools._current_user_id, job_id=jid, title="Plan the trip",
        state=state, final=True, chat_id=None, message_id=None,
        job_type_tag=None, sink=None, session_maker=async_session_maker)
    return '{"job_id": "%s", "steps": []}' % jid


@pytest.mark.asyncio
async def test_a_search_before_delegate_keeps_its_row_and_the_frames_name_the_plan(
        monkeypatch, tmp_path):
    """TW-3. A search the model made BEFORE delegating really ran: it stays on
    the card ahead of the supervisor's plan. And the turn's tracker follows
    the plan, so the delegate's own tool frame names the plan's merge row and
    the closing step event names "Write the answer" — not the template's
    "Search the web" / "Put the answer together" at a total of 3."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-sup",
        said="plan my trip to japan in detail please",
        rounds=[[("s1", "web_search", {"query": "japan"})],
                [("d1", "delegate", {"title": "Plan", "steps": []})]],
        extra_tools={"delegate": _delegate_through_the_card},
    )
    job = (await _jobs_for(out["user_id"]))[0]
    rows = [(s["label"], s["status"]) for s in json.loads(job.steps_json)]
    assert rows == [("Search the web", "done"), ("Helper 0", "done"),
                    ("Helper 1", "done"), ("Check and merge", "done"),
                    ("Write the answer", "done")], rows
    ends = {e["name"]: e for e in out["tool_events"] if e.get("phase") == "end"}
    assert (ends["web_search"]["step_index"], ends["web_search"]["step_name"]) == (
        0, "Search the web")
    d = ends["delegate"]
    assert (d["step_index"], d["step_name"], d["steps_total"]) == (
        3, "Check and merge", 5), d
    final = [e for e in out["step_events"] if e.get("final")]
    assert final, out["step_events"]
    assert (final[-1]["step_index"], final[-1]["step_name"], final[-1]["steps_total"]) == (
        4, "Write the answer", 5), final[-1]


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
    running would sit frozen until the 30-minute reaper closed it.

    R48 changed the WORD, not the guarantee. The card still terminalises from
    the sweep; it closes `cancelled` rather than `completed` because this turn
    was stopped before it had an answer, and the delivered case is covered by
    `test_a_hung_up_turn_whose_answer_landed_still_closes_completed` below.
    Never `failed`: the caller hanging up is not the work breaking.
    """
    import app.agent.voice_jobs as vj

    # The delivery proof is a bounded poll; a ceiling of zero is "ask once,
    # do not wait" — this turn has no answer to find either way.
    monkeypatch.setattr(vj, "_CANCEL_PROOF_POLL_S", 0.0)
    monkeypatch.setattr(vj, "_CANCEL_PROOF_MAX_WAIT_S", 0.0)
    job = vj.VoiceTurnJob(
        user_id=await _make_user(), conversation_id=str(uuid.uuid4()),
        request_text="find the strongest image-generation model",
    )
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    # The sweep runs from run()'s finally, synchronously, on a cancelled task.
    vj.sweep_current_voice_job()
    assert vj.current_voice_job() is None
    await asyncio.sleep(0.4)
    rows = await _jobs_for(job._user_id)  # noqa: SLF001 — the row is the point
    assert len(rows) == 1
    assert rows[0].status == "cancelled", (
        "a voice turn stopped before it delivered anything must not claim "
        "'Done in N steps' — the work happened, the answer never arrived"
    )


# ── Voice round 2026-10-06 (D-T1): a delegated media command's card ──────

@pytest.mark.asyncio
async def test_a_delegated_stop_reads_control_the_music_not_one_skipped(
        monkeypatch, tmp_path):
    """«استاپش کن» delegated: the card was "Work on it · 1 skipped" — the
    only tool a media turn runs is not work, so the generic step never
    started. The plan now declares ONE media step and the media tool (still
    non-work: it never earns a card of its own) starts it."""
    from app.agent.tool_display import ToolResult

    async def _stop(tools, inp):
        r = ToolResult("Stopped: the user's device confirmed the music stopped.",
                       display="Stopped the music")
        r.media = {"action": "stop", "reason": "stopped", "ok": True}
        return r

    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-media", said="استاپش کن",
        rounds=[[("m1", "media_control", {"action": "stop"})]],
        extra_tools={"media_control": _stop},
    )
    jobs = await _jobs_for(out["user_id"])
    assert len(jobs) == 1
    steps = json.loads(jobs[0].steps_json)
    assert [s["label"] for s in steps] == ["کنترل پخش", "جمع‌بندی پاسخ"]
    assert [s["status"] for s in steps] == ["done", "done"], steps
    ends = [e for e in out["tool_events"] if e.get("phase") == "end"]
    assert ends and ends[0]["step_index"] == 0, ends


# ── Voice round 2026-10-06 (D-T2): search OR delegate on round 0 ─────────

@pytest.mark.asyncio
@pytest.mark.parametrize("said,gate,want_delegate", [
    ("compare the three best noise-cancelling headphones for me, search it", True, True),
    ("compare the three best noise-cancelling headphones for me, search it", False, False),
    ("search who won the last F1 race", True, False),
])
async def test_a_multi_part_search_may_go_to_the_supervisor_on_round_zero(
        monkeypatch, tmp_path, said, gate, want_delegate):
    """D-P3 locked round 0 of a search ask to the web tools, so a heavy
    "compare the three best X" could never reach `delegate`: the first round
    searched inline and the model finished inline. The lock is now "search OR
    delegate" — still `required` — and only when the request is multi-part AND
    the gate left `delegate` live for this user."""
    from app.agent import supervisor_gate as g
    from app.config import settings

    monkeypatch.setattr(settings, "supervisor_enabled", gate)
    monkeypatch.setattr(settings, "supervisor_user_allowlist", "")
    g.reset_for_test()
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-sup0", require_web=True,
        said=said, rounds=[[("s1", "web_search", {"query": "x"})]],
    )
    tc = out["llm"].tool_choices[0]
    assert isinstance(tc, dict), tc
    assert tc["allowed_tools"]["mode"] == "required"
    names = {t["function"]["name"] for t in tc["allowed_tools"]["tools"]}
    assert "web_search" in names
    assert ("delegate" in names) is want_delegate, names


# ── Voice round 2026-10-06 (D-T4): the card is titled by its request ─────

def test_the_relays_task_title_wins_over_a_fragment_request():
    """The «ها» delegation was titled «ها». The relay's title (the stuck
    turn's words or its summary) now wins; a fragment never titles; the
    generic no longer reads like a step."""
    import app.agent.voice_jobs as vj

    tj = vj.TurnJob(user_id="u", conversation_id=None, request_text="ها",
                    task_title="مسابقه فرمول یک این هفته کجاست")
    assert tj.title == "مسابقه فرمول یک این هفته کجاست"
    assert vj.TurnJob(user_id="u", conversation_id=None, request_text="ها",
                      task_title="ها").title == "درخواست صوتی"
    # A one-word title from the relay is a fragment too ("huh" survives
    # `derive_job_title`; «ها» does not even get that far).
    assert vj.TurnJob(user_id="u", conversation_id=None,
                      request_text="who won the F1 race",
                      task_title="huh").title == "Who won the F1 race"
    assert vj.TurnJob(user_id="u", conversation_id=None,
                      request_text="who won the F1 race",
                      task_title=None).title == "Who won the F1 race"


def test_the_anchor_rides_the_row_config_and_the_frame():
    import app.agent.voice_jobs as vj

    tj = vj.TurnJob(user_id="u", conversation_id="c", request_text="who won the F1 race",
                    delegation_id="dlg-9", anchor={
                        "parent_user_turn_id": "t-1", "request_turn_ids": ["t-1"],
                        "anchor_occurred_at": "2026-10-06T12:41:09", "junk": "x"})
    cfg = tj._config()
    assert cfg["parent_user_turn_id"] == "t-1" and cfg["request_turn_ids"] == ["t-1"]
    assert cfg["anchor_occurred_at"] == "2026-10-06T12:41:09" and "junk" not in cfg
    assert tj._frame("running", [])["parent_user_turn_id"] == "t-1"
    bare = vj.TurnJob(user_id="u", conversation_id="c", request_text="who won the F1 race")
    assert "parent_user_turn_id" not in bare._config()
    assert "parent_user_turn_id" not in bare._frame("running", [])


@pytest.mark.asyncio
async def test_the_relays_title_and_anchor_reach_the_runners_card(monkeypatch, tmp_path):
    """End to end through `_run_inner`: the body fields the route forwards
    (`task_title`, `voice_anchor`) land on the BuildJob row."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-title", said="ها",
        rounds=[[("s1", "web_search", {"query": "f1"})]],
        extra_kwargs=dict(
            task_title="The F1 race this weekend",
            voice_anchor={"parent_user_turn_id": "t-ask",
                          "anchor_occurred_at": "2026-10-06T12:41:09"},
        ),
    )
    jobs = await _jobs_for(out["user_id"])
    assert jobs and jobs[0].title == "The F1 race this weekend"
    assert (jobs[0].config_json or {}).get("parent_user_turn_id") == "t-ask"


# ── Voice round 2026-10-06 (D-T3): the round-START answer beat ──────────

@pytest.mark.asyncio
async def test_the_answer_step_runs_while_the_answer_is_written(monkeypatch, tmp_path):
    """The `reasoning`/`final` event used to fire AFTER the answer round, so
    «دارم جواب را می‌نویسم» never showed while the answer was written. The
    first text of a tool-less round (≥ 40 chars) now fires a provisional one,
    and the post-round event still follows."""

    class _LongAnswer(_ScriptedLLM):
        async def create_message_stream(self, **kwargs):
            from app.services.openai_agent_service import StreamEvent as E
            if self.calls >= len(self._rounds):
                self.calls += 1
                self.tool_choices.append(kwargs.get("tool_choice"))
                self.messages.append(list(kwargs.get("messages") or []))
                yield E(type="text", text="The strongest image model right now is ")
                yield E(type="text", text="Imagen 4 Ultra, by most published benchmarks.")
                yield E(type="message_end", stop_reason="end_turn",
                        usage={"input_tokens": 10, "output_tokens": 9})
                return
            async for ev in super().create_message_stream(**kwargs):
                yield ev

    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-beat",
                          llm_cls=_LongAnswer)
    finals = [e for e in out["step_events"] if e.get("kind") == "reasoning" and e.get("final")]
    assert [bool(e.get("provisional")) for e in finals] == [True, False], finals
    assert finals[0]["step_index"] == finals[1]["step_index"]


# ── Voice round 2026-10-06 (D-T3): the speculative first search ──────────

@pytest.mark.asyncio
async def test_a_search_delegation_searches_at_t0_and_round_zero_reads_it(
        monkeypatch, tmp_path):
    """The first search no longer waits phase 1 plus a 3–6 s model round: it
    starts at t0 through the real executor (real start/end frames, the card's
    search step running), lands in round 0 as a context block, and releases
    the forced web round — the evidence is already there."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-spec", require_web=True,
        said="search who won the last formula one race",
        rounds=[], speculative=True,
    )
    evs = out["tool_events"]
    spec = [e for e in evs if str(e.get("call_id", "")).startswith("spec_")]
    assert [e["phase"] for e in spec] == ["start", "end"], evs
    assert spec[0]["input"] == {"query": "who won the last formula one race"}
    assert spec[0].get("job_id"), "the search is the card's own step, from t0"
    first_call = out["llm"].messages[0]
    blocks = [m for m in first_call if isinstance(m.get("content"), str)
              and m["content"].startswith("<speculative_search")]
    assert len(blocks) == 1 and "example.org" in blocks[0]["content"]
    tc = out["llm"].tool_choices[0]
    assert not (isinstance(tc, dict) and tc.get("allowed_tools", {}).get("mode") == "required"), tc
    jobs = await _jobs_for(out["user_id"])
    steps = json.loads(jobs[0].steps_json)
    assert steps[0]["status"] == "done" and steps[0]["label"] == "Search the web"


@pytest.mark.asyncio
async def test_a_late_speculative_search_is_dropped_and_the_round_stays_forced(
        monkeypatch, tmp_path):
    import app.agent.agent_runner as ar

    monkeypatch.setattr(ar.settings, "voice_speculative_search_deadline_s", 0.0,
                        raising=False)

    async def _slow(tools, inp):
        await asyncio.sleep(0.5)
        return "1. Late\n   https://late.example/x\n"

    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-spec-late", require_web=True,
        said="search who won the last formula one race",
        rounds=[[("s1", "web_search", {"query": "f1"})]], speculative=True,
        extra_tools={"web_search": _slow},
    )
    assert not any(isinstance(m.get("content"), str)
                   and m["content"].startswith("<speculative_search")
                   for m in out["llm"].messages[0])
    tc = out["llm"].tool_choices[0]
    assert isinstance(tc, dict) and tc["allowed_tools"]["mode"] == "required"
    spec = [e for e in out["tool_events"] if str(e.get("call_id", "")).startswith("spec_")]
    assert [e["phase"] for e in spec] == ["start", "end"], "a dropped search still closes its row"


@pytest.mark.asyncio
async def test_the_first_search_starts_before_phase_one_reads_the_db(monkeypatch, tmp_path):
    """Voice round 2026-10-07 (T1): with a COLD call cache and a slow DB, the
    speculative search's `tool.start` is out before phase 1's first statement
    returns and before its own `disabled_tools` read does — it never queues
    behind the DB to start."""
    import app.agent.agent_runner as ar
    from app.agent import call_context_cache as cc

    cc.reset_for_test()
    marks: dict = {}
    real_tz = ar.AgentRunner._resolve_effective_tz

    async def _slow_tz(self, *a, **kw):
        await asyncio.sleep(0.3)
        marks["phase1_first_statement"] = time.perf_counter()
        return await real_tz(self, *a, **kw)

    async def _slow_read(user_id):
        await asyncio.sleep(0.4)
        marks["disabled_read"] = time.perf_counter()
        return None

    monkeypatch.setattr(ar.AgentRunner, "_resolve_effective_tz", _slow_tz)
    monkeypatch.setattr(ar.AgentRunner, "_read_disabled_tools", staticmethod(_slow_read))
    seen: list = []

    async def _on_tool_event(ev):
        seen.append((time.perf_counter(), dict(ev)))

    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-spec-t0", require_web=True,
        said="search who won the last formula one race", rounds=[], speculative=True,
        extra_kwargs={"on_tool_event": _on_tool_event},
    )
    start = next(t for t, e in seen
                 if str(e.get("call_id", "")).startswith("spec_") and e["phase"] == "start")
    assert start < marks["phase1_first_statement"], "the search waited for phase 1"
    assert start < marks["disabled_read"], "the search waited for its own row read"
    assert out["resp"] is not None
    cc.reset_for_test()


@pytest.mark.asyncio
async def test_the_relays_search_query_is_what_the_first_search_runs(monkeypatch, tmp_path):
    """Voice round 2026-10-07 (T12): the relay's cleaned request wins over the
    display text, which still carries the caller's hesitation words."""
    from app.agent import call_context_cache as cc

    cc.reset_for_test()
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-spec-q", require_web=True,
        said="ok so umm search who won, like, the last formula one race",
        rounds=[], speculative=True,
        extra_kwargs={"search_query": "last formula one race winner"},
    )
    spec = [e for e in out["tool_events"] if str(e.get("call_id", "")).startswith("spec_")]
    assert spec and spec[0]["input"] == {"query": "last formula one race winner"}, spec
    cc.reset_for_test()


@pytest.mark.asyncio
async def test_a_one_word_relay_query_still_starts_the_first_search(monkeypatch, tmp_path):
    """Phase D (search-play-title-1, P0): the relay's cleaner left card 2 of
    the 10-07 call ONE word («آیسآکادمیک»); `speculative_search_query` refuses
    under two, and `search_query or display_request` never reached the display
    text — so the t0 search was OFF and round 0 was forced to web tools only,
    a whole model round before the first tool.  The display text drives it."""
    from app.agent import call_context_cache as cc

    cc.reset_for_test()
    said = "اوکی. بعد به نظرت همین توی آیسآکادمیک چجوریه"
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-spec-oneword", require_web=True,
        said=said, rounds=[], speculative=True,
        extra_kwargs={"search_query": "آیسآکادمیک"},
    )
    spec = [e for e in out["tool_events"] if str(e.get("call_id", "")).startswith("spec_")]
    assert spec, "a one-word relay query switched the t0 search off"
    assert "آیسآکادمیک" in spec[0]["input"]["query"] and len(spec[0]["input"]["query"].split()) >= 2, spec
    cc.reset_for_test()


@pytest.mark.parametrize(("text", "owned"), [
    ("تقویممو سرچ کن ببین فردا چی دارم", True), ("search my gmail for the invoice", True),
    ("جیمیلم چی اومده امروز؟", True), ("تو اسلک سرچ کن ببین علی دیروز چی گفت", True),
    ("تو جلسه‌هام سرچ کن ببین فردا با کی قرار دارم", True), ("ایمیلای امروزم چیه؟", True),
    ("جلسه فردام کیه؟", True), ("فایل قراردادم کجاست؟", True), ("پیام‌هام رو بخون", True),
    # Phase D (lens-tenant-2): research asks the substring list called "owned"
    ("سرچ کن ببین آیفون جدید کی قراره بیاد", False),
    ("سرچ کن آخرین خبرهای چت جی پی تی چیه", False),
    ("سرچ کن گزارش هواشناسی فردا", False),
    ("رویدادهای مهم امروز تهران رو سرچ کن", False),
    ("جلسه امروز مجلس درباره چی بود سرچ کن", False),
    ("قرارداد ترکمنچای کی امضا شد", False), ("جلسه دوم مجلس کی بود", False),
])
def test_owned_data_is_the_callers_own_not_a_substring(text, owned):
    """Phase D lens-tenant-2 (P1): `owned_data_request` used the classifier's
    unanchored substring list («قرار», «چت», «گزارش», «رویداد», «جلسه» …), so
    the T4 guard stripped the relay's `require_web` from ordinary research
    asks.  Own data is a data noun that is the caller's."""
    from app.agent.voice_jobs import owned_data_request

    assert owned_data_request(text) is owned


@pytest.mark.asyncio
@pytest.mark.parametrize("said", [
    "سرچ کن ببین آیفون جدید کی قراره بیاد", "سرچ کن آخرین خبرهای چت جی پی تی چیه",
])
async def test_a_research_ask_keeps_its_forced_search(monkeypatch, tmp_path, caplog, said):
    """Phase D lens-tenant-2: «…قراره بیاد» / «…چت جی پی تی…» with the relay's
    `require_web` (a backstop or decide_web source): the T4 guard read them
    as the caller's own data, dropped `require_web` — no t0 search, no forced
    web round — though the caller said «سرچ کن»."""
    import logging as _logging

    caplog.set_level(_logging.INFO)
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-research-kept", require_web=True,
        said=said, rounds=[], speculative=True,
    )
    assert [e for e in out["tool_events"] if str(e.get("call_id", "")).startswith("spec_")]
    assert not any("require_web ignored" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_an_owned_data_ask_never_declares_the_relays_web_seed(monkeypatch, tmp_path):
    """Phase D search-play-title-3: when T4 drops `require_web` for an ask
    about the caller's own data, the relay's web plan seed ([web, fetch,
    answer]) was still declared verbatim — the card read «جست‌وجو در وب ·
    خواندن منابع» over a Slack read and ended with two skipped web rows.  The
    card is re-planned without the web."""
    from app.agent import call_context_cache as cc

    cc.reset_for_test()
    slack = "تو اسلک سرچ کن ببین علی دیروز چی گفت"
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-owned-seed", require_web=True,
        said=slack, display_request=slack, rounds=[], speculative=True,
        extra_kwargs={"planned_steps": ["web", "fetch", "answer"], "search_query": "اسلک علی"},
    )
    first = [step.get("label") for step in (out["sse"][0].get("steps") or [])]
    assert first and not any("وب" in label or "منابع" in label for label in first), first
    cc.reset_for_test()


@pytest.mark.asyncio
async def test_require_web_is_ignored_for_the_callers_own_data(monkeypatch, tmp_path, caplog):
    """Voice round 2026-10-07 (T4, REQUIRED with the relay's `decide_web`):
    «تقویممو سرچ کن» / "search my gmail" searches the caller's own data, never
    the web — no speculative search and no web-locked round 0. (The card's
    plan still follows the intent classifier, as before.)"""
    import logging as _logging

    caplog.set_level(_logging.INFO)
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-owned", require_web=True,
        said="search my gmail for the invoice", rounds=[], speculative=True,
    )
    assert not [e for e in out["tool_events"] if str(e.get("call_id", "")).startswith("spec_")]
    tc = out["llm"].tool_choices[0]
    assert not (isinstance(tc, dict) and tc.get("allowed_tools", {}).get("mode") == "required"), tc
    assert tc != "required", tc
    assert any("require_web ignored reason=owned_data" in r.getMessage() for r in caplog.records)
    # …and an ordinary research ask keeps its forced search.
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-not-owned", require_web=True,
        said="who is the strongest llm professor at u of t", rounds=[], speculative=True,
    )
    assert [e for e in out["tool_events"] if str(e.get("call_id", "")).startswith("spec_")]


# ── Voice round 2026-10-07: effort, one lineage, the research budget ─────

_BUDGET = {"voice_research_max_searches": 2, "voice_research_max_reads": 1,
           "voice_research_deep_factor": 2}


@pytest.mark.asyncio
async def test_every_round_of_a_voice_delegation_sends_low_effort(monkeypatch, tmp_path):
    """T5: one effort for every round of a voice delegation (n=3 rounds at
    `low` measured ≈1 s faster each). A voice turn that is not a delegation
    and a chat turn keep the per-intent map."""
    over = {"voice_delegation_reasoning_effort": "low"}
    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-effort",
                          settings_over=over)
    assert out["llm"].calls == 3 and out["llm"].efforts == ["low"] * 3, out["llm"].efforts
    plain = await _run_turn(monkeypatch, tmp_path, settings_over=over)
    assert "low" not in plain["llm"].efforts, plain["llm"].efforts
    chat = await _run_turn(monkeypatch, tmp_path, channel="chat", settings_over=over)
    assert "low" not in chat["llm"].efforts, chat["llm"].efforts


@pytest.mark.asyncio
async def test_the_string_tool_choice_sends_no_allowed_tools_dict(monkeypatch, tmp_path):
    """T6 (DARK — `voice_delegation_string_tool_choice` defaults False): with
    the flag on, a voice delegation never sends an `allowed_tools` dict; the
    forced web round is the STRING "required" plus a note."""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-string-tc", require_web=True,
        said="search who won the last formula one race",
        rounds=[[("s1", "web_search", {"query": "f1"})]],
        settings_over={"voice_delegation_string_tool_choice": True},
    )
    tcs = out["llm"].tool_choices
    assert not any(isinstance(tc, dict) for tc in tcs), tcs
    assert tcs[0] == "required", tcs
    assert any(m.get("content") == ar_note("_WEB_FORCE_NOTE") for m in out["llm"].messages[0])
    from app.config import Settings
    assert Settings.model_fields["voice_delegation_string_tool_choice"].default is False


def ar_note(name: str) -> str:
    import app.agent.agent_runner as ar
    return getattr(ar, name)


@pytest.mark.asyncio
async def test_the_research_budget_refuses_quietly_and_forces_the_answer(monkeypatch, tmp_path):
    """T7: 2 searches (the speculative one counts) and 1 read. Over-budget
    calls are NOT run: the model gets a "Not run" result and the call surface
    gets NO frame; once both are spent the model is told to answer now.
    (Phase D lens-tenant-3: that round keeps its tools — only the LAST round
    is tool-less — so an action the request asked for still runs.)"""
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-budget", require_web=True,
        said="search who won the last formula one race", speculative=True,
        rounds=[[("s1", "web_search", {"query": "a"}), ("s2", "web_search", {"query": "b"})],
                [("f1", "web_fetch", {"url": "https://example.org/p"}),
                 ("s3", "web_search", {"query": "c"})]],
        settings_over=_BUDGET,
    )
    ran = [e["call_id"] for e in out["tool_events"] if e["phase"] == "start"]
    assert [c for c in ran if not c.startswith("spec_")] == ["s1", "f1"], ran
    assert len([c for c in ran if c.startswith("spec_")]) == 1
    refused = ar_note("_VR_REFUSED_RESULT")

    def _results(msgs):
        return {b["tool_use_id"]: b["content"] for m in msgs
                if isinstance(m.get("content"), list)
                for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"}

    r = _results(out["llm"].messages[2])
    assert r["s2"] == refused and r["s3"] == refused, r
    assert r["s1"] != refused and r["f1"] != refused
    assert out["llm"].tool_choices[2] != "none", out["llm"].tool_choices
    assert out["llm"].messages[2][-1]["content"] == ar_note("_VR_BUDGET_NOTE")
    assert out["resp"].text == "Right now it's Imagen 4 Ultra."
    line = "Research budget for this spoken answer: 2 web searches and 1 page read(s)"
    assert (any(line in str(m.get("content")) for m in out["llm"].messages[0])
            or line in str(out["llm"].systems[0])), "the model is told the budget"
    # The persisted records hold only what ran.
    assert {e["call_id"] for e in out["tool_events"]}.isdisjoint({"s2", "s3"})


@pytest.mark.asyncio
async def test_a_spent_research_budget_still_lets_the_request_act(monkeypatch, tmp_path):
    """Phase D lens-tenant-3 (P1): the budget counts web tools, but spending
    it forced a `tool_choice="none"` round that forbade EVERY tool — «check
    tomorrow's weather in Toronto and remind me at 8 to take an umbrella»
    spent its searches and read, and the reminder was never set ("Here is
    what I found so far.").  The round after the budget keeps its tools; a
    further web call is still refused in-round."""
    reminded: list = []

    async def remind(_tools, inp):
        reminded.append(inp)
        return "Reminder set for 08:00."

    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-budget-act", require_web=True,
        said="check tomorrow's weather in toronto and remind me at 8 to take an umbrella",
        speculative=True, llm_cls=_HonoursNone, settings_over=_BUDGET,
        rounds=[[("s1", "web_search", {"query": "toronto weather"}),
                 ("f1", "web_fetch", {"url": "https://example.org/w"})],
                [("r1", "routines__remind", {"text": "umbrella", "at": "08:00"}),
                 ("s9", "web_search", {"query": "more"})]],
        extra_tools={"routines__remind": remind},
    )
    assert reminded == [{"text": "umbrella", "at": "08:00"}], out["llm"].tool_choices
    assert out["llm"].tool_choices[1] != "none", out["llm"].tool_choices
    ran = [e["call_id"] for e in out["tool_events"] if e["phase"] == "start"]
    assert "s9" not in ran, ran          # the budget still holds in-round


@pytest.mark.asyncio
async def test_a_late_speculative_search_is_never_announced_as_run(monkeypatch, tmp_path):
    """Phase D lens-tenant-7: the per-turn budget line said "(one already ran
    for you)" whenever the speculative search was CREATED, though it counts
    only when its result is injected — a late, empty or failed one told the
    model a search ran that never reached its context (and it planned one
    search instead of two)."""
    import app.agent.agent_runner as ar

    async def late(self, query, user_id, vjob, steps, emit, *, run_started=None):
        await asyncio.sleep(0.3)
        return None

    monkeypatch.setattr(ar.AgentRunner, "_speculative_search", late)
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-budget-late", require_web=True,
        said="search who won the last formula one race", speculative=True,
        rounds=[[("s1", "web_search", {"query": "a"})]],
        settings_over={**_BUDGET, "voice_speculative_search_deadline_s": 0.0},
    )
    seen = " ".join(str(m.get("content")) for m in out["llm"].messages[0]) + str(out["llm"].systems[0])
    assert "Research budget for this spoken answer" in seen
    assert "already ran" not in seen, "a search that never reached the model was announced"


@pytest.mark.asyncio
async def test_a_depth_ask_gets_the_deep_budget(monkeypatch, tmp_path):
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-deep", require_web=True,
        said="search in depth who won the last formula one race", speculative=True,
        rounds=[[("s1", "web_search", {"query": "a"}), ("s2", "web_search", {"query": "b"})],
                [("s3", "web_search", {"query": "c"})]],
        settings_over=_BUDGET,
    )
    ran = [e["call_id"] for e in out["tool_events"]
           if e["phase"] == "start" and not str(e["call_id"]).startswith("spec_")]
    assert ran == ["s1", "s2", "s3"], ran


class _HonoursNone(_ScriptedLLM):
    """A provider that obeys `tool_choice="none"` / `tools=None`."""

    async def create_message_stream(self, **kwargs):
        if kwargs.get("tool_choice") == "none" or kwargs.get("tools") is None:
            from app.services.openai_agent_service import StreamEvent as E
            self.calls += 1
            self.tool_choices.append(kwargs.get("tool_choice"))
            self.messages.append(list(kwargs.get("messages") or []))
            self.efforts.append(kwargs.get("reasoning_effort"))
            self.systems.append(kwargs.get("system"))
            self.tools_sent.append(kwargs.get("tools") is not None)
            yield E(type="text", text="Here is what I found so far.")
            yield E(type="message_end", stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 2})
            return
        async for ev in super().create_message_stream(**kwargs):
            yield ev


_SIX_ROUNDS = [[(f"m{i}", "memory_search", {"query": str(i)})] for i in range(6)]


@pytest.mark.asyncio
async def test_the_last_allowed_round_of_a_voice_delegation_is_tool_less(monkeypatch, tmp_path):
    """T7: «کیه» ended on the English step-budget boilerplate because its last
    round called a tool and the loop simply ran out. The LAST allowed round of
    every voice delegation is now tool-less, so the answer is an answer."""
    over = {**_BUDGET, "voice_max_tool_iterations": 3}
    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-last",
                          rounds=_SIX_ROUNDS, llm_cls=_HonoursNone, settings_over=over)
    assert out["llm"].tool_choices[2] == "none", out["llm"].tool_choices
    assert out["resp"].text == "Here is what I found so far."
    assert "keep going" not in out["resp"].text


@pytest.mark.asyncio
async def test_the_last_round_is_tool_less_with_the_research_budget_off(monkeypatch, tmp_path, caplog):
    """Phase D lens-tenant-6: the tool-less last round was computed only while
    the research budget was ON, and `voice_research_max_searches=0` is its
    documented kill switch — thrown, a voice delegation whose last round
    called a tool ended EMPTY (`voice turn ended without an answer`), the
    «کیه» shape again, now silent."""
    import logging as _logging

    caplog.set_level(_logging.INFO)
    over = {"voice_research_max_searches": 0, "voice_max_tool_iterations": 3}
    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-last-off",
                          rounds=_SIX_ROUNDS, llm_cls=_HonoursNone, settings_over=over)
    assert out["llm"].tool_choices[2] == "none", out["llm"].tool_choices
    assert out["resp"].text == "Here is what I found so far."
    assert out["llm"].messages[2][-1]["content"] == ar_note("_VR_LAST_ROUND_NOTE")
    assert not any("voice turn ended without an answer" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_claude_tool_less_round_keeps_its_tools(monkeypatch, tmp_path):
    """Phase D lens-tenant-5: on a Claude model the forced round sent
    `tools=None` and no tool_choice while the history held tool_use /
    tool_result blocks — a 400 invalid_request_error from the Messages API,
    and the fallback round then ran WITH tools and no choice (the "tool-less"
    last round could call a tool and the answer end empty).  The round keeps
    its tools and says "none" (AnthropicService maps it to {"type": "none"})."""
    import app.agent.agent_runner as ar

    claude = _ScriptedLLM(_SIX_ROUNDS)
    monkeypatch.setattr(ar, "AnthropicService", lambda *a, **k: claude)
    over = {**_BUDGET, "voice_max_tool_iterations": 3, "anthropic_enabled": True}
    await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-claude-last",
                    rounds=_SIX_ROUNDS, settings_over=over,
                    extra_kwargs={"model_override": "claude-sonnet-4-6"})
    assert claude.calls == 3, claude.tool_choices
    assert claude.tool_choices[2] == "none" and claude.tools_sent[2] is True, (
        claude.tool_choices, claude.tools_sent)


@pytest.mark.asyncio
async def test_the_step_budget_boilerplate_is_never_a_voice_answer(monkeypatch, tmp_path, caplog):
    """T7: even when a provider ignores the tool-less round, a voice delegation
    never answers with the English "keep going" sentence — an empty answer is
    honest (`outcome=empty` on the relay). A chat turn keeps today's text."""
    import logging as _logging
    import app.agent.agent_runner as ar

    caplog.set_level(_logging.INFO)
    over = {**_BUDGET, "voice_max_tool_iterations": 3}
    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-anomaly",
                          rounds=_SIX_ROUNDS, settings_over=over)
    assert out["llm"].calls == 3
    assert "keep going" not in (out["resp"].text or "")
    assert any("voice turn ended without an answer reason=iterations" in r.getMessage()
               for r in caplog.records)
    monkeypatch.setattr(ar.AgentRunner, "_effective_max_iterations", lambda self: 3)
    chat = await _run_turn(monkeypatch, tmp_path, channel="chat", rounds=_SIX_ROUNDS,
                           settings_over=over)
    assert "keep going" in chat["resp"].text


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [True, False], ids=["budget_on", "budget_off"])
async def test_a_search_reentry_never_lands_on_the_tool_less_last_round(
    monkeypatch, tmp_path, budget,
):
    """T7 × D-P3: the LAST round is tool-less — so the explicit-search
    re-entry may only fire when its search round leaves one more round to
    answer in. Otherwise the re-entered round is forced to "none", searches
    nothing, and the answer it replaced is lost.  (Phase D lens-tenant-6: the
    last round is tool-less with the budget OFF too, so the slack follows.)"""

    class _AnswersWithoutSearching(_ScriptedLLM):
        async def create_message_stream(self, **kwargs):
            from app.services.openai_agent_service import StreamEvent as E
            self.calls += 1
            self.tool_choices.append(kwargs.get("tool_choice"))
            self.messages.append(list(kwargs.get("messages") or []))
            if self.calls == 2:
                yield E(type="tool_use_start", tool_name="web_search", tool_id="s1")
                yield E(type="tool_use_end", tool_name="web_search", tool_id="s1",
                        tool_input={"query": "f1"})
                yield E(type="message_end", stop_reason="tool_use",
                        usage={"input_tokens": 10, "output_tokens": 5})
                return
            yield E(type="text", text="Max Verstappen won it.")
            yield E(type="message_end", stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 4})

    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-reentry-last", require_web=True,
        said="search who won the last formula one race", llm_cls=_AnswersWithoutSearching,
        settings_over={**(_BUDGET if budget else {"voice_research_max_searches": 0}),
                       "voice_max_tool_iterations": 2},
    )
    assert out["llm"].calls == 1, out["llm"].tool_choices
    assert out["resp"].text == "Max Verstappen won it."


# ── Voice round 2026-10-07 (T11 / T3): the addendum, the live registry ───

def _live_box(delegation_id="dlg-add", session_id="sess-add"):
    from app.api.api_v1 import _VsLive
    return _VsLive(delegation_id, session_id)


@pytest.mark.asyncio
async def test_an_addendum_is_folded_at_the_next_round_and_nothing_is_discarded(
        monkeypatch, tmp_path):
    """T11: «کیه» arrived while the UofT question ran. It is drained at the
    next round barrier as part of the SAME request — unlike a correction, the
    round's proposed calls still execute — and a late title retitles the card."""
    live = _live_box()
    offered: list = []

    async def _search(tools, inp):
        if not offered:
            offered.append(live.offer("turn:12", "کیه", None))
            live.offer("title:1", "", "Strongest image model this year")
        return f"1. Result for {inp['query']}\n   https://example.org/page\n"

    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-add",
                          extra_tools={"web_search": _search},
                          extra_kwargs={"addendum_check": live})
    assert offered == [{"accepted": True, "state": "queued"}]
    ran = [e["call_id"] for e in out["tool_events"] if e["phase"] == "start"]
    assert ran == ["s1", "s2", "f1"], "the round's calls were not discarded"
    second = out["llm"].messages[1]
    notes = [m for m in second if isinstance(m.get("content"), str) and "«کیه»" in m["content"]]
    assert len(notes) == 1 and "part of the same request" in notes[0]["content"]
    assert not any("«کیه»" in str(m.get("content")) for m in out["llm"].messages[0])
    assert live.applied == ["turn:12"] and live.state == "closing"
    names = [f.get("name") for f in out["sse"] if f.get("type") == "job.update"]
    assert names[-1] == "Strongest image model this year", names
    job = (await _jobs_for(out["user_id"]))[0]
    assert job.title == "Strongest image model this year"


class _AddendumDuringAnswer(_ScriptedLLM):
    """The caller adds words while the FINAL round is being written."""

    def __init__(self, rounds=None):
        super().__init__(rounds)
        self.live = None
        self.finals = 0

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent as E
        if self.calls < len(self._rounds):
            async for ev in super().create_message_stream(**kwargs):
                yield ev
            return
        self.calls += 1
        self.tool_choices.append(kwargs.get("tool_choice"))
        self.messages.append(list(kwargs.get("messages") or []))
        self.finals += 1
        if self.finals == 1 and self.live is not None:
            self.live.offer("turn:20", "and how much does it cost", None)
        yield E(type="text", text=f"answer-{self.finals}")
        yield E(type="message_end", stop_reason="end_turn",
                usage={"input_tokens": 10, "output_tokens": 2})


@pytest.mark.asyncio
async def test_an_addendum_during_the_answer_gets_one_extra_round(monkeypatch, tmp_path):
    """T11: an addendum that lands while the answer is written gets exactly ONE
    extra round (the model sees its answer and the note); the run's text is the
    second answer. After it the delegation closes atomically — a later POST is
    `answer_written`, never queued and dropped."""
    live = _live_box()

    class _LLM(_AddendumDuringAnswer):
        def __init__(self, rounds=None):
            super().__init__(rounds)
            self.live = live

    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-add",
                          llm_cls=_LLM, extra_kwargs={"addendum_check": live})
    llm = out["llm"]
    assert llm.calls == 4 and llm.finals == 2, llm.calls
    assert "«and how much does it cost»" in llm.messages[-1][-1]["content"]
    # Phase D (lens-tenant-4): the extra round's text REPLACES answer-1, so its
    # note asks for the COMPLETE answer — never "do not start over".
    from app.api.api_v1 import _VS_ADDENDUM_AFTER_ANSWER_NOTE
    assert llm.messages[-1][-1]["content"] == _VS_ADDENDUM_AFTER_ANSWER_NOTE.format(
        text="and how much does it cost")
    assert "Do not start over" not in llm.messages[-1][-1]["content"]
    assert llm.messages[-1][-2] == {"role": "assistant",
                                    "content": [{"type": "text", "text": "answer-1"}]}
    assert out["resp"].text == "answer-2"
    assert live.applied == ["turn:20"] and live.state == "closing"
    assert live.offer("turn:21", "one more thing", None) == {
        "accepted": False, "reason": "answer_written"}


@pytest.mark.asyncio
async def test_a_second_late_addendum_is_reported_unapplied(monkeypatch, tmp_path):
    """≤1 extra round: an addendum during the EXTRA round is not taken — the
    delegation closes and `done.addenda_applied` will not name it."""
    live = _live_box()

    class _LLM(_AddendumDuringAnswer):
        def __init__(self, rounds=None):
            super().__init__(rounds)
            self.live = live

        async def create_message_stream(self, **kwargs):
            if self.finals == 1:
                live.offer("turn:22", "and in canada", None)
            async for ev in super().create_message_stream(**kwargs):
                yield ev

    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-add",
                          llm_cls=_LLM, extra_kwargs={"addendum_check": live})
    assert out["llm"].finals == 2 and out["resp"].text == "answer-2"
    assert live.applied == ["turn:20"] and live.state == "closing"
    assert [aid for aid, _ in live.queued] == ["turn:22"]


async def _voice_rows(user_id, session_id, *, job_status="running", delegation_id="dlg-x",
                      answer=None):
    from app.db import async_session_maker
    from app.db.models import BuildJob, Conversation, Message
    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        if not await db.get(Conversation, session_id):
            db.add(Conversation(id=session_id, user_id=user_id, channel="voice"))
        db.add(BuildJob(id=job_id, user_id=user_id, title="t", prompt="t",
                        job_type="agent_task", status=job_status,
                        conversation_id=session_id,
                        config_json={"voice_delegation_id": delegation_id}))
        if answer:
            db.add(Message(id=str(uuid.uuid4()), conversation_id=session_id,
                           role="assistant", content=answer,
                           metadata_json=json.dumps({"voice": {
                               "delegation_id": delegation_id, "source": "delegated",
                               "spoken": True}})))
        await db.commit()
    return job_id


@pytest.mark.asyncio
async def test_the_cancel_route_wakes_the_running_delegation(monkeypatch):
    """T3: the cancel route writes the durable Stop bit AND sets the running
    delegation's in-process event — the watch wakes now, not on a DB poll."""
    from app.api import api_v1
    from app.api.api_v1 import VoiceDelegationCancelRequest as C

    uid = await _make_user()
    monkeypatch.setattr(api_v1.settings, "agent_api_key", "k", raising=False)
    monkeypatch.setattr(api_v1.settings, "user_id", uid, raising=False)
    monkeypatch.setattr(api_v1, "_VS_LIVE", {})
    sid = str(uuid.uuid4())
    job_id = await _voice_rows(uid, sid, delegation_id="dlg-stop")
    live = api_v1._VsLive("dlg-stop", sid)
    api_v1._vs_register(live)

    class _Req:
        headers = {"X-Agent-Key": "k"}

    out = await api_v1.internal_voice_delegation_cancel(
        C(session_id=sid, task_id="dlg-stop", job_id=job_id), _Req())
    assert out["requested"] is True
    assert live.cancel.is_set()


@pytest.mark.asyncio
async def test_status_reads_the_db_once_for_the_ids_that_are_not_live(monkeypatch):
    """T3: one live id (memory) + one finished id (its job row and its answer
    row) — one DB session, both shapes."""
    from app.api import api_v1
    from app.api.api_v1 import VoiceDelegationStatusRequest as S

    uid = await _make_user()
    monkeypatch.setattr(api_v1.settings, "agent_api_key", "k", raising=False)
    monkeypatch.setattr(api_v1.settings, "user_id", uid, raising=False)
    monkeypatch.setattr(api_v1, "_VS_LIVE", {})
    sid = str(uuid.uuid4())
    await _voice_rows(uid, sid, job_status="completed", delegation_id="dlg-done",
                      answer="Colin Raffel.")
    live = api_v1._VsLive("dlg-live", sid)
    live.job_id, live.job_status = "job-live-9", "running"
    api_v1._vs_register(live)
    real = api_v1.async_session_maker
    opened: list = []

    def _counting(*a, **kw):
        opened.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(api_v1, "async_session_maker", _counting)

    class _Req:
        headers = {"X-Agent-Key": "k"}

    out = (await api_v1.internal_voice_delegations_status(
        S(session_id=sid, task_ids=["dlg-live", "dlg-done"]), _Req()))["tasks"]
    assert out["dlg-live"] == {"status": "running", "job_id": "job-live-9",
                               "cancel_requested": False}
    assert out["dlg-done"]["status"] == "completed"
    assert out["dlg-done"]["answer"] == "Colin Raffel."
    assert len(opened) == 1


# ── Voice round 2026-10-07 (D-R2(4), T2): the voice-context cache + prewarm ──

_TORONTO = "America/Toronto"


class _AgentReq:
    headers = {"X-Agent-Key": "k"}


async def _voice_ctx_user(monkeypatch, *, ttl=120.0):
    from collections import OrderedDict as _OD
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo
    from sqlalchemy import update as _upd
    from app.api import api_v1
    from app.agent import call_context_cache as cc
    from app.db import async_session_maker
    from app.db.models import Conversation, Message, User
    from app.db.models.day_chat import DayChat

    uid = await _make_user()
    monkeypatch.setattr(api_v1.settings, "agent_api_key", "k", raising=False)
    monkeypatch.setattr(api_v1.settings, "user_id", uid, raising=False)
    monkeypatch.setattr(api_v1.settings, "voice_context_cache_ttl_s", ttl, raising=False)
    monkeypatch.setattr(api_v1, "_VC_CACHE", _OD())
    monkeypatch.setattr(api_v1, "_voice_prewarm_last", {})
    cc.reset_for_test()
    day_id, conv_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        await db.execute(_upd(User).where(User.id == uid).values(timezone=_TORONTO))
        db.add(DayChat(id=day_id, user_id=uid, local_date=_dt.now(ZoneInfo(_TORONTO)).date(),
                       timezone=_TORONTO))
        db.add(Conversation(id=conv_id, user_id=uid, channel="voice", day_chat_id=day_id))
        db.add(Message(id=str(uuid.uuid4()), conversation_id=conv_id, day_chat_id=day_id,
                       role="user", content="what is the weather"))
        await db.commit()
    return uid, day_id, conv_id


@pytest.mark.asyncio
async def test_the_voice_context_is_cached_until_the_day_changes(monkeypatch):
    """D-R2(4): the tenant's `/voice-context` build is reused for 120 s while
    the day's one-aggregate validator answers the same — a new message is a
    rebuild — and the one clock line is re-rendered at the request's instant."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    from fastapi import BackgroundTasks
    from app.api import api_v1
    from app.api.api_v1 import VoiceContextRequest as V
    from app.agent import voice_context as vc
    from app.db import async_session_maker
    from app.db.models import Message

    uid, day_id, conv_id = await _voice_ctx_user(monkeypatch)
    builds: list = []
    real = vc.build_voice_context

    async def _counting(db, user_id, **kw):
        builds.append(kw)
        return await real(db, user_id, **kw)

    monkeypatch.setattr(vc, "build_voice_context", _counting)
    t1 = _dt(2026, 10, 7, 16, 4, tzinfo=_tz.utc)

    async def ask(now, **kw):
        return await api_v1.internal_voice_context(
            V(live=True, tz_name=_TORONTO, now=now, **kw), _AgentReq(), BackgroundTasks())

    first = await ask(t1)
    assert first.cached is False and first.age_ms == 0 and len(builds) == 1
    second = await ask(t1 + _td(minutes=1))
    assert second.cached is True and second.age_ms >= 0 and len(builds) == 1
    assert "16:05 UTC" in second.instructions and "16:04 UTC" not in second.instructions
    assert second.sections["day_history"] == first.sections["day_history"]
    other = await ask(t1, budget_chars=5000)
    assert other.cached is False and len(builds) == 2, "a different request is a different build"
    async with async_session_maker() as db:
        db.add(Message(id=str(uuid.uuid4()), conversation_id=conv_id, day_chat_id=day_id,
                       role="assistant", content="It is sunny."))
        await db.commit()
    third = await ask(t1 + _td(minutes=1))
    assert third.cached is False and len(builds) == 3, "a new message invalidates the build"
    assert "It is sunny." in third.instructions
    # A write to a memory file / the soul / the agent config in this process
    # bumps the memory generation: the persona and the brain are rebuilt.
    assert (await ask(t1)).cached is True
    from app.agent import call_context_cache as cc
    cc.bump_memory_generation("test")
    assert (await ask(t1)).cached is False and len(builds) == 4
    monkeypatch.setattr(api_v1.settings, "voice_context_cache_ttl_s", 0, raising=False)
    assert (await ask(t1)).cached is False


@pytest.mark.asyncio
async def test_a_warm_refreshes_a_build_past_half_its_life(monkeypatch):
    """Phase D turns-startup-7: a warm answered from the cache kept the
    build's stamp, so warms could not keep the context fresh for a later tap
    — a presence warm at t=0, an app warm at t=70 s (a hit) and the tap at
    t=130 s missed and built on the critical path.  A warm now rebuilds a
    build older than half the TTL; a call still reads it for the whole TTL."""
    from datetime import datetime as _dt, timezone as _tz
    from fastapi import BackgroundTasks
    from app.api import api_v1
    from app.api.api_v1 import VoiceContextRequest as V
    from app.agent import voice_context as vc

    await _voice_ctx_user(monkeypatch)
    builds: list = []
    real = vc.build_voice_context

    async def _counting(db, user_id, **kw):
        builds.append(kw)
        return await real(db, user_id, **kw)

    monkeypatch.setattr(vc, "build_voice_context", _counting)
    t1 = _dt(2026, 10, 7, 16, 4, tzinfo=_tz.utc)

    async def ask(**kw):
        return await api_v1.internal_voice_context(
            V(live=True, tz_name=_TORONTO, now=t1, **kw), _AgentReq(), BackgroundTasks())

    def age(by_s):
        (key, (ctx, val, ts)), = list(api_v1._VC_CACHE.items())
        api_v1._VC_CACHE[key] = (ctx, val, ts - by_s)

    assert (await ask(warm=True)).cached is False and len(builds) == 1   # t = 0
    age(40.0)
    assert (await ask(warm=True)).cached is True and len(builds) == 1    # t = 40: young
    age(30.0)
    warm = await ask(warm=True)                                          # t = 70: refreshed
    assert warm.cached is False and len(builds) == 2, (warm.cached, len(builds))
    age(60.0)
    tap = await ask()                                                    # t = 130: the tap
    assert tap.cached is True and 59_000 <= tap.age_ms <= 61_500 and len(builds) == 2, tap


@pytest.mark.asyncio
async def test_the_voice_context_prewarms_the_call_caches_once(monkeypatch, tmp_path, caplog):
    """T2: card 1 missed every per-call cache (the agent-config row, the voice
    prompt, the day-chat latch) and paid 4.6 s of contended DB reads. A warm or
    a Live `/voice-context` fills them AFTER the response, at most once per
    user per 45 s, and the prompt lands under EXACTLY the key phase 1 derives."""
    import logging as _logging
    from fastapi import BackgroundTasks
    import app.agent.agent_runner as ar
    from app.agent import call_context_cache as cc
    from app.agent.prompt_profile import PromptProfile
    from app.agent.tool_executor import ToolExecutor
    from app.api import api_v1
    from app.api.api_v1 import VoiceContextRequest as V
    from app.db import AgentConfig, async_session_maker

    caplog.set_level(_logging.INFO)
    uid, _day, _conv = await _voice_ctx_user(monkeypatch)
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=uid, disabled_tools='["web_search"]'))
        await db.commit()
    monkeypatch.setattr(ar.settings, "stable_prefix_layout", True, raising=False)

    async def _stub_build(self, db, user_id, user_message, *, client_tz=None,
                          turn_context_out=None, **kw):
        turn_context_out["__tz_name"] = client_tz
        return "PROMPT"

    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt_uncached", _stub_build)
    runner = ar.AgentRunner(llm_service=_ScriptedLLM(), tool_executor=ToolExecutor(
        workspace=str(tmp_path)))
    monkeypatch.setattr(api_v1, "_agent_runner", runner)
    bt = BackgroundTasks()
    await api_v1.internal_voice_context(V(warm=True, live=True, tz_name=_TORONTO),
                                        _AgentReq(), bt)
    assert len(bt.tasks) == 1, "the prewarm runs after the response"
    assert cc.get_agent_config(uid) == (False, None), "…not before it"
    await bt()
    hit, view = cc.get_agent_config(uid)
    assert hit and view.disabled_tools == '["web_search"]'
    phase1_key = runner._prompt_cache_key(
        uid, "voice", _TORONTO, PromptProfile.FULL, None, None, {}, None)
    assert phase1_key is not None and cc.get_prompt(phase1_key) is not None, (
        "the prewarmed prompt must sit under the key phase 1 will derive")
    assert any("voice_prewarm" in r.getMessage() and "prompt=set" in r.getMessage()
               for r in caplog.records)
    again = BackgroundTasks()
    await api_v1.internal_voice_context(V(live=True, tz_name=_TORONTO), _AgentReq(), again)
    assert again.tasks == [], "debounced: one prewarm per user per 45 s"


# ── Phase D (2026-10-06 review) ─────────────────────────────────────────

_HEAVY = "compare the three best noise-cancelling headphones for me, search it"


def _sup_on(monkeypatch, on=True):
    from app.agent import supervisor_gate as g
    from app.config import settings

    monkeypatch.setattr(settings, "supervisor_enabled", on)
    monkeypatch.setattr(settings, "supervisor_user_allowlist", "")
    g.reset_for_test()


@pytest.mark.asyncio
async def test_a_delegated_search_ask_is_not_forced_into_one_more_web_round(
        monkeypatch, tmp_path):
    """correctness-tenant-1: `delegate` is not a web-evidence tool, so after
    the supervisor's workers searched and merged, the explicit-search re-entry
    still forced one more web-only round and rewrote the answer."""
    _sup_on(monkeypatch)

    async def _delegate(tools, inp):
        return "Sony WH-1000XM6 leads; Bose QC Ultra and AirPods Max follow."

    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-deleg", require_web=True,
        said=_HEAVY, rounds=[[("d1", "delegate", {"task": _HEAVY})]],
        extra_tools={"delegate": _delegate},
    )
    assert out["llm"].calls == 2, out["llm"].tool_choices


@pytest.mark.asyncio
async def test_an_empty_speculative_search_is_not_this_turns_search(monkeypatch, tmp_path):
    """correctness-tenant-2: the executor fences every non-empty web_search
    result, so "No search results found." never read as `ERROR` — it was
    injected as round 0's evidence and released the forced search."""

    async def _empty(tools, inp):
        return "No search results found."

    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-spec-empty", require_web=True,
        said="search who won the last formula one race",
        rounds=[[("s1", "web_search", {"query": "f1"})]], speculative=True,
        extra_tools={"web_search": _empty},
    )
    assert not any(isinstance(m.get("content"), str)
                   and m["content"].startswith("<speculative_search")
                   for m in out["llm"].messages[0])
    tc = out["llm"].tool_choices[0]
    assert isinstance(tc, dict) and tc["allowed_tools"]["mode"] == "required", tc


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", [True, False])
async def test_a_landed_speculative_search_keeps_a_heavy_asks_delegate_lock(
        monkeypatch, tmp_path, gate):
    """correctness-tenant-3: the speculative result released round 0 for a
    multi-part ask too, so D-T2's "search OR delegate, required" never ran and
    a heavy request was answered from one search. With `delegate` live the
    lock stays (the result rides along); without it, today's unforced round."""
    _sup_on(monkeypatch, gate)
    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id=f"dlg-spec-heavy-{gate}", require_web=True,
        said=_HEAVY, rounds=[[("s1", "web_search", {"query": "x"})]], speculative=True,
    )
    assert any(isinstance(m.get("content"), str)
               and m["content"].startswith("<speculative_search")
               for m in out["llm"].messages[0])
    tc = out["llm"].tool_choices[0]
    if gate:
        assert isinstance(tc, dict) and tc["allowed_tools"]["mode"] == "required", tc
        names = {t["function"]["name"] for t in tc["allowed_tools"]["tools"]}
        assert "delegate" in names, names
    else:
        assert not (isinstance(tc, dict)
                    and tc.get("allowed_tools", {}).get("mode") == "required"), tc


@pytest.mark.asyncio
async def test_a_beat_before_the_search_reentry_never_closes_the_answer_step_early(
        monkeypatch, tmp_path):
    """correctness-tenant-5: round 0 wrote ≥40 chars without searching (the
    provisional beat moved the card to the answer step), then the explicit-
    search re-entry `continue`d without retracting it — the search's
    `attribute()` closed the answer step DONE before the work it summarises."""

    class _AnswersFirst(_ScriptedLLM):
        async def create_message_stream(self, **kwargs):
            from app.services.openai_agent_service import StreamEvent as E
            self.calls += 1
            self.tool_choices.append(kwargs.get("tool_choice"))
            self.messages.append(list(kwargs.get("messages") or []))
            if self.calls == 2:
                yield E(type="tool_use_start", tool_name="web_search", tool_id="s1")
                yield E(type="tool_use_end", tool_name="web_search", tool_id="s1",
                        tool_input={"query": "f1 winner"})
                yield E(type="message_end", stop_reason="tool_use",
                        usage={"input_tokens": 10, "output_tokens": 5})
                return
            yield E(type="text", text="I believe the last Formula One race was won by ")
            yield E(type="text", text="Max Verstappen, at the Japanese Grand Prix.")
            yield E(type="message_end", stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 9})

    out = await _run_turn(
        monkeypatch, tmp_path, delegation_id="dlg-beat-reentry", require_web=True,
        said="search who won the last formula one race", llm_cls=_AnswersFirst,
    )
    assert out["llm"].calls == 3
    jobs = await _jobs_for(out["user_id"])
    steps = json.loads(jobs[0].steps_json)
    web = next(s for s in steps if s.get("type") == "web" or "web" in str(s.get("label", "")).lower())
    answer = steps[-1]
    assert web is not answer
    assert answer["status"] == "done" and web["status"] == "done"
    assert str(answer.get("started_at") or "") >= str(web.get("started_at") or ""), steps
    assert str(answer.get("completed_at") or "") >= str(web.get("completed_at") or ""), steps


@pytest.mark.asyncio
async def test_a_turn_that_fails_in_phase_one_cancels_its_speculative_search(
        monkeypatch, tmp_path):
    """correctness-tenant-7: the speculative search was awaited or cancelled
    at ONE place (round 0); any earlier exit left it running with no owner —
    it metered a search and wrote a "searched" row onto a failed turn."""
    import app.agent.agent_runner as ar

    finished: list = []

    async def _slow(tools, inp):
        await asyncio.sleep(0.6)
        finished.append(inp)
        return "1. Late\n   https://late.example/x\n"

    async def _boom(self, *a, **kw):
        await asyncio.sleep(0.05)
        raise RuntimeError("phase 1 failed")

    monkeypatch.setattr(ar.AgentRunner, "_get_or_create_session", _boom)
    with pytest.raises(RuntimeError):
        await _run_turn(
            monkeypatch, tmp_path, delegation_id="dlg-spec-orphan", require_web=True,
            said="search who won the last formula one race", rounds=[],
            speculative=True, extra_tools={"web_search": _slow},
        )
    await asyncio.sleep(0.9)
    assert finished == [], "the orphaned speculative search ran to completion"
    assert ar._SPEC_TASK_CV.get() is None


@pytest.mark.asyncio
async def test_a_retracted_answer_beat_walks_the_card_back_with_a_new_row_revision(
        monkeypatch, tmp_path):
    """contract-wire-1: the provisional beat moved the card (prior step done,
    answer running); when the same round then called a tool, `retract_answer`
    restored it — a BACKWARD move both phones' rank merge discarded, because
    no row's `rev` changed. The restored rows now carry a bumped `rev`, on the
    job frame (`public_steps`) as well as on the row."""

    class _Preamble(_ScriptedLLM):
        async def create_message_stream(self, **kwargs):
            from app.services.openai_agent_service import StreamEvent as E
            self.calls += 1
            self.tool_choices.append(kwargs.get("tool_choice"))
            self.messages.append(list(kwargs.get("messages") or []))
            if self.calls == 2:
                yield E(type="text", text="Let me look up the latest standings for you now.")
                yield E(type="tool_use_start", tool_name="web_search", tool_id="s2")
                yield E(type="tool_use_end", tool_name="web_search", tool_id="s2",
                        tool_input={"query": "standings"})
                yield E(type="message_end", stop_reason="tool_use",
                        usage={"input_tokens": 10, "output_tokens": 5})
                return
            if self.calls == 1:
                yield E(type="tool_use_start", tool_name="web_search", tool_id="s1")
                yield E(type="tool_use_end", tool_name="web_search", tool_id="s1",
                        tool_input={"query": "f1"})
                yield E(type="message_end", stop_reason="tool_use",
                        usage={"input_tokens": 10, "output_tokens": 5})
                return
            yield E(type="text", text="Verstappen leads the standings by twelve points.")
            yield E(type="message_end", stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 9})

    out = await _run_turn(monkeypatch, tmp_path, delegation_id="dlg-retract",
                          said="what are the latest formula one standings",
                          llm_cls=_Preamble)
    jobs = [f for f in out["sse"] if isinstance(f.get("steps"), list) and f["steps"]]
    assert jobs, out["sse"]
    last_idx = len(jobs[0]["steps"]) - 1
    moved = next(i for i, f in enumerate(jobs) if f["steps"][last_idx].get("status") == "running")
    back = next(f for f in jobs[moved + 1:] if f["steps"][last_idx].get("status") == "pending")
    assert int(back["steps"][last_idx].get("rev") or 0) >= 1, back["steps"]
    before = jobs[moved]["steps"]
    changed = [i for i, row in enumerate(back["steps"])
               if row.get("status") != before[i].get("status")]
    assert changed and all(int(back["steps"][i].get("rev") or 0) >= 1 for i in changed), (
        before, back["steps"])
