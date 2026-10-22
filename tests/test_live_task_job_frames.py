"""D-P5/D-P6 relay half: the tenant's job frames on the voice socket.

A delegated voice task used to reach the phone as a title and a six-word
summary — the tenant's job, its plan and its workers never crossed the relay.
A client that negotiates `task_job` now gets the job itself (`job_update`,
`step_event`), the job's id on every lifecycle `delegation` observation once
known, and the worker keys on tool rows; every other client is unchanged.
"""

import pytest

import test_live_harness as H
from app.services import live_voice_protocol as live


DID = "d1"


def _job(**extra):
    frame = {
        "type": "job.update", "job_id": "job-1", "name": "Find a dentist",
        "job_type": "research", "status": "running", "step": "Search the web",
        "total_steps": 2, "completed_steps": 0, "channel": "chat",
        "steps": [
            {"id": 0, "type": "search", "label": "Search the web", "status": "running",
             "started_at": "2026-10-05T13:45:00Z"},
            {"id": 1, "type": "answer", "label": "Put the answer together", "status": "pending"},
        ],
    }
    frame.update(extra)
    return frame


_WORKER = {"worker_job_id": "worker-9", "parent_job_id": "job-1",
           "worker_index": 0, "label": "Researcher", "model": "gpt-x"}


# ── the pure builders ──────────────────────────────────────────────────

def test_job_update_keeps_the_chat_keys_and_their_types_and_never_a_channel():
    frame = live.task_job_update_frame(_job(), DID)

    assert frame["type"] == "job_update"
    assert frame["task_id"] == DID and frame["delegation_id"] == DID
    assert "channel" not in frame, "both clients drop a channel they do not own"
    assert isinstance(frame["step"], str) and isinstance(frame["name"], str)
    assert frame["total_steps"] == 2 and frame["completed_steps"] == 0
    assert frame["steps"][0] == {
        "id": 0, "type": "search", "label": "Search the web", "status": "running",
        "started_at": "2026-10-05T13:45:00Z",
    }


def test_job_update_carries_a_rows_revision_so_a_retracted_beat_lands():
    """Phase D (contract-wire-1): a retracted answer beat walks two rows BACK
    (done→running, running→pending); both phones keep the higher status of a
    row unless its `rev` changed, so the relay must forward the tenant's `rev`
    or the card stays on "writing the answer" through the search."""
    frame = live.task_job_update_frame(_job(steps=[
        {"id": 0, "type": "search", "label": "Search the web", "status": "running", "rev": 1},
        {"id": 1, "type": "answer", "label": "Put the answer together", "status": "pending",
         "rev": 1},
        {"id": 2, "type": "answer", "label": "x", "status": "pending"},
    ]), DID)
    assert [s.get("rev") for s in frame["steps"]] == [1, 1, None]


def test_job_update_reads_untrusted_fields_defensively():
    frame = live.task_job_update_frame(_job(
        name=None, title="Compare three laptops",
        step={"type": "search"},             # an object in `step` would crash the web
        total_steps="x", worker_index=True, color="red", attempt=-1,
        steps=[{"label": "Search"}, "junk", {"status": "running"}] + [{"label": "s"}] * 20,
        sources=[{"title": "T", "url": "https://a.example/x", "domain": "a.example"}] * 9,
        result_preview="The answer.",
    ), DID)

    assert frame["name"] == "Compare three laptops"
    assert "step" not in frame and "total_steps" not in frame
    assert "worker_index" not in frame and "color" not in frame and "attempt" not in frame
    assert len(frame["steps"]) <= live._JOB_FRAME_STEPS_MAX
    assert all(s.get("label") or s.get("type") for s in frame["steps"])
    assert len(frame["sources"]) == live._JOB_FRAME_SOURCES_MAX
    assert frame["result_preview"] == "The answer."
    assert live.task_job_update_frame({"type": "job.update"}, DID) is None


def test_job_update_carries_a_fetched_pages_https_icon():
    """Wire §2.7 (voice round 2026-10-06): the tenant stamps a fetched page's
    own icon on its source (`sources[].icon`) and the app draws it on the
    call's job card.  The relay builds this frame key by key, so it must copy
    the key — https only, never an icon the tenant did not send."""

    frame = live.task_job_update_frame(_job(sources=[
        {"title": "A", "url": "https://a.example/x", "domain": "a.example",
         "icon": "https://a.example/apple-touch-icon.png"},
        {"title": "B", "url": "https://b.example/x", "domain": "b.example",
         "icon": "http://b.example/i.png"},
        {"title": "C", "url": "https://c.example/x", "domain": "c.example",
         "icon": "javascript:alert(1)"},
        {"title": "D", "url": "https://d.example/x", "domain": "d.example"},
    ]), DID)

    assert [s.get("icon") for s in frame["sources"]] == [
        "https://a.example/apple-touch-icon.png", None, None, None,
    ], frame["sources"]
    assert frame["sources"][0] == {
        "title": "A", "url": "https://a.example/x", "domain": "a.example",
        "icon": "https://a.example/apple-touch-icon.png",
    }


def test_worker_fields_ride_the_job_update():
    frame = live.task_job_update_frame(_job(
        job_id="worker-9", parent_job_id="job-1", role="executor", worker_index=1,
        label="Researcher", color="#4F46E5", model="gpt-x", plan_step_index=0,
        attempt=2, workers_total=3, workers_done=1,
    ), DID)

    for key, value in {
        "parent_job_id": "job-1", "role": "executor", "worker_index": 1,
        "label": "Researcher", "color": "#4F46E5", "model": "gpt-x",
        "plan_step_index": 0, "attempt": 2, "workers_total": 3, "workers_done": 1,
    }.items():
        assert frame[key] == value, key


def test_step_event_is_the_chat_shape_with_the_task():
    frame = live.task_step_event_frame(
        {"type": "step.event", "kind": "reasoning", "job_id": "job-1", "step_index": 1,
         "elapsed_ms": 812, "final": True, "secret": "x"}, DID,
    )
    assert frame == {
        "type": "step_event", "kind": "reasoning", "job_id": "job-1", "step_index": 1,
        "elapsed_ms": 812, "final": True, "task_id": DID, "delegation_id": DID,
    }
    assert live.task_step_event_frame({"type": "step.event"}, DID) is None


def test_step_event_forwards_the_tenants_provisional_answer_beat():
    """D-T3 (5): the tenant's round-start answer beat is `provisional: true`;
    the relay builds the phone frame key by key, so it must copy the flag —
    and never invent it (an older image's beat carries none)."""

    beat = {"type": "step.event", "kind": "reasoning", "job_id": "job-1",
            "step_index": 2, "final": True, "provisional": True}
    frame = live.task_step_event_frame(beat, DID)
    assert frame["provisional"] is True and frame["final"] is True
    for value in (None, False, "true", 1):
        other = dict(beat, provisional=value)
        assert "provisional" not in live.task_step_event_frame(other, DID), value
    assert "provisional" not in live.task_step_event_frame(
        {k: v for k, v in beat.items() if k != "provisional"}, DID,
    )


# ── end to end through the relay ──────────────────────────────────────

def _features(task_job: bool):
    features = list(H.config()["features"])
    return features + ["task_job"] if task_job else features


async def _run(monkeypatch, think, *, task_job: bool):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("find me a dentist near union station", 0, 400))
            provider.push(H.delegation(DID, 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(features=_features(task_job)), 0.9, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6)
    return client, provider


async def _job_think(user_id, task, session_id, relay=None, out=None, **kwargs):
    await relay.on_event(_job())
    await relay.on_event({"type": "tool.start", "call_id": "i1", "name": "web_search",
                          "args": {"query": "dentist"}, "job_id": "worker-9", **_WORKER})
    await relay.on_event({"type": "tool.end", "call_id": "i1", "name": "web_search",
                          "ok": True, "elapsed_ms": 40, "job_id": "worker-9", **_WORKER})
    await relay.on_event({"type": "step.event", "kind": "step_change", "job_id": "job-1",
                          "step_index": 1})
    return "Found one on Front Street.", "m"


@pytest.mark.asyncio
async def test_a_task_job_client_gets_the_job_before_the_card_is_told_its_id(monkeypatch):
    client, _ = await _run(monkeypatch, _job_think, task_job=True)

    caps = client.of("ready")[0]["capabilities"]
    assert caps.get("task_job") is True and caps.get("supervisor_frames") is True

    jobs = client.of("job_update")
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == "job-1" and jobs[0]["task_id"] == DID
    assert "channel" not in jobs[0]

    frames = client.frames
    job_at = frames.index(jobs[0])
    stamped = [i for i, f in enumerate(frames)
               if f.get("type") == "delegation" and f.get("job_id")]
    assert stamped and stamped[0] > job_at, "the job is known before the card names it"
    # From then on EVERY lifecycle observation names the task's own job — not
    # the worker's job off its tool row (the waiting_for_tool frame and the
    # terminal frame both carry `worker-9` in their extras).  The delivery
    # verdict that follows is not a lifecycle observation.
    after = [f for f in frames[job_at:]
             if f.get("type") == "delegation" and "task_revision" in f]
    assert after and all(f.get("job_id") == "job-1" for f in after), after
    assert after[-1]["phase"] == "completed"

    started = client.of("tool_call.started")[0]
    for key, value in _WORKER.items():
        assert started[key] == value, key
    assert client.of("tool_call.completed")[0]["label"] == "Researcher"

    steps = client.of("step_event")
    assert steps and steps[0]["task_id"] == DID and steps[0]["step_index"] == 1


@pytest.mark.asyncio
async def test_a_client_without_task_job_gets_its_frames_as_before(monkeypatch):
    client, _ = await _run(monkeypatch, _job_think, task_job=False)

    assert "task_job" not in client.of("ready")[0]["capabilities"]
    assert "supervisor_frames" not in client.of("ready")[0]["capabilities"]
    assert not client.of("job_update") and not client.of("step_event")
    assert all(f.get("job_id") != "job-1" for f in client.of("delegation"))


@pytest.mark.asyncio
async def test_workers_and_the_supervisor_become_progress_notes_once_each(monkeypatch):
    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await relay.on_event(_job())
        await relay.on_event({"type": "supervisor.started", "job_id": "job-1",
                              "workers_total": 2, "deadline_s": 240})
        worker = _job(job_id="worker-9", parent_job_id="job-1", worker_index=0,
                      label="Researcher", step="Reading example.com", status="running")
        await relay.on_event(worker)
        await relay.on_event(dict(worker))       # a repeat is not a second note
        return "Merged.", "m"

    _, provider = await _run(monkeypatch, think, task_job=True)

    notes = [e["content"] for e in provider.of("session.thinking.append")
             if e.get("delegation_id") == DID]
    assert any("2 working in parallel" in n for n in notes), notes
    researcher = [n for n in notes if "Researcher: Reading example.com" in n]
    assert len(researcher) == 1, notes


# ── voice round 2026-10-07 (D-R4 (6)): the plan from the FIRST frame ────
# Card 1 of the 10-07 call had no plan for 8.8 s (the tenant's phase 1) and
# card 2 opened on the generic «انجام کار / جمع‌بندی پاسخ».  The relay seeds
# `plan: [{kind, label}]` on every lifecycle observation until the job exists,
# computed with the tenant's OWN `template_plan` and the same inputs, and
# hands the same kinds to the tenant as `planned_steps`.

DENTIST = "find me a dentist near union station"


def _expected_rows(text, *, require_web):
    from app.agent import voice_jobs as vj
    from app.agent.query_intent import classify_query_intent

    labels = vj.template_plan(text, classify_query_intent(text), False, require_web=require_web)
    return labels


@pytest.mark.asyncio
async def test_the_created_frame_seeds_the_plan_the_tenant_will_declare(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    seen: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        seen.update(kwargs)
        await relay.on_event(_job())
        return "Found one on Front Street.", "m"

    client, _ = await _run(monkeypatch, think, task_job=True)

    lifecycle = [f for f in client.of("delegation") if "task_revision" in f]
    created = [f for f in lifecycle if f.get("phase") == "created"]
    assert created, lifecycle
    plan = created[0].get("plan")
    labels = _expected_rows(DENTIST, require_web=False)
    assert [row["label"] for row in plan] == labels, plan
    assert plan[-1]["kind"] == "answer" and len(plan) <= live.PLAN_SEED_MAX_ROWS
    assert all(set(row) == {"kind", "label"} for row in plan)
    # The tenant is handed the same steps, by kind.
    assert seen.get("planned_steps") == [row["kind"] for row in plan], seen.get("planned_steps")
    # Once the job exists its own steps own the card: no seed after that.
    with_job = [f for f in lifecycle if f.get("job_id") == "job-1"]
    assert with_job and all("plan" not in f for f in with_job), with_job
    before_job = [f for f in lifecycle if not f.get("job_id")]
    assert before_job and all(f.get("plan") == plan for f in before_job), before_job
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith("[LIVE] delegation plan id=d1 kind=search rows=3 source=seed")
        for m in messages
    ), messages
    assert any(m.startswith("[LIVE] task job id=d1 job=job-1 steps=2 seed=3") for m in messages)


@pytest.mark.asyncio
async def test_a_client_without_task_job_gets_no_plan_seed(monkeypatch):
    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        return "Found one on Front Street.", "m"

    client, _ = await _run(monkeypatch, think, task_job=False)
    assert all("plan" not in f for f in client.of("delegation"))


def _bare_session(settings_obj):
    session = live._LiveSession.__new__(live._LiveSession)
    session.settings = settings_obj
    session.web_tasks = {}
    session.user_id = "user-1"
    from types import SimpleNamespace

    session.supervisor = SimpleNamespace(running={}, queued=[])
    return session


def test_the_seed_is_planned_after_the_web_decision():
    """A task the relay decided must search (here: the research backstop's
    `require_web`) is seeded with the tenant's SEARCH plan, never the generic
    one its words alone would read as.  And card 2's own sentence — a
    QUESTION the classifier reads as `full` — is seeded with the search plan
    even with no web decision: the tenant's question-shape fallback lives
    inside `template_plan_keys`, the ONE function both sides call (Phase C:
    before TA's function was merged this read «انجام کار»)."""

    from app.config import settings

    session = _bare_session(settings)
    generic = "یه متن کوتاه برای تبریک تولد بنویس"
    task = live.DelegatedTask(delegation_id="relay-research:x", offset_ms=0, transcript=generic)
    task.require_web = True
    session.prepare_dispatch(task)
    assert [k for k, _label in task.plan_rows] == ["web", "fetch", "answer"], task.plan_rows
    assert task.web_source == "backstop"
    plain = live.DelegatedTask(delegation_id="d2", offset_ms=0, transcript=generic)
    session.prepare_dispatch(plain)
    assert [k for k, _label in plain.plan_rows] == ["other", "answer"], plain.plan_rows
    assert plain.web_source == "none"
    card2 = live.DelegatedTask(
        delegation_id="d3", offset_ms=0,
        transcript="اوکی. بعد به نظرت همین توی آیسآکادمیک چجوریه",
    )
    session.prepare_dispatch(card2)
    assert [k for k, _label in card2.plan_rows] == ["web", "fetch", "answer"], card2.plan_rows
    assert card2.web_source == "none"


def test_the_seed_keys_are_the_tenants_own_vocabulary():
    from app.agent import voice_jobs as vj
    from app.api.api_v1 import ChatRequest
    from app.agent.job_titles import is_rtl_text
    from app.agent.query_intent import classify_query_intent

    for text, web in [
        (DENTIST, False), ("تقویممو چک کن", False), ("حالا یه دنه آهنگ دریک واسم پخش کن", False),
        ("explain how large language models are trained", True),
    ]:
        rows = live.template_plan_rows(text, require_web=web)
        assert rows and rows[-1][0] == "answer", rows
        assert all(k in vj._LABELS_EN for k, _label in rows), rows
        # The seed's LABELS are the tenant's own rendering of the same keys —
        # a connector kind names its brand, never "{brand}" (ruling 5).
        intent = classify_query_intent(text)
        rtl = is_rtl_text(vj.clean_request_text(text) or text)
        assert [label for _k, label in rows] == vj.template_plan(
            text, intent, rtl, require_web=web,
        ), rows
        assert not any("{brand}" in label for _k, label in rows), rows
        # …and the tenant DECLARES the seed verbatim: the body key parses as
        # `ChatRequest.planned_steps` and passes `valid_planned_steps`
        # (Phase C seam, D-R4(6): RC's body → TA's TurnJob adoption).
        keys = [k for k, _label in rows]
        req = ChatRequest.model_validate({"message": text, "planned_steps": keys})
        assert req.planned_steps == keys
        assert vj.valid_planned_steps(req.planned_steps, text) == keys, (text, keys)


@pytest.mark.asyncio
async def test_step_event_answer_beat_marks_the_task_answering():
    """The tenant's round-start answer beat (`step.event` reasoning, final)
    sets `answer_started` on the running entry whatever the client
    negotiated — the progress lines read it (D-R4 (2))."""

    session = live._LiveSession.__new__(live._LiveSession)
    session.features = set()
    task = live.DelegatedTask(delegation_id=DID, offset_ms=0, transcript=DENTIST)
    entry = live.RunningDelegation(task=task, started_monotonic=0.0)
    await session.on_tenant_task_event(
        task, entry, {"type": "step.event", "kind": "reasoning", "final": True, "provisional": True},
    )
    assert entry.answer_started is True
    other = live.RunningDelegation(task=task, started_monotonic=0.0)
    await session.on_tenant_task_event(
        task, other, {"type": "step.event", "kind": "step_change", "step_index": 1},
    )
    assert other.answer_started is False
