# agent-mode: automations/automation_threads/_turns/build_jobs are AGENT_ONLY.
"""R38 — the agent becomes a NODE in the workflow.

A v2 step used to be one thing: a connector call. The engine could read
and it could write, and the only judgement in a run happened afterwards,
in the narrator, over material it could no longer change — so an
automation that wanted "summarise this the way I would, then post THAT"
had no way to say it.

`kind: "agent"` is that step. It calls nothing, carries a prompt and an
`output_var`, runs the prompt through the model with everything the
steps before it produced, and binds the answer to `{{var.<name>}}` —
which later steps' templates and the narration read like any other
variable.

What this file proves, on real rows:

  - the answer reaches a later write step's payload, through BOTH
    namespaces the spec exposes (`{{var.x}}` and `{{steps.<id>.text}}`);
  - the step is a STEP everywhere a step is counted — `steps_json`, the
    progress columns, and one thread turn with a sentence and a timing;
  - the prompt is byte-stable over the same facts, and the call is
    temperature 0.0 (the same discipline the triage got this round);
  - a failure obeys `on_error`, and a thinking step is never reported
    as a failed ACCOUNT — it has none to reconnect;
  - and a v2 spec with no agent step behaves IDENTICALLY: same canonical
    spec, same steps, same turns, same outbox payload, and the agent
    seam is never entered at all.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import select

from app.agent.automations import agent_step
from app.agent.automations.spec import validate_spec
from app.db.database import async_session_maker
from app.db.models import Automation, AutomationOutbox, BuildJob

from tests.test_run_ledger_v3 import (  # noqa: F401 — shared fixtures
    REGISTRY_V2, _ISSUES, _OK, _fake_dispatch, _mk_automation_v2,
    _mk_user, _one_run,
)


# ── fixtures ─────────────────────────────────────────────────────────


def _agent_spec(**over):
    """Read Jira → think about it → post what the thinking produced.

    The write reads `{{var.ranked}}` ONLY, so the payload is proof the
    binding happened: nothing else in the run can put that text there.
    """
    spec = {
        "version": 2,
        "name": "Ranked brief",
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
            {"id": "rank", "kind": "agent",
             "prompt": "Rank these by what blocks someone: "
                       "{{steps.issues.text}}",
             "output_var": "ranked"},
            {"id": "post", "connector_id": "slack",
             "tool": "slack__send_message",
             "params": {"channel": "{{grant.target.id}}",
                        "text": "{{var.ranked}}"},
             "grant_id": "g-1",
             "grant_target": {"kind": "channel", "id": "C-PIN",
                              "label": "#platform"}},
        ],
    }
    spec.update(over)
    return validate_spec(spec, REGISTRY_V2)


def _plain_spec():
    """The same automation with the thinking step removed — the shape
    every persisted v2 spec already has."""
    spec = {
        "version": 2,
        "name": "Plain brief",
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
    return validate_spec(spec, REGISTRY_V2)


def _fake_thinking(monkeypatch, answer="TP-482 first — it blocks the export.",
                   *, fail=None):
    """Replace the model seam. Records every prompt it was handed."""
    seen: list[str] = []

    async def _complete(prompt: str) -> str:
        seen.append(prompt)
        if fail is not None:
            raise fail
        return answer

    monkeypatch.setattr(agent_step, "_default_complete", _complete)
    return seen


async def _fire(monkeypatch, uid, a, vspec, *, responses=None):
    from app.agent.automations import executor_v2
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _fake_dispatch(responses or {
            "jira__search_issues": _ISSUES,
            "slack__send_message": _OK,
        }),
    )
    monkeypatch.setattr(
        "app.agent.automations.executor_v2.AUTOMATION_OUTBOX_UNDO_WINDOW_S",
        0,
    )
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        return await executor_v2.run_schedule_fire_v2(
            db, a2, vspec, vspec.schedule_source(),
            fire_key=f"t:{uuid.uuid4()}",
        )


async def _turns(a_id: str) -> list[dict]:
    from app.agent.automations import ledger
    job = await _one_run(a_id)
    async with async_session_maker() as db:
        return await ledger.run_turns(db, run_id=job.id)


# ── it runs, and its answer reaches the write ────────────────────────


@pytest.mark.asyncio
async def test_the_thinking_reaches_the_write_it_feeds(monkeypatch):
    uid = await _mk_user()
    vspec = _agent_spec()
    a = await _mk_automation_v2(uid, vspec)
    seen = _fake_thinking(monkeypatch, "TP-482 first — it blocks the export.")

    assert await _fire(monkeypatch, uid, a, vspec) == "run"

    async with async_session_maker() as db:
        row = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
        )).scalar_one()
    sent = json.loads(row.payload_json)
    # The write's template names ONLY {{var.ranked}} — there is no other
    # route for this string into the payload.
    assert sent["text"] == "TP-482 first — it blocks the export."
    assert sent["channel"] == "C-PIN"
    assert row.status == "executed"

    # The step was handed what the read produced, in its prompt.
    assert len(seen) == 1
    assert "TP-482 Rate-limit the export endpoint" in seen[0]


@pytest.mark.asyncio
async def test_the_answer_is_readable_as_a_step_too(monkeypatch):
    """`{{var.<output_var>}}` is the name the step DECLARES;
    `{{steps.<id>.text}}` is what every other step exposes and what a
    spec author reaches for out of habit. Both work, and they are the
    same value."""
    uid = await _mk_user()
    spec = json.loads(json.dumps(_agent_spec().raw))
    spec["steps"][2]["params"]["text"] = (
        "{{var.ranked}}|{{steps.rank.text}}"
    )
    vspec = validate_spec(spec, REGISTRY_V2)
    a = await _mk_automation_v2(uid, vspec)
    _fake_thinking(monkeypatch, "ONE")

    await _fire(monkeypatch, uid, a, vspec)
    async with async_session_maker() as db:
        row = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
        )).scalar_one()
    assert json.loads(row.payload_json)["text"] == "ONE|ONE"


@pytest.mark.asyncio
async def test_the_run_never_mutates_the_specs_own_variables(monkeypatch):
    """The binding goes into the RUN's context. `vspec.variables` is the
    validated spec's own dict — writing the answer there would leak one
    run's conclusion into the next run parsed from the same object."""
    uid = await _mk_user()
    vspec = _agent_spec()
    a = await _mk_automation_v2(uid, vspec)
    _fake_thinking(monkeypatch, "leak me")
    await _fire(monkeypatch, uid, a, vspec)
    assert "ranked" not in vspec.variables
    assert "ranked" not in (vspec.raw.get("variables") or {})


# ── it is a STEP, everywhere a step is counted ───────────────────────


@pytest.mark.asyncio
async def test_the_thinking_is_a_visible_step_with_a_timing(monkeypatch):
    uid = await _mk_user()
    vspec = _agent_spec()
    a = await _mk_automation_v2(uid, vspec)
    _fake_thinking(monkeypatch)

    await _fire(monkeypatch, uid, a, vspec)
    job = await _one_run(a.id)

    # steps_json — what the phone's job card counts and draws.
    steps = json.loads(job.steps_json)
    assert [s["id"] for s in steps] == [
        "evaluate", "issues", "rank", "post", "record",
    ]
    rank = next(s for s in steps if s["id"] == "rank")
    assert rank["label"] in ("Thinking it through", "Thought it through")
    # The agent's own work, branded as the orb — not as a connector.
    assert rank["brand"] is None
    assert rank["started_at"] and rank["completed_at"]

    # The progress columns four other surfaces read back.
    assert job.progress_total == len(vspec.steps) + 1   # + narration

    # …and one thread turn, with a served sentence and a real timing.
    turns = await _turns(a.id)
    tools = [t for t in turns if t["kind"] == "tool"]
    thinking = [t for t in tools if t["action"] == "Thought it through"]
    assert len(thinking) == 1
    t = thinking[0]
    assert t["ok"] is True
    assert t["ms"] >= 0
    # No account is claimed: it called nothing.
    assert t["account_id"] == ""
    assert t["items"] == [] and t["actions"] == []


def test_the_thinking_sentence_comes_from_the_dictionary():
    """Nothing outside the verb dictionary composes step copy — and an
    action it can emit that `is_served_action` would refuse is a turn
    the ledger silently rewrites into a bare bubble."""
    from app.services import automation_verbs as verbs
    for ok in (True, False):
        act = verbs.engine_action("think", ok=ok)
        assert verbs.is_served_action(act["action"]), act
    # Total: an unknown phase is still served, never a raw phase name.
    unknown = verbs.engine_action("no_such_phase")
    assert verbs.is_served_action(unknown["action"])
    assert "no_such_phase" not in unknown["action"]
    assert verbs.live_sentence(None, None, phase="think") \
        == "thinking it through"


# ── determinism ──────────────────────────────────────────────────────


def test_the_prompt_is_byte_stable_over_the_same_facts():
    """A temperature-0 model is only deterministic over a byte-stable
    prompt, and the run context is assembled from dicts whose insertion
    order is an accident of which step ran first."""
    step = _agent_spec().steps[1]
    a = {"event": {"b": 2, "a": 1}, "var": {"z": "1", "y": "2"},
         "steps": {"two": {"ok": True, "text": "t2"},
                   "one": {"ok": True, "text": "t1", "count": 3}}}
    b = {"var": {"y": "2", "z": "1"}, "event": {"a": 1, "b": 2},
         "steps": {"one": {"count": 3, "text": "t1", "ok": True},
                   "two": {"text": "t2", "ok": True}}}
    assert agent_step.build_prompt("A", step, a) \
        == agent_step.build_prompt("A", step, b)


def test_the_prompt_carries_the_items_the_reads_produced():
    step = _agent_spec().steps[1]
    ctx = {"event": {}, "var": {}, "steps": {"issues": {
        "ok": True, "text": "TP-1 one\nTP-2 two", "count": 2,
        "lines": ["TP-1 one", "TP-2 two"],
        "raw_fields": [{"key": "TP-1"}],
    }}}
    prompt = agent_step.build_prompt("Ranked brief", step, ctx)
    # The prompt is the template, rendered — same rule as any params
    # value, so an author can point at an earlier step.
    assert "Rank these by what blocks someone: TP-1 one\nTP-2 two" in prompt
    assert '"lines"' in prompt and "TP-2 two" in prompt
    # `raw_fields` is the narrator's private material, not the model's.
    assert "raw_fields" not in prompt


def test_the_prompt_is_bounded():
    """A prompt whose length depends on how much a provider happened to
    return is not a stable prompt — and the ask is rendered against the
    SAME bounded view that is printed under it, so the model can never
    see two versions of one string."""
    step = _agent_spec().steps[1]
    ctx = {"event": {}, "var": {}, "steps": {"issues": {
        "ok": True, "text": "x" * 99_999,
        "lines": [f"line {i}" for i in range(500)],
        "raw_fields": [{"key": "TP-1"}],
    }}}
    prompt = agent_step.build_prompt("A", step, ctx)
    assert "x" * (agent_step.MAX_STEP_TEXT_CHARS + 1) not in prompt
    assert f"line {agent_step.MAX_STEP_ITEMS}" not in prompt
    assert len(prompt) < 20_000


@pytest.mark.asyncio
async def test_the_thinking_runs_at_temperature_zero(monkeypatch):
    """Same reasoning as the narrator's pin (rec2): a judgement the
    user acts on must be the same judgement for the same facts."""
    seen = {}

    class _Svc:
        async def complete_with_json(self, *, messages, model,
                                     temperature, max_tokens):
            seen["temperature"] = temperature
            seen["model"] = model
            return '{"text": "done"}'

    monkeypatch.setattr(
        "app.services.llm_service.get_llm_service", lambda: _Svc())
    out = await agent_step._default_complete("p")
    assert seen["temperature"] == 0.0
    assert seen["model"]           # never model=None on a background path
    assert out == "done"


@pytest.mark.asyncio
async def test_prose_instead_of_the_envelope_is_still_the_answer(monkeypatch):
    class _Svc:
        async def complete_with_json(self, *, messages, model,
                                     temperature, max_tokens):
            return "```\nplain prose\n```"

    monkeypatch.setattr(
        "app.services.llm_service.get_llm_service", lambda: _Svc())
    assert await agent_step._default_complete("p") == "plain prose"


@pytest.mark.asyncio
async def test_an_empty_answer_is_a_failure_not_an_empty_binding():
    """Binding "" would put a hole in whatever interpolates it, and
    only the step's on_error is entitled to decide that is acceptable."""
    step = _agent_spec().steps[1]

    async def _nothing(_prompt):
        return "   "

    with pytest.raises(agent_step.AgentStepError):
        await agent_step.run_agent_step(
            automation=type("A", (), {"name": "A"})(), step=step,
            ctx={"steps": {}}, complete=_nothing,
        )


# ── failure ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failed_thinking_step_stops_the_run_by_default(monkeypatch):
    """`fail` is the agent step's default, and the opposite of a read's,
    on purpose: its answer is interpolated, so a swallowed failure does
    not omit a section — it posts a hole."""
    uid = await _mk_user()
    vspec = _agent_spec()
    a = await _mk_automation_v2(uid, vspec)
    _fake_thinking(monkeypatch, fail=RuntimeError("the model refused"))

    assert await _fire(monkeypatch, uid, a, vspec) == "failed"

    job = await _one_run(a.id)
    assert job.status == "failed" and job.outcome == "step_failed"
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
        )).scalars().all()
    assert rows == []                      # nothing staged, nothing sent

    turns = await _turns(a.id)
    failed = [t for t in turns if t["kind"] == "tool" and not t["ok"]]
    assert [t["action"] for t in failed] == ["Could not think it through"]


@pytest.mark.asyncio
async def test_a_skipped_thinking_step_is_never_a_failed_account(monkeypatch):
    """`failed_sources` is a list of ACCOUNTS — every consumer treats it
    as one: the needs-you cards, `accounts_failed`, the reconnect
    buttons, the per-source resume. A thinking step has none, so its
    failure is recorded as its own turn and nowhere else."""
    uid = await _mk_user()
    spec = json.loads(json.dumps(_agent_spec().raw))
    spec["steps"][1]["on_error"] = "continue"
    vspec = validate_spec(spec, REGISTRY_V2)
    a = await _mk_automation_v2(uid, vspec)
    _fake_thinking(monkeypatch, fail=RuntimeError("the model refused"))

    assert await _fire(monkeypatch, uid, a, vspec) == "run"

    job = await _one_run(a.id)
    cfg = job.config_json or {}
    assert cfg.get("steps_partial") is True
    assert not cfg.get("accounts_failed")
    assert not cfg.get("failed_sources")

    turns = await _turns(a.id)
    assert not [t for t in turns if t["kind"] == "needs_you"]
    # The write still ran — and the hole is honest: the name resolves to
    # nothing rather than to a sentence nobody produced.
    async with async_session_maker() as db:
        row = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
        )).scalar_one()
    assert json.loads(row.payload_json)["text"] == ""


# ── the narration seam ───────────────────────────────────────────────


def test_the_narrator_is_told_thinking_is_not_an_account():
    """The model meets a step with no connector, no items and a detail
    full of prose. The only shape it otherwise has for that is "an
    account said this"."""
    from app.agent.automations import narrator
    record = {
        "automation": {"title": "Ranked brief", "mode": "auto"},
        "run_kind": "scheduled", "vocabulary": "brief",
        "status": "completed", "rules": [], "memory_facts": [],
        "steps": [
            {"step_ref": "issues", "connector_name": "Jira",
             "account_id": "jira", "tool_kind": "read", "action": "Checked",
             "detail": "", "ok": True, "failure_reason": None,
             "items": [], "write": None},
            {"step_ref": "rank", "connector_name": "", "account_id": "",
             "tool_kind": "agent", "action": "Thought it through",
             "detail": "TP-482 first.", "ok": True,
             "failure_reason": None, "items": [], "write": None},
        ],
    }
    prompt = narrator.build_prompt(record)
    assert "YOUR OWN THINKING" in prompt
    # Nothing failed, so no failure guidance is dragged in.
    assert "SOME SOURCES FAILED" not in prompt

    record["steps"][1]["ok"] = False
    failed = narrator.build_prompt(record)
    # A thinking step that broke is NOT a source: the "name the account
    # every time" rule would make the model invent one.
    assert "SOME SOURCES FAILED" not in failed
    assert "no account failed, so name none" in failed


def test_a_thinking_step_owes_the_narrator_no_annotate():
    """`validate_drafts` demands one annotate per step that produced
    items, and rejects a turn list that misses one. An agent step
    produces no items — it must not make every narration invalid."""
    from app.agent.automations import narrator
    record = {
        "automation": {"title": "Ranked brief", "mode": "auto"},
        "run_kind": "scheduled", "vocabulary": "brief",
        "status": "completed", "rules": [], "memory_facts": [],
        "steps": [
            {"step_ref": "rank", "connector_name": "", "account_id": "",
             "tool_kind": "agent", "action": "Thought it through",
             "detail": "TP-482 first.", "ok": True,
             "failure_reason": None, "items": [], "write": None},
        ],
    }
    problems = narrator.validate_drafts(
        [{"kind": "agent", "text": "I looked at your morning."}], record,
    )
    assert problems == []


@pytest.mark.asyncio
async def test_the_narration_record_carries_what_the_thinking_produced(
    monkeypatch,
):
    """The narration must be able to build on the run's own earlier
    conclusion instead of re-deriving it from the raw items — and
    reaching a different one, at temperature 0, from a different
    starting point."""
    from app.agent.automations import executor_v2

    captured = {}

    async def _narrate(record, *, complete=None):
        captured["record"] = record
        return {"turns": [], "problems": [], "attempts": 1}

    monkeypatch.setattr(
        "app.agent.automations.narrator.narrate_run", _narrate)

    uid = await _mk_user()
    vspec = _agent_spec()
    a = await _mk_automation_v2(uid, vspec)
    _fake_thinking(monkeypatch, "TP-482 first — it blocks the export.")
    await _fire(monkeypatch, uid, a, vspec)

    steps = {s["step_ref"]: s for s in captured["record"]["steps"]}
    assert steps["rank"]["tool_kind"] == "agent"
    assert steps["rank"]["detail"] == "TP-482 first — it blocks the export."
    assert steps["rank"]["account_id"] == ""
    assert steps["rank"]["connector_name"] == ""
    assert executor_v2 is not None


# ── backwards compatibility ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_spec_with_no_agent_step_behaves_identically(monkeypatch):
    """The guarantee this round owes every automation already running.

    The seam is replaced by something that RAISES: if the executor so
    much as reaches for it on a spec that declares no agent step, this
    fails.
    """
    async def _never(**_kw):
        raise AssertionError("the agent seam was entered for a plain spec")

    monkeypatch.setattr(agent_step, "run_agent_step", _never)

    uid = await _mk_user()
    vspec = _plain_spec()
    a = await _mk_automation_v2(uid, vspec)

    assert await _fire(monkeypatch, uid, a, vspec) == "run"

    # The persisted spec never grows a `kind`.
    assert all("kind" not in s for s in vspec.raw["steps"])

    job = await _one_run(a.id)
    assert [s["id"] for s in json.loads(job.steps_json)] == [
        "evaluate", "issues", "post", "record",
    ]
    assert json.loads(job.steps_json)[1]["brand"] == "jira"
    assert job.status == "completed" and job.outcome == "sent"
    assert job.progress_total == len(vspec.steps) + 1

    async with async_session_maker() as db:
        row = (await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.automation_id == a.id)
        )).scalar_one()
    assert row.idempotency_key.endswith(":w0")
    assert json.loads(row.payload_json) == {
        "channel": "C-PIN",
        "text": "TP-482 Rate-limit the export endpoint\n"
                "TP-476 Flaky memverify test",
    }

    turns = await _turns(a.id)
    tools = [t for t in turns if t["kind"] == "tool"]
    # One read turn and one write turn — no third turn appeared, and no
    # engine action leaked into a run that has no thinking step.
    assert [t["tool_kind"] for t in tools] == ["read", "write"]
    assert all(t["action"] != "Thought it through" for t in tools)
    assert [t["account_id"] for t in tools] == ["jira", "slack"]


def _capture_activity(monkeypatch) -> list[tuple]:
    from app.agent.automations import ledger

    frames: list[tuple] = []

    async def _emit(user_id, *, automation_id, thread_id, phase,
                    run_id=None, tool=None, detail=None):
        frames.append((phase, tool, detail))

    monkeypatch.setattr(ledger, "emit_activity", _emit)
    return frames


@pytest.mark.asyncio
async def test_a_plain_run_puts_the_same_frames_on_the_wire(monkeypatch):
    """`open_run` has always opened with one bare `thinking` frame. A
    plain run gains no second one, and every step it announces is still
    a `tool` frame carrying its connector."""
    frames = _capture_activity(monkeypatch)
    uid = await _mk_user()
    vspec = _plain_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec)

    thinking = [f for f in frames if f[0] == "thinking"]
    assert thinking == [("thinking", None, None)]
    tool_frames = [f for f in frames if f[0] == "tool"]
    assert [f[1]["account_id"] for f in tool_frames] == ["jira"]


@pytest.mark.asyncio
async def test_an_agent_run_announces_thinking_not_a_tool(monkeypatch):
    frames = _capture_activity(monkeypatch)
    uid = await _mk_user()
    vspec = _agent_spec()
    a = await _mk_automation_v2(uid, vspec)
    _fake_thinking(monkeypatch)
    await _fire(monkeypatch, uid, a, vspec)

    # The run's opening frame, then the step's own — and the step's says
    # what it is doing. No connector glyph: naming one would be a lie
    # about what is happening.
    assert [f for f in frames if f[0] == "thinking"] == [
        ("thinking", None, None),
        ("thinking", None, "thinking it through"),
    ]
    assert [f[1]["account_id"] for f in frames if f[0] == "tool"] == ["jira"]


def test_the_canvas_names_the_step_without_inventing_an_account():
    """`turn_action("", "")` answers "Checked the account" — about a
    step that has no account. The steps sheet derives from the raw spec,
    so it needs the same branch the engine has."""
    from app.agent.automations import workflow
    raw = _agent_spec().raw
    automation = type("A", (), {"steps_human_json": None})()
    steps = workflow._steps_human(automation, raw)
    assert [s["n"] for s in steps] == [1, 2, 3]
    # R39: the steps SHEET is the plan ("WHAT IT DOES, IN ORDER"), so the
    # verb dictionary's past tense is re-tensed at this read boundary —
    # `verbs.plan_action`. The pin this test exists for is the line below:
    # the step must not be named after an account it does not have.
    assert steps[1]["text"] == "Thinks it through"
    assert steps[1]["sub"].startswith("Rank these by what blocks someone")
    assert "account" not in steps[1]["text"].lower()
