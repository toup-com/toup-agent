"""Waiting on the user is not failing, and the agent had no way to say so.

Reported 2026-08-13: a job went RED at the exact moment the chat put a
confirmation card in front of the user asking them to approve something.
Two surfaces, side by side, saying opposite things — and the one the user
believes is the red one, so the product reads as broken while it is
working exactly as designed.

The cause was a missing word, not missing machinery. Every piece of the
parked presentation already shipped:

    job_status.STATUS_WAITING_ON_USER      the value
    job_status.PARKED_STATUSES             not-active, not-terminal
    BuildJobCard (web)                     amber "Waiting on you" chip
    JobDetailScreen (mobile)               `isWaiting`, not `isFailed`
    subagent_orchestrator.notify_job_needs_user   keeps the LA card alive

...and the model could not reach any of it, because `update_job`'s
input_schema enum was ["running", "completed", "failed"] and
`create_job`'s CONTRACT sentence named "mark it 'failed'" as the exit for
work it could not finish. An enum is not advice; it is the set of values
the provider will accept. So the model did the only legal thing.

Four layers are asserted here, deliberately, because the first three are
each individually defeatable:

  1. the vocabulary   — the enum, and the contract that no longer says
                        "failed" for blocked work
  2. the coercion     — `update_job` rewrites failed -> waiting_on_user
                        while a card is outstanding, so a model that
                        reaches for `failed` anyway cannot land it
  3. the backstop     — the turn-end finalizer parks instead of writing
                        `completed`, so a turn that says NOTHING about
                        its job does not report a green "Done" for an
                        email that was never sent
  4. the closers      — approval resolves the park (the whole point:
                        green after the user approves), and an ignored
                        card expires it, because `waiting_on_user` has no
                        other closer and an immortal "Waiting on you" is
                        just a politer lie
"""

from __future__ import annotations

import inspect
import json
import os
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("ENVIRONMENT", "test")


# ── 1. Vocabulary ────────────────────────────────────────────────────


def _tool(name: str) -> dict:
    from app.agent.tool_definitions import get_agent_tools, get_extended_tools

    for t in list(get_agent_tools()) + list(get_extended_tools()):
        if t.get("name") == name:
            return t
    raise AssertionError(f"{name} is not in the tool definitions")


def test_update_job_can_express_waiting_on_the_user():
    """The bug in one assertion: without this value in the enum the model
    is structurally incapable of parking a job, whatever any prompt says."""
    enum = _tool("update_job")["input_schema"]["properties"]["status"]["enum"]
    assert "waiting_on_user" in enum, (
        f"update_job cannot express 'blocked on the user'; enum is {enum}. "
        f"The model's only remaining exit is 'failed'."
    )


def test_the_parked_status_the_enum_offers_is_the_one_the_clients_render():
    """A near-miss spelling ('waiting', 'blocked') would pass the test
    above and render as an unknown status on both clients."""
    from app.agent.job_status import PARKED_STATUSES, STATUS_WAITING_ON_USER

    enum = _tool("update_job")["input_schema"]["properties"]["status"]["enum"]
    assert STATUS_WAITING_ON_USER in enum
    assert STATUS_WAITING_ON_USER in PARKED_STATUSES


def test_create_jobs_contract_no_longer_routes_blocked_work_to_failed():
    """The contract sentence is what the model reads when deciding how to
    end a turn. It used to offer exactly three exits, one of which was
    'failed', for a job it could not complete."""
    desc = _tool("create_job")["description"]
    assert "waiting_on_user" in desc, (
        "create_job's CONTRACT never mentions the parked status, so the "
        "model has no reason to believe it may use it"
    )
    lowered = desc.lower()
    assert "waiting on someone is not failing" in lowered, (
        "the contract states the option but not the rule; the model needs "
        "to know which one to pick, not merely that both exist"
    )


def test_waiting_is_not_terminal_so_a_parked_job_is_still_live_work():
    from app.agent.job_status import (
        ACTIVE_STATUSES, STATUS_WAITING_ON_USER, TERMINAL_STATUSES,
    )

    assert STATUS_WAITING_ON_USER not in TERMINAL_STATUSES
    assert STATUS_WAITING_ON_USER not in ACTIVE_STATUSES


def test_the_parked_verdict_asks_the_user_to_act_and_never_reads_as_a_crash():
    from app.agent.job_status import (
        DISPOSITION_NEEDS_USER, ERR_AWAITING_CONFIRMATION, awaiting_confirmation,
    )

    v = awaiting_confirmation()
    assert v.error_class == ERR_AWAITING_CONFIRMATION
    assert v.disposition == DISPOSITION_NEEDS_USER, (
        "a NEEDS_USER disposition is what stops every retry layer from "
        "re-running work that only a human can unblock"
    )
    assert v.required_action, "the client CTA has nothing to render"
    low = (v.user_message or "").lower()
    assert low and "error" not in low and "fail" not in low, v.user_message


# ── 2. Coercion at the single write point ────────────────────────────


class _FakeJob:
    def __init__(self):
        self.id = "job-1"
        self.title = "Send the invoice"
        self.status = "running"
        self.steps_json = "[]"
        self.error_message = None
        self.error_class = None
        self.user_message = None
        self.completed_at = "SENTINEL"
        self.config_json = None


class _FakeDB:
    def __init__(self, job):
        self._job = job
        self.added = []

    async def get(self, _model, _id):
        return self._job

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


async def _run_update(monkeypatch, tools, inp, job=None):
    """Drive the real `_tool_update_job` against a fake session."""
    import app.db.database as dbmod
    from app.agent import subagent_orchestrator as so

    job = job or _FakeJob()
    monkeypatch.setattr(dbmod, "async_session_maker", lambda: _FakeDB(job))

    seen: list[dict] = []

    async def _fake_notify(**kw):
        seen.append({"kind": kw.get("kind"), **kw})

    async def _fake_needs_user(**kw):
        seen.append({"kind": "needs_user", **kw})

    monkeypatch.setattr(so, "_notify_job_event", _fake_notify)
    monkeypatch.setattr(so, "notify_job_needs_user", _fake_needs_user)

    import app.api.ws_chat as wsc

    async def _fake_broadcast(*a, **kw):
        return None

    monkeypatch.setattr(wsc, "broadcast_to_user", _fake_broadcast)

    out = await tools._tool_update_job(inp)
    return job, seen, json.loads(out)


def _executor():
    from app.agent.tool_executor import ToolExecutor

    ex = ToolExecutor.__new__(ToolExecutor)
    ex.staged_pending_action_ids = []
    return ex


@pytest.mark.asyncio
async def test_failed_is_coerced_to_waiting_while_a_card_is_outstanding(monkeypatch):
    """The load-bearing guard. Widening the enum is persuasion — a model
    that just read '[confirmation_required] NOT DONE YET' and reaches for
    `failed` is being reasonable, and the user still sees red."""
    ex = _executor()
    ex.staged_pending_action_ids = ["act-9"]
    job, _seen, res = await _run_update(
        monkeypatch, ex,
        {"job_id": "job-1", "status": "failed", "error_message": "could not send"},
    )
    assert job.status == "waiting_on_user", (
        "a job blocked on a confirmation card was still written as failed"
    )
    assert res["status"] == "waiting_on_user"


@pytest.mark.asyncio
async def test_a_genuine_failure_with_no_card_outstanding_still_fails(monkeypatch):
    """The coercion must not swallow real failures — that would be the
    same class of lie pointed the other way."""
    ex = _executor()
    job, _seen, _res = await _run_update(
        monkeypatch, ex,
        {"job_id": "job-1", "status": "failed", "error_message": "boom"},
    )
    assert job.status == "failed"
    assert job.error_message == "boom"


@pytest.mark.asyncio
async def test_the_coercion_drops_the_models_error_text(monkeypatch):
    """`error_message` is still rendered by older clients. Keeping
    'could not send' on a parked job reintroduces the failure reading in
    the one field built to shout it."""
    ex = _executor()
    ex.staged_pending_action_ids = ["act-9"]
    job, _seen, _res = await _run_update(
        monkeypatch, ex,
        {"job_id": "job-1", "status": "failed", "error_message": "could not send"},
    )
    assert not job.error_message, job.error_message


@pytest.mark.asyncio
async def test_a_parked_job_is_not_stamped_completed_at(monkeypatch):
    """`completed_at` is what History filters on. A parked job carrying
    one appears in History — the terminal-only tab — while still live."""
    ex = _executor()
    ex.staged_pending_action_ids = ["act-9"]
    job, _seen, _res = await _run_update(
        monkeypatch, ex, {"job_id": "job-1", "status": "waiting_on_user"},
    )
    assert job.completed_at is None, job.completed_at


@pytest.mark.asyncio
async def test_a_parked_job_records_which_card_is_holding_it(monkeypatch):
    """Without this the resume path can see an approval land and have no
    way to know which job it just unblocked."""
    ex = _executor()
    ex.staged_pending_action_ids = ["act-9"]
    job, _seen, _res = await _run_update(
        monkeypatch, ex, {"job_id": "job-1", "status": "waiting_on_user"},
    )
    assert (job.config_json or {}).get("pending_action_id") == "act-9"


@pytest.mark.asyncio
async def test_a_parked_job_keeps_its_live_activity_alive(monkeypatch):
    """`mission_failed` alerts and then sends `event=end`, tearing the
    card down. The card IS the user's prompt to act — ending it deletes
    the only thing that would get the job unblocked."""
    ex = _executor()
    ex.staged_pending_action_ids = ["act-9"]
    _job, seen, _res = await _run_update(
        monkeypatch, ex, {"job_id": "job-1", "status": "waiting_on_user"},
    )
    kinds = [s["kind"] for s in seen]
    assert "needs_user" in kinds, kinds
    assert "mission_failed" not in kinds, kinds
    assert "progress" not in kinds, (
        f"a parked job fell through to the progress lane, which draws "
        f"'Working on…' for work that is stopped: {kinds}"
    )


# ── 3. The turn-end backstop ─────────────────────────────────────────


def test_the_turn_end_finalizer_parks_instead_of_reporting_done():
    """A turn that stages a card and then says nothing about its job used
    to have the job closed `completed` — a green "Done" for an email that
    was never sent. That is worse than the red one: red at least stops the
    user believing it went out."""
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._run_inner)
    assert "staged_pending_action_ids" in src, (
        "the finalizer never asks whether a card is outstanding, so it "
        "closes every job `completed` including the parked ones"
    )
    park = src.index("_park = bool(")
    values = src.index('status="completed", completed_at=_now')
    assert park < values, (
        "the completed-values dict is built before _park is decided — the "
        "branch cannot influence what is written"
    )


def test_the_finalizer_reads_a_field_that_save_messages_does_not_consume():
    """`_save_messages` captures AND CLEARS `_last_pending_action`, and it
    runs BEFORE the finalizer. A finalizer reading that attribute would
    see None every time: a guard that looks right, tests green on its
    unit, and is dead in production."""
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._run_inner)
    fin = src.index("_created_job_ids = self.tools.take_created_job_ids")
    tail = src[fin:]
    assert "_last_pending_action" not in tail, (
        "the finalizer reads the consumed attribute; _save_messages has "
        "already set it to None by the time this runs"
    )


class _CapturingDB:
    """Fake session that records the UPDATE statements it is handed."""

    def __init__(self, job):
        self.stmts: list = []
        self._job = job

    async def execute(self, stmt):
        self.stmts.append(stmt)

        class _Res:
            @staticmethod
            def first():
                return ("job-1", "Schedule verification call")

        return _Res()

    async def get(self, _model, _id):
        return self._job

    async def commit(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _drive_interrupted_sweep(staged):
    """Run the real `_close_interrupted_jobs` against a fake session."""
    import asyncio

    import app.db.database as dbmod
    from app.agent import subagent_orchestrator as so
    from app.agent.agent_runner import AgentRunner
    import app.api.ws_chat as wsc

    job = _FakeJob()
    db = _CapturingDB(job)
    seen: list[dict] = []

    async def _fake_notify(**kw):
        seen.append({"kind": kw.get("kind"), **kw})

    async def _fake_needs_user(**kw):
        seen.append({"kind": "needs_user", **kw})

    async def _fake_broadcast(*a, **kw):
        return None

    orig = (
        dbmod.async_session_maker, so._notify_job_event,
        so.notify_job_needs_user, wsc.broadcast_to_user,
    )
    dbmod.async_session_maker = lambda: db
    so._notify_job_event = _fake_notify
    so.notify_job_needs_user = _fake_needs_user
    wsc.broadcast_to_user = _fake_broadcast
    try:
        runner = AgentRunner.__new__(AgentRunner)
        asyncio.run(runner._close_interrupted_jobs(
            ("job-1",), "user-1", staged_action_id=staged,
        ))
    finally:
        (dbmod.async_session_maker, so._notify_job_event,
         so.notify_job_needs_user, wsc.broadcast_to_user) = orig
    return db.stmts, seen, job


def test_an_ordinary_interrupted_turn_still_cancels():
    """Control. The parked branch must not swallow the case it was built
    beside — a turn that died with no card outstanding is genuinely over,
    and `cancelled` / `turn_interrupted` is the honest word for it."""
    stmts, seen, _job = _drive_interrupted_sweep(staged=None)
    params = stmts[0].compile().params
    assert params["status"] == "cancelled", params["status"]
    assert params["error_class"] == "turn_interrupted", params["error_class"]
    assert params["completed_at"] is not None
    assert [s["kind"] for s in seen] == ["mission_failed"]


def test_a_turn_that_DIES_holding_a_card_parks_too():
    """The second half of the production evidence, and a gap the first
    version of this fix had.

    On the reporting account, 2026-08-14, 60 seconds apart:

        00:49:26  Check recent emails       completed   (no card)
        00:50:17  Send verification email   FAILED      <- model wrote it
        00:51:12  Schedule verification call CANCELLED  <- turn died

    The third job took the `finally` sweep, not the happy-path finalizer,
    and that sweep writes `cancelled` / `turn_interrupted` unconditionally.
    Its `calendar__create_event` card — staged four seconds later — was
    still `pending` an hour on, i.e. tappable, i.e. the job was waiting,
    not cancelled. Both clients render `cancelled` in their `isFailed`
    branch, so it looked exactly as broken as the `failed` one.

    Asserted behaviourally, on the values the UPDATE actually binds. A
    substring probe is not enough here: `STATUS_WAITING_ON_USER if parked`
    appears twice in this function (the UPDATE and the WS broadcast), so
    reverting the UPDATE alone leaves the string in place and a
    source-grep passes on the broken code. Both mutations proved it.
    """
    _stmts, seen, _job = _drive_interrupted_sweep(staged="act-9")
    params = _stmts[0].compile().params
    assert params["status"] == "waiting_on_user", (
        f"the interrupted-turn sweep wrote {params['status']!r} while a "
        f"confirmation card was outstanding"
    )
    assert params["completed_at"] is None, params["completed_at"]
    assert params["error_class"] == "awaiting_confirmation", params["error_class"]

    kinds = [s["kind"] for s in seen]
    assert kinds == ["needs_user"], (
        f"a job parked by this path got {kinds} — a terminal push tears "
        f"down the very card the user has to tap"
    )


def test_the_interrupted_sweep_reads_the_staged_ids_without_awaiting():
    """That sweep runs inside a `finally` reached *because the task is
    being cancelled*; an await there re-raises CancelledError immediately
    and the cleanup is skipped by the very condition that requires it."""
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._sweep_unclosed_created_jobs)
    assert "staged_pending_action_ids" in src
    read = src.index("staged_pending_action_ids")
    spawn = src.index("_spawn_bg(")
    assert read < spawn, "the staged ids are read after the work is handed off"
    assert "await " not in src, (
        "the synchronous sweep gained an await — it will be skipped by the "
        "cancellation that makes it necessary"
    )


def test_a_job_parked_by_the_interrupted_path_is_still_resumable():
    """Otherwise approving the card leaves it parked forever: the resume
    path matches on `config_json.pending_action_id`."""
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._close_interrupted_jobs)
    assert 'cfg["pending_action_id"] = staged_action_id' in src


def test_save_messages_still_consumes_the_other_field_and_not_the_new_one():
    """Control for the test above: proves the consume-and-clear it warns
    about is real, and that the new list is NOT swept up by it."""
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._save_messages)
    assert "self.tools._last_pending_action = None" in src
    assert "staged_pending_action_ids" not in src


# ── 4. The closers ───────────────────────────────────────────────────


def test_every_card_outcome_maps_to_a_terminal_job_status():
    from app.api.agent import _ACTION_OUTCOME_TO_JOB_STATUS
    from app.agent.job_status import TERMINAL_STATUSES

    assert set(_ACTION_OUTCOME_TO_JOB_STATUS) == {
        "executed", "failed", "rejected", "expired",
    }, "a card outcome with no mapping leaves its job parked forever"
    for outcome, st in _ACTION_OUTCOME_TO_JOB_STATUS.items():
        assert st in TERMINAL_STATUSES, (outcome, st)


def test_approval_is_what_turns_the_job_green():
    """The half of the report that is not about copy: after the user
    approves, the job has to finish."""
    from app.api.agent import _ACTION_OUTCOME_TO_JOB_STATUS

    assert _ACTION_OUTCOME_TO_JOB_STATUS["executed"] == "completed"


def test_saying_no_cancels_rather_than_fails():
    """Declining is a decision, not a breakage."""
    from app.api.agent import _ACTION_OUTCOME_TO_JOB_STATUS

    assert _ACTION_OUTCOME_TO_JOB_STATUS["rejected"] == "cancelled"
    assert _ACTION_OUTCOME_TO_JOB_STATUS["expired"] == "cancelled"


def test_approve_and_reject_both_resume_the_job():
    """Source-order probe: the resume call has to sit after the commit
    that records THIS card's outcome. The card's own row is the record of
    what happened, and it must be durable before anything downstream is
    told about it.

    Anchored to the outcome write rather than to any `db.commit()`.
    `approve_pending_action` commits three times (lazy-expire, the claim,
    the outcome), so both `index` and `rindex` on a bare commit are
    satisfiable by an unrelated earlier write — a mutation that DELETED
    the outcome commit outright still passed both.
    """
    from app.api import connector_pending_actions as cpa

    cases = [
        (cpa.approve_pending_action, "result_json=json.dumps(result_payload"),
        (cpa.reject_pending_action, 'status="rejected",'),
    ]
    for fn, outcome_anchor in cases:
        src = inspect.getsource(fn)
        assert "_resume_job_for_action" in src, (
            f"{fn.__name__} decides a card and never unblocks the job "
            f"parked on it"
        )
        write = src.index(outcome_anchor)
        resume = src.index("await _resume_job_for_action")
        try:
            commit = src.index("await db.commit()", write)
        except ValueError:
            raise AssertionError(
                f"{fn.__name__} never commits the card's outcome before "
                f"resuming the job"
            )
        assert write < commit < resume, (
            f"{fn.__name__} resumes the job before its own outcome is "
            f"durable (write={write} commit={commit} resume={resume})"
        )


def test_the_resume_hop_can_never_fail_the_users_approval():
    """The send already happened. A tenant agent that is asleep,
    mid-rollout or unreachable must not turn it into a 5xx."""
    from app.api import connector_pending_actions as cpa

    src = inspect.getsource(cpa._resume_job_for_action)
    assert "except Exception" in src
    assert "raise" not in src


def test_an_ignored_card_eventually_releases_its_job():
    """`waiting_on_user` has no other closer: the stalled sweep selects
    queued/running only, and the turn-end finalizer touches running only.
    A card nobody answers is the common case, and an immortal 'Waiting on
    you' is the same lie in a nicer word."""
    from app.agent import job_reaper

    assert hasattr(job_reaper, "sweep_expired_card_parks")
    src = inspect.getsource(job_reaper.sweep_stalled_jobs)
    assert "sweep_expired_card_parks" in src, (
        "the park sweep exists but nothing runs it on the loop"
    )


def test_the_park_timeout_outlives_the_card_it_waits_on():
    """If the job dies first, the user approves a still-tappable card and
    the mail goes out against a job already marked cancelled."""
    from app.agent.job_reaper import PARKED_ON_CARD_STALE_AFTER
    from app.services.connector_dispatcher import _PENDING_ACTION_TTL_HOURS

    assert PARKED_ON_CARD_STALE_AFTER > timedelta(hours=_PENDING_ACTION_TTL_HOURS), (
        f"park timeout {PARKED_ON_CARD_STALE_AFTER} does not outlive the "
        f"card's {_PENDING_ACTION_TTL_HOURS}h TTL"
    )


def test_the_park_sweep_only_touches_confirmation_card_parks():
    """Jobs parked for credits or connector auth have never had a timeout.
    Giving them one here would be an unrelated behaviour change smuggled
    in behind a copy fix."""
    from app.agent import job_reaper

    src = inspect.getsource(job_reaper.sweep_expired_card_parks)
    assert "ERR_AWAITING_CONFIRMATION" in src
    assert "BuildJob.error_class ==" in src, (
        "the sweep does not filter by error_class, so it reaps every "
        "parked job including the credit ones"
    )


def test_the_stalled_reaper_still_ignores_parked_rows():
    """Control: the 30-minute sweep must not start eating parked jobs now
    that they are common."""
    from app.agent import job_reaper

    src = inspect.getsource(job_reaper.sweep_stalled_jobs)
    assert 'BuildJob.status.in_(("queued", "running"))' in src
