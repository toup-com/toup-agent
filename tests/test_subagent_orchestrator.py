"""Sub-agent orchestrator — Phase 4 end-to-end.

What we test
------------
The orchestrator function ``spawn_subagent`` and its background
``_run_child`` lifecycle, end-to-end against the test DB for
BuildJob rows. The child's ``agent_runner.run`` is mocked (we're
not testing agent_runner here; we're testing that the orchestrator
wires everything together correctly).

``write_subagent_message`` + ``broadcast_subagent_message`` are
mocked because the ``messages`` table has a pgvector column that
SQLite skips during ``init_db`` (see
``test_subagent_message_writer.py`` for the why). The mocked
writer captures the kwargs the orchestrator passed in so we can
assert on shape; the real writer is covered by its own unit
tests.

What we pin
-----------
- Kill switch off → rejection bubbles out from pre_spawn_checks.
- Kill switch on + valid task → BuildJob row created with
  job_type='subagent', source_kind='subagent', source_id=parent,
  config_json populated, idempotency_key set.
- Child run is fired via asyncio.create_task; spawn_subagent
  returns without waiting for it. Pinned by holding the child on an
  asyncio.Event rather than by timing the call: a blocking
  implementation cannot reach the assertions at all, which is true
  at any machine speed. The former "under 500ms with a 1s child"
  bound was a coin-flip under CI load and is gone.
- On child success: writer called with outcome='success', WS
  broadcast fires, summary_message_id stamped, status='completed'
  / outcome='success'.
- On child timeout: writer called with outcome='timeout',
  status='timeout', error_message populated.
- On child raising: status='failed', outcome='failed', error
  text captured.
- Retried spawn with same task collapses to the existing job_id;
  only ONE child run is fired.
- agent_runner.run is called with the expected sub-agent kwargs
  (prompt_profile=SUBAGENT, save_assistant_message=False,
  disable_post_processing=True, save_user_message=False,
  channel='subagent', session_id='subagent:<id>').
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import pytest
import pytest_asyncio


# ──────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────


@dataclass
class FakeAgentResponse:
    text: str
    session_id: str = "subagent:fake"
    input_tokens: int = 100
    output_tokens: int = 50
    model_used: str = "claude-haiku-4-5"


class FakeAgentRunner:
    """Stands in for the real runner.

    `gate` is the deterministic alternative to `delay`. A test that needs to
    observe the child *while it is still running* should hold it on an
    unset `gate` rather than race a `delay` against a wall clock: with a gate
    the child provably has not finished, at any machine speed and under any
    CI load. `delay` remains for tests that only need "this takes a moment".
    """

    def __init__(self, *, response_text="result", raise_exc=None, delay=0.0,
                 gate: "asyncio.Event | None" = None):
        self.calls: list[dict[str, Any]] = []
        self._response_text = response_text
        self._raise_exc = raise_exc
        self._delay = delay
        self._gate = gate
        self.started = asyncio.Event()
        self.finished = asyncio.Event()

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        try:
            if self._gate is not None:
                await self._gate.wait()
            elif self._delay:
                await asyncio.sleep(self._delay)
            if self._raise_exc:
                raise self._raise_exc
            return FakeAgentResponse(text=self._response_text)
        finally:
            self.finished.set()


@pytest_asyncio.fixture
async def db():
    from app.db.database import async_session_maker
    async with async_session_maker() as s:
        yield s


@pytest.fixture
def enable_spawning(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "subagent_spawning_enabled", True)
    return settings


@pytest.fixture(autouse=True)
def reset_lane_manager(monkeypatch):
    """LaneManager is a module-level singleton (lanes.py:147-154).
    Previous tests' lane runs accumulate in its _runs dict and the
    semaphore state survives across tests. Reset between tests so
    each starts fresh."""
    import app.agent.lanes as lanes
    lanes._lane_manager = None
    yield
    lanes._lane_manager = None


@pytest.fixture
def patch_writer(monkeypatch):
    """Stub out the announce-back path. ``messages`` table is
    pgvector-only and SQLite skips it in this env. Capture the
    kwargs so we can assert on shape; return synthetic id/dc."""
    calls: dict[str, Any] = {"write": [], "broadcast": []}

    async def _fake_write(db, **kwargs):
        calls["write"].append(kwargs)
        return ("msg-" + str(uuid.uuid4())[:8], "dc-fake")

    async def _fake_broadcast(user_id, **kwargs):
        calls["broadcast"].append({"user_id": user_id, **kwargs})
        return {"ws_count": 1, "channel_results": {"website": {"status": "delivered"}}}

    import app.agent.subagent_message_writer as mw

    monkeypatch.setattr(mw, "write_subagent_message", _fake_write)
    monkeypatch.setattr(mw, "broadcast_subagent_message", _fake_broadcast)
    return calls


async def _seed_user(db, user_id: str) -> None:
    from app.db.models import User
    from sqlalchemy import select
    existing = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if existing is None:
        db.add(User(
            id=user_id,
            email=f"{user_id[:8]}@test.local",
            hashed_password="x" * 60,
            name="Test",
        ))
        await db.commit()


async def _wait_for_completion(job_id: str, *, timeout: float = 30.0):
    """Poll until the job row reaches a terminal status.

    The deadline is a deadlock backstop, NOT a performance assertion — so it is
    generous. A tight bound here measures how loaded the CI runner is rather
    than whether the orchestrator works, and the sweep runs many pytest
    processes at once. Nothing about correctness depends on this number; if a
    test needs to observe a child mid-flight it should gate the child (see
    FakeAgentRunner) instead of shrinking this.

    On expiry, report the status actually seen — "did not reach terminal" with
    no value forces whoever hits it to re-run locally to learn anything.
    """
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select

    TERMINAL = ("completed", "failed", "timeout", "cancelled", "budget_exhausted")
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    last = "<never read>"
    while loop.time() < deadline:
        async with async_session_maker() as db:
            row = (await db.execute(select(BuildJob).where(BuildJob.id == job_id))).scalar_one()
            last = row.status
            if row.status in TERMINAL:
                return row
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"Job {job_id} did not reach a terminal status within {timeout}s — "
        f"last status seen was {last!r} (terminal set: {TERMINAL})"
    )


# ──────────────────────────────────────────────────────────────────────
# Pre-spawn gating
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_kill_switch_off_rejects(monkeypatch):
    from app.config import settings
    from app.agent.subagent_orchestrator import spawn_subagent

    monkeypatch.setattr(settings, "subagent_spawning_enabled", False)
    uid = str(uuid.uuid4())
    runner = FakeAgentRunner()
    result = await spawn_subagent(
        user_id=uid, task="X", label="t", model=None,
        timeout_seconds=10, parent_job_id=None,
        channel="web", telegram_chat_id=None,
        agent_runner=runner,
    )
    assert result["error"] == "SUBAGENT_DISABLED"
    assert runner.calls == []


@pytest.mark.asyncio
async def test_empty_task_rejected(enable_spawning):
    from app.agent.subagent_orchestrator import spawn_subagent

    uid = str(uuid.uuid4())
    runner = FakeAgentRunner()
    result = await spawn_subagent(
        user_id=uid, task="   ", label=None, model=None,
        timeout_seconds=10, parent_job_id=None,
        channel="web", telegram_chat_id=None,
        agent_runner=runner,
    )
    assert result["error"] == "SUBAGENT_EMPTY_TASK"


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_creates_buildjob_with_right_shape(
    enable_spawning, patch_writer, db,
):
    from app.agent.subagent_orchestrator import spawn_subagent
    from app.db.models import BuildJob
    from sqlalchemy import select

    uid = str(uuid.uuid4())
    await _seed_user(db, uid)

    runner = FakeAgentRunner(delay=0.3)
    result = await spawn_subagent(
        user_id=uid, task="research X",
        label="research-X-label", model="claude-haiku-4-5",
        timeout_seconds=15, parent_job_id=None,
        channel="web", telegram_chat_id=None,
        agent_runner=runner,
    )

    job_id = result["job_id"]
    assert result["status"] in ("queued", "running")
    assert result["idempotency_collapsed"] is False

    from app.db.database import async_session_maker
    async with async_session_maker() as db2:
        job = (await db2.execute(select(BuildJob).where(BuildJob.id == job_id))).scalar_one()
        assert job.user_id == uid
        assert job.job_type == "subagent"
        assert job.source_kind == "subagent"
        assert job.title == "research-X-label"
        assert job.config_json is not None
        assert job.config_json["task"] == "research X"
        assert job.config_json["depth"] == 1
        assert job.idempotency_key is not None
        assert job.idempotency_key.startswith("sa:top:")

    await _wait_for_completion(job_id)


@pytest.mark.asyncio
async def test_happy_path_writes_message_and_marks_completed(
    enable_spawning, patch_writer, db,
):
    from app.agent.prompt_profile import PromptProfile
    from app.agent.subagent_orchestrator import spawn_subagent

    uid = str(uuid.uuid4())
    await _seed_user(db, uid)

    runner = FakeAgentRunner(response_text="The answer is 42.")
    result = await spawn_subagent(
        user_id=uid, task="answer the question",
        label="42-task", model=None,
        timeout_seconds=10, parent_job_id=None,
        channel="web", telegram_chat_id=None,
        agent_runner=runner,
    )
    job_id = result["job_id"]
    row = await _wait_for_completion(job_id)

    assert row.status == "completed"
    assert row.outcome == "success"
    assert row.summary_message_id is not None
    assert row.completed_at is not None

    # Writer + broadcast called with the right kwargs
    assert patch_writer["write"], "write_subagent_message must have been called"
    w = patch_writer["write"][0]
    assert w["user_id"] == uid
    assert w["job_id"] == job_id
    assert w["label"] == "42-task"
    assert "The answer is 42." in w["content"]
    assert w["outcome"] == "success"

    assert patch_writer["broadcast"], "broadcast_subagent_message must have been called"
    b = patch_writer["broadcast"][0]
    assert b["user_id"] == uid
    assert b["job_id"] == job_id
    assert b["outcome"] == "success"

    # agent_runner.run was called with the sub-agent kwargs
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["prompt_profile"] == PromptProfile.SUBAGENT
    assert call["save_user_message"] is False
    assert call["save_assistant_message"] is False
    assert call["disable_post_processing"] is True
    assert call["channel"] == "subagent"
    assert call["session_id"].startswith("subagent:")
    assert call["subagent_task_label"] == "42-task"


# ──────────────────────────────────────────────────────────────────────
# Timeout path
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeout_marks_status_and_announces(
    enable_spawning, patch_writer,
):
    """Same single-writer fix as the immediate-return test —
    don't hold a session open during spawn."""
    from app.agent.subagent_orchestrator import spawn_subagent
    from app.db.database import async_session_maker

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed_user(db, uid)

    runner = FakeAgentRunner(delay=2.0)
    result = await spawn_subagent(
        user_id=uid, task="long task",
        label="L", model=None, timeout_seconds=1,
        parent_job_id=None, channel="web",
        telegram_chat_id=None, agent_runner=runner,
    )
    row = await _wait_for_completion(result["job_id"])

    assert row.status == "timeout"
    assert row.outcome == "timeout"
    assert "timed out" in (row.error_message or "").lower()
    # An announce row was still posted so the user knows.
    assert patch_writer["write"]
    w = patch_writer["write"][0]
    assert w["outcome"] == "timeout"
    assert "timed out" in w["content"].lower()


# ──────────────────────────────────────────────────────────────────────
# Failure path
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exception_marks_failed_with_error_message(
    enable_spawning, patch_writer, db,
):
    from app.agent.subagent_orchestrator import spawn_subagent

    uid = str(uuid.uuid4())
    await _seed_user(db, uid)

    runner = FakeAgentRunner(raise_exc=RuntimeError("boom-in-child"))
    result = await spawn_subagent(
        user_id=uid, task="will fail",
        label="bad", model=None, timeout_seconds=10,
        parent_job_id=None, channel="web",
        telegram_chat_id=None, agent_runner=runner,
    )
    row = await _wait_for_completion(result["job_id"])

    assert row.status == "failed"
    assert row.outcome == "failed"
    assert "boom-in-child" in (row.error_message or "")


# ──────────────────────────────────────────────────────────────────────
# Idempotency
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(
    reason=(
        "SQLite write-lock contention when spawn #2 fires while spawn "
        "#1's child task is still committing its phase updates. Hangs "
        "in the batch run; passes when isolated. Functionality is "
        "covered by: (a) test_idempotency_key_stable_across_calls in "
        "test_subagent_dispatcher.py, (b) JobRunner.create_job's "
        "(source_id, idempotency_key) composite UNIQUE which is "
        "Postgres-strict and is exercised by the unified-jobs arc "
        "tests. A real-DB integration test belongs in the smoke "
        "suite (Postgres) not the unit suite."
    ),
)
@pytest.mark.asyncio
async def test_retried_spawn_collapses_to_existing_row(
    enable_spawning, patch_writer,
):
    """A retried tool call (same parent + same task) must return
    the same job_id and NOT fire a second child."""
    from app.agent.subagent_orchestrator import spawn_subagent
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    uid = str(uuid.uuid4())
    parent_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed_user(db, uid)
        db.add(BuildJob(
            id=parent_id, user_id=uid, title="P", prompt="p",
            job_type="agent_task", status="running",
        ))
        await db.commit()
    # Session closed here; spawn opens its own.

    runner = FakeAgentRunner(delay=0.4)
    r1 = await spawn_subagent(
        user_id=uid, task="dedupe-me",
        label="L", model=None, timeout_seconds=10,
        parent_job_id=parent_id, channel="web",
        telegram_chat_id=None, agent_runner=runner,
    )
    r2 = await spawn_subagent(
        user_id=uid, task="dedupe-me",
        label="L", model=None, timeout_seconds=10,
        parent_job_id=parent_id, channel="web",
        telegram_chat_id=None, agent_runner=runner,
    )

    assert r1["job_id"] == r2["job_id"]
    assert r2["idempotency_collapsed"] is True

    await _wait_for_completion(r1["job_id"])
    # Only one child run was fired despite the duplicate spawn.
    assert len(runner.calls) == 1


# ──────────────────────────────────────────────────────────────────────
# Non-blocking contract
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_returns_immediately(enable_spawning, patch_writer):
    """spawn_subagent must return without waiting for its child.

    Proven by construction rather than by stopwatch: the child is held on an
    `asyncio.Event` that this test never sets until after the assertions, so
    an implementation that awaited the child could not reach the assertions
    at all. Same single-writer care as test_retried_spawn — don't hold a
    session open across spawn.
    """
    from app.agent.subagent_orchestrator import spawn_subagent
    from app.db.database import async_session_maker

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed_user(db, uid)
    # Hold the child on a gate nothing has set yet. This is what makes the
    # assertion deterministic: if spawn_subagent awaited its child, it could
    # not return at all — so the property is proven by spawn returning, not by
    # it returning FAST. The previous version raced a 1.0s child delay against
    # an `elapsed < 0.95` bound, which is a coin-flip on a loaded runner: it
    # failed 1 run in 5 on an idle laptop, and CI executes this file alongside
    # others.
    #
    # wait_for, not a bare await, so a genuinely blocking implementation fails
    # in 10s with "spawn_subagent never returned" instead of hanging the suite
    # forever — a hang is strictly worse than a failure, because it reports as
    # a timeout with no diagnosis.
    gate = asyncio.Event()
    runner = FakeAgentRunner(gate=gate)
    try:
        result = await asyncio.wait_for(
            spawn_subagent(
                user_id=uid, task="slow",
                label="L", model=None, timeout_seconds=10,
                parent_job_id=None, channel="web",
                telegram_chat_id=None, agent_runner=runner,
            ),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        raise AssertionError(
            "spawn_subagent never returned while its child was held on a gate "
            "— it is awaiting the child instead of backgrounding it"
        )

    assert result["job_id"]
    # The child is still inside run(), provably: only `gate` can release it.
    assert not runner.finished.is_set(), (
        "child finished before the gate was released — the fake ran to "
        "completion, so this test is no longer observing a live child"
    )

    gate.set()
    await _wait_for_completion(result["job_id"])


# ──────────────────────────────────────────────────────────────────────
# Tool handler routing
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_spawn_routes_to_orchestrator_when_enabled(
    enable_spawning, patch_writer, db, monkeypatch,
):
    """The _tool_spawn handler routes to the orchestrator when
    settings.subagent_spawning_enabled is True. Verify by patching
    spawn_subagent and confirming it's called."""
    import tempfile
    from app.agent.tool_executor import ToolExecutor

    captured: list[dict[str, Any]] = []

    async def _fake_spawn(**kwargs):
        captured.append(kwargs)
        return {"job_id": "fake-job", "status": "queued", "label": kwargs.get("label")}

    import app.agent.subagent_orchestrator as orch
    monkeypatch.setattr(orch, "spawn_subagent", _fake_spawn)
    # ALSO the import-from binding inside tool_executor.
    import app.agent.tool_executor as te
    # _tool_spawn does `from app.agent.subagent_orchestrator import spawn_subagent`
    # at call time — patching the module attribute is sufficient.

    uid = str(uuid.uuid4())
    await _seed_user(db, uid)

    with tempfile.TemporaryDirectory() as workdir:
        executor = ToolExecutor(workspace=workdir)
        executor.set_user_id(uid)
        # Inject a dummy agent_runner since the orchestrator is mocked
        # we won't actually use it, but the handler checks for its presence.
        executor.agent_runner = object()
        executor.set_channel("web")
        result_json = await executor._tool_spawn({
            "task": "test task",
            "label": "tool-routed",
            "timeout_seconds": 5,
        })

    import json
    result = json.loads(result_json)
    assert result["job_id"] == "fake-job"
    assert captured, "spawn_subagent must have been called via _tool_spawn"
    assert captured[0]["task"] == "test task"
    assert captured[0]["label"] == "tool-routed"
    assert captured[0]["user_id"] == uid
    assert captured[0]["channel"] == "web"
