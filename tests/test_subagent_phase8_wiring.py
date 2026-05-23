"""Phase 8 — production-finishing wiring tests.

Pins the three pieces of glue that close the gap between
"PRs merged" and "operator can flip the kill switch":

  1. ContextVars isolation — a child asyncio.create_task that
     overwrites ToolExecutor state via set_user_id / set_chat_id /
     set_channel / set_current_job_id must NOT affect the parent
     task's view of those values. This closes the broader
     concurrency hazard the precondition report flagged at >50
     LOC and Phase 3 deferred.

  2. _current_job_id plumbing — agent_runner.run(current_job_id=X)
     propagates to tool_executor._current_job_id, so _tool_spawn
     can set parent_job_id on the child BuildJob row.

  3. app/main.py wiring — source-grep guards that the three glue
     calls (tool_executor.agent_runner assignment,
     set_telegram_bot_holder, orphan_sweep_on_boot) actually
     ship in the lifespan startup hook.

What we deliberately don't test here:
  - The orphan_sweep_on_boot path itself (covered by Phase 5
    tests).
  - The orchestrator path (covered by Phase 4 tests).
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. ContextVars isolation — parent + child tasks
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_contextvars_isolate_state_across_asyncio_tasks():
    """The headline guarantee. Run a parent flow on the singleton
    ToolExecutor, fire a child asyncio.create_task that overwrites
    user_id / chat_id / channel / job_id, then await both. The
    parent's view of its own state must be unchanged."""
    from app.agent.tool_executor import ToolExecutor

    with tempfile.TemporaryDirectory() as w:
        te = ToolExecutor(workspace=w)

        # Parent state
        te.set_user_id("parent-uid")
        te.set_chat_id(100)
        te.set_channel("web")
        te.set_current_job_id("parent-job")

        # Read fence: capture parent's view before firing the child
        before = {
            "user_id": te._current_user_id,
            "chat_id": te._chat_id,
            "channel": te._current_channel,
            "job_id": te._current_job_id,
        }
        assert before == {
            "user_id": "parent-uid",
            "chat_id": 100,
            "channel": "web",
            "job_id": "parent-job",
        }

        # The child mirrors what agent_runner.run would do at the
        # start of a sub-agent invocation: clobber the shared
        # executor's state with the child's own values.
        child_done = asyncio.Event()
        child_seen = {}

        async def _child():
            te.set_user_id("child-uid")
            te.set_chat_id(999)
            te.set_channel("subagent")
            te.set_current_job_id("child-job")
            # Give the parent a chance to interleave (yield)
            await asyncio.sleep(0)
            child_seen.update({
                "user_id": te._current_user_id,
                "chat_id": te._chat_id,
                "channel": te._current_channel,
                "job_id": te._current_job_id,
            })
            child_done.set()

        task = asyncio.create_task(_child())
        # Yield so the child runs at least once
        await asyncio.sleep(0)

        # Re-read the parent's view AFTER the child mutated. With
        # ContextVars (per-task isolation), the parent still sees
        # its original values. Without contextvars (the
        # pre-Phase-8 instance-attribute behaviour) the parent
        # would see "child-uid" / 999 / "subagent" / "child-job".
        after = {
            "user_id": te._current_user_id,
            "chat_id": te._chat_id,
            "channel": te._current_channel,
            "job_id": te._current_job_id,
        }
        assert after == before, (
            f"ContextVars failed to isolate: parent's view "
            f"after child mutation = {after}, expected {before}"
        )

        # Wait for the child to finish and check it saw ITS OWN values.
        await child_done.wait()
        await task
        assert child_seen == {
            "user_id": "child-uid",
            "chat_id": 999,
            "channel": "subagent",
            "job_id": "child-job",
        }


@pytest.mark.asyncio
async def test_contextvars_default_to_none():
    """A fresh ToolExecutor + a fresh asyncio task with no set_*
    calls reads None for every per-call slot."""
    from app.agent.tool_executor import ToolExecutor

    async def _inner():
        with tempfile.TemporaryDirectory() as w:
            te = ToolExecutor(workspace=w)
            return {
                "user_id": te._current_user_id,
                "chat_id": te._chat_id,
                "channel": te._current_channel,
                "job_id": te._current_job_id,
            }

    seen = await asyncio.create_task(_inner())
    assert seen == {
        "user_id": None,
        "chat_id": None,
        "channel": None,
        "job_id": None,
    }


@pytest.mark.asyncio
async def test_contextvars_share_within_same_task():
    """Within a single asyncio task, ContextVars behave like
    normal state — the same task reading after set_* sees the
    written value. (Sanity: we're not breaking the existing
    in-turn read pattern.)"""
    from app.agent.tool_executor import ToolExecutor

    with tempfile.TemporaryDirectory() as w:
        te = ToolExecutor(workspace=w)
        te.set_user_id("same-task")
        await asyncio.sleep(0)  # yield mid-task; same context
        assert te._current_user_id == "same-task"


@pytest.mark.asyncio
async def test_concurrent_runs_on_singleton_dont_leak_state():
    """The end-to-end scenario the Amendment-3 audit worried about:
    parent's tool loop is mid-iteration when the child fires.
    Simulate by running parent_loop + child_loop concurrently via
    asyncio.gather and assert each loop's reads match its own
    writes throughout."""
    from app.agent.tool_executor import ToolExecutor

    with tempfile.TemporaryDirectory() as w:
        te = ToolExecutor(workspace=w)

        async def _loop(name: str, channel: str, iterations: int = 10):
            te.set_user_id(f"{name}-uid")
            te.set_channel(channel)
            te.set_current_job_id(f"{name}-job")
            for _ in range(iterations):
                await asyncio.sleep(0)  # yield to the other loop
                assert te._current_user_id == f"{name}-uid"
                assert te._current_channel == channel
                assert te._current_job_id == f"{name}-job"
            return name

        results = await asyncio.gather(
            _loop("parent", "web"),
            _loop("child", "subagent"),
            _loop("trigger", "trigger"),
        )
        assert set(results) == {"parent", "child", "trigger"}


# ──────────────────────────────────────────────────────────────────────
# 2. _current_job_id plumbing through AgentRunner.run
# ──────────────────────────────────────────────────────────────────────


def test_agent_runner_run_accepts_current_job_id():
    """Pin the new kwarg with the right default."""
    import inspect
    from app.agent.agent_runner import AgentRunner

    sig = inspect.signature(AgentRunner.run)
    assert "current_job_id" in sig.parameters
    assert sig.parameters["current_job_id"].default is None


_AGENT_RUNNER_PATH = (
    Path(__file__).resolve().parent.parent
    / "app" / "agent" / "agent_runner.py"
)
_AGENT_RUNNER_SRC = _AGENT_RUNNER_PATH.read_text()


def test_agent_runner_calls_set_current_job_id():
    """The run() body must call set_current_job_id with the kwarg
    early enough that any tool call (including _tool_spawn) reads
    the right value."""
    assert "self.tools.set_current_job_id(current_job_id)" in _AGENT_RUNNER_SRC, (
        "agent_runner.run() must propagate current_job_id to the "
        "tool executor via set_current_job_id"
    )


def test_tool_executor_has_set_current_job_id():
    """Pin the new public setter on ToolExecutor."""
    from app.agent.tool_executor import ToolExecutor
    assert hasattr(ToolExecutor, "set_current_job_id")


def test_routine_agent_task_handler_passes_job_id():
    """The routine runner is the highest-volume caller of
    agent_runner.run. Pin that it now forwards the BuildJob id
    so a sub-agent spawned during a routine fire links back to
    the routine's job row, not as a top-level orphan."""
    src = (
        Path(__file__).resolve().parent.parent
        / "app" / "agent" / "routines" / "agent_task_handler.py"
    ).read_text()
    assert "current_job_id=current_job_id" in src, (
        "agent_task_handler._run_via_agent_runner must forward "
        "current_job_id to runner.run"
    )
    assert "current_job_id=job_id" in src, (
        "execute() must extract the BuildJob id from the run shim "
        "and forward to _run_via_agent_runner"
    )


def test_dashboard_task_passes_job_id():
    """The /api/apps/tasks dashboard endpoint creates a BuildJob
    and runs the agent — must forward the job_id."""
    src = (
        Path(__file__).resolve().parent.parent / "app" / "api" / "apps.py"
    ).read_text()
    # Look for the call within the dashboard task path.
    assert "current_job_id=job_id" in src, (
        "Dashboard task runner must forward current_job_id"
    )


# ──────────────────────────────────────────────────────────────────────
# 3. app/main.py wiring — source grep
# ──────────────────────────────────────────────────────────────────────


_MAIN_PATH = Path(__file__).resolve().parent.parent / "app" / "main.py"
_MAIN_SRC = _MAIN_PATH.read_text()


def test_main_wires_tool_executor_agent_runner_attribute():
    """_tool_spawn's Path-A reads self.agent_runner first; without
    this setattr it falls back to subagent_manager._agent_runner
    (works but adds an indirection). Pin the direct wiring."""
    assert "tool_executor.agent_runner = agent_runner" in _MAIN_SRC


def test_main_calls_set_telegram_bot_holder():
    """The orchestrator's Telegram fan-out is a no-op until the bot
    holder is wired. This silent no-op is the kind of footgun
    Phase 7's runbook explicitly warned about."""
    assert "set_telegram_bot_holder" in _MAIN_SRC
    assert "set_telegram_bot_holder(subagent_manager)" in _MAIN_SRC


def test_main_calls_orphan_sweep_on_boot():
    """Phase 5's crash recovery is dead code without a call site.
    Pin the lifespan call."""
    assert "orphan_sweep_on_boot" in _MAIN_SRC
    assert "await orphan_sweep_on_boot()" in _MAIN_SRC


def test_main_wiring_is_after_agent_runner_creation():
    """Ordering matters — the three Phase 8 calls must happen AFTER
    the AgentRunner + ToolExecutor + SubAgentManager have been
    constructed. A naive cut-and-paste that lands them before the
    construction would crash at boot."""
    idx_runner = _MAIN_SRC.find("agent_runner = AgentRunner(")
    idx_wiring = _MAIN_SRC.find("tool_executor.agent_runner = agent_runner")
    idx_sweep = _MAIN_SRC.find("await orphan_sweep_on_boot()")
    assert idx_runner > 0, "agent_runner construction site not found"
    assert idx_wiring > idx_runner, (
        "tool_executor.agent_runner wiring must come AFTER agent_runner construction"
    )
    assert idx_sweep > idx_wiring, (
        "orphan_sweep_on_boot must come after the wiring (so the runner exists)"
    )


def test_main_wiring_wrapped_in_try_except():
    """The Phase 8 wiring is non-fatal — a boot proceeds even if
    the orphan sweep fails (e.g. DB unreachable). Pin the
    try/except guard so a 'helpful cleanup' PR doesn't accidentally
    crash the platform when the sweep can't run."""
    # Find the wiring block and confirm it's inside a try/except.
    wiring_idx = _MAIN_SRC.find("tool_executor.agent_runner = agent_runner")
    # Look backward from the wiring line for the most recent "try:"
    preceding = _MAIN_SRC[max(0, wiring_idx - 2000) : wiring_idx]
    assert preceding.rfind("try:") > 0, (
        "Phase 8 wiring block must be inside a try/except"
    )
    # And the except follows in the next ~3000 chars
    following = _MAIN_SRC[wiring_idx : wiring_idx + 3000]
    assert "except Exception" in following, (
        "Phase 8 wiring block must have a corresponding except clause"
    )
