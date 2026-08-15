"""A cancelled turn must still close the jobs it created.

Observed 2026-07-31 (production, founder's device): two voice-created jobs —
"Find UofT LLM professors" and "Find U of T LLM/NLP professors" — sat in
`running` for 19 minutes. One had a single `Progress: 1/3` event and nothing
after; the other had no events at all, all three steps still `pending`, while
its Live Activity card read "Starting…". Neither would ever reach History,
because History is terminal-only.

Root cause: `AgentRunner._run_inner` closes created jobs at turn end, but
there is NO try/finally anywhere between the top of that method and the
finalizer. An exception, an early return, or a CANCELLATION skips it and the
rows are stranded.

Cancellation is the common case for voice, not an edge case:
`api_v1.internal_agent_turn_stream` runs the turn as a task and its SSE
generator's `finally` does `call_later(1.5, task.cancel)` the moment the
client disconnects — i.e. every time the caller stops talking or hangs up.

The 30-minute `job_reaper` was the only backstop, and it closed them as
**failed** with "⚠️ Didn't finish" — for work the agent had already delivered
out loud. That is the precise lie this pipeline exists to remove, so the
reaper now closes them as `cancelled` with taxonomy copy.

These tests drive the real `AgentRunner.run` wrapper. `_run_inner` and
`_close_interrupted_jobs` are stubbed so the guarantee is tested without
booting an LLM turn — the thing under test is the control flow, which is
exactly what was missing.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.agent.agent_runner import AgentRunner
from app.agent.job_status import (
    DISPOSITION_TERMINAL,
    ERR_TURN_INTERRUPTED,
    STATUS_CANCELLED,
    TERMINAL_STATUSES,
    turn_interrupted,
)

_BACKEND = Path(__file__).resolve().parent.parent


class _Tools:
    """Stands in for ToolExecutor's per-turn created-job ledger."""

    def __init__(self, ids=(), staged=()):
        self._ids = tuple(ids)
        self.takes = 0
        # Confirmation cards this turn staged. The sweep reads this to decide
        # whether an interrupted turn's jobs are dead or merely parked.
        self.staged_pending_action_ids = list(staged)

    def take_created_job_ids(self):
        self.takes += 1
        ids, self._ids = self._ids, ()
        return ids


def _runner(tools, inner):
    """A real AgentRunner with only the two seams stubbed.

    `object.__new__` skips __init__ deliberately: constructing a real runner
    would need an LLM client, a DB and a tool registry, none of which the
    control-flow guarantee depends on.
    """
    r = object.__new__(AgentRunner)
    r.tools = tools
    r._run_inner = inner
    r.closed = []
    r.closed_staged = []

    # This double's signature MUST track the real `_close_interrupted_jobs`.
    # `_sweep_unclosed_created_jobs` wraps the call in a broad
    # `except Exception` (cleanup must never mask the real error), so a
    # signature mismatch does not raise — it is logged and swallowed, and
    # every job is silently stranded. That is exactly how the
    # `staged_action_id` parameter landed: four tests here went red with an
    # empty `r.closed` and a TypeError buried in the captured log.
    async def _close(ids, user_id, staged_action_id=None):
        r.closed.append((tuple(ids), user_id))
        r.closed_staged.append(staged_action_id)

    r._close_interrupted_jobs = _close
    return r


async def _settle():
    """Let the detached close-task run."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


# ── the guarantee ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancelled_turn_still_closes_its_jobs():
    """The production symptom: hang up mid-turn, job stranded forever."""
    tools = _Tools(["job-a", "job-b"])

    async def inner(**kwargs):
        await asyncio.sleep(3600)  # still working when the caller hangs up

    r = _runner(tools, inner)
    task = asyncio.create_task(r.run(user_message="hi", user_id="u1"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _settle()

    assert r.closed == [(("job-a", "job-b"), "u1")], (
        "a cancelled turn must terminalise the jobs it created — this is the "
        "voice path's normal ending, not an edge case"
    )
    assert r.closed_staged == [None], (
        "no card was staged, so the sweep must not claim one — that would "
        "park a genuinely dead job forever"
    )


@pytest.mark.asyncio
async def test_a_cancelled_turn_holding_a_card_hands_the_action_id_over():
    """A card outlives the turn that staged it: it sits in the chat for 24h
    and stays tappable. So a turn cancelled while one is outstanding leaves
    a job that is WAITING, not dead — and the sweep has to hand the action
    id down so the resume path can match on it.

    The id must be read here, synchronously, before the work is detached:
    this sweep runs inside a `finally` usually reached *because* the task is
    being cancelled, and an await would be skipped by that same condition.
    """
    tools = _Tools(["job-a"], staged=["act-42"])

    async def inner(**kwargs):
        await asyncio.sleep(3600)

    r = _runner(tools, inner)
    task = asyncio.create_task(r.run(user_message="hi", user_id="u1"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _settle()

    assert r.closed == [(("job-a",), "u1")]
    assert r.closed_staged == ["act-42"], (
        "the staged card's id never reached the close path, so the parked "
        "job can never be resolved by an approval"
    )


@pytest.mark.asyncio
async def test_successful_turn_does_not_double_close():
    """The happy-path finalizer consumes the ids, so the sweep is a no-op."""
    tools = _Tools(["job-a"])

    async def inner(**kwargs):
        tools.take_created_job_ids()   # what _run_inner's finalizer does
        return "ok"

    r = _runner(tools, inner)
    assert await r.run(user_message="hi", user_id="u1") == "ok"
    await _settle()
    assert r.closed == [], "success must not re-close an already-closed job"


@pytest.mark.asyncio
async def test_exception_closes_jobs_and_still_raises():
    tools = _Tools(["job-a"])

    async def inner(**kwargs):
        raise RuntimeError("boom")

    r = _runner(tools, inner)
    with pytest.raises(RuntimeError):
        await r.run(user_message="hi", user_id="u1")
    await _settle()
    assert r.closed == [(("job-a",), "u1")]


@pytest.mark.asyncio
async def test_early_return_closes_jobs():
    """`finally`, not `except` — an early return strands jobs just as well."""
    tools = _Tools(["job-a"])

    async def inner(**kwargs):
        return "short-circuit"

    r = _runner(tools, inner)
    await r.run(user_message="hi", user_id="u1")
    await _settle()
    assert r.closed == [(("job-a",), "u1")]


@pytest.mark.asyncio
async def test_user_id_read_positionally():
    """`run()` is called positionally across the codebase, not just by kwarg."""
    tools = _Tools(["job-a"])

    async def inner(*a, **k):
        raise RuntimeError("boom")

    r = _runner(tools, inner)
    with pytest.raises(RuntimeError):
        await r.run("hello", "u-positional")
    await _settle()
    assert r.closed == [(("job-a",), "u-positional")]


@pytest.mark.asyncio
async def test_no_jobs_means_no_task():
    tools = _Tools([])

    async def inner(**kwargs):
        return "ok"

    r = _runner(tools, inner)
    await r.run(user_message="hi", user_id="u1")
    await _settle()
    assert r.closed == []


@pytest.mark.asyncio
async def test_sweep_never_masks_the_real_error():
    """A cleanup that raises must not replace the exception the caller needs."""
    class _Exploding:
        def take_created_job_ids(self):
            raise ValueError("ledger exploded")

    async def inner(**kwargs):
        raise RuntimeError("the real failure")

    r = _runner(_Exploding(), inner)
    with pytest.raises(RuntimeError, match="the real failure"):
        await r.run(user_message="hi", user_id="u1")


# ── the copy and the status ──────────────────────────────────────────────

def test_turn_interrupted_is_terminal_and_human():
    v = turn_interrupted()
    assert v.error_class == ERR_TURN_INTERRUPTED
    assert v.disposition == DISPOSITION_TERMINAL
    assert v.user_message and "conversation ended" in v.user_message.lower()
    for banned in ("Traceback", "Error code:", "{'detail'", "None", "asyncio"):
        assert banned not in v.user_message


def test_cancelled_is_terminal_so_history_can_see_it():
    """History is terminal-only; a non-terminal status would strand the row
    in Now forever, which is the bug being fixed."""
    assert STATUS_CANCELLED in TERMINAL_STATUSES


def test_reaper_no_longer_calls_a_stopped_job_failed():
    src = (_BACKEND / "app" / "agent" / "job_reaper.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert 'job.status = "failed"' not in code, (
        "a turn that went away is not a failure — the agent has usually "
        "already delivered the answer"
    )
    assert "STATUS_CANCELLED" in code
    assert "Didn't finish" not in code, "scary copy for non-failures"
    # The taxonomy fields must be populated, or the API's read-time
    # classification falls through to `unknown` — "Something went wrong.
    # We've been notified" — which is worse than the text it replaced.
    assert "job.error_class" in code and "job.user_message" in code


def test_run_wrapper_guards_the_inner_turn():
    """The wrapper is the whole fix; assert it did not get refactored away."""
    src = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "async def _run_inner(" in code
    assert "return await self._run_inner(" in code
    assert "finally:" in code.split("async def run(")[1].split("async def")[0], (
        "the sweep must run in a `finally` — an `except` misses early returns"
    )
