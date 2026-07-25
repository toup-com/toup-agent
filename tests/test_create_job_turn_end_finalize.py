"""A `create_job` job must not outlive the turn that created it.

Observed 2026-07-25 (production rows, founder's device): a voice turn called
the `create_job` tool for "بررسی جدیدترین مدل Anthropic". The Live Activity
opened at "Starting…" 0%, moved once to 33%, and then sat frozen for 31
minutes until the 30-minute `job_reaper` failed it and pushed a
"⚠️ Didn't finish" alert — for a question the agent had already answered out
loud seconds in.

Root cause: nothing closes a `create_job` job at turn end. Its only closers
are the model itself calling `update_job`, and the reaper. `ws_chat` has a
turn-end finalizer but it covers only the regex-intake job
(`source_kind='chat_intent'`); the voice path (`/api/v1/internal/agent-turn`)
has no finalizer at all, so on voice such a job is orphaned by construction.

The fix lives in `AgentRunner.run()` — the one seam every channel shares.
`run()` is a single large coroutine and the guardrails for this work forbid
opportunistic refactoring, so these are **structural** tests in the same
source-grep style as `test_parallel_tool_dispatch` and
`test_agent_runner_tool_events`. They pin the parts that would actually hurt
if they regressed: the scoping predicate (a too-broad one would close other
people's jobs), the guards, and the non-fatal wrapper.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_RUNNER = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
_TOOLS = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()


def _finalizer_block() -> str:
    """The turn-end finalizer, bounded by the `return AgentResponse(` it
    must sit in front of."""
    idx = _RUNNER.index('if session_id and any(tc.get("name") == "create_job"')
    end = _RUNNER.index("return AgentResponse(", idx)
    return _RUNNER[idx:end]


# ── It must run at the end of every channel's turn ──────────────────────

def test_finalizer_runs_before_the_response_is_returned():
    """In AgentRunner.run(), so voice/Telegram/WhatsApp get it too — not in
    ws_chat, which is where the only previous finalizer lived."""
    assert _finalizer_block(), "finalizer must precede `return AgentResponse(`"


def test_finalizer_is_guarded_by_an_actual_create_job_call():
    """No DB round-trip on turns that never made a job."""
    blk = _finalizer_block()
    assert 'any(tc.get("name") == "create_job" for tc in all_tool_calls)' in blk


def test_finalizer_requires_a_session_id():
    """A job created without a session id has conversation_id NULL and cannot
    be scoped safely, so it must fall through to the reaper rather than be
    swept up by a broad match."""
    assert _RUNNER.index("if session_id and any(") > 0


# ── Scoping: the predicate that stops it closing the wrong jobs ─────────

def test_finalizer_scopes_to_this_conversation_and_the_create_job_tool():
    """GATE. `source_kind='manual'` alone is NOT specific enough — the
    dashboard also creates jobs with source_kind 'manual' (see
    ws_chat._detect_and_create_task's comment). conversation_id == session_id
    is what makes this safe."""
    blk = _finalizer_block()
    for clause in (
        "_CJ.user_id == user_id",
        '_CJ.job_type == "agent_task"',
        '_CJ.source_kind == "manual"',
        '_CJ.status == "running"',
        "_CJ.conversation_id == session_id",
    ):
        assert clause in blk, f"missing scoping clause: {clause}"


def test_finalizer_does_not_touch_chat_intent_jobs():
    """ws_chat's own finalizer owns those; double-closing would drop the
    total_tokens/model it writes."""
    blk = _finalizer_block()
    assert "chat_intent" not in blk
    assert '_CJ.source_kind == "manual"' in blk


# ── Honesty + safety ───────────────────────────────────────────────────

def test_outcome_reflects_whether_the_turn_actually_answered():
    """Completed only if the turn produced text; otherwise failed with a
    reason. Never blanket-'completed'."""
    blk = _finalizer_block()
    assert '_answered = bool((final_text or "").strip())' in blk
    assert '_j.status = "completed" if _answered else "failed"' in blk
    assert "mission_completed" in blk and "mission_failed" in blk


def test_finalizer_can_never_fail_the_turn():
    """A turn must not 500 because notification plumbing hiccupped."""
    blk = _finalizer_block()
    assert re.search(r"except Exception as _e:.*\n\s*logger\.warning", blk), blk[-400:]


# ── The card must not open on a frozen "0%" ────────────────────────────

def test_create_job_opens_an_indeterminate_timer_not_progress_zero():
    """`_content_state` picks timer over progress, so `progress=0` shipped a
    card reading a literal frozen "0%" until the first update_job — which for
    an orphaned job never came. A countdown animates on-device with no pushes,
    and the first real update swaps in a discrete bar."""
    idx = _TOOLS.index("async def _tool_create_job(")
    nxt = re.search(r"\n    (?:async )?def ", _TOOLS[idx + 10:])
    body = _TOOLS[idx: idx + 10 + nxt.start()] if nxt else _TOOLS[idx:]

    start_call = body[body.index('kind="mission_started"'):]
    start_call = start_call[: start_call.index(")\n")]
    # Strip comments — the rationale comment names `progress=0` on purpose.
    code = "\n".join(ln for ln in start_call.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "timer_end_ms=" in code, "mission_started must carry a timer"
    assert "progress=0" not in code, "a bare progress=0 renders as '0%'"
