"""A `create_job` job must not outlive the turn that created it.

Observed 2026-07-25 (production rows, founder's device): a voice turn called
the `create_job` tool for "بررسی جدیدترین مدل Anthropic". The Live Activity
opened at "Starting…" 0%, moved once to 33%, then sat frozen for 31 minutes
until the 30-minute `job_reaper` failed it and pushed a "⚠️ Didn't finish"
alert — for a question the agent had already answered out loud seconds in.

Root cause: nothing closes a `create_job` job at turn end. Its only closers
are the model calling `update_job`, and the reaper. `ws_chat` has a turn-end
finalizer but it covers only the regex-intake job (`source_kind='chat_intent'`);
the voice path (`/api/v1/internal/agent-turn`) has none, so on voice such a job
is orphaned by construction. The fix lives in `AgentRunner.run()` — the one
seam every channel shares.

⚠️ THE FIRST ATTEMPT AT THAT FIX WAS INERT, and a file of grep-the-source tests
just like this one passed it anyway. It selected on
`BuildJob.conversation_id == session_id`, but the producer wrote
`conversation_id=getattr(self, "_current_session_id", None)` — an attribute
nothing ever assigned — so the column was always NULL and the predicate never
matched. The behavioural coverage that actually catches that now lives in
`test_create_job_turn_context.py` (it asserts the persisted row). What remains
here is only the loop-integration shape that cannot be reached behaviourally
without booting an LLM turn, in the same source-grep style as
`test_parallel_tool_dispatch`. Read both files as one suite.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_RUNNER = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
_TOOLS = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()


def _code_only(src: str) -> str:
    """Drop whole-line comments. These files explain the bugs they fixed by
    NAMING the old code, so a naive substring assert matches the rationale
    comment and fails on correct code."""
    return "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))


def _finalizer_block() -> str:
    """The turn-end finalizer, bounded by the `return AgentResponse(` it must
    sit in front of."""
    idx = _RUNNER.index("_created_job_ids = self.tools.take_created_job_ids()")
    end = _RUNNER.index("return AgentResponse(", idx)
    return _RUNNER[idx:end]


# ── It must run at the end of every channel's turn ──────────────────────

def test_finalizer_runs_before_the_response_is_returned():
    """In AgentRunner.run(), so voice/Telegram/WhatsApp get it too — not in
    ws_chat, which is where the only previous finalizer lived."""
    assert _finalizer_block(), "finalizer must precede `return AgentResponse(`"


# ── Scoping: exact ids, never a predicate sweep ─────────────────────────

def test_finalizer_closes_exact_recorded_ids():
    """GATE. Targeting the ids the tool recorded is what makes the close
    precise. A conversation_id predicate would also close jobs from EARLIER
    turns of the same long-lived conversation, and `source_kind='manual'` is
    shared with dashboard-created jobs."""
    blk = _finalizer_block()
    assert "take_created_job_ids()" in blk
    assert "_CJ.id == _jid" in blk
    assert "_CJ.user_id == user_id" in blk
    assert "conversation_id ==" not in blk, (
        "the conversation_id predicate is the inert/over-broad design; "
        "close exact ids instead"
    )


def test_the_phantom_session_attribute_never_returns():
    """`getattr(self, "_current_session_id", None)` read an attribute nothing
    assigned — the whole reason the first fix did nothing. conversation_id now
    comes from the ContextVar the runner sets."""
    assert '_current_session_id' not in _code_only(_TOOLS), (
        "phantom attribute is back; conversation_id will silently go NULL again"
    )
    assert "conversation_id=_SESSION_ID_CTX.get()" in _TOOLS


def test_per_turn_state_is_contextvars_not_instance_attrs():
    """One ToolExecutor is shared across concurrent turns and by spawned
    sub-agents, so per-turn state MUST be per-asyncio-task."""
    assert "_SESSION_ID_CTX: contextvars.ContextVar" in _TOOLS
    assert "_CREATED_JOB_IDS_CTX: contextvars.ContextVar" in _TOOLS
    assert "self._turn_created_job_ids" not in _RUNNER
    assert "self._created_job_ids" not in _TOOLS


def test_finalizer_skips_the_sanctioned_handoff():
    """The tool contract permits finishing by handing work to `spawn` /
    `start_mission`, which DO continue the job after the reply. Closing it then
    would cut the legs off live background work."""
    blk = _finalizer_block()
    assert '"spawn", "start_mission"' in blk
    assert "not _handed_off" in blk


# ── Write semantics ────────────────────────────────────────────────────

def test_write_is_a_guarded_update_not_read_then_mutate():
    """`update_job` or the reaper may drive a row terminal while the turn
    finishes; the WHERE must re-check 'running' at write time."""
    blk = _finalizer_block()
    assert '_CJ.status == "running"' in blk
    assert "_upd_cj(_CJ)" in blk, "expected a guarded UPDATE statement"


def test_terminal_write_matches_the_sibling_finalizers():
    """ws_chat (ws_chat.py) and the vibe block both persist total_tokens+model;
    a consumer reading one and not the other sees an inconsistent row."""
    blk = _finalizer_block()
    assert "total_tokens=total_input + total_output" in blk
    assert "model=model_used" in blk


def test_outcome_is_not_keyed_on_reply_text():
    """Reaching this line means run() completed normally. Keying the outcome on
    non-empty `final_text` mislabelled legitimate tool-only / attachment-only
    turns as failures."""
    blk = _finalizer_block()
    assert '_answered' not in blk
    assert 'status="completed"' in blk


def test_card_end_push_survives_turn_cancellation():
    """The row is committed terminal BEFORE the push, and closing it removes
    the reaper backstop — so losing the push to a turn cancellation would
    strand the card on the phone forever. Round 4: the push is a BACKGROUND
    task (held by the module-level set, so it survives the turn's task being
    cancelled — the property the earlier `asyncio.shield` bought) and no
    longer awaited: it held the `done` frame back ~0.5–1 s for an outbox
    write nothing downstream depends on."""
    blk = _finalizer_block()
    assert "_spawn_background(_end_cards())" in blk
    assert "await asyncio.shield(asyncio.create_task(_end_cards()))" not in blk


def test_finalizer_can_never_fail_the_turn():
    """A turn must not 500 because notification plumbing hiccupped."""
    blk = _finalizer_block()
    assert re.search(r"except Exception as _e:.*\n\s*logger\.warning", blk), blk[-400:]


# ── The card must not open on a frozen "0%" ────────────────────────────

def test_create_job_opens_an_indeterminate_timer_not_progress_zero():
    """`_content_state` picks timer over progress, so `progress=0` shipped a
    card reading a literal frozen "0%" until the first update_job — which for
    an orphaned job never came. A countdown animates on-device with no pushes,
    and the first real update swaps in a discrete bar.

    NOTE the residual trade-off: the window is the reaper's 30-minute stall
    cutoff, so on the fallback path where the finalizer does not run the bar
    reaches full about when the reaper gives up. That is the true deadline for
    the job, but it does mean a full bar is not a success signal on its own."""
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
