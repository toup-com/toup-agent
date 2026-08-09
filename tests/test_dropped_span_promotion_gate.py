"""Drop-time memory promotion must obey ``disable_post_processing``.

``AgentRunner.run()`` has three background write paths that fire after (or
alongside) the turn. Two of them were gated on ``disable_post_processing``
from the day the parameter landed:

    if not disable_post_processing:
        _spawn_background(_background_post_processing())      # extraction
    ...
    if not disable_post_processing and _use_day_ctx and _day_chat_id:
        _spawn_background(run_summarizer_if_needed(...))      # day summary

The third — ``_promote_dropped_span``, the A8-6 callback that promotes a
summarized-out span to durable memory before it leaves the context window —
was not. It reached ``compact_messages`` unconditionally at all three
compaction sites, so a run that explicitly asked for NO post-processing
(sub-agent runs, routine turns, ``save=False`` voice turns, the memory
probes) could still write memories.

These tests drive the real ``run()`` with a fake LLM and a fake
``compact_messages`` that reproduces context_manager's own contract
(``if on_drop is not None: on_drop(span)``), then assert on whether
``_extract_memories`` was actually reached.

NOTE ON CI COVERAGE: nothing here touches ``memories`` (the write is
intercepted at ``_extract_memories``), so this file runs in the sqlite
platform-mode sweep as well as under Postgres. It does need the ``users``
table, which is PLATFORM_ONLY — hence RUN_MODE=platform / monolith, not
RUN_MODE=agent.
"""

from __future__ import annotations

import ast
import asyncio
import uuid as _uuid
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parent.parent
_RUNNER_SRC = (BACKEND_DIR / "app" / "agent" / "agent_runner.py").read_text()
_CTX_SRC = (BACKEND_DIR / "app" / "agent" / "context_manager.py").read_text()


# ──────────────────────────────────────────────────────────────────────
# Harness
# ──────────────────────────────────────────────────────────────────────


class _FakeLLM:
    """Duck-typed stand-in for OpenAIAgentService: one text turn."""

    def __init__(self):
        self.calls = []

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent

        self.calls.append({**kwargs, "messages": list(kwargs.get("messages") or [])})
        yield StreamEvent(type="text", text="done")
        yield StreamEvent(
            type="message_end",
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 2},
        )


async def _make_user() -> str:
    from app.db import async_session_maker, User

    user_id = str(_uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"{user_id[:8]}@test.local",
            hashed_password="x" * 60,
            name="Drop-gate",
        ))
        await db.commit()
    return user_id


# The span that compaction "drops". Deliberately memorable content — if the
# gate leaks, this is the text that ends up in the user's durable memory.
_DROPPED_SPAN = [
    {"role": "user", "content": "my sister lands at SFO on the 14th"},
    {"role": "assistant", "content": [{"type": "text", "text": "noted, the 14th"}]},
]


async def _drive_one_turn(monkeypatch, tmp_path, *, disable_post_processing: bool):
    """Run one real turn, force the pre-loop compaction, and report what
    the drop-time promotion path actually did.

    Returns ``(on_drop_seen, extract_calls)`` where ``on_drop_seen`` is the
    exact value handed to ``compact_messages``.
    """
    import app.agent.agent_runner as ar
    from app.agent.prompt_profile import PromptProfile
    from app.agent.tool_executor import ToolExecutor

    extract_calls: list[dict] = []
    spawned: list = []
    seen: dict = {}

    async def _spy_extract(self, db, user_id, user_message, assistant_response,
                           query_was_trivial: bool = False):
        extract_calls.append({
            "user_id": user_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
        })
        return 0

    monkeypatch.setattr(ar.AgentRunner, "_extract_memories", _spy_extract)

    # `messages` and `identities` are AGENT_ONLY, so under RUN_MODE=platform
    # (what the CI sweep runs) neither table exists — that is why the sibling
    # end-to-end file, test_subagent_context_isolation.py, is exiled to the
    # agent-mode lane. Neither input matters here: the gate under test sits
    # BELOW prompt assembly and history loading, and compaction is forced
    # regardless of payload size. Stub both and keep this file in the sweep.
    async def _no_history(self, db, session_id, max_messages: int = 50, client_tz=None):
        return []

    async def _fixed_prompt(self, *a, **kw):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", _no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", _fixed_prompt)

    # Capture every background coroutine instead of letting the loop own it,
    # so the assertions are deterministic rather than racing a scheduler.
    monkeypatch.setattr(ar, "_spawn_bg", spawned.append)

    def _drop_background(coro):
        # The OTHER two gated paths are not under test here; close their
        # coroutines so a `never awaited` warning doesn't muddy the signal.
        coro.close()

    monkeypatch.setattr(ar, "_spawn_background", _drop_background)

    # Force the pre-loop compaction branch. The real needs_compaction() would
    # want a ~200k-token payload to say yes.
    monkeypatch.setattr(ar, "needs_compaction", lambda *a, **kw: True)

    async def _fake_compact(messages, model, conversation_id=None, on_drop=None, **kw):
        seen["on_drop"] = on_drop
        # Reproduce context_manager.compact_messages' own contract verbatim
        # (see test_context_manager_only_promotes_when_on_drop_is_not_none):
        # the promotion block is entered only when on_drop is not None — and
        # a given span is promoted at most ONCE, because the real function
        # advances the persisted compaction_promoted_through cursor. With
        # needs_compaction forced True, the harness now reaches TWO gated
        # sites (pre-loop + the G-7 first-request routed-model gate, whose
        # 128k window for the unknown 'gpt-5.5-mini' override is below the
        # default model's); firing the same span from both would model a
        # cursor that doesn't exist.
        if on_drop is not None and not seen.get("span_promoted"):
            seen["span_promoted"] = True
            on_drop(list(_DROPPED_SPAN))
        return messages

    monkeypatch.setattr(ar, "compact_messages", _fake_compact)

    runner = ar.AgentRunner(
        llm_service=_FakeLLM(),  # type: ignore[arg-type]
        tool_executor=ToolExecutor(workspace=str(tmp_path)),
    )
    user_id = await _make_user()
    response = await runner.run(
        user_message="compare pricing of X and Y",
        user_id=user_id,
        session_id=f"subagent:{_uuid.uuid4()}",
        channel="subagent",
        prompt_profile=PromptProfile.SUBAGENT,
        save_user_message=False,
        save_assistant_message=False,
        disable_post_processing=disable_post_processing,
        model_override="gpt-5.5-mini",
    )
    assert response.text == "done", "the harness itself did not complete a turn"

    # Drain whatever the turn scheduled. The promotion coroutine opens its own
    # session and calls _extract_memories (spied above); anything else the run
    # happened to spawn is harmless here and cannot touch extract_calls.
    if spawned:
        await asyncio.gather(*spawned, return_exceptions=True)

    assert "on_drop" in seen, "compaction never ran — the harness did not reach the gate"
    return seen["on_drop"], extract_calls


# ──────────────────────────────────────────────────────────────────────
# The defect
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_promotion_write_when_post_processing_disabled(monkeypatch, tmp_path):
    """disable_post_processing=True → drop-time promotion writes nothing."""
    on_drop, extract_calls = await _drive_one_turn(
        monkeypatch, tmp_path, disable_post_processing=True
    )

    assert extract_calls == [], (
        "a run that asked for NO post-processing still ran memory extraction "
        f"on the dropped span: {extract_calls}"
    )
    assert on_drop is None, (
        "the promotion callback was still handed to compact_messages — "
        "context_manager advances the persisted compaction_promoted_through "
        "cursor whenever on_drop is not None, so the span would be marked "
        "'already promoted' for the next turn that IS allowed to promote"
    )


@pytest.mark.asyncio
async def test_promotion_still_writes_when_post_processing_enabled(monkeypatch, tmp_path):
    """ANTI-VACUITY CONTROL — the gate must gate, not disable.

    Same turn, same harness, ``disable_post_processing=False``: the dropped
    span must still reach _extract_memories, or "gated" would just mean
    "A8-6 was deleted".
    """
    on_drop, extract_calls = await _drive_one_turn(
        monkeypatch, tmp_path, disable_post_processing=False
    )

    assert on_drop is not None, "A8-6 promotion callback is no longer wired at all"
    assert len(extract_calls) == 1, (
        f"the default path must still promote the dropped span, got {extract_calls}"
    )
    call = extract_calls[0]
    assert "my sister lands at SFO on the 14th" in call["user_message"]
    assert "noted, the 14th" in call["assistant_response"]


# ──────────────────────────────────────────────────────────────────────
# Source pins — same style as the sibling gates in
# tests/test_agent_runner_subagent_params.py, so a refactor cannot
# silently un-gate this path while the behavioural tests above are
# quietly skipped or rewritten.
# ──────────────────────────────────────────────────────────────────────


def _find_funcdef(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def test_promotion_callback_is_bound_through_the_gate():
    """The callback reaches compact_messages ONLY via the gated binding."""
    assert (
        "_on_drop = None if disable_post_processing else _promote_dropped_span"
        in _RUNNER_SRC
    ), "the drop-time promotion gate is gone"
    assert "on_drop=_promote_dropped_span" not in _RUNNER_SRC, (
        "a compaction site binds the raw callback and bypasses the gate"
    )
    # Every compaction site in run() must go through it. Four sites:
    # pre-loop, the G-7 first-request routed-model gate, the
    # context_length_exceeded retry, and the 80% mid-loop.
    assert _RUNNER_SRC.count("on_drop=_on_drop,") == 4, (
        "not every compact_messages() call site routes on_drop through the gate"
    )


def test_all_three_background_write_paths_share_one_gate():
    """The two sibling paths stay gated too — this is the invariant the
    whole parameter exists for, and the defect was that only 2 of 3 held."""
    assert "if not disable_post_processing:" in _RUNNER_SRC, (
        "_background_post_processing scheduling is no longer gated"
    )
    assert (
        "if not disable_post_processing and _use_day_ctx and _day_chat_id:"
        in _RUNNER_SRC
    ), "the day-chat summarizer is no longer gated"


def test_context_manager_only_promotes_when_on_drop_is_not_none():
    """Pins the contract the fake compact_messages above reproduces.

    If context_manager ever promoted with a None callback, handing it None
    would stop gating anything and the behavioural tests would go vacuous.
    """
    assert "if on_drop is not None and cached_summary is None" in _CTX_SRC
    fn = _find_funcdef(_CTX_SRC, "_filter_unpromoted")
    assert fn is not None, (
        "_filter_unpromoted is gone — the promoted-through cursor that makes "
        "withholding the callback (rather than returning early inside it) the "
        "correct gate no longer exists; re-check the fix"
    )
