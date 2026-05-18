"""Integration test for the 2026-05-12 prompt-leak fix.

Source-grep tests (in `test_routine_three_production_bugs.py`) verify
that `save_user_message=False` is in the source. This file exercises
the actual runtime contract — fires the handler against a mocked
AgentRunner and asserts the kwargs the handler passed.

A regression that:
  - Removes the kwarg → source-grep test catches it.
  - Adds the kwarg but somehow passes True at runtime → this test catches it.
  - Swaps the handler to a different code path → BOTH catch it (this test
    fails because the assertion on `run` doesn't fire).

Belt-and-braces. Cheap to maintain.
"""
from __future__ import annotations

import asyncio
import pytest

from app.agent.routines.agent_task_handler import AgentTaskHandler
from app.agent.routines.base_handler import RoutineResult


class _FakeRoutine:
    """Stand-in for the Routine ORM row. Only needs the fields the
    handler reads. The `name`/`kind`/`config_json` fields are
    referenced by the success-path code (delivery channel fan-out,
    Day-as-Chat title) — populate them so the handler doesn't AttributeError
    after the runner returns."""

    def __init__(self, prompt_text: str = "say hi"):
        self.id = "routine-abc"
        self.user_id = "user-xyz"
        self.prompt_text = prompt_text
        self.name = "Test routine"
        self.kind = "agent_task"
        self.config_json = None


class _FakeResponse:
    """AgentRunner.run() returns an object with a .text attribute (or
    .content) — handler reads whichever it finds."""

    def __init__(self, text: str = "hi back"):
        self.text = text


class _SpyAgentRunner:
    """Records every call to `.run(...)` so the test can assert on the
    kwargs. Mirrors the real AgentRunner's `.run(...)` signature loose
    enough to accept the handler's call shape."""

    def __init__(self, response_text: str = "hi back"):
        self.calls: list[dict] = []
        self._response_text = response_text

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._response_text)


# ── The contract ────────────────────────────────────────────────────


async def _spy_writer(db, **kwargs):
    """Stub writer so the handler's persist step doesn't touch a real
    Day-as-Chat row. Returns the (message_id, day_chat_id) tuple the
    handler expects."""
    return ("msg-fake", "daychat-fake")


@pytest.mark.asyncio
async def test_agent_task_handler_passes_save_user_message_false():
    """The handler MUST pass `save_user_message=False` so the routine's
    prompt_text doesn't get persisted as a synthetic user Message.
    Pre-fix the kwarg was omitted and the runner defaulted to True,
    leaking the system-generated prompt into the day-chat as if the
    user had typed it."""
    spy = _SpyAgentRunner()
    handler = AgentTaskHandler(agent_runner=spy, writer=_spy_writer)

    routine = _FakeRoutine(prompt_text="Every day at 1:21 PM Toronto, summarise my Gmail.")
    result = await handler.execute(routine=routine, run=None, db=None)

    # The handler must have invoked the runner exactly once.
    assert len(spy.calls) == 1, f"expected 1 runner.run call, got {len(spy.calls)}"
    kwargs = spy.calls[0]

    # The actual contract under test.
    assert kwargs.get("save_user_message") is False, (
        "AgentTaskHandler must pass save_user_message=False to runner.run. "
        "Without it the routine's prompt_text is persisted as a "
        "role='user' Message in the day-chat — the user sees the "
        "system-generated wrapper text as if they had typed it. "
        f"Got kwargs: {kwargs!r}"
    )

    # And the channel must still be 'routine' so the assistant reply
    # is correctly attributed.
    assert kwargs.get("channel") == "routine", (
        "AgentTaskHandler must pass channel='routine' so the assistant's "
        "reply is attributed to the scheduled run, not a live web turn."
    )

    # The handler should have returned a successful result.
    assert isinstance(result, RoutineResult)
    assert result.status == "success", (
        f"unexpected result status: {result.status} / {result.error_detail}"
    )


@pytest.mark.asyncio
async def test_agent_task_handler_handles_empty_prompt():
    """Edge case: a routine row with no prompt_text returns a failed
    result WITHOUT calling the runner. The pre-leak path could have
    written an empty user Message on this path too — confirm the
    fix doesn't accidentally rely on the early-return."""
    spy = _SpyAgentRunner()
    handler = AgentTaskHandler(agent_runner=spy)

    routine = _FakeRoutine(prompt_text="   ")
    result = await handler.execute(routine=routine, run=None, db=None)

    assert spy.calls == [], "runner.run must NOT be called when prompt_text is empty"
    assert result.status == "failed"
    assert result.error_class == "no_prompt"


# ── Duplicate-message regression (2026-05-18) ─────────────────────────


class _ResponseWithAsstId:
    """AgentResponse-like with `asst_message_id` populated — what the
    real AgentRunner returns after persisting its own Message row."""

    def __init__(self, text: str = "hi back", asst_message_id: str = "msg-from-runner"):
        self.text = text
        self.asst_message_id = asst_message_id
        self.model = "gpt-4o-mini"
        self.tokens_total = 0


@pytest.mark.asyncio
async def test_agent_task_handler_reuses_runner_message_no_duplicate_persist():
    """User-report 2026-05-17: an agent_task routine ("summarize my latest
    Gmail at X o'clock") produced TWO identical assistant bubbles in
    Day-as-Chat per fire. Root cause was a double-persist in
    `_run_via_agent_runner`: AgentRunner saved the assistant Message,
    then `_persist_and_return` wrote a SECOND row with the same content.

    Contract under test: when AgentRunner returns a response with
    `asst_message_id`, the handler must NOT call the writer — instead
    it decorates the runner's row and fans out via the channel
    dispatcher.

    Without this gate the user sees the same summary twice in chat for
    every scheduled run of the routine."""
    spy = _SpyAgentRunner()
    spy._response = _ResponseWithAsstId()

    async def _spy_run(**kwargs):
        spy.calls.append(kwargs)
        return spy._response

    spy.run = _spy_run

    writer_calls: list = []

    async def _failing_writer(db, **kwargs):
        # Catches any attempt to write a SECOND row. If this fires the
        # double-persist bug is back.
        writer_calls.append(kwargs)
        return ("msg-second-row-DO-NOT-WRITE", "daychat-fake")

    handler = AgentTaskHandler(agent_runner=spy, writer=_failing_writer)
    routine = _FakeRoutine(prompt_text="Summarize my latest Gmail.")
    # db=None is fine — the decorate path tolerates a missing Message
    # (logs a warning) and proceeds with broadcast + fan-out.
    result = await handler.execute(routine=routine, run=None, db=None)

    assert isinstance(result, RoutineResult)
    assert result.status == "success", f"unexpected status: {result.status}"
    # ── The actual regression gate ──
    assert writer_calls == [], (
        "AgentTaskHandler wrote a SECOND assistant Message on top of the "
        "one AgentRunner already persisted — this is the duplicate-bubble "
        "bug from the 2026-05-17 user report. The handler must reuse "
        f"response.asst_message_id, not call the writer. writer_calls={writer_calls!r}"
    )
    assert result.summary_message_id == "msg-from-runner", (
        "summary_message_id must point at the runner's row so Mission "
        "Control and the dashboard surface the same canonical id."
    )


@pytest.mark.asyncio
async def test_agent_task_handler_falls_back_to_writer_without_asst_id():
    """If AgentRunner returns a response WITHOUT `asst_message_id`
    (legacy shape, test mock that doesn't model it), the handler must
    still produce a routine artifact via the legacy persist path so the
    routine fire is discoverable. Belt-and-braces against the duplicate
    fix accidentally introducing a silent-success bug for older
    runners."""
    spy = _SpyAgentRunner()  # default _FakeResponse — no asst_message_id

    writer_calls: list = []

    async def _spy_writer(db, **kwargs):
        writer_calls.append(kwargs)
        return ("msg-fallback", "daychat-fake")

    handler = AgentTaskHandler(agent_runner=spy, writer=_spy_writer)
    routine = _FakeRoutine(prompt_text="say hi")
    result = await handler.execute(routine=routine, run=None, db=None)

    assert result.status == "success"
    assert len(writer_calls) == 1, (
        "Fallback path: handler must persist via the injected writer "
        "when AgentRunner doesn't expose asst_message_id."
    )
    assert result.summary_message_id == "msg-fallback"
