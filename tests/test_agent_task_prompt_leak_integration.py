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
