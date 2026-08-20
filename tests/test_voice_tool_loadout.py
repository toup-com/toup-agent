"""Voice must never be handed a tool it cannot run — including on a fallback.

Two defects, one rule.

`_tools_step` filtered the array down to what actually executes on the relay
(V2 hosted agents reach everything through `think`; the raw agent tools need a
tunnel that platform-api does not have). But `_apply_full_context` defaulted to
raw `REALTIME_TOOLS` whenever the context build was slow — so a 10s DB stall
re-armed `web_search`, `edit_file`, `exec` and the rest. The model then called
one directly, got "your terminal agent is not connected", and told the user it
lacked a capability it has. That list was also written to `_instr_cache` and to
`applied_ctx`, so one slow build poisoned every warm reopen for the cache TTL.

The instructions/tools waits also shared a single `try`, which meant an
instructions timeout skipped the tools await entirely — taking the bad fallback
even when the tool list had been ready for thirty seconds.

`_executable_tools` is now the one definition both paths go through.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

from test_voice_turn_survives import FakeClientWS, FakeOpenAIWS, relay  # noqa: F401,E402

pytestmark = pytest.mark.asyncio


@pytest.fixture
def v2_on(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "voice_realtime_v2", True)


@pytest.fixture
def v2_off(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "voice_realtime_v2", False)


# ── The single definition ────────────────────────────────────────────────

def test_v2_keeps_only_what_executes_on_the_relay(v2_on):
    from app.api.ws_realtime import _executable_tools, _REALTIME_NATIVE

    kept = {t["name"] for t in _executable_tools()}
    assert kept and kept <= _REALTIME_NATIVE
    assert "think" in kept, "think is the whole bridge to the user's agent"
    for dead in ("web_search", "edit_file", "exec", "read_file"):
        assert dead not in kept, (
            f"{dead} cannot run on platform-api — offering it makes the model "
            "narrate a broken capability"
        )


def test_v1_still_gets_the_full_array(v2_off):
    """v1 has a tunnel or a local ToolExecutor — the filter would be a
    regression there, not a fix."""
    from app.api.ws_realtime import _executable_tools, REALTIME_TOOLS

    assert len(_executable_tools()) == len(REALTIME_TOOLS)


def test_the_fallback_default_is_executable_not_raw(v2_on):
    """The bug in one line: the fallback and the normal path must agree."""
    from app.api.ws_realtime import _executable_tools, REALTIME_TOOLS

    assert len(_executable_tools()) < len(REALTIME_TOOLS), (
        "under V2 the executable set is a strict subset — if this is equal, "
        "the fallback is handing over tools that cannot run"
    )


# ── The alert ────────────────────────────────────────────────────────────

def test_a_loadout_without_think_is_flagged_unusable():
    from app.api.ws_realtime import _tools_are_usable

    assert _tools_are_usable([{"name": "think"}, {"name": "navigate_to"}])
    assert not _tools_are_usable([{"name": "navigate_to"}])
    assert not _tools_are_usable([]), (
        "an empty loadout can still hold a conversation, which is exactly why "
        "it needs an alert of its own"
    )


# ── End to end: what the live session is actually configured with ────────

async def test_the_session_is_configured_with_a_usable_loadout(relay, monkeypatch, v2_on):
    """The DB read for disabled-tools fails here (no DB in tests), which is the
    same shape as the stall that used to trigger the raw fallback."""
    rt = relay
    client = FakeClientWS()
    openai_ws = FakeOpenAIWS([{"type": "session.created"}])

    async def _connect(*a, **kw):
        return openai_ws
    monkeypatch.setattr(rt.websockets, "connect", _connect)

    import asyncio
    await asyncio.wait_for(
        rt.realtime_voice_ws(client, token="tok", session_id=None, onboarding=False),
        timeout=20,
    )

    updates = [s for s in openai_ws.sent if s.get("type") == "session.update"]
    assert updates, "the session was never configured"
    names = {t["name"] for t in updates[-1]["session"].get("tools", [])}
    assert "think" in names, f"voice cannot act: {names}"
    assert "web_search" not in names, (
        f"a tool that cannot execute on the relay reached the model: {names}"
    )


async def test_a_slow_instructions_build_does_not_re_arm_dead_tools(
    relay, monkeypatch, v2_on,
):
    """The actual regression path, driven.

    On main this test fails twice over: the shared `try` skipped the tools
    await when instructions timed out, and the fallback it landed on was raw
    REALTIME_TOOLS — so the model was configured with `web_search`, `exec` and
    the rest, none of which can execute on the relay.
    """
    import asyncio
    rt = relay

    monkeypatch.setattr(rt, "_CTX_INSTRUCTIONS_TIMEOUT_S", 0.05)
    monkeypatch.setattr(rt, "_CTX_TOOLS_TIMEOUT_S", 5.0)

    async def _hangs(user_id, onboarding=False, now_utc=None):
        await asyncio.sleep(30)
        return "never arrives"
    monkeypatch.setattr(rt, "build_realtime_instructions", _hangs)

    client = FakeClientWS()
    openai_ws = FakeOpenAIWS([{"type": "session.created"}])

    async def _connect(*a, **kw):
        return openai_ws
    monkeypatch.setattr(rt.websockets, "connect", _connect)

    await asyncio.wait_for(
        rt.realtime_voice_ws(client, token="tok", session_id=None, onboarding=False),
        timeout=20,
    )

    updates = [s for s in openai_ws.sent if s.get("type") == "session.update"]
    names = {t["name"] for t in updates[-1]["session"].get("tools", [])}
    assert "web_search" not in names and "exec" not in names, (
        f"a slow context build re-armed tools that cannot run: {sorted(names)}"
    )
    assert "think" in names, (
        "the timeout fallback dropped the one tool voice needs to do anything"
    )


# ── think() with no arguments ────────────────────────────────────────────
# Production 2026-08-20: `Function call: think({})` → agent-turn 422 → the
# tool-less fallback 400 → a non-answer to a question the user did ask.

def test_an_empty_think_falls_back_to_what_the_user_actually_said():
    from app.api.ws_realtime import _think_task

    assert _think_task({"task": "  "}, "چه شرکت‌هایی در AI سرمایه‌گذاری می‌کنن") == (
        "چه شرکت‌هایی در AI سرمایه‌گذاری می‌کنن"
    )
    assert _think_task({}, "what is crunchbase") == "what is crunchbase"


def test_a_real_task_is_passed_through_untouched():
    from app.api.ws_realtime import _think_task

    assert _think_task({"task": "search X"}, "ignored") == "search X"


def test_think_never_returns_empty():
    """An empty message is a 422 at the agent and a 400 at OpenAI — the two
    upstream rejections the user experienced as 'the agent did nothing'."""
    from app.api.ws_realtime import _think_task

    for args, last in (({}, ""), ({"task": ""}, "   "), ({"task": None}, None)):
        out = _think_task(args, last or "")
        assert out and out.strip(), f"empty task survived: {args!r}/{last!r}"
