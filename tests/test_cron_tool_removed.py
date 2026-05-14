"""Pins the 2026-05-14 cron-tool removal.

The legacy `cron` tool only delivered to Telegram and returned
``ERROR: No active Telegram chat`` on every other surface. A user on
website chat asked "in 2 min remind me to drink water" and the agent
picked `cron` (over the new `routines__remind`) because the cron
description literally said "create reminders" — the LLM saw it as
the obvious match. Net result: a real reminder request bounced with
a cryptic Telegram error and nothing was scheduled.

This module pins three contracts so the fix can't drift back:

  1. `cron` is NOT in the tool registry returned by
     `get_tool_definitions`. The agent must not see it as an option.
  2. `_tool_cron` (still present as a defensive backstop) returns
     ERROR-shaped strings that name the replacement tools — calling
     it never falls back to the old CronService.
  3. The agent's system prompt directs reminder requests at
     `routines__remind`, not `cron`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


BACKEND = Path(__file__).resolve().parent.parent
_TOOL_DEFS_SRC = (BACKEND / "app/agent/tool_definitions.py").read_text()
_TOOL_EXEC_SRC = (BACKEND / "app/agent/tool_executor.py").read_text()
_AGENT_RUNNER_SRC = (BACKEND / "app/agent/agent_runner.py").read_text()


# ── Registry ────────────────────────────────────────────────────────


def test_cron_tool_not_in_registry():
    """The agent's tool list must not advertise `cron`. The LLM picks
    tools from the schemas the runner hands it; if `cron` is in the
    list, the model treats it as a valid reminder-creation option and
    chooses it over `routines__remind` (the cron description sat at
    the top of the LLM's attention, ahead of the skill tools)."""
    from app.agent.tool_definitions import get_agent_tools
    defs = get_agent_tools()
    names = {d["name"] for d in defs}
    assert "cron" not in names, (
        "`cron` is still in get_agent_tools(). The agent will keep "
        "calling it for reminder requests and hitting the "
        '"No active Telegram chat" error on website / WhatsApp.'
    )


def test_cron_tool_definition_block_is_removed_source():
    """Source-grep guard: the cron tool's `{"name": "cron", ...}` dict
    must not appear in tool_definitions.py. A future refactor that
    reformats the file shouldn't silently restore the dict."""
    assert '"name": "cron"' not in _TOOL_DEFS_SRC, (
        "Found `\"name\": \"cron\"` in tool_definitions.py — the cron "
        "tool registry entry was supposed to be removed on 2026-05-14."
    )


# ── Handler backstop ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_cron_handler_returns_redirect_not_telegram_error():
    """If anything still calls `_tool_cron` (cached schema, old
    prompt), the response must redirect at `routines__remind` —
    NOT return the old "No active Telegram chat" message. The latter
    would tell the model to keep retrying via a broken path."""
    from app.agent.tool_executor import ToolExecutor

    # Bypass __init__ — the new _tool_cron handler doesn't read any
    # self.* state. The full ctor wants to create a workspace dir
    # which fails outside the Docker image.
    executor = ToolExecutor.__new__(ToolExecutor)
    result = await executor._tool_cron({
        "action": "add",
        "name": "test",
        "schedule": "in 2m",
        "message": "drink water",
    })
    assert result.startswith("ERROR:")
    assert "routines__remind" in result, result
    assert "No active Telegram chat" not in result, (
        "_tool_cron still returns the legacy Telegram-coupled error. "
        "Drop the call to cron_service.add_job and surface the "
        "routines__remind redirect instead."
    )


@pytest.mark.asyncio
async def test_tool_cron_list_action_redirects_to_routines_list():
    from app.agent.tool_executor import ToolExecutor
    # Bypass __init__ — the new _tool_cron handler doesn't read any
    # self.* state. The full ctor wants to create a workspace dir
    # which fails outside the Docker image.
    executor = ToolExecutor.__new__(ToolExecutor)
    result = await executor._tool_cron({"action": "list"})
    assert result.startswith("ERROR:")
    assert "routines__list" in result


@pytest.mark.asyncio
async def test_tool_cron_remove_and_run_redirect_too():
    from app.agent.tool_executor import ToolExecutor
    # Bypass __init__ — the new _tool_cron handler doesn't read any
    # self.* state. The full ctor wants to create a workspace dir
    # which fails outside the Docker image.
    executor = ToolExecutor.__new__(ToolExecutor)
    rem = await executor._tool_cron({"action": "remove", "job_id": "x"})
    assert rem.startswith("ERROR:")
    assert "routines__delete" in rem

    run = await executor._tool_cron({"action": "run", "job_id": "x"})
    assert run.startswith("ERROR:")
    assert "routines__run_now" in run


# ── System prompt ───────────────────────────────────────────────────


def test_system_prompt_does_not_route_reminders_to_cron():
    """The agent's "when user says X, do Y" rule for reminders must
    name `routines__remind`, not `cron`. Without this the LLM sees
    two competing instructions (skill prompt → routines__remind;
    main prompt → cron) and picks the one that loads first."""
    # The decision rule for "remind me" must mention routines__remind
    # and must NOT name `cron` as the call target.
    assert "routines__remind" in _AGENT_RUNNER_SRC
    # The exact old line we replaced. If it ever creeps back it'll
    # cause this exact regression again.
    assert "'remind me at <Y>' → call `cron`" not in _AGENT_RUNNER_SRC, (
        "Decision rule still tells the agent to call `cron` for "
        "reminder requests. That's the 2026-05-14 regression."
    )
