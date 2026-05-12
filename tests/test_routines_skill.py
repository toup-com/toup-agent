"""Tests for the routines skill — the agent-facing CRUD tools.

The underlying API + runner logic is already covered by
test_routine_runner.py and test_email_briefing_handler.py. These tests
target the skill-shaped wrapper: argument validation, JSON shape of the
return value, HTTPException → user-friendly ERROR string translation,
and dispatch routing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import uuid
from typing import Any

import pytest


# ── Helpers ──────────────────────────────────────────────────────────


def _load_skill():
    """Load the routines skill module from its file path the same way the
    SkillLoader does at agent boot. Returns an instantiated skill."""
    here = os.path.dirname(__file__)
    skill_path = os.path.normpath(
        os.path.join(
            here,
            "..",
            "app",
            "agent",
            "skills",
            "builtins",
            "routines",
            "skill.py",
        )
    )
    spec = importlib.util.spec_from_file_location("routines_skill", skill_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RoutinesSkill()


def _ctx():
    from app.agent.skills.base import SkillContext

    return SkillContext(workspace="/tmp", user_id="test-user", session_id="test")


async def _seed_user(timezone: str = "UTC") -> str:
    from app.db import async_session_maker
    from app.db.models import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            User(
                id=user_id,
                email=f"{user_id}@routine-skill.local",
                hashed_password="x",
                name="Routines Skill Test",
                timezone=timezone,
            )
        )
        await db.commit()
    return user_id


@pytest.fixture(autouse=True)
def _enable_routines_flag(monkeypatch):
    """Production code reads `settings.routines_email_briefing_enabled`
    (master flag for both kinds). Without this on, every create call
    returns 404 'Feature not available' which masks the validation
    paths we want to exercise."""
    from app.config import settings

    monkeypatch.setattr(settings, "routines_email_briefing_enabled", True)
    yield


# ── Dispatch + validation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_string():
    """Skill must surface unknown tool names as ERROR, not raise — the
    agent treats ERROR results as recoverable and can apologise to the
    user without crashing the turn."""
    skill = _load_skill()
    result = await skill.execute_tool("routines__bogus", {}, _ctx())
    assert result.startswith("ERROR:")
    assert "Unknown" in result


@pytest.mark.asyncio
async def test_create_missing_kind_returns_error():
    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__create",
        {"schedule_cron_local": "30 6 * * *"},
        _ctx(),
    )
    assert result.startswith("ERROR:")
    assert "kind" in result.lower()


@pytest.mark.asyncio
async def test_create_missing_schedule_returns_error():
    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__create",
        {"kind": "email_briefing"},
        _ctx(),
    )
    assert result.startswith("ERROR:")
    assert "schedule" in result.lower()


@pytest.mark.asyncio
async def test_create_agent_task_without_prompt_returns_error(monkeypatch):
    """agent_task without prompt_text is invalid by API contract — the
    skill must surface the 400 as an ERROR string, not a thrown
    HTTPException."""
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__create",
        {"kind": "agent_task", "schedule_cron_local": "0 7 * * *"},
        _ctx(),
    )
    assert result.startswith("ERROR:")
    assert "prompt_text" in result


@pytest.mark.asyncio
async def test_create_invalid_cron_returns_error(monkeypatch):
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__create",
        {
            "kind": "agent_task",
            "schedule_cron_local": "bogus",
            "prompt_text": "do the thing",
        },
        _ctx(),
    )
    assert result.startswith("ERROR:")
    assert "cron" in result.lower()


# ── Happy paths — JSON shape ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_agent_task_succeeds_and_returns_routine_id(monkeypatch):
    """The agent reads `routine.id` from the JSON to chain follow-up
    actions (update, delete, run_now). If the shape regresses the agent
    silently breaks — pin the contract."""
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__create",
        {
            "kind": "agent_task",
            "schedule_cron_local": "0 9 * * *",
            "name": "Daily standup brief",
            "prompt_text": "Summarise yesterday's commits across all repos.",
        },
        _ctx(),
    )
    payload = json.loads(result)
    assert payload["status"] == "created"
    assert payload["routine"]["kind"] == "agent_task"
    assert payload["routine"]["name"] == "Daily standup brief"
    assert payload["routine"]["schedule_cron_local"] == "0 9 * * *"
    assert payload["routine"]["enabled"] is True
    assert len(payload["routine"]["id"]) >= 32  # uuid


@pytest.mark.asyncio
async def test_list_returns_routines_for_user(monkeypatch):
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    skill = _load_skill()
    # Empty list first.
    empty = json.loads(await skill.execute_tool("routines__list", {}, _ctx()))
    assert empty["count"] == 0
    assert empty["routines"] == []

    # Create + relist.
    await skill.execute_tool(
        "routines__create",
        {
            "kind": "agent_task",
            "schedule_cron_local": "0 10 * * *",
            "prompt_text": "ping me",
            "name": "Ping",
        },
        _ctx(),
    )
    payload = json.loads(await skill.execute_tool("routines__list", {}, _ctx()))
    assert payload["count"] == 1
    assert payload["routines"][0]["name"] == "Ping"
    assert payload["routines"][0]["prompt_text"] == "ping me"


@pytest.mark.asyncio
async def test_update_changes_schedule_and_returns_new_value(monkeypatch):
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    skill = _load_skill()
    created = json.loads(
        await skill.execute_tool(
            "routines__create",
            {
                "kind": "agent_task",
                "schedule_cron_local": "0 6 * * *",
                "prompt_text": "x",
            },
            _ctx(),
        )
    )
    rid = created["routine"]["id"]

    updated = json.loads(
        await skill.execute_tool(
            "routines__update",
            {"routine_id": rid, "schedule_cron_local": "30 7 * * 1-5"},
            _ctx(),
        )
    )
    assert updated["status"] == "updated"
    assert updated["routine"]["schedule_cron_local"] == "30 7 * * 1-5"


@pytest.mark.asyncio
async def test_update_unknown_routine_returns_error(monkeypatch):
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__update",
        {"routine_id": "00000000-0000-0000-0000-000000000000", "enabled": False},
        _ctx(),
    )
    assert result.startswith("ERROR:")
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_delete_routine_succeeds_then_list_empty(monkeypatch):
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    skill = _load_skill()
    created = json.loads(
        await skill.execute_tool(
            "routines__create",
            {
                "kind": "agent_task",
                "schedule_cron_local": "0 6 * * *",
                "prompt_text": "x",
            },
            _ctx(),
        )
    )
    rid = created["routine"]["id"]

    deleted = json.loads(
        await skill.execute_tool("routines__delete", {"routine_id": rid}, _ctx())
    )
    assert deleted["status"] == "deleted"
    assert deleted["routine_id"] == rid

    remaining = json.loads(await skill.execute_tool("routines__list", {}, _ctx()))
    assert remaining["count"] == 0


# ── Feature-flag gate ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_when_flag_off_returns_feature_not_available_error(monkeypatch):
    """Flag-gate must surface as a clear ERROR string the agent can show
    the user — not a silent success or a stack trace. Production
    contract: 404 'Feature not available' becomes 'ERROR: Feature not
    available' to the agent."""
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)
    monkeypatch.setattr(settings, "routines_email_briefing_enabled", False)

    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__create",
        {"kind": "email_briefing", "schedule_cron_local": "30 6 * * *"},
        _ctx(),
    )
    assert result.startswith("ERROR:")
    assert "Feature not available" in result


# ── Tool definitions schema ──────────────────────────────────────────


def test_get_tools_returns_five_tools_with_required_fields():
    """Pin the tool surface so a refactor can't accidentally drop a tool
    the agent's system prompt mentions."""
    skill = _load_skill()
    tools = skill.get_tools()
    names = {t["name"] for t in tools}
    assert names == {
        "routines__create",
        "routines__list",
        "routines__update",
        "routines__delete",
        "routines__run_now",
    }
    for t in tools:
        assert t["name"].startswith("routines__")
        assert "input_schema" in t
        assert t["input_schema"]["type"] == "object"


def test_system_prompt_section_mentions_each_tool():
    """The system-prompt section guides the agent on when to call each
    tool. If a tool is added/removed and the prompt isn't updated, the
    agent stops calling it."""
    skill = _load_skill()
    section = skill.get_system_prompt_section() or ""
    assert "routines__create" in section
    assert "routines__list" in section
    assert "email_briefing" in section
    assert "agent_task" in section
