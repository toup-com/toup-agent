"""Round 4 (2026-08-19) — item 5: reminders fire when asked, once, with a
countdown that reaches the phone in time.

Measured (founder tenant, 2026-08-18 23:18Z): "remind me in 1 min" → the
model passed `at_local="19:19"` (minute-resolution clock, minute-resolution
parameter), the parser zeroed the seconds, APScheduler's DateTrigger fired at
EXACTLY 23:19:00.04 — 25 s after the ask. Not scheduler drift: the boundary.

Fixes pinned here:
  5b — `in_seconds` (relative, computed at execution time, seconds-exact);
       `at_local` keeps seconds; a wall-clock time that passed within the
       last two minutes fires in a moment instead of tomorrow.
  5c — one request = one reminder: an identical enabled reminder within
       90 s is returned as `already_scheduled` instead of a twin.
  5a — the countdown card's `mission_started` rides the ingest fast lane.

RUN_MODE=agent (routines + users live on the agent side; the sibling file
test_routines_skill.py is on the agent-mode list for the same reason).
"""

from __future__ import annotations

import importlib.util
import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest


def _load_skill():
    here = os.path.dirname(__file__)
    skill_path = os.path.normpath(os.path.join(
        here, "..", "app", "agent", "skills", "builtins", "routines", "skill.py"))
    spec = importlib.util.spec_from_file_location("routines_skill_r4", skill_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RoutinesSkill(), mod


def _ctx():
    from app.agent.skills.base import SkillContext
    return SkillContext(workspace="/tmp", user_id="test-user", session_id="test")


async def _seed_user(timezone_name: str = "America/Toronto") -> str:
    from app.db import async_session_maker
    from app.db.models import User
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"{user_id}@r4.local", hashed_password="x",
                    name="R4 reminders", timezone=timezone_name))
        await db.commit()
    return user_id


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "routines_email_briefing_enabled", True, raising=False)
    monkeypatch.setattr(settings, "routines_reminders_enabled", True, raising=False)
    monkeypatch.setattr(settings, "reminder_countdown_live_activity_enabled", False, raising=False)


# ── the parser (5b) ─────────────────────────────────────────────────────

def test_at_local_keeps_seconds_and_accepts_iso_with_seconds():
    from zoneinfo import ZoneInfo
    skill, _ = _load_skill()
    tz = ZoneInfo("America/Toronto")
    dt = skill._parse_local_datetime("2026-12-25 18:00:30", tz)
    assert dt is not None and (dt.hour, dt.minute, dt.second) == (18, 0, 30)
    future = datetime.now(tz) + timedelta(hours=1)
    dt2 = skill._parse_local_datetime(future.strftime("%H:%M:45"), tz)
    assert dt2 is not None and dt2.second == 45


def test_a_wall_clock_time_that_just_passed_fires_soon_not_tomorrow():
    """The model read a minute-resolution clock during a slow turn and asked
    for the CURRENT minute. Yesterday that became tomorrow — a silent
    day-long miss. Now: in a moment. Older than two minutes still rolls."""
    from zoneinfo import ZoneInfo
    skill, _ = _load_skill()
    tz = ZoneInfo("America/Toronto")
    now = datetime.now(tz)
    just_passed = (now - timedelta(seconds=40)).strftime("%H:%M:%S")
    dt = skill._parse_local_datetime(just_passed, tz)
    assert dt is not None
    assert timedelta(0) < (dt - now) < timedelta(seconds=10)
    hour_ago = (now - timedelta(hours=1)).strftime("%H:%M")
    dt2 = skill._parse_local_datetime(hour_ago, tz)
    assert dt2 is not None and dt2 > now + timedelta(hours=20)   # tomorrow


# ── the handler: relative reminders (5b) ────────────────────────────────

async def _create_capture(monkeypatch, mod):
    """Route the skill's create through a capture instead of the real API,
    returning a duck-typed response the skill formats."""
    captured = {}

    async def fake_create_routine(req):
        captured["req"] = req
        class _R:
            id = "rid-1"; name = req.name; enabled = True; config = {"delivery_channels": ["website"]}
            schedule_kind = req.schedule_kind; schedule_cron_local = getattr(req, "schedule_cron_local", "* * * * *")
            schedule_at = getattr(req, "schedule_at", None); schedule_interval_seconds = None
            schedule_window_start_local = None; schedule_window_end_local = None; next_run_at = None
        return _R()

    import app.api.routines as R
    monkeypatch.setattr(R, "create_routine", fake_create_routine)
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase,in_seconds", [
    ("in 1 min", 60), ("in 90 seconds", 90), ("in 2 hours", 7200),
])
async def test_relative_reminders_fire_at_now_plus_in_seconds(monkeypatch, phrase, in_seconds):
    from app.config import settings
    skill, mod = _load_skill()
    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)
    captured = await _create_capture(monkeypatch, mod)

    async def _no_dup(*a, **k):
        return None
    monkeypatch.setattr(mod.RoutinesSkill, "_find_duplicate_reminder", staticmethod(_no_dup))

    t0 = datetime.now(timezone.utc).replace(tzinfo=None)
    result = await skill.execute_tool(
        "routines__remind",
        {"reminder_text": f"test {phrase}", "when": "once", "in_seconds": in_seconds,
         "delivery_channels": ["website"]},
        _ctx(),
    )
    assert not result.startswith("ERROR"), result
    req = captured["req"]
    assert req.schedule_kind == "at"
    delta = (req.schedule_at - t0).total_seconds()
    # ±2 s of the requested delay (the fixed instant, no minute rounding)
    assert in_seconds - 2 <= delta <= in_seconds + 2, delta
    assert json.loads(result)["status"] == "created"


@pytest.mark.asyncio
async def test_absolute_at_8_15_is_a_wall_clock_time_in_the_users_tz(monkeypatch):
    from zoneinfo import ZoneInfo
    from app.config import settings
    skill, mod = _load_skill()
    user_id = await _seed_user("America/Toronto")
    monkeypatch.setattr(settings, "user_id", user_id)
    captured = await _create_capture(monkeypatch, mod)

    async def _no_dup(*a, **k):
        return None
    monkeypatch.setattr(mod.RoutinesSkill, "_find_duplicate_reminder", staticmethod(_no_dup))

    result = await skill.execute_tool(
        "routines__remind",
        {"reminder_text": "standup", "when": "once", "at_local": "08:15",
         "delivery_channels": ["website"]},
        _ctx(),
    )
    assert not result.startswith("ERROR"), result
    at_utc = captured["req"].schedule_at.replace(tzinfo=timezone.utc)
    local = at_utc.astimezone(ZoneInfo("America/Toronto"))
    assert (local.hour, local.minute, local.second) == (8, 15, 0)
    assert at_utc > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_in_seconds_wins_over_a_stale_at_local_and_validates(monkeypatch):
    from app.config import settings
    skill, mod = _load_skill()
    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)
    captured = await _create_capture(monkeypatch, mod)

    async def _no_dup(*a, **k):
        return None
    monkeypatch.setattr(mod.RoutinesSkill, "_find_duplicate_reminder", staticmethod(_no_dup))

    t0 = datetime.now(timezone.utc).replace(tzinfo=None)
    result = await skill.execute_tool(
        "routines__remind",
        {"reminder_text": "x", "when": "once", "in_seconds": 120, "at_local": "23:59",
         "delivery_channels": ["website"]},
        _ctx(),
    )
    assert not result.startswith("ERROR"), result
    assert 118 <= (captured["req"].schedule_at - t0).total_seconds() <= 122
    # validation
    for bad in (0, 4, -10, "soon"):
        r = await skill.execute_tool("routines__remind",
                                     {"reminder_text": "x", "when": "once", "in_seconds": bad}, _ctx())
        assert r.startswith("ERROR"), (bad, r)
    r = await skill.execute_tool("routines__remind", {"reminder_text": "x", "when": "once"}, _ctx())
    assert r.startswith("ERROR") and "in_seconds" in r


# ── one request = one reminder (5c) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_a_second_identical_reminder_within_the_window_is_not_created(monkeypatch):
    """The real create path both times; the second call must find the first
    and return already_scheduled without a second row."""
    from app.config import settings
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import Routine

    skill, mod = _load_skill()
    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    async def _no_reload(*a, **k):
        return None
    import app.api.routines as R
    monkeypatch.setattr(R, "reload_routine", _no_reload, raising=False)

    args = {"reminder_text": "Do the dishes.", "when": "once", "in_seconds": 600,
            "delivery_channels": ["website"]}
    r1 = await skill.execute_tool("routines__remind", dict(args), _ctx())
    assert not r1.startswith("ERROR"), r1
    assert json.loads(r1)["status"] == "created"
    r2 = await skill.execute_tool("routines__remind", dict(args), _ctx())
    assert not r2.startswith("ERROR"), r2
    assert json.loads(r2)["status"] == "already_scheduled"
    assert json.loads(r2)["reminder"]["id"] == json.loads(r1)["reminder"]["id"]

    async with async_session_maker() as db:
        rows = (await db.execute(select(Routine).where(
            Routine.user_id == user_id, Routine.kind == "reminder"))).scalars().all()
    assert len(rows) == 1

    # a DIFFERENT text is a different reminder
    r3 = await skill.execute_tool("routines__remind", {**args, "reminder_text": "Feed the cat"}, _ctx())
    assert json.loads(r3)["status"] == "created"
    # ...and the same text at a clearly different time is too
    r4 = await skill.execute_tool("routines__remind", {**args, "in_seconds": 3600}, _ctx())
    assert json.loads(r4)["status"] == "created"


# ── the prompt + schema tell the model the new contract ─────────────────

def test_schema_and_prompt_route_relative_asks_to_in_seconds():
    skill, mod = _load_skill()
    tool = next(t for t in skill.get_tools() if t["name"] == "routines__remind")
    props = tool["input_schema"]["properties"]
    assert "in_seconds" in props and props["in_seconds"]["type"] == "integer"
    assert "in_seconds" in tool["description"] and "ONCE" in tool["description"]
    assert "HH:MM[:SS]" in props["at_local"]["description"]
    section = skill.get_system_prompt_section() or ""
    assert "in_seconds" in section and "exactly ONCE" in section
    # diet path carries the same contract
    assert "in_seconds" in mod._DIET_TOOL_DESCRIPTIONS["routines__remind"]
    assert "in_seconds" in mod._DIET_PROPERTY_DESCRIPTIONS["routines__remind"]


# ── the countdown start rides the fast lane (5a) ────────────────────────

def test_reminder_countdown_start_is_fast_laned():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "app" / "api"
    an = (root / "agent_notify.py").read_text()
    assert 'str(_data.get("mission_id") or "").startswith("reminder:")' in an
    assert '_data.get("fast_lane")' in an
    rt = (root / "routines.py").read_text()
    assert '"fast_lane": True' in rt
