"""Feature 1 — multi-channel delivery by default, never ask.

Pins three contracts:

  1. `get_connected_channels()` reuses the EXACT deliverability gates
     the fire-time senders use (telegram: bot booted with `.app` AND a
     TelegramUserMapping row; whatsapp: adapter registered AND self-E164
     resolvable; website: always) and never raises — a failed probe
     drops the channel, nothing more.
  2. The routines skill defaults an omitted `delivery_channels` to the
     connected set; explicit lists ("only telegram") pass through
     untouched.
  3. Every "ALWAYS ask"/"ASK WHERE" prompt surface is gone and the
     opposite contract (deliver everywhere by default, do NOT ask) is
     present in both tool descriptions, the skill system prompt, and
     the agent_runner runtime prompt.
"""

from __future__ import annotations

import importlib.util
import json
import os
import uuid
from types import SimpleNamespace

import pytest

import app.agent.routines.channel_dispatcher as cd


# ── Helpers (mirrors test_routines_skill harness) ────────────────────


def _load_skill():
    here = os.path.dirname(__file__)
    skill_path = os.path.normpath(
        os.path.join(here, "..", "app", "agent", "skills", "builtins",
                     "routines", "skill.py")
    )
    spec = importlib.util.spec_from_file_location("routines_skill_mcd", skill_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RoutinesSkill()


def _ctx():
    from app.agent.skills.base import SkillContext

    return SkillContext(workspace="/tmp", user_id="test-user", session_id="test")


async def _seed_user(timezone: str = "America/Toronto") -> str:
    from app.db import async_session_maker
    from app.db.models import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"{user_id}@mcd.local", hashed_password="x",
            name="MCD Test", timezone=timezone,
        ))
        await db.commit()
    return user_id


async def _add_tg_mapping(user_id: str, telegram_id: int = 4242) -> None:
    from app.db import async_session_maker
    from app.db.models import TelegramUserMapping

    async with async_session_maker() as db:
        db.add(TelegramUserMapping(telegram_id=telegram_id, user_id=user_id))
        await db.commit()


def _no_wa(monkeypatch):
    from app.agent.channels.registry import ChannelRegistry
    from app.agent.channels.base import ChannelType

    monkeypatch.setattr(
        ChannelRegistry, "_channels",
        {k: v for k, v in ChannelRegistry._channels.items()
         if k is not ChannelType.WHATSAPP},
    )


def _with_wa(monkeypatch):
    from app.agent.channels.registry import ChannelRegistry
    from app.agent.channels.base import ChannelType

    monkeypatch.setattr(
        ChannelRegistry, "_channels",
        {**ChannelRegistry._channels, ChannelType.WHATSAPP: object()},
    )


# ── get_connected_channels: gate parity with the senders ─────────────


@pytest.mark.asyncio
async def test_no_bot_no_adapter_is_website_only(monkeypatch):
    from app.db import async_session_maker

    monkeypatch.setattr(cd, "_get_telegram_bot", lambda: None)
    _no_wa(monkeypatch)

    out = await cd.get_connected_channels(str(uuid.uuid4()), async_session_maker)
    assert out == ["website"]


@pytest.mark.asyncio
async def test_telegram_requires_booted_bot_and_mapping_row(monkeypatch):
    from app.db import async_session_maker

    user_id = await _seed_user()
    _no_wa(monkeypatch)

    # Bot booted but owner never DM'd it — a send would skip
    # no_recipient, so 'connected' must say no too.
    monkeypatch.setattr(cd, "_get_telegram_bot",
                        lambda: SimpleNamespace(app=object()))
    assert await cd.get_connected_channels(user_id, async_session_maker) == ["website"]

    # Mapping row lands → connected.
    await _add_tg_mapping(user_id)
    assert await cd.get_connected_channels(user_id, async_session_maker) == [
        "website", "telegram",
    ]

    # Bot object without .app (adapter not booted) → not connected,
    # mapping row or not — same gate as _send_telegram_detailed.
    monkeypatch.setattr(cd, "_get_telegram_bot",
                        lambda: SimpleNamespace(app=None))
    assert await cd.get_connected_channels(user_id, async_session_maker) == ["website"]


@pytest.mark.asyncio
async def test_whatsapp_requires_adapter_and_recipient(monkeypatch):
    from app.db import async_session_maker

    monkeypatch.setattr(cd, "_get_telegram_bot", lambda: None)
    _with_wa(monkeypatch)

    async def no_recipient(_maker):
        return None

    monkeypatch.setattr(cd, "_resolve_whatsapp_recipient", no_recipient)
    assert await cd.get_connected_channels("u-1", async_session_maker) == ["website"]

    async def recipient(_maker):
        return "+14375550100"

    monkeypatch.setattr(cd, "_resolve_whatsapp_recipient", recipient)
    assert await cd.get_connected_channels("u-1", async_session_maker) == [
        "website", "whatsapp",
    ]


@pytest.mark.asyncio
async def test_probe_failures_never_raise(monkeypatch):
    from app.db import async_session_maker

    def boom():
        raise RuntimeError("agent_main not importable")

    monkeypatch.setattr(cd, "_get_telegram_bot", boom)
    _with_wa(monkeypatch)

    async def wa_boom(_maker):
        raise RuntimeError("sidecar down")

    monkeypatch.setattr(cd, "_resolve_whatsapp_recipient", wa_boom)

    out = await cd.get_connected_channels("u-1", async_session_maker)
    assert out == ["website"]


# ── Skill defaulting: omitted param → connected set ──────────────────


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "routines_email_briefing_enabled", True)
    monkeypatch.setattr(settings, "routines_reminders_enabled", True)
    yield


@pytest.mark.asyncio
async def test_remind_omitted_channels_default_to_connected(monkeypatch):
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    probed = []

    async def fake_connected(uid, maker):
        probed.append(uid)
        return ["website", "telegram"]

    monkeypatch.setattr(cd, "get_connected_channels", fake_connected)

    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__remind",
        {"reminder_text": "Drink water", "when": "daily",
         "daily_at_local": "07:30"},
        _ctx(),
    )
    assert not result.startswith("ERROR:"), result
    payload = json.loads(result)
    assert payload["reminder"]["delivery_channels"] == ["website", "telegram"]
    assert probed == [user_id]

    # The stored config carries the snapshot the fire-time dispatch reads.
    from app.db import async_session_maker
    from app.db.models import Routine

    async with async_session_maker() as db:
        routine = await db.get(Routine, payload["reminder"]["id"])
        assert routine.config_json["delivery_channels"] == ["website", "telegram"]


@pytest.mark.asyncio
async def test_remind_explicit_channels_pass_through(monkeypatch):
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    async def must_not_probe(uid, maker):  # pragma: no cover — the assert
        raise AssertionError("explicit list must skip the probe")

    monkeypatch.setattr(cd, "get_connected_channels", must_not_probe)

    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__remind",
        {"reminder_text": "stretch", "when": "daily",
         "daily_at_local": "09:00", "delivery_channels": ["telegram"]},
        _ctx(),
    )
    assert not result.startswith("ERROR:"), result
    payload = json.loads(result)
    stored = payload["reminder"]["delivery_channels"]
    assert stored == ["telegram"]
    # Fire-time parse still force-includes the canonical website record.
    assert cd.parse_delivery_channels({"delivery_channels": stored}) == [
        "website", "telegram",
    ]


@pytest.mark.asyncio
async def test_create_omitted_channels_default_to_connected(monkeypatch):
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    async def fake_connected(uid, maker):
        return ["website", "whatsapp"]

    monkeypatch.setattr(cd, "get_connected_channels", fake_connected)

    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__create",
        {"kind": "agent_task", "schedule_cron_local": "0 7 * * *",
         "name": "Deploy check", "prompt_text": "Check the deploys."},
        _ctx(),
    )
    assert not result.startswith("ERROR:"), result
    payload = json.loads(result)
    assert payload["routine"]["delivery_channels"] == ["website", "whatsapp"]


@pytest.mark.asyncio
async def test_remind_probe_failure_falls_back_to_website(monkeypatch):
    from app.config import settings

    user_id = await _seed_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    async def broken(uid, maker):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(cd, "get_connected_channels", broken)

    skill = _load_skill()
    result = await skill.execute_tool(
        "routines__remind",
        {"reminder_text": "stretch", "when": "daily", "daily_at_local": "10:00"},
        _ctx(),
    )
    assert not result.startswith("ERROR:"), result
    payload = json.loads(result)
    assert payload["reminder"]["delivery_channels"] == ["website"]


# ── Prompt surfaces: the asking contract is inverted ─────────────────


def test_tool_descriptions_never_ask():
    skill = _load_skill()
    tools = {t["name"]: t for t in skill.get_tools()}
    blob = json.dumps(tools)

    assert "ALWAYS ask" not in blob
    assert "Ask the user before picking" not in blob

    create_desc = tools["routines__create"]["description"]
    remind_desc = tools["routines__remind"]["description"]
    assert "Do NOT ask" in create_desc
    assert "do NOT ask" in remind_desc

    for tool in ("routines__create", "routines__remind"):
        param = tools[tool]["input_schema"]["properties"]["delivery_channels"]
        assert "OMIT" in param["description"]
        assert "do NOT ask" in param["description"]
    # The stale website-only default is gone from the remind param.
    remind_param = tools["routines__remind"]["input_schema"]["properties"]["delivery_channels"]
    assert 'Defaults to `["website"]`' not in remind_param["description"]


def test_system_prompt_section_never_asks():
    skill = _load_skill()
    section = skill.get_system_prompt_section()

    assert "ASK WHERE" not in section
    assert "Where do you want to see this" not in section
    assert "NEVER ask where to deliver it" in section
    # Explicit restriction mapping survives — overrides keep working.
    assert '"only Telegram" → `["telegram"]`' in section


def test_agent_runner_prompt_has_no_asking_rule():
    import inspect
    import app.agent.agent_runner as ar

    src = inspect.getsource(ar)
    assert "never ask where to send them" in src
    assert (
        "delivery is automatic to chat + every connected channel" in src
    )
