"""Autopilot hand-off + missions API + proxy (Autopilot PR8)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.models import Routine


async def _mk_user(tz: str | None = "America/Toronto") -> str:
    from app.db import async_session_maker
    from app.db.models import User
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"ho-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="HO",
            timezone=tz,
        ))
        await db.commit()
    return user_id


def _agent_api():
    from fastapi import FastAPI
    from app.api.autopilot import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


# ── create_mission helper ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_mission_persists_and_fails_loudly_without_tz():
    from app.api.autopilot import MissionCreateError, create_mission

    user_id = await _mk_user()
    routine = await create_mission(
        user_id=user_id, goal="research and draft the Q3 memo",
        budget_credits=42, urgent=True,
    )
    assert routine.kind == "autopilot" and routine.enabled is True
    assert routine.schedule_kind == "every"
    assert routine.config_json["budget_credits"] == 42.0
    assert routine.config_json["urgent"] is True
    assert routine.last_state_json == {"status": "active"}

    # tz-less user: RoutineRunner would silently never register the
    # trigger — creation must fail loudly instead (routines lesson).
    tzless = await _mk_user(tz=None)
    with pytest.raises(MissionCreateError) as exc:
        await create_mission(user_id=tzless, goal="whatever")
    assert exc.value.reason == "missing_timezone"


# ── Missions API (agent-side) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_mission_lifecycle_endpoints(monkeypatch):
    from app.config import settings

    user_id = await _mk_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    transport = ASGITransport(app=_agent_api())
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        res = await ac.post("/api/autopilot/missions", json={
            "goal": "book the dentist", "name": "Dentist", "budget_credits": 25,
        })
        assert res.status_code == 200, res.text
        mission = res.json()
        assert mission["status"] == "active" and mission["enabled"] is True
        mid = mission["id"]

        res = await ac.get("/api/autopilot/missions")
        assert [m["id"] for m in res.json()] == [mid]

        res = await ac.post(f"/api/autopilot/missions/{mid}/pause")
        assert res.json()["enabled"] is False

        res = await ac.post(f"/api/autopilot/missions/{mid}/resume")
        body = res.json()
        assert body["enabled"] is True and body["status"] == "active"

        res = await ac.post(f"/api/autopilot/missions/{mid}/cancel")
        body = res.json()
        assert body["enabled"] is False and body["status"] == "cancelled"

        res = await ac.get(f"/api/autopilot/missions/{mid}")
        assert res.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_resume_unblocks_blocked_mission(monkeypatch):
    from app.config import settings
    from app.api.autopilot import create_mission
    from app.db import async_session_maker
    from sqlalchemy import update as sa_update

    user_id = await _mk_user()
    monkeypatch.setattr(settings, "user_id", user_id)
    routine = await create_mission(user_id=user_id, goal="g")
    async with async_session_maker() as db:
        await db.execute(
            sa_update(Routine).where(Routine.id == routine.id).values(
                enabled=False,
                last_state_json={"status": "blocked",
                                 "status_reason": "budget_exhausted",
                                 "platform_fail_streak": 2},
            )
        )
        await db.commit()

    transport = ASGITransport(app=_agent_api())
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        res = await ac.post(f"/api/autopilot/missions/{routine.id}/resume")
    body = res.json()
    assert body["enabled"] is True
    assert body["status"] == "active"
    assert body["status_reason"] == "resumed_by_user"


# ── start_mission tool ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_mission_tool(monkeypatch):
    from app.config import settings
    from app.agent.tool_executor import ToolExecutor

    user_id = await _mk_user()
    executor = ToolExecutor.__new__(ToolExecutor)  # skip heavy __init__

    # Flag off → clean refusal, no row.
    monkeypatch.setattr(settings, "autopilot_enabled", False, raising=False)
    out = await executor._tool_start_mission({"goal": "g" * 10})
    assert out.startswith("ERROR") and "not enabled" in out

    monkeypatch.setattr(settings, "autopilot_enabled", True, raising=False)
    from app.agent.tool_executor import _USER_ID_CTX
    token = _USER_ID_CTX.set(user_id)
    try:
        out = await executor._tool_start_mission({
            "goal": "keep researching flights and draft an itinerary",
            "name": "Trip planning", "budget_credits": 30,
        })
    finally:
        _USER_ID_CTX.reset(token)
    assert out.startswith("Mission created"), out
    assert "Mission Control" in out

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        row = (await db.execute(
            select(Routine).where(Routine.user_id == user_id)
        )).scalar_one()
    assert row.kind == "autopilot" and row.name == "Trip planning"


def test_start_mission_tool_registered_and_denied_for_backgrounds():
    from app.agent.tool_definitions import get_agent_tools, get_extended_tools
    from app.agent.prompt_profile import (
        AUTOPILOT_DISABLED_TOOLS, SUBAGENT_DISABLED_TOOLS,
    )

    names = {
        t.get("name")
        for t in [*get_agent_tools(), *get_extended_tools()]
        if isinstance(t, dict)
    }
    assert "start_mission" in names
    # No recursion: missions and sub-agents cannot start missions.
    assert "start_mission" in AUTOPILOT_DISABLED_TOOLS
    assert "start_mission" in SUBAGENT_DISABLED_TOOLS


# ── Platform proxy ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_proxy_action_allowlist_and_auth(client, auth_headers, test_user_id):
    # No AgentConfig → 503 (agent not available), not an open proxy.
    res = await client.get("/api/autopilot/missions", headers=auth_headers)
    assert res.status_code == 503

    res = await client.post(
        f"/api/autopilot/missions/{uuid.uuid4()}/exec",  # not allowlisted
        headers=auth_headers,
    )
    assert res.status_code == 404

    res = await client.get("/api/autopilot/missions")  # no auth
    assert res.status_code in (401, 403)


def test_proxy_mounted_on_platform_main():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "platform_main.py").read_text()
    assert "from app.api.autopilot_proxy import router as autopilot_proxy_router" in src
    assert "autopilot_proxy_router, prefix=settings.api_prefix" in src


# ── Missions are not routines (founder bug 2026-07-16) ────────────
#
# A mission IS a Routine row (kind='autopilot'), but it must never
# surface or be managed through the generic routines API: listing it
# there leaked the "every 5 min" heartbeat internal into the Routines
# UI, delete could destroy a running mission's working state, and a
# generic PATCH could clobber config_json (goal/budget) wholesale.


@pytest.mark.asyncio
async def test_missions_hidden_from_routines_api(monkeypatch):
    from app.config import settings
    from app.api.autopilot import create_mission

    user_id = await _mk_user()
    monkeypatch.setattr(settings, "user_id", user_id)
    routine = await create_mission(user_id=user_id, goal="g", name="M")

    from fastapi import FastAPI
    from app.api.routines import router as routines_router

    app = FastAPI()
    app.include_router(routines_router, prefix="/api")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        listed = await ac.get("/api/routines")
        assert listed.status_code == 200, listed.text
        assert routine.id not in [r["id"] for r in listed.json()]

        patched = await ac.patch(
            f"/api/routines/{routine.id}", json={"enabled": False},
        )
        assert patched.status_code == 409

        deleted = await ac.delete(f"/api/routines/{routine.id}")
        assert deleted.status_code == 409


@pytest.mark.asyncio
async def test_raise_budget_unblocks_and_merges_config(monkeypatch):
    from app.config import settings
    from app.api.autopilot import create_mission
    from app.db import async_session_maker
    from sqlalchemy import update as sa_update

    user_id = await _mk_user()
    monkeypatch.setattr(settings, "user_id", user_id)
    routine = await create_mission(
        user_id=user_id, goal="research tools", budget_credits=100, urgent=True,
    )
    async with async_session_maker() as db:
        await db.execute(
            sa_update(Routine).where(Routine.id == routine.id).values(
                enabled=False,
                last_state_json={"status": "blocked",
                                 "status_reason": "budget_exhausted",
                                 "spent_credits": 218.4,
                                 "progress": 78},
            )
        )
        await db.commit()

    transport = ASGITransport(app=_agent_api())
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        # A budget below what's already spent is a no-op trap — refuse.
        low = await ac.post(
            f"/api/autopilot/missions/{routine.id}/budget",
            json={"budget_credits": 50},
        )
        assert low.status_code == 422

        res = await ac.post(
            f"/api/autopilot/missions/{routine.id}/budget",
            json={"budget_credits": 300},
        )
        body = res.json()
        assert res.status_code == 200, res.text
        assert body["budget_credits"] == 300
        assert body["enabled"] is True
        assert body["status"] == "active"
        assert body["status_reason"] == "budget_raised"
        assert body["progress"] == 78

    # config_json MERGED, not replaced — goal/urgent survive.
    async with async_session_maker() as db:
        r = await db.get(Routine, routine.id)
        assert r.config_json["goal"] == "research tools"
        assert r.config_json["urgent"] is True
        assert r.config_json["budget_credits"] == 300


# ── progress serialization is bulletproof (founder NaN% bug 2026-07-16) ──


def test_progress_int_completed_forces_100():
    from app.api.autopilot import _progress_int

    # Missions completed under pre-progress images have no key at all.
    assert _progress_int({}, "completed") == 100
    assert _progress_int({"progress": 40}, "completed") == 100


def test_progress_int_clamps_and_survives_junk():
    from app.api.autopilot import _progress_int

    assert _progress_int({"progress": 45}, "active") == 45
    assert _progress_int({"progress": "78"}, "active") == 78
    assert _progress_int({"progress": 150}, "active") == 100
    assert _progress_int({"progress": -3}, "active") == 0
    assert _progress_int({"progress": "junk"}, "active") == 0
    assert _progress_int({"progress": None}, "active") == 0
    assert _progress_int({}, "active") == 0
