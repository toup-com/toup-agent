"""Regression for the 2026-07-04 "Connection lost — try sending again" lockout.

The WS proxy hard-gated on AgentConfig.deploy_status == "active". That field
is a wizard/deploy-flow label that drifts (reclaim-healed rows, interrupted
provisions), while agent_url + agent_api_key are the operative facts — the
reclaim sweep successfully auth-probed agents whose status string never said
"active". The gate locked such users (parmida/c847af75) out of chat forever:
the proxy looped "agent starting" and zero bytes ever reached their healthy,
bound, key-accepting agent.

deploy_status must be a HINT: with url+key present the proxy must return
connection info regardless of the label; a genuinely-down agent still fails
at the upstream connect into the same retry path.
"""
import os
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import pytest


async def _seed_cfg(status):
    from app.db import async_session_maker
    from app.db.models import User, AgentConfig
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="", name="T"))
        await db.commit()
        db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                           agent_url=f"https://agent-{uid[:8]}.agents.toup.ai",
                           agent_api_key="k-" + uid[:8], deploy_status=status))
        await db.commit()
    return uid


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["none", "pending", "provisioning", "error", "active", None])
async def test_url_and_key_beat_status_label(status):
    from app.api.ws_chat_proxy import _get_agent_ws_info
    uid = await _seed_cfg(status)
    info = await _get_agent_ws_info(uid)
    assert info is not None, f"status={status!r} must not block a user with url+key"
    ws_url, key = info
    assert ws_url.startswith("wss://") and ws_url.endswith("/api/ws/chat")
    assert key == "k-" + uid[:8]


@pytest.mark.asyncio
async def test_missing_key_still_blocks():
    from app.db import async_session_maker
    from app.db.models import User, AgentConfig
    from app.api.ws_chat_proxy import _get_agent_ws_info
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="", name="T"))
        await db.commit()
        db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                           agent_url=f"https://agent-{uid[:8]}.agents.toup.ai",
                           deploy_status="active"))
        await db.commit()
    assert await _get_agent_ws_info(uid) is None, "no key = genuinely unroutable"
