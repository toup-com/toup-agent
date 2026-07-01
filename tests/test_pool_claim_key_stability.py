"""Regression for the 2026-06-30 / 2026-07-01 double-bind → "Authentication
required" incident.

Root cause: two near-simultaneous pool claims for one user each read
agent_config.agent_api_key=NULL and minted a DIFFERENT random key. The platform
DB kept one key while Caddy routed to the container bound with the other, so the
chat WebSocket JWT (HS256-signed with the DB key, verified by the agent with its
bound copy) failed signature verification on the routed agent — every message
came back "Authentication required".

Fix: `_build_bind_payload` mints agent_api_key via an atomic compare-and-set
(`UPDATE ... WHERE agent_api_key IS NULL`) so exactly one racer wins and every
claim for the user carries an IDENTICAL key. These tests pin that invariant.
"""
import os
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import pytest
from sqlalchemy import select


async def _seed(uid: str) -> None:
    from app.db import async_session_maker
    from app.db.models import User, AgentConfig
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="", name="T"))
        db.add(AgentConfig(user_id=uid))
        await db.commit()


async def _cfg(db, uid: str):
    from app.db.models import AgentConfig
    return (await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == uid)
    )).scalar_one()


@pytest.mark.asyncio
async def test_agent_api_key_minted_and_persisted():
    """A fresh claim mints a key AND persists it to the DB (not just returns it),
    so the platform signs the session JWT with the same value the agent is bound
    with."""
    from app.db import async_session_maker
    from app.services.pool_service import _build_bind_payload

    uid = str(uuid.uuid4())
    await _seed(uid)

    async with async_session_maker() as db:
        cfg = await _cfg(db, uid)
        assert not cfg.agent_api_key  # starts NULL
        payload = await _build_bind_payload(db, uid, cfg)
        await db.commit()

    key = payload["agent_api_key"]
    assert key, "a key must be minted"

    async with async_session_maker() as db:
        cfg = await _cfg(db, uid)
        assert cfg.agent_api_key == key, "minted key must be persisted to agent_config"


@pytest.mark.asyncio
async def test_agent_api_key_is_reused_never_regenerated():
    """Every subsequent claim reuses the SAME key — never regenerates. This is
    what makes a double-bind non-fatal: both containers get an identical key, so
    the routed one can always verify the session JWT."""
    from app.db import async_session_maker
    from app.services.pool_service import _build_bind_payload

    uid = str(uuid.uuid4())
    await _seed(uid)

    async with async_session_maker() as db:
        cfg = await _cfg(db, uid)
        first = (await _build_bind_payload(db, uid, cfg))["agent_api_key"]
        await db.commit()

    async with async_session_maker() as db:
        cfg = await _cfg(db, uid)
        second = (await _build_bind_payload(db, uid, cfg))["agent_api_key"]
        await db.commit()

    # A third, from a session that never observed the mint.
    async with async_session_maker() as db:
        cfg = await _cfg(db, uid)
        third = (await _build_bind_payload(db, uid, cfg))["agent_api_key"]

    assert first == second == third, "agent_api_key must be stable across claims"
