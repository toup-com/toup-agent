"""`/admin/bind` must not erase the agent's name when the payload is silent.

The bind handler's step 2b-2 used to read `agent_name` out of `filtered` — the
whitelist comprehension that drops missing AND null keys — and then write the
result unconditionally. So a payload that said nothing about the name NULLed
whatever the user had chosen. The platform's column is transiently empty around
a failed or racing Soul save, and both Railway replicas push `refresh-config`
on their own schedule, so a bind lands in that window and copies the blank down.

NOTE ON SCOPE. The wider reading — "every bind NULLs the name" — is FALSE and
was checked against live containers: pool-17's runtime.json carries
`agent_name='Aria'` and its tenant row still read 'Aria' after binds on 10 Sep.
f261b564's NULL was inherited from a platform column that was itself NULL
because his Soul save rolled back. The defect fixed here is the narrow one: an
empty value in the bind payload — absent, null or "" alike — must mean "the
platform has nothing to tell me", never "clear it". `POST /api/soul` is the
only writer permitted to clear that column, on both sides.

These tests drive the REAL handler over ASGI against the per-test schema, and
assert on the `agent_configs` row the handler actually wrote.
"""
from __future__ import annotations

import os
from typing import Any, Optional
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import settings

_TOKEN = "pool-secret-identity"


async def _read_agent_name(user_id: str) -> tuple[bool, Optional[str]]:
    """(row_exists, agent_name) straight from the tenant DB."""
    from sqlalchemy import select

    from app.db.database import async_session_maker
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        row = (
            await db.execute(select(AgentConfig).where(AgentConfig.user_id == user_id))
        ).scalar_one_or_none()
        return (row is not None), (row.agent_name if row is not None else None)


async def _seed_agent_config(user_id: str, name: Optional[str], color: str = "#F472B6") -> None:
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=user_id, agent_name=name, agent_color=color))
        await db.commit()


async def _bind(payload: dict[str, Any]) -> int:
    """POST the payload to the real /admin/bind and return the status code."""
    from app.api.admin_pool import router as admin_router

    app = FastAPI()
    app.include_router(admin_router)

    saved_settings = {
        "user_id": settings.user_id,
        "agent_api_key": settings.agent_api_key,
    }
    saved_env = {k: os.environ.get(k) for k in ("USER_ID", "AGENT_API_KEY")}
    try:
        # runtime.json lives at /etc/toup-agent in containers (not writable
        # here); MCP bootstrap would make a real network call. Neither is what
        # this test is about — everything else in the handler runs for real.
        with patch("app.services.runtime_identity.write_runtime"), patch(
            "app.agent.mcp_bootstrap.ensure_mcp_initialized", return_value="skipped"
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as http:
                r = await http.post(
                    "/admin/bind", json=payload, headers={"X-Pool-Admin-Token": _TOKEN}
                )
                return r.status_code
    finally:
        for k, v in saved_settings.items():
            object.__setattr__(settings, k, v)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def _pool_token(monkeypatch):
    monkeypatch.setenv("POOL_ADMIN_TOKEN", _TOKEN)


@pytest.mark.asyncio
async def test_bind_without_agent_name_leaves_the_stored_name_alone():
    """THE regression. The bridge's 7-field refresh-config omits `agent_name`;
    before the fix this wrote NULL over the user's chosen name."""
    uid = "u-name-absent"
    await _seed_agent_config(uid, "Aria")

    assert await _bind({"user_id": uid, "agent_api_key": "k1"}) == 200

    exists, name = await _read_agent_name(uid)
    assert exists, "the bind must not delete the row"
    assert name == "Aria", (
        f"a bind that never mentioned agent_name erased it (got {name!r}) — "
        "this is the f261b564 symptom"
    )


@pytest.mark.asyncio
async def test_bind_with_agent_name_sets_it():
    """The other direction must keep working: an explicit name is authoritative."""
    uid = "u-name-present"
    await _seed_agent_config(uid, "Aria")

    assert await _bind({"user_id": uid, "agent_api_key": "k1", "agent_name": "Nova"}) == 200

    _, name = await _read_agent_name(uid)
    assert name == "Nova"


@pytest.mark.asyncio
async def test_bind_with_an_empty_agent_name_does_not_clear_it():
    """An explicitly empty string is the platform saying "I hold nothing",
    which is exactly what its column says during the Soul-save race. A bind
    may never clear a name; `POST /api/soul` is the only writer that may."""
    uid = "u-name-blank"
    await _seed_agent_config(uid, "Aria")

    assert await _bind({"user_id": uid, "agent_api_key": "k1", "agent_name": "   "}) == 200

    exists, name = await _read_agent_name(uid)
    assert exists
    assert name == "Aria", "a blank in the bind payload cleared the tenant's name"


@pytest.mark.asyncio
async def test_bind_with_null_agent_name_leaves_the_stored_name_alone():
    """JSON null is the same statement as absent and as "" — and by the time
    it reaches the whitelist filter it is literally indistinguishable from
    absence, so all three must fail the same safe way."""
    uid = "u-name-null"
    await _seed_agent_config(uid, "Aria")

    assert await _bind({"user_id": uid, "agent_api_key": "k1", "agent_name": None}) == 200

    _, name = await _read_agent_name(uid)
    assert name == "Aria"


@pytest.mark.asyncio
async def test_a_blank_bind_over_a_nameless_tenant_is_a_quiet_no_op():
    """The common case — a fresh signup who has not named their agent yet —
    must not start logging a kept-name line on every refresh-config."""
    uid = "u-name-none-yet"
    await _seed_agent_config(uid, None)

    assert await _bind({"user_id": uid, "agent_api_key": "k1"}) == 200

    exists, name = await _read_agent_name(uid)
    assert exists and name is None


@pytest.mark.asyncio
async def test_bind_still_creates_a_row_for_a_fresh_user_with_a_name():
    """A slot claimed by a user the tenant DB has never seen gets its row."""
    uid = "u-fresh"
    exists, _ = await _read_agent_name(uid)
    assert not exists

    assert await _bind(
        {"user_id": uid, "agent_api_key": "k1", "agent_name": "Iris", "agent_color": "#00FF00"}
    ) == 200

    exists, name = await _read_agent_name(uid)
    assert exists and name == "Iris"


@pytest.mark.asyncio
async def test_bind_without_color_leaves_the_stored_color_alone():
    """`agent_color` already behaved this way; lock it so the symmetry survives."""
    from sqlalchemy import select

    from app.db.database import async_session_maker
    from app.db.models import AgentConfig

    uid = "u-color-absent"
    await _seed_agent_config(uid, "Aria", color="#123456")

    assert await _bind({"user_id": uid, "agent_api_key": "k1"}) == 200

    async with async_session_maker() as db:
        row = (
            await db.execute(select(AgentConfig).where(AgentConfig.user_id == uid))
        ).scalar_one_or_none()
    assert row is not None and row.agent_color == "#123456"
