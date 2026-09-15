"""`/admin/bind` must carry the platform's timezone down to the tenant (D4).

E1's upstream. The bind is the only moment the platform's known `users.timezone`
can reach a freshly-provisioned tenant BEFORE its first turn — and the owner-row
insert at `admin_pool.py:248` never passed one, so every tenant was born tz-NULL
and the first channel turn without a client tz minted a UTC day chat.

The rule is NULL-FILL, not write: once the tenant row holds a local value that
value is authority (ws_chat learns it from `msg.tz`), and a bind that is silent
about tz must never clear it. Four cases below say exactly that, and the last
one says a rejected tz never fails the bind.

A NEW file rather than an extension of test_admin_bind_identity.py: CI routes
one process per file and this one seeds `users`, which that file does not.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_admin_bind_owner_row.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import os
from typing import Any, Optional
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import settings

_TOKEN = "pool-secret-owner-row"


@pytest.fixture(autouse=True)
def _pool_token(monkeypatch):
    monkeypatch.setenv("POOL_ADMIN_TOKEN", _TOKEN)


def _require_bind_tz():
    """Skip until lane B3 adds `user_timezone` to the bind contract."""
    from app.api import admin_pool

    if "user_timezone" not in admin_pool._BIND_FIELDS:
        pytest.skip("'user_timezone' not in admin_pool._BIND_FIELDS yet — lane B3 (D4)")


async def _seed_user(user_id: str, tz: Optional[str]) -> None:
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"{user_id[:8]}@agent.local",
            hashed_password="", name="Agent Owner", timezone=tz,
        ))
        await db.commit()


async def _read_user(user_id: str):
    from sqlalchemy import select
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        return (
            await db.execute(select(User).where(User.id == user_id))
        ).scalar_one_or_none()


async def _bind(payload: dict[str, Any]) -> int:
    """POST the payload to the REAL /admin/bind and return the status code.

    Harness copied from tests/test_admin_bind_identity.py:60-92 — runtime.json
    is not writable here and MCP bootstrap would make a network call; nothing
    else in the handler is stubbed."""
    from app.api.admin_pool import router as admin_router

    app = FastAPI()
    app.include_router(admin_router)

    saved_settings = {"user_id": settings.user_id, "agent_api_key": settings.agent_api_key}
    saved_env = {k: os.environ.get(k) for k in ("USER_ID", "AGENT_API_KEY")}
    try:
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


@pytest.mark.asyncio
async def test_bind_seeds_the_timezone_on_a_created_owner_row():
    """THE fix. A fresh tenant's very first bind is the only chance to be
    right before the first channel turn resolves a day."""
    _require_bind_tz()
    uid = "bt-crt-0001"

    assert await _bind({
        "user_id": uid, "agent_api_key": "k1", "user_timezone": "America/Toronto",
    }) == 200

    row = await _read_user(uid)
    assert row is not None, "the bind must still create the owner row"
    assert row.timezone == "America/Toronto"


@pytest.mark.asyncio
async def test_bind_backfills_a_null_timezone_on_an_existing_row():
    """The repair path for every tenant already provisioned tz-NULL — they
    get it on the next refresh-config, with no manual SQL."""
    _require_bind_tz()
    uid = "bt-bkf-0001"
    await _seed_user(uid, None)

    assert await _bind({
        "user_id": uid, "agent_api_key": "k1", "user_timezone": "America/Toronto",
    }) == 200

    assert (await _read_user(uid)).timezone == "America/Toronto"


@pytest.mark.asyncio
async def test_bind_never_overwrites_a_non_null_tenant_timezone():
    """The tenant is authority once it knows. ws_chat writes the value the
    device actually reports; the platform's copy can be stale by a trip."""
    _require_bind_tz()
    uid = "bt-aut-0001"
    await _seed_user(uid, "Europe/London")

    assert await _bind({
        "user_id": uid, "agent_api_key": "k1", "user_timezone": "America/Toronto",
    }) == 200

    assert (await _read_user(uid)).timezone == "Europe/London"


@pytest.mark.asyncio
async def test_a_silent_payload_never_clears_a_stored_timezone():
    """The `agent_name` lesson (test_admin_bind_identity.py's whole subject),
    applied to tz: absent / null / '' all mean 'the platform has nothing to
    tell me', never 'clear it'. An OLD bridge omits the key entirely."""
    _require_bind_tz()
    for uid, payload in (
        ("bt-abs-0001", {}),
        ("bt-nul-0001", {"user_timezone": None}),
        ("bt-blk-0001", {"user_timezone": "   "}),
    ):
        await _seed_user(uid, "Europe/London")
        assert await _bind({"user_id": uid, "agent_api_key": "k1", **payload}) == 200
        assert (await _read_user(uid)).timezone == "Europe/London", uid


@pytest.mark.asyncio
async def test_an_unloadable_timezone_is_rejected_and_the_bind_still_succeeds():
    """A bad tz is a bad field, not a bad bind — the owner-row block is
    non-fatal by construction (admin_pool.py:276) and must stay that way, or
    one malformed profile value strands a container unbound."""
    _require_bind_tz()
    uid = "bt-inv-0001"

    assert await _bind({
        "user_id": uid, "agent_api_key": "k1", "user_timezone": "Mars/Olympus",
    }) == 200

    row = await _read_user(uid)
    assert row is not None
    assert row.timezone is None, "an unloadable tz must not be stored"


@pytest.mark.asyncio
async def test_utc_is_never_stored_from_a_bind():
    """'UTC' is the unknown-state default on the agent side (D4). Persisting
    it makes 'we never learned' indistinguishable from 'they live there'."""
    _require_bind_tz()
    uid = "bt-utc-0001"

    assert await _bind({"user_id": uid, "agent_api_key": "k1", "user_timezone": "UTC"}) == 200
    assert (await _read_user(uid)).timezone is None
