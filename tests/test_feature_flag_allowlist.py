"""Per-user feature-flag allowlist (R28-D) — platform lane.

The pct rollout moves whole hash buckets, so "turn this on for the
dev/test tenant, nobody else" was impossible: the R28-D simulator
round found the app pointing at a dark production where the only
lever would have enrolled a percentage of the real fleet. The
allowlist (`platform_settings` key `<setting_key>.allow`, admin
route PUT /api/admin/feature-flags/flag/{flag}/allow) is that lever.

Proves:
  - service round trip normalizes, dedupes, caps, clears
  - is_enabled: listed user ON at pct 0; everyone else exactly as
    before (unlisted OFF, anonymous OFF, pct 100 still ON for all)
  - the public readout reflects the allowlist for the listed caller
  - the admin route replaces the list and snapshots it; non-admins
    403; unknown flags 404
  - the dark-launch gate honors it: at pct 0 the automations surface
    404s for an unlisted user and serves for the listed one
"""

import uuid

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.db.database import async_session_maker
from app.services import feature_flags


@pytest_asyncio.fixture
async def app() -> FastAPI:
    from app.api.auth import router as auth_router
    from app.api.automations_proxy import router as proxy_router
    from app.api.onboarding_events import admin_router
    a = FastAPI()
    a.include_router(auth_router, prefix=settings.api_prefix)
    a.include_router(admin_router, prefix=settings.api_prefix)
    a.include_router(proxy_router, prefix=settings.api_prefix)
    return a


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac


async def _mk_user(role: str = "user") -> dict:
    from app.db import User
    from app.services.auth_service import create_access_token, get_password_hash
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"u-{uid[:8]}@example.com",
                    hashed_password=get_password_hash("pw-123456"),
                    name="U", role=role))
        await db.commit()
    return {"id": uid,
            "h": {"Authorization": f"Bearer {create_access_token(uid)}"}}


# ── Service layer ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_allowlist_round_trip_normalizes_and_clears():
    async with async_session_maker() as db:
        ids = await feature_flags.set_allowlist(
            db, "automations", ["  u1 ", "u2", "u1", "", "u3"],
        )
        assert ids == ["u1", "u2", "u3"]
        assert await feature_flags.get_allowlist(db, "automations") \
            == ["u1", "u2", "u3"]
        assert await feature_flags.set_allowlist(db, "automations", []) == []
        assert await feature_flags.get_allowlist(db, "automations") == []


@pytest.mark.asyncio
async def test_allowlist_caps_at_fifty():
    async with async_session_maker() as db:
        ids = await feature_flags.set_allowlist(
            db, "automations", [f"u{i}" for i in range(80)],
        )
        assert len(ids) == 50
        await feature_flags.set_allowlist(db, "automations", [])


@pytest.mark.asyncio
async def test_listed_user_is_on_at_pct_zero_everyone_else_untouched():
    listed = str(uuid.uuid4())
    unlisted = str(uuid.uuid4())
    async with async_session_maker() as db:
        await feature_flags.set_rollout_pct(db, "automations", 0)
        await feature_flags.set_allowlist(db, "automations", [listed])
        assert await feature_flags.is_enabled(db, "automations", listed) is True
        assert await feature_flags.is_enabled(db, "automations", unlisted) is False
        assert await feature_flags.is_enabled(db, "automations", None) is False
        # pct 100 still turns everyone on — the allowlist never restricts.
        await feature_flags.set_rollout_pct(db, "automations", 100)
        assert await feature_flags.is_enabled(db, "automations", unlisted) is True
        # cleanup
        await feature_flags.set_rollout_pct(db, "automations", 0)
        await feature_flags.set_allowlist(db, "automations", [])


@pytest.mark.asyncio
async def test_readout_reflects_the_allowlist_for_the_listed_caller():
    listed = str(uuid.uuid4())
    async with async_session_maker() as db:
        await feature_flags.set_rollout_pct(db, "automations", 0)
        await feature_flags.set_allowlist(db, "automations", [listed])
        flags = await feature_flags.all_flags_for(db, listed)
        assert flags["automations"] is True
        flags = await feature_flags.all_flags_for(db, str(uuid.uuid4()))
        assert flags["automations"] is False
        await feature_flags.set_allowlist(db, "automations", [])


# ── Admin route ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_admin_put_allow_replaces_and_snapshots(client):
    admin = await _mk_user("admin")
    dev = await _mk_user()
    r = await client.put(
        "/api/admin/feature-flags/flag/automations/allow",
        headers=admin["h"], json={"user_ids": [dev["id"]]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["flag"] == "automations"
    assert body["allow_user_ids"] == [dev["id"]]
    assert body["rollout_pct"] == 0
    r = await client.get(
        "/api/admin/feature-flags/flag/automations", headers=admin["h"],
    )
    assert r.json()["allow_user_ids"] == [dev["id"]]
    # clear
    r = await client.put(
        "/api/admin/feature-flags/flag/automations/allow",
        headers=admin["h"], json={"user_ids": []},
    )
    assert r.json()["allow_user_ids"] == []


@pytest.mark.asyncio
async def test_non_admin_cannot_touch_the_allowlist(client):
    plain = await _mk_user()
    r = await client.put(
        "/api/admin/feature-flags/flag/automations/allow",
        headers=plain["h"], json={"user_ids": [plain["id"]]},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_unknown_flag_404s(client):
    admin = await _mk_user("admin")
    r = await client.put(
        "/api/admin/feature-flags/flag/nope/allow",
        headers=admin["h"], json={"user_ids": []},
    )
    assert r.status_code == 404


# ── Dark-launch gate integration ─────────────────────────────────────


@pytest.mark.asyncio
async def test_dark_surface_serves_the_listed_user_only(client):
    admin = await _mk_user("admin")
    dev = await _mk_user()
    other = await _mk_user()
    async with async_session_maker() as db:
        await feature_flags.set_rollout_pct(db, "automations", 0)
    r = await client.put(
        "/api/admin/feature-flags/flag/automations/allow",
        headers=admin["h"], json={"user_ids": [dev["id"]]},
    )
    assert r.status_code == 200
    r = await client.get("/api/automations/templates", headers=other["h"])
    assert r.status_code == 404
    r = await client.get("/api/automations/templates", headers=dev["h"])
    assert r.status_code == 200
    # cleanup
    await client.put(
        "/api/admin/feature-flags/flag/automations/allow",
        headers=admin["h"], json={"user_ids": []},
    )
