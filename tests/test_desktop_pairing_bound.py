"""The shipped pairing behaviours, re-proved against a BOUND pairing.

Why this file exists
--------------------
`pair/init` now requires the Mac's account bearer and binds the pairing to
that account (CONNECTIONS.md §0, §4.1 and F-K/T5: device-code phishing).
`tests/test_desktop_pairing.py` and `tests/test_desktop_relay_socket.py`
were written against the unauthenticated `init` that shipped, so their
shared helpers now get 401 at the first line and 38 tests fail without ever
reaching the behaviour they assert.

Those helpers are the supervisor's to update — this file does not touch
them. What it does is separate "the harness starts differently" from "the
behaviour changed", by re-asserting, with a signed-in Mac, exactly the
properties the brief names as already reviewed and shipped: the device
token's own audience and scope, expiry-is-not-denial, the verification
URI's shape, `/ws/chat` refusing a device token, and the pairing session
lifecycle (one bundle, once, with no token left at rest).

It also pins §4.6, the other structural change: the relay socket moved to
`ws_router`, which only `agent_main` mounts, so a platform replica serves
no `/api/ws/desktop` and cannot answer "Online" for a Mac the agent has
never seen (F-A).

Lane: RUN_MODE=platform, as `test_desktop_pairing.py`. No COVERAGE_DEBT
entry: every table here is PLATFORM_ONLY.

Run:
    cd backend && DATABASE_URL="sqlite+aiosqlite:///:memory:" RUN_MODE=platform \
        python -m pytest tests/test_desktop_pairing_bound.py -q
"""

from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update as sa_update

from app.api import desktop as desktop_api
from app.config import settings
from app.db.database import async_session_maker
from app.db.models import DesktopDevice, DesktopPairing

from tests.test_desktop_connections_security import (  # helpers only
    Account,
    _app,
    _approve,
    _init,
    _make_account,
)


@pytest.fixture(autouse=True)
def _enabled_account_cohort(monkeypatch):
    # These tests exercise the enabled protocol; rollout-off has its own tests.
    monkeypatch.setattr(settings, "desktop_connections_rollout_pct", 100)
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)


@pytest.fixture(autouse=True)
def _fresh_rate_buckets():
    desktop_api._RATE_BUCKETS.clear()
    yield
    desktop_api._RATE_BUCKETS.clear()


@pytest.fixture
async def api() -> AsyncClient:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def mac() -> Account:
    """A signed-in Mac. No feature flag: the typed path is ungated, and
    everything in this file is the pre-QR behaviour."""
    return await _make_account()


async def _bundle(api: AsyncClient, acc: Account) -> dict:
    started = await _init(api, acc)
    ok = await _approve(api, acc, label="mac", user_code=started["user_code"])
    assert ok.status_code == 200, ok.text
    poll = await api.post("/api/desktop/pair/poll",
                          json={"device_code": started["device_code"]})
    assert poll.status_code == 200, poll.text
    return {"started": started, **poll.json()}


# ── The pairing session lifecycle ────────────────────────────────────
async def test_the_bundle_is_issued_once_and_leaves_no_token_at_rest(api, mac):
    got = await _bundle(api, mac)
    assert got["user_id"] == mac.user_id
    assert got["device_id"] and got["access_token"] and got["relay_ws_url"]

    again = await api.post("/api/desktop/pair/poll",
                           json={"device_code": got["started"]["device_code"]})
    assert again.status_code != 200

    async with async_session_maker() as db:
        row = (await db.execute(select(DesktopPairing).where(
            DesktopPairing.user_code == got["started"]["user_code"],
        ))).scalar_one()
        assert row.status == "consumed"
        assert row.approved_token is None
        dev = (await db.execute(select(DesktopDevice).where(
            DesktopDevice.id == got["device_id"],
        ))).scalar_one()

    from jose import jwt

    assert dev.token_jti == jwt.get_unverified_claims(got["access_token"])["jti"]


async def test_a_pending_poll_is_202_and_a_fast_poll_is_slow_down(api, mac):
    started = await _init(api, mac)
    first = await api.post("/api/desktop/pair/poll",
                           json={"device_code": started["device_code"]})
    assert first.status_code == 202
    assert first.json()["error"] == "authorization_pending"

    fast = await api.post("/api/desktop/pair/poll",
                          json={"device_code": started["device_code"]})
    assert fast.status_code == 429
    assert fast.json()["error"] == "slow_down"


# ── Expiry is not denial, and both are flat ──────────────────────────
#
# `PairingClient.pollOnce` reads `json["error"]` off the ROOT. Nested under
# `detail`, "you pressed Deny" and "the code ran out" both reach the Mac as
# `PairingError.server(400)`, i.e. a number.
async def test_an_expired_code_says_expired_at_the_top_level(api, mac):
    started = await _init(api, mac)
    async with async_session_maker() as db:
        await db.execute(
            sa_update(DesktopPairing)
            .where(DesktopPairing.user_code == started["user_code"])
            .values(expires_at=datetime.utcnow() - timedelta(seconds=1))
        )
        await db.commit()

    out = await api.post("/api/desktop/pair/poll",
                         json={"device_code": started["device_code"]})
    assert out.status_code == 400
    assert out.json() == {"error": "expired"}


async def test_a_denied_code_says_denied_and_not_expired(api, mac):
    started = await _init(api, mac)
    denied = await api.post("/api/desktop/pair/deny",
                            json={"user_code": started["user_code"]},
                            headers=mac.headers("mac"))
    assert denied.json()["denied"] is True

    out = await api.post("/api/desktop/pair/poll",
                         json={"device_code": started["device_code"]})
    assert out.status_code == 400
    assert out.json() == {"error": "denied"}


async def test_an_unknown_code_is_expired_rather_than_a_new_word(api):
    out = await api.post("/api/desktop/pair/poll",
                         json={"device_code": "not-a-real-device-code"})
    assert out.status_code == 400
    assert out.json() == {"error": "expired"}


# ── The verification URI the Mac is willing to open ──────────────────
async def test_the_verification_uri_carries_no_query_fragment_or_userinfo(api, mac):
    started = await _init(api, mac)
    parsed = urlparse(started["verification_uri"])
    assert parsed.scheme == "https"
    assert parsed.hostname and "@" not in parsed.netloc
    assert not parsed.query and not parsed.fragment
    assert parsed.port in (None, 443)


# ── The device token's own authority ─────────────────────────────────
async def test_the_device_token_has_its_own_aud_and_scope(api, mac):
    from jose import jwt

    token = (await _bundle(api, mac))["access_token"]
    claims = jwt.get_unverified_claims(token)
    assert claims["aud"] == "toup-desktop" != "toup-agent-session"
    assert claims["scope"] == "desktop" != "extension"
    assert claims["iss"] == "toup-platform"
    assert claims["jti"] and claims["did"]
    assert 0 < claims["exp"] - claims["iat"] <= 24 * 3600


async def test_the_device_token_is_header_safe(api, mac):
    """It rides a `Sec-WebSocket-Protocol` value, so it has to be a valid
    HTTP token: no `=`, no `,`, no space."""
    token = (await _bundle(api, mac))["access_token"]
    assert token.isascii()
    assert not any(c in token for c in (",", " ", "=", "\t"))


async def test_ws_chat_refuses_the_device_token(api, mac):
    """The divergence that matters most for a credential with filesystem
    reach: the extension's token works on both its own socket and
    `/ws/chat`, which makes its scope a label. Both of `/ws/chat`'s
    authenticators must refuse this one."""
    from app.api.ws_chat import _authenticate_ws, _authenticate_ws_session_token
    from app.services.auth_service import decode_access_token

    token = (await _bundle(api, mac))["access_token"]
    assert decode_access_token(token) is None
    assert await _authenticate_ws(token) is None
    assert await _authenticate_ws_session_token(token) is None


async def test_an_account_jwt_is_not_a_device_token(mac):
    account_jwt = mac.sessions["mac"][0]
    assert desktop_api._decode_device_token(account_jwt) is None


async def test_the_device_token_is_not_signed_with_the_account_secret(api, mac):
    from jose import jwt

    token = (await _bundle(api, mac))["access_token"]
    with pytest.raises(Exception):
        jwt.decode(token, settings.jwt_secret, algorithms=["HS256"],
                   audience="toup-desktop")


# ── §4.6: only the agent serves the socket (F-A) ─────────────────────
def _paths(app: FastAPI) -> set:
    return {getattr(r, "path", "") for r in app.routes}


def test_the_platform_facing_router_serves_no_relay_socket():
    """A platform replica that answers `/api/ws/desktop` verifies the token
    locally and stamps `last_seen_at`, so the Mac reads Online while the
    tenant agent's registry is empty and every desktop tool says "not
    connected". A route it does not serve cannot lie that way."""
    platform_like = FastAPI()
    platform_like.include_router(desktop_api.router, prefix=settings.api_prefix)
    assert "/api/ws/desktop" not in _paths(platform_like)
    # …and the pairing routes ARE still there, so this is not an empty app.
    assert "/api/desktop/pair/init" in _paths(platform_like)


def test_the_agent_facing_app_does_serve_it():
    agent_like = FastAPI()
    agent_like.include_router(desktop_api.router, prefix=settings.api_prefix)
    agent_like.include_router(desktop_api.ws_router, prefix=settings.api_prefix)
    assert "/api/ws/desktop" in _paths(agent_like)


def test_agent_main_mounts_the_ws_router():
    """The source, not the import: `agent_main` starts a scheduler and an
    MCP server, so this reads the mount rather than building the app. Without
    it the socket has no home at all and no Mac can connect."""
    import pathlib

    src = pathlib.Path("agent_main.py").read_text()
    assert "ws_router as desktop_ws_router" in src
    assert "include_router(desktop_ws_router" in src
