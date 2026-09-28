"""Toup for Mac — device-code pairing and the device token's own authority.

What this file pins
-------------------
`desktop/docs/RELAY_PROTOCOL.md` §1.1-§1.3 (the pairing flow), §1.2's answer
vocabulary, §2 (the token's own `aud`/`scope`, refused wherever an account
JWT is accepted), and §7.1 (the account the bundle names is the account that
approved).

The answer vocabulary is not decoration. `PairingClient.pollOnce`
(`desktop/Packages/ToupRelay/Sources/ToupRelay/PairingClient.swift:156-176`)
reads `json["error"]` at the TOP LEVEL of the body and maps `"denied"` /
`"expired"` to the two outcomes the user is shown; anything else becomes
`PairingError.server(status:)`, i.e. "the server answered 400". So a body
shaped `{"detail": {"error": "denied"}}` — which is what
`raise HTTPException(400, {"error": "denied"})` produces — turns "you said
no on the website" into an opaque number on the Mac. The tests below assert
the flat shape both the protocol document and the reference server
(`desktop/mock/relay/relay.mjs:79-93`) emit.

Lane: RUN_MODE=platform. All three `desktop_*` tables are PLATFORM_ONLY, so
this file belongs in the platform sweep and needs no COVERAGE_DEBT entry.

Run:
    cd backend && DATABASE_URL="sqlite+aiosqlite:///:memory:" RUN_MODE=platform \
        python -m pytest tests/test_desktop_pairing.py -q
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update as sa_update

from app.api import desktop as desktop_api
from app.config import settings
from app.db.database import async_session_maker
from app.db.models import DesktopDevice, DesktopPairing, UserSession


# ── A test app carrying only what pairing touches ────────────────────
def _app() -> FastAPI:
    from app.api.auth import router as auth_router

    app = FastAPI()
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(desktop_api.router, prefix=settings.api_prefix)
    return app


@pytest.fixture(autouse=True)
def _enabled_account_cohort(monkeypatch):
    # These tests exercise the enabled protocol; rollout-off has its own tests.
    monkeypatch.setattr(settings, "desktop_connections_rollout_pct", 100)
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)


@pytest.fixture(autouse=True)
def _fresh_rate_buckets():
    """`_RATE_BUCKETS` is module-level and per-process, so the 10/min/IP cap
    on `pair/init` is shared by every test in this file (and, in production,
    by every user behind one NAT — see the return notes)."""
    desktop_api._RATE_BUCKETS.clear()
    yield
    desktop_api._RATE_BUCKETS.clear()


@pytest.fixture(autouse=True)
async def _agent_provisioned(test_user_id):
    """An account whose agent container exists.

    `pair/approve` refuses when there is no `relay_ws_url` to hand over —
    a bundle without one spends the one-shot device code on a credential
    the Mac's `RelayHostPolicy` will refuse (§4). `pool_service` mints
    `https://agent-<prefix>.agents.toup.ai`, which is what the host policy
    accepts, so that is the shape used here."""
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=test_user_id,
            agent_url="https://agent-testtenant.agents.toup.ai",
            deploy_status="active",
        ))
        await db.commit()
    yield


@pytest.fixture
async def api() -> AsyncClient:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# The Mac's own sign-in, as `pair/init` now requires it.
#
# `init` used to need no login at all, which is exactly what F-K
# (CONNECTIONS.md §1.2, threat T5) turns into a device token on someone
# else's account: an attacker starts the pairing, a victim approves the
# code they were shown, and the attacker polls and collects. The route
# now binds the pairing to the caller's account AND to the caller's
# session, so a bearer alone is not enough — `_session_jti` reads
# `request.state.user_session_jti`, which `get_current_user` sets only
# when a live `user_sessions` row carries that JWT's `jti`
# (`app/api/auth.py:262-295`). With no row the route answers 403
# `session_unknown`. Same construction as
# `tests/test_desktop_connections_security.py::_sign_in`.
_MAC_HEADERS: dict[str, str] | None = None


@pytest.fixture(autouse=True)
async def _mac_signed_in(_test_user):
    from app.services.auth_service import decode_platform_jwt

    jti = decode_platform_jwt(_test_user["token"]).get("jti")
    async with async_session_maker() as db:
        db.add(UserSession(
            id=str(uuid.uuid4()), user_id=_test_user["id"], jti=jti,
            device_label="Mac", created_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(), is_revoked=False,
        ))
        await db.commit()

    global _MAC_HEADERS
    _MAC_HEADERS = {"Authorization": f"Bearer {_test_user['token']}"}
    yield
    _MAC_HEADERS = None


async def _init(api: AsyncClient, **over) -> dict:
    body = {
        "device_name": "Nariman's MacBook Pro",
        "app_version": "0.1.0",
        "os_version": "15.7",
        "flavor": "direct",
    }
    body.update(over)
    r = await api.post(
        "/api/desktop/pair/init", json=body, headers=_MAC_HEADERS,
    )
    assert r.status_code == 200, r.text
    return r.json()


# ══════════════════════════════════════════════════════════════════════
# 1. The happy path
# ══════════════════════════════════════════════════════════════════════
async def test_pairing_happy_path_issues_one_bundle(api, auth_headers, test_user_id):
    """init → lookup → approve → poll, and the bundle is collectable ONCE."""
    started = await _init(api)
    assert started["expires_in"] == desktop_api.PAIR_TTL_S
    assert started["interval"] == desktop_api.PAIR_POLL_INTERVAL_S
    assert len(started["device_code"]) >= 22          # >= 128 bits, url-safe
    assert "-" in started["user_code"]

    look = await api.get(
        "/api/desktop/pair/lookup",
        params={"user_code": started["user_code"]},
        headers=auth_headers,
    )
    assert look.status_code == 200, look.text
    assert look.json()["device_name"] == "Nariman's MacBook Pro"
    assert look.json()["flavor"] == "direct"

    ok = await api.post(
        "/api/desktop/pair/approve",
        json={"user_code": started["user_code"]},
        headers=auth_headers,
    )
    assert ok.status_code == 200, ok.text
    device_id = ok.json()["device_id"]

    poll = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert poll.status_code == 200, poll.text
    bundle = poll.json()
    assert bundle["access_token"]
    assert bundle["device_id"] == device_id
    # §7.1: the app compares this to the account signed in inside the app and
    # discards the token BEFORE the Keychain if they differ.
    assert bundle["user_id"] == test_user_id
    assert bundle["expires_in"] == settings.desktop_token_ttl_s
    # §4: wss, a dot-suffix of toup.ai, no query, no fragment, no port —
    # `RelayHostPolicy` is the whole backstop on a client with no CSP.
    ws = bundle["relay_ws_url"]
    assert ws == "wss://agent-testtenant.agents.toup.ai/api/ws/desktop"
    assert not any(c in ws for c in ("?", "#", "@"))

    # §1.2: "the record is EVICTED, so the bundle cannot be replayed".
    again = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert again.status_code == 400
    assert again.json() == {"error": "expired"}


async def test_approved_device_row_carries_the_tokens_jti(api, auth_headers):
    """The row IS the denylist (§9.1), so the jti has to be ON it."""
    from jose import jwt

    started = await _init(api)
    await api.post(
        "/api/desktop/pair/approve",
        json={"user_code": started["user_code"]},
        headers=auth_headers,
    )
    poll = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    token = poll.json()["access_token"]
    claims = jwt.get_unverified_claims(token)

    async with async_session_maker() as db:
        row = (await db.execute(
            select(DesktopDevice).where(DesktopDevice.id == claims["did"])
        )).scalar_one()
    assert row.token_jti == claims["jti"]
    assert row.revoked_at is None
    assert row.device_name == "Nariman's MacBook Pro"


async def test_a_consumed_pairing_row_holds_no_token_at_rest(api, auth_headers):
    """The bundle sits in a row between approve and poll; after the poll it
    must not. A retained pairing record is an audit trail, not a credential
    store."""
    started = await _init(api)
    await api.post(
        "/api/desktop/pair/approve",
        json={"user_code": started["user_code"]},
        headers=auth_headers,
    )
    await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    async with async_session_maker() as db:
        row = (await db.execute(
            select(DesktopPairing).where(
                DesktopPairing.device_code == started["device_code"]
            )
        )).scalar_one()
    assert row.status == "consumed"
    assert row.approved_token is None


# ══════════════════════════════════════════════════════════════════════
# 2. The answer vocabulary the Mac actually parses (§1.2)
# ══════════════════════════════════════════════════════════════════════
async def test_pending_poll_is_202_authorization_pending(api):
    started = await _init(api)
    r = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert r.status_code == 202
    # Still exact equality, and still the whole body: the two fields the design
    # added (CONNECTIONS.md §4.1) are what lets the Mac take its QR off the
    # screen and name the phone holding the code, and nothing else belongs here.
    assert r.json() == {
        "error": "authorization_pending",
        "claimed": False,
        "claimed_on": None,
    }


async def test_polling_faster_than_the_interval_is_429_slow_down(api):
    started = await _init(api)
    first = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert first.status_code == 202
    second = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert second.status_code == 429
    assert second.json() == {"error": "slow_down"}


async def test_an_expired_code_says_expired_in_a_flat_body(api):
    """The client reads `error` at the top level. `{"detail": {...}}` is a
    400 with no vocabulary, which it reports as "the server answered 400"."""
    started = await _init(api)
    async with async_session_maker() as db:
        await db.execute(
            sa_update(DesktopPairing)
            .where(DesktopPairing.device_code == started["device_code"])
            .values(expires_at=datetime.utcnow() - timedelta(seconds=1))
        )
        await db.commit()

    r = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert r.status_code == 400
    assert r.json() == {"error": "expired"}, (
        "PairingClient.pollOnce reads json[\"error\"] at the top level; a "
        "nested detail object is an unreadable 400 on the Mac"
    )


async def test_a_code_swept_by_SOMEONE_ELSES_request_still_says_expired(api):
    """The lazy sweep runs on `pair/init`, `pair/lookup` and `pair/approve`,
    so a code that timed out is routinely rewritten before its own device
    polls again. Writing `denied` there made that poll answer
    `{"error": "denied"}` — which `PairingClient` maps to `.denied` — and
    the Mac told its user they had declined a pairing on a page they never
    opened. A clock and a person are different answers (§1.2)."""
    started = await _init(api)
    async with async_session_maker() as db:
        await db.execute(
            sa_update(DesktopPairing)
            .where(DesktopPairing.device_code == started["device_code"])
            .values(expires_at=datetime.utcnow() - timedelta(seconds=1))
        )
        await db.commit()

    # Someone else's request runs the sweep first.
    await _init(api)
    async with async_session_maker() as db:
        swept = (await db.execute(
            select(DesktopPairing).where(
                DesktopPairing.device_code == started["device_code"]
            )
        )).scalar_one()
    assert swept.status == "expired", (
        "the sweep wrote a status that means 'the user said no'"
    )

    r = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert r.status_code == 400
    assert r.json() == {"error": "expired"}


async def test_an_unknown_code_answers_expired_and_not_a_different_word(api):
    """Unknown and expired are one answer on purpose — a device code is a
    bearer and "that code exists but is not yours" is an oracle."""
    r = await api.post(
        "/api/desktop/pair/poll", json={"device_code": "x" * 40},
    )
    assert r.status_code == 400
    assert r.json() == {"error": "expired"}


async def test_a_denied_code_says_denied_in_a_flat_body(api, auth_headers):
    started = await _init(api)
    deny = await api.post(
        "/api/desktop/pair/deny",
        json={"user_code": started["user_code"]},
        headers=auth_headers,
    )
    assert deny.status_code == 200 and deny.json()["denied"] is True

    r = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert r.status_code == 400
    assert r.json() == {"error": "denied"}, (
        "a user who pressed Deny must be told that, not given a bare 400"
    )


async def test_approving_with_no_agent_to_connect_to_mints_nothing(
    api, auth_headers, test_user_id,
):
    """A bundle with no `relay_ws_url` is a dead pairing: the Mac's
    `RelayHostPolicy` refuses the empty string (§4), and by then the
    one-shot device code is spent and a device row exists. Refuse the
    approval instead, and say which of the two things is missing."""
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == test_user_id)
        )).scalar_one()
        cfg.agent_url = None
        cfg.deploy_status = "none"
        await db.commit()

    started = await _init(api)
    resp = await api.post(
        "/api/desktop/pair/approve",
        json={"user_code": started["user_code"]}, headers=auth_headers,
    )
    assert resp.status_code == 409
    assert "isn't running yet" in resp.json()["detail"]

    async with async_session_maker() as db:
        assert (await db.execute(select(DesktopDevice))).scalars().all() == []
        row = (await db.execute(
            select(DesktopPairing).where(
                DesktopPairing.device_code == started["device_code"]
            )
        )).scalar_one()
    # Still tappable once the agent comes up — nothing was consumed.
    assert row.status == "pending"
    assert row.approved_token is None


async def test_approving_a_denied_code_is_refused(api, auth_headers):
    started = await _init(api)
    await api.post(
        "/api/desktop/pair/deny",
        json={"user_code": started["user_code"]},
        headers=auth_headers,
    )
    r = await api.post(
        "/api/desktop/pair/approve",
        json={"user_code": started["user_code"]},
        headers=auth_headers,
    )
    assert r.status_code == 409


async def test_lookup_refuses_an_expired_code(api, auth_headers):
    started = await _init(api)
    async with async_session_maker() as db:
        await db.execute(
            sa_update(DesktopPairing)
            .where(DesktopPairing.device_code == started["device_code"])
            .values(expires_at=datetime.utcnow() - timedelta(seconds=1))
        )
        await db.commit()
    r = await api.get(
        "/api/desktop/pair/lookup",
        params={"user_code": started["user_code"]},
        headers=auth_headers,
    )
    assert r.status_code == 404


async def test_lookup_and_approve_require_a_signed_in_user(api):
    started = await _init(api)
    assert (await api.get(
        "/api/desktop/pair/lookup", params={"user_code": started["user_code"]},
    )).status_code in (401, 403)
    assert (await api.post(
        "/api/desktop/pair/approve", json={"user_code": started["user_code"]},
    )).status_code in (401, 403)


# ══════════════════════════════════════════════════════════════════════
# 3. §4 — what may be emitted in a URL
# ══════════════════════════════════════════════════════════════════════
async def test_verification_uri_carries_no_query_fragment_or_userinfo(api):
    """`RelayHostPolicy` is the whole backstop on a client with no CSP, and
    it refuses any of these. Emitting one is our bug, not the Mac's."""
    started = await _init(api)
    uri = started["verification_uri"]
    assert uri.startswith("https://") or uri.startswith("http://localhost"), uri
    assert "?" not in uri and "#" not in uri and "@" not in uri
    assert uri.endswith("/settings/desktop/pair")


# ══════════════════════════════════════════════════════════════════════
# 4. §2 / §9.2 — the token is NOT an account credential
# ══════════════════════════════════════════════════════════════════════
async def _paired_token(api: AsyncClient, auth_headers: dict) -> str:
    started = await _init(api)
    await api.post(
        "/api/desktop/pair/approve",
        json={"user_code": started["user_code"]},
        headers=auth_headers,
    )
    poll = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert poll.status_code == 200, poll.text
    return poll.json()["access_token"]


async def test_the_device_token_has_its_own_aud_and_scope(api, auth_headers):
    from jose import jwt

    token = await _paired_token(api, auth_headers)
    claims = jwt.get_unverified_claims(token)
    assert claims["aud"] == "toup-desktop" != "toup-agent-session"
    assert claims["scope"] == "desktop" != "extension"
    assert claims["iss"] == "toup-platform"
    assert claims["jti"] and claims["did"]
    # §2: hours, not the extension's 90 days.
    assert 0 < claims["exp"] - claims["iat"] <= 24 * 3600


async def test_the_device_token_is_header_safe(api, auth_headers):
    """§3.1: it rides a `Sec-WebSocket-Protocol` value, so it must be a valid
    HTTP token — no `=`, no `,`, no space. The Mac raises before opening a
    socket rather than presenting a mangled header as a mysterious 4001."""
    token = await _paired_token(api, auth_headers)
    assert token.isascii()
    assert not any(c in token for c in (",", " ", "=", "\t"))


async def test_ws_chat_refuses_the_device_token(api, auth_headers):
    """§9.2, the divergence that matters most for a credential with
    filesystem reach: the extension's token deliberately works on BOTH
    `/ws/extension` and `/ws/chat`, which makes its scope a label rather
    than a restriction. Both of `/ws/chat`'s authenticators must refuse
    this one."""
    from app.api.ws_chat import _authenticate_ws, _authenticate_ws_session_token
    from app.services.auth_service import decode_access_token

    token = await _paired_token(api, auth_headers)

    # The account-JWT path: wrong signing key, and a non-"full" scope.
    assert decode_access_token(token) is None
    assert await _authenticate_ws(token) is None
    # The direct-to-agent session-token path: wrong audience, wrong key.
    assert await _authenticate_ws_session_token(token) is None


async def test_an_account_jwt_is_not_a_device_token(_test_user):
    """The other direction. Domain separation means the signature fails
    first, which is a stronger boundary than a claim check."""
    assert desktop_api._decode_device_token(_test_user["token"]) is None


async def test_the_device_token_is_not_signed_with_the_account_jwt_secret(
    api, auth_headers,
):
    """§2's fourth requirement, as a fact rather than a comment: a reader
    holding `jwt_secret` cannot verify this token."""
    from jose import jwt

    token = await _paired_token(api, auth_headers)
    with pytest.raises(Exception):
        jwt.decode(
            token, settings.jwt_secret, algorithms=["HS256"],
            audience="toup-desktop", issuer="toup-platform",
        )
    # …and the key it IS signed with is derived, not the agent's API key.
    assert desktop_api._signing_key() != (settings.agent_api_key or "")
    assert desktop_api._signing_key() != settings.jwt_secret
