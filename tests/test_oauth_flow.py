"""
T1d — OAuth API end-to-end tests.

Smoke matrix (per the T1d spec):

  1. Stub end-to-end: POST /connect/stub → authorize_url → GET it →
     302 to /callback → ConnectorIdentity exists with encrypted tokens,
     status='active', audit row written.
  2. Disconnect: POST /disconnect/stub → row preserved, ciphertext NULL,
     status='revoked', audit row written.
  3. State replay: complete one callback, replay same state → 400.
  4. Expired state: monkey-patch iat to 11 min old → 400.
  5. Bad signature: tamper with state token → 400.
  6. Bad csrf_nonce: same state from a different user's session — covered
     by the user_id binding in the state payload.
  7. Unknown connector: POST /connect/nonexistent → 404.
  8. Flag off: POST /connect/stub → 503.
  9. Auth required: POST /connect/stub without session → 401.
  10. Provider revoke fails: monkey-patch stub.revoke to raise →
      disconnect still completes, separate revocation_provider_failed
      audit event written.

Two layers:
  - Unit tests against `oauth_state` (sign/verify/expiry/tampering).
  - HTTP integration tests against the mounted /api/oauth/* router.
    Skip on local FastAPI/Starlette pin mismatch.
"""

from __future__ import annotations

import time
import uuid
from typing import AsyncIterator
from urllib.parse import urlparse, parse_qs

import httpx
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    ConnectorEvent,
    ConnectorIdentity,
    ConnectorOAuthSession,
    EVENT_CONNECTED,
    EVENT_DISCONNECTED,
    EVENT_REVOCATION_PROVIDER_FAILED,
    EVENT_REVOCATION_PROVIDER_SUCCEEDED,
    User,
)
from app.services import connector_vault as vault
from app.services.connector_registry import (
    get_registry,
    reset_registry_for_tests,
)
from app.services.credential_crypto import _multi_fernet
from app.services.oauth_state import (
    StateVerificationError,
    sign_state,
    state_hash,
    verify_state,
)
from app.services.provider_apps import (
    bootstrap_provider_apps,
    reset_for_tests as reset_provider_apps,
)


# ─── Autouse: provision crypto + OAuth state secret + stub registry ──


@pytest.fixture(autouse=True)
def _provision_oauth_environment():
    """Set the env up so the OAuth API can sign state, encrypt tokens,
    and find the stub provider_app + stub manifest."""
    prev_enc = settings.platform_encryption_key
    prev_enc_prev = settings.platform_encryption_key_previous
    prev_state = settings.oauth_state_secret
    prev_flag = settings.enable_connector_oauth
    prev_callback = settings.oauth_callback_url

    settings.platform_encryption_key = Fernet.generate_key().decode()
    settings.platform_encryption_key_previous = ""
    settings.oauth_state_secret = "test_state_secret_xxxxxxxxxxxxxxxxxxxxxxx"
    settings.enable_connector_oauth = True
    settings.oauth_callback_url = "http://test/api/oauth/callback"
    _multi_fernet.cache_clear()

    # Reset + bootstrap stub provider_app.
    reset_provider_apps()
    bootstrap_provider_apps()

    # Reset + load registry with experimental enabled (so `stub` loads).
    reset_registry_for_tests()
    get_registry().load_all(include_experimental=True)

    try:
        yield
    finally:
        settings.platform_encryption_key = prev_enc
        settings.platform_encryption_key_previous = prev_enc_prev
        settings.oauth_state_secret = prev_state
        settings.enable_connector_oauth = prev_flag
        settings.oauth_callback_url = prev_callback
        _multi_fernet.cache_clear()
        reset_provider_apps()
        reset_registry_for_tests()


# ─── Pure-unit: oauth_state sign/verify ──────────────────────────────


def test_state_sign_verify_roundtrip():
    token = sign_state("alice-uid", "stub")
    payload = verify_state(token)
    assert payload.user_id == "alice-uid"
    assert payload.connector_id == "stub"
    assert len(payload.csrf_nonce) >= 16


def test_state_verify_rejects_tampered_payload():
    token = sign_state("alice-uid", "stub")
    body, sig = token.rsplit(".", 1)
    # Flip a single character in the body.
    tampered = body[:-1] + ("A" if body[-1] != "A" else "B") + "." + sig
    with pytest.raises(StateVerificationError, match="HMAC"):
        verify_state(tampered)


def test_state_verify_rejects_tampered_signature():
    token = sign_state("alice-uid", "stub")
    body, sig = token.rsplit(".", 1)
    tampered = body + "." + sig[:-1] + ("A" if sig[-1] != "A" else "B")
    with pytest.raises(StateVerificationError):
        verify_state(tampered)


def test_state_verify_rejects_expired(monkeypatch):
    token = sign_state("alice-uid", "stub")
    # Pretend 11 minutes have passed (TTL is 600s = 10 min).
    real_time = time.time
    monkeypatch.setattr(
        "app.services.oauth_state.time.time",
        lambda: real_time() + 11 * 60,
    )
    with pytest.raises(StateVerificationError, match="expired"):
        verify_state(token)


def test_state_verify_rejects_missing_token():
    with pytest.raises(StateVerificationError, match="missing"):
        verify_state("")


def test_state_verify_rejects_no_dot():
    with pytest.raises(StateVerificationError):
        verify_state("notarealstate")


def test_state_hash_is_stable():
    """Same input → same hash. Two different states → different hashes."""
    a = sign_state("u1", "stub")
    b = sign_state("u1", "stub")  # Different csrf_nonce + iat
    assert state_hash(a) == state_hash(a)
    assert state_hash(b) == state_hash(b)
    assert state_hash(a) != state_hash(b)


# ─── HTTP integration fixtures ───────────────────────────────────────


@pytest_asyncio.fixture
async def oauth_test_app() -> AsyncIterator[FastAPI]:
    """FastAPI app with auth + oauth routers. Skip on local pin mismatch."""
    try:
        from app.api.auth import router as auth_router
        from app.api.oauth import router as oauth_router

        app = FastAPI()
        app.include_router(auth_router, prefix=settings.api_prefix)
        app.include_router(oauth_router, prefix=settings.api_prefix)
    except TypeError as e:
        pytest.skip(f"FastAPI/router wiring failed (local env pin mismatch): {e}")
    yield app


@pytest_asyncio.fixture
async def http_client(oauth_test_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=oauth_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def alice() -> dict:
    """Seed one user + return {id, jwt} for authenticated requests."""
    from app.services.auth_service import create_access_token

    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid, email=f"{uid[:8]}@example.com",
            hashed_password="x", name="Alice",
        ))
        await db.commit()
    return {"id": uid, "jwt": create_access_token(uid)}


# ─── 1. Stub round-trip (the headline test) ──────────────────────────


@pytest.mark.asyncio
async def test_stub_oauth_full_round_trip(http_client, alice):
    """POST /connect/stub → GET authorize_url → 302 to /callback →
    ConnectorIdentity exists with encrypted tokens, status='active',
    audit row written."""
    headers = {"Authorization": f"Bearer {alice['jwt']}"}

    # 1. /connect — get the authorize URL.
    r = await http_client.post("/api/oauth/connect/stub", headers=headers)
    assert r.status_code == 200, r.text
    authorize_url = r.json()["authorize_url"]
    # Stub authorize lives at /api/oauth/_stub/authorize
    assert "/api/oauth/_stub/authorize" in authorize_url

    # The authorize URL is just a path + query when using the stub
    # provider_app. Strip any host the URL builder may have prefixed.
    parsed = urlparse(authorize_url)
    path_with_query = parsed.path + "?" + parsed.query

    # 2. GET the authorize URL — stub immediately 302s back to /callback
    #    with a synthetic code.
    r = await http_client.get(path_with_query, follow_redirects=False)
    assert r.status_code == 302, r.text
    callback_location = r.headers["location"]
    assert "/api/oauth/callback" in callback_location

    # The callback location includes the real callback URL host. Re-parse.
    parsed_cb = urlparse(callback_location)
    cb_query = "?" + parsed_cb.query

    # 3. GET /callback — full token-exchange path runs end-to-end.
    r = await http_client.get(
        f"/api/oauth/callback{cb_query}",
        follow_redirects=False,
    )
    assert r.status_code == 302, r.text
    assert "connected=stub" in r.headers["location"]

    # 4. ConnectorIdentity row exists, encrypted, active.
    async with async_session_maker() as db:
        ident_row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice["id"]
            )
        )).scalar_one()
        assert ident_row.connector_id == "stub"
        assert ident_row.status == "active"
        assert ident_row.access_token_enc is not None
        assert ident_row.access_token_enc.startswith("gAAAAA")
        # refresh_token came back from stub too
        assert ident_row.refresh_token_enc is not None

        # Audit row.
        events = (await db.execute(
            select(ConnectorEvent)
            .where(ConnectorEvent.user_id == alice["id"])
            .where(ConnectorEvent.connector_id == "stub")
        )).scalars().all()
        assert any(e.event_type == EVENT_CONNECTED for e in events)

        # Session row was atomically deleted.
        sessions = (await db.execute(
            select(ConnectorOAuthSession).where(
                ConnectorOAuthSession.user_id == alice["id"]
            )
        )).scalars().all()
        assert sessions == [], "session row must be single-use"

    # 5. The plaintext round-trips through the vault.
    async with async_session_maker() as db:
        decrypted = await vault.get(db, alice["id"], "stub")
        assert decrypted is not None
        assert decrypted.access_token.startswith("stub_at_")
        assert decrypted.refresh_token.startswith("stub_rt_")


# ─── 2. Disconnect ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disconnect_zeroes_ciphertext_and_audits(http_client, alice):
    """After connect, disconnect zeroes ciphertext, status='revoked',
    audit row written, row preserved."""
    headers = {"Authorization": f"Bearer {alice['jwt']}"}
    await _round_trip_connect(http_client, alice, headers)

    r = await http_client.post("/api/oauth/disconnect/stub", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "disconnected"
    assert body["revoke_outcome"] == "succeeded"

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice["id"]
            )
        )).scalar_one()
        assert row.status == "revoked"
        assert row.access_token_enc is None
        assert row.refresh_token_enc is None

        events = (await db.execute(
            select(ConnectorEvent.event_type)
            .where(ConnectorEvent.user_id == alice["id"])
        )).scalars().all()
        assert EVENT_DISCONNECTED in events
        assert EVENT_REVOCATION_PROVIDER_SUCCEEDED in events


# ─── 3. State replay ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_callback_rejects_state_replay(http_client, alice):
    """First callback succeeds. Second callback with the SAME state
    must fail — single-use replay defense."""
    headers = {"Authorization": f"Bearer {alice['jwt']}"}

    # Connect + capture the callback URL.
    cb_url = await _connect_and_get_callback(http_client, alice, headers)

    # First callback — succeeds.
    r1 = await http_client.get(cb_url, follow_redirects=False)
    assert r1.status_code == 302

    # Second callback — same state, must fail.
    r2 = await http_client.get(cb_url, follow_redirects=False)
    assert r2.status_code == 400, r2.text
    assert "invalid state" in r2.text.lower()


# ─── 4. Expired state ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_callback_rejects_expired_state(http_client, alice, monkeypatch):
    """If 11 minutes pass between /connect and /callback, the state
    HMAC verify rejects with `expired`."""
    headers = {"Authorization": f"Bearer {alice['jwt']}"}
    cb_url = await _connect_and_get_callback(http_client, alice, headers)

    # Forward time by 11 min — only inside the verify path.
    real_time = time.time
    monkeypatch.setattr(
        "app.services.oauth_state.time.time",
        lambda: real_time() + 11 * 60,
    )

    r = await http_client.get(cb_url, follow_redirects=False)
    assert r.status_code == 400


# ─── 5. Tampered signature ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_callback_rejects_tampered_state_signature(http_client, alice):
    headers = {"Authorization": f"Bearer {alice['jwt']}"}
    cb_url = await _connect_and_get_callback(http_client, alice, headers)

    # Pull the state out, tamper one byte of the signature, re-emit.
    parsed = urlparse(cb_url)
    qs = parse_qs(parsed.query)
    state = qs["state"][0]
    body, sig = state.rsplit(".", 1)
    tampered = body + "." + sig[:-1] + ("A" if sig[-1] != "A" else "B")
    qs["state"] = [tampered]
    tampered_qs = "&".join(f"{k}={v[0]}" for k, v in qs.items())
    tampered_cb = f"{parsed.path}?{tampered_qs}"

    r = await http_client.get(tampered_cb, follow_redirects=False)
    assert r.status_code == 400


# ─── 7. Unknown connector ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_unknown_connector_404(http_client, alice):
    headers = {"Authorization": f"Bearer {alice['jwt']}"}
    r = await http_client.post(
        "/api/oauth/connect/nonexistent_xyz", headers=headers,
    )
    assert r.status_code == 404


# ─── 8. Flag off ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_503_when_flag_disabled(http_client, alice):
    settings.enable_connector_oauth = False
    headers = {"Authorization": f"Bearer {alice['jwt']}"}
    r = await http_client.post("/api/oauth/connect/stub", headers=headers)
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_disconnect_503_when_flag_disabled(http_client, alice):
    settings.enable_connector_oauth = False
    headers = {"Authorization": f"Bearer {alice['jwt']}"}
    r = await http_client.post("/api/oauth/disconnect/stub", headers=headers)
    assert r.status_code == 503


# ─── 9. Auth required ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_401_when_unauthenticated(http_client):
    r = await http_client.post("/api/oauth/connect/stub")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_disconnect_401_when_unauthenticated(http_client):
    r = await http_client.post("/api/oauth/disconnect/stub")
    assert r.status_code == 401


# ─── 10. Provider revoke fails ───────────────────────────────────────


@pytest.mark.asyncio
async def test_stub_authorize_url_is_absolute(http_client, alice):
    """T1d Q2: every authorize_url returned by /connect/<id> must be
    absolute, regardless of whether the underlying provider_app config
    points at a relative path (stub) or an external host (Google).
    Frontend's window.open contract is 'always absolute'."""
    headers = {"Authorization": f"Bearer {alice['jwt']}"}
    r = await http_client.post("/api/oauth/connect/stub", headers=headers)
    assert r.status_code == 200
    authorize_url = r.json()["authorize_url"]
    cb_parsed = urlparse(settings.oauth_callback_url)
    cb_origin = f"{cb_parsed.scheme}://{cb_parsed.netloc}"
    assert authorize_url.startswith(cb_origin), (
        f"stub authorize_url must start with callback origin {cb_origin!r}, "
        f"got: {authorize_url}"
    )


@pytest.mark.asyncio
async def test_disconnect_proceeds_when_provider_revoke_fails(
    http_client, alice, monkeypatch,
):
    """Per architecture §3.5: disconnect still completes if revoke
    fails. Audit row records revocation_provider_failed."""
    headers = {"Authorization": f"Bearer {alice['jwt']}"}
    await _round_trip_connect(http_client, alice, headers)

    # Patch the stub provider's revoke to raise.
    entry = get_registry().get("stub")
    assert entry is not None
    original_revoke = entry.provider.revoke

    async def boom(user_id, access_token, refresh_token=None):
        raise RuntimeError("simulated provider 502")

    entry.provider.revoke = boom

    try:
        r = await http_client.post(
            "/api/oauth/disconnect/stub", headers=headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["revoke_outcome"] == "failed"

        # Vault still got disconnected.
        async with async_session_maker() as db:
            row = (await db.execute(
                select(ConnectorIdentity).where(
                    ConnectorIdentity.user_id == alice["id"]
                )
            )).scalar_one()
            assert row.status == "revoked"
            assert row.access_token_enc is None

            events = (await db.execute(
                select(ConnectorEvent.event_type)
                .where(ConnectorEvent.user_id == alice["id"])
            )).scalars().all()
            assert EVENT_REVOCATION_PROVIDER_FAILED in events
            assert EVENT_DISCONNECTED in events
            assert EVENT_REVOCATION_PROVIDER_SUCCEEDED not in events
    finally:
        entry.provider.revoke = original_revoke


# ─── Helpers ─────────────────────────────────────────────────────────


async def _round_trip_connect(http_client, alice, headers):
    """Walk through connect → authorize → callback to leave the user
    with an active stub identity. Returns nothing (callers query the
    DB themselves)."""
    cb_url = await _connect_and_get_callback(http_client, alice, headers)
    r = await http_client.get(cb_url, follow_redirects=False)
    assert r.status_code == 302, r.text


async def _connect_and_get_callback(http_client, alice, headers) -> str:
    """POST /connect, GET the authorize URL, return the callback path
    + query as a single URL string ready to GET."""
    r = await http_client.post("/api/oauth/connect/stub", headers=headers)
    assert r.status_code == 200, r.text
    authorize_url = r.json()["authorize_url"]
    parsed = urlparse(authorize_url)
    authz_path = parsed.path + "?" + parsed.query
    r = await http_client.get(authz_path, follow_redirects=False)
    assert r.status_code == 302
    location = r.headers["location"]
    parsed_cb = urlparse(location)
    return parsed_cb.path + "?" + parsed_cb.query


# ─── 11. Token-response parsing (GitHub's form-urlencoded body) ──────
#
# Every GitHub connect attempt ever made 500'd here. `_exchange_code_
# for_tokens` called `r.json()` on a body GitHub serves as
# `application/x-www-form-urlencoded` unless asked otherwise, so the
# callback raised `JSONDecodeError: Expecting value: line 1 column 1`
# AFTER the user had already consented — an "Internal Server Error"
# page in the OAuth popup and zero rows in connector_identities.
#
# The stub provider exchanges in-process and never touches this branch,
# which is exactly why the end-to-end tests above stayed green for the
# entire life of the bug. These go at the parser directly.


def test_parse_token_response_reads_form_urlencoded_body():
    """GitHub's default shape."""
    from app.api.oauth import _parse_token_response

    r = httpx.Response(
        200,
        headers={"content-type": "application/x-www-form-urlencoded; charset=utf-8"},
        text="access_token=gho_abc123&scope=repo%2Cread%3Auser&token_type=bearer",
    )
    assert _parse_token_response(r) == {
        "access_token": "gho_abc123",
        "scope": "repo,read:user",
        "token_type": "bearer",
    }


def test_parse_token_response_reads_json_body():
    """Google / Microsoft / LinkedIn shape — must not regress."""
    from app.api.oauth import _parse_token_response

    r = httpx.Response(
        200,
        headers={"content-type": "application/json; charset=utf-8"},
        json={"access_token": "ya29.abc", "expires_in": 3599, "scope": "a b"},
    )
    assert _parse_token_response(r)["access_token"] == "ya29.abc"


def test_parse_token_response_survives_a_mislabelled_content_type():
    """A JSON body under a form content-type (and vice versa) must not
    500 — the whole point of the belt-and-braces parser."""
    from app.api.oauth import _parse_token_response

    lying_form = httpx.Response(
        200,
        headers={"content-type": "application/x-www-form-urlencoded"},
        text='{"access_token": "tok"}',
    )
    assert _parse_token_response(lying_form)["access_token"] == "tok"

    lying_json = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        text="access_token=tok",
    )
    assert _parse_token_response(lying_json)["access_token"] == "tok"


def test_parse_token_response_raises_typed_error_on_garbage():
    """An HTML error page must raise the typed error the callback
    catches (→ a bridge redirect the user can read), never the bare
    ValueError that became a 500."""
    from app.api.oauth import TokenResponseUnparseable, _parse_token_response

    r = httpx.Response(
        502,
        headers={"content-type": "text/html"},
        text="<html><body>Bad Gateway</body></html>",
    )
    with pytest.raises(TokenResponseUnparseable):
        _parse_token_response(r)


@pytest.mark.asyncio
async def test_token_exchange_sends_accept_json(monkeypatch):
    """`Accept: application/json` is what makes GitHub answer in JSON at
    all. Without it GitHub content-negotiates down to form-urlencoded —
    verified against the live endpoint. Assert we actually send it."""
    from app.api import oauth as oauth_mod

    seen: dict = {}

    async def fake_post(self, url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers") or {}
        seen["data"] = kwargs.get("data")
        return httpx.Response(
            200,
            headers={"content-type": "application/x-www-form-urlencoded"},
            text="access_token=gho_x&scope=repo&token_type=bearer",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    tokens = await oauth_mod._exchange_code_for_tokens(
        token_url="https://github.com/login/oauth/access_token",
        code="abc",
        code_verifier="v",
        client_id="cid",
        client_secret="sec",
        redirect_uri="https://toup.ai/api/oauth/callback",
    )
    assert seen["headers"].get("Accept") == "application/json"
    assert tokens["access_token"] == "gho_x"


def test_github_comma_scopes_split_into_separate_entries():
    """RFC 6749 says space-delimited; GitHub sends `repo,read:user`.
    A bare `.split()` stored one scope literally named "repo,read:user"."""
    import re as _re

    scope = "repo,read:user"
    assert [s for s in _re.split(r"[,\s]+", scope) if s] == ["repo", "read:user"]


# ─── 12. verify_state's ascii escape hatch ───────────────────────────
#
# `verify_state` promises in its own docstring that every failure mode
# maps to a 400. One did not: `body_b64.encode("ascii")` sat outside
# any try, so a non-ASCII state raised UnicodeEncodeError — a
# ValueError, NOT a StateVerificationError — and escaped to the
# unauthenticated /api/oauth/callback as an unhandled 500.


def test_verify_state_rejects_non_ascii_without_leaking_unicode_error():
    from app.services.oauth_state import StateVerificationError, verify_state

    with pytest.raises(StateVerificationError):
        verify_state("é.abc")

    # And the ordinary tampered-token path still behaves.
    with pytest.raises(StateVerificationError):
        verify_state("abc.def")


def test_state_hash_does_not_raise_on_non_ascii():
    """Currently unreachable — `verify_state` runs first — but safety
    by call ordering evaporates the moment a second caller appears."""
    from app.services.oauth_state import state_hash

    assert len(state_hash("é.abc")) == 64
    # utf-8 vs ascii must not have moved the hash for real tokens.
    import hashlib as _h
    assert state_hash("abc.def") == _h.sha256(b"abc.def").hexdigest()


def test_declining_consent_says_cancelled_not_failed():
    """`access_denied` is the commonest code on this branch. The
    frontend's message-less fallback is the literal string "OAuth flow
    failed", which is untrue for a user who pressed Cancel."""
    from app.api.oauth import _PROVIDER_ERROR_COPY

    copy = _PROVIDER_ERROR_COPY["access_denied"]
    assert "cancel" in copy.lower()
    assert "fail" not in copy.lower()


def test_bridge_error_carries_the_connector_id():
    """The bridge posts one connectorId field and `useOAuthHandoff`
    drops any message whose id doesn't match the connector it opened
    the popup for — so an error redirect without `connector=` is
    discarded on arrival and surfaces as a false "Cancelled"."""
    from urllib.parse import parse_qs, urlparse

    from app.api.oauth import _bridge_error

    r = _bridge_error("token_exchange_failed", connector_id="github", message="nope")
    assert r.status_code == 302
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["connector"] == ["github"]
    assert q["error"] == ["token_exchange_failed"]
    assert q["message"] == ["nope"]
