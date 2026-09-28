"""Toup for Mac — the relay socket: subprotocol auth, close codes, revocation.

What this file pins
-------------------
`desktop/docs/RELAY_PROTOCOL.md` §3 end to end, and §9.1 — the divergence the
audit calls "the single most important thing not to reproduce": the Chrome
extension's revoke sets a flag no authenticator reads, leaves the live socket
up, and the 90-day token keeps working
(`agent-tool-relay.md` §1.8, `extension.py:841-848`, `:364-397`).

Three of these tests would be green against a server that never checks
anything, so each one names the specific thing it can see fail:

  * `4001`/`4003` make the app DELETE its token and stop permanently (§3.2),
    so a transient platform failure answering either is a laptop that has to
    be paired again by hand. `unavailable` must close `1001`.
  * `URLSessionWebSocketTask` joins its subprotocol list with `", "`, so the
    bearer arrives with a LEADING SPACE. A server that compares untrimmed
    rejects every real client and nothing else in the stack can see it.
  * A revoked device must fail BOTH doors — the next connect and the socket
    it is holding right now.

Deliberately not `TestClient`: what is under test is what arrives on the
wire, including the close code, and TestClient hides an ASGI exception
behind its portal thread (same reasoning as
`tests/test_ws_chat_infra_fault_close.py`).

Lane: RUN_MODE=platform — `desktop_devices` is PLATFORM_ONLY and
`_platform_db_local()` is what makes the relay authenticate against the
local table instead of an HTTP hop to itself.

Run:
    cd backend && DATABASE_URL="sqlite+aiosqlite:///:memory:" RUN_MODE=platform \
        python -m pytest tests/test_desktop_relay_socket.py -q
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.agent import desktop_bridge
from app.api import desktop as desktop_api
from app.config import settings
from app.db.database import async_session_maker
from app.db.models import DesktopDevice, UserSession


# ── The app under test ────────────────────────────────────────────────
def _app() -> FastAPI:
    from app.api.auth import router as auth_router

    app = FastAPI()
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(desktop_api.router, prefix=settings.api_prefix)
    app.include_router(desktop_api.ws_router, prefix=settings.api_prefix)
    return app


@pytest.fixture(autouse=True)
def _enabled_account_cohort(monkeypatch):
    # These tests exercise the enabled protocol; rollout-off has its own tests.
    monkeypatch.setattr(settings, "desktop_connections_rollout_pct", 100)
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)


@pytest.fixture(autouse=True)
def _clean_registry():
    desktop_bridge.reset_for_tests()
    desktop_api._RATE_BUCKETS.clear()
    yield
    desktop_bridge.reset_for_tests()


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


@pytest.fixture(autouse=True)
async def _mac_signed_in(_test_user):
    """Pairing init belongs to a live account session on the Mac."""
    from app.services.auth_service import decode_platform_jwt

    jti = decode_platform_jwt(_test_user["token"]).get("jti")
    async with async_session_maker() as db:
        db.add(UserSession(
            id=str(uuid.uuid4()), user_id=_test_user["id"], jti=jti,
            device_label="Mac", created_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(), is_revoked=False,
        ))
        await db.commit()
    yield


@pytest.fixture
async def api() -> AsyncClient:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── One client socket, spoken straight into the ASGI app ─────────────
class AsgiWs:
    def __init__(self, app: Any, subprotocols: Optional[List[str]] = None) -> None:
        self._app = app
        self._subprotocols = list(subprotocols or [])
        self._to_app: asyncio.Queue = asyncio.Queue()
        self.received: asyncio.Queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None
        self.accepted: Optional[Dict[str, Any]] = None
        self.close_code: Optional[int] = None

    async def _pump(self, message: Dict[str, Any]) -> None:
        await self.received.put(message)

    async def open(self) -> Dict[str, Any]:
        scope = {
            "type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "scheme": "ws", "path": "/api/ws/desktop",
            "raw_path": b"/api/ws/desktop", "query_string": b"",
            "root_path": "", "headers": [(b"host", b"testserver")],
            "client": ("127.0.0.1", 51234), "server": ("testserver", 80),
            "subprotocols": self._subprotocols, "state": {},
        }
        self._to_app.put_nowait({"type": "websocket.connect"})
        self.task = asyncio.create_task(
            self._app(scope, self._to_app.get, self._pump)
        )
        self.accepted = await asyncio.wait_for(self.received.get(), timeout=5.0)
        return self.accepted

    def send_json(self, obj: Any) -> None:
        self._to_app.put_nowait(
            {"type": "websocket.receive", "text": json.dumps(obj)}
        )

    def send_raw(self, text: str) -> None:
        self._to_app.put_nowait({"type": "websocket.receive", "text": text})

    async def next_message(self, timeout: float = 5.0) -> Dict[str, Any]:
        msg = await asyncio.wait_for(self.received.get(), timeout=timeout)
        if msg["type"] == "websocket.close":
            self.close_code = msg.get("code", 1000)
        return msg

    async def next_frame(self, timeout: float = 5.0) -> Dict[str, Any]:
        msg = await self.next_message(timeout)
        assert msg["type"] == "websocket.send", msg
        return json.loads(msg["text"])

    async def settle(self, timeout: float = 5.0) -> None:
        """Block until the handler is inside its receive loop.

        `accept` is sent BEFORE the token is verified (§3.2 — refusals are
        close codes, so the handshake has to complete first), and
        registration happens after. So "open() returned" says nothing about
        whether the endpoint exists yet; a pong does, because answering one
        is the first thing past `register()`."""
        self.send_json({"type": "ping", "ts": "settle"})
        frame = await self.next_frame(timeout)
        assert frame["type"] == "pong", frame

    async def wait_closed(self, timeout: float = 5.0) -> int:
        while True:
            msg = await self.next_message(timeout)
            if msg["type"] == "websocket.close":
                return msg.get("code", 1000)

    async def dispose(self) -> None:
        self._to_app.put_nowait({"type": "websocket.disconnect", "code": 1000})
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=5.0)
            except (asyncio.TimeoutError, Exception):  # noqa: B014
                if not self.task.done():
                    self.task.cancel()


# ── Fixtures: a paired device and its token ──────────────────────────
async def _pair(api: AsyncClient, auth_headers: dict) -> Dict[str, Any]:
    started = await api.post(
        "/api/desktop/pair/init",
        json={"device_name": "A Mac", "app_version": "0.1.0",
              "os_version": "15.7", "flavor": "direct"},
        headers=auth_headers,
    )
    assert started.status_code == 200, started.text
    code = started.json()
    approved = await api.post(
        "/api/desktop/pair/approve",
        json={"user_code": code["user_code"]}, headers=auth_headers,
    )
    assert approved.status_code == 200, approved.text
    poll = await api.post(
        "/api/desktop/pair/poll", json={"device_code": code["device_code"]},
    )
    assert poll.status_code == 200, poll.text
    return poll.json()


async def _open(subprotocols: List[str]) -> AsgiWs:
    ws = AsgiWs(_app(), subprotocols=subprotocols)
    await ws.open()
    return ws


def _auth(token: str, *, leading_space: bool = False) -> List[str]:
    bearer = f"bearer.{token}"
    return ["toup.auth.v1", (" " + bearer) if leading_space else bearer]


# ══════════════════════════════════════════════════════════════════════
# 1. §3.1 — auth rides the subprotocol, and the echo is the contract
# ══════════════════════════════════════════════════════════════════════
async def test_a_paired_device_connects_and_the_server_echoes_the_subprotocol(
    api, auth_headers,
):
    bundle = await _pair(api, auth_headers)
    ws = await _open(_auth(bundle["access_token"]))
    try:
        assert ws.accepted["type"] == "websocket.accept"
        # The app treats a connection that does not echo this as a server
        # that did not understand the auth channel: it closes and backs off.
        assert ws.accepted.get("subprotocol") == "toup.auth.v1"
        await ws.settle()
        assert desktop_bridge.is_connected(bundle["user_id"]) is True
    finally:
        await ws.dispose()
    assert desktop_bridge.is_connected(bundle["user_id"]) is False


async def test_a_bearer_with_a_leading_space_still_authenticates(
    api, auth_headers,
):
    """`URLSessionWebSocketTask` joins its subprotocol array with ", ", and
    parsers that split on "," alone hand the second token over as
    " bearer.eyJ…". A server that compares untrimmed rejects every real
    client — and nothing else in this stack can see that."""
    bundle = await _pair(api, auth_headers)
    ws = await _open(_auth(bundle["access_token"], leading_space=True))
    try:
        assert ws.accepted.get("subprotocol") == "toup.auth.v1"
        await ws.settle()
        assert desktop_bridge.is_connected(bundle["user_id"]) is True
    finally:
        await ws.dispose()


async def test_a_query_string_token_is_not_a_way_in(api, auth_headers):
    """§9.5: the extension still accepts `?token=` and a first-frame
    `{type:"auth"}`. A token in a URL reaches every proxy log between here
    and the tenant, and the app refuses a relay URL with any query string at
    all — so accepting one could only ever serve something that is not this
    client."""
    bundle = await _pair(api, auth_headers)
    app = _app()
    ws = AsgiWs(app, subprotocols=[])
    ws._to_app.put_nowait({"type": "websocket.connect"})
    scope = {
        "type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "scheme": "ws", "path": "/api/ws/desktop",
        "raw_path": b"/api/ws/desktop",
        "query_string": f"token={bundle['access_token']}".encode(),
        "root_path": "", "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 1), "server": ("testserver", 80),
        "subprotocols": [], "state": {},
    }
    ws.task = asyncio.create_task(app(scope, ws._to_app.get, ws._pump))
    try:
        assert (await ws.next_message())["type"] == "websocket.accept"
        assert await ws.wait_closed() == 4001
        assert desktop_bridge.is_connected(bundle["user_id"]) is False
    finally:
        await ws.dispose()


async def test_no_auth_subprotocol_is_accepted_then_closed(api, auth_headers):
    """§3.2: ACCEPT the handshake, then close with the code. An upgrade
    refused with an HTTP status carries no close code, and a native client
    cannot tell that apart from a dropped network — so it would retry a dead
    token for the life of the laptop."""
    ws = await _open([])
    try:
        assert ws.accepted["type"] == "websocket.accept"
        assert ws.accepted.get("subprotocol") in (None, "")
        assert await ws.wait_closed() == 4001
    finally:
        await ws.dispose()


async def test_the_auth_subprotocol_without_a_bearer_is_closed(api):
    ws = await _open(["toup.auth.v1"])
    try:
        assert await ws.wait_closed() == 4001
    finally:
        await ws.dispose()


async def test_a_forged_token_is_4001(api):
    ws = await _open(_auth("not.a.jwt"))
    try:
        assert await ws.wait_closed() == 4001
    finally:
        await ws.dispose()


async def test_a_token_for_a_device_row_that_is_gone_is_4001(api, auth_headers):
    """`unknown`, not `revoked` — but both are permanent, and neither may be
    answered by accepting the socket."""
    bundle = await _pair(api, auth_headers)
    async with async_session_maker() as db:
        row = await db.get(DesktopDevice, bundle["device_id"])
        await db.delete(row)
        await db.commit()

    ws = await _open(_auth(bundle["access_token"]))
    try:
        assert await ws.wait_closed() == 4001
    finally:
        await ws.dispose()


async def test_an_expired_token_is_4001(api, auth_headers):
    from jose import jwt

    bundle = await _pair(api, auth_headers)
    claims = jwt.get_unverified_claims(bundle["access_token"])
    stale = jwt.encode(
        {**claims, "iat": claims["iat"] - 90_000, "exp": claims["iat"] - 60},
        desktop_api._signing_key(), algorithm="HS256",
    )
    ws = await _open(_auth(stale))
    try:
        assert await ws.wait_closed() == 4001
    finally:
        await ws.dispose()


# ══════════════════════════════════════════════════════════════════════
# 2. §9.1 — revocation, through BOTH doors
# ══════════════════════════════════════════════════════════════════════
async def test_a_revoked_device_cannot_reconnect(api, auth_headers):
    bundle = await _pair(api, auth_headers)
    r = await api.post(
        "/api/desktop/devices/revoke",
        json={"device_id": bundle["device_id"]}, headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["revoked"] == 1

    ws = await _open(_auth(bundle["access_token"]))
    try:
        assert await ws.wait_closed() == 4003, (
            "a revoked Mac reconnected — this is exactly extension.py's "
            "_decode_extension_token, which consults neither revoked_at nor "
            "any denylist"
        )
    finally:
        await ws.dispose()


async def test_revoked_at_alone_refuses_the_connect(api, auth_headers):
    """The `revoked_at` check ON ITS OWN, with a jti that still matches.

    Added after a mutation survived: `POST /devices/revoke` ALSO nulls
    `token_jti`, so `test_a_revoked_device_cannot_reconnect` stays green
    against an authenticator that consults only the jti — it cannot tell the
    two guards apart, which is precisely the `extension.py:364-397` failure
    ("consults neither `revoked_at` nor any denylist") it exists to pin.

    The state below is reachable: any revocation that marks the row without
    reaching through to the jti column — an admin action, a data fix, a
    future second revoke path — lands here."""
    bundle = await _pair(api, auth_headers)
    async with async_session_maker() as db:
        row = await db.get(DesktopDevice, bundle["device_id"])
        row.revoked_at = datetime.utcnow()          # token_jti left intact
        await db.commit()
        assert row.token_jti

    ws = await _open(_auth(bundle["access_token"]))
    try:
        assert await ws.wait_closed() == 4003
    finally:
        await ws.dispose()


async def test_revoking_closes_the_LIVE_socket_with_4003(api, auth_headers):
    """The half the precedent never shipped. "Mark the row revoked" and
    "denylist the jti" both leave a socket that is already open still
    reading tasks."""
    bundle = await _pair(api, auth_headers)
    ws = await _open(_auth(bundle["access_token"]))
    try:
        await ws.settle()
        assert desktop_bridge.is_connected(bundle["user_id"]) is True
        r = await api.post(
            "/api/desktop/devices/revoke",
            json={"device_id": bundle["device_id"]}, headers=auth_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["sockets_closed"] == 1
        assert await ws.wait_closed() == 4003
    finally:
        await ws.dispose()


async def test_a_device_token_can_revoke_only_its_own_device(
    api, auth_headers, test_user_id,
):
    """The Mac's own *Disconnect* sends the device token as `Bearer`. The
    device id comes off the TOKEN, never off the body, so a stolen device
    token cannot disconnect the user's other Macs."""
    first = await _pair(api, auth_headers)
    second = await _pair(api, auth_headers)
    assert first["device_id"] != second["device_id"]

    r = await api.post(
        "/api/desktop/devices/revoke",
        json={"device_id": second["device_id"]},
        headers={"Authorization": f"Bearer {first['access_token']}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["revoked"] == 1

    devices = (await api.get(
        "/api/desktop/devices", headers=auth_headers,
    )).json()["devices"]
    alive = {d["id"] for d in devices}
    assert second["device_id"] in alive, (
        "a device token revoked a DIFFERENT device by naming it in the body"
    )
    assert first["device_id"] not in alive


async def test_revoke_needs_a_credential(api):
    assert (await api.post(
        "/api/desktop/devices/revoke", json={},
    )).status_code == 401


async def test_a_superseded_token_is_refused(api, auth_headers):
    """Re-pairing mints a new jti on a new row; the old credential's row
    still exists and is not revoked, so `token_jti` is the only thing that
    can refuse it."""
    bundle = await _pair(api, auth_headers)
    async with async_session_maker() as db:
        row = await db.get(DesktopDevice, bundle["device_id"])
        row.token_jti = uuid.uuid4().hex
        await db.commit()

    ws = await _open(_auth(bundle["access_token"]))
    try:
        assert await ws.wait_closed() == 4003
    finally:
        await ws.dispose()


# ══════════════════════════════════════════════════════════════════════
# 3. §3.2 — a platform blip must never delete a good token
# ══════════════════════════════════════════════════════════════════════
async def test_an_unverifiable_token_closes_1001_and_not_4001(
    api, auth_headers, monkeypatch,
):
    """`4001` and `4003` are never retried and cause the app to DELETE its
    token. "I could not check" is not "you are not who you say you are" —
    the same distinction `_authenticate_ws` had to learn after incident 3."""
    bundle = await _pair(api, auth_headers)

    async def _unavailable(_token: str):
        return {"ok": False, "reason": "unavailable"}

    monkeypatch.setattr(desktop_api, "_verify_device_token", _unavailable)
    ws = await _open(_auth(bundle["access_token"]))
    try:
        code = await ws.wait_closed()
        assert code == 1001, (
            f"closed {code}; the app deletes its token on 4001/4003, so a "
            "platform blip would unpair every Mac in the fleet"
        )
    finally:
        await ws.dispose()


# ══════════════════════════════════════════════════════════════════════
# 4. §3.3, §5 — liveness and frame hygiene
# ══════════════════════════════════════════════════════════════════════
async def test_every_ping_is_answered(api, auth_headers):
    """The only detector for a half-open socket: after a Mac sleeps the
    socket commonly reports OPEN and `send` keeps succeeding into a void."""
    bundle = await _pair(api, auth_headers)
    ws = await _open(_auth(bundle["access_token"]))
    try:
        ws.send_json({"type": "ping", "ts": 12345})
        frame = await ws.next_frame()
        assert frame == {"type": "pong", "ts": 12345}
    finally:
        await ws.dispose()


async def test_hello_records_the_device_and_its_tools(api, auth_headers):
    bundle = await _pair(api, auth_headers)
    ws = await _open(_auth(bundle["access_token"]))
    try:
        ws.send_json({
            "type": "hello", "protocol": 1,
            "device": {"name": "A Mac", "os_version": "15.7",
                       "app_version": "0.1.0", "flavor": "direct",
                       "secret": "should be dropped"},
            "tools": [{"name": "desktop__fs_read", "family": "files",
                       "description": "…", "inputSchema": {"type": "object"},
                       "mutates": False}],
        })
        ws.send_json({"type": "ping"})
        await ws.next_frame()                       # the pong; hello is done

        tools = desktop_bridge.device_tools(bundle["user_id"])
        assert [t["name"] for t in tools] == ["desktop__fs_read"]
        # camelCase survives: the type is frozen in ToupSeam with no
        # CodingKeys, so it is the one camelCase key on the wire.
        assert "inputSchema" in tools[0]
        summary = desktop_bridge.device_summary(bundle["user_id"])
        assert summary["device"]["name"] == "A Mac"
        assert "secret" not in summary["device"]
    finally:
        await ws.dispose()


async def test_an_oversize_frame_is_dropped_without_being_parsed(
    api, auth_headers,
):
    bundle = await _pair(api, auth_headers)
    ws = await _open(_auth(bundle["access_token"]))
    try:
        ws.send_raw("x" * (desktop_bridge.MAX_INBOUND_FRAME_BYTES + 1))
        ws.send_json({"type": "ping"})
        assert (await ws.next_frame())["type"] == "pong"
        eps = desktop_bridge._active_eps(bundle["user_id"])
        assert eps[-1].counters["oversize_frames"] == 1
    finally:
        await ws.dispose()


async def test_an_unknown_frame_type_is_counted_and_never_fatal(
    api, auth_headers,
):
    """§5: a newer app build may send frames this server has never heard
    of."""
    bundle = await _pair(api, auth_headers)
    ws = await _open(_auth(bundle["access_token"]))
    try:
        ws.send_json({"type": "something_from_a_newer_build", "x": 1})
        ws.send_raw("{not json")
        ws.send_json({"type": "ping"})
        assert (await ws.next_frame())["type"] == "pong"
        eps = desktop_bridge._active_eps(bundle["user_id"])
        assert eps[-1].counters["unknown_frames"] == 1
        assert eps[-1].counters["bad_json"] == 1
    finally:
        await ws.dispose()


# ══════════════════════════════════════════════════════════════════════
# 5. §8 — presence is a PLATFORM DB fact, not a registry read
# ══════════════════════════════════════════════════════════════════════
async def test_connecting_marks_the_device_online_in_the_database(
    api, auth_headers,
):
    """`agent-tool-relay.md` §1.9: the extension's session routes read an
    in-process registry from the platform, which runs two replicas with
    their own empty copies — so on production they answer []. Everything the
    web UI reads here comes off the row."""
    bundle = await _pair(api, auth_headers)
    before = (await api.get(
        "/api/desktop/devices", headers=auth_headers,
    )).json()["devices"][0]
    assert before["online"] is False

    ws = await _open(_auth(bundle["access_token"]))
    try:
        await ws.settle()
        status = (await api.get(
            "/api/desktop/status", headers=auth_headers,
        )).json()
        assert status["paired"] is True
        assert status["connected"] is True
        assert status["relay_enabled"] is True  # enabled account cohort
    finally:
        await ws.dispose()


async def test_the_heartbeat_reports_a_revoked_device_as_revoked(
    api, auth_headers,
):
    """The third layer of revocation (§1.4): a revoke whose close-push was
    lost still shuts the socket within one heartbeat interval, because the
    heartbeat's ANSWER carries the verdict."""
    bundle = await _pair(api, auth_headers)
    from jose import jwt
    jti = jwt.get_unverified_claims(bundle["access_token"])["jti"]

    live = await desktop_api._touch_last_seen(
        bundle["user_id"], bundle["device_id"], jti, online=True,
    )
    assert live == {"ok": True, "revoked": False}

    async with async_session_maker() as db:
        row = await db.get(DesktopDevice, bundle["device_id"])
        row.revoked_at = datetime.utcnow()
        await db.commit()

    after = await desktop_api._touch_last_seen(
        bundle["user_id"], bundle["device_id"], jti, online=True,
    )
    assert after["revoked"] is True


async def test_the_heartbeat_never_materialises_a_device_row(
    api, auth_headers, test_user_id,
):
    """`extension.py:341-356` heals a missing row on heartbeat. A heartbeat
    that can CREATE a device row is a heartbeat that can un-revoke a Mac."""
    ghost = str(uuid.uuid4())
    res = await desktop_api._touch_last_seen(
        test_user_id, ghost, "nope", online=True,
    )
    assert res["revoked"] is True
    async with async_session_maker() as db:
        assert await db.get(DesktopDevice, ghost) is None


async def test_a_stale_last_seen_reads_as_offline(api, auth_headers):
    bundle = await _pair(api, auth_headers)
    async with async_session_maker() as db:
        row = await db.get(DesktopDevice, bundle["device_id"])
        row.last_seen_at = datetime.utcnow() - timedelta(
            seconds=desktop_api.DEVICE_ONLINE_WINDOW_S + 5,
        )
        await db.commit()
    devices = (await api.get(
        "/api/desktop/devices", headers=auth_headers,
    )).json()["devices"]
    assert devices[0]["online"] is False
