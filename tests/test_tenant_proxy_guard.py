"""PR-15: the platform's read proxies stop piling onto a stalled tenant.

2026-10-02, founder tenant 871bac24 (FINDINGS §3): the tenant's loop stalled,
and every platform read proxy kept dialling it — the job board's 5 s poll of
`GET /api/apps/jobs/` tried the tunnel (15 s) and then re-dialled over HTTPS
even right after a tunnel forward to the same tenant had just timed out; the
autopilot missions poll, day-chats and sessions ran their own ladders on top.

These tests pin the guard in `app/services/tenant_proxy_guard.py`:
  * a timeout (tunnel forward or proxied read) marks the tenant; the next
    guarded read inside the window answers 503 `agent_unreachable` with a
    Retry-After at once — no tunnel, no HTTPS re-dial; after the window the
    dial resumes;
  * a per-tenant cap on in-flight guarded reads answers 503 fast past it;
  * health probes, writes and the voice relay's own tenant calls are exempt.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from app.api import apps_proxy, autopilot_proxy, day_chats, sessions
from app.api import ws_agent_tunnel as tunnel
from app.config import settings
from app.services import tenant_proxy_guard as guard

USER = "tenant-871bac24"
OTHER = "tenant-other"
BACKEND = Path(__file__).resolve().parents[1]


# ── fixtures / fakes ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    tunnel._tunnels.clear()
    tunnel._pending_calls.clear()
    guard.reset_for_test()
    monkeypatch.setattr(settings, "tenant_proxy_timeout_memo_s", 10.0, raising=False)
    monkeypatch.setattr(settings, "tenant_proxy_max_inflight", 8, raising=False)
    yield
    tunnel._tunnels.clear()
    tunnel._pending_calls.clear()
    guard.reset_for_test()


class _Clock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(guard, "_clock", c)
    return c


class _Resp:
    def __init__(self, status: int = 200, payload=None):
        self.status_code = status
        self._payload = [] if payload is None else payload
        self.content = json.dumps(self._payload).encode()
        self.text = self.content.decode()
        self.headers = {}

    def json(self):
        return self._payload


class _Client:
    """Fake shared agent_http client: records every dial."""

    def __init__(self, *, exc: Exception | None = None, status: int = 200,
                 payload=None, gate: asyncio.Event | None = None):
        self.calls: list[tuple[str, str]] = []
        self.exc = exc
        self.status = status
        self.payload = payload
        self.gate = gate

    async def _answer(self, method: str, url: str):
        self.calls.append((method, url))
        if self.gate is not None:
            await self.gate.wait()
        if self.exc is not None:
            raise self.exc
        return _Resp(self.status, self.payload)

    async def get(self, url, **kw):
        return await self._answer("GET", url)

    async def post(self, url, **kw):
        return await self._answer("POST", url)

    async def patch(self, url, **kw):
        return await self._answer("PATCH", url)

    async def delete(self, url, **kw):
        return await self._answer("DELETE", url)

    async def request(self, method, url, **kw):
        return await self._answer(method, url)


def _install_client(monkeypatch, client: _Client) -> _Client:
    monkeypatch.setattr(
        "app.services.agent_http.get_agent_http_client", lambda: client,
    )
    return client


def _no_backoff(monkeypatch):
    monkeypatch.setattr(day_chats, "_PROXY_BACKOFF_S", 0.0)
    monkeypatch.setattr(sessions, "_SESSIONS_PROXY_BACKOFF_S", 0.0)


def _user(uid: str = USER):
    return SimpleNamespace(id=uid)


def _req(query: str = ""):
    return SimpleNamespace(url=SimpleNamespace(query=query))


def _body(resp):
    return json.loads(resp.body)


def _assert_fast_503(resp, why: str = "recent_timeout"):
    assert resp.status_code == 503, resp.status_code
    assert resp.headers["x-toup-reason"] == "agent_unreachable"
    assert resp.headers["x-toup-shed"] == why
    retry_after = int(resp.headers["retry-after"])
    assert 1 <= retry_after <= 10, "the app honours Retry-After only up to 10 s"
    assert _body(resp)["code"] == "agent_unreachable"


class _SilentWebSocket:
    """A tunnel to a stalled tenant: the forward is sent, nothing comes back."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload):
        self.sent.append(payload)


def _install_stalled_tunnel(monkeypatch, user_id: str = USER) -> _SilentWebSocket:
    ws = _SilentWebSocket()
    conn = tunnel.TunnelConnection(user_id, ws)
    conn.accepting_calls = True
    tunnel._tunnels[user_id] = conn
    real_forward = tunnel.send_http_forward

    async def short_forward(uid, method, path, headers=None, body=None, timeout=15.0):
        # The real forward, with the deadline shrunk so the test is fast.
        return await real_forward(uid, method, path, headers, body, timeout=0.05)

    monkeypatch.setattr(tunnel, "send_http_forward", short_forward)
    return ws


async def _agent(*_a, **_k):
    return ("https://agent-871bac24.invalid", "key", None)


# ── settings ──────────────────────────────────────────────────────────


def test_settings_are_declared_in_config_with_safe_defaults():
    from app.config import Settings

    fields = Settings.model_fields
    assert fields["tenant_proxy_timeout_memo_s"].default == 10.0
    assert fields["tenant_proxy_max_inflight"].default == 8
    # The app's retry ladder ignores a Retry-After above 10 s; the memo's
    # Retry-After is the window left, so the default must stay inside it.
    assert fields["tenant_proxy_timeout_memo_s"].default <= 10


# ── (1) the memo: a tunnel timeout skips the HTTPS re-dial ────────────


@pytest.mark.asyncio
async def test_tunnel_forward_timeout_marks_the_tenant():
    ws = _SilentWebSocket()
    conn = tunnel.TunnelConnection(USER, ws)
    conn.accepting_calls = True
    tunnel._tunnels[USER] = conn

    assert guard.recent_timeout_remaining(USER) == 0.0
    result = await tunnel.send_http_forward(USER, "GET", "/api/apps/jobs/", timeout=0.05)
    assert result is None
    assert len(ws.sent) == 1
    assert guard.recent_timeout_remaining(USER) > 0
    assert guard.recent_timeout_remaining(OTHER) == 0.0, "per tenant, not global"


@pytest.mark.asyncio
async def test_list_jobs_after_a_tunnel_timeout_answers_503_without_an_https_redial(monkeypatch):
    ws = _install_stalled_tunnel(monkeypatch)
    direct = []

    async def forbidden_direct(*a, **k):
        direct.append(a)
        raise AssertionError("HTTPS re-dial right after a tunnel timeout")

    monkeypatch.setattr(apps_proxy, "_get_agent", _agent)
    monkeypatch.setattr(apps_proxy, "_proxy", forbidden_direct)

    resp = await apps_proxy.list_jobs(_req(), current_user=_user(), db=object())

    _assert_fast_503(resp)
    assert len(ws.sent) == 1, "the tunnel was tried once"
    assert direct == []


@pytest.mark.asyncio
async def test_list_apps_after_a_tunnel_timeout_answers_503_without_an_https_redial(monkeypatch):
    ws = _install_stalled_tunnel(monkeypatch)
    client = _install_client(monkeypatch, _Client(payload=[]))
    monkeypatch.setattr(apps_proxy, "_get_agent", _agent)

    resp = await apps_proxy.list_apps(current_user=_user(), db=object())

    _assert_fast_503(resp)
    assert len(ws.sent) == 1
    assert client.calls == []


@pytest.mark.asyncio
async def test_jobs_events_after_a_tunnel_timeout_answers_503_without_an_https_redial(monkeypatch):
    ws = _install_stalled_tunnel(monkeypatch)
    client = _install_client(monkeypatch, _Client(payload={"events": []}))
    monkeypatch.setattr(apps_proxy, "_get_agent", _agent)

    resp = await apps_proxy.jobs_events_proxy(_req("limit=5"), current_user=_user(), db=object())

    _assert_fast_503(resp)
    assert len(ws.sent) == 1
    assert client.calls == []


@pytest.mark.asyncio
async def test_next_reads_inside_the_window_never_dial_any_route(monkeypatch, clock):
    """After one timeout, every guarded read to that tenant — apps, jobs,
    events, a job, autopilot missions/approvals, day-chats, sessions — is a
    fast 503 with the window's Retry-After; nothing reaches the tunnel or
    HTTPS. Another tenant is untouched."""
    guard.note_timeout(USER)
    clock.now += 3.2  # 6.8 s left → Retry-After 7
    ws = _SilentWebSocket()
    conn = tunnel.TunnelConnection(USER, ws)
    conn.accepting_calls = True
    tunnel._tunnels[USER] = conn
    client = _install_client(monkeypatch, _Client(payload=[]))
    monkeypatch.setattr(apps_proxy, "_get_agent", _agent)

    async def _ap_agent(*_a, **_k):
        return ("https://agent-871bac24.invalid", "key")

    monkeypatch.setattr(autopilot_proxy, "_agent_info", _ap_agent)

    responses = [
        await apps_proxy.list_jobs(_req(), current_user=_user(), db=object()),
        await apps_proxy.list_apps(current_user=_user(), db=object()),
        await apps_proxy.jobs_events_proxy(_req(), current_user=_user(), db=object()),
        await apps_proxy.get_job("job-1", current_user=_user(), db=object()),
        await autopilot_proxy.list_missions_proxy(_req(), current_user=_user(), db=object()),
        await autopilot_proxy.list_approvals_proxy(_req(), current_user=_user(), db=object()),
    ]
    for resp in responses:
        _assert_fast_503(resp)
        assert resp.headers["retry-after"] == "7"

    with pytest.raises(day_chats.AgentUnreachable) as dc:
        await day_chats._proxy_day_chats("https://a", "k", "", {}, user_id=USER)
    http = day_chats._unreachable(dc.value)
    assert http.status_code == 503
    assert http.headers["Retry-After"] == "7"
    assert http.headers["X-Toup-Reason"] == "agent_unreachable"
    assert http.headers["X-Toup-Shed"] == "recent_timeout"

    with pytest.raises(sessions.SessionsAgentUnreachable) as ss:
        await sessions._proxy_sessions("https://a", "k", "", {}, user_id=USER)
    http = sessions._sessions_unreachable(ss.value)
    assert http.status_code == 503
    assert http.headers["Retry-After"] == "7"
    assert http.headers["X-Toup-Shed"] == "recent_timeout"

    assert ws.sent == [], "no tunnel forward inside the window"
    assert client.calls == [], "no HTTPS dial inside the window"

    # Another tenant still dials.
    other = await day_chats._proxy_day_chats("https://b", "k", "", {}, user_id=OTHER)
    assert other == []
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_after_the_window_the_dial_resumes(monkeypatch, clock):
    guard.note_timeout(USER)
    client = _install_client(monkeypatch, _Client(payload=[{"id": "j1"}]))
    monkeypatch.setattr(tunnel, "is_agent_connected", lambda _uid: False)
    monkeypatch.setattr(apps_proxy, "_get_agent", _agent)

    clock.now += 9.9
    resp = await apps_proxy.list_jobs(_req(), current_user=_user(), db=object())
    assert resp.status_code == 503 and client.calls == []

    clock.now += 0.2  # window over
    resp = await apps_proxy.list_jobs(_req(), current_user=_user(), db=object())
    assert resp.status_code == 200
    assert _body(resp) == [{"id": "j1"}]
    assert len(client.calls) == 1
    assert guard.recent_timeout_remaining(USER) == 0.0


@pytest.mark.asyncio
async def test_an_https_read_timeout_marks_the_tenant_and_the_next_read_is_fast(monkeypatch):
    """day-chats/sessions never use the tunnel: their own timed-out ladder is
    what marks the tenant (in the incident trail, attempt 1 ReadTimeout at
    02:28:34.811, then the sessions and missions polls piled on)."""
    _no_backoff(monkeypatch)
    client = _install_client(monkeypatch, _Client(exc=httpx.ReadTimeout("")))

    with pytest.raises(day_chats.AgentUnreachable) as first:
        await day_chats._proxy_day_chats("https://a", "k", "", {}, user_id=USER)
    assert first.value.shed is None, "the first one really dialled"
    assert len(client.calls) == 2, "the existing 2-attempt ladder is unchanged"
    assert guard.recent_timeout_remaining(USER) > 0

    with pytest.raises(sessions.SessionsAgentUnreachable) as second:
        await sessions._proxy_sessions("https://a", "k", "by-date/2026-10-02/messages",
                                       {}, user_id=USER)
    assert second.value.shed == "recent_timeout"
    assert len(client.calls) == 2, "no second 12 s ladder"


@pytest.mark.asyncio
async def test_apps_direct_read_timeout_marks_the_tenant(monkeypatch):
    client = _install_client(monkeypatch, _Client(exc=httpx.ReadTimeout("")))
    monkeypatch.setattr(tunnel, "is_agent_connected", lambda _uid: False)
    monkeypatch.setattr(apps_proxy, "_get_agent", _agent)

    with pytest.raises(HTTPException) as exc:
        await apps_proxy.list_jobs(_req(), current_user=_user(), db=object())
    assert exc.value.status_code == 502  # unchanged first-failure shape
    assert guard.recent_timeout_remaining(USER) > 0

    resp = await apps_proxy.list_jobs(_req(), current_user=_user(), db=object())
    _assert_fast_503(resp)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_a_non_timeout_failure_does_not_mark(monkeypatch):
    _no_backoff(monkeypatch)
    _install_client(monkeypatch, _Client(exc=httpx.ConnectError("refused")))
    with pytest.raises(day_chats.AgentUnreachable):
        await day_chats._proxy_day_chats("https://a", "k", "", {}, user_id=USER)
    assert guard.recent_timeout_remaining(USER) == 0.0

    # The platform's own pool being full is not the tenant's fault.
    _install_client(monkeypatch, _Client(exc=httpx.PoolTimeout("")))
    with pytest.raises(day_chats.AgentUnreachable):
        await day_chats._proxy_day_chats("https://a", "k", "", {}, user_id=USER)
    assert guard.recent_timeout_remaining(USER) == 0.0


@pytest.mark.asyncio
async def test_memo_disabled_by_zero_window(monkeypatch):
    monkeypatch.setattr(settings, "tenant_proxy_timeout_memo_s", 0.0, raising=False)
    guard.note_timeout(USER)
    assert guard.recent_timeout_remaining(USER) == 0.0


@pytest.mark.asyncio
async def test_helpers_called_without_user_id_dial_exactly_as_before(monkeypatch):
    """Back-compat: the helpers' existing positional callers are unguarded."""
    guard.note_timeout(USER)
    client = _install_client(monkeypatch, _Client(payload=[]))
    assert await day_chats._proxy_day_chats("https://a", "k") == []
    assert await sessions._proxy_sessions("https://a", "k", "") == []
    assert len(client.calls) == 2


# ── (2) the in-flight cap ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inflight_cap_answers_503_fast_and_frees_slots(monkeypatch):
    monkeypatch.setattr(settings, "tenant_proxy_max_inflight", 2, raising=False)
    gate = asyncio.Event()
    client = _install_client(monkeypatch, _Client(payload=[], gate=gate))

    first = asyncio.create_task(
        day_chats._proxy_day_chats("https://a", "k", "", {}, user_id=USER))
    second = asyncio.create_task(
        sessions._proxy_sessions("https://a", "k", "", {}, user_id=USER))
    for _ in range(5):
        await asyncio.sleep(0)
    assert guard.inflight(USER) == 2
    assert len(client.calls) == 2

    # The third read to the same tenant: refused before dialling.
    with pytest.raises(day_chats.AgentUnreachable) as third:
        await day_chats._proxy_day_chats("https://a", "k", "", {}, user_id=USER)
    assert third.value.shed == "inflight_cap"
    assert third.value.retry_after == guard.INFLIGHT_RETRY_AFTER_S
    http = day_chats._unreachable(third.value)
    assert http.status_code == 503
    assert http.headers["X-Toup-Reason"] == "agent_unreachable"
    assert http.headers["X-Toup-Shed"] == "inflight_cap"

    monkeypatch.setattr(apps_proxy, "_get_agent", _agent)
    monkeypatch.setattr(tunnel, "is_agent_connected", lambda _uid: False)
    resp = await apps_proxy.list_jobs(_req(), current_user=_user(), db=object())
    _assert_fast_503(resp, why="inflight_cap")
    assert len(client.calls) == 2

    # Another tenant has its own budget.
    other = asyncio.create_task(
        day_chats._proxy_day_chats("https://b", "k", "", {}, user_id=OTHER))
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(client.calls) == 3

    gate.set()
    assert await first == [] and await second == [] and await other == []
    assert guard.inflight(USER) == 0 and guard.inflight(OTHER) == 0

    # Slots freed: the next read dials again.
    assert await day_chats._proxy_day_chats("https://a", "k", "", {}, user_id=USER) == []
    assert len(client.calls) == 4


@pytest.mark.asyncio
async def test_a_failing_read_releases_its_slot(monkeypatch):
    _no_backoff(monkeypatch)
    monkeypatch.setattr(settings, "tenant_proxy_max_inflight", 1, raising=False)
    _install_client(monkeypatch, _Client(exc=httpx.ConnectError("refused")))
    for _ in range(3):
        with pytest.raises(day_chats.AgentUnreachable) as exc:
            await day_chats._proxy_day_chats("https://a", "k", "", {}, user_id=USER)
        assert exc.value.shed is None
    assert guard.inflight(USER) == 0


@pytest.mark.asyncio
async def test_inflight_cap_disabled_by_zero(monkeypatch):
    monkeypatch.setattr(settings, "tenant_proxy_max_inflight", 0, raising=False)
    gate = asyncio.Event()
    client = _install_client(monkeypatch, _Client(payload=[], gate=gate))
    tasks = [
        asyncio.create_task(day_chats._proxy_day_chats("https://a", "k", "", {}, user_id=USER))
        for _ in range(12)
    ]
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(client.calls) == 12
    gate.set()
    await asyncio.gather(*tasks)


def test_default_cap_is_eight():
    assert guard._max_inflight() == 8


# ── (3) exemptions: health probes, writes, the voice relay ────────────


def _saturate(user_id: str = USER):
    """The worst case for an exempt path: marked AND at the cap."""
    guard.note_timeout(user_id)
    guard._inflight[user_id] = 10_000


@pytest.mark.asyncio
async def test_health_probe_is_never_shed(monkeypatch):
    _saturate()
    client = _install_client(monkeypatch, _Client(payload={"cpu": 1}))
    monkeypatch.setattr(apps_proxy, "_get_agent", _agent)
    resp = await apps_proxy.get_server_info(current_user=_user(), db=object())
    assert resp.status_code == 200
    assert _body(resp)["status"] == "online"
    assert client.calls == [("GET", "https://agent-871bac24.invalid/agent/system")]


@pytest.mark.asyncio
async def test_writes_are_never_shed(monkeypatch):
    _saturate()
    client = _install_client(monkeypatch, _Client(payload={"ok": True}))
    monkeypatch.setattr(apps_proxy, "_get_agent", _agent)

    async def _ap_agent(*_a, **_k):
        return ("https://agent-871bac24.invalid", "key")

    monkeypatch.setattr(autopilot_proxy, "_agent_info", _ap_agent)

    resp = await apps_proxy.create_job_proxy({"title": "t"}, current_user=_user(), db=object())
    assert resp.status_code == 200
    resp = await apps_proxy.update_job_proxy("j1", {"status": "done"},
                                             current_user=_user(), db=object())
    assert resp.status_code == 200
    resp = await autopilot_proxy._forward(_req(), USER, object(), "missions/m1/pause",
                                          method="POST", body={})
    assert resp.status_code == 200
    assert [m for m, _ in client.calls] == ["POST", "PATCH", "POST"]


@pytest.mark.asyncio
async def test_the_voice_relays_own_tenant_calls_are_never_shed(monkeypatch):
    from app.api import ws_realtime

    _saturate()
    client = _install_client(monkeypatch, _Client(payload={"id": "row"}))
    got = await ws_realtime._vps_api(
        "https://agent-871bac24.invalid", "key", "POST",
        "/api/sessions/s1/messages", json_body={"role": "user", "content": "hi"},
    )
    assert got == {"id": "row"}
    got = await ws_realtime._vps_api(
        "https://agent-871bac24.invalid", "key", "GET", "/api/voice-tasks/t1",
    )
    assert got == {"id": "row"}
    assert len(client.calls) == 2


def test_relay_and_health_modules_never_consult_the_guard():
    """Structural half of the exemption: the guard is opt-in per helper, and
    the relay / health-probe modules do not opt in."""
    for rel in (
        "app/services/live_voice_protocol.py",
        "app/api/ws_realtime.py",
        "app/api/agent_setup.py",
    ):
        src = (BACKEND / rel).read_text()
        assert "tenant_proxy_guard" not in src, f"{rel} must stay exempt"
    src = (BACKEND / "app/api/apps_proxy.py").read_text()
    server_info = src.split("async def get_server_info", 1)[1].split("@router", 1)[0]
    assert "_guard" not in server_info and "_guarded_read" not in server_info


def test_retry_after_rounds_up_and_never_says_zero(clock):
    guard.note_timeout(USER)
    clock.now += 9.95
    with pytest.raises(guard.TenantProxyShed) as exc:
        guard.check_recent_timeout(USER)
    assert exc.value.retry_after == 1
    clock.now += 0.04
    with pytest.raises(guard.TenantProxyShed) as exc:
        guard.check_recent_timeout(USER)
    assert exc.value.retry_after == math.ceil(10.0 - 9.99) == 1
