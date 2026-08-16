"""Admin Dispatch — the PLATFORM half (contract §10, items 1-6 and 8-13).

An operator composes one message and it lands on two surfaces: a chat card
written into the tenant agent over HTTP, and an ``announcement``
notification. The agent-side writer/retract (§10.7) is covered by its own
suite under RUN_MODE=agent; everything here runs RUN_MODE=platform.

What each test is actually defending — these are behaviours, not lines:

  1. Every route on the dispatch router refuses a non-admin. Enumerated FROM
     THE ROUTER, so a route added later without ``Depends(require_admin)``
     fails here rather than shipping an open admin endpoint.
  2. A broadcast is unrecallable, so it costs a typed confirmation.
  3. One dispatch → exactly one target, one queue row whose data_json is the
     §4 shape field-for-field, and one agent hop with the §5 body. The URL,
     the X-Agent-Key header and the JSON body are all asserted: the mobile
     and web clients render off that payload.
  4. An announcement must escape the two suppressions the user's own agent
     is subject to (``autopilot_push`` off, daily cap spent) — it is not the
     agent talking — while still respecting quiet hours unless urgent.
  5. It opens a Live Activity card, and yields to a live ``reminder:``
     countdown (founder rule D6).
  6. The tenant ingest route may not author one.
  8. The read receipt is the only one in the system: it is recorded even
     when the agent is unreachable, and only ``once`` proxies a retract.
     It is also exactly-once per target — the CAS on ``read_at IS NULL`` is
     the only thing licensing the blind ``read_count + 1`` — and it is
     scoped to the reader, so a stranger's ack moves nobody's counters.
  9. A broadcast drips, or it starves the reminder lane behind it (D8).
 10. Two replicas fan out the same dispatch; the CAS means each target is
     delivered exactly once.
 11. All three serializers emit ``admin_notice`` — the clients fall back
     between them, and a field on only one path vanishes on the fallback.
 12. ``admin`` is a known + system channel AND is covered by the partial
     unique index: the resolver's IntegrityError→re-SELECT race recovery is
     only correct when something actually rejects the second insert.
 13. Table partitioning still passes (run separately; not duplicated here).
"""

from __future__ import annotations

import asyncio
import ast
import json
import types
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.api.admin.dispatch import router as dispatch_router
from app.api.auth import get_current_user
from app.api.notices import router as notices_router
from app.config import settings
from app.db import get_db, async_session_maker
from app.db.models import (
    AdminDispatch,
    AdminDispatchTarget,
    AdminThreadMessage,
    AgentConfig,
    LiveActivity,
    LiveActivityDevice,
    NotificationQueue,
    User,
    LA_STARTED,
    NQ_QUEUED,
)


BACKEND = Path(__file__).resolve().parent.parent


# ── Harness ───────────────────────────────────────────────────────
#
# Same shape as tests/test_cache_telemetry.py::admin_app_client — a minimal
# app mounting only the routers under test, with get_db overridden onto the
# real session maker and get_current_user swapped per test.


def _principal(uid: str, role: str = "user", email: str = "u@x.com"):
    return types.SimpleNamespace(id=uid, role=role, email=email, name="U")


@pytest_asyncio.fixture
async def dispatch_client():
    app = FastAPI()
    app.include_router(dispatch_router, prefix=settings.api_prefix)
    app.include_router(notices_router, prefix=settings.api_prefix)

    async def _override_db():
        async with async_session_maker() as db:
            yield db

    app.dependency_overrides[get_db] = _override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c, app


async def _mk_user(*, role: str = "user", created_at: datetime | None = None) -> str:
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"ad-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12),
            name="Dispatch Test",
            role=role,
            timezone="America/Toronto",
            created_at=created_at or datetime.utcnow(),
            notification_preferences={
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
            },
        ))
        await db.commit()
    return user_id


async def _mk_agent(
    user_id: str,
    url: str = "https://agent.example",
    deploy_status: str = "active",
) -> str:
    """`deploy_status` is a parameter because "no agent" and "the agent is not
    reachable right now" are different facts that `agent_proxy_info` collapses
    into one None — and the retract path has to tell them apart."""
    key = f"tk-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=user_id, agent_api_key=key, agent_url=url,
            deploy_status=deploy_status,
        ))
        await db.commit()
    return key


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _AgentSpy:
    """Stands in for the shared agent HTTP client. Records every hop and
    answers 201, which the worker must accept (contract §6.3: any 2xx)."""

    def __init__(self, status_code: int = 201, boom: Exception | None = None):
        self.calls: list[dict] = []
        self.status_code = status_code
        self.boom = boom

    async def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {},
                           "json": json, "timeout": timeout})
        if self.boom is not None:
            raise self.boom
        return _FakeResponse(
            self.status_code,
            {"message_id": f"msg-{len(self.calls)}", "day_chat_id": None, "ws_count": 0},
        )


def _patch_agent_http(monkeypatch, spy: _AgentSpy) -> _AgentSpy:
    # The worker imports get_agent_http_client INSIDE _agent_hop, so patching
    # the source module is what the call site actually resolves.
    from app.services import agent_http

    monkeypatch.setattr(agent_http, "get_agent_http_client", lambda: spy)
    return spy


def _compose(**overrides) -> dict:
    body = {
        "mode": "once",
        "audience": "user",
        "title": "Scheduled maintenance",
        "body": "Toup will be briefly unavailable at 02:00 UTC.",
    }
    body.update(overrides)
    return body


# ── 1. require_admin on EVERY dispatch route ──────────────────────


def _sample_path(path: str) -> str:
    """Fill path params with values that exist for nobody — a guarded route
    must answer 403 before it can discover they are missing."""
    return (path
            .replace("{dispatch_id}", str(uuid.uuid4()))
            .replace("{user_id}", str(uuid.uuid4())))


@pytest.mark.asyncio
async def test_every_dispatch_route_refuses_a_non_admin(dispatch_client):
    """Enumerated from the router itself, so a route added later without
    Depends(require_admin) fails HERE instead of shipping open. The
    /api/admin/ prefix is a naming convention — admin/system.py is mounted
    under it with no guard at all."""
    client, app = dispatch_client
    non_admin = await _mk_user(role="user")
    app.dependency_overrides[get_current_user] = lambda: _principal(non_admin, role="user")

    checked = []
    for route in dispatch_router.routes:
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            url = settings.api_prefix + _sample_path(route.path)
            res = await client.request(method, url, json={} if method != "GET" else None)
            checked.append((method, route.path, res.status_code))
            assert res.status_code == 403, (
                f"{method} {url} answered {res.status_code}, not 403 — "
                "every dispatch route needs Depends(require_admin)"
            )

    # The enumeration itself must not silently cover nothing.
    assert len(checked) >= 7, checked


@pytest.mark.asyncio
async def test_admin_reaches_the_same_routes(dispatch_client):
    """Counterpart to the above: 403 for everyone would also pass it."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    res = await client.get("/api/admin/dispatch")
    assert res.status_code == 200, res.text
    assert res.json() == {"dispatches": []}


# ── 2. The broadcast confirmation gate ────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_requires_the_typed_confirmation(dispatch_client, monkeypatch):
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    spawned: list[str] = []

    async def _fake_spawn(dispatch_id: str):
        spawned.append(dispatch_id)

    monkeypatch.setattr("app.api.admin.dispatch.spawn_dispatch_fanout", _fake_spawn)

    res = await client.post("/api/admin/dispatch", json=_compose(audience="all"))
    assert res.status_code == 400, res.text
    assert not spawned, "an unconfirmed broadcast must not reach the fan-out"
    async with async_session_maker() as db:
        assert (await db.execute(select(AdminDispatch))).scalars().all() == []

    res = await client.post(
        "/api/admin/dispatch", json=_compose(audience="all", confirm="BROADCAST"),
    )
    assert res.status_code == 201, res.text
    d = res.json()["dispatch"]
    assert d["audience"] == "all" and d["target_user_id"] is None
    assert d["status"] == "queued"
    assert spawned == [d["id"]]

    # A near-miss is still a refusal — the word is the guard rail.
    res = await client.post(
        "/api/admin/dispatch", json=_compose(audience="all", confirm="broadcast"),
    )
    assert res.status_code == 400, res.text


# ── 3. Single-user fan-out: one target, one queue row, one agent hop ──


@pytest.mark.asyncio
async def test_single_user_dispatch_writes_target_queue_row_and_agent_hop(
    dispatch_client, monkeypatch,
):
    from app.services.admin_dispatch_worker import run_dispatch_fanout

    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    recipient = await _mk_user()
    agent_key = await _mk_agent(recipient, url="https://tenant-7.example/")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    spy = _patch_agent_http(monkeypatch, _AgentSpy())

    async def _no_spawn(dispatch_id: str):
        return None

    monkeypatch.setattr("app.api.admin.dispatch.spawn_dispatch_fanout", _no_spawn)

    res = await client.post("/api/admin/dispatch", json=_compose(
        mode="persistent", audience="user", target_user_id=recipient,
        title="Your plan changed", body="You are now on Builder.",
    ))
    assert res.status_code == 201, res.text
    dispatch_id = res.json()["dispatch"]["id"]

    summary = await run_dispatch_fanout(dispatch_id)
    assert summary["target_count"] == 1, summary
    assert summary["delivered_count"] == 1, summary
    assert summary["failed_count"] == 0, summary
    assert summary["status"] == "sent"

    async with async_session_maker() as db:
        targets = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().all()
        assert len(targets) == 1, "exactly one target row per recipient"
        target = targets[0]
        assert target.user_id == recipient
        assert target.state == "done"
        assert target.chat_status == "delivered"
        assert target.chat_message_id == "msg-1"
        assert target.attempts == 1

        rows = (await db.execute(
            select(NotificationQueue).where(NotificationQueue.user_id == recipient)
        )).scalars().all()
        assert len(rows) == 1, "exactly one notification row"
        nq = rows[0]
        assert target.notification_id == nq.id

        # Contract §4, field for field.
        assert nq.source == "platform"
        assert nq.event_kind == "announcement"
        assert nq.title == "Your plan changed"
        assert nq.body == "You are now on Builder."
        assert nq.priority == "default"          # urgent=False
        assert nq.idempotency_key == f"admin-dispatch:{dispatch_id}:{recipient}"
        assert nq.scheduled_for is None          # first of the burst
        assert nq.data_json == {
            "kind": "announcement",
            "mission_id": f"admin:{dispatch_id}",
            "dispatch_id": dispatch_id,
            "mode": "persistent",
            "deep_link": f"toup://notices?mission=admin:{dispatch_id}",
            "cap_exempt": True,
            "urgent": False,
        }
        # mission_id rides in a 64-char Live Activity attribute.
        assert len(nq.data_json["mission_id"]) <= 64
        # No colon in the deep-link ROUTE segment (WHATWG reads it as a port);
        # the ?mission= value may carry one.
        assert ":" not in nq.data_json["deep_link"].split("://", 1)[1].split("?", 1)[0]

        # persistent mode opens the thread with the operator's own row.
        thread = (await db.execute(
            select(AdminThreadMessage).where(AdminThreadMessage.user_id == recipient)
        )).scalars().all()
        assert len(thread) == 1
        assert thread[0].direction == "out"
        assert thread[0].dispatch_id == dispatch_id

    # Contract §5: the agent hop.
    assert len(spy.calls) == 1, spy.calls
    call = spy.calls[0]
    assert call["url"] == "https://tenant-7.example/api/internal/admin-notice"
    assert call["headers"].get("X-Agent-Key") == agent_key
    body = call["json"]
    assert set(body) == {
        "user_id", "dispatch_id", "mode", "title", "body", "sender_name", "sent_at",
    }, body
    assert body["user_id"] == recipient
    assert body["dispatch_id"] == dispatch_id
    assert body["mode"] == "persistent"
    assert body["title"] == "Your plan changed"
    assert body["body"] == "You are now on Builder."
    assert body["sender_name"] == settings.admin_dispatch_sender_name
    datetime.fromisoformat(body["sent_at"])  # ISO-8601, parseable


@pytest.mark.asyncio
async def test_recipient_with_no_agent_is_done_not_failed(dispatch_client, monkeypatch):
    """The notification still landed — a user whose container is down is a
    recorded fact (`no_agent`), not a failed send."""
    from app.services.admin_dispatch_worker import run_dispatch_fanout

    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    recipient = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    spy = _patch_agent_http(monkeypatch, _AgentSpy())
    monkeypatch.setattr("app.api.admin.dispatch.spawn_dispatch_fanout",
                        lambda _id: asyncio.sleep(0))

    res = await client.post("/api/admin/dispatch", json=_compose(
        audience="user", target_user_id=recipient))
    dispatch_id = res.json()["dispatch"]["id"]
    await run_dispatch_fanout(dispatch_id)

    assert spy.calls == [], "no agent config → no hop"
    async with async_session_maker() as db:
        target = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()
        assert (target.state, target.chat_status) == ("done", "no_agent")
        assert target.notification_id is not None


# ── 4. evaluate_policy: the two suppressions an announcement escapes ──


def _prefs(**overrides):
    from app.api.account import _merged_prefs

    base = _merged_prefs(None)
    base.update(overrides)
    return base


def _announcement_row(dispatch_id: str = "d-1", *, urgent: bool = False):
    return NotificationQueue(
        event_kind="announcement",
        title="Scheduled maintenance",
        body="Back at 02:30 UTC.",
        priority="high" if urgent else "default",
        data_json={
            "kind": "announcement",
            "mission_id": f"admin:{dispatch_id}",
            "dispatch_id": dispatch_id,
            "mode": "once",
            "deep_link": f"toup://notices?mission=admin:{dispatch_id}",
            "cap_exempt": True,
            "urgent": urgent,
        },
    )


def test_announcement_escapes_autopilot_toggle_and_daily_cap():
    """The two suppressions that exist because the AGENT is acting. An
    operator message is not the agent, so neither may silence it: a user who
    turned autopilot pushes off did not opt out of hearing from Toup, and cap
    slot #11 behind ten of the agent's own pushes is where the one message we
    ever send would land."""
    import app.services.notification_dispatcher as nd

    # Fixed daytime instant — utcnow() at night would trip the default
    # quiet-hours window and turn 'send' into 'defer'.
    now = datetime(2026, 8, 13, 15, 0)
    prefs = _prefs(autopilot_push=False, daily_push_cap=3)

    decision, reason = nd.evaluate_policy(
        _announcement_row(), prefs, "UTC", now, 99, False,
    )
    assert (decision, reason) == ("send", None)

    # Control: the same prefs DO suppress an agent-authored mission row —
    # so the assertion above is about the kind, not about broken prefs.
    agent_row = NotificationQueue(event_kind="mission_completed", priority="default")
    assert nd.evaluate_policy(agent_row, prefs, "UTC", now, 99, False)[0] == "suppress"

    # The bypass is on the EVENT KIND, not only on data.cap_exempt: a
    # producer that forgets the data key must still not be silenceable.
    bare = NotificationQueue(
        event_kind="announcement", priority="default", data_json={},
    )
    assert nd.evaluate_policy(bare, prefs, "UTC", now, 99, False) == ("send", None)


def test_announcement_still_defers_under_quiet_hours_unless_urgent():
    """Escaping the agent's suppressions is not a licence to wake someone at
    23:00. `urgent` is the operator's explicit judgement, per dispatch."""
    import app.services.notification_dispatcher as nd

    now_utc = datetime(2026, 8, 13, 3, 0)  # 23:00 America/Toronto
    prefs = _prefs(quiet_hours={"enabled": True, "start": "22:00", "end": "08:00"})

    decision, until = nd.evaluate_policy(
        _announcement_row(), prefs, "America/Toronto", now_utc, 0, False,
    )
    assert decision == "defer"
    assert until == datetime(2026, 8, 13, 12, 0)  # 08:00 EDT

    decision, _ = nd.evaluate_policy(
        _announcement_row(urgent=True), prefs, "America/Toronto", now_utc, 0, False,
    )
    assert decision == "send"


def test_announcement_never_fans_out_to_a_third_party_channel():
    """One broadcast must not become every user's Telegram message (§4)."""
    import app.services.notification_dispatcher as nd

    assert "announcement" in nd._CAP_BYPASS_KINDS
    assert "announcement" not in nd._FALLBACK_KINDS


# ── 5. Live Activity lane: start branch, and REMINDER WINS ────────


def _patch_apns(monkeypatch, sent: list, status: int = 200):
    from app.services import live_activity_service as las

    async def fake_send(token, payload, *, environment="development", priority=10):
        sent.append({"token": token, "payload": payload})
        return status, ""

    monkeypatch.setattr(las.apns_push, "send_live_activity", fake_send)
    monkeypatch.setattr(settings, "apns_key_b64", "eA==")
    monkeypatch.setattr(settings, "apns_key_id", "KEY123")
    monkeypatch.setattr(settings, "apns_team_id", "TEAM123")


async def _mk_la_device(user_id: str) -> str:
    device_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(LiveActivityDevice(
            id=device_id, user_id=user_id,
            push_to_start_token=uuid.uuid4().hex + uuid.uuid4().hex,
            apns_environment="development",
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return device_id


async def _mk_started(user_id: str, device_id: str, mission_id: str) -> None:
    async with async_session_maker() as db:
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=user_id, mission_id=mission_id,
            device_id=device_id, status=LA_STARTED, started_at=datetime.utcnow(),
        ))
        await db.commit()


async def _enqueue_announcement(user_id: str, dispatch_id: str) -> str:
    row_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        row = _announcement_row(dispatch_id)
        db.add(NotificationQueue(
            id=row_id, user_id=user_id, source="platform",
            event_kind=row.event_kind, title=row.title, body=row.body,
            priority=row.priority, data_json=row.data_json,
            idempotency_key=f"admin-dispatch:{dispatch_id}:{user_id}",
            status=NQ_QUEUED, created_at=datetime.utcnow(),
        ))
        await db.commit()
    return row_id


@pytest.mark.asyncio
async def test_announcement_opens_a_card_via_the_start_branch(monkeypatch):
    from app.services import live_activity_service as las

    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    dispatch_id = str(uuid.uuid4())
    row_id = await _enqueue_announcement(user_id, dispatch_id)

    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        out = await las.handle_notification_row(db, row, datetime.utcnow())
        await db.commit()

    assert out["status"] == "ok" and out["delivered"] is True, out
    assert [s["payload"]["aps"]["event"] for s in sent] == ["start"], sent
    aps = sent[0]["payload"]["aps"]
    assert aps["attributes"]["name"] == f"admin:{dispatch_id}"
    # Visibly disowned from the agent: brand hue, not the user's orb colour.
    assert aps["attributes"]["orbColor"] == "#7C6BF5"

    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(
                LiveActivity.mission_id == f"admin:{dispatch_id}")
        )).scalars().one()
        assert la.status == LA_STARTED


@pytest.mark.asyncio
async def test_announcement_yields_to_a_live_reminder_card(monkeypatch):
    """D6 — an operator message is not more urgent than the alarm the user
    set. It still lands as a chat card and as its own row; only the Island
    yields."""
    from app.services import live_activity_service as las

    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started(user_id, device_id, "reminder:cd0001")
    sent.clear()

    dispatch_id = str(uuid.uuid4())
    row_id = await _enqueue_announcement(user_id, dispatch_id)
    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        out = await las.handle_notification_row(db, row, datetime.utcnow())
        await db.commit()

    assert sent == [], "no start push while a countdown owns the device"
    reasons = [d.get("reason") for d in out["devices"].values()]
    assert reasons == ["yields_to_reminder"], out

    async with async_session_maker() as db:
        # The countdown row is untouched — not preempted, not orphaned.
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "reminder:cd0001")
        )).scalars().one()
        assert la.status == LA_STARTED
        assert (await db.execute(
            select(LiveActivity).where(
                LiveActivity.mission_id == f"admin:{dispatch_id}")
        )).scalars().all() == []


@pytest.mark.asyncio
async def test_preempt_refuses_to_end_a_reminder_for_an_admin_card():
    """Belt-and-braces at the preempt itself: even a direct call must not end
    a countdown to make room for `admin:`."""
    from app.services import live_activity_service as las

    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started(user_id, device_id, "reminder:cd0002")

    async with async_session_maker() as db:
        device = (await db.execute(
            select(LiveActivityDevice).where(LiveActivityDevice.id == device_id)
        )).scalars().one()
        preempted = await las._preempt_device(
            db, device, f"admin:{uuid.uuid4()}", datetime.utcnow(),
        )
        await db.commit()
        assert preempted == 0
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "reminder:cd0002")
        )).scalars().one()
        assert la.status == LA_STARTED


# ── 6. The tenant ingest route may not author an announcement ─────


@pytest.mark.asyncio
async def test_agent_notify_rejects_announcement(client, test_user_id):
    """agent_notify authenticates a TENANT key, so anything arriving with
    this kind is a user's own agent asking to speak as the operator."""
    key = f"tk-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=test_user_id, agent_api_key=key))
        await db.commit()

    body = {
        "user_id": test_user_id,
        "idempotency_key": f"outbox-{uuid.uuid4().hex}",
        "event_kind": "announcement",
        "title": "Scheduled maintenance",
        "body": "impersonating the operator",
        "priority": "default",
    }
    res = await client.post("/api/agent/notify", json=body, headers={"X-Agent-Key": key})
    assert res.status_code == 422, res.text

    async with async_session_maker() as db:
        assert (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.event_kind == "announcement")
        )).scalars().all() == [], "a rejected ingest must enqueue nothing"

    # Control: a legitimate kind on the same route still works, so the 422
    # above is about `announcement` and not a broken request.
    body["event_kind"] = "mission_completed"
    res = await client.post("/api/agent/notify", json=body, headers={"X-Agent-Key": key})
    assert res.status_code == 200, res.text


# ── 8. The read receipt, and the `once` retract ───────────────────


async def _seed_delivered(
    mode: str, user_id: str, chat_status: str = "delivered",
) -> str:
    """One dispatch + one target for `user_id`.

    `chat_status` is a parameter because it is the ONLY record of whether a
    tenant row was ever written, and the retract path has to read it to tell
    "no agent" from "the agent is unreachable right now".
    """
    dispatch_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(AdminDispatch(
            id=dispatch_id, created_by_user_id=None, mode=mode, audience="user",
            target_user_id=user_id, sender_name="Toup", title="Heads up",
            body="Something changed.", urgent=False, status="sent",
            target_count=1, delivered_count=1, created_at=datetime.utcnow(),
        ))
        db.add(AdminDispatchTarget(
            id=str(uuid.uuid4()), dispatch_id=dispatch_id, user_id=user_id,
            state="done", chat_status=chat_status, attempts=1,
            notification_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        await db.commit()
    return dispatch_id


async def _receipt(dispatch_id: str) -> tuple[datetime | None, int]:
    """(target.read_at, dispatch.read_count) — the whole read ledger for a
    single-target dispatch, read back through a fresh session."""
    async with async_session_maker() as db:
        target = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()
        dispatch = await db.get(AdminDispatch, dispatch_id)
        return target.read_at, dispatch.read_count


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expect_retract", [("once", True), ("persistent", False)])
async def test_read_receipt_retracts_only_a_once_notice(
    dispatch_client, monkeypatch, mode, expect_retract,
):
    client, app = dispatch_client
    user_id = await _mk_user()
    await _mk_agent(user_id)
    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    dispatch_id = await _seed_delivered(mode, user_id)

    proxied: list[dict] = []

    async def _fake_proxy(agent_url, agent_api_key, path, method="GET", **kw):
        proxied.append({"url": agent_url, "path": path, "method": method,
                        "json": kw.get("json_body")})
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _fake_proxy)

    res = await client.post(f"/api/notices/{dispatch_id}/read")
    assert res.status_code == 204, res.text

    async with async_session_maker() as db:
        target = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()
        assert target.read_at is not None, "the read receipt is the only one there is"
        dispatch = await db.get(AdminDispatch, dispatch_id)
        assert dispatch.read_count == 1

    if expect_retract:
        assert len(proxied) == 1, proxied
        assert proxied[0]["path"] == "internal/admin-notice/retract"
        assert proxied[0]["method"] == "POST"
        assert proxied[0]["json"] == {"user_id": user_id, "dispatch_id": dispatch_id}
        async with async_session_maker() as db:
            target = (await db.execute(
                select(AdminDispatchTarget).where(
                    AdminDispatchTarget.dispatch_id == dispatch_id)
            )).scalars().one()
            assert target.chat_status == "retracted"
    else:
        assert proxied == [], "a persistent notice stays in the thread — never retracted"
        async with async_session_maker() as db:
            target = (await db.execute(
                select(AdminDispatchTarget).where(
                    AdminDispatchTarget.dispatch_id == dispatch_id)
            )).scalars().one()
            assert target.chat_status == "delivered"


@pytest.mark.asyncio
async def test_read_receipt_survives_an_unreachable_agent(dispatch_client, monkeypatch):
    """The receipt is committed BEFORE the hop. Losing it to a dead container
    would re-serve the notice forever."""
    client, app = dispatch_client
    user_id = await _mk_user()
    await _mk_agent(user_id)
    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    dispatch_id = await _seed_delivered("once", user_id)

    async def _boom(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _boom)

    res = await client.post(f"/api/notices/{dispatch_id}/read")
    assert res.status_code == 204, res.text

    async with async_session_maker() as db:
        target = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()
        assert target.read_at is not None
        # Not 'retracted': only the agent's confirmed delete may claim that,
        # and the failure is recorded for the retry sweep.
        assert target.chat_status == "delivered"
        assert target.last_error and "retract failed" in target.last_error

    # And the badge no longer counts it.
    res = await client.get("/api/notices/state")
    assert res.status_code == 200, res.text
    assert res.json()["unread_notices"] == 0


@pytest.mark.asyncio
async def test_read_receipt_of_a_notice_that_is_not_yours_is_404(dispatch_client):
    client, app = dispatch_client
    owner = await _mk_user()
    stranger = await _mk_user()
    dispatch_id = await _seed_delivered("once", owner)
    app.dependency_overrides[get_current_user] = lambda: _principal(stranger)

    res = await client.post(f"/api/notices/{dispatch_id}/read")
    assert res.status_code == 404, res.text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["once", "persistent"])
async def test_a_second_read_receipt_moves_nothing(dispatch_client, monkeypatch, mode):
    """Exactly-once per target, which is what makes `read_count` mean anything.

    A repeat ack is the ORDINARY case, not an edge one: the web card posts a
    receipt from its mount effect for every `persistent` notice it draws
    (AdminNoticeCard.tsx), so every reload of the thread acks again, and a
    user reading on two devices acks twice regardless of mode. The CAS on
    `read_at IS NULL` is the only thing standing between that and a
    `read_count` that climbs past `target_count` — the increment beneath it
    is blind.
    """
    client, app = dispatch_client
    user_id = await _mk_user()
    await _mk_agent(user_id)
    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    dispatch_id = await _seed_delivered(mode, user_id)

    hops: list[dict] = []

    async def _fake_proxy(agent_url, agent_api_key, path, method="GET", **kw):
        hops.append({"path": path, "json": kw.get("json_body")})
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _fake_proxy)

    res = await client.post(f"/api/notices/{dispatch_id}/read")
    assert res.status_code == 204, res.text
    first_read_at, first_count = await _receipt(dispatch_id)
    assert first_read_at is not None, "the first ack is the receipt"
    assert first_count == 1
    hops_after_first = len(hops)

    # utcnow() carries microseconds, so a re-stamp would be visible here.
    await asyncio.sleep(0.01)

    res = await client.post(f"/api/notices/{dispatch_id}/read")
    assert res.status_code == 204, res.text
    second_read_at, second_count = await _receipt(dispatch_id)
    assert second_read_at == first_read_at, "read_at is when they FIRST read it"
    assert second_count == 1, "the blind increment is licensed by the CAS, and only by it"
    # A `once` card the agent has already deleted must not be deleted again:
    # `chat_status == 'retracted'` is what closes that hop off.
    assert len(hops) == hops_after_first, hops


@pytest.mark.asyncio
async def test_a_second_read_retries_a_retract_that_failed(dispatch_client, monkeypatch):
    """The one thing a second ack DOES do, and the reason the route cannot be
    made idempotent by returning early on `read_at IS NOT NULL`.

    While the hop has not landed, `chat_status` is still 'delivered' and the
    card is still in the user's chat — so the next read has to try again. The
    receipt itself stays exactly-once across both attempts.
    """
    client, app = dispatch_client
    user_id = await _mk_user()
    await _mk_agent(user_id)
    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    dispatch_id = await _seed_delivered("once", user_id)

    attempts: list[str] = []

    async def _boom(agent_url, agent_api_key, path, method="GET", **kw):
        attempts.append(path)
        raise RuntimeError("connection refused")

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _boom)

    assert (await client.post(f"/api/notices/{dispatch_id}/read")).status_code == 204
    read_at, count = await _receipt(dispatch_id)
    assert read_at is not None and count == 1

    async def _ok(agent_url, agent_api_key, path, method="GET", **kw):
        attempts.append(path)
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _ok)

    assert (await client.post(f"/api/notices/{dispatch_id}/read")).status_code == 204
    assert attempts == ["internal/admin-notice/retract"] * 2, attempts

    retried_read_at, retried_count = await _receipt(dispatch_id)
    assert retried_read_at == read_at
    assert retried_count == 1
    async with async_session_maker() as db:
        target = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()
        assert target.chat_status == "retracted"


@pytest.mark.asyncio
async def test_the_receipt_is_committed_before_the_hop_not_after_it(
    dispatch_client, monkeypatch,
):
    """What actually protects the receipt is the commit ORDER, not the
    `except Exception` around the hop — and only a BaseException tells the two
    apart, because that handler catches everything else.

    The case is real: the phone posts "Got it" and the user backgrounds the app,
    so the ASGI task is cancelled mid-retract. Commit the receipt after the hop
    instead and the session closes unflushed — the next history fetch re-serves
    the notice and the card the user just dismissed is back (B6).
    """
    client, app = dispatch_client
    user_id = await _mk_user()
    await _mk_agent(user_id)
    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    dispatch_id = await _seed_delivered("once", user_id)

    async def _cancelled(*a, **kw):
        raise asyncio.CancelledError()

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _cancelled)

    with pytest.raises(asyncio.CancelledError):
        await client.post(f"/api/notices/{dispatch_id}/read")

    read_at, count = await _receipt(dispatch_id)
    assert read_at is not None, "the receipt outlives a hop that never returned"
    assert count == 1


@pytest.mark.asyncio
async def test_a_stranger_ack_moves_nobody_elses_counters(dispatch_client, monkeypatch):
    """The 404 is not the whole assertion — what it must also do is nothing.

    Two dispatches, two users, neither a target of the other's. The lookup is
    scoped by BOTH `dispatch_id` and `user_id`, so dropping either half is a
    cross-tenant write: on one, a stranger stamps the owner's receipt and
    retracts the owner's card; on the other, they credit their own unrelated
    notice against the id in the URL.
    """
    client, app = dispatch_client
    owner = await _mk_user()
    stranger = await _mk_user()
    await _mk_agent(stranger)
    owner_dispatch = await _seed_delivered("once", owner)
    stranger_dispatch = await _seed_delivered("once", stranger)

    # Recorded, not raised: the route's retract guard is a bare
    # `except Exception`, so an AssertionError thrown in here would be
    # swallowed into `last_error` and the test would pass through the bug.
    hops: list[str] = []

    async def _record(agent_url, agent_api_key, path, method="GET", **kw):
        hops.append(path)
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _record)

    app.dependency_overrides[get_current_user] = lambda: _principal(stranger)
    res = await client.post(f"/api/notices/{owner_dispatch}/read")
    assert res.status_code == 404, res.text
    assert hops == [], "a rejected ack must not reach anyone's agent"

    assert await _receipt(owner_dispatch) == (None, 0), "the owner has not read it"
    assert await _receipt(stranger_dispatch) == (None, 0), (
        "and the stranger has not read their own by asking for someone else's"
    )


# ── 9. Broadcast drip (D8) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_drips_in_bursts_of_ten(monkeypatch):
    """A broadcast must never occupy more than a slice of the dispatcher's
    20-row/30s claim batch, or every reminder queues behind it."""
    from app.services import admin_dispatch_worker as w

    n = 23
    base = datetime(2026, 1, 1, 12, 0)
    order: list[str] = []
    for i in range(n):
        order.append(await _mk_user(created_at=base + timedelta(seconds=i)))

    dispatch_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(AdminDispatch(
            id=dispatch_id, created_by_user_id=None, mode="once", audience="all",
            target_user_id=None, sender_name="Toup", title="Everyone",
            body="Broadcast body.", urgent=False, status="queued",
            created_at=datetime.utcnow(),
        ))
        await db.commit()

    t0 = datetime.utcnow()
    summary = await w.run_dispatch_fanout(dispatch_id)
    t1 = datetime.utcnow()
    assert summary["target_count"] == n, summary

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(NotificationQueue.user_id, NotificationQueue.scheduled_for)
        )).all()
    sched = {uid: when for uid, when in rows}
    assert len(sched) == n

    for index, user_id in enumerate(order):
        step = index // w.ADMIN_DISPATCH_BURST
        when = sched[user_id]
        if step == 0:
            assert when is None, f"target {index} must send immediately, got {when}"
        else:
            assert when is not None, f"target {index} must be deferred"
            gap = timedelta(seconds=30 * step)
            assert t0 + gap <= when <= t1 + gap, (
                f"target {index} scheduled {when}, expected ≈ now+{gap}"
            )

    # The shape the rule is actually about: at most BURST rows share a slot.
    buckets: dict = {}
    for when in sched.values():
        buckets.setdefault(None if when is None else round(
            (when - t0).total_seconds() / 30), 0)
        buckets[None if when is None else round((when - t0).total_seconds() / 30)] += 1
    assert max(buckets.values()) <= w.ADMIN_DISPATCH_BURST, buckets


# ── 10. Fan-out claim is a CAS ────────────────────────────────────


@pytest.mark.asyncio
async def test_two_concurrent_fanouts_deliver_each_target_exactly_once(monkeypatch):
    """platform-api runs 2 Railway replicas with no leader election, and the
    retry route can re-enter a fan-out that is still running. The per-row
    status CAS is the whole defence: the loser sees rowcount=0 and skips."""
    from app.services import admin_dispatch_worker as w

    spy = _patch_agent_http(monkeypatch, _AgentSpy())

    recipients = []
    base = datetime(2026, 1, 1, 12, 0)
    for i in range(6):
        uid = await _mk_user(created_at=base + timedelta(seconds=i))
        await _mk_agent(uid, url=f"https://tenant-{i}.example")
        recipients.append(uid)

    dispatch_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(AdminDispatch(
            id=dispatch_id, created_by_user_id=None, mode="persistent",
            audience="all", target_user_id=None, sender_name="Toup",
            title="Racy", body="Twice would be a bug.", urgent=False,
            status="queued", created_at=datetime.utcnow(),
        ))
        await db.commit()

    a, b = await asyncio.gather(
        w.run_dispatch_fanout(dispatch_id),
        w.run_dispatch_fanout(dispatch_id),
    )

    # Every target claimed exactly once, across both workers.
    assert a["claimed"] + b["claimed"] == len(recipients), (a, b)

    async with async_session_maker() as db:
        targets = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().all()
        assert len(targets) == len(recipients), "UNIQUE(dispatch_id,user_id)"
        assert {t.user_id for t in targets} == set(recipients)
        for t in targets:
            assert t.state == "done", t.last_error
            assert t.attempts == 1, "attempts is bumped by the winning CAS only"

        # One side effect per recipient, not two.
        nq = (await db.execute(select(NotificationQueue.user_id))).all()
        assert sorted(u for (u,) in nq) == sorted(recipients)
        thread = (await db.execute(select(AdminThreadMessage.user_id))).all()
        assert sorted(u for (u,) in thread) == sorted(recipients)

    hop_users = [c["json"]["user_id"] for c in spy.calls]
    assert sorted(hop_users) == sorted(recipients), (
        f"one agent hop per recipient; got {hop_users}"
    )


@pytest.mark.asyncio
async def test_fanout_survives_its_own_replica_insert_race(monkeypatch):
    """The same race as the test above, driven DETERMINISTICALLY and in ONE
    session — so the outcome cannot be blamed on the sqlite harness.

    ``_ensure_targets`` documents this exact path: "a second replica
    enumerated the same audience concurrently … the rollback drops the whole
    batch, so re-read and insert what is still genuinely missing". We
    reproduce it by making the FIRST existing-targets read stale (the other
    replica's row is not visible yet) while that row really is in the table,
    so the batch INSERT violates uq_admin_dispatch_target.

    The recovery must leave the broadcast intact: every target delivered,
    every counter honest."""
    from app.services import admin_dispatch_worker as w

    spy = _patch_agent_http(monkeypatch, _AgentSpy())

    base = datetime(2026, 1, 1, 12, 0)
    recipients = []
    for i in range(3):
        uid = await _mk_user(created_at=base + timedelta(seconds=i))
        await _mk_agent(uid, url=f"https://tenant-{i}.example")
        recipients.append(uid)

    dispatch_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(AdminDispatch(
            id=dispatch_id, mode="once", audience="all", target_user_id=None,
            sender_name="Toup", title="Everyone", body="Broadcast body.",
            urgent=False, status="queued", created_at=datetime.utcnow(),
        ))
        # The other replica already materialised one of the targets.
        db.add(AdminDispatchTarget(
            id=str(uuid.uuid4()), dispatch_id=dispatch_id, user_id=recipients[1],
            state="pending", chat_status="pending", attempts=0,
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        await db.commit()

    original = w._target_user_ids
    seen = {"n": 0}

    async def _stale_first_read(db, did):
        seen["n"] += 1
        if seen["n"] == 1:
            return set()  # …and we cannot see it yet
        return await original(db, did)

    monkeypatch.setattr(w, "_target_user_ids", _stale_first_read)

    summary = await w.run_dispatch_fanout(dispatch_id)

    assert summary["target_count"] == len(recipients), summary
    assert summary["failed_count"] == 0, (
        f"the documented insert-race recovery must not fail targets: {summary}"
    )
    assert summary["delivered_count"] == len(recipients), summary
    assert sorted(c["json"]["user_id"] for c in spy.calls) == sorted(recipients)

    async with async_session_maker() as db:
        targets = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().all()
        assert {t.state for t in targets} == {"done"}, [
            (t.user_id, t.state, t.last_error) for t in targets
        ]
        dispatch = await db.get(AdminDispatch, dispatch_id)
        # A dispatch that reads `sent` while every target failed is worse than
        # one that reads `failed`: the panel says the operator was heard.
        assert dispatch.failed_count == 0
        assert dispatch.status == "sent"


@pytest.mark.asyncio
async def test_claim_target_cas_returns_false_for_the_loser():
    """The primitive, in isolation — the end-to-end test above can only see
    its effect."""
    from app.services import admin_dispatch_worker as w

    user_id = await _mk_user()
    dispatch_id = str(uuid.uuid4())
    target_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(AdminDispatch(
            id=dispatch_id, mode="once", audience="user", target_user_id=user_id,
            sender_name="Toup", title="t", body="b", status="queued",
            created_at=datetime.utcnow(),
        ))
        db.add(AdminDispatchTarget(
            id=target_id, dispatch_id=dispatch_id, user_id=user_id,
            state="pending", chat_status="pending", attempts=0,
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        await db.commit()

    async with async_session_maker() as db:
        assert await w._claim_target(db, target_id, datetime.utcnow()) is True
        assert await w._claim_target(db, target_id, datetime.utcnow()) is False
        target = await db.get(AdminDispatchTarget, target_id)
        await db.refresh(target)
        assert target.attempts == 1, "the losing CAS must not bump attempts"


# ── 11. All three serializers emit admin_notice ───────────────────


_NOTICE = {
    "dispatch_id": "d-abc",
    "mode": "once",
    "title": "Scheduled maintenance",
    "sender_name": "Toup",
    "sent_at": "2026-08-13T12:00:00",
}


def _fake_message(metadata: dict | None):
    return types.SimpleNamespace(
        id="m-1", role="assistant", content="Back at 02:30 UTC.",
        created_at=datetime(2026, 8, 13, 12, 0),
        conversation_id="c-1", day_chat_id=None, channel="admin",
        metadata_json=json.dumps(metadata) if metadata is not None else None,
        memories_retrieved_json=None, tokens_prompt=None, tokens_completion=None,
        model_used=None, processing_time_ms=None, attachments=None,
        reply_to_message_id=None,
    )


def test_day_chats_serializer_emits_the_notice():
    from app.api.day_chats import _serialize_admin_notice

    assert _serialize_admin_notice(_fake_message({"admin_notice": _NOTICE})) == _NOTICE
    assert _serialize_admin_notice(_fake_message({"media": {"x": 1}})) is None
    assert _serialize_admin_notice(_fake_message(None)) is None
    # Malformed metadata must be None, never an exception on a history read.
    broken = _fake_message(None)
    broken.metadata_json = "{not json"
    assert _serialize_admin_notice(broken) is None


def test_sessions_serializer_emits_the_notice():
    from app.api.sessions import _message_to_response
    from app.schemas import ChatMessageResponse

    # Declared on the response model, or pydantic drops it silently.
    assert "admin_notice" in ChatMessageResponse.model_fields

    resp = _message_to_response(_fake_message({"admin_notice": _NOTICE}))
    assert resp.admin_notice == _NOTICE
    assert _message_to_response(_fake_message({"media": {"x": 1}})).admin_notice is None
    assert _message_to_response(_fake_message(None)).admin_notice is None


def test_messages_recover_serializer_emits_the_notice():
    """`/messages/since` is the recovery path the clients fall back to, and
    its serializer is inline in the route — so assert (a) it reuses the very
    function the day-chats test above proved, and (b) the payload dict it
    builds actually carries the key."""
    import app.api.messages_recover as mr
    from app.api.day_chats import _serialize_admin_notice

    assert mr._serialize_admin_notice is _serialize_admin_notice

    tree = ast.parse((BACKEND / "app/api/messages_recover.py").read_text())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "messages_since"
    )
    emitted = {
        key.value
        for dict_node in (n for n in ast.walk(fn) if isinstance(n, ast.Dict))
        for key in dict_node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "admin_notice" in emitted, (
        "messages_since builds no admin_notice key — a notice would arrive on "
        "the recovery path as a plain assistant bubble owned by the agent"
    )


# ── 12. `admin` is a real channel, and the index covers it ────────


def test_admin_is_a_known_and_system_channel():
    from app.agent.channel_util import KNOWN_CHANNELS
    from app.agent.conversation_resolver import SYSTEM_CHANNELS

    assert "admin" in KNOWN_CHANNELS, (
        "an unmapped channel logs `resolve_channel default` on every path "
        "that touches the row"
    )
    assert "admin" in SYSTEM_CHANNELS, (
        "without this, repeated dispatches on one day fork a Conversation each"
    )


def test_system_channel_index_predicate_matches_the_resolver():
    """STRUCTURAL, and load-bearing: `resolve_or_create_day_conversation`
    recovers from a concurrent insert by catching IntegrityError and
    re-SELECTing. Nothing raises that error unless the partial unique index
    actually covers the channel — so a SYSTEM_CHANNELS member missing from
    the predicate has silent, advisory-only dedupe."""
    from app.agent.conversation_resolver import (
        INDEXED_SYSTEM_CHANNELS, SYSTEM_CHANNELS,
    )

    src = (BACKEND / "app/db/database.py").read_text()
    assert "ix_conversations_system_channel_per_day" in src

    # The predicate must be BUILT from the resolver's tuple, not hand-copied
    # a third time — three literals is exactly what let them drift.
    assert "_system_channel_sql = " in src
    assert "from app.agent.conversation_resolver import INDEXED_SYSTEM_CHANNELS" in src
    assert "WHERE channel IN ({_system_channel_sql})" in src
    # …and edited in place it would never reach a tenant that already has the
    # index: both DROP-less forms skip on the index NAME.
    assert "DROP INDEX IF EXISTS ix_conversations_system_channel_per_day" in src
    assert "CREATE UNIQUE INDEX ix_conversations_system_channel_per_day" in src

    covered = set(INDEXED_SYSTEM_CHANNELS)
    assert "admin" in covered, "admin dispatch's day-dedup race recovery is dead"
    assert covered <= SYSTEM_CHANNELS, (
        f"indexed but not a system channel: {covered - SYSTEM_CHANNELS}"
    )
    # `subagent` is the ONE documented exception (sub-agents finish
    # concurrently, so live tenants already hold duplicate rows and CREATE
    # UNIQUE INDEX would fail on them, taking every other channel's
    # constraint down with it). Any NEW gap is a bug, and fails here.
    assert SYSTEM_CHANNELS - covered == {"subagent"}, (
        f"system channels with no index behind them: {SYSTEM_CHANNELS - covered}"
    )


def test_dispatch_tables_are_platform_only():
    from app.db.models.base import (
        AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES, SHARED_TABLES,
    )

    for t in ("admin_dispatches", "admin_dispatch_targets", "admin_thread_messages"):
        assert t in PLATFORM_ONLY_TABLES, f"{t} must be PLATFORM_ONLY"
        assert t not in AGENT_ONLY_TABLES
        assert t not in SHARED_TABLES


@pytest.mark.asyncio
async def test_a_delivered_once_card_records_a_deferred_retract_when_the_agent_is_down(
    dispatch_client,
):
    """"No agent" and "the agent is unreachable right now" are different facts,
    and only one of them means there is nothing to do.

    `agent_proxy_info` returns None for BOTH — it requires
    `deploy_status == 'active'`, so a container that is redeploying or wedged
    fails that test while its `messages` row sits there intact. The route used
    to return silently on either, justified by "there is no tenant row to
    delete", which is false in the second case: the card IS on screen, the
    receipt has just been committed, and the `once` notice therefore stays
    visible permanently. That is B6, reintroduced through the recovery path
    rather than the delivery one.

    The receipt must still stand — an unreachable agent must not cost the user
    their read, or the notice is re-served forever — so what is asserted is
    that read_at is set AND the un-retracted card is recorded rather than
    forgotten.
    """
    client, app = dispatch_client
    recipient = await _mk_user()
    # NOT 'active': the container exists and holds the row, it simply cannot
    # be reached this second.
    await _mk_agent(recipient, url="https://tenant-down.example",
                    deploy_status="deploying")
    app.dependency_overrides[get_current_user] = lambda: _principal(recipient)

    dispatch_id = await _seed_delivered("once", recipient)
    assert (await client.post(f"/api/notices/{dispatch_id}/read")).status_code == 204

    async with async_session_maker() as db:
        row = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()

    assert row.read_at is not None, (
        "the receipt must survive an unreachable agent, or the notice is "
        "re-served to the user forever"
    )
    assert row.chat_status == "delivered", (
        "nothing confirmed the delete, so the card must NOT be recorded as "
        "retracted"
    )
    assert row.last_error and "retract" in row.last_error.lower(), (
        "a delivered card that could not be retracted was forgotten silently "
        f"— last_error={row.last_error!r}"
    )


@pytest.mark.asyncio
async def test_a_never_delivered_card_records_nothing_when_the_agent_is_down(
    dispatch_client,
):
    """The companion, and the reason the branch tests `chat_status` rather than
    just "was the proxy missing".

    A recipient whose container was down at FAN-OUT time carries `no_agent` and
    has no row anywhere, so there is genuinely nothing to retract and nothing
    to report. Without this, "record an error whenever the proxy is missing"
    would pass the test above while filling the operator's panel with failures
    for recipients who never had a card.
    """
    client, app = dispatch_client
    recipient = await _mk_user()
    await _mk_agent(recipient, url="https://tenant-down.example",
                    deploy_status="deploying")
    app.dependency_overrides[get_current_user] = lambda: _principal(recipient)

    dispatch_id = await _seed_delivered("once", recipient, chat_status="no_agent")
    assert (await client.post(f"/api/notices/{dispatch_id}/read")).status_code == 204

    async with async_session_maker() as db:
        row = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()

    assert row.read_at is not None
    assert row.last_error is None, (
        "nothing was ever written for this recipient, so there is nothing to "
        f"report — last_error={row.last_error!r}"
    )
