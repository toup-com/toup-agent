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
from sqlalchemy import func, select, update as sa_update

from app.api.admin.dispatch import DELETED_BODY, router as dispatch_router
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


# ── RUN_MODE precondition ─────────────────────────────────────────
#
# This suite is the PLATFORM half, and CI runs it under the job-level
# RUN_MODE=platform (.github/workflows/test-backend.yml). Under `monolith` —
# which is the DEFAULT, and therefore what a local `pytest` gets — the whole
# retract path silently changes shape: `tenant_proxy.serving_locally()` is true
# for monolith, so `agent_proxy_info` returns None for every user and every
# test that proxies to a tenant fails.
#
# Those failures look exactly like real defects. On 2026-08-15 four of them
# were reported as "main is red, verified by running it", used as a quality
# baseline across five PRs, and cited as evidence for a defect that the tests
# were not in fact demonstrating. CI was green throughout and was correct.
#
# So this is a hard failure, not a skip: a skip would let the same local run
# report "29 passed" and still say nothing about the routes that matter.
def _require_platform_run_mode() -> None:
    from app.config import settings
    mode = (settings.run_mode or "").strip().lower()
    if mode != "platform":
        raise RuntimeError(
            f"test_admin_dispatch.py needs RUN_MODE=platform (got {mode!r}).\n"
            "\n"
            "Under 'monolith' or 'agent', tenant_proxy.serving_locally() is true, so\n"
            "agent_proxy_info() returns None for every user and the retract/read-ack\n"
            "tests fail for a reason that has nothing to do with the code under test.\n"
            "\n"
            "    RUN_MODE=platform pytest tests/test_admin_dispatch.py\n"
            "\n"
            "This is the exact invocation CI uses (test-backend.yml, 'Admin Dispatch\n"
            "suite — platform half')."
        )


_require_platform_run_mode()


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
    # Exact, not a subset — see this test's docstring. `broadcast_enabled` is
    # part of the shape now: the compose form must read the rail off the SERVER
    # rather than carrying its own copy, which would say "disabled" while the
    # API happily accepted the send.
    assert res.json() == {"dispatches": [], "broadcast_enabled": False}


# ── 2. The broadcast confirmation gate ────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_requires_the_typed_confirmation(dispatch_client, monkeypatch):
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    # G-DISPATCH-BROADCAST on, because this test is about the CONFIRMATION WORD
    # and the flag is checked first. The rail's own default-off behaviour is
    # covered by test_broadcast_is_refused_by_default_even_with_the_right_word;
    # leaving it off here would silently turn this into a second copy of that
    # test while its name went on promising something else.
    monkeypatch.setattr(settings, "dispatch_broadcast_enabled", True, raising=False)

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
            # The operator's chosen notification sound, as a catalogue id.
            # Deliberately NOT under the key `sound` — that one marks a row
            # ALARM-CLASS in live_activity_service and would re-ring this
            # announcement three times (test_dispatch_tone.py pins it).
            "tone": "default",
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


# ── R3: an operator's message costs the recipient nothing ────────────


@pytest.mark.asyncio
async def test_a_full_dispatch_moves_no_credit_and_writes_no_ledger_row(
    dispatch_client, monkeypatch,
):
    """R3, asserted rather than assumed.

    An operator's message is not the user's spend. It runs no agent turn —
    `_agent_hop` POSTs to `/internal/admin-notice`, which writes one row and
    broadcasts — so there is no inference to bill and nothing to meter. That
    is TRUE TODAY BY CONSTRUCTION, which is exactly why it needs a test: the
    property is invisible in the code (it is the absence of a call), so the
    first person to route a dispatch through anything that thinks would break
    it with nothing turning red.

    Every credit surface is snapshotted around a complete delivery: the live
    balance in all four of its buckets, the daily counter, and the ledger row
    count. Nothing may move.
    """
    from decimal import Decimal
    from app.db.models import CreditBalance, CreditLedger

    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    recipient = await _mk_user()
    await _mk_agent(recipient, url="https://tenant-r3.example")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    spy = _AgentSpy()
    _patch_agent_http(monkeypatch, spy)

    now = datetime.utcnow()
    async with async_session_maker() as db:
        db.add(CreditBalance(
            user_id=recipient, plan_id="free",
            message_credits_remaining=Decimal("100.00"),
            integration_credits_remaining=Decimal("50.00"),
            message_credits_used_today=Decimal("7.00"),
            purchased_credits_remaining=Decimal("25.00"),
            day_anchor_local_date=now.date().isoformat(),
            period_start=now, period_end=now + timedelta(days=30),
        ))
        await db.commit()

    async def _snapshot():
        async with async_session_maker() as db:
            bal = await db.get(CreditBalance, recipient)
            ledger = (await db.execute(
                select(func.count()).select_from(CreditLedger)
                .where(CreditLedger.user_id == recipient)
            )).scalar_one()
            return (
                bal.message_credits_remaining,
                bal.integration_credits_remaining,
                bal.message_credits_used_today,
                bal.purchased_credits_remaining,
                int(ledger or 0),
            )

    before = await _snapshot()

    created = await client.post("/api/admin/dispatch", json={
        "mode": "once", "audience": "user", "target_user_id": recipient,
        "title": "Scheduled maintenance", "body": "We are moving some things.",
    })
    assert created.status_code == 201, created.text
    dispatch_id = created.json()["dispatch"]["id"]

    from app.services import admin_dispatch_worker as w
    summary = await w.run_dispatch_fanout(dispatch_id)
    assert summary["status"] == "sent", summary
    # Precondition: this must be a REAL delivery, or "nothing was billed" is
    # only true because nothing happened.
    assert summary["delivered_count"] == 1, summary
    assert spy.calls, "the agent hop never ran — the zero-cost claim would be vacuous"

    # And the read, which is the other half of the lifecycle.
    app.dependency_overrides[get_current_user] = lambda: _principal(recipient)
    assert (await client.post(f"/api/notices/{dispatch_id}/read")).status_code == 204

    after = await _snapshot()
    assert after == before, (
        "an operator's message charged the recipient. "
        f"before={before} after={after} — R3 says a dispatch is free to the "
        "user in every bucket, and that the delivery path runs no inference."
    )


def test_the_delivery_path_calls_nothing_that_thinks():
    """The structural half of R3, and the one that survives a refactor.

    The test above proves no credit MOVED. This proves there is nothing on the
    path that could move one: the fan-out module must not reach for an LLM, a
    credit debit, or a turn runner. A future edit that routes a dispatch
    through the agent would still pass the balance assertion whenever the model
    happened to be stubbed out — this one fails on the import.
    """
    src = Path(__file__).resolve().parents[1] / "app" / "services" / "admin_dispatch_worker.py"
    text = src.read_text()
    tree = ast.parse(text)

    reached: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            reached.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            reached.add(node.module)

    forbidden = (
        "app.agent.orchestrator", "app.agent.turn", "app.services.llm",
        "app.services.credit", "app.services.credits", "app.llm",
        "anthropic", "openai",
    )
    hits = sorted(m for m in reached if any(m.startswith(f) for f in forbidden))
    assert not hits, (
        f"the admin-dispatch fan-out imports {hits}. R3: an operator's message "
        "must trigger no inference and consume no credits — the delivery path "
        "writes a row and sends a notification, and nothing on it may think."
    )


async def _mk_dispatch(dispatch_id: str, **overrides) -> None:
    """A dispatch row with the cached counters the fan-out leaves at 0 until
    `_reconcile` runs — which is exactly the state B2a is about."""
    fields = dict(
        id=dispatch_id, created_by_user_id=None, mode="once", audience="user",
        sender_name="Toup", title="t", body="b", urgent=False,
        status="sending", created_at=datetime.utcnow(),
    )
    fields.update(overrides)
    async with async_session_maker() as db:
        db.add(AdminDispatch(**fields))
        await db.commit()


async def _mk_target(dispatch_id: str, user_id: str, **overrides) -> str:
    target_id = str(uuid.uuid4())
    fields = dict(
        id=target_id, dispatch_id=dispatch_id, user_id=user_id,
        state="done", chat_status="delivered", attempts=1,
        created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
    )
    fields.update(overrides)
    async with async_session_maker() as db:
        db.add(AdminDispatchTarget(**fields))
        await db.commit()
    return target_id


@pytest.mark.asyncio
async def test_sent_list_row_carries_one_vintage_not_two(dispatch_client):
    """The list recomputed read_count from the ledger but took target_count and
    delivered_count from the parent's cached columns, which only `_reconcile`
    writes. So a dispatch that had already delivered rendered a live Read
    beside a stale 0/0 — and an operator reads 0/0 as "nothing was sent".

    The list and the detail route describe the same dispatch, so they must
    agree field for field."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    recipient = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    dispatch_id = str(uuid.uuid4())
    await _mk_dispatch(dispatch_id, target_user_id=recipient)
    await _mk_target(
        dispatch_id, recipient,
        notification_id=str(uuid.uuid4()), read_at=datetime.utcnow(),
    )

    listed = await client.get("/api/admin/dispatch")
    assert listed.status_code == 200, listed.text
    row = next(d for d in listed.json()["dispatches"] if d["id"] == dispatch_id)

    detail = await client.get(f"/api/admin/dispatch/{dispatch_id}")
    assert detail.status_code == 200, detail.text
    one = detail.json()["dispatch"]

    # The parent's cached columns are still 0 — that is the whole premise.
    async with async_session_maker() as db:
        parent = await db.get(AdminDispatch, dispatch_id)
        assert parent.target_count == 0 and parent.delivered_count == 0

    assert row["target_count"] == 1, (
        f"the list must recompute target_count from the ledger, got {row}"
    )
    assert row["delivered_count"] == 1, row
    for key in ("target_count", "delivered_count", "chat_delivered_count",
                "no_agent_count", "read_count", "failed_count"):
        assert row[key] == one[key], (
            f"list and detail disagree on {key}: {row[key]} vs {one[key]}"
        )


@pytest.mark.asyncio
async def test_the_list_mirrors_the_delivered_predicate_exactly(dispatch_client):
    """`delivered` is NOT a plain chat_status test — `notification_id` is
    non-null whether or not the agent hop after it succeeded. A FAILED target
    is never also delivered, or a broadcast that reached hundreds of banners
    and no chats reports itself fully delivered."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    recipient = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    dispatch_id = str(uuid.uuid4())
    await _mk_dispatch(dispatch_id, target_user_id=recipient)
    # Notified, then the agent hop failed: the banner row exists, the send did not.
    await _mk_target(
        dispatch_id, recipient, state="failed", chat_status="failed",
        notification_id=str(uuid.uuid4()), last_error="agent 503",
    )

    row = next(
        d for d in (await client.get("/api/admin/dispatch")).json()["dispatches"]
        if d["id"] == dispatch_id
    )
    one = (await client.get(f"/api/admin/dispatch/{dispatch_id}")).json()["dispatch"]

    assert row["delivered_count"] == 0, (
        f"a failed target is never delivered, got {row}"
    )
    assert row["failed_count"] == 1, row
    assert row["target_count"] == 1, row
    assert (row["delivered_count"], row["failed_count"]) == (
        one["delivered_count"], one["failed_count"]
    )


@pytest.mark.asyncio
async def test_a_dispatch_with_no_targets_still_reads_zero(dispatch_client):
    """The GROUP BY returns no row for a dispatch that has no targets, so the
    fallback to the stored columns has to stay — this is the honest 0/0."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    dispatch_id = str(uuid.uuid4())
    await _mk_dispatch(dispatch_id, audience="all", status="queued")

    row = next(
        d for d in (await client.get("/api/admin/dispatch")).json()["dispatches"]
        if d["id"] == dispatch_id
    )
    assert (row["target_count"], row["delivered_count"], row["read_count"]) == (0, 0, 0)


@pytest.mark.asyncio
async def test_a_stalled_dispatch_is_swept_to_failed_with_a_visible_reason(
    dispatch_client,
):
    """`run_dispatch_fanout` writes `sending` at the top and only its own
    `except` writes `failed`. A worker that is OOM-killed or redeployed runs
    NEITHER — and it typically dies before materialising a single target, so
    there is no per-target `last_error` to explain it either. The row read
    `sending` forever and the panel told the operator a message was on its way
    that nothing was carrying."""
    from app.services import admin_dispatch_worker as w

    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    now = datetime.utcnow()
    dispatch_id = str(uuid.uuid4())
    await _mk_dispatch(
        dispatch_id, audience="all", status="sending",
        created_at=now - w._STALL_MAX_AGE - timedelta(minutes=1),
    )

    async with async_session_maker() as db:
        assert await w.sweep_stalled_dispatches(db, now) == 1

    async with async_session_maker() as db:
        row = await db.get(AdminDispatch, dispatch_id)
        assert row.status == "failed", "a stall must reach a TERMINAL state"
        assert row.completed_at is not None
        assert row.last_error, "a swept dispatch with no reason is still a mystery"
        # The remedy has to be in the string: a stalled dispatch is fully
        # recoverable and nothing else on the row says so.
        assert "Retry" in row.last_error, row.last_error

    # …and the panel can actually read it, on both routes.
    listed = next(
        d for d in (await client.get("/api/admin/dispatch")).json()["dispatches"]
        if d["id"] == dispatch_id
    )
    detail = (await client.get(f"/api/admin/dispatch/{dispatch_id}")).json()["dispatch"]
    assert listed["last_error"] == detail["last_error"] == row.last_error
    assert listed["status"] == "failed"


@pytest.mark.asyncio
async def test_the_sweep_spares_a_young_dispatch_and_a_moving_one():
    """Two ways to be alive. The age test alone would sweep a broadcast still
    walking its targets, and — because `created_at` is the parent row's only
    clock — a RETRIED old dispatch the instant its new fan-out set `sending`.
    Progress on any target is what holds a dispatch out of the sweep."""
    from app.services import admin_dispatch_worker as w

    now = datetime.utcnow()
    user_id = await _mk_user()

    young = str(uuid.uuid4())
    await _mk_dispatch(young, status="sending", created_at=now - timedelta(minutes=1))

    # Created long ago — a retry of an old dispatch looks exactly like this —
    # but a target moved a moment ago, so the fan-out is demonstrably alive.
    moving = str(uuid.uuid4())
    await _mk_dispatch(
        moving, status="sending",
        created_at=now - w._STALL_MAX_AGE - timedelta(hours=4),
    )
    await _mk_target(
        moving, user_id, state="sending", chat_status="pending",
        updated_at=now - timedelta(seconds=5),
    )

    async with async_session_maker() as db:
        assert await w.sweep_stalled_dispatches(db, now) == 0

    async with async_session_maker() as db:
        for did in (young, moving):
            row = await db.get(AdminDispatch, did)
            assert row.status == "sending", f"{did} was swept while alive"
            assert row.last_error is None


@pytest.mark.asyncio
async def test_the_sweep_is_a_status_cas_and_leaves_terminal_rows_alone():
    """Both Railway replicas run this with no leader election, so the UPDATE's
    `WHERE status='sending'` is the whole defence: the loser sees rowcount=0.
    A dispatch that already settled must never be reopened."""
    from app.services import admin_dispatch_worker as w

    now = datetime.utcnow()
    old = now - w._STALL_MAX_AGE - timedelta(minutes=1)

    stalled = str(uuid.uuid4())
    await _mk_dispatch(stalled, status="sending", created_at=old)
    settled = str(uuid.uuid4())
    await _mk_dispatch(settled, status="sent", created_at=old)

    async def _sweep():
        async with async_session_maker() as db:
            return await w.sweep_stalled_dispatches(db, now)

    a, b = await asyncio.gather(_sweep(), _sweep())
    assert sorted((a, b)) == [0, 1], (
        f"exactly one replica may claim a stalled row, got {(a, b)}"
    )

    async with async_session_maker() as db:
        assert (await db.get(AdminDispatch, stalled)).status == "failed"
        done = await db.get(AdminDispatch, settled)
        assert done.status == "sent" and done.last_error is None


@pytest.mark.asyncio
async def test_the_stall_sweep_runs_on_the_notification_tick(monkeypatch):
    """The sweep is only a fix if something calls it. It rides the notification
    dispatch loop — the platform's one periodic task that provably runs
    (APScheduler was found dead on this deployment on 2026-07-10, which is why
    that loop exists at all) — so this drives the real tick, not the sweep."""
    from app.services import admin_dispatch_worker as w
    from app.services.notification_dispatcher import run_notification_dispatch

    monkeypatch.setattr(settings, "notification_dispatch_enabled", True)

    dispatch_id = str(uuid.uuid4())
    await _mk_dispatch(
        dispatch_id, status="sending",
        created_at=datetime.utcnow() - w._STALL_MAX_AGE - timedelta(minutes=1),
    )

    await run_notification_dispatch()

    async with async_session_maker() as db:
        row = await db.get(AdminDispatch, dispatch_id)
        assert row.status == "failed", (
            "nothing sweeps automatically — the tick must call the sweep"
        )
        assert row.last_error


@pytest.mark.asyncio
async def test_a_re_run_fanout_clears_the_stall_reason(dispatch_client, monkeypatch):
    """`last_error` is this attempt's story. A retry that inherited the
    sweeper's reason would report a dead worker on a dispatch that has just
    delivered — the same one-row-two-vintages defect as B2a, in prose."""
    from app.services import admin_dispatch_worker as w

    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    recipient = await _mk_user()
    await _mk_agent(recipient, url="https://tenant-9.example")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    _patch_agent_http(monkeypatch, _AgentSpy())

    now = datetime.utcnow()
    dispatch_id = str(uuid.uuid4())
    await _mk_dispatch(
        dispatch_id, audience="user", target_user_id=recipient, status="sending",
        created_at=now - w._STALL_MAX_AGE - timedelta(minutes=1),
    )

    async with async_session_maker() as db:
        assert await w.sweep_stalled_dispatches(db, now) == 1

    summary = await w.run_dispatch_fanout(dispatch_id)
    assert summary["status"] == "sent", summary

    async with async_session_maker() as db:
        row = await db.get(AdminDispatch, dispatch_id)
        assert row.last_error is None, (
            f"a delivered dispatch still blames a dead worker: {row.last_error}"
        )

    detail = (await client.get(f"/api/admin/dispatch/{dispatch_id}")).json()["dispatch"]
    assert detail["last_error"] is None and detail["status"] == "sent"


@pytest.mark.asyncio
async def test_reconcile_clears_a_stall_reason_left_on_a_fanout_that_was_alive():
    """The sweep can be WRONG, and this is what happens next.

    Its clock cannot see the window before the first target exists, so a
    genuinely-alive fan-out can be declared dead — and that worker then
    finishes normally and lands in `_reconcile`, not in the re-run path.
    `run_dispatch_fanout` clears `last_error` at its top, so the existing
    re-run test passes while this ordering stays broken: the row settles to
    `sent`, every counter correct, still carrying "the worker is gone, press
    Retry" for a delivery that completed.

    Whoever writes the outcome owns every field that describes it.
    """
    from app.services import admin_dispatch_worker as w

    now = datetime.utcnow()
    recipient = await _mk_user()
    dispatch_id = str(uuid.uuid4())
    await _mk_dispatch(
        dispatch_id, audience="user", target_user_id=recipient, status="sending",
        created_at=now - w._STALL_MAX_AGE - timedelta(minutes=1),
    )
    async with async_session_maker() as db:
        assert await w.sweep_stalled_dispatches(db, now) == 1
        row = await db.get(AdminDispatch, dispatch_id)
        assert row.last_error, "precondition: the sweep must have stamped a reason"

    # The still-live worker delivers its one target and reconciles. NOT
    # run_dispatch_fanout — that clears at the top and would hide the defect.
    await _mk_target(dispatch_id, recipient, state="done", chat_status="delivered")
    async with async_session_maker() as db:
        summary = await w._reconcile(db, dispatch_id)
    assert summary["status"] == "sent", summary

    async with async_session_maker() as db:
        row = await db.get(AdminDispatch, dispatch_id)
        assert row.last_error is None, (
            f"a delivered dispatch still blames a dead worker: {row.last_error}"
        )


@pytest.mark.asyncio
async def test_the_sweep_also_terminates_a_dispatch_that_died_while_queued():
    """B2b was only half-fixed by sweeping `sending`.

    Both producers commit `status='queued'` and THEN await
    `spawn_dispatch_fanout`, which is a bare `asyncio.create_task` in the API
    process. A redeploy in that window — the exact scenario the sweep exists
    for — or a throw inside `run_dispatch_fanout` before its status UPDATE
    lands, leaves the row `queued` forever with nobody walking it.

    Sweeping only `sending` fixes the second half and leaves the first, which
    is worse than not fixing it: the panel looks like it now reports stalls.
    """
    from app.services import admin_dispatch_worker as w

    now = datetime.utcnow()
    recipient = await _mk_user()
    dispatch_id = str(uuid.uuid4())
    await _mk_dispatch(
        dispatch_id, audience="user", target_user_id=recipient, status="queued",
        created_at=now - w._STALL_MAX_AGE - timedelta(minutes=1),
    )

    async with async_session_maker() as db:
        assert await w.sweep_stalled_dispatches(db, now) == 1
        row = await db.get(AdminDispatch, dispatch_id)
        assert row.status == "failed"
        assert row.completed_at is not None
        assert "Retry" in (row.last_error or ""), row.last_error


@pytest.mark.asyncio
async def test_the_stall_sweep_survives_the_notification_kill_switch(monkeypatch):
    """The sweep must NOT share `notification_dispatch_enabled`.

    It first sat below that early return, justified by "with notification
    dispatch off an announcement reaches nobody anyway". That is false: the
    fan-out writes the chat card itself over HTTP (`_agent_hop`) and the
    persistent thread row itself (`_ensure_thread_row`), and neither touches
    this queue. So with the switch off, dispatches keep delivering and keep
    stalling — and the one thing that reports a dead fan-out would be the one
    thing switched off. A kill switch for SENDING is not a kill switch for
    BOOKKEEPING.
    """
    from app.services import notification_dispatcher as nd

    calls: list = []

    async def _spy(db, now):
        calls.append(now)
        return 0

    monkeypatch.setattr(
        nd.admin_dispatch_worker, "sweep_stalled_dispatches", _spy,
    )
    monkeypatch.setattr(nd.settings, "notification_dispatch_enabled", False)

    out = await nd.run_notification_dispatch()
    assert out == {"claimed": 0}, out
    assert calls, (
        "the stall sweep did not run with notification dispatch disabled — "
        "a dispatch that stalls while the switch is off can never be reported"
    )


def _stub_agent_resolution(monkeypatch, *, url="https://agent.example", key="tk-test"):
    """Make tenant RESOLUTION succeed, so a revoke test measures revoke.

    `agent_proxy_info` answers None in this harness — the same seam whose
    ambiguity (`None` means both "no agent" and "agent redeploying") is the
    defect #637 fixes, and the reason two read-receipt tests are red on main
    today. Leaving it unstubbed here would make every revoke assertion pass or
    fail for a reason that has nothing to do with revoking.
    """
    async def _info(user_id, db):
        return (url, key)
    monkeypatch.setattr("app.api.admin.dispatch.agent_proxy_info", _info)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["once", "persistent"])
async def test_revoke_pulls_the_card_for_either_mode(dispatch_client, monkeypatch, mode):
    """`once` already retracted itself on read; recall must work for BOTH.

    A persistent dispatch is the one an operator most needs back — it is the
    mode that STAYS in the chat — and it is exactly the mode the read path
    never retracts.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    user_id = await _mk_user()
    await _mk_agent(user_id)
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    dispatch_id = await _seed_delivered(mode, user_id)

    proxied: list[dict] = []

    async def _fake_proxy(agent_url, agent_api_key, path, method="GET", **kw):
        proxied.append({"path": path, "method": method, "json": kw.get("json_body")})
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.admin.dispatch.proxy_to_agent", _fake_proxy)
    _stub_agent_resolution(monkeypatch)

    res = await client.post(f"/api/admin/dispatch/{dispatch_id}/revoke")
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["revoked"] == 1 and payload["failed"] == 0, payload

    # The SAME tenant route the read path uses — no second pipeline.
    assert len(proxied) == 1, proxied
    assert proxied[0]["path"] == "internal/admin-notice/retract"
    assert proxied[0]["json"] == {"user_id": user_id, "dispatch_id": dispatch_id}

    async with async_session_maker() as db:
        d = await db.get(AdminDispatch, dispatch_id)
        assert d.revoked_at is not None
        assert d.revoked_by_user_id == admin, "a recall records WHO, not just that"
        t = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()
        assert t.chat_status == "retracted"


@pytest.mark.asyncio
async def test_a_revoke_that_cannot_reach_an_agent_is_not_counted_as_one(
    dispatch_client, monkeypatch,
):
    """The card is still sitting in that tenant's DB, so this is not a recall.

    The whole value of the number this route returns is that an operator can
    tell "930 of 1000" from "done". Counting an unreachable tenant as revoked
    would make the reassuring answer the wrong one, which is the one failure
    mode a recall cannot afford.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    user_id = await _mk_user()
    await _mk_agent(user_id)
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    dispatch_id = await _seed_delivered("persistent", user_id)

    async def _boom(*a, **kw):
        raise RuntimeError("tenant down")

    monkeypatch.setattr("app.api.admin.dispatch.proxy_to_agent", _boom)
    _stub_agent_resolution(monkeypatch)

    res = await client.post(f"/api/admin/dispatch/{dispatch_id}/revoke")
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["revoked"] == 0 and payload["failed"] == 1

    async with async_session_maker() as db:
        t = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()
        assert t.chat_status == "delivered", "the card is still there; do not claim otherwise"
        assert t.last_error and "revoke" in t.last_error

        # ...but the operator's INTENT is recorded regardless, so a retry
        # after the tenant recovers is a retry, not a fresh decision.
        d = await db.get(AdminDispatch, dispatch_id)
        assert d.revoked_at is not None


@pytest.mark.asyncio
async def test_revoke_never_touches_the_notification(dispatch_client, monkeypatch):
    """R-of-record: a recall pulls the CARD. Nothing un-sends a push.

    If someone later wires this route into the notification queue, this fails —
    which is the point. Promising an operator that a lock-screen banner can be
    taken back is a worse lie than the one this feature exists to fix.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    user_id = await _mk_user()
    await _mk_agent(user_id)
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    dispatch_id = await _seed_delivered("once", user_id)

    async def _fake_proxy(*a, **kw):
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.admin.dispatch.proxy_to_agent", _fake_proxy)
    _stub_agent_resolution(monkeypatch)

    async with async_session_maker() as db:
        before = (await db.execute(
            select(func.count()).select_from(NotificationQueue)
        )).scalar_one()

    res = await client.post(f"/api/admin/dispatch/{dispatch_id}/revoke")
    assert res.status_code == 200, res.text

    async with async_session_maker() as db:
        after = (await db.execute(
            select(func.count()).select_from(NotificationQueue)
        )).scalar_one()
    assert after == before, "a recall must not write, cancel or amend a notification row"


# ── 15. Deleting from the thread ──────────────────────────────────
#
# The operator could send into a thread and never take anything back. These
# pin the two scopes, and — more importantly — that the OPERATOR keeps the
# record while the USER stops seeing it.


async def _mk_thread_msg(user_id: str, direction: str, body: str) -> str:
    mid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(AdminThreadMessage(
            id=mid, user_id=user_id, dispatch_id=None, direction=direction,
            body=body, author_admin_id=None, sender_name="Toup",
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return mid


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["user_side", "everyone"])
async def test_a_deleted_message_leaves_the_users_thread(dispatch_client, scope):
    """Either scope removes it from what the USER can see."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    user_id = await _mk_user()
    mid = await _mk_thread_msg(user_id, "out", "sent by mistake")

    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    res = await client.delete(
        f"/api/admin/dispatch/threads/{user_id}/messages/{mid}?scope={scope}")
    assert res.status_code == 200, res.text

    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    thread = await client.get("/api/notices/thread")
    assert thread.status_code == 200, thread.text
    assert [m["id"] for m in thread.json()["messages"]] == []

    # ...and it stops badging. A thread badged for content the user cannot
    # find is worse than the message itself.
    state = await client.get("/api/notices/state")
    assert state.json()["thread_unread"] == 0
    assert state.json()["has_thread"] is False


@pytest.mark.asyncio
async def test_theirs_keeps_the_operators_copy_and_both_keeps_the_turn(dispatch_client):
    """The asymmetry IS the feature.

    `user_side` unsends the words and keeps the record — the state an operator
    needs most at exactly the moment they are undoing something. `everyone` clears
    the body but keeps the row, so the reply that follows it still has
    something to follow.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    user_id = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    a = await _mk_thread_msg(user_id, "out", "hide from them")
    b = await _mk_thread_msg(user_id, "out", "gone for everyone")

    await client.delete(f"/api/admin/dispatch/threads/{user_id}/messages/{a}?scope=user_side")
    await client.delete(f"/api/admin/dispatch/threads/{user_id}/messages/{b}?scope=everyone")

    res = await client.get(f"/api/admin/dispatch/threads/{user_id}")
    assert res.status_code == 200, res.text
    by_id = {m["id"]: m for m in res.json()["messages"]}

    assert by_id[a]["body"] == "hide from them", "theirs must not destroy the operator's copy"
    assert by_id[a]["hidden_from_user_at"] and not by_id[a]["deleted_at"]

    assert by_id[b]["body"] == "This message was deleted.", "both clears the words"
    assert by_id[b]["deleted_at"]

    async with async_session_maker() as db:
        row = await db.get(AdminThreadMessage, b)
        assert row is not None, "both must leave a tombstone, never a hard delete"
        assert row.deleted_by_user_id == admin


@pytest.mark.asyncio
async def test_an_operator_may_delete_a_users_own_reply(dispatch_client):
    """Someone pastes a password into a support thread. The asymmetry of
    'operators may only delete their own words' would be arbitrary here."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    user_id = await _mk_user()
    mid = await _mk_thread_msg(user_id, "in", "my password is hunter2")

    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    res = await client.delete(
        f"/api/admin/dispatch/threads/{user_id}/messages/{mid}?scope=everyone")
    assert res.status_code == 200, res.text

    async with async_session_maker() as db:
        row = await db.get(AdminThreadMessage, mid)
        assert row.body == "This message was deleted."


@pytest.mark.asyncio
async def test_a_message_from_another_users_thread_is_404(dispatch_client):
    """Scoped to the thread, not just the id. The route reads as "this user's
    thread" and a mismatch must not silently reach into someone else's."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    owner = await _mk_user()
    stranger = await _mk_user()
    mid = await _mk_thread_msg(owner, "out", "not yours")

    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    res = await client.delete(
        f"/api/admin/dispatch/threads/{stranger}/messages/{mid}?scope=everyone")
    assert res.status_code == 404, res.text

    async with async_session_maker() as db:
        assert (await db.get(AdminThreadMessage, mid)).body == "not yours"


@pytest.mark.asyncio
async def test_deleting_twice_does_not_overwrite_the_first_record(dispatch_client):
    """Idempotent. A second press must not re-stamp who did it, nor replace
    the tombstone with a tombstone of the tombstone."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    other = await _mk_user(role="admin")
    user_id = await _mk_user()
    mid = await _mk_thread_msg(user_id, "out", "x")

    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    await client.delete(f"/api/admin/dispatch/threads/{user_id}/messages/{mid}?scope=everyone")
    async with async_session_maker() as db:
        first = await db.get(AdminThreadMessage, mid)
        stamp, who, body = first.deleted_at, first.deleted_by_user_id, first.body

    app.dependency_overrides[get_current_user] = lambda: _principal(other, role="admin")
    res = await client.delete(
        f"/api/admin/dispatch/threads/{user_id}/messages/{mid}?scope=everyone")
    assert res.status_code == 200, res.text

    async with async_session_maker() as db:
        again = await db.get(AdminThreadMessage, mid)
        assert (again.deleted_at, again.deleted_by_user_id, again.body) == (stamp, who, body)


@pytest.mark.asyncio
async def test_broadcast_is_refused_by_default_even_with_the_right_word(
    dispatch_client, monkeypatch,
):
    """The default deployment may not broadcast. No flag set, correct word."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    spawned: list[str] = []

    async def _fake_spawn(dispatch_id: str):
        spawned.append(dispatch_id)

    monkeypatch.setattr("app.api.admin.dispatch.spawn_dispatch_fanout", _fake_spawn)

    res = await client.post(
        "/api/admin/dispatch", json=_compose(audience="all", confirm="BROADCAST"),
    )
    assert res.status_code == 403, res.text
    assert not spawned, "a flag-disabled broadcast must not reach the fan-out"
    async with async_session_maker() as db:
        assert (await db.execute(select(AdminDispatch))).scalars().all() == [], \
            "nothing may be persisted by a refused broadcast"

    # The refusal must not hand the caller a way through. An operator who has
    # just been stopped reads "send confirm='BROADCAST' to proceed" as the way
    # past the thing that stopped them, and here it is not — the word does not
    # unlock this. Asserted on the INSTRUCTION, not on the bare token: the
    # message legitimately names docs/DISPATCH_BROADCAST.md, and a substring
    # test for "BROADCAST" fails on the filename while proving nothing about
    # the hazard. (It did, which is how this comment came to exist.)
    detail = res.json().get("detail", "")
    assert "confirm=" not in detail and "to proceed" not in detail, \
        f"the 403 must not offer the confirmation word as a workaround: {detail!r}"


@pytest.mark.asyncio
async def test_the_flag_is_checked_before_the_confirmation_word(
    dispatch_client, monkeypatch,
):
    """ORDER, not just presence — the two checks must fire in this sequence.

    Flag off AND no confirmation word: both gates would refuse, so the status
    code is the only thing that says which ran first. It has to be the flag's
    403. With the checks reversed the caller gets the word's 400, whose body
    reads "send confirm='BROADCAST' to proceed" — telling an operator how to
    do the thing this deployment forbids, and sending them round the loop to
    a refusal they could have been given once.

    Written because the reversed ordering passed every other test in this
    section: they all supply the correct word, so the guard fires either way
    and the sequence is invisible to them.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    monkeypatch.setattr(settings, "dispatch_broadcast_enabled", False, raising=False)

    res = await client.post("/api/admin/dispatch", json=_compose(audience="all"))
    assert res.status_code == 403, (
        f"flag must be evaluated before the confirmation word; got {res.status_code}: {res.text}"
    )
    detail = res.json().get("detail", "")
    assert "confirm=" not in detail and "to proceed" not in detail, \
        f"a flag-refused broadcast must not be told how to type past it: {detail!r}"


@pytest.mark.asyncio
async def test_the_flag_is_off_in_the_shipped_defaults():
    """Pins the DEFAULT, not the runtime value.

    The route test above monkeypatches, so it would keep passing if someone
    flipped the default to True and left the code alone. This is the assertion
    that notices.
    """
    from app.config import Settings
    assert Settings.model_fields["dispatch_broadcast_enabled"].default is False


@pytest.mark.asyncio
async def test_a_single_user_send_is_untouched_by_the_broadcast_flag(
    dispatch_client, monkeypatch,
):
    """The rail is about blast radius, not about the feature.

    Guards the obvious over-correction: gating `create_dispatch` as a whole
    would take the one audience this deployment IS meant to use down with it.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    target = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    spawned: list[str] = []

    async def _fake_spawn(dispatch_id: str):
        spawned.append(dispatch_id)

    monkeypatch.setattr("app.api.admin.dispatch.spawn_dispatch_fanout", _fake_spawn)
    monkeypatch.setattr(settings, "dispatch_broadcast_enabled", False, raising=False)

    res = await client.post(
        "/api/admin/dispatch",
        json=_compose(audience="user", target_user_id=target),
    )
    assert res.status_code == 201, res.text
    assert len(spawned) == 1


@pytest.mark.asyncio
async def test_both_read_surfaces_report_the_flag(dispatch_client, monkeypatch):
    """The compose form must not carry its own copy of the flag.

    Two surfaces because they are read at different moments: the list on mount
    (before 'Everyone' can be clicked) and the preview after. A client constant
    that disagrees with the deployment is either a dead button or a broadcast
    the operator believed was impossible.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    for flag in (False, True):
        monkeypatch.setattr(settings, "dispatch_broadcast_enabled", flag, raising=False)

        listed = await client.get("/api/admin/dispatch")
        assert listed.status_code == 200, listed.text
        assert listed.json()["broadcast_enabled"] is flag

        prev = await client.get("/api/admin/dispatch/preview?audience=all")
        assert prev.status_code == 200, prev.text
        assert prev.json()["broadcast_enabled"] is flag


async def _seed_thread(user_id: str, rows: list[tuple[str, str]],
                       *, base: datetime | None = None) -> list[str]:
    """Write (direction, body) pairs into one user's thread, oldest first.

    Timestamps are explicit and one minute apart: `created_at` orders every
    query under test, and rows written in the same tick sort arbitrarily, which
    makes an ordering assertion pass or fail on machine speed.
    """
    from app.db.models import AdminThreadMessage

    start = base or (datetime.utcnow() - timedelta(hours=1))
    ids = []
    async with async_session_maker() as db:
        for i, (direction, body) in enumerate(rows):
            mid = str(uuid.uuid4())
            ids.append(mid)
            db.add(AdminThreadMessage(
                id=mid,
                user_id=user_id,
                direction=direction,
                body=body,
                created_at=start + timedelta(minutes=i),
                sender_name="Toup" if direction == "out" else None,
            ))
        await db.commit()
    return ids


@pytest.mark.asyncio
async def test_reading_a_thread_does_not_mark_it_read(dispatch_client):
    """F6. The console polls the open thread every few seconds; if GET stamped
    `admin_read_at`, a background refresh would clear the operator's reply
    queue with the window blurred and nobody looking. `admin_read_at` is the
    ONLY record that a reply was seen, so there is nothing to recover it from.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _seed_thread(user, [("out", "Are you there?"), ("in", "Yes — what's up?")])

    res = await client.get(f"/api/admin/dispatch/threads/{user}")
    assert res.status_code == 200, res.text
    assert res.json()["unread_in"] == 1

    # The whole point: read it as many times as a poller would.
    for _ in range(3):
        await client.get(f"/api/admin/dispatch/threads/{user}")

    from app.db.models import AdminThreadMessage
    async with async_session_maker() as db:
        still_unread = (await db.execute(
            select(func.count()).select_from(AdminThreadMessage).where(
                AdminThreadMessage.user_id == user,
                AdminThreadMessage.direction == "in",
                AdminThreadMessage.admin_read_at.is_(None),
            )
        )).scalar_one()
    assert still_unread == 1, (
        "GET marked the thread read — a poll would clear the reply queue"
    )


@pytest.mark.asyncio
async def test_the_explicit_read_route_marks_only_inbound_rows(dispatch_client):
    """Counterpart: 'never marks read' would also pass the test above.

    Only `in` rows carry `admin_read_at`. Stamping `out` rows would corrupt the
    USER's unread count, which reads the same column family from the other side.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _seed_thread(user, [("out", "Ping"), ("in", "Pong"), ("in", "Still here")])

    res = await client.post(f"/api/admin/dispatch/threads/{user}/read")
    assert res.status_code == 204, res.text

    from app.db.models import AdminThreadMessage
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AdminThreadMessage.direction, AdminThreadMessage.admin_read_at)
            .where(AdminThreadMessage.user_id == user)
        )).all()

    by_dir = {}
    for direction, stamp in rows:
        by_dir.setdefault(direction, []).append(stamp)
    assert all(s is not None for s in by_dir["in"]), "inbound rows were not marked"
    assert all(s is None for s in by_dir["out"]), (
        "outbound rows were stamped — that column belongs to the user's side"
    )

    assert (await client.get(
        f"/api/admin/dispatch/threads/{user}"
    )).json()["unread_in"] == 0


@pytest.mark.asyncio
async def test_read_route_is_idempotent_and_404s_for_a_stranger(dispatch_client):
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _seed_thread(user, [("in", "hello")])

    assert (await client.post(f"/api/admin/dispatch/threads/{user}/read")).status_code == 204
    assert (await client.post(f"/api/admin/dispatch/threads/{user}/read")).status_code == 204

    res = await client.post(f"/api/admin/dispatch/threads/{uuid.uuid4()}/read")
    assert res.status_code == 404, res.text


@pytest.mark.asyncio
async def test_thread_list_reports_who_spoke_last(dispatch_client):
    """The sidebar prefixes the operator's own words with 'You: '. Without a
    direction on the summary every row reads as if the user said it, which
    inverts the one thing the column is scanned for."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    waiting = await _mk_user()   # user spoke last — needs an answer
    answered = await _mk_user()  # operator spoke last
    await _seed_thread(waiting, [("out", "Hi"), ("in", "I have a question")])
    await _seed_thread(answered, [("in", "A question"), ("out", "Here is the answer")])

    rows = (await client.get("/api/admin/dispatch/threads")).json()["threads"]
    by_id = {r["user_id"]: r for r in rows}

    assert by_id[waiting]["last_direction"] == "in"
    assert by_id[waiting]["last_body"] == "I have a question"
    assert by_id[answered]["last_direction"] == "out"
    assert by_id[answered]["last_body"] == "Here is the answer"


@pytest.mark.asyncio
async def test_thread_search_matches_person_and_content(dispatch_client):
    """An operator remembers the person or the words, never reliably which."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    alice = await _mk_user()
    bob = await _mk_user()
    async with async_session_maker() as db:
        await db.execute(sa_update(User).where(User.id == alice).values(
            name="Alice Nakamura", email="alice@example.com"))
        await db.execute(sa_update(User).where(User.id == bob).values(
            name="Bob Silva", email="bob@example.com"))
        await db.commit()

    await _seed_thread(alice, [("in", "my invoice is wrong")])
    await _seed_thread(bob, [("in", "cannot log in")])

    def ids(payload):
        return {t["user_id"] for t in payload["threads"]}

    by_name = (await client.get("/api/admin/dispatch/threads?query=nakamura")).json()
    assert ids(by_name) == {alice}, "name search missed"

    by_email = (await client.get("/api/admin/dispatch/threads?query=bob@")).json()
    assert ids(by_email) == {bob}, "email search missed"

    by_body = (await client.get("/api/admin/dispatch/threads?query=invoice")).json()
    assert ids(by_body) == {alice}, "message-body search missed"

    assert ids((await client.get(
        "/api/admin/dispatch/threads?query=zzzznomatch")).json()) == set()


@pytest.mark.asyncio
async def test_search_does_not_rewrite_the_conversation_length(dispatch_client):
    """`total` counts the THREAD, not the matches.

    Filtering the rows being grouped is the obvious implementation and it makes
    a two-word search report a 40-message conversation as having 1 message.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _seed_thread(user, [
        ("out", "Welcome aboard"),
        ("in", "thanks"),
        ("in", "one question about billing"),
        ("out", "happy to help"),
    ])

    hit = (await client.get(
        "/api/admin/dispatch/threads?query=billing")).json()["threads"]
    assert len(hit) == 1
    assert hit[0]["total"] == 4, (
        f"search rewrote the thread length to {hit[0]['total']} — it must count "
        "every message, not the matching ones"
    )


@pytest.mark.asyncio
async def test_thread_list_pages_by_keyset(dispatch_client):
    """Cursor paging, and no cursor on a short page — a cursor at the end of
    the list makes the client fetch one empty page per scroll forever."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    base = datetime.utcnow() - timedelta(days=1)
    users = []
    for i in range(5):
        u = await _mk_user()
        users.append(u)
        await _seed_thread(u, [("in", f"message {i}")],
                           base=base + timedelta(hours=i))

    page1 = (await client.get("/api/admin/dispatch/threads?limit=2")).json()
    assert len(page1["threads"]) == 2
    assert page1["next_cursor"] is not None

    page2 = (await client.get(
        f"/api/admin/dispatch/threads?limit=2&cursor={page1['next_cursor']}"
    )).json()
    assert len(page2["threads"]) == 2

    seen = [t["user_id"] for t in page1["threads"] + page2["threads"]]
    assert len(set(seen)) == 4, f"pages overlapped: {seen}"
    # Newest first, and strictly descending across the page boundary.
    stamps = [t["last_message_at"] for t in page1["threads"] + page2["threads"]]
    assert stamps == sorted(stamps, reverse=True), stamps

    last = (await client.get(
        f"/api/admin/dispatch/threads?limit=2&cursor={page2['next_cursor']}"
    )).json()
    assert len(last["threads"]) == 1
    assert last["next_cursor"] is None, "a short page must end the scroll"


@pytest.mark.asyncio
async def test_a_malformed_cursor_is_rejected_not_ignored(dispatch_client):
    """Swallowing it serves page 1 forever, which reads as a list that will not
    scroll rather than as the client bug it is."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    res = await client.get("/api/admin/dispatch/threads?cursor=not-a-date")
    assert res.status_code == 422, res.text


# ── Unit 2: deleting a whole conversation ─────────────────────────
#
# The scopes are the same two a single message has, applied to every row. What
# these defend is the half a naive implementation misses: a persistent dispatch
# leaves a CARD in the user's chat as well as a row in the thread, and clearing
# only the thread leaves that card sitting there with a Reply action pointing
# at a conversation that is no longer there.


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["user_side", "everyone"])
async def test_deleting_a_conversation_clears_it_for_the_user(dispatch_client, scope):
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _seed_thread(user, [("out", "first"), ("in", "reply"), ("out", "second")])

    res = await client.request(
        "DELETE", f"/api/admin/dispatch/threads/{user}?scope={scope}")
    assert res.status_code == 200, res.text
    assert res.json()["messages"] == 3

    # The USER's own read of their thread is empty either way.
    app.dependency_overrides[get_current_user] = lambda: _principal(user, role="user")
    theirs = (await client.get("/api/notices/thread")).json()
    assert theirs["messages"] == [], theirs
    state = (await client.get("/api/notices/state")).json()
    assert state["has_thread"] is False, "the drawer row must go with the thread"


@pytest.mark.asyncio
async def test_user_side_keeps_the_operators_copy_and_everyone_does_not(dispatch_client):
    """The whole reason there are two scopes. `user_side` is an unsend that
    keeps the record — which is what an operator almost always wants, and is
    exactly what they need most at the moment they are undoing something."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    kept = await _mk_user()
    gone = await _mk_user()
    await _seed_thread(kept, [("out", "the operator still sees this")])
    await _seed_thread(gone, [("out", "nobody sees this")])

    await client.request("DELETE", f"/api/admin/dispatch/threads/{kept}?scope=user_side")
    await client.request("DELETE", f"/api/admin/dispatch/threads/{gone}?scope=everyone")

    kept_body = (await client.get(f"/api/admin/dispatch/threads/{kept}")).json()
    assert [m["body"] for m in kept_body["messages"]] == ["the operator still sees this"]
    assert kept_body["retracted_at"] is not None
    assert kept_body["user_visible_total"] == 0

    gone_body = (await client.get(f"/api/admin/dispatch/threads/{gone}")).json()
    assert [m["body"] for m in gone_body["messages"]] == [DELETED_BODY], (
        "an `everyone` delete must clear the words for the operator too"
    )
    # ...but the turn survives, so the conversation does not silently lose one.
    assert len(gone_body["messages"]) == 1


@pytest.mark.asyncio
async def test_deleting_a_conversation_pulls_the_chat_cards_too(dispatch_client, monkeypatch):
    """The defect this exists for: clear the thread, leave the card.

    A persistent dispatch writes BOTH a thread row and a card in the user's
    chat. Removing only the thread leaves that card in place carrying a Reply
    action that opens a conversation which is no longer there.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _mk_agent(user)
    dispatch_id = await _seed_delivered("persistent", user)
    await _seed_thread(user, [("out", "from the dispatch")])
    async with async_session_maker() as db:
        await db.execute(
            sa_update(AdminThreadMessage)
            .where(AdminThreadMessage.user_id == user)
            .values(dispatch_id=dispatch_id)
        )
        await db.commit()

    hops: list[dict] = []

    async def _fake_proxy(url, key, path, method, json_body=None, **kw):
        hops.append({"path": path, "body": json_body})
        return {"ok": True}

    monkeypatch.setattr("app.api.admin.dispatch.proxy_to_agent", _fake_proxy)

    res = await client.request(
        "DELETE", f"/api/admin/dispatch/threads/{user}?scope=user_side")
    assert res.status_code == 200, res.text
    assert res.json()["cards_retracted"] == 1, res.json()

    assert [h["path"] for h in hops] == ["internal/admin-notice/retract"], hops
    assert hops[0]["body"] == {"user_id": user, "dispatch_id": dispatch_id}

    async with async_session_maker() as db:
        chat_status = (await db.execute(
            select(AdminDispatchTarget.chat_status).where(
                AdminDispatchTarget.dispatch_id == dispatch_id,
                AdminDispatchTarget.user_id == user,
            )
        )).scalar_one()
    assert chat_status == "retracted"


@pytest.mark.asyncio
async def test_an_unreachable_tenant_is_counted_not_swallowed(dispatch_client, monkeypatch):
    """"3 of 4 cards" is a different sentence from "done", and the operator has
    to be told which one happened. The thread rows are removed regardless — one
    dead container must not cost them the rest of the removal."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _mk_agent(user)
    dispatch_id = await _seed_delivered("persistent", user)
    await _seed_thread(user, [("out", "hello")])
    async with async_session_maker() as db:
        await db.execute(
            sa_update(AdminThreadMessage)
            .where(AdminThreadMessage.user_id == user)
            .values(dispatch_id=dispatch_id)
        )
        await db.commit()

    async def _boom(*a, **kw):
        raise RuntimeError("tenant is down")

    monkeypatch.setattr("app.api.admin.dispatch.proxy_to_agent", _boom)

    body = (await client.request(
        "DELETE", f"/api/admin/dispatch/threads/{user}?scope=user_side")).json()
    assert body["cards_failed"] == 1 and body["cards_retracted"] == 0, body
    assert body["messages"] == 1, "the thread rows come out even when a card cannot"

    async with async_session_maker() as db:
        err = (await db.execute(
            select(AdminDispatchTarget.last_error).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalar_one()
    assert err and "retract failed" in err, err


@pytest.mark.asyncio
async def test_deleting_a_conversation_twice_does_not_move_the_date(dispatch_client):
    """Idempotent by construction — the second press matches no rows. Re-
    stamping would lose WHEN the conversation was actually removed, which is
    the only question anyone asks afterwards."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _seed_thread(user, [("out", "one"), ("out", "two")])

    first = (await client.request(
        "DELETE", f"/api/admin/dispatch/threads/{user}?scope=user_side")).json()
    assert first["messages"] == 2
    stamped = (await client.get(f"/api/admin/dispatch/threads/{user}")).json()["retracted_at"]

    second = (await client.request(
        "DELETE", f"/api/admin/dispatch/threads/{user}?scope=user_side")).json()
    assert second["messages"] == 0, "a second removal has nothing left to remove"
    again = (await client.get(f"/api/admin/dispatch/threads/{user}")).json()["retracted_at"]
    assert again == stamped, "the removal date moved on a repeat press"


@pytest.mark.asyncio
async def test_the_operator_can_start_the_conversation_again(dispatch_client):
    """§3.3's "Start new conversation", and the reason no admin_threads table
    was needed for it: the next message simply carries no stamp."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _seed_thread(user, [("out", "old business"), ("in", "old reply")])
    await client.request("DELETE", f"/api/admin/dispatch/threads/{user}?scope=user_side")

    res = await client.post(f"/api/admin/dispatch/threads/{user}",
                            json={"body": "a fresh start"})
    assert res.status_code == 201, res.text

    # The operator keeps everything, and the thread is no longer "removed".
    mine = (await client.get(f"/api/admin/dispatch/threads/{user}")).json()
    assert len(mine["messages"]) == 3
    assert mine["retracted_at"] is None
    assert mine["user_visible_total"] == 1

    # The USER sees exactly the new one.
    app.dependency_overrides[get_current_user] = lambda: _principal(user, role="user")
    theirs = (await client.get("/api/notices/thread")).json()
    assert [m["body"] for m in theirs["messages"]] == ["a fresh start"]


@pytest.mark.asyncio
async def test_a_partly_deleted_conversation_is_not_a_removed_one(dispatch_client):
    """`retracted_at` is set only when NOTHING is left for them. Otherwise an
    ordinary one-message unsend would badge the whole conversation as wiped."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    user = await _mk_user()
    await _seed_thread(user, [("out", "keep me"), ("out", "unsend me")])
    mine = (await client.get(f"/api/admin/dispatch/threads/{user}")).json()
    doomed = mine["messages"][1]["id"]

    await client.request(
        "DELETE",
        f"/api/admin/dispatch/threads/{user}/messages/{doomed}?scope=user_side")

    after = (await client.get(f"/api/admin/dispatch/threads/{user}")).json()
    assert after["retracted_at"] is None, "one unsend is not a removed conversation"
    assert after["user_visible_total"] == 1

    listed = (await client.get("/api/admin/dispatch/threads")).json()["threads"]
    row = next(t for t in listed if t["user_id"] == user)
    assert row["retracted_at"] is None and row["user_visible_total"] == 1


@pytest.mark.asyncio
async def test_deleting_a_conversation_for_a_stranger_is_404(dispatch_client):
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    res = await client.request(
        "DELETE", f"/api/admin/dispatch/threads/{uuid.uuid4()}?scope=user_side")
    assert res.status_code == 404, res.text


# ── Unit 3: the recipient picker ──────────────────────────────────


@pytest.mark.asyncio
async def test_a_blank_query_browses_rather_than_refusing(dispatch_client):
    """SUPERSEDES ``test_an_empty_query_returns_nobody`` (reversed 2026-08-17).

    That test asserted `[] == GET /recipients`, on the reasoning that the
    compose form used to render the entire user base as chips and a picker doing
    the same on focus is that wall with extra steps.

    The reasoning survives; the conclusion does not. The wall's defects were that
    it rendered EVERY account, unlabelled, through a route with two aggregate
    subqueries per row. A capped, labelled, cheap list is none of those — and the
    rule it produced only helped an operator who already knew the address, which
    is not the common case.

    Whitespace still counts as blank: `?query=%20%20` browses, it does not search
    for two spaces.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    await _mk_user()
    await _mk_user()

    assert (await client.get("/api/admin/dispatch/recipients")).json() != []
    assert (await client.get("/api/admin/dispatch/recipients?query=%20%20")).json() != []


@pytest.mark.asyncio
async def test_recipients_match_email_name_and_exact_id(dispatch_client):
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    target = await _mk_user()
    other = await _mk_user()
    async with async_session_maker() as db:
        await db.execute(sa_update(User).where(User.id == target).values(
            name="Parasto Fahimi", email="parasto@example.com"))
        await db.execute(sa_update(User).where(User.id == other).values(
            name="Someone Else", email="else@example.com"))
        await db.commit()

    def ids(res):
        return {u["id"] for u in res.json()}

    assert ids(await client.get("/api/admin/dispatch/recipients?query=parasto@")) == {target}
    assert ids(await client.get("/api/admin/dispatch/recipients?query=fahimi")) == {target}
    # Pasting an id out of a log or a support ticket is a real path in.
    assert ids(await client.get(f"/api/admin/dispatch/recipients?query={target}")) == {target}
    assert ids(await client.get("/api/admin/dispatch/recipients?query=nobodyhere")) == set()


@pytest.mark.asyncio
async def test_a_recipient_says_whether_a_card_can_reach_them(dispatch_client):
    """`has_agent` changes what sending DOES — a user with no reachable
    container still gets the notification but no in-chat card. That belongs
    beside the name while choosing, not in a count afterwards."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    with_agent = await _mk_user()
    await _mk_agent(with_agent)
    without = await _mk_user()
    async with async_session_maker() as db:
        await db.execute(sa_update(User).where(User.id == with_agent).values(
            email="haz@agentsearch.test"))
        await db.execute(sa_update(User).where(User.id == without).values(
            email="noz@agentsearch.test"))
        await db.commit()

    rows = (await client.get(
        "/api/admin/dispatch/recipients?query=agentsearch.test")).json()
    by_id = {r["id"]: r for r in rows}
    assert by_id[with_agent]["has_agent"] is True
    assert by_id[without]["has_agent"] is False


@pytest.mark.asyncio
async def test_the_picker_is_capped_and_ordered(dispatch_client):
    """Unbounded, one broad query returns the whole user base down a dropdown.
    Ordered, because a list that reshuffles between keystrokes is unusable."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    for i in range(12):
        u = await _mk_user()
        async with async_session_maker() as db:
            await db.execute(sa_update(User).where(User.id == u).values(
                email=f"cap{i:02d}@capped.test"))
            await db.commit()

    rows = (await client.get(
        "/api/admin/dispatch/recipients?query=capped.test&limit=5")).json()
    assert len(rows) == 5
    emails = [r["email"] for r in rows]
    assert emails == sorted(emails), emails


@pytest.mark.asyncio
async def test_recipients_browses_on_an_empty_query_and_narrows_on_a_typed_one(dispatch_client):
    """REVERSED 2026-08-17, at the founder's request.

    This route used to answer an empty query with `[]`, on the reasoning that
    the recipient list is not something to browse. That is right for a chip wall
    of 10,000 and wrong for an operator who knows the person but not the
    address: an empty box that answers nothing is a dead end, not a discipline.

    Newest-first, to match the admin Users list (users.py:212) — the same people
    in the same order in both places.

    The old rule WAS pinned, by ``test_an_empty_query_returns_nobody`` directly
    above — now rewritten as its counterpart rather than deleted, so the reversal
    is legible instead of looking like coverage that quietly went missing. This
    test pins the new behaviour in both directions, so "browse" cannot become
    "return everything" and "search" cannot stop filtering.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    base = datetime.utcnow() - timedelta(days=10)
    ids = [await _mk_user(created_at=base + timedelta(days=i)) for i in range(3)]

    res = await client.get("/api/admin/dispatch/recipients")
    assert res.status_code == 200, res.text
    browsed = res.json()
    assert browsed, "an empty query must BROWSE, not return []"

    returned = [r["id"] for r in browsed]
    # Newest first — assert on our own rows' relative order, since the fixture DB
    # carries other users from sibling tests and an absolute index would be a
    # test that fails when a neighbour is added.
    positions = [returned.index(u) for u in ids if u in returned]
    assert positions == sorted(positions, reverse=True), (
        f"expected newest-first; our users landed at {positions}"
    )

    # The cap is real, and it is what keeps "browse" honest at 10,000 accounts.
    capped = await client.get("/api/admin/dispatch/recipients", params={"limit": 2})
    assert len(capped.json()) == 2, capped.text

    # And a typed query still narrows rather than browsing.
    async with async_session_maker() as db:
        target = (await db.execute(select(User).where(User.id == ids[0]))).scalar_one()
        email = target.email
    hit = await client.get("/api/admin/dispatch/recipients", params={"query": email})
    assert [r["id"] for r in hit.json()] == [ids[0]], hit.text

    miss = await client.get(
        "/api/admin/dispatch/recipients", params={"query": "zzz-no-such-account-zzz"})
    assert miss.json() == [], miss.text


# ── Attachments (migration 093) ───────────────────────────────────


def _png(nbytes: int = 64) -> bytes:
    """A real PNG header plus filler. The header matters: a route that ever
    starts sniffing content must not be fooled by this fixture."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * max(0, nbytes - 8)


@pytest.mark.asyncio
async def test_a_user_can_attach_a_picture_and_only_they_and_admins_can_fetch_it(
    dispatch_client, monkeypatch,
):
    """The whole point, end to end, plus the thing that would make it a leak.

    The bytes live in the platform DB and are served from our own origin, so
    "who may GET this id" is the entire security model. A second user must not
    be able to fetch it by id — these are uuids in a table shared by everyone.
    """
    from app.api.notices import router as notices_router

    client, app = dispatch_client
    app.include_router(notices_router, prefix=settings.api_prefix + "/notices")

    owner = await _mk_user()
    stranger = await _mk_user()
    admin = await _mk_user(role="admin")

    app.dependency_overrides[get_current_user] = lambda: _principal(owner)
    res = await client.post(
        "/api/notices/thread/attachment",
        files={"file": ("shot.png", _png(256), "image/png")},
        data={"body": "it looks like this"},
    )
    assert res.status_code == 201, res.text
    msg = res.json()["message"]
    assert msg["body"] == "it looks like this"
    assert len(msg["attachments"]) == 1, msg
    att = msg["attachments"][0]
    assert att["mime_type"] == "image/png"
    assert att["size_bytes"] == 256
    # The BYTES are not in the JSON. A thread GET returns up to 500 messages;
    # inlining them would re-download every picture on every poll.
    assert "data" not in att and "base64" not in res.text.lower()

    # The owner can fetch it.
    got = await client.get(f"/api/notices/thread/attachments/{att['id']}")
    assert got.status_code == 200, got.text
    assert got.content == _png(256)
    assert got.headers["content-type"].startswith("image/png")
    # User-supplied bytes from our own origin.
    assert got.headers.get("x-content-type-options") == "nosniff"

    # A different user cannot, by id.
    app.dependency_overrides[get_current_user] = lambda: _principal(stranger)
    assert (await client.get(f"/api/notices/thread/attachments/{att['id']}")).status_code == 404

    # An admin can, through the admin route scoped to that user's thread...
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    ok = await client.get(f"/api/admin/dispatch/threads/{owner}/attachments/{att['id']}")
    assert ok.status_code == 200, ok.text
    assert ok.content == _png(256)
    # ...but NOT by pointing that route at the wrong user. Being an admin is not
    # a licence to read an id out of whatever thread it happens to live in.
    wrong = await client.get(f"/api/admin/dispatch/threads/{stranger}/attachments/{att['id']}")
    assert wrong.status_code == 404, wrong.text

    # And it appears in the operator's thread view.
    thread = await client.get(f"/api/admin/dispatch/threads/{owner}")
    assert thread.json()["messages"][0]["attachments"][0]["id"] == att["id"]


@pytest.mark.asyncio
async def test_an_attachment_is_refused_by_type_and_by_size(dispatch_client, monkeypatch):
    """Both caps, and the size one specifically because it is the DoS.

    ``support_attachments`` and this table are the only blobs in the platform
    DB, and the audio blob cache reaching 96% of that database is why the cap is
    not decorative.
    """
    from app.api.notices import router as notices_router

    client, app = dispatch_client
    app.include_router(notices_router, prefix=settings.api_prefix + "/notices")
    user = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(user)

    # HEIC is the one a phone offers most eagerly and no browser renders.
    bad = await client.post(
        "/api/notices/thread/attachment",
        files={"file": ("photo.heic", _png(32), "image/heic")},
    )
    assert bad.status_code == 415, bad.text

    monkeypatch.setattr(settings, "admin_thread_attachment_max_bytes", 100, raising=False)
    big = await client.post(
        "/api/notices/thread/attachment",
        files={"file": ("big.png", _png(101), "image/png")},
    )
    assert big.status_code == 413, big.text

    empty = await client.post(
        "/api/notices/thread/attachment",
        files={"file": ("nothing.png", b"", "image/png")},
    )
    assert empty.status_code == 400, empty.text

    # A refused upload must leave NO message behind. The row is added to the
    # session before the file is validated, so this asserts the rollback rather
    # than assuming it.
    async with async_session_maker() as db:
        n = (await db.execute(
            select(func.count()).select_from(AdminThreadMessage)
            .where(AdminThreadMessage.user_id == user)
        )).scalar_one()
    assert n == 0, f"{n} orphaned message rows survived a rejected upload"


@pytest.mark.asyncio
async def test_deleting_a_message_for_everyone_takes_its_picture_with_it(dispatch_client):
    """092 clears the words. A picture that outlived them would be a deletion
    that only looked complete — and it is the half anyone would actually look
    at."""
    from app.api.notices import router as notices_router

    client, app = dispatch_client
    app.include_router(notices_router, prefix=settings.api_prefix + "/notices")

    owner = await _mk_user()
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(owner)
    res = await client.post(
        "/api/notices/thread/attachment",
        files={"file": ("shot.png", _png(64), "image/png")},
    )
    att_id = res.json()["message"]["attachments"][0]["id"]

    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    gone = await client.delete(
        f"/api/admin/dispatch/threads/{owner}", params={"scope": "everyone"})
    assert gone.status_code == 200, gone.text

    # 404 for the OPERATOR too, by id. This assertion started life as
    # `in (200, 404)`, which accepts either answer and therefore asserts
    # nothing — and it was hiding the real question: "delete for everyone"
    # clears the body, so leaving the bytes fetchable would make the
    # destructive scope the one that destroys less. The blob is now purged.
    assert (await client.get(
        f"/api/admin/dispatch/threads/{owner}/attachments/{att_id}")).status_code == 404
    # ...and not described in either thread view.
    thread = await client.get(f"/api/admin/dispatch/threads/{owner}")
    for m in thread.json()["messages"]:
        assert m["attachments"] == [], m

    app.dependency_overrides[get_current_user] = lambda: _principal(owner)
    assert (await client.get(f"/api/notices/thread/attachments/{att_id}")).status_code == 404


@pytest.mark.asyncio
async def test_removing_for_the_user_only_keeps_the_operators_copy_of_the_picture(
    dispatch_client,
):
    """The other scope, which must NOT destroy anything.

    `user_side` exists so the operator keeps a record of what they sent. If it
    purged the bytes like `everyone` does, the two scopes would differ only in
    wording — and the read-only transcript the whole feature promises would be
    missing the one part that mattered.
    """
    from app.api.notices import router as notices_router

    client, app = dispatch_client
    app.include_router(notices_router, prefix=settings.api_prefix + "/notices")

    owner = await _mk_user()
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(owner)
    att_id = (await client.post(
        "/api/notices/thread/attachment",
        files={"file": ("shot.png", _png(64), "image/png")},
    )).json()["message"]["attachments"][0]["id"]

    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")
    assert (await client.delete(
        f"/api/admin/dispatch/threads/{owner}", params={"scope": "user_side"},
    )).status_code == 200

    # The operator still has it, described AND fetchable.
    thread = await client.get(f"/api/admin/dispatch/threads/{owner}")
    assert thread.json()["messages"][0]["attachments"][0]["id"] == att_id
    kept = await client.get(f"/api/admin/dispatch/threads/{owner}/attachments/{att_id}")
    assert kept.status_code == 200, kept.text
    assert kept.content == _png(64)

    # The user does not — neither the bytes nor the row.
    app.dependency_overrides[get_current_user] = lambda: _principal(owner)
    assert (await client.get(f"/api/notices/thread/attachments/{att_id}")).status_code == 404
    assert (await client.get("/api/notices/thread")).json()["messages"] == []


def test_the_serializer_refuses_to_describe_a_deleted_messages_attachment():
    """Defence in depth, tested directly because nothing else reaches it.

    In the live path `everyone` PURGES the bytes, so by the time any thread is
    serialised there is nothing left to describe and this guard never fires — a
    mutation that removes it kills no test. That is exactly the shape CLAUDE.md
    warns about: a guard whose precondition something above it satisfies is
    invisible to every check in the repo.

    It is kept rather than deleted because the purge is one statement in one
    transaction, and any future path that tombstones a row without calling it
    would otherwise start advertising an image for a message whose words are
    gone. So it is exercised here at the unit level, with an attachment that
    still exists alongside a tombstone — a state the routes do not produce.
    """
    import types as _t
    from app.api.admin.dispatch import THREAD_IN, thread_message_out

    att = _t.SimpleNamespace(id="a-1", mime_type="image/png", size_bytes=10)
    live = _t.SimpleNamespace(
        id="m-1", direction=THREAD_IN, body="here", created_at=datetime.utcnow(),
        dispatch_id=None, read_at=None, sender_name=None,
        hidden_from_user_at=None, deleted_at=None,
    )
    assert len(thread_message_out(live, attachments=[att]).attachments) == 1

    tombstoned = _t.SimpleNamespace(
        id="m-2", direction=THREAD_IN, body=DELETED_BODY, created_at=datetime.utcnow(),
        dispatch_id=None, read_at=None, sender_name=None,
        hidden_from_user_at=None, deleted_at=datetime.utcnow(),
    )
    assert thread_message_out(tombstoned, attachments=[att]).attachments == []
