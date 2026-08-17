"""product_events — the Admin Dispatch funnel.

Nine event names are defined; five have a producer today. What each test
here is actually defending:

  1. A full lifecycle (create → fan-out → read → reply) writes exactly the
     five live events, once each, with the right subject and actor. The
     "exactly" is the load-bearing half: it is also what proves the four
     dark names are not being emitted by accident — in particular that the
     `once` retract, which fires on read, is NOT recorded as a revoke.
  2. A retry re-walks every recipient (the fan-out is idempotent by
     construction, so it must), and that must not inflate the funnel.
  3. …except where a retry genuinely delivers something it did not before:
     a recipient whose container was down gets `no_agent`, and the pass
     that finally lands the chat card is a second delivery, not a repeat.
  4. Telemetry may never break delivery. With the event write failing on
     every call, the dispatch still delivers and the read receipt is still
     recorded.
  5. A `persistent` notice's read is counted. `mark_notice_read` returns
     early for anything that needs no retract, so an emit one line lower
     would count `once` reads only and report zero for the whole
     persistent half — with every other test still green.
  6. Only the RECIPIENT's reply counts. The operator's own follow-up in
     the thread writes no event, or the reply rate rises every time the
     operator talks to themselves.
"""

from __future__ import annotations

import json
import types
import uuid
from datetime import datetime

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
    AgentConfig,
    NotificationQueue,
    ProductEvent,
    User,
    DISPATCH_PRODUCT_EVENTS,
    PE_DISPATCH_CREATED,
    PE_DISPATCH_DELIVERED,
    PE_DISPATCH_READ,
    PE_DISPATCH_REPLIED,
    PE_DISPATCH_SCREENSHOT_DETECTED,
    PE_DISPATCH_SENT,
    PE_ENTITY_DISPATCH,
    PE_ENTITY_THREAD_MESSAGE,
)
from app.services.admin_dispatch_worker import run_dispatch_fanout


LIVE_EVENTS = {
    PE_DISPATCH_CREATED,
    PE_DISPATCH_SENT,
    PE_DISPATCH_DELIVERED,
    PE_DISPATCH_READ,
    PE_DISPATCH_REPLIED,
}
# Has a producer, but is NOT a delivery milestone — so it is deliberately out
# of LIVE_EVENTS, which the lifecycle test asserts as an exact set. A
# screenshot is caused by the USER, can happen any number of times, and can
# happen when nothing is being delivered at all. Folding it in would have made
# the lifecycle test demand a screenshot on every dispatch.
SIGNAL_EVENTS = {PE_DISPATCH_SCREENSHOT_DETECTED}
# The three still with no producer: revoke and delete are capabilities landing
# in sibling PRs, and `viewed` needs a client-side viewport rule neither client
# ships (D4: read is an ack, never a render).
DARK_EVENTS = DISPATCH_PRODUCT_EVENTS - LIVE_EVENTS - SIGNAL_EVENTS


# ── Harness (same shape as test_admin_dispatch.py) ────────────────


def _principal(uid: str, role: str = "user", email: str = "u@x.com"):
    return types.SimpleNamespace(id=uid, role=role, email=email, name="U")


@pytest.fixture(autouse=True)
def _no_background_fanout(monkeypatch):
    """Both producing routes fire the fan-out as a background task; every
    test here drives `run_dispatch_fanout` explicitly instead.

    Letting the spawned one race the explicit one is not "testing the
    two-replica case" — on this harness both sessions share ONE sqlite
    connection (StaticPool), so their transactions interleave in ways two
    Postgres replicas never do, and the first `_reconcile` to settle
    reported `sent` with zero deliveries. Genuine concurrency is covered by
    test_admin_dispatch.py's `gather()` of two fan-outs.
    """
    async def _no_spawn(_dispatch_id):
        return None

    monkeypatch.setattr("app.api.admin.dispatch.spawn_dispatch_fanout", _no_spawn)


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


async def _mk_user(*, role: str = "user") -> str:
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"pe-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12),
            name="Product Events Test",
            role=role,
            timezone="America/Toronto",
            created_at=datetime.utcnow(),
            notification_preferences={
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
            },
        ))
        await db.commit()
    return user_id


async def _mk_agent(user_id: str, url: str = "https://agent.example") -> str:
    key = f"tk-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=user_id, agent_api_key=key, agent_url=url,
            deploy_status="active",
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
    def __init__(self, status_code: int = 201):
        self.calls: list[dict] = []
        self.status_code = status_code

    async def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        return _FakeResponse(
            self.status_code,
            {"message_id": f"msg-{len(self.calls)}", "day_chat_id": None, "ws_count": 0},
        )


def _patch_agent_http(monkeypatch, spy: _AgentSpy) -> _AgentSpy:
    from app.services import agent_http

    monkeypatch.setattr(agent_http, "get_agent_http_client", lambda: spy)
    return spy


async def _events(event: str | None = None) -> list[ProductEvent]:
    async with async_session_maker() as db:
        stmt = select(ProductEvent).order_by(ProductEvent.created_at.asc())
        if event is not None:
            stmt = stmt.where(ProductEvent.event == event)
        return list((await db.execute(stmt)).scalars().all())


async def _compose_and_send(client, admin_id: str, recipient_id: str, *, mode="once") -> str:
    """POST the dispatch as an admin, then run the fan-out to completion.

    The route only spawns the fan-out; awaiting it here is what the retry
    route and every other dispatch test do, so the assertions are about the
    events and not about task scheduling.
    """
    res = await client.post("/api/admin/dispatch", json={
        "mode": mode,
        "audience": "user",
        "target_user_id": recipient_id,
        "title": "Scheduled maintenance",
        "body": "Toup will be briefly unavailable at 02:00 UTC.",
    })
    assert res.status_code == 201, res.text
    dispatch_id = res.json()["dispatch"]["id"]
    await run_dispatch_fanout(dispatch_id)
    return dispatch_id


# ── 1. The five live events, at the true points ───────────────────


@pytest.mark.asyncio
async def test_a_full_lifecycle_writes_the_five_live_events_once_each(
    dispatch_client, monkeypatch,
):
    client, app = dispatch_client
    admin_id = await _mk_user(role="admin")
    user_id = await _mk_user()
    await _mk_agent(user_id)
    _patch_agent_http(monkeypatch, _AgentSpy())

    async def _fake_proxy(agent_url, agent_api_key, path, method="GET", **kw):
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _fake_proxy)

    app.dependency_overrides[get_current_user] = lambda: _principal(admin_id, role="admin")
    dispatch_id = await _compose_and_send(client, admin_id, user_id, mode="once")

    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    assert (await client.post(f"/api/notices/{dispatch_id}/read")).status_code == 204
    assert (await client.post("/api/notices/thread", json={"body": "got it, thanks"})).status_code == 201

    rows = await _events()
    by_event: dict[str, list[ProductEvent]] = {}
    for r in rows:
        by_event.setdefault(r.event, []).append(r)

    assert set(by_event) == LIVE_EVENTS, (
        f"expected exactly {sorted(LIVE_EVENTS)}, got {sorted(by_event)}"
    )
    assert all(len(v) == 1 for v in by_event.values()), {
        k: len(v) for k, v in by_event.items()
    }
    # The `once` retract fired on this read (the fake proxy above answered
    # it). It is the automatic end of a `once` card's life, NOT an operator
    # recalling a message, and recording it as one would make the revoke
    # metric read one-per-read forever.
    assert not (set(by_event) & DARK_EVENTS), sorted(set(by_event) & DARK_EVENTS)

    created = by_event[PE_DISPATCH_CREATED][0]
    assert created.actor_user_id == admin_id, "the operator is the actor"
    assert created.user_id == user_id, "a single-user dispatch has a subject"
    assert (created.entity_type, created.entity_id) == (PE_ENTITY_DISPATCH, dispatch_id)
    assert created.payload_json["mode"] == "once"

    delivered = by_event[PE_DISPATCH_DELIVERED][0]
    assert delivered.user_id == user_id
    assert delivered.actor_user_id is None, "nobody DID a delivery"
    assert delivered.payload_json["chat_status"] == "delivered"
    assert delivered.payload_json["notified"] is True

    sent = by_event[PE_DISPATCH_SENT][0]
    assert sent.entity_id == dispatch_id
    assert sent.payload_json["target_count"] == 1
    assert sent.payload_json["chat_delivered_count"] == 1

    read = by_event[PE_DISPATCH_READ][0]
    assert (read.user_id, read.actor_user_id) == (user_id, user_id)
    assert read.entity_id == dispatch_id

    replied = by_event[PE_DISPATCH_REPLIED][0]
    assert (replied.user_id, replied.actor_user_id) == (user_id, user_id)
    assert replied.entity_type == PE_ENTITY_THREAD_MESSAGE
    # Length, never the words.
    assert replied.payload_json["body_chars"] == len("got it, thanks")
    assert "got it" not in json.dumps(replied.payload_json)


@pytest.mark.asyncio
async def test_a_broadcast_created_event_has_no_subject(dispatch_client, monkeypatch):
    """`user_id` is the SUBJECT. A broadcast has no single one, and stamping
    the operator there would make "events about this account" return every
    broadcast the operator ever sent."""
    client, app = dispatch_client
    admin_id = await _mk_user(role="admin")
    await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(admin_id, role="admin")
    # G-DISPATCH-BROADCAST defaults OFF; this test is about the event's
    # subject, not the kill switch, so turn it on explicitly.
    monkeypatch.setattr(settings, "dispatch_broadcast_enabled", True, raising=False)

    res = await client.post("/api/admin/dispatch", json={
        "mode": "once", "audience": "all", "confirm": "BROADCAST",
        "title": "Heads up", "body": "Maintenance tonight.",
    })
    assert res.status_code == 201, res.text

    created = await _events(PE_DISPATCH_CREATED)
    assert len(created) == 1
    assert created[0].user_id is None
    assert created[0].actor_user_id == admin_id
    assert created[0].payload_json["audience"] == "all"


# ── 2/3. Retries ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_retry_does_not_inflate_the_funnel(dispatch_client):
    """The recipient here has NO agent container, which is deliberate and is
    the whole point of the test.

    Retry re-queues three classes of target, and `done` + `no_agent` is the
    one it reaches in practice. A target that is `done` + `delivered` is not
    reset at all, so for THAT one the ledger alone stops a second walk and
    an event with no dedupe key would still look correct. `no_agent` is
    reset on purpose — and if the container is still down, the second pass
    delivers exactly what the first did. The dedupe key is the only thing
    between an operator pressing Retry four times and a funnel reporting
    four people reached.
    """
    client, app = dispatch_client
    admin_id = await _mk_user(role="admin")
    user_id = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(admin_id, role="admin")

    dispatch_id = await _compose_and_send(client, admin_id, user_id)
    assert len(await _events(PE_DISPATCH_DELIVERED)) == 1
    assert len(await _events(PE_DISPATCH_SENT)) == 1

    for _ in range(2):
        res = await client.post(f"/api/admin/dispatch/{dispatch_id}/retry")
        assert res.status_code == 200, res.text
        assert res.json()["retried"] == 1, "the no_agent target IS re-queued"
        await run_dispatch_fanout(dispatch_id)
    # And once more with nothing left to claim, because a second replica
    # reconciling the same dispatch settles the status again.
    await run_dispatch_fanout(dispatch_id)

    delivered = await _events(PE_DISPATCH_DELIVERED)
    assert len(delivered) == 1, [r.payload_json for r in delivered]
    assert delivered[0].payload_json["chat_status"] == "no_agent"
    assert len(await _events(PE_DISPATCH_SENT)) == 1
    assert len(await _events(PE_DISPATCH_CREATED)) == 1


@pytest.mark.asyncio
async def test_a_retry_that_lands_the_chat_card_is_a_second_delivery(
    dispatch_client, monkeypatch,
):
    """The deliberate NON-duplicate. A recipient whose container was down is
    `no_agent` — banner, no chat card. When Retry finally lands the card
    that IS a second delivery, and collapsing it would erase the only
    number that says whether the Retry button works.

    Recipients reached stays COUNT(DISTINCT user_id) = 1; deliveries is
    COUNT(*) = 2, and the difference is exactly the upgrades.
    """
    client, app = dispatch_client
    admin_id = await _mk_user(role="admin")
    user_id = await _mk_user()  # deliberately NO agent config yet
    app.dependency_overrides[get_current_user] = lambda: _principal(admin_id, role="admin")

    dispatch_id = await _compose_and_send(client, admin_id, user_id)

    first = await _events(PE_DISPATCH_DELIVERED)
    assert len(first) == 1
    assert first[0].payload_json["chat_status"] == "no_agent"
    assert first[0].payload_json["notified"] is True, "the banner still landed"

    # The container comes back, the operator presses Retry.
    await _mk_agent(user_id)
    _patch_agent_http(monkeypatch, _AgentSpy())
    assert (await client.post(f"/api/admin/dispatch/{dispatch_id}/retry")).status_code == 200
    await run_dispatch_fanout(dispatch_id)

    rows = await _events(PE_DISPATCH_DELIVERED)
    assert [r.payload_json["chat_status"] for r in rows] == ["no_agent", "delivered"]
    assert len({r.user_id for r in rows}) == 1, "one recipient, two deliveries"


# ── 4. Telemetry may never break delivery ─────────────────────────


@pytest.mark.asyncio
async def test_a_dead_event_writer_does_not_break_a_dispatch(
    dispatch_client, monkeypatch,
):
    """The whole design constraint, exercised: with every event write
    raising, the dispatch still reaches its recipient and the read receipt
    is still recorded. Telemetry that can break delivery is worse than no
    telemetry."""
    client, app = dispatch_client
    admin_id = await _mk_user(role="admin")
    user_id = await _mk_user()
    await _mk_agent(user_id)
    spy = _patch_agent_http(monkeypatch, _AgentSpy())

    async def _fake_proxy(agent_url, agent_api_key, path, method="GET", **kw):
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _fake_proxy)

    # Break the session the emitter opens — the deepest failure it can meet
    # short of the process dying, and the one that on a shared session would
    # poison the caller's transaction.
    def _boom(*a, **kw):
        raise RuntimeError("telemetry DB is on fire")

    monkeypatch.setattr("app.db.database.async_session_maker", _boom)

    app.dependency_overrides[get_current_user] = lambda: _principal(admin_id, role="admin")
    dispatch_id = await _compose_and_send(client, admin_id, user_id)

    async with async_session_maker() as db:
        target = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()
        assert target.state == "done", target.last_error
        assert target.chat_status == "delivered", target.last_error
        assert target.notification_id is not None
        dispatch = await db.get(AdminDispatch, dispatch_id)
        assert dispatch.status == "sent"
        assert (await db.execute(
            select(NotificationQueue).where(NotificationQueue.user_id == user_id)
        )).scalars().all() != []
    assert len(spy.calls) == 1, "the agent hop still happened"

    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    assert (await client.post(f"/api/notices/{dispatch_id}/read")).status_code == 204
    async with async_session_maker() as db:
        target = (await db.execute(
            select(AdminDispatchTarget).where(
                AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalars().one()
        assert target.read_at is not None, "the only read receipt in the system"

    assert await _events() == [], "the writer was broken for the whole run"


# ── 5. Guard order: a persistent read is still a read ─────────────


@pytest.mark.asyncio
async def test_a_persistent_notice_read_is_counted(dispatch_client, monkeypatch):
    """`mark_notice_read` returns early once it knows no retract is needed,
    which is EVERY persistent notice. An emit below that return counts
    `once` reads only — and reports zero for the persistent half of the
    product, silently, with every other assertion in this file still green.
    """
    client, app = dispatch_client
    admin_id = await _mk_user(role="admin")
    user_id = await _mk_user()
    await _mk_agent(user_id)
    _patch_agent_http(monkeypatch, _AgentSpy())

    proxied: list[str] = []

    async def _fake_proxy(agent_url, agent_api_key, path, method="GET", **kw):
        proxied.append(path)
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _fake_proxy)

    app.dependency_overrides[get_current_user] = lambda: _principal(admin_id, role="admin")
    dispatch_id = await _compose_and_send(client, admin_id, user_id, mode="persistent")

    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    assert (await client.post(f"/api/notices/{dispatch_id}/read")).status_code == 204

    assert proxied == [], "a persistent notice is never retracted — the early return"
    read = await _events(PE_DISPATCH_READ)
    assert len(read) == 1, "the persistent half of the funnel must not read zero"
    assert read[0].entity_id == dispatch_id
    assert read[0].payload_json["mode"] == "persistent"


@pytest.mark.asyncio
async def test_a_second_got_it_is_not_a_second_read(dispatch_client, monkeypatch):
    client, app = dispatch_client
    admin_id = await _mk_user(role="admin")
    user_id = await _mk_user()
    await _mk_agent(user_id)
    _patch_agent_http(monkeypatch, _AgentSpy())

    async def _fake_proxy(agent_url, agent_api_key, path, method="GET", **kw):
        return {"deleted": 1, "ws_count": 0}

    monkeypatch.setattr("app.api.notices.proxy_to_agent", _fake_proxy)

    app.dependency_overrides[get_current_user] = lambda: _principal(admin_id, role="admin")
    dispatch_id = await _compose_and_send(client, admin_id, user_id, mode="persistent")

    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    for _ in range(3):
        assert (await client.post(f"/api/notices/{dispatch_id}/read")).status_code == 204

    assert len(await _events(PE_DISPATCH_READ)) == 1


# ── 6. Only the recipient's reply counts ──────────────────────────


@pytest.mark.asyncio
async def test_an_operator_follow_up_is_not_a_reply(dispatch_client, monkeypatch):
    """`dispatch_replied` answers "did this message start a conversation".
    Counting the operator's own `out` rows makes that rate rise whenever the
    operator talks to themselves."""
    client, app = dispatch_client
    admin_id = await _mk_user(role="admin")
    user_id = await _mk_user()
    await _mk_agent(user_id)
    _patch_agent_http(monkeypatch, _AgentSpy())
    app.dependency_overrides[get_current_user] = lambda: _principal(admin_id, role="admin")

    await _compose_and_send(client, admin_id, user_id, mode="persistent")

    res = await client.post(
        f"/api/admin/dispatch/threads/{user_id}", json={"body": "any questions?"},
    )
    assert res.status_code == 201, res.text
    assert await _events(PE_DISPATCH_REPLIED) == []

    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)
    assert (await client.post("/api/notices/thread", json={"body": "no, all good"})).status_code == 201

    replied = await _events(PE_DISPATCH_REPLIED)
    assert len(replied) == 1
    assert replied[0].actor_user_id == user_id


# ── The names themselves ──────────────────────────────────────────


def test_the_nine_event_names_are_declared_and_distinct():
    """Four have no producer in this PR and are declared anyway: a later PR
    that ships revoke/delete/screenshot detection needs one place to look,
    and a name invented at that call site instead is a metric that reads
    zero forever under a spelling nobody queries."""
    assert len(DISPATCH_PRODUCT_EVENTS) == 9, sorted(DISPATCH_PRODUCT_EVENTS)
    assert LIVE_EVENTS < DISPATCH_PRODUCT_EVENTS
    # Screenshot moved out of the dark set when it gained a producer. It is not
    # in LIVE_EVENTS either — it is not a delivery milestone, so the lifecycle
    # test must not demand one.
    assert len(DARK_EVENTS) == 3, sorted(DARK_EVENTS)
    assert DARK_EVENTS == {
        "dispatch_viewed",
        "dispatch_revoked",
        "dispatch_deleted",
    }
    assert SIGNAL_EVENTS == {"dispatch_screenshot_detected"}
    assert not (LIVE_EVENTS & SIGNAL_EVENTS), "an event is a milestone or a signal, not both"
    assert LIVE_EVENTS | SIGNAL_EVENTS | DARK_EVENTS == DISPATCH_PRODUCT_EVENTS


# ── Screenshot signal ─────────────────────────────────────────────
#
# Detection, not prevention. These pin the two things most likely to be got
# wrong later: that a repeat is COUNTED (it is the whole metric), and that a
# failure in telemetry can never reach the user.


@pytest.mark.asyncio
async def test_every_screenshot_is_counted_not_deduped(dispatch_client):
    """A second screenshot is a second screenshot.

    Every other event in this table dedupes on a fact that can only happen
    once — created, sent, read. This one is a repeatable act, and the count IS
    the signal an operator reads. A dedupe key here would silently report
    "1" forever no matter how many times a message left the app.
    """
    client, app = dispatch_client
    user_id = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)

    for _ in range(3):
        res = await client.post("/api/notices/screenshot", json={"surface": "thread"})
        assert res.status_code == 204, res.text

    rows = await _events(PE_DISPATCH_SCREENSHOT_DETECTED)
    assert len(rows) == 3, f"expected 3 recorded screenshots, got {len(rows)}"
    assert all(r.user_id == user_id and r.actor_user_id == user_id for r in rows)


@pytest.mark.asyncio
async def test_a_broken_telemetry_write_never_reaches_the_user(
    dispatch_client, monkeypatch,
):
    """The user took a screenshot of their own screen. Whatever happens to our
    logging of it, they must not see an error — and the client must not be
    handed a status that invites a retry loop."""
    client, app = dispatch_client
    user_id = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)

    import app.services.product_events as pe

    async def _boom(*a, **kw):
        raise RuntimeError("telemetry DB is on fire")

    monkeypatch.setattr(pe, "emit_product_event", _boom)

    res = await client.post("/api/notices/screenshot", json={"surface": "thread"})
    assert res.status_code == 204, res.text


@pytest.mark.asyncio
async def test_a_card_screenshot_is_attributed_to_its_dispatch(dispatch_client):
    """A thread screenshot and a card screenshot are different facts, and an
    operator asks a different question of each."""
    client, app = dispatch_client
    user_id = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(user_id)

    res = await client.post("/api/notices/screenshot",
                            json={"surface": "notice", "dispatch_id": "dsp-9"})
    assert res.status_code == 204, res.text

    rows = await _events(PE_DISPATCH_SCREENSHOT_DETECTED)
    assert len(rows) == 1
    assert rows[0].entity_type == "dispatch" and rows[0].entity_id == "dsp-9"
    assert rows[0].payload_json == {"surface": "notice"}
