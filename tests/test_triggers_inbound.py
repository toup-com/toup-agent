"""Tests for the agent-side trigger inbound endpoint (Gate T1).

The endpoint is hot-path Pub/Sub dispatch ingest. Production invariants:
  - X-Agent-Key auth: missing / wrong key → 401, never 200.
  - Cross-tenant routing bug → 401 (body.user_id != container owner).
  - INSERT … ON CONFLICT DO NOTHING dedupe: duplicate event_dedupe_id
    on a retry returns dedupe_hit, NOT a fresh row.
  - No enabled trigger → 200 with no_trigger count, NOT a 4xx
    (Pub/Sub interprets 4xx as "stop retrying, drop the event"; we
    want it dropped, but only after we've authenticated the call —
    the response payload is the only audit trail we keep at the
    inbound layer).

Tests bypass FastAPI's full app fixture and call the endpoint
directly via a minimal app with just the inbound router mounted.
That avoids the slow conftest reset_database autouse and keeps
the endpoint reachable on any DB backend.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select


# Force a known agent identity into settings BEFORE app import so the
# X-Agent-Key contract has a fixed key to compare against. `setdefault`
# so conftest's ENVIRONMENT=test takes precedence and we don't shadow
# anything CI sets.
os.environ.setdefault("AGENT_API_KEY", "test-agent-key-trigger-inbound")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-000000000aaa")


CONTAINER_USER_ID = "00000000-0000-0000-0000-000000000aaa"
CONTAINER_KEY = "test-agent-key-trigger-inbound"


def _build_app() -> FastAPI:
    from app.api.triggers_inbound import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    app = _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def trigger_id() -> str:
    """Seed an enabled email_received trigger owned by the container."""
    from app.db import async_session_maker
    from app.db.models import Trigger

    tid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Trigger(
            id=tid,
            user_id=CONTAINER_USER_ID,
            kind="email_received",
            action="summarize_and_post",
            enabled=True,
        ))
        await db.commit()
    return tid


# ── Auth ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_x_agent_key_returns_401(client):
    """Bare POST with no auth header — must 401. Pub/Sub will not retry
    on 401 (good — a wrong key is permanent), so we never accidentally
    silence a misconfigured platform."""
    r = await client.post("/api/triggers/inbound", json={
        "trigger_kind": "email_received",
        "user_id": CONTAINER_USER_ID,
        "events": [{"event_dedupe_id": "msg_x"}],
    })
    assert r.status_code == 401, r.text


@pytest.mark.asyncio
async def test_wrong_x_agent_key_returns_401(client):
    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": "wrong-key"},
        json={
            "trigger_kind": "email_received",
            "user_id": CONTAINER_USER_ID,
            "events": [{"event_dedupe_id": "msg_x"}],
        },
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_cross_tenant_user_id_returns_401(client):
    """Container belongs to user A; envelope says user B. This is a
    platform routing bug — agent must refuse rather than write to the
    wrong tenant's table. Documented invariant in the endpoint
    docstring; without this assertion a regression silently
    cross-contaminates trigger_events rows."""
    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": CONTAINER_KEY},
        json={
            "trigger_kind": "email_received",
            "user_id": "different-user-uuid",
            "events": [{"event_dedupe_id": "msg_x"}],
        },
    )
    assert r.status_code == 401


# ── Validation ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_kind_returns_422(client):
    """trigger_kind outside TRIGGER_KINDS is a Pydantic validation
    error — 422. Future kinds added to the frozenset enable themselves
    via the same path."""
    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": CONTAINER_KEY},
        json={
            "trigger_kind": "totally_made_up_kind",
            "user_id": CONTAINER_USER_ID,
            "events": [{"event_dedupe_id": "msg_x"}],
        },
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_empty_event_list_returns_422(client):
    """An empty events list is malformed (platform should never send
    one); reject at the Pydantic layer rather than handling silently."""
    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": CONTAINER_KEY},
        json={
            "trigger_kind": "email_received",
            "user_id": CONTAINER_USER_ID,
            "events": [],
        },
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_blank_dedupe_id_returns_422(client):
    """An all-whitespace dedupe id is unusable as a UNIQUE key. Pydantic
    rejects rather than silently slipping a blank into the constraint."""
    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": CONTAINER_KEY},
        json={
            "trigger_kind": "email_received",
            "user_id": CONTAINER_USER_ID,
            "events": [{"event_dedupe_id": "   "}],
        },
    )
    assert r.status_code == 422


# ── Happy path ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_event_inserts_row(client, trigger_id):
    """The basic happy-path: one event, one trigger, one row inserted.
    Response shape pinned so downstream platform code can rely on it."""
    from app.db import async_session_maker
    from app.db.models import TriggerEvent

    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": CONTAINER_KEY},
        json={
            "trigger_kind": "email_received",
            "user_id": CONTAINER_USER_ID,
            "events": [{"event_dedupe_id": "gmail_001"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted_events"] == 1
    assert body["inserted"] == 1
    assert body["dedupe_hits"] == 0
    assert body["no_trigger"] == 0
    assert body["results"][0]["outcome"] == "inserted"
    assert body["results"][0]["trigger_id"] == trigger_id

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(TriggerEvent).where(TriggerEvent.trigger_id == trigger_id)
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].event_dedupe_id == "gmail_001"
    assert rows[0].status == "queued"


@pytest.mark.asyncio
async def test_pubsub_retry_returns_dedupe_hit(client, trigger_id):
    """The load-bearing invariant. Pub/Sub retries the same message,
    we MUST NOT double-insert. First call inserts, second call returns
    dedupe_hit with zero new rows."""
    from app.db import async_session_maker
    from app.db.models import TriggerEvent

    payload = {
        "trigger_kind": "email_received",
        "user_id": CONTAINER_USER_ID,
        "events": [{"event_dedupe_id": "gmail_retry_test"}],
    }
    headers = {"X-Agent-Key": CONTAINER_KEY}

    r1 = await client.post("/api/triggers/inbound", headers=headers, json=payload)
    assert r1.status_code == 200
    assert r1.json()["inserted"] == 1

    r2 = await client.post("/api/triggers/inbound", headers=headers, json=payload)
    assert r2.status_code == 200
    body = r2.json()
    assert body["inserted"] == 0
    assert body["dedupe_hits"] == 1
    assert body["results"][0]["outcome"] == "dedupe_hit"

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(TriggerEvent).where(
                TriggerEvent.trigger_id == trigger_id,
                TriggerEvent.event_dedupe_id == "gmail_retry_test",
            )
        )).scalars().all()
    assert len(rows) == 1, "Pub/Sub retry should NOT double-insert"


@pytest.mark.asyncio
async def test_batch_of_events_inserts_all(client, trigger_id):
    """Coalesced inbox arrival — one Pub/Sub dispatch carrying N events.
    Each event becomes one row (per trigger). Status code 200, totals
    accurate, results array has 1 entry per event."""
    events = [{"event_dedupe_id": f"gmail_batch_{i}"} for i in range(5)]
    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": CONTAINER_KEY},
        json={
            "trigger_kind": "email_received",
            "user_id": CONTAINER_USER_ID,
            "events": events,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["accepted_events"] == 5
    assert body["inserted"] == 5
    assert len(body["results"]) == 5


@pytest.mark.asyncio
async def test_disabled_trigger_returns_no_trigger(client):
    """Trigger exists but is disabled → no row inserted, outcome is
    no_trigger, status code remains 200 (the dispatch was valid; we
    just have nothing to do). 4xx here would make Pub/Sub retry
    needlessly."""
    from app.db import async_session_maker
    from app.db.models import Trigger

    tid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Trigger(
            id=tid,
            user_id=CONTAINER_USER_ID,
            kind="email_received",
            action="notify_only",
            enabled=False,
        ))
        await db.commit()

    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": CONTAINER_KEY},
        json={
            "trigger_kind": "email_received",
            "user_id": CONTAINER_USER_ID,
            "events": [{"event_dedupe_id": "gmail_disabled"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["inserted"] == 0
    assert body["no_trigger"] == 1
    assert body["results"][0]["outcome"] == "no_trigger"


@pytest.mark.asyncio
async def test_no_trigger_at_all_returns_no_trigger(client):
    """No trigger row exists for this user/kind — still 200 with
    no_trigger. (Race window: user just deleted their trigger but the
    platform watch hasn't been torn down yet.)"""
    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": CONTAINER_KEY},
        json={
            "trigger_kind": "email_received",
            "user_id": CONTAINER_USER_ID,
            "events": [{"event_dedupe_id": "gmail_no_trigger"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["inserted"] == 0
    assert body["no_trigger"] == 1


@pytest.mark.asyncio
async def test_multiple_triggers_fan_out_inserts(client, trigger_id):
    """Two enabled triggers of the same kind for the same user → one
    inbound dispatch becomes two rows (one per trigger). Lets a user
    have, e.g., two different filter configs against the same Gmail
    account."""
    from app.db import async_session_maker
    from app.db.models import Trigger, TriggerEvent

    tid2 = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Trigger(
            id=tid2,
            user_id=CONTAINER_USER_ID,
            kind="email_received",
            action="notify_only",
            enabled=True,
        ))
        await db.commit()

    r = await client.post(
        "/api/triggers/inbound",
        headers={"X-Agent-Key": CONTAINER_KEY},
        json={
            "trigger_kind": "email_received",
            "user_id": CONTAINER_USER_ID,
            "events": [{"event_dedupe_id": "gmail_fanout"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["inserted"] == 2  # one per trigger
    assert body["accepted_events"] == 1

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(TriggerEvent).where(
                TriggerEvent.event_dedupe_id == "gmail_fanout",
            )
        )).scalars().all()
    trigger_ids_seen = {r.trigger_id for r in rows}
    assert trigger_ids_seen == {trigger_id, tid2}
