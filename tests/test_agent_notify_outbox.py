"""Agent notify outbox + deliver endpoint (Autopilot PR4).

The durability contract under test: a notification survives lost acks
and container restarts because the outbox row is only marked flushed
on a platform ack (queued/duplicate), and the row id is the platform
idempotency key so replays are no-ops.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.models import AgentNotifyOutbox
from app.services import agent_notify_client as anc


@pytest.fixture(autouse=True)
def _configured_agent(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "platform_api_url", "https://toup.ai")
    monkeypatch.setattr(settings, "agent_api_key", "test-agent-key")
    monkeypatch.setattr(settings, "user_id", str(uuid.uuid4()))
    # Deterministic flushes: only the explicit calls in each test may
    # consume the scripted HTTP responses.
    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)
    yield


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self._body = body or {}
        self.text = str(body)

    def json(self):
        return self._body


def _fake_http(responses: list):
    """AsyncClient replacement yielding scripted responses; records posts."""
    calls: list[dict] = []

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            calls.append({"url": url, "json": json, "headers": headers})
            if not responses:
                raise ConnectionError("no scripted response")
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

    return _FakeClient, calls


async def _outbox_row(row_id: str) -> AgentNotifyOutbox:
    from app.db import async_session_maker

    async with async_session_maker() as db:
        row = await db.get(AgentNotifyOutbox, row_id)
        db.expunge(row)
        return row


@pytest.mark.asyncio
async def test_notify_persists_and_flush_acks(monkeypatch):
    fake, calls = _fake_http([_FakeResponse(200, {"status": "queued", "id": "q1"})])
    monkeypatch.setattr(anc.httpx, "AsyncClient", fake)

    row_id = await anc.notify(
        event_kind="mission_completed",
        title="Done",
        body="finished",
        data={"route": "mission-control"},
        priority="default",
        dedup_key="m1:done",
    )
    assert row_id

    stats = await anc.flush_notify_outbox()
    assert stats["acked"] >= 1
    row = await _outbox_row(row_id)
    assert row.flushed_at is not None and row.last_error is None

    # The POST used the row id as the platform idempotency key and
    # normalized the /api suffix.
    sent = calls[-1]
    assert sent["url"] == "https://toup.ai/api/agent/notify"
    assert sent["json"]["idempotency_key"] == row_id
    assert sent["headers"]["X-Agent-Key"] == "test-agent-key"


@pytest.mark.asyncio
async def test_lost_ack_replays_and_duplicate_acks(monkeypatch):
    # First flush: network dies (ack lost). Second: platform says
    # duplicate (it actually got the first one) — row must resolve.
    fake, calls = _fake_http([
        ConnectionError("boom"),
        _FakeResponse(200, {"status": "duplicate", "id": "q1"}),
    ])
    monkeypatch.setattr(anc.httpx, "AsyncClient", fake)

    row_id = await anc.notify(event_kind="needs_input", title="Q", priority="high")

    stats = await anc.flush_notify_outbox()
    assert stats["retried"] == 1
    row = await _outbox_row(row_id)
    assert row.flushed_at is None and row.attempts == 1
    assert row.next_attempt_at is not None

    # Force the backoff cursor into the past, then re-flush.
    from app.db import async_session_maker
    from sqlalchemy import update
    async with async_session_maker() as db:
        await db.execute(
            update(AgentNotifyOutbox)
            .where(AgentNotifyOutbox.id == row_id)
            .values(next_attempt_at=datetime.utcnow() - timedelta(seconds=1))
        )
        await db.commit()

    stats = await anc.flush_notify_outbox()
    assert stats["acked"] == 1
    row = await _outbox_row(row_id)
    assert row.flushed_at is not None
    # Both attempts carried the SAME idempotency key.
    assert calls[0]["json"]["idempotency_key"] == calls[1]["json"]["idempotency_key"]


@pytest.mark.asyncio
async def test_permanent_rejection_does_not_wedge_queue(monkeypatch):
    fake, _ = _fake_http([_FakeResponse(422, {"detail": "bad kind"})])
    monkeypatch.setattr(anc.httpx, "AsyncClient", fake)

    row_id = await anc.notify(event_kind="mission_failed", title="X")
    await anc.flush_notify_outbox()
    row = await _outbox_row(row_id)
    assert row.flushed_at is not None
    assert row.last_error.startswith("rejected")


@pytest.mark.asyncio
async def test_stale_rows_expire(monkeypatch):
    fake, calls = _fake_http([])
    monkeypatch.setattr(anc.httpx, "AsyncClient", fake)

    row_id = await anc.notify(event_kind="digest", title="Old")
    from app.db import async_session_maker
    from sqlalchemy import update
    async with async_session_maker() as db:
        await db.execute(
            update(AgentNotifyOutbox)
            .where(AgentNotifyOutbox.id == row_id)
            .values(created_at=datetime.utcnow() - timedelta(hours=25))
        )
        await db.commit()

    stats = await anc.flush_notify_outbox()
    assert stats["expired"] == 1
    assert calls == []  # never POSTed
    row = await _outbox_row(row_id)
    assert row.last_error == "expired"


@pytest.mark.asyncio
async def test_unconfigured_container_noops(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "platform_api_url", "")
    stats = await anc.flush_notify_outbox()
    assert stats == {"skipped_unconfigured": 1}


# ── /notify/deliver (agent-side endpoint) ─────────────────────────


def _deliver_app():
    from fastapi import FastAPI
    from app.api.notify_deliver import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


@pytest.mark.asyncio
async def test_deliver_rejects_bad_key():
    transport = ASGITransport(app=_deliver_app())
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        res = await ac.post(
            "/api/notify/deliver",
            json={"event_kind": "needs_input", "title": "Q"},
            headers={"X-Agent-Key": "wrong"},
        )
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_deliver_fans_out_and_reports_delivered(monkeypatch):
    async def fake_deliver(**kwargs):
        assert kwargs["delivery_channels"] == ["telegram", "whatsapp"]
        return {
            "telegram": {"status": "delivered", "telegram_message_id": 5},
            "whatsapp": {"status": "skipped", "reason": "no_recipient"},
        }

    import app.agent.routines.channel_dispatcher as cd
    monkeypatch.setattr(cd, "deliver_to_extra_channels_detailed", fake_deliver)

    transport = ASGITransport(app=_deliver_app())
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        res = await ac.post(
            "/api/notify/deliver",
            json={
                "event_kind": "needs_approval",
                "title": "Approve?",
                "body": "Send the email to Sam?",
            },
            headers={"X-Agent-Key": "test-agent-key"},
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["delivered"] is True
    assert body["channels"]["telegram"]["status"] == "delivered"


@pytest.mark.asyncio
async def test_deliver_no_recipient_reports_not_delivered(monkeypatch):
    async def fake_deliver(**kwargs):
        return {
            "telegram": {"status": "skipped", "reason": "no_recipient"},
            "whatsapp": {"status": "skipped", "reason": "no_recipient"},
        }

    import app.agent.routines.channel_dispatcher as cd
    monkeypatch.setattr(cd, "deliver_to_extra_channels_detailed", fake_deliver)

    transport = ASGITransport(app=_deliver_app())
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        res = await ac.post(
            "/api/notify/deliver",
            json={"event_kind": "mission_completed", "title": "Done"},
            headers={"X-Agent-Key": "test-agent-key"},
        )
    assert res.status_code == 200
    assert res.json()["delivered"] is False
