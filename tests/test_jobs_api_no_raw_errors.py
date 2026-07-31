"""The jobs API must never ship internals to a client.

The audited path was five hops with zero mapping: `repr(exc)` →
`build_jobs.error_message` → `JobResponse.error_message` → byte-for-byte
platform proxy → `Job.errorMessage` → a React Native `<Text>`. Users read
402 JSON blobs and `AttributeError("'BuildJob' object has no attribute
'event_dedupe_id'")`.

Every string below is verbatim from the founder's production tenant
(2026-07-29). These tests are the contract that keeps them off screen.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient

PROD_402 = (
    "Error code: 402 - {'detail': {'error': 'out_of_credits', 'reason': "
    "'insufficient_message_credits', 'bucket': 'message', 'balance_after': '0.00'}}"
)
PROD_UPSTREAM = (
    'BadRequestError: Error code: 400 - {\'detail\': \'{"type":"error","error":'
    '{"type":"invalid_request_error","message":"Your credit balance is too low"}}\'}'
)
PROD_ATTR = (
    "all_retries_exhausted: AttributeError(\"'BuildJob' object has no "
    "attribute 'event_dedupe_id'\")"
)
PROD_RESTART = "Agent restarted during execution"

#: Substrings that must never appear in any client-facing string.
BANNED = (
    "Error code:", "Traceback", "AttributeError", "BadRequestError",
    "{'", '{"', "event_dedupe_id", "balance_after", "insufficient_message_credits",
    "restarted during execution",
)


async def _mk_user() -> str:
    from app.db import User, async_session_maker

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=f"{uid[:8]}@t.test", name="T",
            hashed_password="x", role="beta_user",
        ))
        await db.commit()
    return uid


async def _mk_job(user_id: str, **kw) -> str:
    from app.db import async_session_maker
    from app.db.models import BuildJob

    jid = str(uuid.uuid4())
    payload = dict(
        id=jid, user_id=user_id, title=kw.pop("title", "a task"), prompt="p",
        job_type="agent_task", status="failed", created_at=datetime.utcnow(),
    )
    payload.update(kw)
    async with async_session_maker() as db:
        db.add(BuildJob(**payload))
        await db.commit()
    return jid


def _client():
    from fastapi import FastAPI
    from app.api.apps import router as apps_router

    app = FastAPI()
    app.include_router(apps_router, prefix="/api")
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://agent")


@pytest.mark.parametrize("raw", [PROD_402, PROD_UPSTREAM, PROD_ATTR, PROD_RESTART])
@pytest.mark.asyncio
async def test_legacy_raw_error_is_humanized_on_read(monkeypatch, raw):
    """No data migration required: rows written before the taxonomy are
    classified at serialization time, so the leak stops the moment this
    deploys — including on 79-day-old corpses."""
    from app.config import settings

    uid = await _mk_user()
    monkeypatch.setattr(settings, "user_id", uid)
    jid = await _mk_job(uid, error_message=raw)

    async with _client() as ac:
        row = next(j for j in (await ac.get("/api/apps/jobs/")).json() if j["id"] == jid)

    assert row["error_class"], "every failure must carry a taxonomy class"
    msg = row["user_message"]
    if msg is not None:
        for token in BANNED:
            assert token not in msg, f"{token!r} leaked into user_message"


@pytest.mark.asyncio
async def test_technical_detail_is_never_serialized(monkeypatch):
    """`technical_detail` is internal-only telemetry. If it ever appears in
    a response the whole taxonomy is pointless — the raw text is back."""
    from app.config import settings

    uid = await _mk_user()
    monkeypatch.setattr(settings, "user_id", uid)
    jid = await _mk_job(
        uid, error_message=PROD_402, error_class="credits_toup",
        user_message="Your agent ran out of Toup credits partway through this task.",
        technical_detail=PROD_402,
    )

    async with _client() as ac:
        listed = next(j for j in (await ac.get("/api/apps/jobs/")).json() if j["id"] == jid)
        detail = (await ac.get(f"/api/apps/jobs/{jid}")).json()

    for payload in (listed, detail):
        assert "technical_detail" not in payload
        blob = json.dumps(payload)
        assert "event_dedupe_id" not in blob
        # error_message is still present for legacy clients, but the
        # taxonomy fields must be populated so new clients ignore it.
        assert payload["error_class"] == "credits_toup"
        assert "credits" in payload["user_message"]


@pytest.mark.asyncio
async def test_stored_taxonomy_wins_over_read_time_classification(monkeypatch):
    """A writer that already classified must not be second-guessed."""
    from app.config import settings

    uid = await _mk_user()
    monkeypatch.setattr(settings, "user_id", uid)
    jid = await _mk_job(
        uid, error_message=PROD_402,
        error_class="connector_auth", user_message="Reconnect Gmail to continue.",
    )

    async with _client() as ac:
        row = (await ac.get(f"/api/apps/jobs/{jid}")).json()

    assert row["error_class"] == "connector_auth"
    assert row["user_message"] == "Reconnect Gmail to continue."


@pytest.mark.asyncio
async def test_successful_job_carries_no_error_fields(monkeypatch):
    from app.config import settings

    uid = await _mk_user()
    monkeypatch.setattr(settings, "user_id", uid)
    jid = await _mk_job(uid, status="completed", error_message=None)

    async with _client() as ac:
        row = (await ac.get(f"/api/apps/jobs/{jid}")).json()

    assert row["error_class"] is None
    assert row["user_message"] is None


# ── archive (soft retirement) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_archive_hides_without_deleting(monkeypatch):
    """History is never destroyed. Archiving stamps `archived_at`; the row
    stays fetchable by id so a deep link still resolves."""
    from app.config import settings

    uid = await _mk_user()
    monkeypatch.setattr(settings, "user_id", uid)
    jid = await _mk_job(uid, status="completed")

    async with _client() as ac:
        r = await ac.patch(f"/api/apps/jobs/{jid}", json={"archived": True})
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        # Still retrievable directly — archive is not delete.
        assert (await ac.get(f"/api/apps/jobs/{jid}")).status_code == 200

        # And reversible.
        un = await ac.patch(f"/api/apps/jobs/{jid}", json={"archived": False})
        assert un.json()["archived_at"] is None


@pytest.mark.asyncio
async def test_patch_accepts_new_statuses(monkeypatch):
    """SYMPTOM: 400 on `waiting_on_user`.

    The old validator hard-coded a 4-value tuple, so it rejected
    `waiting_on_user` and the `cancelled`/`timeout`/`budget_exhausted` the
    sub-agent arc already writes.
    """
    from app.config import settings

    uid = await _mk_user()
    monkeypatch.setattr(settings, "user_id", uid)
    jid = await _mk_job(uid, status="running")

    async with _client() as ac:
        for status in ("waiting_on_user", "cancelled", "timeout", "budget_exhausted"):
            r = await ac.patch(f"/api/apps/jobs/{jid}", json={"status": status})
            assert r.status_code == 200, f"{status} rejected: {r.text}"
            assert r.json()["status"] == status

        bad = await ac.patch(f"/api/apps/jobs/{jid}", json={"status": "nonsense"})
        assert bad.status_code == 400

        empty = await ac.patch(f"/api/apps/jobs/{jid}", json={})
        assert empty.status_code == 400, "a no-op PATCH should be rejected"
