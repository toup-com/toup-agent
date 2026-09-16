"""A dead database is not an empty list.

2026-09-15, 15:08:19.140Z, container toup-agent-pool-38:

    INFO: 172.17.0.1:36212 - "GET /api/routines HTTP/1.1" 200 OK

immediately preceded by `list_routines: catastrophic failure (returning [])`.
The host's pgbouncer had been dead since 15:08:13; every DB access in that
container was raising `ConnectionRefusedError`, and it stayed that way until
15:19:33Z. For eleven minutes a user looking at their automations was told,
with a 200, that they had none.

The rule (already law on the app side for playlists: an unreachable agent
returns 503, never an empty list) has two halves and this file pins both:

  * an INFRASTRUCTURE failure — the store is unreachable — is a 503 carrying
    `code="backend_unavailable"`, because the honest answer to "what do I
    have?" is "I cannot tell you", never "nothing";
  * a PER-ROW failure still degrades. One routine whose columns have drifted
    must not take the panel down; that inner handler is untouched.

Listed in tests/COVERAGE_DEBT.txt as `# agent-mode`: `routines` and
`build_jobs` are AGENT_ONLY, so `init_db()` does not create them under
RUN_MODE=platform.

Run:
    cd backend && RUN_MODE=agent PYTHONPATH=. \
        pytest tests/test_routines_list_honesty.py -q
"""
from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError

pytestmark = pytest.mark.asyncio


async def _seed(user_id: str, *, name: str = "Morning brief") -> str:
    from app.db import async_session_maker
    from app.db.models import Routine, User

    rid = str(uuid.uuid4())
    async with async_session_maker() as db:
        if not await db.get(User, user_id):
            db.add(User(id=user_id, email=f"{user_id}@honesty.test",
                        hashed_password="x", name="Owner", timezone="UTC"))
        db.add(Routine(
            id=rid, user_id=user_id, kind="agent_task", name=name,
            enabled=True, schedule_kind="cron", schedule_cron_local="0 8 * * *",
            prompt_text="say hello", created_at=datetime.utcnow(),
        ))
        await db.commit()
    return rid


@pytest.fixture()
def owner(monkeypatch):
    from app.config import settings
    uid = str(uuid.uuid4())
    monkeypatch.setattr(settings, "user_id", uid, raising=False)
    return uid


def _broken_session_maker(exc: BaseException):
    """Stand in for `async_session_maker` while the pooler is dead: the raise
    happens on the FIRST touch, exactly as asyncpg's connect did."""
    class _Maker:
        def __call__(self, *a, **k):
            return self

        async def __aenter__(self):
            raise exc

        async def __aexit__(self, *a):
            return False
    return _Maker()


# ── the honest empty ─────────────────────────────────────────────


async def test_a_reachable_database_still_lists_routines(owner):
    """Anti-vacuity: if this returned [] the 503 tests below would prove
    nothing, because everything would be 'empty' either way."""
    from app.api.routines import list_routines

    rid = await _seed(owner)
    rows = await list_routines()
    assert [r.id for r in rows] == [rid]


async def test_a_user_with_no_routines_still_gets_an_empty_list(owner):
    from app.api.routines import list_routines

    assert await list_routines() == []


# ── the lie ──────────────────────────────────────────────────────


@pytest.mark.parametrize("exc", [
    ConnectionRefusedError(111, "Connection refused"),
    OperationalError("SELECT 1", {}, Exception("connection was closed")),
])
async def test_an_unreachable_database_is_503_not_an_empty_list(owner, monkeypatch, exc):
    import app.db.database as _db
    from app.api.routines import list_routines

    monkeypatch.setattr(_db, "async_session_maker", _broken_session_maker(exc))
    with pytest.raises(HTTPException) as ei:
        await list_routines()
    assert ei.value.status_code == 503
    assert ei.value.detail["code"] == "backend_unavailable"


async def test_the_503_body_names_a_code_a_client_can_branch_on(owner, monkeypatch):
    """The app's error copy prefers a CODE over a raw string — without one it
    fell to its transport regex and told the user to check their internet
    during a server-side outage."""
    import app.db.database as _db
    from app.api.routines import list_routines

    monkeypatch.setattr(
        _db, "async_session_maker",
        _broken_session_maker(ConnectionRefusedError(111, "Connection refused")),
    )
    with pytest.raises(HTTPException) as ei:
        await list_routines()
    assert isinstance(ei.value.detail, dict)
    assert set(ei.value.detail) >= {"code", "message"}
    assert "internet" not in ei.value.detail["message"].lower()


# ── what must NOT become a 503 ───────────────────────────────────


async def test_schema_drift_still_degrades_to_an_empty_list(owner, monkeypatch):
    """One stale column must not take a tenant's whole panel offline — that
    degrade is the reason the broad handler exists and it stays."""
    import app.db.database as _db
    from app.api.routines import list_routines

    monkeypatch.setattr(
        _db, "async_session_maker",
        _broken_session_maker(AttributeError("Routine has no attribute 'phase'")),
    )
    assert await list_routines() == []


async def test_one_unrenderable_row_does_not_take_the_list_down(owner, monkeypatch):
    import app.api.routines as mod

    good = await _seed(owner, name="Good")
    bad = await _seed(owner, name="Bad")
    real = mod._row_to_response

    def _flaky(routine, recent_runs=(), recent_jobs=()):
        if routine.id == bad:
            raise ValueError("mig 042 columns not backfilled")
        return real(routine, recent_runs, recent_jobs)

    monkeypatch.setattr(mod, "_row_to_response", _flaky)
    rows = await mod.list_routines()
    assert [r.id for r in rows] == [good]
