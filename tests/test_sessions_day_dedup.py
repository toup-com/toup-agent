"""Regression tests for the /api/sessions IntegrityError landmine and the
conversation resolver's savepoint semantics (SOTA assessment, unit W1.6).

  * ``SessionCreate.channel`` DEFAULTS to ``"api"`` (schemas.py), which sits
    inside the partial unique index ``ix_conversations_system_channel_per_day``.
    The handler stamps a real ``day_chat_id`` and used to blind-insert a
    Conversation — so a user's 2nd default-channel create in one local day
    500'd with an IntegrityError (exact bug class 543739ab fixed in the
    runner). It now routes system channels through
    ``resolve_or_create_day_conversation``.

  * The runner's blind Conversation insert was still reachable with
    ``force_new=True`` + a client-supplied indexed system channel over
    /ws/chat (both fields are client-controlled). ``force_new`` is now a
    deliberate no-op for indexed system channels.

  * The resolver's race-recovery arm said "savepoint" but called
    ``db.rollback()`` — a FULL transaction rollback that discarded the
    caller's uncommitted work. The INSERT attempt is now wrapped in
    ``begin_nested()`` so only the savepoint rolls back.
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select


@pytest_asyncio.fixture
async def sessions_client():
    """Client against an app with the sessions router mounted (the shared
    conftest app doesn't include it)."""
    from app.api.sessions import router as sessions_router
    from app.config import settings

    app = FastAPI()
    app.include_router(sessions_router, prefix=settings.api_prefix)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def seeded_user_and_day():
    """Create a User row + DayChat row for today. Returns ids.

    Mirrors the fixture in test_conversation_resolver.py.
    """
    from app.db import User, async_session_maker
    from app.db.models import DayChat

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"w16-test-{user_id[:8]}@example.com",
            hashed_password="x",
            name="W1.6 Test",
            timezone="UTC",
        ))
        day = DayChat(
            id=str(uuid.uuid4()),
            user_id=user_id,
            local_date=date.today(),
            timezone="UTC",
        )
        db.add(day)
        await db.commit()
        return {"user_id": user_id, "day_chat_id": day.id}


class TestSessionsCreateDayDedup:
    """POST /api/sessions must not 500 on the 2nd same-day default create."""

    @pytest.mark.asyncio
    async def test_second_default_channel_create_reuses_day_thread(
        self, sessions_client, auth_headers, test_user_id
    ):
        """Mirror of test_cross_channel_continuity's raw-index collision for
        the sessions.py path: the 2nd default-channel ('api') create of the
        same local day must return the SAME conversation, not an
        IntegrityError 500."""
        r1 = await sessions_client.post("/api/sessions", json={}, headers=auth_headers)
        assert r1.status_code == 201, r1.text
        assert r1.json()["channel"] == "api"
        # day_chat_id must actually be stamped — a NULL would evade the
        # partial unique index and make this test vacuous.
        from app.db import async_session_maker
        from app.db.models import Conversation
        async with async_session_maker() as db:
            first = (await db.execute(
                select(Conversation).where(Conversation.id == r1.json()["id"])
            )).scalar_one()
            assert first.day_chat_id is not None

        r2 = await sessions_client.post("/api/sessions", json={}, headers=auth_headers)
        assert r2.status_code == 201, r2.text  # pre-fix: IntegrityError → 500
        assert r2.json()["id"] == r1.json()["id"]

        async with async_session_maker() as db:
            convs = (await db.execute(
                select(Conversation).where(
                    Conversation.user_id == test_user_id,
                    Conversation.channel == "api",
                    Conversation.is_active.is_(True),
                )
            )).scalars().all()
        assert len(convs) == 1

    @pytest.mark.asyncio
    async def test_non_system_channel_still_creates_new_each_time(
        self, sessions_client, auth_headers
    ):
        """User-driven channels keep their pre-fix semantics: every create
        is a fresh Conversation."""
        r1 = await sessions_client.post(
            "/api/sessions", json={"channel": "web"}, headers=auth_headers
        )
        r2 = await sessions_client.post(
            "/api/sessions", json={"channel": "web"}, headers=auth_headers
        )
        assert r1.status_code == 201 and r2.status_code == 201
        assert r1.json()["id"] != r2.json()["id"]


class TestResolverSavepoint:
    """The race-recovery arm must roll back ONLY the failed INSERT."""

    @pytest.mark.asyncio
    async def test_race_arm_preserves_callers_pending_work(
        self, seeded_user_and_day, monkeypatch
    ):
        """Winner row committed; the resolver's dedup SELECT is patched to
        miss so the INSERT collides (deterministic stand-in for the race
        window — StaticPool sqlite shares one connection, so a true
        concurrent writer can't be simulated). The caller's flushed-but-
        uncommitted work must survive the IntegrityError recovery.
        Pre-fix, db.rollback() silently discarded it.

        The pending work is a 'web' Conversation (not a Message — the
        messages table needs pgvector, so sqlite runs skip creating it)."""
        import app.agent.conversation_resolver as cr
        from sqlalchemy import false as sa_false
        from app.db import async_session_maker
        from app.db.models import Conversation

        uid = seeded_user_and_day["user_id"]
        day_id = seeded_user_and_day["day_chat_id"]

        # init_db skips non-ALTER statements on sqlite, so install the same
        # partial unique index database.py declares — the collision under
        # test is impossible without it.
        from sqlalchemy import text
        from app.db.database import engine
        if engine.dialect.name == "sqlite":
            async with engine.begin() as conn:
                await conn.execute(text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "ix_conversations_system_channel_per_day "
                    "ON conversations (user_id, day_chat_id, channel) "
                    "WHERE channel IN ('routine','trigger','api','digest') "
                    "AND is_active = 1"
                ))

        # The row a concurrent fire already committed.
        winner_id = str(uuid.uuid4())
        async with async_session_maker() as db:
            db.add(Conversation(
                id=winner_id,
                user_id=uid,
                channel="api",
                day_chat_id=day_id,
                is_active=True,
            ))
            await db.commit()

        # First select (the dedup check) misses; every later select —
        # including the winner re-fetch in the race arm — is untouched.
        real_select = cr.select
        calls = {"n": 0}

        def racey_select(*args, **kwargs):
            calls["n"] += 1
            stmt = real_select(*args, **kwargs)
            if calls["n"] == 1:
                stmt = stmt.where(sa_false())
            return stmt

        monkeypatch.setattr(cr, "select", racey_select)

        pending_id = str(uuid.uuid4())
        async with async_session_maker() as db:
            # Caller's uncommitted work, pending in the same transaction.
            db.add(Conversation(
                id=pending_id,
                user_id=uid,
                channel="web",  # user channel — outside the unique index
                day_chat_id=day_id,
                is_active=True,
            ))
            await db.flush()

            conv = await cr.resolve_or_create_day_conversation(
                db, user_id=uid, day_chat_id=day_id, channel="api",
            )
            assert conv.id == winner_id  # race arm re-fetched the winner
            await db.commit()

        assert calls["n"] >= 2, "race arm never ran — test setup is broken"

        async with async_session_maker() as db:
            row = (await db.execute(
                select(Conversation).where(Conversation.id == pending_id)
            )).scalar_one_or_none()
        assert row is not None, (
            "caller's pending work was discarded — the resolver did a full "
            "transaction rollback instead of a savepoint rollback"
        )


# ---------------------------------------------------------------------------
# Wiring pins (source-grep style — see test_cross_channel_continuity.py).
# ---------------------------------------------------------------------------

_BACKEND = Path(__file__).resolve().parent.parent
_RUNNER = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
_SESSIONS = (_BACKEND / "app" / "api" / "sessions.py").read_text()
_RESOLVER = (_BACKEND / "app" / "agent" / "conversation_resolver.py").read_text()


class TestWiringPins:
    def test_sessions_create_routes_system_channels_through_resolver(self):
        seg = _SESSIONS[_SESSIONS.index("async def create_session("):]
        seg = seg[:seg.index("async def list_sessions(")]
        assert "resolve_or_create_day_conversation" in seg
        # the resolver return must appear before the blind insert
        assert seg.index("return _session_to_response(session)") < seg.index(
            "session = Conversation("
        )

    def test_runner_force_new_is_noop_for_indexed_system_channels(self):
        """force_new + client-supplied system channel over /ws/chat must not
        reach the blind insert — the guard no longer honors force_new."""
        guard = "if _channel in _INDEXED_SYSTEM_CHANNELS and _day_chat_id is not None:"
        assert guard in _RUNNER
        assert (
            "if _channel in _INDEXED_SYSTEM_CHANNELS and _day_chat_id is not None "
            "and not force_new:"
        ) not in _RUNNER

    def test_resolver_race_arm_uses_savepoint_not_full_rollback(self):
        assert "async with db.begin_nested():" in _RESOLVER
        assert "await db.rollback()" not in _RESOLVER
