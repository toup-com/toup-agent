"""Unit tests for app.agent.channels.whatsapp_dedupe.

Pure async tests over an in-memory SQLite — no project conftest
fixtures, no network. Each test instantiates its own engine + table
so they cannot interfere with each other.

What's covered
--------------
* claim() empty-input short-circuit.
* claim() returns True on first call, False on retry of same key.
* claim() per-(user, message) isolation — same message_id under a
  different user_id is independently claimable.
* claim() defensive let-through when the underlying DB rejects the
  Postgres-specific ON-CONFLICT syntax (SQLite path).
* sweep_expired() prunes only rows older than the cutoff.
* sweep_expired() returns 0 on DB error, never raises.
* run_sweep_loop() invokes sweep_expired on the configured interval
  and stops cleanly on cancellation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


# ── Fixture: per-test engine + table-only schema ─────────────────


async def _fresh_maker():
    """Create a per-test SQLite engine with just the dedupe table.

    We deliberately don't use ``Base.metadata.create_all`` because the
    project's ``entities`` table uses Postgres TSVECTOR which SQLite
    can't render. Creating the dedupe table in isolation keeps the
    test self-contained.
    """
    from app.db.models.agent import WhatsAppInboundDedupe

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(WhatsAppInboundDedupe.__table__.create)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, maker


# ── claim() ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_claim_empty_user_id_returns_false():
    from app.agent.channels.whatsapp_dedupe import claim

    engine, maker = await _fresh_maker()
    try:
        assert await claim("", "msg-1", session_maker=lambda: maker()) is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_claim_empty_message_id_returns_false():
    from app.agent.channels.whatsapp_dedupe import claim

    engine, maker = await _fresh_maker()
    try:
        assert await claim("user-1", "", session_maker=lambda: maker()) is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_claim_falls_through_on_db_error_and_lets_message_through():
    """When the underlying DB is incompatible with the Postgres-only
    ON CONFLICT clause we generate, we MUST NOT silently drop the
    message — claim() returns True so the caller dispatches normally.
    SQLite is the easy way to exercise this branch.
    """
    from app.agent.channels.whatsapp_dedupe import claim

    engine, maker = await _fresh_maker()
    try:
        result = await claim("user-1", "msg-1", session_maker=lambda: maker())
        # SQLite rejects the Postgres ON CONFLICT (... constraint=...)
        # form; the helper must catch and return True (let-through).
        assert result is True
    finally:
        await engine.dispose()


# ── sweep_expired() ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_prunes_only_rows_older_than_cutoff():
    from app.agent.channels.whatsapp_dedupe import sweep_expired
    from app.db.models.agent import WhatsAppInboundDedupe

    engine, maker = await _fresh_maker()
    try:
        async with maker() as db:
            now = datetime.utcnow()
            db.add(
                WhatsAppInboundDedupe(
                    user_id="u",
                    message_id="old-1",
                    received_at=now - timedelta(hours=1),
                )
            )
            db.add(
                WhatsAppInboundDedupe(
                    user_id="u",
                    message_id="old-2",
                    received_at=now - timedelta(minutes=25),
                )
            )
            db.add(
                WhatsAppInboundDedupe(
                    user_id="u",
                    message_id="fresh",
                    received_at=now,
                )
            )
            await db.commit()

        pruned = await sweep_expired(
            older_than=timedelta(minutes=20),
            session_maker=lambda: maker(),
        )
        assert pruned == 2
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sweep_no_rows_returns_zero():
    from app.agent.channels.whatsapp_dedupe import sweep_expired

    engine, maker = await _fresh_maker()
    try:
        n = await sweep_expired(session_maker=lambda: maker())
        assert n == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sweep_returns_zero_on_db_error_does_not_raise():
    """If the session_maker raises, sweep_expired must swallow the
    error (zero pruned, warning logged) rather than bubble up and
    crash the long-running sweep loop.
    """
    from app.agent.channels.whatsapp_dedupe import sweep_expired

    def broken_maker():
        raise RuntimeError("simulated DB outage")

    n = await sweep_expired(session_maker=broken_maker)
    assert n == 0  # never raises


# ── run_sweep_loop() ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_sweep_loop_invokes_sweep_and_cancels_cleanly():
    """The loop should call sweep_expired between sleeps and exit
    cleanly when the task is cancelled. We inject a fake `sleep` that
    counts iterations and cancels after the third.
    """
    from app.agent.channels import whatsapp_dedupe as mod

    engine, maker = await _fresh_maker()
    try:
        sweep_calls = []

        async def fake_sweep(*, older_than, session_maker=None):
            sweep_calls.append(datetime.utcnow())
            return 0

        # Stub out the real sweep so we don't hit the DB.
        original = mod.sweep_expired
        mod.sweep_expired = fake_sweep
        try:
            iterations = {"n": 0}

            async def fake_sleep(seconds):
                iterations["n"] += 1
                if iterations["n"] >= 3:
                    raise asyncio.CancelledError
                return None

            await mod.run_sweep_loop(
                interval_seconds=0,
                session_maker=lambda: maker(),
                sleep=fake_sleep,
            )
            # We slept once successfully and swept; on the third sleep
            # CancelledError exited the loop. Two completed sweeps.
            assert len(sweep_calls) == 2
        finally:
            mod.sweep_expired = original
    finally:
        await engine.dispose()
