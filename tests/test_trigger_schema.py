"""Schema invariant tests for the Triggers tables (Gate T1).

Covers the load-bearing invariants the runner + webhook will rely on
in Gate T2+:
  - Tables are correctly partitioned (AGENT_ONLY_TABLES).
  - Sensible defaults on insert (enabled=true, fire_count=0,
    last_status='never_fired', event.status='queued').
  - `UNIQUE (trigger_id, event_dedupe_id)` is the idempotency gate —
    a duplicate INSERT must raise IntegrityError. Pub/Sub retries
    rely on this; if it silently fails to dedupe, every retry runs
    the handler again.
  - `triggers.id` deletion cascades to `trigger_events.trigger_id`.
  - `coalesced_into_event_id` self-FK accepts null and a real event_id.

Pinned-value tests on TRIGGER_KINDS / TRIGGER_ACTIONS / TRIGGER_STATUSES
catch silent enum drift (someone deletes 'notify_only' from the set
without coordinating with the API validator / handlers).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError


# ── Helpers ──────────────────────────────────────────────────────────


async def _make_user() -> str:
    from app.db import async_session_maker
    from app.db.models import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            User(
                id=user_id,
                email=f"{user_id}@trigger-test.local",
                hashed_password="x",
                name="Trigger Test User",
            )
        )
        await db.commit()
    return user_id


async def _make_trigger(
    user_id: str,
    *,
    kind: str = "email_received",
    action: str = "summarize_and_post",
    enabled: bool = True,
    config: dict | None = None,
) -> str:
    from app.db import async_session_maker
    from app.db.models import Trigger

    tid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            Trigger(
                id=tid,
                user_id=user_id,
                kind=kind,
                action=action,
                enabled=enabled,
                config_json=config,
            )
        )
        await db.commit()
    return tid


# ── Partition + categorisation ───────────────────────────────────────


def test_triggers_tables_are_agent_only():
    """The whole point of the agent-side architecture: the platform must
    not write to these tables. The partition test in
    test_table_partitioning.py covers metadata-vs-set drift; this is the
    explicit, named assertion that triggers in particular live on the
    agent side."""
    from app.db.models.base import (
        AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES, SHARED_TABLES,
    )

    assert "triggers" in AGENT_ONLY_TABLES
    assert "trigger_events" in AGENT_ONLY_TABLES
    assert "triggers" not in PLATFORM_ONLY_TABLES
    assert "trigger_events" not in PLATFORM_ONLY_TABLES
    assert "triggers" not in SHARED_TABLES
    assert "trigger_events" not in SHARED_TABLES


def test_trigger_enum_constants_pinned():
    """Pin the canonical enum values. Removing or renaming one of these
    is a contract change that requires migrating existing rows + the API
    validator + every handler that maps to the value. The test fails
    loudly so the change can't slip through."""
    from app.db.models import (
        TRIGGER_KINDS, TRIGGER_ACTIONS, TRIGGER_STATUSES,
        TRIGGER_EVENT_STATUSES,
    )

    assert TRIGGER_KINDS == frozenset({"email_received"})
    assert TRIGGER_ACTIONS == frozenset({
        "summarize_and_post", "notify_only", "forward_to_telegram",
    })
    assert TRIGGER_STATUSES == frozenset({
        "never_fired", "active", "failed", "skipped_reauth",
        "provisioning_failed",
    })
    assert TRIGGER_EVENT_STATUSES == frozenset({
        "queued", "running", "success", "failed",
        "skipped_rate_limit", "skipped_filter", "coalesced",
    })


# ── Defaults ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trigger_defaults_match_spec():
    """A bare insert with only the NOT NULL columns must yield the
    documented defaults — enabled=True, fire_count=0,
    last_status='never_fired'. If a future migration changes a default
    silently the runner will fire on supposedly-disabled triggers."""
    from app.db import async_session_maker
    from app.db.models import Trigger

    user_id = await _make_user()
    tid = await _make_trigger(user_id)

    async with async_session_maker() as db:
        row = (await db.execute(select(Trigger).where(Trigger.id == tid))).scalar_one()
    assert row.enabled is True
    assert row.fire_count == 0
    assert row.last_status == "never_fired"
    assert row.last_error is None
    assert row.last_fired_at is None
    assert row.provider_state_json is None


@pytest.mark.asyncio
async def test_trigger_event_defaults_match_spec():
    """trigger_events.status must default to 'queued' and received_at
    must be auto-populated by the model default. The runner relies on
    'queued' as the discriminator for 'not yet claimed by a handler.'"""
    from app.db import async_session_maker
    from app.db.models import TriggerEvent

    user_id = await _make_user()
    tid = await _make_trigger(user_id)
    eid = str(uuid.uuid4())

    async with async_session_maker() as db:
        db.add(TriggerEvent(
            id=eid,
            trigger_id=tid,
            user_id=user_id,
            event_dedupe_id="gmail_msg_001",
        ))
        await db.commit()

    async with async_session_maker() as db:
        row = (await db.execute(select(TriggerEvent).where(TriggerEvent.id == eid))).scalar_one()
    assert row.status == "queued"
    assert row.received_at is not None
    assert row.started_at is None
    assert row.finished_at is None
    assert row.coalesced_into_event_id is None


# ── Idempotency — the production-critical invariant ──────────────────


@pytest.mark.asyncio
async def test_unique_trigger_id_event_dedupe_id_blocks_duplicate():
    """Pub/Sub WILL deliver duplicates — retries on 5xx, retries on
    missing-200-within-timeout. Two INSERTs with the same
    (trigger_id, event_dedupe_id) MUST collide on uq_trigger_events_dedupe.

    Without this gate, every retry runs the handler again and the user
    gets 4× summaries for one email. Production-critical."""
    from app.db import async_session_maker
    from app.db.models import TriggerEvent

    user_id = await _make_user()
    tid = await _make_trigger(user_id)

    # First insert — succeeds.
    async with async_session_maker() as db:
        db.add(TriggerEvent(
            id=str(uuid.uuid4()),
            trigger_id=tid,
            user_id=user_id,
            event_dedupe_id="gmail_msg_dup_test",
        ))
        await db.commit()

    # Second insert with same (trigger_id, event_dedupe_id) — must fail.
    with pytest.raises(IntegrityError):
        async with async_session_maker() as db:
            db.add(TriggerEvent(
                id=str(uuid.uuid4()),
                trigger_id=tid,
                user_id=user_id,
                event_dedupe_id="gmail_msg_dup_test",
            ))
            await db.commit()


@pytest.mark.asyncio
async def test_same_event_dedupe_id_allowed_for_different_triggers():
    """Two different triggers can each see the same external event id
    (e.g., user has two email_received triggers on the same Gmail
    account with different filters) — the dedupe is per-trigger, not
    global. Both must independently land a row."""
    from app.db import async_session_maker
    from app.db.models import TriggerEvent

    user_id = await _make_user()
    t1 = await _make_trigger(user_id)
    t2 = await _make_trigger(user_id)

    async with async_session_maker() as db:
        db.add(TriggerEvent(
            id=str(uuid.uuid4()),
            trigger_id=t1,
            user_id=user_id,
            event_dedupe_id="shared_msg_id",
        ))
        db.add(TriggerEvent(
            id=str(uuid.uuid4()),
            trigger_id=t2,
            user_id=user_id,
            event_dedupe_id="shared_msg_id",
        ))
        await db.commit()

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(TriggerEvent).where(TriggerEvent.event_dedupe_id == "shared_msg_id")
        )).scalars().all()
    assert len(rows) == 2


def _is_sqlite() -> bool:
    return os.environ.get("DATABASE_URL", "").startswith("sqlite")


async def _ensure_sqlite_fks_on() -> None:
    """SQLite has FK enforcement OFF by default — it is a per-connection
    setting, and neither conftest nor `_build_engine` turns it on. So an
    `ondelete="CASCADE"` the model genuinely declares is simply not enforced
    by the test engine, and a cascade test fails against correct production
    code. (Verified: PRAGMA foreign_keys reads 0; with it ON this file is
    11/11 green.) Copied from tests/test_connector_schema.py, which hit this
    first. No-op on Postgres.
    """
    if not _is_sqlite():
        return
    from app.db.database import engine
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA foreign_keys = ON"))


# ── FK lifecycle ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deleting_trigger_cascades_to_events():
    """Deleting a Trigger must wipe its trigger_events rows
    (ON DELETE CASCADE). Otherwise UX delete leaves orphaned event rows
    that ix_trigger_events_status_received scans forever."""
    from app.db import async_session_maker
    from app.db.models import Trigger, TriggerEvent

    await _ensure_sqlite_fks_on()
    user_id = await _make_user()
    tid = await _make_trigger(user_id)

    async with async_session_maker() as db:
        for i in range(3):
            db.add(TriggerEvent(
                id=str(uuid.uuid4()),
                trigger_id=tid,
                user_id=user_id,
                event_dedupe_id=f"msg_{i}",
            ))
        await db.commit()

    # Sanity — 3 rows present.
    async with async_session_maker() as db:
        before = (await db.execute(
            select(TriggerEvent).where(TriggerEvent.trigger_id == tid)
        )).scalars().all()
        assert len(before) == 3

    # Delete the Trigger.
    async with async_session_maker() as db:
        t = (await db.execute(select(Trigger).where(Trigger.id == tid))).scalar_one()
        await db.delete(t)
        await db.commit()

    async with async_session_maker() as db:
        after = (await db.execute(
            select(TriggerEvent).where(TriggerEvent.trigger_id == tid)
        )).scalars().all()
    assert after == []


@pytest.mark.asyncio
async def test_coalesced_into_event_id_accepts_self_ref():
    """The self-FK on trigger_events allows one event row to point at
    another (sibling that was folded into an in-flight handler run).
    Setting the column to a real existing event_id must succeed; NULL
    is also valid (default state)."""
    from app.db import async_session_maker
    from app.db.models import TriggerEvent

    user_id = await _make_user()
    tid = await _make_trigger(user_id)

    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())

    async with async_session_maker() as db:
        db.add(TriggerEvent(
            id=parent_id, trigger_id=tid, user_id=user_id,
            event_dedupe_id="parent_msg",
        ))
        await db.commit()

    async with async_session_maker() as db:
        db.add(TriggerEvent(
            id=child_id, trigger_id=tid, user_id=user_id,
            event_dedupe_id="child_msg",
            coalesced_into_event_id=parent_id,
            status="coalesced",
        ))
        await db.commit()

    async with async_session_maker() as db:
        child = (await db.execute(
            select(TriggerEvent).where(TriggerEvent.id == child_id)
        )).scalar_one()
    assert child.coalesced_into_event_id == parent_id
    assert child.status == "coalesced"


# ── JSON column round-trip ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_config_and_provider_state_roundtrip():
    """JSON columns must survive a write→read cycle with nested
    structures intact. config_json carries rate_limit dict; provider
    state carries the Gmail watch handle. SQLite's JSON1 + Postgres JSONB
    are both expected to preserve nested dicts."""
    from app.db import async_session_maker
    from app.db.models import Trigger

    user_id = await _make_user()
    cfg = {
        "connector_identity_id": "abc-123",
        "rate_limit": {
            "per_hour": 30,
            "per_minute": 5,
            "coalesce_window_sec": 120,
        },
        "delivery_channels": ["website", "telegram"],
    }
    provider_state = {
        "gmail_history_id": "987654",
        "watch_expires_at": "2026-05-19T08:00:00Z",
        "pubsub_topic": "projects/toup-prod/topics/gmail-events",
    }
    tid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Trigger(
            id=tid,
            user_id=user_id,
            kind="email_received",
            action="summarize_and_post",
            config_json=cfg,
            provider_state_json=provider_state,
        ))
        await db.commit()

    async with async_session_maker() as db:
        row = (await db.execute(
            select(Trigger).where(Trigger.id == tid)
        )).scalar_one()
    assert row.config_json == cfg
    assert row.provider_state_json == provider_state


# ── Hot-path index sanity ────────────────────────────────────────────


def test_triggers_has_user_kind_enabled_index():
    """The (user_id, kind, enabled) composite is the index the webhook
    will hit for "find enabled email_received triggers for this user."
    Drift = the runner scans the table linearly per Pub/Sub push."""
    from app.db.models import Trigger

    index_columns = {
        tuple(col.name for col in idx.columns)
        for idx in Trigger.__table__.indexes
    }
    # SQLAlchemy may also auto-emit a single-column user_id index from
    # the FK declaration — we only care that the composite is present.
    assert ("user_id", "kind", "enabled") in index_columns, (
        f"missing composite index. Found: {index_columns}"
    )


def test_trigger_events_has_required_indexes():
    """trigger_events needs three indexes — per-trigger timeline,
    per-user timeline, and status-sweep — to keep Mission Control
    queries fast as the table grows."""
    from app.db.models import TriggerEvent

    index_columns = {
        tuple(col.name for col in idx.columns)
        for idx in TriggerEvent.__table__.indexes
    }
    assert ("trigger_id", "received_at") in index_columns
    assert ("user_id", "received_at") in index_columns
    assert ("status", "received_at") in index_columns
