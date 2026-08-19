"""The migration/maintenance pass (docs/memory/rebuild-2026-08.md §3.7) and
the RC3 decay repairs.

Pins: legacy working rows are repaired (standing arrangements unleased and
tagged, born-permanent one-offs leased from their last sign of life and then
ARCHIVED by the ordinary sweep — with events, nothing deleted), the pass is
idempotent, the curation policy fires once and then only on change+age, and
agent_main's scheduler registrations carry cron triggers, not intervals
(source probes — the interval-on-a-restarting-fleet hole is invisible to
every runtime test).
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.base import Base
from app.db.models.user import User
from app.db.models.memory import BrainStats, Memory, MemoryEvent, MemoryFile, memory_relationships
from app.db.models.entity import Entity, EntityLink
from app.services.active_task_service import normalize_working_leases
from app.services.memory_file_migration import _needs_curation, migrate_user_memory_files

BACKEND = Path(__file__).resolve().parent.parent


class _StubEmbedding:
    async def embed_async(self, text, api_key=None):
        return None

    def embed(self, text, api_key=None):
        return None


@pytest.fixture(autouse=True)
def _offline_embeddings(monkeypatch):
    monkeypatch.setattr(
        "app.services.memory_service.get_embedding_service", lambda: _StubEmbedding()
    )


async def _make_session():
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[
                User.__table__, Memory.__table__, MemoryFile.__table__,
                MemoryEvent.__table__, BrainStats.__table__,
                Entity.__table__, EntityLink.__table__, memory_relationships,
            ],
        )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = maker()
    user_id = str(uuid.uuid4())
    db.add(User(id=user_id, email=f"{user_id[:8]}@test.local", hashed_password="x"))
    await db.commit()
    return db, user_id


def _task(user_id, content, *, days_old, expires_at=None):
    when = datetime.utcnow() - timedelta(days=days_old)
    return Memory(
        user_id=user_id, content=content, category="active_task",
        memory_type="task", brain_type="user", memory_level="working",
        importance=0.9, created_at=when, updated_at=when, expires_at=expires_at,
    )


async def test_normalize_working_leases_repairs_all_three_legacy_shapes():
    db, user_id = await _make_session()
    # The founder's corpus, in miniature:
    # 1. A recurring arrangement born BEFORE leases existed (May) — NULL lease.
    may_standing = _task(user_id, "Send a daily Gmail briefing every day at 11:49 AM", days_old=97)
    # 2. The same arrangement re-stated AFTER leases shipped (Aug 10) — leased,
    #    so the sweep would archive a standing arrangement the user relies on.
    aug_standing = _task(
        user_id, "You get one motivational quote every day at 5:06 PM",
        days_old=9, expires_at=datetime.utcnow() - timedelta(days=2),
    )
    # 3. A born-permanent one-off from May — invisible to the sweep forever.
    may_oneoff = _task(user_id, "Research the three best PM tools", days_old=97)
    # 4. A live, correctly-leased one-off — must be untouched.
    live = _task(
        user_id, "Finish the sidebar CSS fix", days_old=1,
        expires_at=datetime.utcnow() + timedelta(days=6),
    )
    db.add_all([may_standing, aug_standing, may_oneoff, live])
    await db.commit()

    changed = await normalize_working_leases(db, user_id)
    await db.commit()
    assert changed == {"standing": 2, "leased": 1}

    for row in (may_standing, aug_standing):
        await db.refresh(row)
        assert row.expires_at is None
        assert json.loads(row.tags_json) == ["standing"]
    await db.refresh(may_oneoff)
    assert may_oneoff.expires_at is not None
    assert may_oneoff.expires_at < datetime.utcnow()  # already past → sweep archives
    await db.refresh(live)
    assert live.tags_json is None and live.expires_at > datetime.utcnow()

    # Idempotent.
    assert await normalize_working_leases(db, user_id) == {"standing": 0, "leased": 0}


async def test_migration_archives_stale_working_rows_with_events_not_deletes():
    db, user_id = await _make_session()
    stale = _task(user_id, "Draft the conference talk outline", days_old=97)
    standing = _task(user_id, "Send a daily Gmail briefing every day at 11:49 AM", days_old=97)
    fresh = _task(user_id, "Finish the sidebar CSS fix", days_old=1)
    db.add_all([stale, standing, fresh])
    await db.commit()

    report = await migrate_user_memory_files(db, user_id, consolidate=False)
    assert report["organized"] == 3          # every row got a file_slug
    assert report["leases"]["standing"] == 1
    assert report["expired"] == 1            # the May one-off faded

    await db.refresh(stale); await db.refresh(standing); await db.refresh(fresh)
    assert stale.is_active is False and stale.is_deleted is False  # archived, kept
    assert standing.is_active and fresh.is_active
    assert stale.file_slug == standing.file_slug == "working"

    events = (await db.execute(
        select(MemoryEvent).where(MemoryEvent.memory_id == stale.id)
    )).scalars().all()
    assert any(e.event_type == "decayed" for e in events)

    # Second run: nothing to do — the pass is safe to fire on every boot.
    report2 = await migrate_user_memory_files(db, user_id, consolidate=False)
    assert (report2["organized"], report2["expired"]) == (0, 0)
    assert report2["leases"] == {"standing": 0, "leased": 0}


async def test_curation_policy_fires_once_then_only_on_change_and_age():
    now = datetime.utcnow()
    f = {"entry_count": 5, "consolidated_at": None, "updated_at": now.isoformat()}
    assert _needs_curation(f, now)                          # never curated
    f = {"entry_count": 1, "consolidated_at": None, "updated_at": now.isoformat()}
    assert not _needs_curation(f, now)                      # nothing to merge
    recent = (now - timedelta(days=2)).isoformat()
    f = {"entry_count": 5, "consolidated_at": recent, "updated_at": now.isoformat()}
    assert not _needs_curation(f, now)                      # changed but too recent
    old = (now - timedelta(days=9)).isoformat()
    f = {"entry_count": 5, "consolidated_at": old, "updated_at": now.isoformat()}
    assert _needs_curation(f, now)                          # changed and aged
    f = {"entry_count": 5, "consolidated_at": old, "updated_at": old}
    assert not _needs_curation(f, now)                      # aged but unchanged


# ── Scheduler wiring: source probes on agent_main ─────────────────────

def test_agent_scheduler_uses_cron_not_interval_for_memory_jobs():
    src = (BACKEND / "agent_main.py").read_text()
    block = src[src.index("Memory maintenance (audit A6-1)"):]
    block = block[:block.index("Agent initialization error")]
    # Probe CODE, not commentary — comments legitimately name the retired
    # function while explaining why it was retired.
    block = "\n".join(
        line for line in block.splitlines() if not line.strip().startswith("#")
    )
    # The RC3.1 hole: an IntervalTrigger on a fleet that restarts faster
    # than the interval never fires. No memory job may use one.
    assert "_MMIvl(" not in block, "a memory-maintenance job regressed to an interval trigger"
    assert '_MMCron(hour="2,8,14,20", minute=0)' in block          # decay
    assert "run_memory_file_maintenance" in block                   # consolidation = file curation
    assert "run_consolidation_for_all_users" not in block           # additive pass retired
    assert '"memory_file_migration_boot"' in block                  # boot one-shot registered
    assert "_MMCron(minute=0))" in block                            # day_archival hourly cron
