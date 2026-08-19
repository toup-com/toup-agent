"""The write path after the rebuild (docs/memory/rebuild-2026-08.md §3.2–3.3):
every write gets a memory-file home, active tasks stop minting variants, a
standing arrangement is never leased, and all three prompts store the user's
second person.

The prompt checks are SOURCE probes on purpose — no test executes those LLM
calls, so presence-in-source is the only guard that exists (same lesson as
scripts/check-boot.js on the app side).
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
from app.schemas import BrainType, MemoryCreate
from app.services.active_task_service import (
    _same_task,
    decay_expired_tasks,
    get_active_tasks,
    store_active_task,
)

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
    monkeypatch.setattr(
        "app.services.memory_dedup_service.get_embedding_service", lambda: _StubEmbedding()
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


# ── Active tasks: variants merge, standing arrangements aren't leased ──

async def test_same_task_matches_the_founders_variants():
    # The exact duplicates from the 2026-08 recording: three phrasings each.
    gmail = [
        "Send a summary of my Gmail inbox every day at 11:49 AM",
        "You want to receive a daily briefing of your Gmail at 11:49",
        "Daily Gmail briefing at 11:49 AM America/Toronto",
    ]
    quote = [
        "You want Nariman to receive one short motivational quote at 5:06 PM",
        "Get a motivational quote daily at 5:06 PM",
    ]
    for other in gmail[1:]:
        assert _same_task(gmail[0], other), other
    assert _same_task(quote[0], quote[1])
    # Different arrangements at the same clock time stay distinct…
    assert not _same_task("Go to the gym at 5:06 PM", quote[1])
    # …and near-miss tasks that share a template stay distinct.
    assert not _same_task("Fix the login bug", "Fix the signup bug")
    assert not _same_task("", "anything")


async def test_store_active_task_reinforces_variants_into_one_row():
    db, user_id = await _make_session()
    first = await store_active_task(
        db, user_id, "Send me a summary of my Gmail inbox every day at 11:49 AM"
    )
    second = await store_active_task(
        db, user_id, "Receive a daily briefing of your Gmail at 11:49"
    )
    assert first == second  # reinforced, not duplicated
    rows = (await db.execute(
        select(Memory).where(Memory.user_id == user_id, Memory.category == "active_task")
    )).scalars().all()
    assert len(rows) == 1
    assert rows[0].file_slug == "working"


async def test_standing_arrangement_gets_no_lease_and_a_tag():
    db, user_id = await _make_session()
    standing_id = await store_active_task(
        db, user_id, "Send me a motivational quote every day at 5:06 PM"
    )
    oneoff_id = await store_active_task(
        db, user_id, "Finish debugging the sidebar CSS bug"
    )
    standing = (await db.execute(select(Memory).where(Memory.id == standing_id))).scalar_one()
    oneoff = (await db.execute(select(Memory).where(Memory.id == oneoff_id))).scalar_one()
    assert standing.expires_at is None
    assert json.loads(standing.tags_json) == ["standing"]
    assert oneoff.expires_at is not None  # 7-day lease
    assert oneoff.tags_json is None
    assert standing.file_slug == oneoff.file_slug == "working"
    # The standing arrangement survives the TTL sweep; a stale one-off doesn't.
    oneoff.expires_at = datetime.utcnow() - timedelta(days=1)
    oneoff.created_at = datetime.utcnow() - timedelta(days=9)
    await db.flush()
    archived = await decay_expired_tasks(db, user_id)
    assert archived == 1
    await db.refresh(standing); await db.refresh(oneoff)
    assert standing.is_active and not oneoff.is_active


async def test_get_active_tasks_excludes_expired_but_unswept_leases():
    db, user_id = await _make_session()
    live_id = await store_active_task(db, user_id, "Finish the migration script draft")
    stale_id = await store_active_task(db, user_id, "Review the quarterly budget numbers")
    stale = (await db.execute(select(Memory).where(Memory.id == stale_id))).scalar_one()
    stale.expires_at = datetime.utcnow() - timedelta(hours=2)  # lease ran out, sweep hasn't
    await db.flush()
    tasks = await get_active_tasks(db, user_id)
    ids = {t["id"] for t in tasks}
    assert live_id in ids and stale_id not in ids


# ── Dedup boundary: every write gets a memory-file home ──────────────

async def test_smart_create_routes_rows_into_files():
    db, user_id = await _make_session()
    from app.services.memory_dedup_service import MemoryDedupService

    dedup = MemoryDedupService(db)
    results = await dedup.smart_create_memories(
        new_memories=[
            MemoryCreate(content="You are applying to the UofT MScAC program.",
                         brain_type=BrainType.USER, category="goals", memory_type="fact"),
            MemoryCreate(content="Majid Tajik is your IELTS tutor and teaches on Tuesdays.",
                         brain_type=BrainType.USER, category="people", memory_type="person"),
            MemoryCreate(content="You prefer replies that get to the point quickly.",
                         brain_type=BrainType.AGENT, category="conversation_style", memory_type="preference"),
        ],
        user_id=user_id,
        person_names=[None, ["Majid Tajik"], None],
    )
    assert [action for _, action in results] == ["created", "created", "created"]
    (goal, _), (person, _), (learned, _) = results
    assert goal.file_slug == "areas/work"
    assert person.file_slug == "people/majid-tajik"
    assert learned.file_slug == "learned"
    assert goal.file_position is not None

    # The person file row was created with the person's name as title.
    row = (await db.execute(
        select(MemoryFile).where(
            MemoryFile.user_id == user_id, MemoryFile.slug == "people/majid-tajik"
        )
    )).scalar_one()
    assert row.title == "Majid Tajik" and row.section == "people"


async def test_supersede_inherits_the_old_rows_place():
    db, user_id = await _make_session()
    from app.services.memory_dedup_service import MemoryDedupService

    dedup = MemoryDedupService(db)
    old = Memory(
        user_id=user_id, content="You work at Acme Corp.", category="work",
        memory_type="fact", brain_type="user", file_slug="areas/work", file_position=3,
    )
    db.add(old)
    await db.commit()

    replacement = await dedup._supersede_with_new(
        old_memory_id=old.id,
        new_memory_data=MemoryCreate(
            content="You left Acme Corp and now work at Initech.",
            brain_type=BrainType.USER, category="work", memory_type="fact",
        ),
        user_id=user_id,
        reason="changed jobs",
        file_slug="knowledge",  # routed slug loses to inheritance
    )
    await db.refresh(old)
    assert old.is_active is False and old.superseded_by == replacement.id
    assert replacement.file_slug == "areas/work"   # inherited, not the routed slug
    assert replacement.file_position == 3          # same place in the file


# ── Voice: the three prompts store the user's second person ──────────

def test_extractor_prompt_speaks_to_the_user():
    src = (BACKEND / "app/services/memory_extractor.py").read_text()
    assert "written in the second person" in src
    assert '"You are applying to the UofT MScAC program for graduate studies"' in src
    assert '"Nariman is applying to UofT MScAC program' not in src
    # The noise rules exist: assistant outcomes and bodily states are skipped.
    assert "Playback, search or tool OUTCOMES" in src
    assert "Momentary bodily states" in src


def test_consolidation_prompt_speaks_to_the_user():
    src = (BACKEND / "app/services/consolidation_service.py").read_text()
    assert "third person" not in src.lower()
    assert src.count('Write in the second person, to the user') == 2


def test_merge_prompt_speaks_to_the_user():
    src = (BACKEND / "app/services/memory_dedup_service.py").read_text()
    assert 'Write in the second person, to the user ("You…", "Your…")' in src
