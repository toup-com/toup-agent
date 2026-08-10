"""
Tests for the Memory Consolidation System

Tests the episodic to semantic memory promotion and memory linking.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import init_db, drop_db, async_session_maker
from app.db.models import Memory, MemoryEvent, MemoryEventType, MemoryLevel
from app.services import create_user
from app.services.consolidation_service import ConsolidationService


@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    """Create a fresh database for each test"""
    await init_db()
    yield
    await drop_db()


@pytest_asyncio.fixture
async def db_session():
    """Get a database session"""
    async with async_session_maker() as session:
        yield session


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession):
    """Create a test user"""
    user = await create_user(
        db_session,
        email="test@example.com",
        name="Test User"
    )
    return user


@pytest_asyncio.fixture
async def consolidation_service(db_session: AsyncSession):
    """Create a consolidation service instance"""
    return ConsolidationService(db_session)


@pytest_asyncio.fixture
async def episodic_memory(db_session: AsyncSession, test_user):
    """Create a test episodic memory that is ready for consolidation"""
    old_date = datetime.utcnow() - timedelta(days=14)  # 2 weeks old
    
    memory = Memory(
        user_id=test_user.id,
        content="I met with Dr. Smith about the project yesterday",
        category="events",
        memory_type="event",
        importance=0.7,
        strength=0.8,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.6,
        access_count=5,  # Accessed multiple times
        created_at=old_date,
        last_reinforced_at=old_date,
        embedding=[0.1] * 384,
    )
    db_session.add(memory)
    await db_session.commit()
    await db_session.refresh(memory)
    return memory


@pytest_asyncio.fixture
async def similar_memories(db_session: AsyncSession, test_user):
    """A cluster of similar episodic memories that is ACTUALLY eligible.

    Eligibility is a conjunction, and this fixture used to satisfy only
    half of it: created 10 days ago (old enough — MIN_AGE_DAYS is 7) but
    also last reinforced 10 days ago, which fails the "still relevant"
    condition (MAX_LAST_ACCESS_DAYS is 7). The candidate query therefore
    returned nothing and any test that thought it was exercising
    consolidation was exercising the empty path instead.

    So: created old, reinforced recently — the shape consolidation is
    actually for.
    """
    old_date = datetime.utcnow() - timedelta(days=10)
    recent = datetime.utcnow() - timedelta(days=1)

    memories = []
    contents = [
        "Python is great for data analysis",
        "I use Python for my daily programming tasks",
        "Python's pandas library is very useful",
        "Learning Python has improved my productivity",
    ]
    
    for i, content in enumerate(contents):
        memory = Memory(
            user_id=test_user.id,
            content=content,
            category="knowledge",
            memory_type="fact",
            importance=0.6,
            strength=0.7,
            memory_level=MemoryLevel.EPISODIC,
            emotional_salience=0.5,
            access_count=3 + i,  # Variable access counts
            created_at=old_date - timedelta(days=i),
            last_reinforced_at=recent,
            embedding=[0.1 + i * 0.01] * 384,  # Slightly different embeddings
        )
        db_session.add(memory)
        memories.append(memory)
    
    await db_session.commit()
    for m in memories:
        await db_session.refresh(m)
    
    return memories


# ============ Consolidation Service Tests ============
#
# Rewritten 2026-08-10 (finish run, W-8). The previous bodies called
# `_find_consolidation_candidates`, `_find_related_memories`,
# `_link_memories`, `_promote_to_procedural`, `_create_meta_memory` and
# `get_consolidation_metrics` — none of which exist on ConsolidationService
# any more, so 18 of 19 tests failed on import-time attribute access and the
# file was parked in COVERAGE_DEBT.txt as "references code that no longer
# exists". These cover the API that IS there, and the eligibility rules that
# decide what consolidation is even allowed to touch — the part with real
# blast radius, since consolidation rewrites stored rows.


@pytest.mark.asyncio
async def test_service_exposes_the_documented_api(consolidation_service):
    """Pin the surface the schedulers and the admin route call.

    `run_decay_for_all_users`' sibling job and `POST /admin/.../consolidate`
    both bind to these names; losing one is the failure that parked this
    file in the first place.
    """
    for name in ("run_consolidation", "promote_to_semantic",
                 "_find_similar_groups", "_consolidate_group"):
        assert hasattr(consolidation_service, name), (
            f"ConsolidationService lost {name} — a caller is now broken"
        )


@pytest.mark.asyncio
async def test_promote_to_semantic_changes_level_and_slows_decay(
    db_session: AsyncSession, test_user, episodic_memory, consolidation_service
):
    """Promotion is the one destructive-ish write with a simple contract:
    level up, consolidation_count up, decay HALVED (a semantic fact should
    outlive the episode it came from) with a 0.05 floor."""
    before_rate = episodic_memory.decay_rate
    before_count = episodic_memory.consolidation_count or 0

    promoted = await consolidation_service.promote_to_semantic(
        memory_id=episodic_memory.id, user_id=test_user.id
    )

    assert promoted is not None
    assert promoted.memory_level == "semantic"
    assert promoted.consolidation_count == before_count + 1
    assert promoted.decay_rate == max(0.05, before_rate * 0.5)
    assert promoted.decay_rate <= before_rate


@pytest.mark.asyncio
async def test_promote_to_semantic_writes_a_consolidated_event(
    db_session: AsyncSession, test_user, episodic_memory, consolidation_service
):
    """The audit trail is the only way to answer 'why did this row change?'
    after the fact — `memory_events` is where a surprised operator looks."""
    await consolidation_service.promote_to_semantic(
        memory_id=episodic_memory.id, user_id=test_user.id
    )

    rows = (await db_session.execute(
        select(MemoryEvent).where(MemoryEvent.memory_id == episodic_memory.id)
    )).scalars().all()

    consolidated = [e for e in rows if e.event_type == MemoryEventType.CONSOLIDATED]
    assert consolidated, "promotion wrote no CONSOLIDATED event"
    assert consolidated[0].trigger_source == "api"


@pytest.mark.asyncio
async def test_promote_refuses_another_users_memory(
    test_user, episodic_memory, consolidation_service
):
    """Tenant isolation at the service boundary: a wrong user_id must return
    None, not silently rewrite the row."""
    result = await consolidation_service.promote_to_semantic(
        memory_id=episodic_memory.id, user_id=str(uuid4())
    )
    assert result is None


@pytest.mark.asyncio
async def test_promote_refuses_a_deleted_memory(
    db_session: AsyncSession, test_user, episodic_memory, consolidation_service
):
    episodic_memory.is_deleted = True
    await db_session.commit()

    result = await consolidation_service.promote_to_semantic(
        memory_id=episodic_memory.id, user_id=test_user.id
    )
    assert result is None


@pytest.mark.asyncio
async def test_run_consolidation_returns_its_three_counters(
    test_user, similar_memories, consolidation_service
):
    """(considered, groups, consolidated) is what the scheduler logs; the
    shape is the contract, and `considered` must actually see the eligible
    fixture rows rather than silently returning zeros."""
    considered, groups, consolidated = await consolidation_service.run_consolidation(
        user_id=test_user.id
    )

    assert isinstance(considered, int) and isinstance(groups, int)
    assert isinstance(consolidated, int)
    assert considered >= len(similar_memories), (
        "eligible episodic rows were not even considered — the candidate "
        "query no longer matches what the fixtures describe as eligible"
    )
    assert consolidated <= considered


@pytest.mark.asyncio
async def test_run_consolidation_on_a_user_with_nothing(consolidation_service):
    considered, groups, consolidated = await consolidation_service.run_consolidation(
        user_id=str(uuid4())
    )
    assert (considered, groups, consolidated) == (0, 0, 0)


@pytest.mark.asyncio
async def test_eligibility_excludes_young_low_access_weak_and_deleted(
    db_session: AsyncSession, test_user, consolidation_service
):
    """Four independent disqualifiers, one per row, asserted together.

    Consolidation REWRITES memories, so its candidate query is the safety
    boundary: a row that is too new, barely used, mostly decayed, or
    deleted must never be swept into a merge.
    """
    now = datetime.utcnow()
    old = now - timedelta(days=14)
    disqualified = {
        "young": dict(created_at=now, last_reinforced_at=now,
                      access_count=9, strength=0.9, is_deleted=False),
        "rarely_accessed": dict(created_at=old, last_reinforced_at=old,
                                access_count=0, strength=0.9, is_deleted=False),
        "decayed": dict(created_at=old, last_reinforced_at=old,
                        access_count=9, strength=0.05, is_deleted=False),
        "deleted": dict(created_at=old, last_reinforced_at=old,
                        access_count=9, strength=0.9, is_deleted=True),
    }
    for label, kw in disqualified.items():
        db_session.add(Memory(
            user_id=test_user.id,
            content=f"disqualified because it is {label}",
            category="knowledge", memory_type="fact",
            importance=0.6, emotional_salience=0.5,
            memory_level=MemoryLevel.EPISODIC,
            embedding=[0.2] * 384,
            **kw,
        ))
    await db_session.commit()

    considered, _groups, _consolidated = await consolidation_service.run_consolidation(
        user_id=test_user.id
    )
    assert considered == 0, (
        f"{considered} disqualified row(s) entered the candidate set — "
        "consolidation rewrites rows, so this query is a safety boundary"
    )


@pytest.mark.asyncio
async def test_find_similar_groups_needs_at_least_two(consolidation_service):
    """MIN_GROUP_SIZE is 2: a lone memory has nothing to consolidate WITH,
    and a 'group' of one would merge a row into itself."""
    assert await consolidation_service._find_similar_groups([]) == []

    lone = Memory(
        user_id=str(uuid4()), content="only one", category="knowledge",
        memory_type="fact", memory_level=MemoryLevel.EPISODIC,
        embedding=[0.3] * 384,
    )
    assert await consolidation_service._find_similar_groups([lone]) == []


@pytest.mark.asyncio
async def test_consolidation_count_increments_are_cumulative(
    db_session: AsyncSession, test_user, episodic_memory, consolidation_service
):
    """`consolidation_count` feeds the decay curve's stability multiplier
    (decay_service: +20% per consolidation, capped at 5), so it has to
    accumulate rather than latch at 1."""
    first = await consolidation_service.promote_to_semantic(
        memory_id=episodic_memory.id, user_id=test_user.id
    )
    assert first.consolidation_count == 1

    second = await consolidation_service.promote_to_semantic(
        memory_id=episodic_memory.id, user_id=test_user.id
    )
    assert second.consolidation_count == 2
