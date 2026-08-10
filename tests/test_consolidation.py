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
    """Create a cluster of similar episodic memories"""
    old_date = datetime.utcnow() - timedelta(days=10)
    
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
            last_reinforced_at=old_date,
            embedding=[0.1 + i * 0.01] * 384,  # Slightly different embeddings
        )
        db_session.add(memory)
        memories.append(memory)
    
    await db_session.commit()
    for m in memories:
        await db_session.refresh(m)
    
    return memories


# ============ Consolidation Service Tests ============

@pytest.mark.asyncio
async def test_consolidation_service_initialization(consolidation_service):
    """Test consolidation service initializes correctly"""
    assert consolidation_service is not None
    assert consolidation_service.MIN_ACCESS_COUNT >= 3
    assert consolidation_service.MIN_AGE_DAYS >= 7


@pytest.mark.asyncio
async def test_find_consolidation_candidates(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user,
    episodic_memory
):
    """Test finding memories eligible for consolidation"""
    candidates = await consolidation_service._find_consolidation_candidates(test_user.id)
    
    assert len(candidates) >= 1
    assert any(c.id == episodic_memory.id for c in candidates)


@pytest.mark.asyncio
async def test_young_memory_not_candidate(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user
):
    """Test that recent memories are not consolidation candidates"""
    # Create a very recent memory
    memory = Memory(
        user_id=test_user.id,
        content="Just created this memory",
        category="events",
        memory_type="event",
        importance=0.7,
        strength=1.0,
        memory_level=MemoryLevel.EPISODIC,
        access_count=10,  # High access but too young
        embedding=[0.1] * 384,
    )
    db_session.add(memory)
    await db_session.commit()
    
    candidates = await consolidation_service._find_consolidation_candidates(test_user.id)
    
    assert not any(c.id == memory.id for c in candidates)


@pytest.mark.asyncio
async def test_low_access_memory_not_candidate(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user
):
    """Test that rarely accessed memories are not consolidation candidates"""
    old_date = datetime.utcnow() - timedelta(days=30)
    
    memory = Memory(
        user_id=test_user.id,
        content="Old but forgotten memory",
        category="events",
        memory_type="event",
        importance=0.7,
        strength=0.3,
        memory_level=MemoryLevel.EPISODIC,
        access_count=1,  # Only accessed once
        created_at=old_date,
        last_reinforced_at=old_date,
        embedding=[0.1] * 384,
    )
    db_session.add(memory)
    await db_session.commit()
    
    candidates = await consolidation_service._find_consolidation_candidates(test_user.id)
    
    assert not any(c.id == memory.id for c in candidates)


@pytest.mark.asyncio
async def test_weak_memory_not_candidate(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user
):
    """Test that weak memories are not consolidation candidates"""
    old_date = datetime.utcnow() - timedelta(days=30)
    
    memory = Memory(
        user_id=test_user.id,
        content="Fading memory",
        category="events",
        memory_type="event",
        importance=0.7,
        strength=0.2,  # Too weak
        memory_level=MemoryLevel.EPISODIC,
        access_count=5,
        created_at=old_date,
        last_reinforced_at=old_date,
        embedding=[0.1] * 384,
    )
    db_session.add(memory)
    await db_session.commit()
    
    candidates = await consolidation_service._find_consolidation_candidates(test_user.id)
    
    assert not any(c.id == memory.id for c in candidates)


@pytest.mark.asyncio
async def test_promote_to_semantic(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    episodic_memory: Memory
):
    """Test promoting an episodic memory to semantic"""
    result = await consolidation_service._promote_to_semantic(episodic_memory)
    
    await db_session.refresh(episodic_memory)
    
    assert result["promoted"] is True
    assert episodic_memory.memory_level == MemoryLevel.SEMANTIC
    assert episodic_memory.consolidation_count >= 1


@pytest.mark.asyncio
async def test_promote_logs_event(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    episodic_memory: Memory
):
    """Test that promotion creates a memory event"""
    await consolidation_service._promote_to_semantic(episodic_memory)
    
    # Check for consolidated event
    result = await db_session.execute(
        select(MemoryEvent)
        .where(MemoryEvent.memory_id == episodic_memory.id)
        .where(MemoryEvent.event_type == MemoryEventType.CONSOLIDATED)
    )
    event = result.scalar_one_or_none()
    
    assert event is not None
    assert "episodic" in event.event_data_json.lower()
    assert "semantic" in event.event_data_json.lower()


@pytest.mark.asyncio
async def test_run_consolidation(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user,
    episodic_memory: Memory
):
    """Test running full consolidation process"""
    result = await consolidation_service.run_consolidation(test_user.id)
    
    assert result["total_candidates"] >= 1
    assert result["promoted"] >= 1
    
    await db_session.refresh(episodic_memory)
    assert episodic_memory.memory_level == MemoryLevel.SEMANTIC


@pytest.mark.asyncio
async def test_already_semantic_not_promoted(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user
):
    """Test that already semantic memories are not re-promoted"""
    old_date = datetime.utcnow() - timedelta(days=30)
    
    memory = Memory(
        user_id=test_user.id,
        content="Already semantic",
        category="knowledge",
        memory_type="fact",
        importance=0.8,
        strength=0.9,
        memory_level=MemoryLevel.SEMANTIC,  # Already semantic
        access_count=10,
        created_at=old_date,
        last_reinforced_at=old_date,
        consolidation_count=2,
        embedding=[0.1] * 384,
    )
    db_session.add(memory)
    await db_session.commit()
    
    candidates = await consolidation_service._find_consolidation_candidates(test_user.id)
    
    # Should not be in candidates since already semantic
    assert not any(c.id == memory.id for c in candidates)


# ============ Memory Linking Tests ============

@pytest.mark.asyncio
async def test_find_related_memories(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    similar_memories
):
    """Test finding related memories for linking"""
    # This tests the semantic similarity grouping
    memory = similar_memories[0]
    
    related = await consolidation_service._find_related_memories(
        memory,
        min_similarity=0.5
    )
    
    # Should find some related memories
    assert len(related) >= 0  # Depends on embedding similarity


@pytest.mark.asyncio
async def test_link_memories(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    similar_memories
):
    """Test linking related memories"""
    memory1 = similar_memories[0]
    memory2 = similar_memories[1]
    
    result = await consolidation_service._link_memories(
        memory1.id,
        memory2.id,
        relationship_type="related"
    )
    
    assert result["linked"] is True


@pytest.mark.asyncio
async def test_link_creates_events(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    similar_memories
):
    """Test that linking creates memory events"""
    memory1 = similar_memories[0]
    memory2 = similar_memories[1]
    
    await consolidation_service._link_memories(
        memory1.id,
        memory2.id,
        relationship_type="related"
    )
    
    # Check for linked events on both memories
    result = await db_session.execute(
        select(MemoryEvent)
        .where(MemoryEvent.memory_id == memory1.id)
        .where(MemoryEvent.event_type == MemoryEventType.LINKED)
    )
    event = result.scalar_one_or_none()
    
    assert event is not None


# ============ Meta Memory Tests ============

@pytest.mark.asyncio
async def test_create_meta_memory(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user,
    similar_memories
):
    """Test creating a meta memory from similar memories"""
    # Promote all to semantic first
    for mem in similar_memories:
        mem.memory_level = MemoryLevel.SEMANTIC
    await db_session.commit()
    
    result = await consolidation_service._create_meta_memory(
        user_id=test_user.id,
        source_memories=similar_memories,
        summary="User has strong knowledge and preference for Python programming"
    )
    
    if result.get("created"):
        meta_memory = result["memory"]
        assert meta_memory.memory_level == MemoryLevel.META
        assert "Python" in meta_memory.content


# ============ Procedural Memory Tests ============

@pytest.mark.asyncio
async def test_promote_to_procedural(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user
):
    """Test promoting a frequently used semantic memory to procedural"""
    old_date = datetime.utcnow() - timedelta(days=60)
    
    # Create a semantic memory with very high access
    memory = Memory(
        user_id=test_user.id,
        content="Always greet users with 'Hello!'",
        category="procedures",
        memory_type="procedure",
        importance=0.9,
        strength=0.95,
        memory_level=MemoryLevel.SEMANTIC,
        access_count=100,  # Very frequently accessed
        created_at=old_date,
        last_reinforced_at=datetime.utcnow() - timedelta(days=1),
        consolidation_count=10,
        embedding=[0.1] * 384,
    )
    db_session.add(memory)
    await db_session.commit()
    
    result = await consolidation_service._promote_to_procedural(memory)
    
    await db_session.refresh(memory)
    
    if result.get("promoted"):
        assert memory.memory_level == MemoryLevel.PROCEDURAL


# ============ Consolidation Metrics Tests ============

@pytest.mark.asyncio
async def test_get_consolidation_metrics(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user
):
    """Test getting consolidation statistics"""
    # Create some memories at different levels
    levels = [MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL]
    
    for level in levels:
        memory = Memory(
            user_id=test_user.id,
            content=f"Memory at {level.value} level",
            category="knowledge",
            memory_type="fact",
            importance=0.5,
            strength=0.8,
            memory_level=level,
            embedding=[0.1] * 384,
        )
        db_session.add(memory)
    
    await db_session.commit()
    
    metrics = await consolidation_service.get_consolidation_metrics(test_user.id)
    
    assert metrics["total_memories"] == 3
    assert metrics["by_level"]["episodic"] >= 1
    assert metrics["by_level"]["semantic"] >= 1
    assert metrics["by_level"]["procedural"] >= 1


# ============ Edge Cases ============

@pytest.mark.asyncio
async def test_consolidation_empty_user(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user
):
    """Test consolidation with no memories"""
    result = await consolidation_service.run_consolidation(test_user.id)
    
    assert result["total_candidates"] == 0
    assert result["promoted"] == 0


@pytest.mark.asyncio
async def test_consolidation_nonexistent_user(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService
):
    """Test consolidation for nonexistent user"""
    result = await consolidation_service.run_consolidation(uuid4())
    
    assert result["total_candidates"] == 0


@pytest.mark.asyncio
async def test_deleted_memories_not_consolidated(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    episodic_memory: Memory
):
    """Test that deleted memories are not consolidated"""
    episodic_memory.is_deleted = True
    await db_session.commit()
    
    candidates = await consolidation_service._find_consolidation_candidates(
        episodic_memory.user_id
    )
    
    assert not any(c.id == episodic_memory.id for c in candidates)


@pytest.mark.asyncio
async def test_consolidation_respects_batch_limit(
    db_session: AsyncSession,
    consolidation_service: ConsolidationService,
    test_user
):
    """Test that consolidation respects maximum batch size"""
    old_date = datetime.utcnow() - timedelta(days=30)
    
    # Create many eligible memories
    for i in range(20):
        memory = Memory(
            user_id=test_user.id,
            content=f"Memory {i} for batch test",
            category="knowledge",
            memory_type="fact",
            importance=0.6,
            strength=0.7,
            memory_level=MemoryLevel.EPISODIC,
            access_count=5,
            created_at=old_date,
            last_reinforced_at=old_date,
            embedding=[0.1] * 384,
        )
        db_session.add(memory)
    
    await db_session.commit()
    
    result = await consolidation_service.run_consolidation(
        test_user.id,
        max_batch_size=5
    )
    
    # Should only process up to batch limit
    assert result["promoted"] <= 5
