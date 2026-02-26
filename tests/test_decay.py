"""
Tests for the Memory Decay System

Tests the Ebbinghaus forgetting curve implementation and memory strength decay.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
import math

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import init_db, drop_db, async_session_maker
from app.db.models import Memory, MemoryEvent, MemoryEventType, MemoryLevel
from app.services import create_user
from app.services.decay_service import DecayService


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
        password="testpassword123",
        name="Test User"
    )
    return user


@pytest_asyncio.fixture
async def decay_service(db_session: AsyncSession):
    """Create a decay service instance"""
    return DecayService(db_session)


@pytest_asyncio.fixture
async def test_memory(db_session: AsyncSession, test_user):
    """Create a test memory"""
    memory = Memory(
        user_id=test_user.id,
        content="Test memory for decay testing",
        category="knowledge",
        memory_type="fact",
        importance=0.5,
        strength=1.0,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.5,
        decay_rate=1.0,
        embedding=[0.1] * 384,  # Dummy embedding
    )
    db_session.add(memory)
    await db_session.commit()
    await db_session.refresh(memory)
    return memory


# ============ Decay Formula Tests ============

class TestDecayFormula:
    """Test the Ebbinghaus forgetting curve formula"""
    
    def test_decay_formula_zero_days(self):
        """Memory should retain full strength at day 0"""
        # R = e^(-t/S) where t=0 gives R=1
        t = 0
        S = 7  # base half-life
        result = math.exp(-t / S)
        assert result == 1.0
    
    def test_decay_formula_half_life(self):
        """Memory should be at ~37% at one time constant (S days)"""
        t = 7  # days
        S = 7  # stability (time constant)
        result = math.exp(-t / S)
        # e^(-1) ≈ 0.368
        assert abs(result - 0.368) < 0.01
    
    def test_decay_formula_two_half_lives(self):
        """Memory should be at ~13.5% at 2x time constant"""
        t = 14  # days
        S = 7  # stability
        result = math.exp(-t / S)
        # e^(-2) ≈ 0.135
        assert abs(result - 0.135) < 0.01
    
    def test_higher_stability_means_slower_decay(self):
        """Higher stability should mean slower decay"""
        t = 7  # days
        low_stability = 7
        high_stability = 14
        
        low_result = math.exp(-t / low_stability)
        high_result = math.exp(-t / high_stability)
        
        # Higher stability = slower decay = higher retention
        assert high_result > low_result


# ============ Decay Service Tests ============

@pytest.mark.asyncio
async def test_decay_service_initialization(decay_service: DecayService):
    """Test decay service initializes correctly"""
    assert decay_service is not None
    assert decay_service.BASE_HALF_LIFE_DAYS == 7.0
    assert decay_service.MIN_STRENGTH == 0.1
    assert decay_service.MAX_STRENGTH == 1.0


@pytest.mark.asyncio
async def test_calculate_decay_recent_memory(
    decay_service: DecayService,
    test_memory: Memory
):
    """Test decay calculation for a recent memory"""
    # Memory created just now should have minimal decay
    now = datetime.utcnow()
    test_memory.last_reinforced_at = now
    
    new_strength = decay_service._calculate_decayed_strength(test_memory, now)
    assert new_strength == test_memory.strength  # No decay for 0 time


@pytest.mark.asyncio
async def test_calculate_decay_old_memory(
    decay_service: DecayService,
    test_memory: Memory
):
    """Test decay calculation for an older memory"""
    now = datetime.utcnow()
    test_memory.last_reinforced_at = now - timedelta(days=7)
    
    new_strength = decay_service._calculate_decayed_strength(test_memory, now)
    
    # Should be less than original strength
    assert new_strength < test_memory.strength
    # With importance=0.5, emotional_salience=0.5, decay should be moderate
    # The exact value depends on the modifier factors
    assert new_strength > decay_service.MIN_STRENGTH


@pytest.mark.asyncio
async def test_calculate_decay_respects_minimum(
    decay_service: DecayService,
    test_memory: Memory
):
    """Test that decay respects minimum strength"""
    now = datetime.utcnow()
    # Very old memory
    test_memory.last_reinforced_at = now - timedelta(days=365)
    
    new_strength = decay_service._calculate_decayed_strength(test_memory, now)
    assert new_strength >= decay_service.MIN_STRENGTH


@pytest.mark.asyncio
async def test_apply_decay_to_user(
    db_session: AsyncSession,
    decay_service: DecayService,
    test_user
):
    """Test applying decay to all memories for a user"""
    # Create multiple memories with old timestamps
    old_date = datetime.utcnow() - timedelta(days=7)
    
    for i in range(3):
        memory = Memory(
            user_id=test_user.id,
            content=f"Test memory {i}",
            category="knowledge",
            memory_type="fact",
            importance=0.5,
            strength=1.0,
            memory_level=MemoryLevel.EPISODIC,
            last_reinforced_at=old_date,
            decay_rate=1.0,
            embedding=[0.1] * 384,
        )
        db_session.add(memory)
    
    await db_session.commit()
    
    processed, updated = await decay_service.apply_decay_to_user(str(test_user.id))
    
    assert processed == 3
    assert updated == 3  # All should have decayed


@pytest.mark.asyncio
async def test_reinforce_memory(
    db_session: AsyncSession,
    decay_service: DecayService,
    test_memory: Memory
):
    """Test reinforcing a memory"""
    # First let it have lower strength
    test_memory.strength = 0.5
    test_memory.last_reinforced_at = datetime.utcnow() - timedelta(days=7)
    await db_session.commit()
    
    old_strength = test_memory.strength
    reinforced = await decay_service.reinforce_memory(
        str(test_memory.id),
        str(test_memory.user_id)
    )
    
    assert reinforced is not None
    assert reinforced.strength > old_strength  # Strength increased
    
    # Check that last_reinforced_at was updated
    time_diff = datetime.utcnow() - reinforced.last_reinforced_at
    assert time_diff.total_seconds() < 5  # Should be very recent


@pytest.mark.asyncio
async def test_reinforce_memory_logs_event(
    db_session: AsyncSession,
    decay_service: DecayService,
    test_memory: Memory
):
    """Test that reinforcement creates a memory event"""
    await decay_service.reinforce_memory(
        str(test_memory.id),
        str(test_memory.user_id)
    )
    
    # Check for reinforced event
    result = await db_session.execute(
        select(MemoryEvent)
        .where(MemoryEvent.memory_id == str(test_memory.id))
        .where(MemoryEvent.event_type == MemoryEventType.REINFORCED)
    )
    event = result.scalar_one_or_none()
    
    assert event is not None


@pytest.mark.asyncio
async def test_semantic_memories_decay_slower(
    db_session: AsyncSession,
    decay_service: DecayService,
    test_user
):
    """Test that semantic memories decay slower than episodic"""
    old_date = datetime.utcnow() - timedelta(days=7)
    
    # Create episodic memory
    episodic = Memory(
        user_id=test_user.id,
        content="Episodic memory",
        category="knowledge",
        memory_type="fact",
        importance=0.5,
        strength=1.0,
        memory_level=MemoryLevel.EPISODIC,
        last_reinforced_at=old_date,
        decay_rate=1.0,
        embedding=[0.1] * 384,
    )
    
    # Create semantic memory
    semantic = Memory(
        user_id=test_user.id,
        content="Semantic memory",
        category="knowledge",
        memory_type="fact",
        importance=0.5,
        strength=1.0,
        memory_level=MemoryLevel.SEMANTIC,
        last_reinforced_at=old_date,
        decay_rate=1.0,
        embedding=[0.1] * 384,
    )
    
    db_session.add(episodic)
    db_session.add(semantic)
    await db_session.commit()
    
    now = datetime.utcnow()
    episodic_strength = decay_service._calculate_decayed_strength(episodic, now)
    semantic_strength = decay_service._calculate_decayed_strength(semantic, now)
    
    # Semantic memory should retain more strength
    assert semantic_strength > episodic_strength


@pytest.mark.asyncio
async def test_high_importance_decays_slower(
    db_session: AsyncSession,
    decay_service: DecayService,
    test_user
):
    """Test that high importance memories decay slower"""
    old_date = datetime.utcnow() - timedelta(days=7)
    
    # Low importance memory
    low_importance = Memory(
        user_id=test_user.id,
        content="Low importance",
        category="knowledge",
        memory_type="fact",
        importance=0.1,
        strength=1.0,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.0,
        last_reinforced_at=old_date,
        decay_rate=1.0,
        embedding=[0.1] * 384,
    )
    
    # High importance memory
    high_importance = Memory(
        user_id=test_user.id,
        content="High importance",
        category="knowledge",
        memory_type="fact",
        importance=0.9,
        strength=1.0,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.0,
        last_reinforced_at=old_date,
        decay_rate=1.0,
        embedding=[0.1] * 384,
    )
    
    db_session.add(low_importance)
    db_session.add(high_importance)
    await db_session.commit()
    
    now = datetime.utcnow()
    low_strength = decay_service._calculate_decayed_strength(low_importance, now)
    high_strength = decay_service._calculate_decayed_strength(high_importance, now)
    
    # High importance memory should retain more strength
    assert high_strength > low_strength


@pytest.mark.asyncio
async def test_emotional_salience_resists_decay(
    db_session: AsyncSession,
    decay_service: DecayService,
    test_user
):
    """Test that emotionally salient memories decay slower"""
    old_date = datetime.utcnow() - timedelta(days=7)
    
    # Low emotional salience
    low_emotion = Memory(
        user_id=test_user.id,
        content="Low emotion",
        category="knowledge",
        memory_type="fact",
        importance=0.5,
        strength=1.0,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.1,
        last_reinforced_at=old_date,
        decay_rate=1.0,
        embedding=[0.1] * 384,
    )
    
    # High emotional salience
    high_emotion = Memory(
        user_id=test_user.id,
        content="High emotion",
        category="knowledge",
        memory_type="fact",
        importance=0.5,
        strength=1.0,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.9,
        last_reinforced_at=old_date,
        decay_rate=1.0,
        embedding=[0.1] * 384,
    )
    
    db_session.add(low_emotion)
    db_session.add(high_emotion)
    await db_session.commit()
    
    now = datetime.utcnow()
    low_strength = decay_service._calculate_decayed_strength(low_emotion, now)
    high_strength = decay_service._calculate_decayed_strength(high_emotion, now)
    
    # High emotional salience memory should retain more strength
    assert high_strength > low_strength


@pytest.mark.asyncio
async def test_reinforce_nonexistent_memory(
    decay_service: DecayService
):
    """Test handling of nonexistent memory"""
    from uuid import uuid4
    
    result = await decay_service.reinforce_memory(str(uuid4()), str(uuid4()))
    assert result is None  # Should return None for nonexistent memory


@pytest.mark.asyncio
async def test_get_weak_memories(
    db_session: AsyncSession,
    decay_service: DecayService,
    test_user
):
    """Test getting weak memories below threshold"""
    # Create memories with varying strengths
    for i, strength in enumerate([0.2, 0.5, 0.8]):
        memory = Memory(
            user_id=test_user.id,
            content=f"Memory with strength {strength}",
            category="knowledge",
            memory_type="fact",
            importance=0.5,
            strength=strength,
            memory_level=MemoryLevel.EPISODIC,
            decay_rate=1.0,
            embedding=[0.1] * 384,
        )
        db_session.add(memory)
    
    await db_session.commit()
    
    # Get weak memories (threshold 0.3)
    weak_memories = await decay_service.get_weak_memories(
        str(test_user.id),
        threshold=0.3
    )
    
    assert len(weak_memories) == 1
    assert weak_memories[0].strength == 0.2


@pytest.mark.asyncio
async def test_consolidation_count_slows_decay(
    db_session: AsyncSession,
    decay_service: DecayService,
    test_user
):
    """Test that consolidated memories decay slower"""
    old_date = datetime.utcnow() - timedelta(days=7)
    
    # Not consolidated
    not_consolidated = Memory(
        user_id=test_user.id,
        content="Not consolidated",
        category="knowledge",
        memory_type="fact",
        importance=0.5,
        strength=1.0,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.5,
        consolidation_count=0,
        last_reinforced_at=old_date,
        decay_rate=1.0,
        embedding=[0.1] * 384,
    )
    
    # Consolidated multiple times
    consolidated = Memory(
        user_id=test_user.id,
        content="Consolidated",
        category="knowledge",
        memory_type="fact",
        importance=0.5,
        strength=1.0,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.5,
        consolidation_count=5,
        last_reinforced_at=old_date,
        decay_rate=1.0,
        embedding=[0.1] * 384,
    )
    
    db_session.add(not_consolidated)
    db_session.add(consolidated)
    await db_session.commit()
    
    now = datetime.utcnow()
    not_cons_strength = decay_service._calculate_decayed_strength(not_consolidated, now)
    cons_strength = decay_service._calculate_decayed_strength(consolidated, now)
    
    # Consolidated memory should retain more strength
    assert cons_strength > not_cons_strength
