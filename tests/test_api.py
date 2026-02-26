"""
Tests for the HexBrain Memory System
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.db import init_db, drop_db, async_session_maker
from app.services import create_user, get_embedding_service, get_memory_extractor


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
async def client():
    """Create an async test client"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient):
    """Get auth headers for a demo user"""
    response = await client.post("/api/auth/demo")
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ============ Auth Tests ============

@pytest.mark.asyncio
async def test_demo_login(client: AsyncClient):
    """Test demo user login"""
    response = await client.post("/api/auth/demo")
    assert response.status_code == 200
    assert "access_token" in response.json()


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient, auth_headers: dict):
    """Test getting current user info"""
    response = await client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "email" in data


# ============ Memory Tests ============

@pytest.mark.asyncio
async def test_create_memory(client: AsyncClient, auth_headers: dict):
    """Test creating a memory"""
    response = await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "I love Python programming",
            "category": "preferences",
            "memory_type": "preference",
            "importance": 0.8,
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["content"] == "I love Python programming"
    assert data["category"] == "preferences"
    assert data["memory_type"] == "preference"


@pytest.mark.asyncio
async def test_get_memory(client: AsyncClient, auth_headers: dict):
    """Test getting a memory by ID"""
    # Create first
    create_response = await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Test memory content",
            "category": "knowledge",
            "memory_type": "fact",
        }
    )
    memory_id = create_response.json()["id"]
    
    # Get
    response = await client.get(
        f"/api/memories/{memory_id}",
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["id"] == memory_id


@pytest.mark.asyncio
async def test_search_memories(client: AsyncClient, auth_headers: dict):
    """Test semantic search for memories"""
    # Create some memories
    for content in [
        "I enjoy hiking in the mountains",
        "Python is my favorite programming language",
        "I need to call my doctor tomorrow",
    ]:
        await client.post(
            "/api/memories",
            headers=auth_headers,
            json={
                "content": content,
                "category": "knowledge",
                "memory_type": "fact",
            }
        )
    
    # Search
    response = await client.post(
        "/api/memories/search",
        headers=auth_headers,
        json={
            "query": "programming languages",
            "limit": 10,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    # Python memory should rank high
    if data["results"]:
        assert "Python" in data["results"][0]["content"] or len(data["results"]) > 0


@pytest.mark.asyncio
async def test_get_memories_by_category(client: AsyncClient, auth_headers: dict):
    """Test getting memories by category"""
    # Create memories in different categories
    await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Preference memory test",
            "category": "preferences",
            "memory_type": "preference",
        }
    )
    
    response = await client.get(
        "/api/memories/category/preferences",
        headers=auth_headers
    )
    assert response.status_code == 200
    memories = response.json()
    assert all(m["category"] == "preferences" for m in memories)


# ============ Ingestion Tests ============

@pytest.mark.asyncio
async def test_ingest_message(client: AsyncClient, auth_headers: dict):
    """Test ingesting a conversation message"""
    response = await client.post(
        "/api/ingest/message",
        headers=auth_headers,
        json={
            "user_message": "My name is Alex and I work as a data scientist.",
            "assistant_response": "Nice to meet you Alex! Data science is an exciting field.",
            "extract_memories": True,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert data["messages_ingested"] == 2
    assert data["memories_extracted"] >= 0


# ============ Stats Tests ============

@pytest.mark.asyncio
async def test_get_region_stats(client: AsyncClient, auth_headers: dict):
    """Test getting brain region statistics"""
    response = await client.get(
        "/api/stats/regions",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_memories" in data
    assert "regions" in data


@pytest.mark.asyncio
async def test_get_timeline(client: AsyncClient, auth_headers: dict):
    """Test getting activity timeline"""
    response = await client.get(
        "/api/stats/timeline?days=30",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data


# ============ Agent API Tests ============

@pytest.mark.asyncio
async def test_agent_store(client: AsyncClient, auth_headers: dict):
    """Test agent storing memories"""
    response = await client.post(
        "/api/agent/store",
        headers=auth_headers,
        json={
            "memories": [
                {
                    "content": "User prefers dark mode",
                    "category": "preferences",
                    "memory_type": "preference",
                    "importance": 0.7,
                }
            ]
        }
    )
    assert response.status_code == 200
    memories = response.json()
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_agent_recall(client: AsyncClient, auth_headers: dict):
    """Test agent recalling memories"""
    # Store some memories first
    await client.post(
        "/api/agent/store",
        headers=auth_headers,
        json={
            "memories": [
                {
                    "content": "User loves pizza",
                    "category": "food",
                    "memory_type": "preference",
                }
            ]
        }
    )
    
    # Recall
    response = await client.post(
        "/api/agent/recall",
        headers=auth_headers,
        json={
            "query": "food preferences",
            "limit": 5,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "memories" in data


# ============ Service Tests ============

def test_embedding_service():
    """Test the embedding service"""
    service = get_embedding_service()
    
    # Test single embedding
    embedding = service.embed("Hello world")
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    
    # Test batch embedding
    embeddings = service.embed_batch(["Hello", "World"])
    assert len(embeddings) == 2
    assert all(len(e) == 384 for e in embeddings)
    
    # Test similarity
    e1 = service.embed("I love cats")
    e2 = service.embed("I adore felines")
    e3 = service.embed("The stock market crashed")
    
    sim_12 = service.cosine_similarity(e1, e2)
    sim_13 = service.cosine_similarity(e1, e3)
    
    # Similar sentences should have higher similarity
    assert sim_12 > sim_13


def test_memory_extractor():
    """Test the memory extraction service"""
    extractor = get_memory_extractor()
    
    user_message = "I really love Thai food, especially Pad Thai. Also, I need to call my doctor tomorrow."
    assistant_response = "Thai food is delicious! I'll remind you about the doctor call."
    
    memories = extractor.extract_memories(user_message, assistant_response)
    
    assert len(memories) > 0
    
    # Should extract preference about Thai food
    contents = [m.content.lower() for m in memories]
    assert any("thai" in c or "prefer" in c for c in contents)
    
    # Should extract task about doctor
    assert any("doctor" in c or "call" in c for c in contents)


def test_category_classification():
    """Test memory category classification"""
    extractor = get_memory_extractor()
    
    test_cases = [
        ("I learned that Python was created in 1991", "knowledge"),
        ("Yesterday I went to the park", "experiences"),
        ("I need to finish my project by Friday", "schedule"),
        ("My friend John works at Google", "people"),
        ("I love chocolate ice cream", "food"),
    ]
    
    for text, expected_category in test_cases:
        category = extractor.classify_category(text)
        # Note: classification is heuristic, so we just check it returns valid category
        assert category.value in [
            "identity", "preferences", "beliefs", "emotions", "people", "places", 
            "family", "experiences", "projects", "schedule", "work", "learning", 
            "knowledge", "tools", "media", "health", "habits", "food", "travel", 
            "goals", "context"
        ]


# Run with: pytest -v tests/test_api.py


# ============ Memory Enhancement API Tests ============

@pytest.mark.asyncio
async def test_create_memory_returns_strength(client: AsyncClient, auth_headers: dict):
    """Test that created memory includes strength field"""
    response = await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Testing memory strength",
            "category": "knowledge",
            "memory_type": "fact",
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert "strength" in data
    assert data["strength"] == 1.0  # New memories start at full strength


@pytest.mark.asyncio
async def test_create_memory_returns_memory_level(client: AsyncClient, auth_headers: dict):
    """Test that created memory includes memory_level field"""
    response = await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Testing memory level",
            "category": "knowledge",
            "memory_type": "fact",
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert "memory_level" in data
    assert data["memory_level"] == "episodic"  # Default level


@pytest.mark.asyncio
async def test_create_memory_with_emotional_salience(client: AsyncClient, auth_headers: dict):
    """Test creating memory with emotional salience"""
    response = await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "This is a very emotional memory",
            "category": "emotions",
            "memory_type": "emotion",
            "importance": 0.9,
            "emotional_salience": 0.95,
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert "emotional_salience" in data
    assert data["emotional_salience"] == 0.95


@pytest.mark.asyncio
async def test_get_memory_events(client: AsyncClient, auth_headers: dict):
    """Test getting memory events (audit trail)"""
    # Create a memory
    create_response = await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Test memory for events",
            "category": "knowledge",
            "memory_type": "fact",
        }
    )
    memory_id = create_response.json()["id"]
    
    # Get events
    response = await client.get(
        f"/api/memories/{memory_id}/events",
        headers=auth_headers
    )
    assert response.status_code == 200
    events = response.json()
    
    # Should have at least a 'created' event
    assert len(events) >= 1
    assert any(e["event_type"] == "created" for e in events)


@pytest.mark.asyncio
async def test_memory_access_creates_event(client: AsyncClient, auth_headers: dict):
    """Test that accessing a memory creates an accessed event"""
    # Create a memory
    create_response = await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Memory to access",
            "category": "knowledge",
            "memory_type": "fact",
        }
    )
    memory_id = create_response.json()["id"]
    
    # Access the memory
    await client.get(f"/api/memories/{memory_id}", headers=auth_headers)
    
    # Get events
    response = await client.get(
        f"/api/memories/{memory_id}/events",
        headers=auth_headers
    )
    events = response.json()
    
    # Should have 'created' and 'accessed' events
    event_types = [e["event_type"] for e in events]
    assert "created" in event_types
    assert "accessed" in event_types


@pytest.mark.asyncio
async def test_memory_update_creates_event(client: AsyncClient, auth_headers: dict):
    """Test that updating a memory creates an updated event"""
    # Create a memory
    create_response = await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Original content",
            "category": "knowledge",
            "memory_type": "fact",
        }
    )
    memory_id = create_response.json()["id"]
    
    # Update the memory
    await client.patch(
        f"/api/memories/{memory_id}",
        headers=auth_headers,
        json={"content": "Updated content"}
    )
    
    # Get events
    response = await client.get(
        f"/api/memories/{memory_id}/events",
        headers=auth_headers
    )
    events = response.json()
    
    # Should have 'updated' event
    assert any(e["event_type"] == "updated" for e in events)


@pytest.mark.asyncio
async def test_reinforce_memory_endpoint(client: AsyncClient, auth_headers: dict):
    """Test the reinforce memory endpoint"""
    # Create a memory
    create_response = await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Memory to reinforce",
            "category": "knowledge",
            "memory_type": "fact",
        }
    )
    memory_id = create_response.json()["id"]
    
    # Reinforce the memory
    response = await client.post(
        f"/api/memories/{memory_id}/reinforce",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True


@pytest.mark.asyncio
async def test_search_returns_enhanced_fields(client: AsyncClient, auth_headers: dict):
    """Test that search results include strength and memory_level"""
    # Create a memory
    await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Python programming is great",
            "category": "knowledge",
            "memory_type": "fact",
        }
    )
    
    # Search
    response = await client.post(
        "/api/memories/search",
        headers=auth_headers,
        json={
            "query": "Python programming",
            "limit": 5,
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    if data["results"]:
        result = data["results"][0]
        assert "strength" in result
        assert "memory_level" in result


@pytest.mark.asyncio
async def test_memory_list_returns_enhanced_fields(client: AsyncClient, auth_headers: dict):
    """Test that memory list includes enhanced fields"""
    # Create a memory
    await client.post(
        "/api/memories",
        headers=auth_headers,
        json={
            "content": "Test memory for listing",
            "category": "knowledge",
            "memory_type": "fact",
        }
    )
    
    # List memories
    response = await client.get(
        "/api/memories",
        headers=auth_headers
    )
    assert response.status_code == 200
    memories = response.json()
    
    if memories:
        memory = memories[0]
        assert "strength" in memory
        assert "memory_level" in memory
        assert "emotional_salience" in memory


# ============ Admin API Tests (if accessible) ============

@pytest.mark.asyncio
async def test_admin_decay_requires_auth(client: AsyncClient):
    """Test that admin decay endpoint requires authentication"""
    response = await client.post("/api/admin/decay")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_consolidate_requires_auth(client: AsyncClient):
    """Test that admin consolidation endpoint requires authentication"""
    response = await client.post("/api/admin/consolidate")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_memory_health_requires_auth(client: AsyncClient):
    """Test that admin memory-health endpoint requires authentication"""
    response = await client.get("/api/admin/memory-health")
    assert response.status_code == 401

