"""API smoke tests for the routes that still exist.

WHAT LEFT, 2026-08-20 (memory v3)

This file was mostly a driver for the round-8 ROW API — `POST /memories`,
`/{id}/events`, `/{id}/reinforce`, `/category/{c}`, `/region/{r}`, plus the
admin decay/consolidate endpoints. All of that is deleted
(docs/memory/rebuild-2026-08-v3.md §1.1, §4), so those 15 tests retired with
their subject; see tests/RETIRED_WITH_MEMORY_V3.md for the accounting.

What is left has a LIVE subject: auth, the ingest routes (whose message
storage survives even though their extraction was severed), the stats
routes, the `/agent/store` + `/agent/recall` API, and the two pure-function
service tests.

WHY THE AUTH FIXTURE CHANGED, AND WHAT IT DOES NOT CHANGE

Every test here took its token from `POST /api/auth/demo`. That route has
been gated OFF by default since 2026-08-09 (c83c575b, "demo login is an open
door") — it is unauthenticated with the password in source, so it 404s
everywhere, deliberately. The fixture asserted 200, so it raised, so all 19
tests that depend on it ERRORED. That is a PRE-EXISTING red, eleven days
older than this rebuild and nothing to do with memory.

The fixture now mints a token directly, the way tests/conftest.py already
does. No product code changes and the gate is untouched: a TEST should not
depend on a production route that is disabled on purpose, and
`test_demo_login` below now asserts the shipped behaviour (404 when
disabled) instead of the behaviour PR #528 removed.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.config import settings
from app.db import init_db, drop_db, async_session_maker
from app.services import create_user, get_embedding_service, get_memory_extractor
from app.services.auth_service import create_access_token


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
    """A token for a real user, minted directly.

    NOT via `POST /api/auth/demo`: that route is gated off by default and
    404s (see the module docstring), which is why every test in this file
    used to error in setup. Same primitive tests/conftest.py uses.
    """
    async with async_session_maker() as session:
        user = await create_user(
            session, email="apitest@toup.local", password="apitest123456",
            name="API Test User",
        )
        user_id = user.id
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


# ============ Auth Tests ============

@pytest.mark.asyncio
async def test_demo_login_is_closed_unless_explicitly_enabled(client: AsyncClient):
    """PR #528: the demo route is an open door, so it ships disabled.

    It is unauthenticated and its password is in source, so anywhere it is
    reachable anyone can mint a valid session. 404 rather than 403 so the
    disabled route does not advertise its own existence — that is the
    assertion, and asserting 200 (as this test used to) asserts the hole.
    """
    response = await client.post("/api/auth/demo")
    if settings.demo_login_enabled:
        assert response.status_code == 200
        assert "access_token" in response.json()
    else:
        assert response.status_code == 404, (
            "the demo login route answered while disabled — anyone who can "
            "reach this deployment can now mint a session"
        )


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient, auth_headers: dict):
    """Test getting current user info"""
    response = await client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "email" in data


# ============ Memory Tests ============









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



















# ============ Admin API Tests (if accessible) ============





@pytest.mark.asyncio
async def test_admin_memory_health_requires_auth(client: AsyncClient):
    """The per-user memory-health snapshot must not be readable unguarded.

    This asked for `/api/admin/memory-health` — a path that has never
    existed; the route has always been `/memory-health/{user_id}`. So it got
    a 404 from the ROUTER and asserted 401, which means it has never once
    tested authentication. Its subject survives (the route is still there,
    still an operator surface over one user's data), so it is repaired
    rather than retired.
    """
    response = await client.get(
        "/api/admin/memory-health/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 401, (
        "the per-user memory-health snapshot answered without a token"
    )

