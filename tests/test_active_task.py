"""
Tests for active_task memory extraction, TTL decay, and injection.

Run: python tests/test_active_task.py
"""

import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.db.models.user import User
from app.db.models.memory import Memory

# Direct import to avoid heavy app.services.__init__
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "active_task_service",
    str(Path(__file__).resolve().parent.parent / "app" / "services" / "active_task_service.py"),
)
_ats = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ats)

detect_active_tasks = _ats.detect_active_tasks
store_active_task = _ats.store_active_task
decay_expired_tasks = _ats.decay_expired_tasks
get_active_tasks = _ats.get_active_tasks
build_active_tasks_block = _ats.build_active_tasks_block


async def _make_engine():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        for stmt in [
            """CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY, email VARCHAR(255) UNIQUE,
                hashed_password VARCHAR(255), password_plain VARCHAR(255),
                name VARCHAR(255), password_changed_at TIMESTAMP,
                role VARCHAR(20) DEFAULT 'beta_user', created_at TIMESTAMP,
                updated_at TIMESTAMP, is_active BOOLEAN DEFAULT 1,
                stripe_customer_id VARCHAR(255), timezone VARCHAR(50)
            )""",
            """CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR(50) PRIMARY KEY, conversation_id VARCHAR(36),
                day_chat_id VARCHAR(36), role VARCHAR(20), content TEXT,
                created_at TIMESTAMP, tokens_prompt INTEGER,
                tokens_completion INTEGER, model_used VARCHAR(50),
                memories_retrieved_json TEXT, processing_time_ms INTEGER,
                metadata_json TEXT, embedding_json TEXT, embedding BLOB
            )""",
            """CREATE TABLE IF NOT EXISTS memories (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36),
                brain_type VARCHAR(20) DEFAULT 'user', content TEXT,
                summary VARCHAR(500), category VARCHAR(20), memory_type VARCHAR(20),
                embedding_json TEXT, importance FLOAT DEFAULT 0.5,
                confidence FLOAT DEFAULT 1.0, strength FLOAT DEFAULT 1.0,
                memory_level VARCHAR(20) DEFAULT 'episodic',
                emotional_salience FLOAT DEFAULT 0.5,
                last_reinforced_at TIMESTAMP, consolidation_count INTEGER DEFAULT 0,
                decay_rate FLOAT DEFAULT 0.1, created_at TIMESTAMP, updated_at TIMESTAMP,
                last_accessed_at TIMESTAMP, access_count INTEGER DEFAULT 0,
                source_message_id VARCHAR(36), source_type VARCHAR(50) DEFAULT 'conversation',
                metadata_json TEXT, tags_json TEXT, canonical_content TEXT,
                history_json TEXT, merged_from_json TEXT, superseded_by VARCHAR(36),
                is_active BOOLEAN DEFAULT 1, is_deleted BOOLEAN DEFAULT 0, deleted_at TIMESTAMP,
                embedding BLOB, search_vector TEXT
            )""",
        ]:
            await conn.run_sync(lambda c, s=stmt: c.execute(text(s)))
    return engine


# ── Tests ─────────────────────────────────────────────────────────────

async def test_detect_active_tasks_patterns():
    """Pattern matching finds active task descriptions in user messages."""
    positives = [
        "I'm working on the sidebar CSS bug",
        "We're debugging the login flow",
        "Still need to fix the deployment script",
        "I'm building a React component for the dashboard",
        "Next step is to deploy to production",
        "I'm in the middle of refactoring the auth module",
        "Can you help me continue working on the API?",
        "Currently trying to figure out why the tests fail",
    ]
    for msg in positives:
        tasks = detect_active_tasks(msg, "Sure, I can help.")
        assert len(tasks) > 0, f"Expected to detect task in: '{msg}'"

    negatives = [
        "Hello",
        "What's the weather?",
        "Tell me a joke",
        "Thanks!",
    ]
    for msg in negatives:
        tasks = detect_active_tasks(msg, "Here you go.")
        assert len(tasks) == 0, f"False positive for: '{msg}', got {tasks}"


async def test_store_active_task_creates_memory():
    """Storing an active task creates a Memory with category=active_task."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u1", email="u1@test.local", hashed_password="x", name="Test"))
        await db.commit()

    async with sm() as db:
        mem_id = await store_active_task(db, "u1", "Debugging the CSS sidebar alignment issue")
        await db.commit()

    assert mem_id is not None

    async with sm() as db:
        mem = (await db.execute(select(Memory).where(Memory.id == mem_id))).scalar_one()
        assert mem.category == "active_task"
        assert mem.brain_type == "user"
        assert mem.importance == 0.9
        assert mem.is_active is True
        assert "sidebar" in mem.content.lower()

    await engine.dispose()


async def test_store_active_task_reinforces_existing():
    """Storing a similar task reinforces the existing one instead of creating a duplicate."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u2", email="u2@test.local", hashed_password="x", name="Test"))
        await db.commit()

    # Create initial task
    async with sm() as db:
        id1 = await store_active_task(db, "u2", "Working on the sidebar CSS bug")
        await db.commit()

    # Store similar task — should reinforce, not duplicate
    async with sm() as db:
        id2 = await store_active_task(db, "u2", "Still working on the sidebar CSS bug fix")
        await db.commit()

    assert id1 == id2, f"Should have reinforced same task, got different IDs: {id1} vs {id2}"

    # Only one active_task memory in DB
    async with sm() as db:
        count = (await db.execute(
            select(func.count()).select_from(Memory).where(
                Memory.category == "active_task"
            )
        )).scalar()
    assert count == 1

    await engine.dispose()


async def test_decay_expired_tasks():
    """Tasks older than 7 days without reinforcement are archived."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u3", email="u3@test.local", hashed_password="x", name="Test"))

        # Old task (8 days ago, never reinforced)
        old_task = Memory(
            id=str(uuid.uuid4()), user_id="u3", brain_type="user",
            content="Old task that expired", category="active_task",
            memory_type="semantic", importance=0.9, strength=1.0,
            created_at=datetime.utcnow() - timedelta(days=8),
            is_active=True,
        )
        # Fresh task (1 day ago)
        fresh_task = Memory(
            id=str(uuid.uuid4()), user_id="u3", brain_type="user",
            content="Fresh task still active", category="active_task",
            memory_type="semantic", importance=0.9, strength=1.0,
            created_at=datetime.utcnow() - timedelta(days=1),
            is_active=True,
        )
        # Reinforced task (created 10 days ago, reinforced 2 days ago)
        reinforced_task = Memory(
            id=str(uuid.uuid4()), user_id="u3", brain_type="user",
            content="Reinforced task", category="active_task",
            memory_type="semantic", importance=0.9, strength=1.0,
            created_at=datetime.utcnow() - timedelta(days=10),
            last_reinforced_at=datetime.utcnow() - timedelta(days=2),
            is_active=True,
        )
        db.add_all([old_task, fresh_task, reinforced_task])
        await db.commit()
        old_id = old_task.id
        fresh_id = fresh_task.id
        reinforced_id = reinforced_task.id

    # Run decay
    async with sm() as db:
        archived = await decay_expired_tasks(db, "u3")
        await db.commit()

    assert archived == 1  # Only the old task

    # Verify states
    async with sm() as db:
        old = (await db.execute(select(Memory).where(Memory.id == old_id))).scalar_one()
        fresh = (await db.execute(select(Memory).where(Memory.id == fresh_id))).scalar_one()
        reinforced = (await db.execute(select(Memory).where(Memory.id == reinforced_id))).scalar_one()

        assert old.is_active is False, "Old task should be archived"
        assert fresh.is_active is True, "Fresh task should still be active"
        assert reinforced.is_active is True, "Reinforced task should still be active"

    await engine.dispose()


async def test_get_active_tasks_returns_only_active():
    """get_active_tasks only returns active, non-deleted tasks."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u4", email="u4@test.local", hashed_password="x", name="Test"))
        db.add(Memory(
            id="active-1", user_id="u4", brain_type="user",
            content="Active task 1", category="active_task",
            memory_type="semantic", is_active=True,
            created_at=datetime.utcnow(),
        ))
        db.add(Memory(
            id="archived-1", user_id="u4", brain_type="user",
            content="Archived task", category="active_task",
            memory_type="semantic", is_active=False,
            created_at=datetime.utcnow(),
        ))
        db.add(Memory(
            id="other-1", user_id="u4", brain_type="user",
            content="Regular memory", category="work",
            memory_type="semantic", is_active=True,
            created_at=datetime.utcnow(),
        ))
        await db.commit()

    async with sm() as db:
        tasks = await get_active_tasks(db, "u4")

    assert len(tasks) == 1
    assert tasks[0]["id"] == "active-1"
    assert "Active task 1" in tasks[0]["content"]


async def test_build_active_tasks_block():
    """System prompt block is formatted correctly."""
    tasks = [
        {"content": "Debugging CSS sidebar", "created_at": "2026-04-07T09:00:00", "last_reinforced_at": None, "strength": 1.0},
        {"content": "Building React dashboard", "created_at": "2026-04-06T14:00:00", "last_reinforced_at": "2026-04-07T10:00:00", "strength": 1.0},
    ]
    block = build_active_tasks_block(tasks)
    assert "<active_tasks>" in block
    assert "</active_tasks>" in block
    assert "Debugging CSS sidebar" in block
    assert "Building React dashboard" in block
    assert "last mentioned:" in block  # For reinforced task

    # Empty tasks = empty block
    assert build_active_tasks_block([]) == ""


async def test_active_tasks_always_injected():
    """Active tasks are injected regardless of intent — guards against re-adding skip logic."""
    # This test verifies the code path in agent_runner.py where active tasks
    # are injected AFTER the memory retrieval section (which no longer has
    # skip_memory_retrieval gating). The injection is unconditional.

    # We test this by checking the build_active_tasks_block works with
    # any intent type's output — it doesn't check intent at all.
    tasks = [{"content": "Active task", "created_at": "2026-04-07T09:00:00",
              "last_reinforced_at": None, "strength": 1.0}]

    block = build_active_tasks_block(tasks)
    assert block != "", "Active tasks block should not be empty when tasks exist"
    assert "<active_tasks>" in block

    # Also verify via grep that agent_runner.py has no intent gate on active_tasks
    import re
    runner_code = open(str(Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py")).read()

    # Find the active_tasks injection block
    match = re.search(r'# ── 3b\. Active tasks.*?# ── 4\.', runner_code, re.DOTALL)
    assert match, "Could not find active_tasks injection block in agent_runner.py"

    block_code = match.group()
    # There should be NO reference to intent.skip or intent.category in this block
    assert "intent.skip" not in block_code, "Active tasks must not be gated by intent"
    assert "intent.category" not in block_code, "Active tasks must not be gated by intent category"


async def test_no_skip_memory_retrieval_references():
    """Grep test: codebase contains zero functional references to skip_memory_retrieval."""
    import os

    backend_dir = str(Path(__file__).resolve().parent.parent)
    hits = []

    for root, dirs, files in os.walk(backend_dir):
        # Skip __pycache__, .git, node_modules
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", "venv")]
        for f in files:
            if not f.endswith(".py"):
                continue
            filepath = os.path.join(root, f)
            with open(filepath) as fh:
                for i, line in enumerate(fh, 1):
                    if "skip_memory_retrieval" in line:
                        # Allow comments (lines starting with # or containing "NOTE:" / "removed")
                        stripped = line.strip()
                        if stripped.startswith("#") or "NOTE:" in stripped or "removed" in stripped.lower() or "Previously" in stripped:
                            continue
                        # Allow test files referencing it
                        if "test_" in f:
                            continue
                        hits.append(f"{filepath}:{i}: {stripped}")

    assert not hits, f"Found {len(hits)} functional references to skip_memory_retrieval:\n" + "\n".join(hits)


# ── Standalone runner ─────────────────────────────────────────────────

if __name__ == "__main__":
    async def _run_all():
        tests = [
            ("test_detect_active_tasks_patterns", test_detect_active_tasks_patterns),
            ("test_store_active_task_creates_memory", test_store_active_task_creates_memory),
            ("test_store_active_task_reinforces_existing", test_store_active_task_reinforces_existing),
            ("test_decay_expired_tasks", test_decay_expired_tasks),
            ("test_get_active_tasks_returns_only_active", test_get_active_tasks_returns_only_active),
            ("test_build_active_tasks_block", test_build_active_tasks_block),
            ("test_active_tasks_always_injected", test_active_tasks_always_injected),
            ("test_no_skip_memory_retrieval_references", test_no_skip_memory_retrieval_references),
        ]
        passed = 0
        failed = 0
        for name, fn in tests:
            try:
                await fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
        return failed == 0

    success = asyncio.run(_run_all())
    sys.exit(0 if success else 1)
