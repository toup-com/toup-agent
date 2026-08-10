"""
Behavioral test that the active_tasks block actually reaches the assembled
system prompt — not just that it's built.

Catches the F1 regression class from docs/memory/continuity-audit.md:
  section_parts["active_tasks"] = ...
  ...
  SECTION_ORDER = [..., # MISSING "active_tasks"]
  sections = [section_parts[k] for k in SECTION_ORDER if k in section_parts]
  # ← active_tasks silently dropped at this filter

Pre-existing test_active_task.py only verified the build block was intact and
not intent-gated. It did NOT verify the section reached the assembled prompt.
This file fixes that gap.

Two tests:
  1. test_active_tasks_marker_in_built_prompt — behavioral. Builds a prompt
     with an active_task fixture, asserts <active_tasks> marker survives.
  2. test_built_keys_are_subset_of_section_order — structural / static.
     Catches the F1 bug class for any future section, not just active_tasks.
"""

import asyncio
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Set non-prod environment BEFORE importing app.config (which validates
# Stripe key prefix when ENVIRONMENT=production).
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("USE_DAY_CHAT_CONTEXT", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.db.models.user import User
from app.db.models.memory import Memory


# ── Fixture: minimal sqlite schema covering what _build_system_prompt touches ──

async def _make_engine():
    """Build an in-memory sqlite with the minimum schema _build_system_prompt
    queries against. Mirrors the column set used by tests/test_active_task.py
    plus the identity / agent_config tables the prompt builder reads."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        for stmt in [
            """CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY, email VARCHAR(255) UNIQUE,
                hashed_password VARCHAR(255), name VARCHAR(255),
                password_changed_at TIMESTAMP,
                role VARCHAR(20) DEFAULT 'beta_user', created_at TIMESTAMP,
                updated_at TIMESTAMP, is_active BOOLEAN DEFAULT 1,
                stripe_customer_id VARCHAR(255), timezone VARCHAR(50),
                is_canary BOOLEAN DEFAULT 0
            )""",
            """CREATE TABLE IF NOT EXISTS identities (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36),
                identity_type VARCHAR(30), name VARCHAR(255), content TEXT,
                priority INTEGER DEFAULT 0, is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP, updated_at TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS agent_configs (
                user_id VARCHAR(36) PRIMARY KEY, agent_name VARCHAR(255),
                agent_color VARCHAR(20), onboarding_completed BOOLEAN DEFAULT 1,
                disabled_tools TEXT, created_at TIMESTAMP, updated_at TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS memories (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36),
                brain_type VARCHAR(20) DEFAULT 'user', content TEXT,
                summary VARCHAR(500), category VARCHAR(20),
                memory_type VARCHAR(20),
                embedding_json TEXT, importance FLOAT DEFAULT 0.5,
                confidence FLOAT DEFAULT 1.0, strength FLOAT DEFAULT 1.0,
                memory_level VARCHAR(20) DEFAULT 'episodic',
                emotional_salience FLOAT DEFAULT 0.5,
                last_reinforced_at TIMESTAMP,
                consolidation_count INTEGER DEFAULT 0,
                decay_rate FLOAT DEFAULT 0.1,
                created_at TIMESTAMP, updated_at TIMESTAMP,
                last_accessed_at TIMESTAMP, access_count INTEGER DEFAULT 0,
                source_message_id VARCHAR(36),
                source_type VARCHAR(50) DEFAULT 'conversation',
                metadata_json TEXT, tags_json TEXT, canonical_content TEXT,
                history_json TEXT, merged_from_json TEXT,
                superseded_by VARCHAR(36),
                is_active BOOLEAN DEFAULT 1, is_deleted BOOLEAN DEFAULT 0,
                deleted_at TIMESTAMP, embedding BLOB, search_vector TEXT
            )""",
        ]:
            await conn.run_sync(lambda c, s=stmt: c.execute(text(s)))
    return engine


# ── Test 1: behavioral — section actually reaches the assembled prompt ──

async def test_active_tasks_marker_in_built_prompt():
    """F1 regression test: when an active_task Memory exists, the
    `<active_tasks>` marker MUST appear in the assembled system prompt.

    This is a behavioral test — it calls the real `_build_system_prompt`
    and inspects the returned string. It explicitly catches the bug class
    where a section is built into `section_parts` but its key is missing
    from `SECTION_ORDER`, so the filter at assembly time silently drops it.
    """
    from app.agent.agent_runner import AgentRunner

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    user_id = str(uuid.uuid4())

    async with sm() as db:
        db.add(User(
            id=user_id, email="t@t.local", hashed_password="x",
            name="Test", role="beta_user", is_active=True, is_canary=False,
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        db.add(Memory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            brain_type="user",
            content="Debugging the CSS sidebar alignment issue",
            category="active_task",
            memory_type="semantic",
            importance=0.9,
            confidence=1.0,
            strength=1.0,
            memory_level="working",
            decay_rate=0.05,
            is_active=True,
            is_deleted=False,
            source_type="active_task_extraction",
            last_reinforced_at=datetime.utcnow(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ))
        await db.commit()

    # Stub out heavy services that would call OpenAI / embedding APIs.
    # The active_tasks path doesn't depend on either — it's a plain SQL query —
    # so stubbing these doesn't compromise the F1 assertion.
    runner = AgentRunner(llm_service=AsyncMock(), tool_executor=AsyncMock())

    with patch(
        "app.services.memory_service.MemoryService.hybrid_search",
        AsyncMock(return_value=[]),
    ), patch(
        "app.services.user_portrait_service.UserPortraitService.get_or_build_portrait",
        AsyncMock(return_value=""),
    ):
        async with sm() as db:
            prompt = await runner._build_system_prompt(
                db=db,
                user_id=user_id,
                user_message="hello",
            )

    assert "<active_tasks>" in prompt, (
        "F1 regression: <active_tasks> marker missing from assembled prompt. "
        "Either 'active_tasks' is missing from SECTION_ORDER in "
        "agent_runner.py, or the build path is broken. "
        "See docs/memory/continuity-audit.md F1."
    )
    assert "Debugging the CSS sidebar" in prompt, (
        "Active task content is missing from the prompt body."
    )

    await engine.dispose()


# ── Test 2: structural — catches F1 bug class for any future section ──

def test_built_keys_are_subset_of_section_order():
    """Structural invariant: every key assigned to `section_parts` in
    `_build_system_prompt` MUST appear in `SECTION_ORDER`. Anything else
    is silently dropped at assembly — that's the F1 bug class.

    This test runs without instantiating the runner or hitting a DB —
    it's a pure source grep. It generalises beyond active_tasks: any
    future section with the same bug pattern fails this test.
    """
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "app" / "agent" / "agent_runner.py"
    ).read_text()

    # Every section_parts["KEY"] = ... assignment in the prompt builder.
    assigned = set(re.findall(r'section_parts\["([a-z_]+)"\]\s*=', src))

    # The SECTION_ORDER list contents.
    order_match = re.search(r'SECTION_ORDER\s*=\s*\[(.*?)\]', src, re.DOTALL)
    assert order_match, (
        "SECTION_ORDER list not found in agent_runner.py — "
        "test cannot verify the invariant."
    )
    order_keys = set(re.findall(r'"([a-z_]+)"', order_match.group(1)))

    dropped = assigned - order_keys
    assert not dropped, (
        f"section_parts keys assigned but missing from SECTION_ORDER: "
        f"{sorted(dropped)}. These sections are silently dropped at "
        f"assembly time. See docs/memory/continuity-audit.md F1 for the "
        f"bug class. Add the missing key(s) to SECTION_ORDER, picking "
        f"the right position for the section's intent."
    )

    # Spot-check: active_tasks must be in both halves (current F1 fix).
    assert "active_tasks" in assigned, (
        "section_parts['active_tasks'] assignment missing from "
        "agent_runner.py — the build block was removed?"
    )
    assert "active_tasks" in order_keys, (
        "active_tasks missing from SECTION_ORDER — F1 has regressed."
    )


# ── Allow direct invocation: `python tests/agent/test_system_prompt_assembly.py` ──

if __name__ == "__main__":
    test_built_keys_are_subset_of_section_order()
    print("OK structural")
    asyncio.run(test_active_tasks_marker_in_built_prompt())
    print("OK behavioral")
