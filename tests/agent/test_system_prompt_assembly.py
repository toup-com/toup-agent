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
from app.db.models.memory import Memory, MemoryFile


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
        # `users` comes FROM THE ORM MODEL, never from a copy of it. The copy
        # that used to live here went red the moment the model gained
        # `first_media_played_at` (migration 086), because the ORM insert below
        # emits every mapped column. A hand-written schema is a second source
        # of truth nothing keeps in sync; the tables below stay raw because
        # they are seeded by raw SQL, not by the ORM.
        await conn.run_sync(User.__table__.create, checkfirst=True)
        # `memories` is ORM-seeded too (the tests below add Memory objects),
        # so it gets the same treatment — a hand-written copy went red when
        # the model gained file_slug/file_position (memory files, 2026-08).
        # memory_files comes along because _build_system_prompt's user_brain
        # now reads the file index.
        await conn.run_sync(Memory.__table__.create, checkfirst=True)
        await conn.run_sync(MemoryFile.__table__.create, checkfirst=True)
        for stmt in [
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
                # NOT "hello". TKT-LAT-019 added a trivial-query skip that
                # short-circuits the active_tasks DB load entirely (a
                # one-word answer does not need to know what threads are
                # open), so a greeting exercises the skip branch rather
                # than the assembly this test exists to guard.
                user_message="what am I in the middle of working on right now?",
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
    `_build_system_prompt` MUST appear in the FULL profile's section order.
    Anything else is silently dropped at assembly — that's the F1 bug class.

    SECTION_ORDER is no longer a literal list in agent_runner.py — it's
    `list(sections_for(_profile))`, resolved per-run from
    app/agent/prompt_profile.py. FULL is the superset profile (every other
    profile is pinned as a strict subset of it there), so a key missing
    from FULL is dropped for every profile.

    Assignments are still gathered by source grep — a key assigned inside a
    conditional branch our fixtures never trigger must still be ordered.
    It generalises beyond active_tasks: any future section with the same
    bug pattern fails this test.
    """
    from app.agent.prompt_profile import PromptProfile, sections_for

    src = (
        Path(__file__).resolve().parent.parent.parent
        / "app" / "agent" / "agent_runner.py"
    ).read_text()

    # Every section_parts["KEY"] = ... assignment in the prompt builder.
    assigned = set(re.findall(r'section_parts\["([a-z_]+)"\]', src))
    assert assigned, (
        "No section_parts[...] assignments found in agent_runner.py — "
        "the builder was restructured and this grep no longer sees it; "
        "repoint the test."
    )

    order_keys = set(sections_for(PromptProfile.FULL))

    dropped = assigned - order_keys
    assert not dropped, (
        f"section_parts keys assigned in _build_system_prompt but missing "
        f"from the FULL profile's section order: {sorted(dropped)}. These "
        f"sections are silently dropped at assembly time. See "
        f"docs/memory/continuity-audit.md F1 for the bug class. Add the "
        f"missing key(s) to _FULL_SECTIONS in app/agent/prompt_profile.py, "
        f"picking the right position for the section's intent."
    )

    # Spot-check: active_tasks must be in both halves (current F1 fix).
    assert "active_tasks" in assigned, (
        "section_parts['active_tasks'] assignment missing from "
        "agent_runner.py — the build block was removed?"
    )
    assert "active_tasks" in order_keys, (
        "active_tasks missing from the FULL profile's section order — "
        "F1 has regressed."
    )


# ── Allow direct invocation: `python tests/agent/test_system_prompt_assembly.py` ──

if __name__ == "__main__":
    test_built_keys_are_subset_of_section_order()
    print("OK structural")
    asyncio.run(test_active_tasks_marker_in_built_prompt())
    print("OK behavioral")
