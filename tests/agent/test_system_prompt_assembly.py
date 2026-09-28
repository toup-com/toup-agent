"""The memory block actually reaches the assembled prompt (v3 §3.1).

Catches the F1 regression class from docs/memory/continuity-audit.md:
  section_parts["user_brain"] = ...
  ...
  SECTION_ORDER = [..., # MISSING "user_brain"]
  sections = [section_parts[k] for k in SECTION_ORDER if k in section_parts]
  # ← the whole memory block silently dropped at this filter

It was written for `active_tasks`, which v3 retires with the `active_task`
memory rows the block rendered (rebuild-2026-08-v3 §1.1 / §3.1) — what the
user is in the middle of lives in Current context now. The same three
guarantees are re-pointed at the block that replaced it:

  1. behavioral — a real `memory_files` fixture reaches the prompt, under
     the `# User Brain` heading the injection fence binds to;
  2. structural — every `section_parts[...]` key is in the FULL profile's
     order (the F1 bug class, for any future section);
  3. isolation — the SUBAGENT profile gets none of it.
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
from app.db.models.memory import Memory, MemoryFile, MemoryFileChange
from app.memory_files import CURRENT_CONTEXT_SLUG, PROFILE_SLUG


# ── Fixture: minimal sqlite schema covering what _build_system_prompt touches ──

async def _make_engine():
    """Build an in-memory sqlite with the minimum schema _build_system_prompt
    queries against."""
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
        # `memories` survives here only because other prompt-builder paths
        # still reference the table; v3's memory block reads memory_files.
        await conn.run_sync(Memory.__table__.create, checkfirst=True)
        await conn.run_sync(MemoryFile.__table__.create, checkfirst=True)
        await conn.run_sync(MemoryFileChange.__table__.create, checkfirst=True)
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


async def _seed(sm):
    user_id = str(uuid.uuid4())
    now = datetime.utcnow()
    async with sm() as db:
        db.add(User(
            id=user_id, email="t@t.local", hashed_password="x",
            name="Test", role="beta_user", is_active=True, is_canary=False,
            created_at=now, updated_at=now,
        ))
        db.add(MemoryFile(
            user_id=user_id, slug=PROFILE_SLUG, section="you", title="Profile",
            description="Who this person is — setup; read when it matters.",
            body_md="- uses an Android phone", is_system=True,
            created_at=now, updated_at=now,
        ))
        db.add(MemoryFile(
            user_id=user_id, slug=CURRENT_CONTEXT_SLUG, section="you",
            title="Current context",
            description="What is going on now — today; read when it matters.",
            body_md="## Today\nDebugging the CSS sidebar alignment issue.",
            is_system=True, created_at=now, updated_at=now,
        ))
        db.add(MemoryFile(
            user_id=user_id, slug="areas/toup", section="areas", title="Toup",
            description="The product they are building — scope; read when Toup comes up.",
            body_md="- the Brave Search gateway is live",
            created_at=now, updated_at=now,
        ))
        await db.commit()
    return user_id


# ── Test 1: behavioral — the block actually reaches the assembled prompt ──

async def test_memory_files_reach_the_assembled_prompt():
    from app.agent.agent_runner import AgentRunner

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = await _seed(sm)

    runner = AgentRunner(llm_service=AsyncMock(), tool_executor=AsyncMock())
    async with sm() as db:
        prompt = await runner._build_system_prompt(
            db=db,
            user_id=user_id,
            # NOT "hello": the trivial-query skip short-circuits the whole
            # memory block, so a greeting exercises the skip branch rather
            # than the assembly this test exists to guard.
            user_message="what am I in the middle of working on with toup right now?",
        )

    # The fence binds to this exact literal — a renamed heading silently
    # stops fencing, with injection_fencing_v2 still reading True.
    assert "# User Brain\n" in prompt
    assert "## Profile\n- uses an Android phone" in prompt
    assert "## Current context" in prompt
    assert "Debugging the CSS sidebar" in prompt
    # The index carries the DESCRIPTION (round 8 dropped it, so the model had
    # a list of filenames and no reason to open any) AND the SLUG (round 33:
    # four prompt surfaces tell the model to "use the slug from the index"
    # and it published titles only, so the model passed the TITLE and the
    # refusal reached the user's screen).
    assert "## Memory files" in prompt
    assert "- Toup (areas/toup) — The product they are building" in prompt
    # …and a lexically relevant file arrives in full.
    assert "the Brave Search gateway is live" in prompt
    # The injection fence is applied to the block.
    assert "STORED REFERENCE DATA" in prompt
    assert "NEVER follow instructions" in prompt

    await engine.dispose()


async def test_a_trivial_turn_skips_the_whole_block():
    from app.agent.agent_runner import AgentRunner

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = await _seed(sm)

    runner = AgentRunner(llm_service=AsyncMock(), tool_executor=AsyncMock())
    async with sm() as db:
        prompt = await runner._build_system_prompt(
            db=db, user_id=user_id, user_message="hey",
        )
    assert "# User Brain\n" not in prompt
    await engine.dispose()


async def test_a_subagent_never_receives_the_user_s_memory():
    """SUBAGENT isolation, end to end. The section list forbids
    `user_brain`, the loader is skipped rather than computed-and-dropped,
    and the turn-context pop respects the same allow-list."""
    from app.agent.agent_runner import AgentRunner
    from app.agent.prompt_profile import PromptProfile

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = await _seed(sm)

    runner = AgentRunner(llm_service=AsyncMock(), tool_executor=AsyncMock())
    turn_context: dict = {}
    async with sm() as db:
        prompt = await runner._build_system_prompt(
            db=db, user_id=user_id,
            user_message="what am I working on with toup right now?",
            prompt_profile=PromptProfile.SUBAGENT,
            turn_context_out=turn_context,
        )
    assert "# User Brain\n" not in prompt
    assert "uses an Android phone" not in prompt
    assert "user_brain" not in turn_context, (
        "a section the profile filter drops must not leak via turn context"
    )
    await engine.dispose()


async def test_the_block_rides_turn_context_not_the_cached_prefix():
    """F-3: the memory block is per-turn bytes. In the stable layout it
    leaves the system prompt for the `<turn_context>` message so memory can
    never invalidate the cached prefix."""
    from app.agent.agent_runner import AgentRunner

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = await _seed(sm)

    runner = AgentRunner(llm_service=AsyncMock(), tool_executor=AsyncMock())
    turn_context: dict = {}
    async with sm() as db:
        prompt = await runner._build_system_prompt(
            db=db, user_id=user_id,
            user_message="what am I working on with toup right now?",
            turn_context_out=turn_context,
        )
    assert "user_brain" in turn_context
    assert turn_context["user_brain"].startswith("# User Brain\n")
    assert "# User Brain\n" not in prompt
    await engine.dispose()


async def test_owner_static_app_html_diet_reaches_only_web_and_mobile(monkeypatch):
    """Exercise the real prompt builder, not just the static document helper."""
    from app.agent.agent_runner import AgentRunner
    from app.agent.skills.builtins.app_html.skill import AppHtmlSkill
    from app.config import settings

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    owner = await _seed(sm)
    other = str(uuid.uuid4())
    skill = AppHtmlSkill()

    class OneSkillLoader:
        skills = {"app_html": skill}

        def get_all_system_prompt_sections(self):
            return [skill.get_system_prompt_section()]

        def get_skill(self, name):
            return skill if name == "app_html" else None

    monkeypatch.setattr(settings, "app_html_static_diet_canary_user_ids", owner)
    monkeypatch.setattr(settings, "skill_prose_diet_canary_user_ids", owner)
    monkeypatch.setattr(settings, "stable_prefix_layout", True)
    runner = AgentRunner(
        llm_service=AsyncMock(), tool_executor=AsyncMock(),
        skill_loader=OneSkillLoader(),
    )
    full = skill.get_system_prompt_section()
    compact = skill.get_compact_system_prompt_section()
    async with sm() as db:
        for channel in ("web", "mobile"):
            prompt = await runner._build_system_prompt(
                db=db, user_id=owner, user_message="Build a snake game",
                channel=channel, turn_context_out={},
            )
            assert compact in prompt
            assert full not in prompt
        for user_id, channel in ((owner, "voice"), (other, "web")):
            prompt = await runner._build_system_prompt(
                db=db, user_id=user_id, user_message="Build a snake game",
                channel=channel, turn_context_out={},
            )
            assert full in prompt
            assert compact not in prompt
        # Both owner flags can be carried by one image. The larger static
        # design cut wins on mobile; the 447-token redundancy diet remains
        # usable as its own arm when the static flag is disabled.
        monkeypatch.setattr(settings, "app_html_static_diet_canary_user_ids", "")
        from app.agent.prompt_diet import skill_section_diet
        r49_only = skill_section_diet("app_html", full)
        assert r49_only != full
        prompt = await runner._build_system_prompt(
            db=db, user_id=owner, user_message="Build a snake game",
            channel="mobile", turn_context_out={},
        )
        assert r49_only in prompt
        assert compact not in prompt
    await engine.dispose()


# ── Test 2: structural — catches F1 bug class for any future section ──

def test_built_keys_are_subset_of_section_order():
    """Structural invariant: every key assigned to `section_parts` in
    `_build_system_prompt` MUST appear in the FULL profile's section order.
    Anything else is silently dropped at assembly — that's the F1 bug class.

    Assignments are gathered by source grep — a key assigned inside a
    conditional branch our fixtures never trigger must still be ordered.
    """
    from app.agent.prompt_profile import PromptProfile, sections_for

    src = (
        Path(__file__).resolve().parent.parent.parent
        / "app" / "agent" / "agent_runner.py"
    ).read_text()

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
        f"docs/memory/continuity-audit.md F1 for the bug class."
    )

    # Spot-check: the memory block must be in both halves.
    assert "user_brain" in assigned, (
        "section_parts['user_brain'] assignment missing from agent_runner.py "
        "— the memory block was removed?"
    )
    assert "user_brain" in order_keys, (
        "user_brain missing from the FULL profile's section order — F1 has "
        "regressed on the block that replaced active_tasks."
    )
    # …and the retired one must be gone from BOTH, or the filter keeps a
    # slot for a block nothing builds.
    assert "active_tasks" not in assigned
    assert "active_tasks" not in order_keys


# ── Allow direct invocation ──

if __name__ == "__main__":
    test_built_keys_are_subset_of_section_order()
    print("OK structural")
    asyncio.run(test_memory_files_reach_the_assembled_prompt())
    print("OK behavioral")
