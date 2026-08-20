"""The OTHER two connection-leak sites, and the reason they were missed.

WHY THIS EXISTS

#407 fixed `day_summarizer.generate_summary` / `generate_archival_summary`:
they held a pooled AsyncSession across a multi-second LLM round-trip, and
because the caller is fire-and-forget on a path that cancels routinely (a
voice caller hanging up cancels the parent 1.5s later), a death mid-call left
the connection checked out forever. The GC then terminated it, the pool
degraded, and later turns died on PendingRollbackError -> HTTP 500 from
/internal/agent-turn. Measured on the canary 2026-08-01: turn latency
14s -> 88s -> 148s under sustained load, then every turn 500'd.

The follow-up scan (5e722750) concluded those two "appear to be the only
remaining instances". They are not. The scan's LLM predicate was
`call_system_llm` / `call_openai_system` / `call_anthropic_system`, and BOTH
remaining sites reach the model through `LLMService.complete_with_json`
instead, so no depth of call-graph walking could have matched them:

    agent_runner._extract_memories                      [v3: now the curator]
        -> memory_extractor.extract_memories_with_llm   [v3: curate_turn]
        -> _complete_json_with_retry -> LLMService.complete_with_json

    agent_reflection.reflect_on_turn
        -> extract_agent_reflections -> LLMService.complete_with_json

Both are reached from `_background_post_processing`, the SAME fire-and-forget
task, and extraction is the hotter of all four sites: the summarizer is
debounced, extraction runs on every non-trivial turn.

The scan's stated limits (annotation-keyed, three levels deep) did not cover
this one, so the rule is worth writing down: the predicate has to be "reaches
a provider", not "reaches one named helper".

Also covers the second, independent defect on the same path — a bare
`asyncio.create_task(...)` whose result nobody stores. The loop keeps only a
weak reference, so the task can be collected mid-await; that is the
cancellation. Releasing the connection first makes the death harmless,
keeping a reference stops it happening.

Run:
  cd backend && pytest tests/test_background_connection_leak.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text as _sqltext
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.agent_reflection as AREF


# ──────────────────────────────────────────────────────────────
# Harness
# ──────────────────────────────────────────────────────────────

async def _agent_config_engine(user_id: str, api_key: str = "sk-tenant-key"):
    """sqlite engine carrying a real `agent_configs` row.

    The real table, via ORM metadata — `_resolve_tenant_api_key` must perform
    a REAL read for this test to mean anything. A read is what autobegins the
    transaction that pins the connection.
    """
    from app.db import AgentConfig

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda c: AgentConfig.metadata.create_all(c, tables=[AgentConfig.__table__])
        )
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        db.add(AgentConfig(id=str(uuid.uuid4()), user_id=user_id, openai_api_key=api_key))
        await db.commit()
    return engine, sm


# ──────────────────────────────────────────────────────────────
# reflect_on_turn — the runtime assertion
# ──────────────────────────────────────────────────────────────

class TestReflectOnTurn:
    @pytest.mark.asyncio
    async def test_connection_released_before_the_llm_call(self):
        user_id = str(uuid.uuid4())
        engine, sm = await _agent_config_engine(user_id)
        seen: dict = {}
        holder: dict = {}

        async def _probe(user_message, assistant_response, api_key=None):
            # in_transaction() is sync on AsyncSession and reports whether a
            # connection is currently checked out for this session.
            seen["in_transaction"] = holder["db"].in_transaction()
            seen["api_key"] = api_key
            return []

        with patch.object(AREF, "should_reflect", lambda _m: True), \
             patch.object(AREF, "extract_agent_reflections", _probe):
            async with sm() as db:
                holder["db"] = db
                await AREF.reflect_on_turn(db, user_id, "always call me Nariman", "ok")

        assert "in_transaction" in seen, "the LLM call never happened — test is vacuous"
        assert seen["in_transaction"] is False, (
            "reflect_on_turn held the pooled connection across the LLM call "
            "(idle-in-transaction; leaks outright if the task is cancelled)"
        )
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_the_tenant_key_is_still_resolved(self):
        """Load-bearing companion: without a real read there is no transaction
        to release, and the test above would pass on a no-op implementation.
        This pins that the read still happens and its result still reaches the
        model — i.e. the commit was added, not the query removed."""
        user_id = str(uuid.uuid4())
        engine, sm = await _agent_config_engine(user_id, api_key="sk-the-tenants-own")
        seen: dict = {}

        async def _probe(user_message, assistant_response, api_key=None):
            seen["api_key"] = api_key
            return []

        with patch.object(AREF, "should_reflect", lambda _m: True), \
             patch.object(AREF, "extract_agent_reflections", _probe):
            async with sm() as db:
                await AREF.reflect_on_turn(db, user_id, "always call me Nariman", "ok")

        assert seen["api_key"] == "sk-the-tenants-own"
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_a_failed_release_does_not_cost_the_turn(self):
        """The release is an optimisation; the reflection is the user's data.

        Caught by CI on the first cut of this change: the commit was inside the
        caller's try, so a session that could not commit skipped the work
        entirely and logged 'extraction failed'. A pinned connection is a much
        smaller loss than a dropped turn — so a failing release must degrade to
        'connection stays pinned', never to 'nothing ran'."""
        user_id = str(uuid.uuid4())
        engine, sm = await _agent_config_engine(user_id)
        seen: dict = {}

        async def _probe(user_message, assistant_response, api_key=None):
            seen["called"] = True
            return []

        async def _boom():
            raise RuntimeError("pool is gone")

        with patch.object(AREF, "should_reflect", lambda _m: True), \
             patch.object(AREF, "extract_agent_reflections", _probe):
            async with sm() as db:
                with patch.object(db, "commit", _boom):
                    await AREF.reflect_on_turn(db, user_id, "always call me Nariman", "ok")

        assert seen.get("called") is True, (
            "a failing connection release swallowed the reflection turn"
        )
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_the_gate_still_short_circuits(self):
        """An ungated commit on every turn would be a behaviour change of its
        own. The cheap regex gate must still return before touching the DB."""
        user_id = str(uuid.uuid4())
        engine, sm = await _agent_config_engine(user_id)
        called = {"n": 0}

        async def _never(*a, **kw):
            called["n"] += 1
            return []

        with patch.object(AREF, "should_reflect", lambda _m: False), \
             patch.object(AREF, "extract_agent_reflections", _never):
            async with sm() as db:
                out = await AREF.reflect_on_turn(db, user_id, "thanks!", "np")

        assert out == 0 and called["n"] == 0
        await engine.dispose()


# ──────────────────────────────────────────────────────────────
# _extract_memories + the fire-and-forget sites — against real source
# ──────────────────────────────────────────────────────────────

class TestAgentRunnerSites:
    @pytest.fixture(scope="class")
    def src(self):
        import app.agent.agent_runner as AR
        return Path(AR.__file__).read_text()

    def test_the_curator_commits_before_its_model_call(self):
        """The ordering IS the fix: a commit after the call releases nothing.

        v3 moved this site. `_extract_memories` is deleted; the writer is
        `memory_curator.curate_turn`, reached from the SAME fire-and-forget
        `_background_post_processing`, and it still does DB reads (identity,
        system files, the file bodies) before a multi-second model round
        trip. Same hazard, same fix, one file over.
        """
        import app.services.memory_curator as MC

        src = Path(MC.__file__).read_text()
        at = src.index("async def curate_turn(")
        body = src[at:]
        read = body.index("await ops.ensure_system_files(db, user_id)")
        llm = body.index("await _run_ops(")
        block = body[read:llm]
        assert "await db.commit()" in block, (
            "curate_turn holds the pooled connection across the model call"
        )

    def test_the_curator_still_does_the_reads_that_make_the_ordering_matter(self):
        """Deleting the reads would make the ordering assertion vacuous."""
        import app.services.memory_curator as MC

        src = Path(MC.__file__).read_text()
        assert "resolve_user_identity(db, user_id)" in src
        assert "ops._all_files(db, user_id)" in src

    def test_background_post_processing_keeps_a_strong_ref(self, src):
        assert "_spawn_background(_background_post_processing())" in src
        assert "asyncio.create_task(_background_post_processing())" not in src

    def test_summarizer_schedule_keeps_a_strong_ref(self, src):
        assert "_spawn_background(run_summarizer_if_needed(" in src
        assert "asyncio.create_task(run_summarizer_if_needed(" not in src


class TestSpawnBackground:
    @pytest.mark.asyncio
    async def test_holds_a_reference_while_running_and_drops_it_after(self):
        import app.agent.agent_runner as AR

        gate = asyncio.Event()

        async def _work():
            await gate.wait()

        before = len(AR._background_tasks)
        AR._spawn_background(_work())
        await asyncio.sleep(0)
        assert len(AR._background_tasks) == before + 1, "no strong ref retained"

        gate.set()
        await asyncio.sleep(0.05)
        assert len(AR._background_tasks) == before, "done task was never discarded"

    @pytest.mark.asyncio
    async def test_a_raising_task_is_also_discarded(self):
        """The done-callback fires on failure too — otherwise a crashing
        background turn would leak a Task object per request."""
        import app.agent.agent_runner as AR

        async def _boom():
            raise RuntimeError("nope")

        before = len(AR._background_tasks)
        AR._spawn_background(_boom())
        await asyncio.sleep(0.05)
        assert len(AR._background_tasks) == before
