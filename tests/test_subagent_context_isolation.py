"""W1.2 — Sub-agent context isolation (2026-07 SOTA assessment).

Headline leak fixed here: a SUBAGENT child run inherited the parent's
~30k-token day history and resent it on every iteration (~300k
tokens/spawn). Four fixes pinned:

  (a) HISTORY GATE — the day-context load in run() is gated on an
      EXPLICIT ``prompt_profile != PromptProfile.SUBAGENT`` check
      (not allows_post_builder_blocks, which is also False for
      AUTOPILOT — autopilot deliberately KEEPS day context).
  (b) CACHE KEY — SUBAGENT runs scope prompt_cache_key on their own
      'subagent:{job_id}' session sentinel, never the parent's day
      shard.
  (c) SKIP DISCARDED WORK — _build_system_prompt short-circuits the
      memory-retrieval fan-out (hybrid_search + entity_search +
      portrait) and the active_tasks load when the profile's section
      list excludes 'user_brain' / 'active_tasks' — the same
      predicate the pop/filter sites use.
  (d) NO ORPHAN ROW — SUBAGENT runs create no Conversation row; the
      orchestrator's 'subagent:{job_id}' id is kept as an in-memory
      sentinel (announce-back persists via write_subagent_message,
      which resolves its own conversation — untouched).

Mix of source-grep pins (convention: test_agent_runner_subagent_params
/ test_stable_prefix) and DB-backed behavior tests with a fake LLM.
"""
from __future__ import annotations

import re
import uuid as _uuid
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parent.parent
_SRC = (BACKEND_DIR / "app" / "agent" / "agent_runner.py").read_text()


# ──────────────────────────────────────────────────────────────────────
# (a) History gate — source pins
# ──────────────────────────────────────────────────────────────────────


def test_day_context_load_is_gated_on_explicit_subagent_check():
    """The day-context feature-flag check itself sits under the profile
    gate — a SUBAGENT run never even asks should_use_day_chat_context."""
    assert re.search(
        r"if prompt_profile != PromptProfile\.SUBAGENT:\s*\n\s*try:\s*\n"
        r"\s*from app\.agent\.day_chat_resolver import should_use_day_chat_context",
        _SRC,
    ), "day-context load must be gated on prompt_profile != SUBAGENT"


def test_day_context_gate_is_not_allows_post_builder_blocks():
    """AUTOPILOT also has allows_post_builder_blocks == False but
    deliberately keeps day context — the gate must be the explicit
    enum comparison, never the post-builder predicate."""
    gate_idx = _SRC.index("if prompt_profile != PromptProfile.SUBAGENT:")
    flag_idx = _SRC.index("_use_day_ctx = await should_use_day_chat_context()")
    assert gate_idx < flag_idx, "gate must precede the flag check"
    # No allows_post_builder_blocks CALL between the _use_day_ctx init
    # and the load (a comment may name it — only a call is a violation)
    region = _SRC[_SRC.index("_use_day_ctx = False"):flag_idx]
    assert "allows_post_builder_blocks(" not in region


def test_autopilot_still_allows_day_context():
    """Regression guard for the gate's polarity: AUTOPILOT (and FULL)
    must pass the gate — only SUBAGENT is excluded."""
    from app.agent.prompt_profile import PromptProfile

    for profile in (PromptProfile.FULL, PromptProfile.AUTOPILOT):
        assert profile != PromptProfile.SUBAGENT  # the gate lets these through


# ──────────────────────────────────────────────────────────────────────
# (b) Cache key — source pins
# ──────────────────────────────────────────────────────────────────────


def test_cache_scope_is_session_sentinel_for_subagent():
    assert re.search(
        r"if prompt_profile == PromptProfile\.SUBAGENT:\s*\n"
        r"\s*_cache_scope = session_id\s*\n"
        r"\s*else:\s*\n"
        r"\s*_cache_scope = _day_chat_id or session_id",
        _SRC,
    ), "SUBAGENT cache scope must be the child's own session sentinel"
    # Key shape unchanged: user_id-prefixed
    assert '_cache_key = f"{user_id}:{_cache_scope}"' in _SRC


# ──────────────────────────────────────────────────────────────────────
# (c) Skip discarded work — source pins
# ──────────────────────────────────────────────────────────────────────


def test_memory_retrieval_short_circuits_on_profile():
    """Same predicate as the pop site: 'user_brain' not in the
    profile's section list → the hybrid_search fan-out never runs."""
    assert '"user_brain" not in _profile_sections' in _SRC
    assert "reason=profile_no_user_brain" in _SRC
    # The short-circuit precedes the retrieval body
    assert _SRC.index('"user_brain" not in _profile_sections') < _SRC.index(
        "memories = await mem_svc.hybrid_search("
    )


def test_active_tasks_short_circuits_on_profile():
    assert '"active_tasks" not in _profile_sections' in _SRC
    assert _SRC.index('"active_tasks" not in _profile_sections') < _SRC.index(
        "active_tasks = await get_active_tasks(db, user_id)"
    )


# ──────────────────────────────────────────────────────────────────────
# (d) No orphan Conversation row — source pins
# ──────────────────────────────────────────────────────────────────────


def test_subagent_run_skips_get_or_create_session():
    """SUBAGENT keeps the caller's 'subagent:{job_id}' id as an
    in-memory sentinel instead of inserting a Conversation row."""
    assert re.search(
        r"if prompt_profile == PromptProfile\.SUBAGENT:\s*\n(?:\s*#[^\n]*\n)*"
        r'\s*session_id = session_id or f"subagent:\{uuid\.uuid4\(\)\.hex\[:20\]\}"',
        _SRC,
    ), "SUBAGENT branch must keep/synthesize a sentinel session_id"
    # The ContextVar stamp survives on both branches (a child tool call
    # must not inherit the parent's conversation id).
    assert "self.tools.set_session_id(session_id)" in _SRC


# ──────────────────────────────────────────────────────────────────────
# Spawn tool description — ambient context must travel in the task text
# ──────────────────────────────────────────────────────────────────────


def test_spawn_description_says_child_starts_blank():
    from app.agent.tool_definitions import get_agent_tools, get_extended_tools

    spawn = next(
        t for t in get_agent_tools() + get_extended_tools()
        if t.get("name") == "spawn"
    )
    desc = spawn["description"]
    assert "NO conversation history" in desc
    assert "task text" in desc


# ──────────────────────────────────────────────────────────────────────
# Behavior: _build_system_prompt skips retrieval + active_tasks for
# SUBAGENT, keeps them for FULL
# ──────────────────────────────────────────────────────────────────────


async def _make_user() -> str:
    from app.db import async_session_maker, User

    user_id = str(_uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"{user_id[:8]}@test.local",
            hashed_password="x" * 60,
            name="Ctx-iso",
        ))
        await db.commit()
    return user_id


def _make_runner(workdir: str):
    from app.agent.agent_runner import AgentRunner
    from app.agent.tool_executor import ToolExecutor
    from app.services.openai_agent_service import OpenAIAgentService

    return AgentRunner(
        llm_service=OpenAIAgentService(),
        tool_executor=ToolExecutor(workspace=workdir),
    )


# Non-trivial message — must NOT hit the trivial-query skip, so the
# only thing standing between the profile and the retrieval fan-out is
# the W1.2(c) short-circuit.
_NON_TRIVIAL_MSG = "research competitor pricing pages for acme corp and summarize"


@pytest.mark.asyncio
async def test_subagent_profile_never_calls_retrieval_or_active_tasks(monkeypatch, tmp_path):
    from app.agent.prompt_profile import PromptProfile
    from app.db import async_session_maker
    import app.services.active_task_service as at_svc
    from app.services.memory_service import MemoryService

    calls = {"hybrid": 0, "active": 0}

    async def _spy_hybrid(self, **kwargs):
        calls["hybrid"] += 1
        return []

    async def _spy_active(db, user_id):
        calls["active"] += 1
        return []

    monkeypatch.setattr(MemoryService, "hybrid_search", _spy_hybrid)
    monkeypatch.setattr(at_svc, "get_active_tasks", _spy_active)

    user_id = await _make_user()
    runner = _make_runner(str(tmp_path))
    async with async_session_maker() as db:
        prompt = await runner._build_system_prompt(
            db=db,
            user_id=user_id,
            user_message=_NON_TRIVIAL_MSG,
            channel="subagent",
            client_tz="UTC",
            prompt_profile=PromptProfile.SUBAGENT,
            subagent_task_label="research task",
        )

    assert calls["hybrid"] == 0, "SUBAGENT must not run hybrid_search — output is discarded"
    assert calls["active"] == 0, "SUBAGENT must not load active_tasks — output is discarded"
    assert runner._memory_health["retrieved"] == 0
    assert runner._memory_health["active_tasks"] == 0
    assert "# User Brain" not in prompt


@pytest.mark.asyncio
async def test_full_profile_still_calls_retrieval_and_active_tasks(monkeypatch, tmp_path):
    """Control: the short-circuit must not leak into FULL."""
    from app.agent.prompt_profile import PromptProfile
    from app.db import async_session_maker
    import app.services.active_task_service as at_svc
    from app.services.memory_service import MemoryService
    from app.services.user_portrait_service import UserPortraitService

    calls = {"hybrid": 0, "active": 0}

    async def _spy_hybrid(self, **kwargs):
        calls["hybrid"] += 1
        return []

    async def _spy_active(db, user_id):
        calls["active"] += 1
        return []

    async def _no_portrait(self, user_id):
        return None

    monkeypatch.setattr(MemoryService, "hybrid_search", _spy_hybrid)
    monkeypatch.setattr(at_svc, "get_active_tasks", _spy_active)
    monkeypatch.setattr(UserPortraitService, "get_or_build_portrait", _no_portrait)

    user_id = await _make_user()
    runner = _make_runner(str(tmp_path))
    async with async_session_maker() as db:
        await runner._build_system_prompt(
            db=db,
            user_id=user_id,
            user_message=_NON_TRIVIAL_MSG,
            channel="web",
            client_tz="UTC",
            prompt_profile=PromptProfile.FULL,
        )

    assert calls["hybrid"] == 1, "FULL must still run memory retrieval"
    assert calls["active"] == 1, "FULL must still load active_tasks"


# ──────────────────────────────────────────────────────────────────────
# Behavior: end-to-end SUBAGENT run() with a fake LLM — empty history,
# sentinel cache key, no Conversation row, no day-context flag check
# ──────────────────────────────────────────────────────────────────────


class _FakeLLM:
    """Duck-typed stand-in for OpenAIAgentService: one text turn,
    records every kwarg the runner passes."""

    def __init__(self):
        self.calls = []

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent

        # Snapshot: the runner mutates the messages list in place after
        # the stream (appends the assistant turn for the next iteration).
        self.calls.append({**kwargs, "messages": list(kwargs.get("messages") or [])})
        yield StreamEvent(type="text", text="done")
        yield StreamEvent(
            type="message_end",
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 2},
        )


@pytest.mark.asyncio
async def test_subagent_run_isolated_end_to_end(monkeypatch, tmp_path):
    import app.agent.day_chat_resolver as dc_resolver
    from app.agent.agent_runner import AgentRunner
    from app.agent.prompt_profile import PromptProfile
    from app.agent.tool_executor import ToolExecutor
    from app.db import async_session_maker
    from app.db.models import Conversation
    from sqlalchemy import func, select

    flag_checks = {"n": 0}

    async def _spy_flag(*a, **kw):
        flag_checks["n"] += 1
        return False

    monkeypatch.setattr(dc_resolver, "should_use_day_chat_context", _spy_flag)

    # conversations/messages are AGENT_ONLY — the conftest platform-mode
    # init_db doesn't create them, but run() reads history from messages
    # and this test asserts the conversations row count.
    from app.db import database as db_database
    from app.db.models import Message
    from app.db.models.base import Base

    async with db_database.engine.begin() as conn:
        await conn.run_sync(lambda sc: Base.metadata.create_all(
            sc,
            tables=[Conversation.__table__, Message.__table__],
            checkfirst=True,
        ))

    user_id = await _make_user()
    job_id = str(_uuid.uuid4())
    fake = _FakeLLM()
    runner = AgentRunner(
        llm_service=fake,  # type: ignore[arg-type]
        tool_executor=ToolExecutor(workspace=str(tmp_path)),
    )

    response = await runner.run(
        user_message="[Sub-agent task]\ncompare pricing of X and Y",
        user_id=user_id,
        session_id=f"subagent:{job_id}",
        channel="subagent",
        prompt_profile=PromptProfile.SUBAGENT,
        subagent_task_label="compare pricing",
        save_user_message=False,
        save_assistant_message=False,
        disable_post_processing=True,
        model_override="gpt-5.5-mini",
    )

    assert response.text == "done"

    # (a) No day-context: the flag itself was never consulted for SUBAGENT
    assert flag_checks["n"] == 0, "SUBAGENT must not even check the day-chat flag"

    # (a) Empty history: exactly the current user message went to the LLM
    assert len(fake.calls) == 1
    sent = fake.calls[0]["messages"]
    assert len(sent) == 1, f"child must start with empty history, got {len(sent)} messages"
    assert sent[0]["role"] == "user"
    assert "compare pricing of X and Y" in str(sent[0]["content"])

    # (b) Cache key scoped to the child's own sentinel — user:subagent:job
    assert fake.calls[0]["prompt_cache_key"] == f"{user_id}:subagent:{job_id}"

    # (d) No orphan Conversation row, and the sentinel survived the run
    assert response.session_id == f"subagent:{job_id}"
    async with async_session_maker() as db:
        n_convs = (await db.execute(
            select(func.count()).select_from(Conversation).where(
                Conversation.user_id == user_id
            )
        )).scalar_one()
    assert n_convs == 0, "SUBAGENT run must not create a Conversation row"
