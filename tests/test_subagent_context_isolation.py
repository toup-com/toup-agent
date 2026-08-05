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

2026-08-04 — this file used to be half source-text greps over
agent_runner.py, and that half was worthless twice over.

It was worthless as coverage: ``assert re.search(r"...", _SRC)`` passes
against a function body of ``return``. It never executes a line of the
code it claims to protect.

And it was worthless in practice: #367 (CHANNEL_CONVERGE W2.3a) inserted
an ``elif _channel_converge:`` branch between the SUBAGENT branch and the
day/session ``else``. Every guarded BEHAVIOUR was unchanged — SUBAGENT is
still first, so it still wins — but the regex demanded the two branches be
textually adjacent, so it went red and got parked in COVERAGE_DEBT.txt.
A test that fails on a refactor it should not care about, and passes on a
deletion it should catch, has its polarity backwards on both ends.

Every pin here is now behavioural: drive the real ``run()`` with a fake
LLM and assert on the kwargs it received, the spy counters, and the rows
in the database.
"""
from __future__ import annotations

import uuid as _uuid

import pytest


# ──────────────────────────────────────────────────────────────────────
# Spawn tool description — ambient context must travel in the task text
# ──────────────────────────────────────────────────────────────────────


def test_spawn_description_says_child_starts_blank():
    """Not a source grep: this reads the live tool definition that is
    actually shipped to the model, so it goes red if the description is
    changed — which is the thing that would mislead the model."""
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
# Behavior: end-to-end run() with a fake LLM
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


def _quiet_prompt_build(monkeypatch):
    """Silence the retrieval fan-out for the non-SUBAGENT profiles.

    FULL and AUTOPILOT really do call hybrid_search, and the sqlite test
    database has no pgvector — so without this the run dies on the
    embedding column rather than on the thing under test.
    """
    import app.services.active_task_service as at_svc
    from app.services.memory_service import MemoryService
    from app.services.user_portrait_service import UserPortraitService

    async def _no_hybrid(self, **kwargs):
        return []

    async def _no_active(db, user_id):
        return []

    async def _no_portrait(self, user_id):
        return None

    monkeypatch.setattr(MemoryService, "hybrid_search", _no_hybrid)
    monkeypatch.setattr(at_svc, "get_active_tasks", _no_active)
    monkeypatch.setattr(UserPortraitService, "get_or_build_portrait", _no_portrait)


async def _run_with_fake(monkeypatch, tmp_path, *, profile, session_id, user_id):
    """Drive one real run() against a fake LLM and hand back the recorder."""
    from app.agent.agent_runner import AgentRunner
    from app.agent.tool_executor import ToolExecutor

    fake = _FakeLLM()
    runner = AgentRunner(
        llm_service=fake,  # type: ignore[arg-type]
        tool_executor=ToolExecutor(workspace=str(tmp_path)),
    )
    response = await runner.run(
        user_message="compare pricing of X and Y",
        user_id=user_id,
        session_id=session_id,
        channel="subagent" if session_id.startswith("subagent:") else "web",
        prompt_profile=profile,
        save_user_message=False,
        save_assistant_message=False,
        disable_post_processing=True,
        model_override="gpt-5.5-mini",
    )
    return fake, response


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

    # (a) Empty history: no PARENT conversation reached the child.
    #
    # This asserted `len(sent) == 1` until 2026-08-05, which used message
    # COUNT as a proxy for isolation. That proxy only held while
    # stable_prefix_layout defaulted OFF: with the layout on — which is what
    # 59 of 61 fleet containers were already running — a <turn_context>
    # message is appended ahead of the task, so the count is 2 and the test
    # failed for a reason that has nothing to do with isolation. It was
    # therefore a test that only passed in a configuration production does
    # not use.
    #
    # Isolation is about CONTENT, so assert content: the child sees its own
    # task and nothing of the parent's history, memories or day. The
    # turn_context a SUBAGENT gets is clock-only (verified below), which is
    # exactly what it should be.
    assert len(fake.calls) == 1
    sent = fake.calls[0]["messages"]
    assert all(m["role"] == "user" for m in sent), (
        f"child was handed non-user turns: {[m['role'] for m in sent]}"
    )
    task_msgs = [m for m in sent if "compare pricing of X and Y" in str(m["content"])]
    assert len(task_msgs) == 1, "the child's own task must be present exactly once"

    others = [m for m in sent if m not in task_msgs]
    assert len(others) <= 1, (
        f"child received {len(others)} messages beyond its task — expected at "
        f"most one <turn_context>: {[str(m['content'])[:80] for m in others]}"
    )
    for m in others:
        raw = str(m["content"])
        assert raw.lstrip().startswith("<turn_context>"), (
            f"unexpected message prepended to a SUBAGENT run: {raw[:200]}"
        )
        # Inspect the BODY only. The block's fixed preamble names every kind
        # of state it CAN carry ("recalled memories, open threads, day
        # summary"), so scanning the whole string matches those words in the
        # boilerplate and reports a leak on an empty block.
        body = raw.split(")\n\n", 1)[1].split("</turn_context>")[0]
        lines = [ln for ln in body.splitlines() if ln.strip()]
        assert lines and all(ln.startswith("Current time:") for ln in lines), (
            "a SUBAGENT's turn_context must be clock-only — anything else is "
            f"parent state reaching an isolated child: {lines}"
        )

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


@pytest.mark.asyncio
async def test_autopilot_still_checks_the_day_context_flag(monkeypatch, tmp_path):
    """(a) POLARITY — the gate must be the enum comparison, not the
    post-builder predicate.

    ``allows_post_builder_blocks`` is False for AUTOPILOT *as well as*
    SUBAGENT, so gating the day-context load on it would silently strip
    autopilot's day history too. Only SUBAGENT is excluded.

    This replaces two source greps (one asserting the textual order of the
    gate and the flag check, one asserting the string
    "allows_post_builder_blocks(" was absent from a slice of the file) and
    a third that asserted ``PromptProfile.FULL != PromptProfile.SUBAGENT``
    — a tautology about the enum that executed none of our code.

    MUTATION: change the gate in run() from
    ``if prompt_profile != PromptProfile.SUBAGENT:`` to
    ``if allows_post_builder_blocks(prompt_profile):`` — AUTOPILOT stops
    consulting the flag and this goes red, while the SUBAGENT assertion in
    the test above stays green. That asymmetry is the whole point.
    """
    import app.agent.day_chat_resolver as dc_resolver
    from app.agent.prompt_profile import PromptProfile

    _quiet_prompt_build(monkeypatch)

    flag_checks = {"n": 0}

    async def _spy_flag(*a, **kw):
        flag_checks["n"] += 1
        return False

    monkeypatch.setattr(dc_resolver, "should_use_day_chat_context", _spy_flag)

    user_id = await _make_user()
    await _run_with_fake(
        monkeypatch, tmp_path,
        profile=PromptProfile.AUTOPILOT,
        session_id=str(_uuid.uuid4()),
        user_id=user_id,
    )

    assert flag_checks["n"] >= 1, (
        "AUTOPILOT must still consult the day-chat flag — it deliberately "
        "KEEPS day context. If this is zero the gate has been widened from "
        "the SUBAGENT enum check to allows_post_builder_blocks."
    )


# ──────────────────────────────────────────────────────────────────────
# (b) Cache scope — behavioural, across the CHANNEL_CONVERGE branch
# ──────────────────────────────────────────────────────────────────────
#
# The scope chain in run() is three-way:
#
#     if prompt_profile == PromptProfile.SUBAGENT:  _cache_scope = session_id
#     elif _channel_converge:                       _cache_scope = "all"
#     else:                                         _cache_scope = _day_chat_id or session_id
#
# and ``_channel_converge`` is ``stable_prefix_enabled(user_id) and
# settings.channel_converge``, both off by default. The end-to-end test
# above only ever exercises the flag-OFF path, so before these three the
# ORDER of the first two branches was pinned by nothing but a regex — and
# reordering them is precisely the mistake that would route a child's
# prefix into the parent's shared shard, which is the leak W1.2(b) exists
# to prevent.


def _enable_channel_converge(monkeypatch, *, on: bool):
    from app.config import settings

    monkeypatch.setattr(settings, "stable_prefix_layout", True, raising=False)
    monkeypatch.setattr(settings, "channel_converge", on, raising=False)


@pytest.mark.asyncio
async def test_subagent_scope_wins_over_channel_converge(monkeypatch, tmp_path):
    """SUBAGENT must keep its own sentinel even when CHANNEL_CONVERGE is on.

    MUTATION: swap the first two branches so ``elif _channel_converge`` is
    tested first — this goes red (key becomes ``{user}:all``) while
    test_subagent_run_isolated_end_to_end stays green, because that one
    runs with the flag off.
    """
    from app.agent.prompt_profile import PromptProfile

    _enable_channel_converge(monkeypatch, on=True)

    user_id = await _make_user()
    job_id = str(_uuid.uuid4())
    fake, _ = await _run_with_fake(
        monkeypatch, tmp_path,
        profile=PromptProfile.SUBAGENT,
        session_id=f"subagent:{job_id}",
        user_id=user_id,
    )

    assert fake.calls[0]["prompt_cache_key"] == f"{user_id}:subagent:{job_id}"


@pytest.mark.asyncio
async def test_full_scope_is_the_stable_shard_when_channel_converge_is_on(monkeypatch, tmp_path):
    """Control for the branch above: FULL *does* take the "all" shard.

    MUTATION: delete the ``elif _channel_converge:`` branch — the key falls
    back to ``{user}:{session}`` and this goes red.
    """
    from app.agent.prompt_profile import PromptProfile

    _quiet_prompt_build(monkeypatch)
    _enable_channel_converge(monkeypatch, on=True)

    user_id = await _make_user()
    fake, _ = await _run_with_fake(
        monkeypatch, tmp_path,
        profile=PromptProfile.FULL,
        session_id=str(_uuid.uuid4()),
        user_id=user_id,
    )

    assert fake.calls[0]["prompt_cache_key"] == f"{user_id}:all"


@pytest.mark.asyncio
async def test_full_scope_is_the_session_when_channel_converge_is_off(monkeypatch, tmp_path):
    """Default path: no day chat resolved, so the scope is the session.

    MUTATION: replace the ``else`` scope with a constant — this goes red.
    """
    from app.agent.prompt_profile import PromptProfile

    _quiet_prompt_build(monkeypatch)
    _enable_channel_converge(monkeypatch, on=False)

    user_id = await _make_user()
    fake, response = await _run_with_fake(
        monkeypatch, tmp_path,
        profile=PromptProfile.FULL,
        session_id=str(_uuid.uuid4()),
        user_id=user_id,
    )

    assert fake.calls[0]["prompt_cache_key"] == f"{user_id}:{response.session_id}"
