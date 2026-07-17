"""agent_runner.run() — Phase 3 sub-agent parameter wiring.

Static-introspection + targeted-mock tests for the three new
parameters on ``AgentRunner.run()``:

  - ``prompt_profile``: PromptProfile.FULL (default) | SUBAGENT
  - ``save_assistant_message``: True (default) | False — gates the
    save_messages call at agent_runner.py:1182-1198
  - ``disable_post_processing``: False (default) | True — gates
    asyncio.create_task(_background_post_processing()) AND the
    day-chat summarizer

We DO NOT exercise the full agent loop here (no LLM mocks; that's
a Phase 4-level integration test once the spawn handler invokes
these end-to-end). What we DO pin:

  - The new parameters appear in the run() signature with the right
    defaults.
  - The _build_system_prompt method accepts the threading
    parameters and the agent_runner.py source wires them through.
  - Memory-write tool defaults are applied to self._disabled_tool_names
    when prompt_profile=SUBAGENT.
  - Day-chat summarizer call site is gated behind disable_post_processing.
  - save_assistant_message gate is in place around the _save_messages call.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parent.parent
_AGENT_RUNNER_PATH = BACKEND_DIR / "app" / "agent" / "agent_runner.py"
_AGENT_RUNNER_SRC = _AGENT_RUNNER_PATH.read_text()


# ──────────────────────────────────────────────────────────────────────
# Signature pinning
# ──────────────────────────────────────────────────────────────────────


def test_run_signature_has_new_params_with_correct_defaults():
    from app.agent.agent_runner import AgentRunner

    sig = inspect.signature(AgentRunner.run)
    params = sig.parameters

    # New required-named params with their defaults
    expected = {
        "save_assistant_message": True,
        "disable_post_processing": False,
        "prompt_profile": None,    # late-resolved to FULL inside run()
        "subagent_task_label": None,
        "credit_budget": None,
    }
    for name, default in expected.items():
        assert name in params, f"run() must accept {name}"
        assert params[name].default == default, (
            f"run().{name} default must be {default!r}, "
            f"got {params[name].default!r}"
        )


def test_run_default_prompt_profile_resolves_to_full():
    """When prompt_profile is omitted (None), run() must resolve to
    PromptProfile.FULL — preserving historical behaviour for every
    existing caller (telegram_bot, ws_chat, ws_realtime, routines,
    triggers, voice, app_builder, etc.)."""
    # Source-level check: the body contains the resolution to FULL.
    assert (
        "if prompt_profile is None:" in _AGENT_RUNNER_SRC
        and "prompt_profile = PromptProfile.FULL" in _AGENT_RUNNER_SRC
    ), "run() must default prompt_profile to FULL when None passed"


def test_build_system_prompt_accepts_profile_and_task_label():
    from app.agent.agent_runner import AgentRunner

    sig = inspect.signature(AgentRunner._build_system_prompt)
    assert "prompt_profile" in sig.parameters
    assert "subagent_task_label" in sig.parameters


# ──────────────────────────────────────────────────────────────────────
# Wiring pinning via source grep
#
# These are belt-and-suspenders for the surgery on a 3000+-line file —
# they read the source and assert the gating constructs are in place.
# Beat a fragile mock-everything integration test for changes this
# narrow.
# ──────────────────────────────────────────────────────────────────────


def test_save_assistant_message_gates_save_messages_call():
    """The save_messages call lives under an `if save_assistant_message:`
    block — sub-agent runs (save_assistant_message=False) skip it."""
    # Match the gate's prefix and the following call shape.
    assert re.search(
        r"if save_assistant_message:\s*\n[^\n]*async with async_session_maker",
        _AGENT_RUNNER_SRC,
    ), "save_assistant_message gate must wrap the _save_messages call"


def test_disable_post_processing_gates_background_task():
    """The _background_post_processing scheduling is gated."""
    assert (
        "if not disable_post_processing:" in _AGENT_RUNNER_SRC
        and "asyncio.create_task(_background_post_processing())" in _AGENT_RUNNER_SRC
    ), "_background_post_processing scheduling must be gated"


def test_disable_post_processing_gates_day_chat_summarizer():
    """The day-chat summarizer is also gated — sub-agent runs do not
    bump the user's day summary."""
    assert "if not disable_post_processing and _use_day_ctx and _day_chat_id:" in _AGENT_RUNNER_SRC, (
        "Day-chat summarizer must be gated behind disable_post_processing"
    )


def test_amendment_3_closure_captures_retrieved_memories():
    """Amendment 3: _last_retrieved_memories is captured into a
    closure-local copy before the background task is scheduled. The
    background coroutine reads the local copy, not self.* — so a
    concurrent sub-agent run's _build_system_prompt write to the
    singleton instance doesn't corrupt the parent's retrieval-feedback
    log."""
    assert "_retrieved_for_bg = list(self._last_retrieved_memories" in _AGENT_RUNNER_SRC, (
        "Phase 3 must capture self._last_retrieved_memories into a "
        "closure-local _retrieved_for_bg before scheduling "
        "_background_post_processing (Amendment 3 fix)"
    )
    # And the closure is what feeds log_retrieval_feedback — not self.*
    assert "retrieved_memories=_retrieved_for_bg" in _AGENT_RUNNER_SRC, (
        "log_retrieval_feedback must read the closure-local copy"
    )


def test_subagent_profile_extends_disabled_tools():
    """When prompt_profile=SUBAGENT, the disabled-tools set is merged
    with the SUBAGENT_DISABLED_TOOLS default."""
    assert (
        "from app.agent.prompt_profile import disabled_tools_for" in _AGENT_RUNNER_SRC
        and "_profile_disabled = disabled_tools_for(prompt_profile)" in _AGENT_RUNNER_SRC
    ), "run() must merge profile-default disabled tools into self._disabled_tool_names"


def test_build_system_prompt_uses_profile_for_section_order():
    """_build_system_prompt no longer ships a hardcoded SECTION_ORDER
    list; instead it pulls from prompt_profile.sections_for()."""
    # The historical hardcoded order has been replaced with a
    # dynamic call. Both forms exist for clarity, but the assignment
    # must come from sections_for, not from a raw list literal.
    assert "from app.agent.prompt_profile import sections_for" in _AGENT_RUNNER_SRC
    assert "SECTION_ORDER = list(sections_for(" in _AGENT_RUNNER_SRC, (
        "_build_system_prompt must derive SECTION_ORDER from the "
        "profile-keyed allow-list (Phase 3)"
    )


def test_subagent_task_preamble_section_built_when_profile_is_subagent():
    """Source-level check: when profile is SUBAGENT, the builder
    sets section_parts['subagent_task_preamble']."""
    assert 'section_parts["subagent_task_preamble"]' in _AGENT_RUNNER_SRC, (
        "_build_system_prompt must emit a subagent_task_preamble "
        "section when prompt_profile is SUBAGENT"
    )


def test_post_builder_blocks_gated_by_allows_post_builder_blocks():
    """The today_so_far + reply_to_directive + recent_days appends
    use the same _allow_post_builder gate."""
    assert "_allow_post_builder = allows_post_builder_blocks(prompt_profile)" in _AGENT_RUNNER_SRC
    assert "_allow_post_builder and _use_day_ctx" in _AGENT_RUNNER_SRC
    assert "_allow_post_builder and _stripped_um.startswith" in _AGENT_RUNNER_SRC


def test_run_log_line_emits_profile_and_budget():
    """The early structured log line stamps profile and
    credit_budget — so an operator grepping for sub-agent runs sees
    them at a glance."""
    # The new opening log line includes "profile=%s" and "credit_budget=%s"
    assert "profile=%s" in _AGENT_RUNNER_SRC
    assert "credit_budget=%s" in _AGENT_RUNNER_SRC


# ──────────────────────────────────────────────────────────────────────
# Backward-compat: every existing caller is unaffected
# ──────────────────────────────────────────────────────────────────────


def test_existing_call_sites_do_not_pass_new_params():
    """Existing callers (telegram_bot, ws_chat, etc.) must NOT have
    been updated to pass the new sub-agent parameters — that's Phase
    4's job. This pins the "Phase 3 ships invisible to existing
    users" invariant."""
    backend_root = BACKEND_DIR / "app"
    for path in backend_root.rglob("*.py"):
        if "subagent" in path.name:
            continue  # Phase 4 will wire these
        if path.name == "agent_runner.py":
            continue  # the definition site
        if path.name == "prompt_profile.py":
            continue  # the definition site
        if path.name == "autopilot_handler.py":
            # Headless ticks (PR #282, 2026-07-16) intentionally pass
            # save_assistant_message=False + disable_post_processing=True
            # so raw AUTOPILOT_* marker replies stop landing in chat.
            continue
        try:
            src = path.read_text()
        except Exception:
            continue
        for forbidden in (
            "prompt_profile=PromptProfile.SUBAGENT",
            "disable_post_processing=True",
            "save_assistant_message=False",
        ):
            assert forbidden not in src, (
                f"Phase 3 should NOT wire {forbidden!r} from {path} — "
                "that's Phase 4's job. If you intentionally landed "
                "the spawn handler ahead of schedule, remove this "
                "test."
            )


# ──────────────────────────────────────────────────────────────────────
# Subagent prompt builder smoke — verify the assembled prompt for a
# SUBAGENT profile drops the persona/memory sections
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_system_prompt_subagent_drops_persona_and_memory():
    """End-to-end on _build_system_prompt: feed it a real DB session,
    profile=SUBAGENT, and confirm the assembled string is missing
    the persona/memory section headers.

    The function is large (~1150 LOC) and has many side effects, but
    its OUTPUT is one string. We assert against the string.

    NOTE: this skips on missing OPENAI_API_KEY because
    classify_query_intent uses lightweight regex (no LLM), but the
    hybrid_search inside the user_brain section can attempt an
    embedding call if the user has memories. With a fresh test user
    it returns nothing fast.
    """
    import uuid as _uuid
    from app.agent.agent_runner import AgentRunner
    from app.agent.prompt_profile import PromptProfile
    from app.db import async_session_maker, User
    from app.services.openai_agent_service import OpenAIAgentService
    from app.agent.tool_executor import ToolExecutor

    user_id = str(_uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"{user_id[:8]}@test.local",
            hashed_password="x" * 60,
            name="Sub-test",
        ))
        await db.commit()

    import tempfile
    with tempfile.TemporaryDirectory() as workdir:
        runner = AgentRunner(
            llm_service=OpenAIAgentService(),
            tool_executor=ToolExecutor(workspace=workdir),
        )
        async with async_session_maker() as db:
            prompt = await runner._build_system_prompt(
                db=db,
                user_id=user_id,
                user_message="research X for me",
                channel="web",
                client_tz="UTC",
                prompt_profile=PromptProfile.SUBAGENT,
                subagent_task_label="research X for me",
            )

    # SUBAGENT prompt must contain the task preamble
    assert "# Sub-agent task" in prompt
    assert "research X for me" in prompt

    # And must NOT contain the persona/memory section headers
    forbidden_headers = [
        "# Core Identity",
        "# Identity\nYour name is",
        "# Identity\nYou don't have a name yet",
        "# Voice — Always Apply",
        "# Platform Knowledge",
        "# About You (the User)",
        "# User Brain",
        "# Your Memory",
        "# Onboarding Mode",
        "# Media Playback",
    ]
    for header in forbidden_headers:
        assert header not in prompt, (
            f"SUBAGENT prompt must not contain header {header!r} — "
            f"found in: {prompt[:200]}..."
        )


@pytest.mark.asyncio
async def test_build_system_prompt_full_still_contains_persona():
    """Regression: the FULL profile path is unchanged. Persona,
    voice rules, and platform knowledge all still land."""
    import uuid as _uuid
    from app.agent.agent_runner import AgentRunner
    from app.agent.prompt_profile import PromptProfile
    from app.db import async_session_maker, User
    from app.services.openai_agent_service import OpenAIAgentService
    from app.agent.tool_executor import ToolExecutor

    user_id = str(_uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"{user_id[:8]}@test.local",
            hashed_password="x" * 60,
            name="Full-test",
        ))
        await db.commit()

    import tempfile
    with tempfile.TemporaryDirectory() as workdir:
        runner = AgentRunner(
            llm_service=OpenAIAgentService(),
            tool_executor=ToolExecutor(workspace=workdir),
        )
        async with async_session_maker() as db:
            prompt = await runner._build_system_prompt(
                db=db,
                user_id=user_id,
                user_message="hello",
                channel="web",
                client_tz="UTC",
                prompt_profile=PromptProfile.FULL,
            )

    # FULL must contain the persona stack
    assert "# Core Identity" in prompt
    assert "# Voice — Always Apply" in prompt
    assert "# Platform Knowledge" in prompt
    # And NOT the sub-agent preamble
    assert "# Sub-agent task" not in prompt
