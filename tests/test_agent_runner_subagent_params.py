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
    """The _background_post_processing scheduling is gated.

    The SPAWN FORM changed after this test was written and the test did not:
    it still asserted `asyncio.create_task(...)`, while the runner uses
    `_spawn_background(...)`. That is not cosmetic — a bare `create_task`
    keeps no strong reference, so the GC can collect the task mid-flight and
    the whole post-processing block silently stops running. The strong-ref
    rule is pinned in `tests/test_background_connection_leak.py`, and this
    test now asserts the form that satisfies BOTH rules rather than a form
    that would violate one of them.
    """
    assert "if not disable_post_processing:" in _AGENT_RUNNER_SRC, (
        "_background_post_processing scheduling must be gated"
    )
    assert "_spawn_background(_background_post_processing())" in _AGENT_RUNNER_SRC, (
        "the background block must be spawned with a strong reference held"
    )
    assert "asyncio.create_task(_background_post_processing())" not in _AGENT_RUNNER_SRC


def test_disable_post_processing_gates_day_chat_summarizer():
    """The day-chat summarizer is also gated — sub-agent runs do not
    bump the user's day summary."""
    assert "if not disable_post_processing and _use_day_ctx and _day_chat_id:" in _AGENT_RUNNER_SRC, (
        "Day-chat summarizer must be gated behind disable_post_processing"
    )


def test_amendment_3_has_nothing_left_to_capture():
    """Amendment 3 captured `_last_retrieved_memories` into a closure-local
    copy before scheduling the background task, so a concurrent sub-agent
    run's `_build_system_prompt` could not corrupt the parent's
    retrieval-feedback row.

    Memory v3 (§3.1) retires sentence retrieval, and with it the attribute,
    the closure, `log_retrieval_feedback`'s per-turn caller and the weekly
    job that read `retrieval_events`. The singleton race it defended
    against cannot occur when there is nothing to race over — so what is
    pinned now is the ABSENCE, in both halves. A future writer that
    reintroduces per-turn retrieval state must reintroduce the capture with
    it, and this test is where they find that out."""
    # CODE only — the comment that records the retirement legitimately
    # names what it retired.
    code = "\n".join(
        line for line in _AGENT_RUNNER_SRC.splitlines()
        if not line.strip().startswith("#")
    )
    for gone in (
        "_last_retrieved_memories",
        "_retrieved_for_bg",
        "log_retrieval_feedback",
    ):
        assert gone not in code, (
            f"{gone} is back in agent_runner without the Amendment-3 "
            "closure capture — see the docstring above"
        )
    # The trivial-turn classification still rides the same pattern, and it
    # is the reason the pattern must stay documented.
    assert "_trivial_for_bg = _query_was_trivial" in code


def test_subagent_profile_extends_disabled_tools():
    """When prompt_profile=SUBAGENT, the disabled-tools set is merged
    with the SUBAGENT_DISABLED_TOOLS default.

    Asserted on the AST, not on the import's TEXT. The old form matched the
    literal `from app.agent.prompt_profile import disabled_tools_for`, and
    broke the day a second name was added and black wrapped it in
    parentheses — a reflow, with the behaviour completely unchanged. A guard
    that a formatter can turn red is a guard people learn to ignore.
    """
    import ast as _ast

    tree = _ast.parse(_AGENT_RUNNER_SRC)
    imported = {
        alias.name
        for node in _ast.walk(tree)
        if isinstance(node, _ast.ImportFrom)
        and node.module == "app.agent.prompt_profile"
        for alias in node.names
    }
    assert "disabled_tools_for" in imported, (
        "run() no longer imports disabled_tools_for from prompt_profile"
    )
    assert "_profile_disabled = disabled_tools_for(prompt_profile)" in _AGENT_RUNNER_SRC, (
        "run() must merge profile-default disabled tools into self._disabled_tool_names"
    )


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


#: Who is allowed to pass the three sub-agent parameters, and why.
#:
#: This replaces `test_existing_call_sites_do_not_pass_new_params`, which
#: asserted that NOBODY outside agent_runner passed them — the "Phase 3 ships
#: invisible to existing users" invariant. Phase 4 landed long ago and wired
#: eight call sites, so the old test was asserting the absence of shipped
#: behaviour; it stayed green only because the file was quarantined in
#: COVERAGE_DEBT.txt as "fails for a reason not yet triaged".
#:
#: The invariant that is actually LIVE, and that memory v3 depends on, is the
#: opposite one: a turn whose `user_message` is MACHINE-AUTHORED must pass
#: `disable_post_processing=True`, or the memory curator mines the platform's
#: own prompt as something the user said. That is where rows like "Gmail
#: briefing fetches messages from Gmail" and "5:06 PM motivational quote
#: routine is scheduled in America/Toronto" came from.
_SUBAGENT_PARAM_CALLERS = {
    "app/agent/cron_service.py":
        "'[Scheduled task: <name>] <prompt>' — the job's own text.",
    "app/agent/heartbeat_service.py":
        "'[Heartbeat] <settings.heartbeat_prompt>' — the platform talking to itself.",
    "app/agent/subagent.py":
        "'[Background Task: <label>] TASK: …' — machine-authored, and a "
        "sub-agent turn is not a user-facing event.",
    "app/agent/subagent_orchestrator.py":
        "The live sub-agent spawn path: SUBAGENT profile, no save, no mining.",
    "app/agent/routines/agent_task_handler.py":
        "A routine's prompt_text. The original RCA for this whole class.",
    "app/agent/routines/autopilot_handler.py":
        "Headless ticks (PR #282): raw AUTOPILOT_* marker replies must not "
        "land in chat and must not be mined.",
    "app/agent/triggers/email_received_handler.py":
        "A trigger fire carries the EMAIL's text, not the user's.",
    "app/api/ws_realtime.py":
        "`think`'s task string is the REALTIME MODEL's synthesis, not what "
        "the user said — mining it is the 409A incident. Voice memory is "
        "written from the real transcript via /internal/curate-turn.",
    "app/api/api_v1.py":
        "`disable_post_processing=not req.save` — the internal agent-turn "
        "routes map the caller's `save` flag onto all three.",
    "app/agent/agent_runner.py":
        "The definition site; it forwards `prompt_profile` to itself.",
}


def _subagent_param_call_sites() -> dict:
    """{relative path: {param}} for every call passing one of the three.

    AST, so the retirement comments that now name these parameters all over
    the tree cannot make a file look like a caller. That mistake is exactly
    how `app/api/ingest.py` hid as a memory producer through a whole
    convergence pass.
    """
    import ast as _ast

    params = {"disable_post_processing", "save_assistant_message", "prompt_profile"}
    found: dict = {}
    for path in (BACKEND_DIR / "app").rglob("*.py"):
        try:
            tree = _ast.parse(path.read_text())
        except (SyntaxError, OSError):  # pragma: no cover
            continue
        rel = str(path.relative_to(BACKEND_DIR))
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg in params:
                    found.setdefault(rel, set()).add(kw.arg)
    return found


def test_every_synthetic_prompt_runner_disables_post_processing():
    """The live invariant: a machine-authored turn is never mined."""
    sites = _subagent_param_call_sites()
    for rel in (
        "app/agent/cron_service.py",
        "app/agent/heartbeat_service.py",
        "app/agent/subagent.py",
        "app/agent/subagent_orchestrator.py",
        "app/agent/routines/agent_task_handler.py",
    ):
        assert "disable_post_processing" in sites.get(rel, set()), (
            f"{rel} builds a machine-authored user_message and does NOT pass "
            "disable_post_processing — the curator will mine the platform's "
            "own prompt as something the user said"
        )


def test_the_set_of_callers_is_audited():
    """Fails if the set GROWS or SHRINKS.

    A new caller is a new decision about whose words reach memory, and it
    should not be possible to make it silently. A removed one means a path
    started being mined again.
    """
    sites = set(_subagent_param_call_sites())
    allowed = set(_SUBAGENT_PARAM_CALLERS)
    new = sorted(sites - allowed)
    gone = sorted(allowed - sites)
    assert not new, (
        f"{new} newly pass a sub-agent parameter. If the turn is "
        "machine-authored that is correct — add it to "
        "_SUBAGENT_PARAM_CALLERS with the reason."
    )
    assert not gone, (
        f"{gone} stopped passing sub-agent parameters. If that is a "
        "machine-authored turn, its text is now being mined into memory."
    )


def test_every_audited_caller_states_a_reason():
    for rel, reason in _SUBAGENT_PARAM_CALLERS.items():
        assert len(reason) > 30, f"{rel}: an exception without a reason is a hole"


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
