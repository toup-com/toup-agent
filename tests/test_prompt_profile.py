"""PromptProfile module — sections allow-list + post-builder gate +
tool-disable defaults.

Phase 3 of the sub-agent spawning arc. These pin the contract that
``agent_runner._build_system_prompt`` and ``agent_runner.run()``
depend on:

  - FULL profile is the historical behaviour (every section).
  - SUBAGENT profile is a strict subset (no persona, no memory, no
    continuity).
  - The subagent_task_preamble key is in FULL too even though only
    SUBAGENT renders it — so a careless caller passing
    subagent_task_label=... to a FULL run doesn't silently drop the
    section due to a missing entry.
  - Tool-disable defaults exclude memory_search (sub-agent may still
    READ user memory) and the LLM-callable mutation surfaces are
    blocked.
"""
from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────
# Enum identity + str-subclass
# ──────────────────────────────────────────────────────────────────────


def test_promptprofile_is_str_subclass():
    """Subclassing str gives us JSON-serialisable values + grep-able
    log lines. ``profile=full`` reads more clearly than ``profile=0``."""
    from app.agent.prompt_profile import PromptProfile

    assert isinstance(PromptProfile.FULL, str)
    assert PromptProfile.FULL == "full"
    assert PromptProfile.SUBAGENT == "subagent"


def test_promptprofile_has_only_full_and_subagent_v1():
    """Amendment 1: MINIMAL is intentionally omitted from v1 to avoid
    speculative API surface. Pin the membership so a 'helpful' PR
    that adds MINIMAL up-front fails review here."""
    from app.agent.prompt_profile import PromptProfile

    assert {p.value for p in PromptProfile} == {"full", "subagent"}


# ──────────────────────────────────────────────────────────────────────
# sections_for() / SUBAGENT is a strict subset
# ──────────────────────────────────────────────────────────────────────


def test_full_profile_has_every_historic_section():
    """FULL mirrors agent_runner.py:2595-2615 historical
    SECTION_ORDER one-for-one. A typo here would silently drop a
    section."""
    from app.agent.prompt_profile import PromptProfile, sections_for

    expected = (
        "identity",
        "identity_anchor",
        "voice_rules",
        "self_knowledge",
        "platform_knowledge",
        "about_you",
        "owner_recognition",
        "user_brain",
        "active_tasks",
        "agent_brain",
        "work_brain",
        "skills",
        "environment",
        "doc_generation",
        "media",
        "runtime",
        "vibecoding",
        "formatting",
        "onboarding",
        "activation",
        "verbose",
        "subagent_task_preamble",  # present in FULL list so a FULL
                                   # run with subagent_task_label set
                                   # isn't silently dropped
    )
    assert sections_for(PromptProfile.FULL) == expected


def test_subagent_profile_is_a_strict_subset_of_full():
    """A typo in the SUBAGENT list (e.g. 'runtim') would create a
    section the builder never emits. Pin the subset relation."""
    from app.agent.prompt_profile import PromptProfile, sections_for

    full = set(sections_for(PromptProfile.FULL))
    sub = set(sections_for(PromptProfile.SUBAGENT))
    missing = sub - full
    assert not missing, (
        f"SUBAGENT contains sections not in FULL — typo? {missing}"
    )


def test_subagent_profile_drops_persona_memory_continuity():
    """The whole point of SUBAGENT: a stripped prompt. Pin the
    sections that must be absent so a 'helpful' PR adding the user
    brain back lights up here."""
    from app.agent.prompt_profile import PromptProfile, sections_for

    sub = set(sections_for(PromptProfile.SUBAGENT))
    must_be_absent = {
        # Persona stack
        "identity",
        "identity_anchor",
        "voice_rules",
        "self_knowledge",
        # User memory
        "user_brain",
        "active_tasks",
        "about_you",
        # Platform context
        "platform_knowledge",
        # Continuity / onboarding
        "onboarding",
        # Channel-specific overrides (sub-agents don't get vibecoding mode)
        "vibecoding",
        "media",
    }
    leaked = must_be_absent & sub
    assert not leaked, (
        f"SUBAGENT must NOT carry these sections: {leaked}"
    )


def test_subagent_profile_keeps_tool_surface_sections():
    """Even stripped, the sub-agent needs to know its tools, runtime
    context, and how to format the final summary. Pin the must-have
    sections."""
    from app.agent.prompt_profile import PromptProfile, sections_for

    sub = set(sections_for(PromptProfile.SUBAGENT))
    must_be_present = {
        "subagent_task_preamble",  # the task statement
        "skills",                  # tool catalog
        "environment",             # exec/file/web tool surface
        "runtime",                 # date, channel
        "formatting",              # markdown rules
    }
    missing = must_be_present - sub
    assert not missing, (
        f"SUBAGENT must carry these sections: {missing}"
    )


def test_subagent_task_preamble_is_first_section():
    """The task statement is the model's 'why' — it must be at the
    top so it's the highest-attention chunk."""
    from app.agent.prompt_profile import PromptProfile, sections_for

    sub = sections_for(PromptProfile.SUBAGENT)
    assert sub[0] == "subagent_task_preamble"


# ──────────────────────────────────────────────────────────────────────
# is_section_allowed / allows_post_builder_blocks
# ──────────────────────────────────────────────────────────────────────


def test_is_section_allowed_matches_sections_for():
    """is_section_allowed must be a pure membership check on
    sections_for — pin the contract."""
    from app.agent.prompt_profile import (
        PromptProfile, is_section_allowed, sections_for,
    )

    for profile in PromptProfile:
        for section in sections_for(profile):
            assert is_section_allowed(profile, section)
        assert not is_section_allowed(profile, "no_such_section")


def test_allows_post_builder_blocks_full_yes_subagent_no():
    """The three post-builder appends (today_so_far,
    reply_to_directive, recent_days) are user-facing day-chat
    continuity surfaces. SUBAGENT skips them."""
    from app.agent.prompt_profile import (
        PromptProfile, allows_post_builder_blocks,
    )

    assert allows_post_builder_blocks(PromptProfile.FULL) is True
    assert allows_post_builder_blocks(PromptProfile.SUBAGENT) is False


# ──────────────────────────────────────────────────────────────────────
# Tool-disable defaults
# ──────────────────────────────────────────────────────────────────────


def test_full_profile_disables_no_tools_by_default():
    """A FULL run must not have extra tool restrictions beyond the
    user's own AgentConfig.disabled_tools — the profile is not where
    user preferences live."""
    from app.agent.prompt_profile import PromptProfile, disabled_tools_for

    assert disabled_tools_for(PromptProfile.FULL) == frozenset()


def test_subagent_profile_disables_expected_tools():
    """The closed set the sub-agent default applies. Pinned so a
    careless 'let me add X back' lights up here."""
    from app.agent.prompt_profile import (
        PromptProfile, disabled_tools_for, SUBAGENT_DISABLED_TOOLS,
    )

    assert disabled_tools_for(PromptProfile.SUBAGENT) == SUBAGENT_DISABLED_TOOLS

    # The shape contract — what the closed set MUST include.
    must_include = {
        "memory_store",
        "memory_delete",
        "spawn",
        "create_job",
        "update_job",
        "routines__create",
        "routines__remind",
        "routines__update",
        "routines__delete",
        "routines__run_now",
        "triggers__create",
        "triggers__update",
        "triggers__delete",
    }
    assert must_include.issubset(SUBAGENT_DISABLED_TOOLS), (
        f"SUBAGENT_DISABLED_TOOLS missing: "
        f"{must_include - SUBAGENT_DISABLED_TOOLS}"
    )


def test_subagent_profile_does_not_disable_memory_search():
    """Read-only memory access is fine — a sub-agent may legitimately
    need to look up what the user has stored. The block is on
    WRITES, not reads."""
    from app.agent.prompt_profile import SUBAGENT_DISABLED_TOOLS

    assert "memory_search" not in SUBAGENT_DISABLED_TOOLS


def test_subagent_disabled_tools_is_immutable():
    """frozenset, not set — the caller can't mutate the default
    accidentally."""
    from app.agent.prompt_profile import SUBAGENT_DISABLED_TOOLS

    assert isinstance(SUBAGENT_DISABLED_TOOLS, frozenset)
