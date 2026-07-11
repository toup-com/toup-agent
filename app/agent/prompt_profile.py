"""Prompt profile — gates which sections ``_build_system_prompt``
assembles for a given ``agent_runner.run()`` call.

Phase 3 of the sub-agent spawning arc. The profile is a discrete
enum the caller passes in; ``_build_system_prompt`` looks up the
allowed-section list and the assembly-time filter at
``agent_runner.py:2586-2587`` does the rest. Sections not in the
profile's list are silently dropped from the assembled prompt.

Profile semantics
-----------------
- ``FULL`` — every section the prompt builder produces, in the
  historic order. Today's behaviour for every user-facing turn
  (web, mobile, telegram, voice, extension, voice-realtime).
- ``SUBAGENT`` — a stripped prompt suitable for a non-user-facing
  child agent run. Drops persona/identity stack, user memories,
  active tasks, day-as-chat continuity (today_so_far / recent_days /
  reply_to), platform map, onboarding. Keeps just enough to use
  tools coherently: a task preamble, skills, environment,
  runtime context, formatting.

Why two profiles, not "minimal":
  Per Amendment 1, ``MINIMAL`` is intentionally omitted from v1.
  It would be needed for a non-tool utility caller (classify /
  title / summarize) but no such caller exists today; adding it
  speculatively risks evolving it in a direction that conflicts
  with the eventual real use case. We will add it when a concrete
  caller arrives.

What this module does NOT do
----------------------------
- Does not build the prompt sections themselves — that's
  ``_build_system_prompt``. This module is pure declarative
  metadata.
- Does not affect post-builder appended blocks (today_so_far,
  reply_to_directive, recent_days). Those are gated separately in
  ``run()`` via the same enum (the gate sits where the appends
  happen, not here).
"""
from __future__ import annotations

import enum


class PromptProfile(str, enum.Enum):
    """Controls which sections land in the assembled system prompt.

    Subclasses ``str`` so the value is JSON-serialisable and grep-able
    in logs (``profile=full`` reads more clearly than ``profile=0``).
    """

    FULL = "full"
    SUBAGENT = "subagent"
    AUTOPILOT = "autopilot"


# ──────────────────────────────────────────────────────────────────────
# Section allow-lists per profile.
#
# The order here is the ORDER sections appear in the assembled prompt
# when present. Mirrors agent_runner.py:2559-2580 SECTION_ORDER for
# FULL (changes here without a corresponding update there will silently
# drop sections — keep the lists in sync; the builder uses this module's
# allow-list directly so there's only one source of truth).
#
# Adding a new section to the builder requires adding it here too.
# ──────────────────────────────────────────────────────────────────────


# FULL — every section the builder produces. This is the historical
# ordering; reordering changes prompt behaviour, do not reorder
# casually.
_FULL_SECTIONS: tuple[str, ...] = (
    "identity",          # WHO the agent is (soul + behavioral)
    "identity_anchor",   # Don't break white-label by naming underlying LLM
    "voice_rules",       # Always-apply anti-chatbot tone
    "self_knowledge",    # HOW your memory works (F7)
    "platform_knowledge",  # WHAT Toup is — pages, capabilities
    "about_you",         # User's name + local time-of-day
    "owner_recognition", # Founder-only: this user owns Toup (gated)
    "user_brain",        # WHO the user is
    "active_tasks",      # CONTINUITY — open threads
    "agent_brain",       # Agent brain (env-flag gated)
    "work_brain",        # Work brain (env-flag gated)
    "skills",            # WHAT the agent can do
    "environment",       # WHAT the agent has access to
    "doc_generation",    # Document generation (flag-gated)
    "media",             # Media playback (web/app only)
    "runtime",           # WHEN/WHERE
    "vibecoding",        # Vibe coding mode override
    "formatting",        # HOW to respond
    "onboarding",        # Temporary onboarding instructions
    "activation",        # Optional activation prompt
    "verbose",           # Optional verbose mode
    "subagent_task_preamble",  # Sub-agent task statement (only
                               # present when the run was kicked off
                               # by the dispatcher; ignored when
                               # absent for FULL profile)
)


# SUBAGENT — stripped. Order chosen so the task statement is read
# first by the model (right after the section assembly point), then
# the tool catalog, then runtime + environment + formatting. No
# persona / memory / continuity surface.
#
# Pinned by test against `_FULL_SECTIONS` to guarantee it's a strict
# subset — that prevents typos here from creating sections the
# builder doesn't actually emit.
_SUBAGENT_SECTIONS: tuple[str, ...] = (
    "subagent_task_preamble",  # Task statement — the child's "why"
    "skills",                  # Skill catalog
    "environment",             # Terminal / DB / file / web tool surface
    "runtime",                 # Date, channel ("subagent"), workspace
    "formatting",              # Markdown rules (web-shape default)
)


# AUTOPILOT — autonomous mission ticks (Autopilot arc PR7). Richer
# than SUBAGENT (the mission acts FOR the user, so it needs identity +
# user context to make good choices) but stripped of foreground-only
# surfaces (media/onboarding/vibecoding/activation). Strict subset of
# _FULL_SECTIONS, pinned by test.
_AUTOPILOT_SECTIONS: tuple[str, ...] = (
    "identity",
    "voice_rules",
    "about_you",
    "user_brain",
    "active_tasks",
    "skills",
    "environment",
    "runtime",
    "formatting",
)


_SECTION_LISTS: dict[PromptProfile, tuple[str, ...]] = {
    PromptProfile.FULL: _FULL_SECTIONS,
    PromptProfile.SUBAGENT: _SUBAGENT_SECTIONS,
    PromptProfile.AUTOPILOT: _AUTOPILOT_SECTIONS,
}


def sections_for(profile: PromptProfile) -> tuple[str, ...]:
    """Allowed-section list for a profile, in assembly order.

    Used by ``agent_runner._build_system_prompt`` at the
    SECTION_ORDER filter (line ~2586) — anything in
    ``section_parts`` but not in this list is silently dropped, same
    as today's behaviour with the static SECTION_ORDER tuple.
    """
    return _SECTION_LISTS[profile]


def is_section_allowed(profile: PromptProfile, section_key: str) -> bool:
    """Fast O(N) membership check. N is small (≤ 21 for FULL,
    ≤ 5 for SUBAGENT) — sets aren't worth the boilerplate."""
    return section_key in _SECTION_LISTS[profile]


# ──────────────────────────────────────────────────────────────────────
# Post-builder block gates
#
# The three blocks appended in ``run()`` AFTER _build_system_prompt
# returns (today_so_far, reply_to_directive, recent_days at
# agent_runner.py:541-624) are tagged by a profile-aware boolean so
# the call site at run() doesn't have to know section names.
# ──────────────────────────────────────────────────────────────────────


_POST_BUILDER_ALLOWED: dict[PromptProfile, bool] = {
    PromptProfile.FULL: True,
    PromptProfile.SUBAGENT: False,
    # Missions carry their own continuity (goal + note + last summary
    # in the tick prompt) — day-chat blocks would just burn tokens on
    # every tick.
    PromptProfile.AUTOPILOT: False,
}


def allows_post_builder_blocks(profile: PromptProfile) -> bool:
    """Whether the post-builder appends (today_so_far,
    reply_to_directive, recent_days) should be included.

    FULL: True (today's behaviour).
    SUBAGENT: False (no day-chat continuity for child runs).
    """
    return _POST_BUILDER_ALLOWED[profile]


# ──────────────────────────────────────────────────────────────────────
# Tool-disable defaults for SUBAGENT profile
#
# Memory-write tools, spawn (no recursive grandchildren), and the
# job/routine/trigger mutators are removed from the LLM-visible tool
# list when a sub-agent run starts. The LLM literally cannot see
# the tools — prompt guidance is not load-bearing because prompts
# are advisory; tool-list omission is hard.
#
# memory_search is intentionally NOT in this set — a sub-agent may
# still need to read user memory to do its task. Only writes are
# blocked.
# ──────────────────────────────────────────────────────────────────────


SUBAGENT_DISABLED_TOOLS: frozenset[str] = frozenset({
    # Memory writes — child must not pollute user brain
    "memory_store",
    "memory_delete",
    # No grandchildren — v1 depth = 1
    "spawn",
    # No missions from sub-agents (Autopilot PR8)
    "start_mission",
    # Dashboard / sidebar surfaces are user-intent shapes; a sub-agent
    # creating jobs would confuse the activity feed
    "create_job",
    "update_job",
    # Schedule mutators — sub-agent should not change user's automations
    "routines__create",
    "routines__remind",
    "routines__update",
    "routines__delete",
    "routines__run_now",
    "triggers__create",
    "triggers__update",
    "triggers__delete",
    # Extension tools route through the user's Chrome side panel via a
    # WebSocket round-trip. They're meant for foreground UX where the
    # user's tab provides DOM context. For a background sub-agent doing
    # research, every request bounces user→server→user→server, adding
    # multi-second latency per call vs. the agent's native HTTP fetch.
    # Caught live 2026-05-25: nariman's research sub-agent spent ~3m on
    # work the native web_search/web_fetch pair finishes in ~30s. Force
    # sub-agents to the direct path. (User-facing turns keep them.)
    "extension_search",
    "extension_read",
    "extension_research",
})


# Unsupervised-action policy for autonomous mission ticks
# (docs/autopilot/PLAN.md D3). Deny-by-default for anything that
# mutates OUTSIDE the tenant workspace or rewires the user's
# automations; workspace file ops / exec / research stay available —
# that is how missions make progress. Outward mutation via CONNECTOR
# tools is separately denied at the channel layer
# (connector_dispatcher._MUTATES_DEFAULT_DENY_CHANNELS includes
# "autopilot"; per-user explicit allows still override there).
AUTOPILOT_DISABLED_TOOLS: frozenset[str] = frozenset({
    # Brain hygiene: missions may store findings, never delete.
    "memory_delete",
    # Dashboard / automation mutators — user-intent surfaces.
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
    # Extension tools need the user's foreground Chrome — pointless
    # (and slow) while the user is away. Same rationale as SUBAGENT.
    "extension_search",
    "extension_read",
    "extension_research",
    # Credential vault — never unsupervised (also channel-blocked).
    "save_streaming_credential",
    # No mission-from-mission recursion (Autopilot PR8).
    "start_mission",
})


def disabled_tools_for(profile: PromptProfile) -> frozenset[str]:
    """Default tool-disable set per profile. Merged with the user's
    own ``AgentConfig.disabled_tools`` at agent_runner.run() time."""
    if profile == PromptProfile.SUBAGENT:
        return SUBAGENT_DISABLED_TOOLS
    if profile == PromptProfile.AUTOPILOT:
        return AUTOPILOT_DISABLED_TOOLS
    return frozenset()
