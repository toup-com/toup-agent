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
  child agent run. Drops persona/identity stack, user memory files,
  day-as-chat continuity (today_so_far / recent_days / reply_to),
  platform map, onboarding. Keeps just enough to use tools
  coherently: a task preamble, skills, environment, runtime context,
  formatting.

  v3 note: ``active_tasks`` left every list with the block itself —
  the working-file render was built from ``active_task`` memory rows,
  which the file model retires (rebuild-2026-08-v3 §1.1). What the
  user is in the middle of lives in Current context now.

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
    "user_brain",        # WHO the user is — the memory files (v3 §3.1)
    "agent_brain",       # Agent brain (env-flag gated)
    "work_brain",        # Work brain (env-flag gated)
    "skills",            # WHAT the agent can do
    "environment",       # WHAT the agent has access to
    "doc_generation",    # Document generation (flag-gated)
    "media",             # Media playback (web/app only)
    "runtime",           # WHEN/WHERE
    "vibecoding",        # Vibe coding mode override
    "automation_session",  # R29: automation session-thread context —
                           # only present when the turn is addressed
                           # to an automation's thread
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
    # v3: a sub-agent gets no user_brain section (prompt_profile above),
    # so handing it a tool that returns a whole memory file in full would
    # reopen the isolation the section list closes. `memory_search` keeps
    # its exemption below — it answers a scoped question; this one hands
    # over Profile.
    "memory_read_file",
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


# ──────────────────────────────────────────────────────────────────────
# Voice: the deferral tools are removed, not discouraged
#
# Voice is a real-time interface. The user is holding a live audio
# session open, so an answer that arrives "later, in Mission Control" is
# not a slower answer — it is no answer, delivered to a surface they are
# not looking at. Every tool below moves work OUT of the current turn.
#
# Measured on the founder's account, 2026-08-01T00:25Z. He asked, in
# Farsi, for U of T professors working on LLMs. In 145 seconds the agent
# produced THREE background jobs and zero spoken answers:
#
#   00:25:24  create_job → "Find UofT LLM professors"     (cancelled)
#   00:25:57  spawn      → "UofT LLM/NLP professor …"     (failed)
#   00:27:23  spawn      → "UofT LLM/NLP professor …"     (failed)
#
# Both subagent rows recorded total_tokens=0 and credit_spent=0.0 over
# the 19 minutes they were alive: they never ran at all, then an agent
# restart marked them infra_interrupted. Meanwhile the voice model said
# «یه گزارش جمع‌وجور و به‌دردبخور به فارسی برات میاد» — "a compact, useful
# report in Farsi is coming to you". Nothing ever came. He re-asked twice,
# and each re-ask minted another job, because nothing dedupes a request
# against work already in flight.
#
# Prompt guidance cannot fix this, for the reason SUBAGENT_DISABLED_TOOLS
# already states: prompts are advisory, tool-list omission is hard. The
# model followed the prompt correctly — the FULL profile's decision rules
# literally say "research … → call `create_job` FIRST". So voice loses
# the tools instead.
#
# `start_mission` is deliberately NOT here. It is the one deferral the
# user asks for in words ("while I'm away", "keep working on X"), it is
# the surface Mission Control was built for, and it reports back through
# channels the user will actually see. Removing it would leave a real
# request with nowhere to go. `create_job`/`update_job` draw a progress
# card and perform no work; `spawn` hands the work to a child whose
# result this turn never waits for. None of the three can put an answer
# in the user's ear.
VOICE_DISABLED_TOOLS: frozenset[str] = frozenset({
    "create_job",
    "update_job",
    "spawn",
})


# G-19b: an email trigger's runner turn is UNATTENDED background work.
# It must not schedule MORE background work — a trigger that spawns
# missions/jobs on every inbound email is a fork bomb with a Gmail
# fuse, and unlike voice there is no user in the loop to notice.
# `start_mission` IS here (voice keeps it because the user asks for it
# in words; no one asked for anything on a trigger turn).
# `save_streaming_credential` is also denied — belt and braces with
# VAULT_TOOL_CHANNEL_BLOCK in agent_runner, which already strips the
# tool for channel="trigger": prompts are advisory, tool-list omission
# is hard, and this set survives if the vault block set ever drifts.
TRIGGER_DISABLED_TOOLS: frozenset[str] = frozenset({
    "create_job",
    "update_job",
    "spawn",
    "start_mission",
    "save_streaming_credential",
})


# Round 33, item 8: an automation THREAD turn is the user asking a
# question inside one automation. Until now it was answered by a bare
# `llm_service.complete()` with no tools at all, so "give me my last
# five gmail" was answered "I could not read Gmail" in the thread and
# answered correctly in the main chat a minute later, on the same
# account. The thread now runs the SAME agent loop the main chat runs —
# same MCP connector tools, same `connector_dispatcher.execute`, same
# credentials — which is the only way the two surfaces can agree.
#
# What it loses is the set with no business in a thread:
# `create_job`/`update_job` would draw a progress card in a surface that
# has its own run ledger; `spawn`/`start_mission` move the answer
# somewhere the user is not looking; the memory writers are what filed a
# run's connector failures as facts about the user (item 6); the
# routine/trigger mutators would let "why did this fail?" quietly
# reschedule something. Prompts are advisory — tool-list omission is
# hard.
#
# What it deliberately KEEPS is the connector READ surface. Writes are
# refused one layer down: `automation_thread` IS in
# `connector_dispatcher._MUTATES_UNATTENDED_DENY_CHANNELS` (b44815f7),
# because the elevation-confirmation card a mutating call would need
# has no proven surface in the thread yet, and a confirmation the
# surface cannot show is a wedge. The earlier position recorded here —
# attended surface, so writes should meet the main chat's elevation
# flow ("Post that to Slack", asked inside the automation whose job is
# posting to Slack, is a reasonable thing to be able to say) — is the
# direction of travel once the thread can render that card; until then
# the deny list is the enforcement and this comment matches it.
AUTOMATION_THREAD_DISABLED_TOOLS: frozenset[str] = frozenset({
    "create_job",
    "update_job",
    "spawn",
    "start_mission",
    "save_streaming_credential",
    "memory_store",
    "memory_write_file",
    "memory_edit_file",
    "memory_delete",
    "routines__create",
    "routines__update",
    "routines__delete",
    "routines__remind",
    "triggers__create",
    "triggers__update",
    "triggers__delete",
})


def disabled_tools_for_channel(channel: str | None) -> frozenset[str]:
    """Extra tool-disable set implied by the SURFACE, independent of profile.

    Merged on top of the profile set and the user's own disabled list, so a
    voice sub-agent loses the union of both — which is already what both
    sets independently want.
    """
    if (channel or "").strip().lower() == "voice":
        return VOICE_DISABLED_TOOLS
    if (channel or "").strip().lower() == "trigger":
        return TRIGGER_DISABLED_TOOLS
    if (channel or "").strip().lower() == "automation_thread":
        return AUTOMATION_THREAD_DISABLED_TOOLS
    return frozenset()
