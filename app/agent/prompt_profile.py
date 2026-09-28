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
    "vibecoding",        # Vibe coding mode override (flag-off path only
                         # once CHANNEL_ENVELOPE is on — see
                         # surface_contracts below)
    "surface_contracts", # W-A3: every long per-surface contract (voice,
                         # extension, vibecoding), always present and each
                         # scoped by the <runtime_envelope> Contract line.
                         # A key missing from this tuple is built and then
                         # SILENTLY DROPPED by the assembly filter.
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
# Toup for Mac — the tools that act on the user's OWN computer
#
# `agent-tool-relay.md` §2.5 item 9 names this file as a site the
# `desktop__*` family touches: "sub-agent and voice turns must not drive
# the Mac." Two sets, because the two halves fail differently.
#
# The names are a LITERAL here rather than an import from the skill. That
# module is loaded by the skill loader at boot and is gated on
# `settings.desktop_relay_enabled`; importing it from a module on the
# prompt path would evaluate it unconditionally and make a withheld
# capability's absence depend on an import side effect. The literal is
# kept honest by `tests/test_desktop_unattended_turns.py`, which asserts
# it equals the skill's own `ALL_TOOLS` / `CONSENT_TOOLS`.
#
# Withholding a name that is not in the array is a no-op, so with the flag
# off (the shipped default) every line below changes nothing.
DESKTOP_LOCAL_TOOLS: frozenset[str] = frozenset({
    "desktop__fs_list", "desktop__fs_read", "desktop__fs_search",
    "desktop__fs_write", "desktop__fs_mkdir", "desktop__fs_move",
    "desktop__fs_trash", "desktop__exec_run", "desktop__screen_capture",
    "desktop__ui_snapshot", "desktop__ui_click", "desktop__ui_type",
    "desktop__ui_key",
})

#: The half that changes something, runs something or drives the machine.
#: Each one stages a confirmation card before it reaches the Mac.
DESKTOP_MUTATING_TOOLS: frozenset[str] = frozenset({
    "desktop__fs_write", "desktop__fs_mkdir", "desktop__fs_move",
    "desktop__fs_trash", "desktop__exec_run", "desktop__ui_click",
    "desktop__ui_type", "desktop__ui_key",
})


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
}) | DESKTOP_LOCAL_TOOLS
# ^ Toup for Mac, the WHOLE family, reads included.
#
# A sub-agent is the turn furthest from the user: it runs unattended, and
# its input is routinely text the agent did not author — a fetched page, a
# file, an email — which is the `_EXTERNAL_CONTENT_TOOLS` threat arriving
# with a tool that can read `~/.ssh`. The reads are withheld along with
# the writes for that reason and not the extension's (latency): a refusal
# the user never sees is not a refusal, and the mutating half's
# confirmation card would be drawn against a turn nobody is watching.
#
# This is the conservative default, not a permanent product decision. If
# sub-agent research on local files is wanted later, it wants its own
# consent surface first.


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
}) | DESKTOP_LOCAL_TOOLS
# ^ Toup for Mac: a mission tick runs while the user is away, by
# definition. The mutating half would stage a card onto a surface nobody
# is looking at, and the read half would take a machine's contents into a
# turn with no one to notice — the same "unsupervised outward mutation"
# line this set already draws, applied to the one executor that is the
# user's own laptop.


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
# Toup for Mac is DELIBERATELY ABSENT from this set, and the reasoning is
# recorded because `agent-tool-relay.md` §2.5 item 9 asks for the opposite
# ("sub-agent and voice turns must not drive the Mac") and a reader will
# look for it here.
#
# Voice is ATTENDED — the user is speaking to the agent — so the safety
# argument that withholds the family from SUBAGENT/AUTOPILOT/trigger does
# not apply. What remains is a UX objection: `voice` is not in
# `connector_dispatcher._CONFIRMABLE_CHANNELS`, so a spoken
# `desktop__exec_run` ends the turn saying "a confirmation card is on your
# screen" about a screen the speaker may not be looking at.
#
# That is not a reason to put it HERE. This set is half of the
# channel-converge mechanism (`agent_runner.tool_defs_ignoring`): its
# members stay in the wire array for cache identity and are banned through
# `allowed_tools` plus the executor's disabled set. Its two tests
# (`test_channel_converge_voice_array.py`,
# `test_voice_answers_inline.py::test_voice_loses_the_three_deferral_tools`)
# pin the membership exactly, and every member is a CORE def that is always
# in the array — which a flag-gated skill tool is not. Fixing the voice UX
# belongs in the skill, where the card's surface is known, not in a
# cache-lineage set.



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
}) | DESKTOP_LOCAL_TOOLS
# ^ Toup for Mac: "unlike voice there is no user in the loop to notice"
# is the whole argument, and it applies hardest to a tool whose executor
# is the user's own laptop. An inbound email must not be able to ask for
# a file off it.


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
# What it deliberately KEEPS is the connector READ surface. Connector
# WRITES are decided one layer down, in `connector_dispatcher`: since
# R38 a mutating connector call from this channel is STAGED as a
# pending action and rendered in the thread as a `needs_you` turn with
# `fix: "approve"` — the elevation surface whose absence was the reason
# for the older blanket deny (b44815f7). Nothing runs until the user
# taps; what changed is that there is now a tap to make.
#
# R38: the set became a dict so every withholding carries its REASON in
# the same literal. A tool cannot be added here without one, and
# `test_r38_agent_hands.py::test_the_withheld_set_is_exactly_this`
# fails on any add or removal that does not also move the pinned list —
# so "why is this tool missing from the thread?" is answerable from the
# code rather than from a commit archaeology.
AUTOMATION_THREAD_WITHHELD: dict[str, str] = {
    # Progress cards in a surface that already has a run ledger: the
    # thread draws its own turns, so a second card is two accounts of
    # the same work.
    "create_job": "the thread has its own run ledger; a job card would "
                  "be a second, disagreeing record of the same work",
    "update_job": "it can only edit a card this surface never draws",
    # The answer has to land where the user is looking.
    "spawn": "hands the work to a child whose result this turn never "
             "waits for — the answer lands nowhere the user is reading",
    "start_mission": "moves the work out of the thread the user is "
                     "sitting in, and reports back somewhere else",
    "save_streaming_credential": "belt and braces with agent_runner's "
                                 "VAULT_TOOL_CHANNEL_BLOCK: a thread "
                                 "turn never has a credential to bank",
    # The memory writers STAY withheld, and this is the reason of
    # record (Round 33, item 6): the thread's turns are mostly about a
    # RUN — which account failed, why a read came back empty — and a
    # writer on this surface filed those as durable facts about the
    # USER. "Gmail could not be read" became a fact about the person,
    # surfaced weeks later in an unrelated chat. Reading memory is
    # still allowed (`memory_search`, `automations__memory_recall`);
    # what is withheld is the ability to write a run's transient
    # failure into the one store that outlives it. Nothing in R38
    # changes this — the elevation surface is about CONNECTOR writes,
    # which the user approves one at a time and can see the target of.
    # A memory write has no target the user could be shown.
    "memory_store": "a run's transient failure is not a fact about the "
                    "user, and this surface talks mostly about runs "
                    "(Round 33, item 6)",
    "memory_write_file": "same as memory_store: a durable write out of "
                         "a conversation about one run",
    "memory_edit_file": "same as memory_store: a durable write out of "
                        "a conversation about one run",
    "memory_delete": "destructive, unreviewable, and never what the "
                     "user meant by a question about an automation",
    # "Why did this fail?" must not quietly reschedule anything.
    "routines__create": "a question about a failure must not create "
                        "recurring work as a side effect",
    "routines__update": "the same edit, made invisibly, to something "
                        "the user is not looking at",
    "routines__delete": "destructive, and out of this thread's scope — "
                        "this automation is not that routine",
    "routines__remind": "a reminder created here surfaces somewhere "
                        "this thread cannot show or take back",
    "triggers__create": "a question about a failure must not arm a new "
                        "trigger as a side effect",
    "triggers__update": "the same edit, made invisibly, to something "
                        "the user is not looking at",
    "triggers__delete": "destructive, and out of this thread's scope",
}

AUTOMATION_THREAD_DISABLED_TOOLS: frozenset[str] = frozenset(
    AUTOMATION_THREAD_WITHHELD
)


def disabled_tools_for_channel(channel: str | None) -> frozenset[str]:
    """Extra tool-disable set implied by the SURFACE, independent of profile.

    Merged on top of the profile set and the user's own disabled list, so a
    voice sub-agent loses the union of both — which is already what both
    sets independently want.
    """
    normalized = (channel or "").strip().lower()
    # Only these attended surfaces can request access to the user's Mac.
    # New/background channel names stay dark until deliberately reviewed.
    desktop_denied = (
        frozenset() if normalized in {"web", "app", "mobile", "desktop", "voice"}
        else DESKTOP_LOCAL_TOOLS
    )
    if normalized == "voice":
        return VOICE_DISABLED_TOOLS | desktop_denied
    if normalized == "trigger":
        return TRIGGER_DISABLED_TOOLS | desktop_denied
    if normalized == "automation_thread":
        return AUTOMATION_THREAD_DISABLED_TOOLS | desktop_denied
    return desktop_denied
