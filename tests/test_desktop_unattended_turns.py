"""Toup for Mac — which turns may reach the user's own computer.

`agent-tool-relay.md` §2.5 item 9: "sub-agent and voice turns must not
drive the Mac." The gate is tool-list OMISSION, not prompt guidance, for
the reason `SUBAGENT_DISABLED_TOOLS` already states in this repo: prompts
are advisory, a missing tool is hard.

Four surfaces, three answers:

    SUBAGENT   nothing   an unattended child turn whose input is routinely
                         text the agent did not author — the injection
                         threat `_EXTERNAL_CONTENT_TOOLS` exists for,
                         holding a tool that can read ~/.ssh
    AUTOPILOT  nothing   a mission tick runs while the user is away
    trigger    nothing   "unlike voice there is no user in the loop"
    voice      reads     a spoken "what's the error in that file?" answers
                         inside the turn; a spoken `exec_run` ends it with
                         "a card is on your screen" into an audio session

Lane: RUN_MODE=platform. Pure set algebra plus one loader assembly.

Run:
    cd backend && DATABASE_URL="sqlite+aiosqlite:///:memory:" RUN_MODE=platform \
        python -m pytest tests/test_desktop_unattended_turns.py -q
"""

from __future__ import annotations

import pytest

from app.agent.prompt_profile import (
    DESKTOP_LOCAL_TOOLS,
    DESKTOP_MUTATING_TOOLS,
    PromptProfile,
    disabled_tools_for,
    disabled_tools_for_channel,
)
from app.agent.skills.builtins.desktop.skill import (
    ALL_TOOLS,
    CONSENT_TOOLS,
    READ_TOOLS,
)


def test_the_literal_has_not_drifted_from_the_skill():
    """`prompt_profile` names these as a literal on purpose — it must not
    import the skill module, which is loaded behind a launch flag. This is
    the drift guard that buys that."""
    assert DESKTOP_LOCAL_TOOLS == ALL_TOOLS
    assert DESKTOP_MUTATING_TOOLS == CONSENT_TOOLS
    assert DESKTOP_LOCAL_TOOLS - DESKTOP_MUTATING_TOOLS == READ_TOOLS


def test_a_subagent_gets_none_of_the_family():
    withheld = disabled_tools_for(PromptProfile.SUBAGENT)
    assert DESKTOP_LOCAL_TOOLS <= withheld


def test_an_autopilot_mission_tick_gets_none_of_the_family():
    withheld = disabled_tools_for(PromptProfile.AUTOPILOT)
    assert DESKTOP_LOCAL_TOOLS <= withheld


def test_a_trigger_turn_gets_none_of_the_family():
    withheld = disabled_tools_for_channel("trigger")
    assert DESKTOP_LOCAL_TOOLS <= withheld


def test_a_voice_turn_keeps_the_whole_family_and_that_is_on_purpose():
    """Voice is ATTENDED, so the unattended argument does not reach it —
    and `VOICE_DISABLED_TOOLS` is half of the channel-converge mechanism,
    not a safety set. `test_channel_converge_voice_array.py` requires every
    member to be a core def that is always in the array, which a flag-gated
    skill tool is not.

    Recorded as a test rather than only a comment because "we considered it
    and said no" and "nobody thought of it" look identical in a diff. The
    open UX issue — a spoken mutating call ends the turn pointing at a card
    on a screen the speaker may not be looking at — is in the return notes,
    and it belongs in the skill."""
    withheld = disabled_tools_for_channel("voice")
    assert not (DESKTOP_LOCAL_TOOLS & withheld)
    assert withheld == frozenset({"create_job", "update_job", "spawn"})


def test_an_ordinary_turn_keeps_the_whole_family():
    for profile in (PromptProfile.FULL,):
        assert not (DESKTOP_LOCAL_TOOLS & disabled_tools_for(profile))
    for channel in ("web", "app", "mobile", "desktop", "voice"):
        assert not (DESKTOP_LOCAL_TOOLS & disabled_tools_for_channel(channel))


async def test_the_real_assembly_drops_them_for_a_subagent():
    """Not set algebra: the actual filter in `AgentRunner.tool_defs` removes
    names from the COMBINED array (core + skills + connectors)."""
    from app.agent import tool_entitlements as te
    from app.agent.skills.loader import SkillLoader
    from app.config import settings

    prev = getattr(settings, "desktop_relay_enabled", False)
    prev_fams = settings.agent_tool_families
    settings.desktop_relay_enabled = True
    te._RESOLVED = None
    try:
        loader = SkillLoader()
        await loader.load_all()
        assert loader.get_skill("desktop") is not None, "nothing to withhold"

        from app.agent.agent_runner import AgentRunner

        runner = AgentRunner(llm_service=None, tool_executor=None)
        runner.skill_loader = loader

        runner._disabled_tool_names = frozenset()
        lit = {t.get("name") for t in runner.tool_defs}
        assert DESKTOP_LOCAL_TOOLS <= lit

        runner._disabled_tool_names = disabled_tools_for(PromptProfile.SUBAGENT)
        dark = {t.get("name") for t in runner.tool_defs}
        assert not (DESKTOP_LOCAL_TOOLS & dark)
        # …and nothing else vanished with them.
        assert lit - dark == DESKTOP_LOCAL_TOOLS | (
            lit & disabled_tools_for(PromptProfile.SUBAGENT)
        )
    finally:
        settings.desktop_relay_enabled = prev
        settings.agent_tool_families = prev_fams
        te._RESOLVED = None


@pytest.mark.parametrize("channel", [None, "", "automation", "zzz", "cron", "routine",
    "heartbeat", "agent_task", "email_briefing", "background", "subagent", "app_builder",
    "automation_thread"])
def test_unattended_and_unknown_channels_have_no_mac_tools(channel):
    assert DESKTOP_LOCAL_TOOLS <= disabled_tools_for_channel(channel)
