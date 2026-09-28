"""Toup for Mac — merging this must change nothing, for anybody.

Why this file is the one that gates the merge
---------------------------------------------
The tools array serializes AHEAD of system+history in the provider prompt
prefix, so **any byte difference starts a separate provider cache lineage**
and every turn in that lineage re-bills the whole system+history tail at full
price (`tool_entitlements.py:21-39`; the audit behind `prefix_stability.py`
measured a 0 % production cache-hit rate from exactly this churn).

Two claims are made about this work and both are checked here against the
REAL skill loader rather than against a description of it:

  1. **Flag off, nothing moves.** With `desktop_relay_enabled=False` — the
     shipped default — the assembled skill array and the assembled
     system-prompt sections are byte-identical to what the same loader
     produces with the `desktop` skill directory not on disk at all. That
     second array is "main", constructed rather than remembered.

  2. **Flag on, presence is not in the array.** The 13 defs appear ALWAYS,
     never keyed on whether a Mac is connected. A lid closing would
     otherwise fork the cache lineage several times an hour, which is
     `agent-tool-relay.md` §2.6's headline finding and the reason
     availability is an execution-time question.

The channel-policy half is here too, because the same merge adds `desktop`
to `mcp_auth._KNOWN_CHANNELS` and to the dispatcher's confirmable set, and an
unregistered channel is silently clamped to `background` — the state
`extension` is in today (§2.4).

Lane: RUN_MODE=platform. Pure module-level assembly; no DB rows.

Run:
    cd backend && DATABASE_URL="sqlite+aiosqlite:///:memory:" RUN_MODE=platform \
        python -m pytest tests/test_desktop_tools_array_inertness.py -q
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

from app.agent import desktop_bridge, tool_entitlements as te
from app.agent.skills.loader import SkillLoader
from app.config import settings

DESKTOP_NAMES = {
    "desktop__fs_list", "desktop__fs_read", "desktop__fs_search",
    "desktop__fs_write", "desktop__fs_mkdir", "desktop__fs_move",
    "desktop__fs_trash", "desktop__exec_run", "desktop__screen_capture",
    "desktop__ui_snapshot", "desktop__ui_click", "desktop__ui_type",
    "desktop__ui_key",
}


@pytest.fixture(autouse=True)
def _pristine_entitlements():
    """`entitled_families()` memoizes per process on purpose — that memo is
    what makes the gate container-stable. Tests have to reach past it."""
    prev_flag = getattr(settings, "desktop_relay_enabled", False)
    prev_fams = settings.agent_tool_families
    te._RESOLVED = None
    desktop_bridge.reset_for_tests()
    yield
    settings.desktop_relay_enabled = prev_flag
    settings.agent_tool_families = prev_fams
    te._RESOLVED = None
    desktop_bridge.reset_for_tests()


async def _assemble(*, without_desktop_on_disk: bool = False) -> Tuple[str, str]:
    """(wire tools JSON, prompt sections JSON) from the REAL loader."""
    loader = SkillLoader()
    if without_desktop_on_disk:
        real = loader._discover_skill_files

        def pruned(dirs=None):
            return [(e, f) for (e, f) in real(dirs) if e != "desktop"]

        loader._discover_skill_files = pruned          # the tree before this work
    await loader.load_all()
    tools: List[Dict[str, Any]] = loader.get_all_tool_definitions()
    sections: List[str] = loader.get_all_system_prompt_sections()
    return (
        json.dumps(tools, separators=(",", ":"), sort_keys=False),
        json.dumps(sections, separators=(",", ":")),
    )


# ══════════════════════════════════════════════════════════════════════
# 1. Flag off — the merge is inert
# ══════════════════════════════════════════════════════════════════════
async def test_the_shipped_default_is_off():
    from app.config import Settings

    assert Settings.model_fields["desktop_relay_enabled"].default is False
    assert te.skill_enabled("desktop") is False


async def test_flag_off_is_byte_identical_to_the_tree_without_this_work():
    settings.desktop_relay_enabled = False
    te._RESOLVED = None

    actual_tools, actual_sections = await _assemble()
    main_tools, main_sections = await _assemble(without_desktop_on_disk=True)

    assert actual_tools == main_tools, (
        "the wire tools array moved with the flag OFF — every tenant starts a "
        "new provider cache lineage on merge and re-bills the whole "
        "system+history prefix behind it"
    )
    assert actual_sections == main_sections, (
        "the skill system-prompt sections moved with the flag off"
    )
    assert "desktop__" not in actual_tools


async def test_flag_off_removes_the_tools_the_prompt_AND_the_execution_path():
    """`skills/loader.py:390-409`: "a skill that is half-gated — tools hidden
    but still callable, or callable but absent from the capabilities listing
    — is worse than either extreme"."""
    settings.desktop_relay_enabled = False
    te._RESOLVED = None
    loader = SkillLoader()
    await loader.load_all()

    assert loader.get_skill("desktop") is None
    assert all(
        not n.startswith("desktop__")
        for n in (t["name"] for t in loader.get_all_tool_definitions())
    )
    assert not any(
        "user's Mac" in s for s in loader.get_all_system_prompt_sections()
    )
    for name in sorted(DESKTOP_NAMES):
        assert loader.is_skill_tool(name) is False


# ══════════════════════════════════════════════════════════════════════
# 2. Flag on — the family appears, and presence never enters the array
# ══════════════════════════════════════════════════════════════════════
async def test_flag_on_adds_exactly_the_thirteen_and_nothing_else():
    settings.desktop_relay_enabled = False
    te._RESOLVED = None
    off = json.loads((await _assemble())[0])

    settings.desktop_relay_enabled = True
    te._RESOLVED = None
    on = json.loads((await _assemble())[0])

    off_names = [t["name"] for t in off]
    on_names = [t["name"] for t in on]
    assert set(on_names) - set(off_names) == DESKTOP_NAMES
    assert set(off_names) - set(on_names) == set()
    # Everything that was already there is unchanged, byte for byte.
    kept = [t for t in on if not t["name"].startswith("desktop__")]
    assert json.dumps(kept, separators=(",", ":")) == json.dumps(
        off, separators=(",", ":")
    )


async def test_flag_on_the_array_is_identical_online_and_offline():
    """§2.6, and the whole reason `is_connected` is never consulted during
    assembly. Device presence flips on every laptop lid close."""
    settings.desktop_relay_enabled = True
    te._RESOLVED = None

    desktop_bridge.reset_for_tests()
    offline_tools, offline_sections = await _assemble()

    class _Sock:
        async def send_text(self, text: str) -> None: ...
        async def close(self, code: int = 1000, reason: str = "") -> None: ...

    await desktop_bridge.register(
        "00000000-0000-4000-8000-00000000000f", _Sock(), device_id="dev-1",
    )
    assert desktop_bridge.is_connected("00000000-0000-4000-8000-00000000000f")
    online_tools, online_sections = await _assemble()

    assert online_tools == offline_tools, (
        "the tools array is keyed on whether a Mac is online — that forks the "
        "provider cache lineage every time a lid closes"
    )
    assert online_sections == offline_sections


async def test_a_withheld_family_keeps_the_array_identical_even_with_the_flag_on():
    """The family is the per-tenant withhold (§5.7); the flag is the launch
    gate. `skill_enabled` consults BOTH, and either one alone is enough."""
    settings.desktop_relay_enabled = True
    settings.agent_tool_families = "doc_generation,app_builder"
    te._RESOLVED = None

    tools, sections = await _assemble()
    main_tools, main_sections = await _assemble(without_desktop_on_disk=True)
    assert tools == main_tools
    assert sections == main_sections


async def test_the_family_is_named_in_the_shipped_loadout_so_one_flag_launches_it():
    """A family absent from `AGENT_TOOL_FAMILIES` would make
    `DESKTOP_RELAY_ENABLED=1` a silent no-op — the two-gates-one-intent trap
    the automations rollout already paid for."""
    from app.config import Settings

    default = Settings.model_fields["agent_tool_families"].default
    assert "desktop" in {f.strip() for f in default.split(",")}

    settings.agent_tool_families = default
    settings.desktop_relay_enabled = True
    te._RESOLVED = None
    assert te.skill_enabled("desktop") is True
    assert "desktop__exec_run" in (await _assemble())[0]


def test_the_family_exists_and_names_only_this_skill():
    fam = te.FAMILIES["desktop"]
    assert fam.skills == frozenset({"desktop"})
    assert fam.tool_names == frozenset()          # no core defs; it is a skill
    assert "desktop" in te.ALL_FAMILIES


# ══════════════════════════════════════════════════════════════════════
# 3. Channel policy — registered on purpose, not clamped by accident
# ══════════════════════════════════════════════════════════════════════
def test_the_desktop_channel_is_registered_and_not_clamped():
    """§2.4: an unrecognised channel is clamped to `background`, which is in
    the unattended deny set — "the right policy by accident", logged as a
    warning on every tool call, and silently different the day the clamp
    target moves. `extension` is in that state today."""
    from app.mcp_auth import _KNOWN_CHANNELS, _UNKNOWN_CHANNEL_CLAMP

    assert "desktop" in _KNOWN_CHANNELS
    assert _UNKNOWN_CHANNEL_CLAMP == "background"


def test_desktop_can_draw_a_confirmation_card():
    """`connector_dispatcher.py:638-657`: a tool needing confirmation on a
    channel that cannot draw a card is REFUSED, never silently run. Being in
    this set is what makes the refusal unnecessary rather than the card
    impossible."""
    from app.services.connector_dispatcher import _CONFIRMABLE_CHANNELS

    assert "desktop" in _CONFIRMABLE_CHANNELS


def test_desktop_is_attended_so_its_connector_writes_are_not_denied():
    from app.services.connector_dispatcher import (
        _MUTATES_DEFAULT_DENY_CHANNELS,
        _MUTATES_UNATTENDED_DENY_CHANNELS,
    )

    assert "desktop" not in _MUTATES_DEFAULT_DENY_CHANNELS
    assert "desktop" not in _MUTATES_UNATTENDED_DENY_CHANNELS


def test_every_channel_the_deny_sets_name_is_a_known_channel():
    """`mcp_auth.py:99-105` states the invariant: `_KNOWN_CHANNELS` must stay
    a superset of every name the dispatcher's deny sets reference, or the
    deny is bypassed by the transport's rewrite before policy ever sees it.

    This is a CONTROL as much as an assertion: it is what would have caught
    `desktop` being added to one file and not the other."""
    from app.mcp_auth import _KNOWN_CHANNELS
    from app.services.connector_dispatcher import (
        _CONFIRMABLE_CHANNELS,
        _MUTATES_CONFIRM_CHANNELS,
        _MUTATES_DEFAULT_DENY_CHANNELS,
        _MUTATES_UNATTENDED_DENY_CHANNELS,
    )

    referenced = (
        set(_CONFIRMABLE_CHANNELS) | set(_MUTATES_CONFIRM_CHANNELS)
        | set(_MUTATES_DEFAULT_DENY_CHANNELS)
        | set(_MUTATES_UNATTENDED_DENY_CHANNELS)
    )
    assert referenced <= set(_KNOWN_CHANNELS), sorted(referenced - set(_KNOWN_CHANNELS))


# ══════════════════════════════════════════════════════════════════════
# 4. The routes are NOT gated on the flag
# ══════════════════════════════════════════════════════════════════════
async def test_revocation_works_with_the_feature_dark():
    """A kill switch that the kill switch can switch off is not one. A user
    who paired a Mac and then had the flag turned off must still be able to
    list and revoke it."""
    from app.api import desktop as desktop_api

    settings.desktop_relay_enabled = False
    paths = {getattr(r, "path", "") for r in desktop_api.router.routes}
    assert "/desktop/devices" in paths
    assert "/desktop/devices/revoke" in paths
    import inspect
    assert "desktop_relay_enabled" not in inspect.getsource(desktop_api.revoke_device)
    assert "_connections_on" not in inspect.getsource(desktop_api.revoke_device)
