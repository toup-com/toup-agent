"""Under `channel_converge`, EVERY channel must send the same wire array.

Why this is stricter than test_channel_converge_voice_array.py
--------------------------------------------------------------
That file pins the one channel that was broken. This one pins the
*invariant*, over every channel string the runner actually routes — so the
next channel added to the product cannot quietly re-open the hole.

The mechanism it guards: the tools array serializes AHEAD of the system
prompt and the history, so a channel whose array differs by a single byte
starts its own provider cache lineage. Every hop into that channel then
re-bills the entire prefix behind the array at full price, and the loss is
invisible in behaviour — the model answers correctly, it just costs more.

Two independent mechanisms can drop a tool for a channel, and converge was
originally written for only the first:

  A. ``strip_tools_for_channel`` — the vault confirmation card, the
     ``app_builder__*`` family, and the ``app`` channel's mutation strip.
  B. ``disabled_tools_for_channel`` — the SURFACE disable set
     (``VOICE_DISABLED_TOOLS``), applied in ``run()`` long before the array
     is built, so ``tool_defs`` filters it out of the array itself.

Measured on the real definitions, 2026-08-05 — before the (B) fix:

    web-family (11 channels): 52 defs, 9,116 tok
    voice:                    49 defs, 8,233 tok    -> a second lineage

Converge is worth having only if the answer is exactly one lineage, so the
count is asserted directly rather than inferred.

Run:
    cd backend && RUN_MODE=agent PYTHONPATH=. \
        pytest tests/test_all_channels_one_lineage.py
"""
from __future__ import annotations

import json

import pytest

from app.agent.agent_runner import (
    AgentRunner,
    _RUN_DISABLED_TOOLS_CTX,
    strip_vault_tool_for_channel,
)
from app.agent.prefix_stability import channel_banned_names, strip_tools_for_channel
from app.agent.prompt_profile import disabled_tools_for_channel
from app.agent.tool_definitions import get_agent_tools, get_extended_tools

# Every channel string the runner routes, plus the unset case. `autopilot`
# and the subagent profile are deliberately absent: they carry a different
# SYSTEM prompt, so they are a separate lineage by design and converging
# their tool array would buy nothing.
CHANNELS = (
    "web", "telegram", "whatsapp", "voice", "mobile",
    "extension", "app", "vibecoding", "sms", "email", "api", "",
)

# `app_builder__*` tools are skill-provided, so the core array has none and
# the vibecoding/app branch of strip_tools_for_channel would otherwise be
# exercised against zero matches — a green that proves nothing.
_SKILL_TOOLS = [
    {"name": f"app_builder__{n}", "description": f"app builder {n}",
     "input_schema": {"type": "object"}}
    for n in ("create", "deploy", "list")
]


def _all_tools():
    return get_agent_tools() + get_extended_tools() + list(_SKILL_TOOLS)


def _runner(tools):
    """A bare runner — only the tool-composition surface is exercised."""
    r = AgentRunner.__new__(AgentRunner)
    r._core_tool_defs = list(tools)
    r.skill_loader = None
    r.tools = None
    r._disabled_tool_names = frozenset()
    return r


def _wire_array(channel: str, tools) -> str:
    """The bytes this channel would put on the wire under converge."""
    surface = disabled_tools_for_channel(channel)
    r = _runner(tools)
    token = _RUN_DISABLED_TOOLS_CTX.set(frozenset(surface))
    try:
        return json.dumps(r.tool_defs_ignoring(surface), separators=(",", ":"))
    finally:
        _RUN_DISABLED_TOOLS_CTX.reset(token)


@pytest.fixture(autouse=True)
def _clean_ctx():
    token = _RUN_DISABLED_TOOLS_CTX.set(None)
    yield
    _RUN_DISABLED_TOOLS_CTX.reset(token)


def test_every_channel_shares_one_cache_lineage():
    """The contract. Identical bytes => one provider cache lineage.

    MUTATION: revert the converge branch to `list(all_tools)` and this goes
    red with voice split into its own lineage.
    """
    tools = _all_tools()
    lineages: dict[str, list[str]] = {}
    for ch in CHANNELS:
        lineages.setdefault(_wire_array(ch, tools), []).append(ch or "(unset)")

    assert len(lineages) == 1, (
        "channels do not share one wire array, so each extra lineage re-bills "
        "the whole system+history prefix on every hop into it: "
        + " | ".join(
            f"{len(json.loads(k))} defs: {', '.join(sorted(v))}"
            for k, v in lineages.items()
        )
    )


def test_the_array_under_test_is_real():
    """Anti-vacuity control.

    An empty or near-empty array would make the equality above trivially
    true. Pin that the fixture carries the real tool set, including the
    skill tools whose strip branch would otherwise never fire.
    """
    tools = _all_tools()
    assert len(tools) > 20, f"tool array too small to prove anything: {len(tools)}"
    assert sum(1 for t in tools if t.get("name", "").startswith("app_builder__")) == 3

    # And show the channels genuinely DID differ before the fix, on this
    # same data — otherwise the test above cannot be detecting anything.
    unfixed: set[str] = set()
    for ch in CHANNELS:
        r = _runner(tools)
        token = _RUN_DISABLED_TOOLS_CTX.set(
            frozenset(disabled_tools_for_channel(ch))
        )
        try:
            unfixed.add(json.dumps(r.tool_defs, separators=(",", ":")))
        finally:
            _RUN_DISABLED_TOOLS_CTX.reset(token)
    assert len(unfixed) > 1, (
        "the pre-fix arrays are already identical on this fixture, so the "
        "one-lineage assertion cannot be detecting the fix"
    )


@pytest.mark.parametrize("channel", CHANNELS)
def test_channel_policy_survives_convergence(channel):
    """Converge keeps the array whole — so the policy it used to encode by
    ABSENCE must be carried by the banned set instead.

    Every name either strip mechanism would have removed has to end up in
    `channel_banned_names | disabled_tools_for_channel`, which feeds both
    the allowed_tools restriction and the executor's refusal path. A name
    that falls through both is a tool the channel was never meant to expose.
    """
    tools = _all_tools()
    surface = frozenset(disabled_tools_for_channel(channel))
    banned = channel_banned_names(
        tools, channel, strip_vault_tool_for_channel=strip_vault_tool_for_channel,
    ) | surface

    kept = {
        (t.get("name") or "")
        for t in strip_tools_for_channel(
            list(tools), channel,
            strip_vault_tool_for_channel=strip_vault_tool_for_channel,
        )
    }
    stripped = {(t.get("name") or "") for t in tools} - kept

    leaked = sorted((stripped | surface) - banned)
    assert not leaked, (
        f"{channel or '(unset)'} would expose {leaked} under converge — the "
        "array no longer hides them and nothing bans them"
    )


def test_at_least_one_channel_actually_bans_something():
    """Anti-vacuity control for the parametrized test above.

    If every channel banned nothing, all twelve cases would pass while
    asserting nothing at all.
    """
    tools = _all_tools()
    total = {
        ch: len(
            channel_banned_names(
                tools, ch,
                strip_vault_tool_for_channel=strip_vault_tool_for_channel,
            )
            | frozenset(disabled_tools_for_channel(ch))
        )
        for ch in CHANNELS
    }
    assert sum(total.values()) > 0, "no channel bans anything — nothing was tested"
    assert total["voice"] >= 4, (
        f"voice should ban the vault card + the 3 surface-disabled tools, got "
        f"{total['voice']}"
    )
    assert total["app"] >= 8, (
        f"app should ban the exec family + app_builder__*, got {total['app']}"
    )
