"""channel_converge must converge VOICE too, not just the strip channels.

The flag's whole purpose is one provider cache lineage across channels:
the wire tools array serializes AHEAD of system+history, so any channel
sending a different array re-bills the entire prefix behind it.

It had a hole. Two mechanisms remove tools for a channel:

  1. the per-channel STRIPS (vault card, app_builder, exec family) —
     `strip_tools_for_channel`, which converge correctly replaces with an
     allowed_tools restriction; and
  2. the SURFACE disable set — `disabled_tools_for_channel`, which for
     voice is VOICE_DISABLED_TOOLS (create_job/update_job/spawn). That
     one is applied in run() long before the array is built, and
     `tool_defs` filters it out of the wire array itself.

So converge was applied to an array that had already diverged. Measured
on the real definitions, 2026-08-05:

    web   array: 52 defs, 9,116 tok
    voice array: 49 defs, 8,233 tok      -> 883 tok, different bytes

which is a separate cache lineage, i.e. a voice<->web hop inside one day
re-bills the whole system+history tail at full price.

Run:
    cd backend && RUN_MODE=agent PYTHONPATH=. pytest tests/test_channel_converge_voice_array.py
"""
from __future__ import annotations

import json

import pytest

from app.agent.agent_runner import AgentRunner, _RUN_DISABLED_TOOLS_CTX
from app.agent.prompt_profile import VOICE_DISABLED_TOOLS


def _runner() -> AgentRunner:
    """A bare runner — only the tool-composition surface is exercised."""
    r = AgentRunner.__new__(AgentRunner)
    r._core_tool_defs = [
        {"name": n, "description": f"d {n}", "input_schema": {"type": "object"}}
        for n in ("spawn", "create_job", "update_job", "web_search",
                  "memory_search", "save_streaming_credential")
    ]
    r.skill_loader = None
    r.tools = None
    r._disabled_tool_names = frozenset()
    return r


def _names(defs):
    return [d.get("name") or d.get("function", {}).get("name") for d in defs]


@pytest.fixture(autouse=True)
def _clean_ctx():
    tok = _RUN_DISABLED_TOOLS_CTX.set(None)
    yield
    _RUN_DISABLED_TOOLS_CTX.reset(tok)


def test_tool_defs_filters_the_surface_disabled_set_out_of_the_wire_array():
    """The premise. If this ever stops being true the bug cannot recur, but
    the fix below would also be pointless — so pin it."""
    r = _runner()
    _RUN_DISABLED_TOOLS_CTX.set(frozenset(VOICE_DISABLED_TOOLS))
    got = _names(r.tool_defs)
    for banned in VOICE_DISABLED_TOOLS:
        assert banned not in got, (
            f"{banned} is still in the wire array — premise changed"
        )


def test_tool_defs_ignoring_restores_exactly_the_exempted_names():
    """The fix's mechanism: same array, minus nothing but the exemption."""
    r = _runner()
    _RUN_DISABLED_TOOLS_CTX.set(frozenset(VOICE_DISABLED_TOOLS))

    converged = _names(r.tool_defs_ignoring(VOICE_DISABLED_TOOLS))
    for name in VOICE_DISABLED_TOOLS:
        assert name in converged, f"{name} was not restored to the array"

    # And it must be exactly the full set — not a superset that also
    # un-disables something the USER turned off.
    _RUN_DISABLED_TOOLS_CTX.set(frozenset(VOICE_DISABLED_TOOLS) | {"web_search"})
    partial = _names(r.tool_defs_ignoring(VOICE_DISABLED_TOOLS))
    assert "web_search" not in partial, (
        "tool_defs_ignoring un-disabled a tool outside the exemption"
    )


def test_the_context_var_is_restored_afterwards():
    """It swaps a ContextVar. A leak would silently un-disable tools for the
    rest of the turn, including at execute time."""
    r = _runner()
    before = frozenset(VOICE_DISABLED_TOOLS)
    _RUN_DISABLED_TOOLS_CTX.set(before)
    r.tool_defs_ignoring(VOICE_DISABLED_TOOLS)
    assert _RUN_DISABLED_TOOLS_CTX.get() == before, (
        "the disabled set leaked — voice would keep these tools callable"
    )


def test_voice_and_web_wire_arrays_are_byte_identical_under_converge():
    """The actual contract: same bytes => same cache lineage.

    MUTATION: change the converge branch back to `list(all_tools)` and this
    goes red, because voice's array is 3 defs short.
    """
    r = _runner()

    _RUN_DISABLED_TOOLS_CTX.set(frozenset())            # a web turn
    web = json.dumps(r.tool_defs, separators=(",", ":"))

    _RUN_DISABLED_TOOLS_CTX.set(frozenset(VOICE_DISABLED_TOOLS))   # a voice turn
    voice = json.dumps(
        r.tool_defs_ignoring(VOICE_DISABLED_TOOLS), separators=(",", ":")
    )

    assert voice == web, (
        "voice and web still send different tool arrays under channel_converge, "
        "so they remain separate provider cache lineages and every hop between "
        "them re-bills the whole system+history prefix"
    )


def test_the_arrays_really_did_differ_before_the_fix():
    """Anti-vacuity control.

    If the fixture or the runner stub produced empty arrays, the equality
    above would pass while proving nothing. Show the UNFIXED comparison is
    genuinely unequal on this same data.
    """
    r = _runner()
    _RUN_DISABLED_TOOLS_CTX.set(frozenset())
    web = json.dumps(r.tool_defs, separators=(",", ":"))
    _RUN_DISABLED_TOOLS_CTX.set(frozenset(VOICE_DISABLED_TOOLS))
    voice_unfixed = json.dumps(r.tool_defs, separators=(",", ":"))

    assert len(json.loads(web)) > 3, "stub array too small to prove anything"
    assert voice_unfixed != web, (
        "the pre-fix arrays are equal on this fixture, so the test above "
        "cannot be detecting the fix"
    )
