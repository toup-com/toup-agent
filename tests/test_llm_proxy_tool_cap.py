"""OpenAI's 128-tool cap is a cliff, and we walked off it.

On 2026-08-08 a tenant connected GitHub — a 5-tool connector — and went
from 124 tools to 129. OpenAI rejects the whole request with
`array_above_max_length`, so *every* chat turn 400'd regardless of what
was typed, and `ws_chat._friendly_error` rendered that 400 as "There was
an issue with the request. Please try rephrasing your message."

Nothing about that message points at a tool count, and nothing in the
stack capped or even counted the array. The user's agent was simply dead
until they switched providers.

These cover the helper. The wiring (OpenAI-only, after routing) is
asserted by reading the source, because the endpoints need a live
provider config to exercise end-to-end.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.api.llm_proxy import (
    _OPENAI_MAX_TOOLS, _cap_tools, _prune_tool_choice,
)


def _tools(n: int) -> list:
    return [{"name": f"tool_{i}", "description": "x"} for i in range(n)]


def test_the_cap_matches_openais_documented_limit():
    assert _OPENAI_MAX_TOOLS == 128


def test_an_array_at_the_limit_is_untouched():
    """128 is fine. Only 129 breaks — capping at the boundary would
    silently remove a tool from every tenant sitting exactly on it."""
    at_limit = _tools(_OPENAI_MAX_TOOLS)
    kept, dropped = _cap_tools(at_limit)
    assert dropped == []
    assert kept is at_limit


def test_a_short_array_is_untouched():
    short = _tools(12)
    kept, dropped = _cap_tools(short)
    assert dropped == []
    assert kept is short


def test_one_over_drops_exactly_one_from_the_tail():
    """The real incident: 129."""
    kept, dropped = _cap_tools(_tools(129))
    assert len(kept) == 128
    assert dropped == ["tool_128"]
    # Core tools come first in the agent's assembly order, so the head
    # must survive — the agent is broken without memory/files/messaging.
    assert kept[0]["name"] == "tool_0"


def test_a_large_overflow_reports_every_dropped_name():
    kept, dropped = _cap_tools(_tools(140))
    assert len(kept) == 128
    assert len(dropped) == 12
    assert dropped[0] == "tool_128" and dropped[-1] == "tool_139"


def test_dropped_names_survive_the_responses_shape():
    """Responses-style tools nest the name under `function`. A dropped
    tool that logs as <unnamed> is a tool nobody can identify later."""
    tools = _tools(128) + [{"type": "function", "function": {"name": "github__search_code"}}]
    _, dropped = _cap_tools(tools)
    assert dropped == ["github__search_code"]


def test_an_unnamed_tool_still_reports_something():
    _, dropped = _cap_tools(_tools(128) + [{"type": "function"}])
    assert dropped == ["<unnamed>"]


def test_the_cap_is_wired_on_the_openai_path_only_and_after_routing():
    """Two properties no unit test of the helper can see.

    1. Anthropic has no such limit, so capping there drops tools the
       provider would have accepted.
    2. The limit belongs to the PROVIDER, not the requested model id —
       an alias can resolve across backends — so the check has to sit
       after `_route_chat`, not before it.
    """
    src = Path(__file__).resolve().parents[1] / "app" / "api" / "llm_proxy.py"
    text = src.read_text()

    # Both endpoints cap.
    assert text.count("_cap_tools(_tools)") == 2, "one endpoint is missing the cap"

    # proxy_chat gates on the resolved backend, not on the model string.
    assert 'if backend.name == "openai":' in text

    # …and does so after routing, not before.
    route_at = text.index("backend, api_key = _route_chat(model, config)")
    gate_at = text.index('if backend.name == "openai":')
    assert route_at < gate_at, "the cap must read the backend routing decided"

    # No accidental global cap: the helper is never called unguarded on a
    # path that could be Anthropic.
    for m in re.finditer(r"_cap_tools\(_tools\)", text):
        window = text[max(0, m.start() - 700):m.start()]
        assert "openai" in window.lower(), "a _cap_tools call with no OpenAI guard above it"


# ── The half this file never tested: the CHOICE, not just the array ──────
#
# `_cap_tools` was pinned in isolation, so nothing noticed that capping the
# array left `tool_choice` naming tools the same request no longer offered.
# OpenAI answers 400 "Tool choice 'X' not found in 'tools' parameter".


def _chat_allowed(*names):
    return {"type": "allowed_tools",
            "allowed_tools": {"mode": "auto",
                              "tools": [{"type": "function",
                                         "function": {"name": n}}
                                        for n in names]}}


def _responses_allowed(*names):
    return {"type": "allowed_tools", "mode": "auto",
            "tools": [{"type": "function", "name": n} for n in names]}


def test_the_production_failure_chat_shape():
    """The founder's voice turn: 141 tools, 13 capped from the tail, and
    `slack__list_channels` still named in the allowlist."""
    body = {"tool_choice": _chat_allowed(
        "memory_search", "slack__list_channels", "navigate_to")}
    pruned = _prune_tool_choice(body, ["slack__list_channels"])

    assert pruned == ["slack__list_channels"]
    left = [t["function"]["name"]
            for t in body["tool_choice"]["allowed_tools"]["tools"]]
    assert left == ["memory_search", "navigate_to"]
    assert "slack__list_channels" not in left


def test_the_responses_wire_shape_too():
    """The 400 was on /v1/responses, whose allowlist is FLAT — a fix that
    only understood the chat shape would not have helped the request that
    actually failed."""
    body = {"tool_choice": _responses_allowed("a", "jira__search_issues")}
    pruned = _prune_tool_choice(body, ["jira__search_issues"])

    assert pruned == ["jira__search_issues"]
    assert [t["name"] for t in body["tool_choice"]["tools"]] == ["a"]


def test_an_allowlist_emptied_by_the_cap_drops_the_restriction():
    """An allowlist whose every entry was capped away would forbid every
    tool the model was offered. Dropping the restriction is the honest
    degradation; keeping an empty one is a guaranteed dead turn."""
    body = {"tool_choice": _chat_allowed("slack__list_channels")}
    pruned = _prune_tool_choice(body, ["slack__list_channels"])

    assert pruned == ["slack__list_channels"]
    assert "tool_choice" not in body


def test_a_choice_naming_only_surviving_tools_is_untouched():
    """The common case must not be disturbed — no cap, no pruning, and the
    object is left byte-identical."""
    tc = _chat_allowed("memory_search", "navigate_to")
    body = {"tool_choice": tc}
    assert _prune_tool_choice(body, ["some__other_tool"]) == []
    assert body["tool_choice"] is tc
    assert [t["function"]["name"]
            for t in tc["allowed_tools"]["tools"]] == [
        "memory_search", "navigate_to"]
    # ...and no dropped names at all is a no-op.
    assert _prune_tool_choice({"tool_choice": tc}, []) == []


def test_a_forced_single_tool_is_reported_not_silently_rewritten():
    """A forced `{"type":"function"}` naming a capped tool has no valid
    repair — pruning it would invent a different request than the caller
    asked for. It is left alone and logged instead."""
    body = {"tool_choice": {"type": "function",
                            "function": {"name": "slack__list_channels"}}}
    assert _prune_tool_choice(body, ["slack__list_channels"]) == []
    assert body["tool_choice"]["function"]["name"] == "slack__list_channels"

    # "auto" / "required" / absent are all untouched.
    for tc in ("auto", "required", None):
        b = {"tool_choice": tc} if tc else {}
        assert _prune_tool_choice(b, ["x"]) == []


# ── The cut must never reach a tool the agent OWNS ──────────────────────


def test_the_cap_can_only_ever_reach_the_mcp_block():
    """Two rules in this repo point in opposite directions.

    Prefix stability says a new tool may only join at the **END** — that is
    why `automations__memory_recall` is last in its skill and why
    CONTRACTS §14 leans on the append-only property. **The cap drops from
    the END.** So the newest tool occupies simultaneously the cache-safest
    position and the first-to-vanish one, and nothing anywhere says which
    property wins when the array overflows.

    Today the collision is invisible: wire order is core → skill →
    MCP/connector (`agent_runner.py:1038`, `:1054`), and the MCP block
    absorbs the whole cut. That is luck, not design — the boundary is
    pinned nowhere, so the moment the MCP block shrinks or the skill block
    grows, the cut walks into skill tools **silently**, which is ND-24's
    actual defect rather than the Slack writes specifically.

    It also protects `tool_choice` INTEGRITY, not only capability
    presence — R30-C's point, about their own change. `_ALWAYS_INCLUDED_TOOLS`
    does not merely widen the array; it feeds the allowlist
    (`filter_tools_by_intent` → `_gated_names` → `_allowed_tool_names` →
    `build_allowed_tools_choice`), and `automations__list` was added to
    that set for ND-18. So an allowlisted tool cut by the cap is exactly
    the ND-22 path: `tool_choice` naming a tool absent from `tools`, three
    400s, a silent fallback to a weaker model, and a 20.7s turn returning
    two output tokens. Skill tools sit upstream of the MCP block, so
    bounding that block is what keeps an always-included entry from
    becoming that failure.

    This does not decide the precedence question — deliberately. It makes
    the day the question becomes urgent arrive as a red test instead of as
    a capability quietly missing from production.
    """
    from app.agent.tool_definitions import (
        get_agent_tools, get_extended_tools, get_navigation_tools,
    )
    from app.agent.skills.loader import SkillLoader
    from app.config import settings

    settings.automations_enabled = True   # widest skill set we ship
    core = get_agent_tools() + get_extended_tools() + get_navigation_tools()
    import asyncio
    loader = SkillLoader()
    asyncio.run(loader.load_all())
    skills = loader.get_all_tool_definitions()

    owned = len(core) + len(skills)
    assert owned <= _OPENAI_MAX_TOOLS, (
        f"core+skill tools ({owned}) exceed the OpenAI cap "
        f"({_OPENAI_MAX_TOOLS}), so `_cap_tools` would trim a tool the "
        f"agent OWNS rather than an MCP tool. Decide the precedence "
        f"question before shipping this: which yields, append-only "
        f"prefix stability or capability presence?"
    )
    # Headroom is the thing worth watching; name it in the failure.
    headroom = _OPENAI_MAX_TOOLS - owned
    assert headroom >= 1, headroom


def test_pruning_announces_itself_because_it_removes_the_last_signal(caplog):
    """The strip makes the request valid — and silences the only symptom.

    Before pruning existed the mismatch announced itself as a 400: three
    retries and a visible model downgrade. Making the request valid means
    the model quietly takes whatever survived the cut instead of failing,
    so the log is now the ONLY way a human learns capability was removed.

    Measured on the founder 2026-08-26: asked which Slack channels it
    could see with every `slack__*` tool capped away, the agent answered
    correctly via `automations__list_targets` — a skill tool upstream of
    the cut. Reads have a survivor; writes do not. The agent looks
    capable when asked to LOOK and is silently incapable when asked to
    ACT.
    """
    import logging

    body = {"tool_choice": _chat_allowed("memory_search",
                                         "slack__send_message")}
    with caplog.at_level(logging.ERROR, logger="app.api.llm_proxy"):
        pruned = _prune_tool_choice(
            body, ["slack__send_message"],
            model_name="gpt-5.6-terra", original_len=144)

    assert pruned == ["slack__send_message"]
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "capability was removed and nothing said so"
    said = errors[0].getMessage()
    assert "CAPABILITY REMOVED" in said
    assert "slack__send_message" in said
    # The numbers a human needs to act on it, not just the fact.
    assert "144" in said and str(_OPENAI_MAX_TOOLS) in said
