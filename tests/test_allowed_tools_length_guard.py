"""Never emit an allow-list the provider will refuse.

The chain, measured on the founder's tenant 2026-09-20 (Loki 18:49:17–18:49:27):

  `[PERF] query_intent: category=full, tools=all`
  `[PERF] tool_filter: 172 total → 171 for intent=full`   tools_sent=175
  proxy: `174 > 128, dropped 4 (namespace-fair)` — only the four names the
         170-entry allow-list did not protect could go
  OpenAI: `400 … param: tool_choice.type, "Invalid value: 'allowed_tools'"`
  `[AGENT] provider rejected the allowed_tools tool_choice … retrying the
         same call unrestricted`
  proxy, every round after: `dropped 46 (namespace-fair)` — including
         `play_media`, `browser`, `recall_day` and every `generate_*`

So the WIDEST intent produced the NARROWEST effective toolset: dropping the
restriction on retry also removed the protection the cap was reading, and the
unprotected cap took 46 capabilities. One wasted round trip, and the only
trace was a drop list inside a proxy warning.

The guard lives in `build_allowed_tools_choice` — the single place the
restriction is built — rather than at a call site, because a check at one
caller is invisible to the other and to the next one. Dropping the restriction
costs nothing in policy: tool bans are enforced at EXECUTE time, which is the
same argument the runner's own rejection branch already makes when it retries
unrestricted. This just makes it happen before the wire instead of after a 400.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.agent.prefix_stability import (
    ALLOWED_TOOLS_MAX,
    build_allowed_tools_choice,
)


def _names(n: int) -> list:
    return [f"tool_{i:03d}" for i in range(n)]


def test_the_limit_matches_the_proxys_tools_cap():
    """Same number for the same reason: the provider will not take an
    allow-list longer than the array it accepts in the first place."""
    from app.api.llm_proxy import _OPENAI_MAX_TOOLS
    assert ALLOWED_TOOLS_MAX == _OPENAI_MAX_TOOLS == 128


def test_a_restriction_at_exactly_the_limit_is_still_sent():
    """The boundary is where this kind of guard goes wrong. 128 is valid;
    refusing it would silently un-gate every tenant sitting exactly on it."""
    choice = build_allowed_tools_choice(_names(ALLOWED_TOOLS_MAX))
    assert choice is not None
    assert choice["type"] == "allowed_tools"
    assert len(choice["allowed_tools"]["tools"]) == ALLOWED_TOOLS_MAX


def test_one_over_the_limit_drops_the_restriction_entirely():
    assert build_allowed_tools_choice(_names(ALLOWED_TOOLS_MAX + 1)) is None


def test_the_production_case_drops_it():
    """171 names on a `full`-intent voice turn is what the recording shows."""
    assert build_allowed_tools_choice(_names(171)) is None


def test_the_ordinary_narrow_case_is_untouched():
    choice = build_allowed_tools_choice(["web_search", "exec"])
    assert choice == {
        "type": "allowed_tools",
        "allowed_tools": {"mode": "auto", "tools": [
            {"type": "function", "function": {"name": "exec"}},
            {"type": "function", "function": {"name": "web_search"}},
        ]},
    }


def test_an_empty_list_is_not_turned_into_a_dropped_restriction():
    """Empty is a different question from too-long, and the caller already
    guards it (`and _allowed_tool_names`). Keep the shapes distinguishable."""
    choice = build_allowed_tools_choice([])
    assert choice is not None
    assert choice["allowed_tools"]["tools"] == []


def test_the_mode_still_rides_through_under_the_limit():
    choice = build_allowed_tools_choice(["exec"], mode="required")
    assert choice["allowed_tools"]["mode"] == "required"


def test_dropping_it_says_so_once_with_the_count(caplog):
    """A restriction that vanishes silently is a capability change nobody can
    attribute. The line carries the count, because a set this large means the
    intent gate stopped narrowing anything — a defect upstream of here."""
    with caplog.at_level(logging.WARNING, logger="app.agent.prefix_stability"):
        build_allowed_tools_choice(_names(170))
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    said = warnings[0].getMessage()
    assert "170" in said and str(ALLOWED_TOOLS_MAX) in said


def test_an_over_long_required_list_keeps_the_FORCING_it_only_drops_the_list():
    """Dropping the restriction may not also drop the forcing.

    `agent_runner` sets `_tool_choice = "required"` for vibecoding's first
    iteration and then assigns this function's return value straight over it.
    A plain None there degrades forced tool use to `auto` on exactly the turns
    whose allow-list is widest — a silent capability change on a channel this
    round is not otherwise touching. The bare string is the value that call
    site used before the allow-list shape existed, and the rejection-retry
    ladder is keyed on `isinstance(_tool_choice, dict)`, so a string cannot
    re-enter it."""
    assert build_allowed_tools_choice(_names(171), mode="required") == "required"
    # …and `auto` is still a full drop: there is nothing to preserve.
    assert build_allowed_tools_choice(_names(171), mode="auto") is None


def test_the_runner_still_sets_required_before_it_asks_for_the_choice():
    """The source half. The preserved "required" is only worth anything while
    the call site still derives `mode=` from a `_tool_choice` it set first —
    if that line moved, this helper would be preserving a mode nobody asks
    for, and the vibecoding path would be broken somewhere else entirely."""
    runner = (Path(__file__).resolve().parents[1]
              / "app" / "agent" / "agent_runner.py").read_text()
    assign = runner.index('_tool_choice: Any = "required" if (channel == "vibecoding"')
    build = runner.index("_tool_choice = build_allowed_tools_choice(")
    assert assign < build
    assert 'mode="required" if _tool_choice == "required" else "auto"' in runner


def test_the_caller_treats_none_as_no_restriction():
    """Two source properties no unit test of the helper can see.

    1. The runner assigns the result straight to `_tool_choice`, whose
       no-restriction value is already None — so returning None is the
       existing "send nothing" path and needs no branch of its own.
    2. The tool policy is NOT weakened by dropping the restriction: the
       executor's disabled set is computed elsewhere and is untouched here.
       If a future change ever made `tool_choice` the only enforcement, this
       guard would become a security hole rather than a latency fix, and the
       probe below is what turns that into a red test.
    """
    runner = (Path(__file__).resolve().parents[1]
              / "app" / "agent" / "agent_runner.py").read_text()

    assert "_tool_choice = build_allowed_tools_choice(" in runner
    # The default value the assignment can safely land back on.
    assert '_tool_choice: Any = None' in runner
    # Execute-time enforcement is a separate mechanism and still present.
    assert "_RUN_DISABLED_TOOLS_CTX" in runner


def test_the_guard_is_in_the_builder_not_at_a_call_site():
    """Where it lives IS the property. A length check written at the runner
    would not cover the next caller, and this function is the only place the
    restriction is constructed."""
    src = (Path(__file__).resolve().parents[1]
           / "app" / "agent" / "prefix_stability.py").read_text()
    body = src[src.index("def build_allowed_tools_choice("):]
    body = body[:body.index("\ndef ")]
    assert "ALLOWED_TOOLS_MAX" in body
    assert "return None" in body
    # …and the check must precede the construction, or it guards nothing.
    # Compared against the CODE, not the prose: the docstring quotes the wire
    # shape, and a probe that matched there would pass with the guard deleted.
    code = body[body.index('"""', body.index('"""') + 3) + 3:]
    assert code.index("if len(allowed_names) > ALLOWED_TOOLS_MAX:") \
        < code.index("return None") < code.index("return {")
