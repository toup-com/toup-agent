"""When the allow-list itself overflows, something has to give — and it must
not be the player.

Two defects, one function.

**R44** (measured 2026-09-15, founder tenant): 173 tools on the wire, of which
the voice channel's intent filter allow-listed 130. `_cap_tools` dropped all 43
unprotected tools, reached `n_kept=130 > 128`, found no droppable victim and
`break`'d — returning an array two OVER the cap. `_prune_tool_choice` then
pruned nothing, because no allow-listed name had been dropped. The request went
upstream with 130 tools AND a 130-entry allow-list; OpenAI answered `400
param=tool_choice.type "Invalid value: 'allowed_tools'"`. Three identical 400s,
then a silent fallback to gpt-4o: three minutes, two output tokens. The WARNING
said "dropped 4", which reads like a success.

**R48**: the R44 policy trims the protected names namespace-fair and tail-first,
with no idea which of them the product cannot work without — and on this very
tenant `play_media` was in the drop list on every unprotected round, so its
survival was positional luck. A floor of channel-essential names goes under the
allow-list: allow-listed names are trimmed first, the floor only if the floor
alone still does not fit.

Determinism is a named requirement rather than a nicety: the kept ORDER is what
the provider's prefix cache keys on, and A3-6 is the same defect one layer up
(iteration 0 protected the allow-list, iteration 1 protected nothing, so two
different 128-arrays went upstream inside one turn and the first two rounds
cached nothing at all).
"""
from __future__ import annotations

import logging

import pytest

from app.api.llm_proxy import (
    _OPENAI_MAX_TOOLS,
    PROTECTED_CORE_TOOLS,
    _cap_tools,
    _prune_tool_choice,
    _protected_core_tools,
)

LIMIT = _OPENAI_MAX_TOOLS


def _named(*names) -> list:
    return [{"name": n, "description": "x"} for n in names]


def _voice_like_array(n_core: int = 40, n_conn: int = 140) -> list:
    """A tenant shaped like the founder's: a core block that includes the
    floor, then a long connector tail across several namespaces.

    The floor names sit at the END of the core block on purpose. Core is the
    largest namespace, so rule 3 trims it first and rule "tail within it"
    reaches its last entries first — which is why `play_media`'s survival on
    any given turn was positional luck rather than policy. Putting the floor
    at the head of the array would make every test below pass with the floor
    deleted."""
    core = [f"core_{i}" for i in range(n_core)] + sorted(PROTECTED_CORE_TOOLS)
    conn = [f"ns{i % 7}__t{i}" for i in range(n_conn)]
    return _named(*core, *conn)


# ── R44: an over-long allow-list is trimmed rather than shipped ──────────

def test_an_allowlist_alone_over_the_cap_is_trimmed_until_the_array_fits():
    tools = _named(*[f"t{i}" for i in range(174)])
    protected = {f"t{i}" for i in range(170)}
    kept, dropped = _cap_tools(tools, protected=protected, floor=frozenset())

    assert len(kept) == LIMIT, (
        "the array went upstream over the cap — a guaranteed 400 for the "
        "whole turn"
    )
    # The trimmed protected names must be REPORTED, or `_prune_tool_choice`
    # leaves them in the allow-list and the request is inconsistent instead.
    assert len(dropped) == 174 - LIMIT
    assert any(n in protected for n in dropped)


def test_the_trimmed_names_are_then_removed_from_the_allowlist():
    """The two halves only work together: trimming the array without pruning
    the choice swaps one 400 for another ("Tool choice 'X' not found")."""
    tools = _named(*[f"t{i}" for i in range(174)])
    protected = [f"t{i}" for i in range(170)]
    kept, dropped = _cap_tools(tools, protected=set(protected), floor=frozenset())

    body = {"tool_choice": {
        "type": "allowed_tools",
        "allowed_tools": {"mode": "auto", "tools": [
            {"type": "function", "function": {"name": n}} for n in protected]},
    }}
    pruned = _prune_tool_choice(body, dropped)

    left = {t["function"]["name"]
            for t in body["tool_choice"]["allowed_tools"]["tools"]}
    kept_names = {t["name"] for t in kept}
    assert pruned, "nothing was pruned although protected names were dropped"
    assert left <= kept_names, (
        f"allow-list still names tools the array no longer offers: "
        f"{sorted(left - kept_names)}"
    )


def test_it_says_so_at_error_level_with_the_number_to_act_on(caplog):
    """Trimming makes the request valid — and silences the only symptom.
    Before this, the mismatch announced itself as three 400s and a visible
    model downgrade; now the model quietly takes whatever survived, so the log
    is the only way a human learns the allow-list is being built too wide."""
    tools = _named(*[f"t{i}" for i in range(174)])
    with caplog.at_level(logging.ERROR, logger="app.api.llm_proxy"):
        _cap_tools(tools, protected={f"t{i}" for i in range(170)},
                   floor=frozenset())

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "an allow-list larger than the cap was trimmed silently"
    said = errors[0].getMessage()
    assert "170" in said and str(LIMIT) in said


# ── R48: the floor is last ───────────────────────────────────────────────

def test_the_floor_survives_an_overflowing_allowlist():
    """The specific casualty this exists to prevent. `play_media` is in the
    core block, and on every unprotected round of the founder's voice turns it
    was in the drop list."""
    tools = _voice_like_array()
    assert len(tools) > LIMIT
    # Everything is allow-listed, which is the `full`-intent voice case.
    protected = {t["name"] for t in tools}
    kept, dropped = _cap_tools(tools, protected=protected)

    kept_names = {t["name"] for t in kept}
    assert len(kept) == LIMIT
    for name in PROTECTED_CORE_TOOLS:
        assert name in kept_names, f"{name} was trimmed off the floor"
    assert "play_media" in kept_names


def test_without_the_floor_the_player_dies_which_is_the_point():
    """The mutation, encoded: run the identical array through the identical
    policy with an EMPTY floor and `play_media` is trimmed. So this test
    fails the moment the floor stops being consulted, rather than passing
    because some other property happened to keep the tool alive."""
    tools = _voice_like_array()
    protected = {t["name"] for t in tools}
    _, dropped = _cap_tools(tools, protected=set(protected), floor=frozenset())
    assert "play_media" in dropped, (
        "the floor is not what is keeping play_media on the wire — this test "
        "proves nothing as written"
    )


def test_the_floor_is_protected_even_when_the_request_names_nothing():
    """`protected` is whatever the REQUEST named — on iterations ≥1 the agent
    sends no tool_choice at all, so that set is EMPTY and the cap trims purely
    by position. That is the round on which `play_media` died."""
    tools = _voice_like_array()
    kept, _ = _cap_tools(tools, protected=set())
    kept_names = {t["name"] for t in kept}
    assert "play_media" in kept_names
    assert PROTECTED_CORE_TOOLS <= kept_names


def test_allowlisted_names_are_trimmed_before_the_floor():
    """Ordering, not merely membership: with both over the cap, the cut has to
    land on the allow-list first."""
    floor = frozenset({"play_media", "web_search"})
    tools = _named("play_media", "web_search",
                   *[f"t{i}" for i in range(LIMIT)])   # 130 total
    protected = {f"t{i}" for i in range(LIMIT)} | set(floor)
    kept, dropped = _cap_tools(tools, protected=protected, floor=floor)

    assert len(kept) == LIMIT
    assert "play_media" not in dropped and "web_search" not in dropped
    assert all(n.startswith("t") for n in dropped), dropped


def test_the_floor_yields_only_when_the_floor_alone_does_not_fit():
    """An absurd floor must still produce a VALID request — a 400 for
    everyone is worse than a degraded turn, which is the same trade rule 2
    already makes."""
    floor = frozenset(f"f{i}" for i in range(LIMIT + 5))
    tools = _named(*sorted(floor))
    kept, dropped = _cap_tools(tools, protected=set(), floor=floor)
    assert len(kept) == LIMIT
    assert len(dropped) == 5


def test_the_floor_never_evicts_a_tool_the_request_is_allowed_to_call():
    """The floor may only ever be a GAIN, and on a restricted request holding a
    name the allow-list omits is a pure loss.

    Under `allowed_tools` a tool that is in the array but not in the allow-list
    cannot be called at all, so protecting it costs an allow-listed tool its
    place and buys nothing. `_prune_tool_choice` then reports the casualty as
    CAPABILITY REMOVED at ERROR — on a request that, before the floor existed,
    was valid and fully honoured. This function runs on platform-api for every
    user and every channel the moment the platform deploys, and the reachable
    case is ordinary: every non-`full` intent omits one to three floor names,
    so a `code`-intent turn on a connector-heavy tenant forces a trim on every
    single turn.
    """
    floor = frozenset({"play_media", "web_search", "web_fetch",
                       "recall_day", "memory_search"})
    # 175 on the wire — the founder's tenant shape. The floor sits in the
    # array but OUTSIDE the allow-list, which is what a non-`full` intent
    # produces.
    allowed = [f"t{i}" for i in range(126)]
    tools = _named(*allowed, *sorted(floor),
                   *[f"ns{i % 5}__x{i}" for i in range(175 - 126 - len(floor))])
    assert len(tools) == 175

    kept, dropped = _cap_tools(tools, protected=set(allowed), floor=floor)
    assert len(kept) == LIMIT
    lost = [n for n in dropped if n in set(allowed)]
    assert lost == [], (
        f"the floor evicted {len(lost)} allow-listed tool(s) in favour of "
        f"names this request may not call: {lost}"
    )

    # …and the control: with no floor at all the same request loses nothing
    # either, which is what makes the loss above attributable to the floor
    # rather than to the array simply being too long.
    _, dropped_nofloor = _cap_tools(tools, protected=set(allowed),
                                    floor=frozenset())
    assert [n for n in dropped_nofloor if n in set(allowed)] == []


def test_the_error_line_attributes_each_count_to_the_right_cause(caplog):
    """The old line read "tool_choice's allow-list ALONE exceeds the cap (128
    named > 128)" — 128 is not greater than 128, and it blamed whatever built
    the allow-list for a trim the floor had caused. A log line that states a
    false comparison is worse than no line: it sends the reader upstream.

    The assertions are POSITIONAL, and the reason is a mutation that survived a
    review round: with `'130' in said[0] and '128' in said[0]`, swapping
    `n_named` and `n_floored` in the logger call left every test green while
    the line said "2 named by tool_choice, 130 on the protected core floor" —
    a NEW falsehood of exactly the class this fix was written against. A count
    that cannot be told from its neighbour is not reported, it is decoration.
    The floor-outside-the-allow-list case below is the other half: it is the
    only one where the two counts differ from each other AND from the total.
    """
    floor = frozenset({"play_media", "web_search"})
    tools = _named("play_media", "web_search",
                   *[f"t{i}" for i in range(LIMIT)])
    protected = {f"t{i}" for i in range(LIMIT)} | set(floor)
    with caplog.at_level(logging.ERROR, logger="app.api.llm_proxy"):
        _cap_tools(tools, protected=protected, floor=floor)

    said = [r.getMessage() for r in caplog.records
            if r.levelno >= logging.ERROR]
    assert said, "an over-cap protected set was trimmed silently"
    assert "ALONE exceeds the cap" not in said[0]
    # 130 untrimmable, all 130 named by tool_choice, 2 of them also on the
    # floor, against a limit of 128.
    assert ("130 untrimmable (130 named by tool_choice, 2 on the protected "
            "core floor)") in said[0], said[0]
    assert "vs limit 128" in said[0], said[0]


def test_the_error_line_reports_a_zero_floor_when_the_floor_is_not_the_cause(caplog):
    """The mirror, and the case that makes a swapped or wrong variable die:
    here the floor name is in the array but OUTSIDE the allow-list, so the
    intersection is empty and `n_floored` is 0 while `n_named` is the total.
    Any of the three counts printed in another's place changes this line."""
    floor = frozenset({"play_media"})
    allowed = [f"t{i}" for i in range(LIMIT + 2)]
    tools = _named("play_media", *allowed)
    with caplog.at_level(logging.ERROR, logger="app.api.llm_proxy"):
        _cap_tools(tools, protected=set(allowed), floor=floor)

    said = [r.getMessage() for r in caplog.records
            if r.levelno >= logging.ERROR]
    assert said, "an over-cap protected set was trimmed silently"
    assert ("130 untrimmable (130 named by tool_choice, 0 on the protected "
            "core floor)") in said[0], said[0]


def test_a_floor_name_absent_from_the_array_costs_nothing():
    tools = _named(*[f"t{i}" for i in range(140)])
    kept, _ = _cap_tools(tools, protected=set(),
                         floor=frozenset({"not_a_real_tool"}))
    assert len(kept) == LIMIT


# ── Determinism, because the prefix cache keys on the kept order ─────────

def test_the_same_input_caps_to_a_byte_identical_array():
    tools = _voice_like_array()
    protected = {t["name"] for t in tools}
    a_kept, a_dropped = _cap_tools(tools, protected=set(protected))
    b_kept, b_dropped = _cap_tools(tools, protected=set(protected))
    assert [t["name"] for t in a_kept] == [t["name"] for t in b_kept]
    assert a_dropped == b_dropped


def test_the_kept_array_preserves_wire_order():
    """The cap may remove entries; it may never REORDER them. Tools serialize
    ahead of system and history, so a reorder invalidates the whole prefix."""
    tools = _voice_like_array()
    kept, _ = _cap_tools(tools, protected=set())
    wire = [t["name"] for t in tools]
    assert [t["name"] for t in kept] == [n for n in wire
                                         if n in {k["name"] for k in kept}]


# ── The env override ─────────────────────────────────────────────────────

def test_the_floor_can_be_widened_or_emptied_without_a_code_change(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_PROTECTED_CORE_TOOLS", "a_tool, b_tool ,")
    assert _protected_core_tools() == frozenset({"a_tool", "b_tool"})
    monkeypatch.setenv("LLM_PROXY_PROTECTED_CORE_TOOLS", "")
    assert _protected_core_tools() == frozenset()
    monkeypatch.delenv("LLM_PROXY_PROTECTED_CORE_TOOLS")
    assert "play_media" in _protected_core_tools()


def test_the_floor_is_frozen_at_import():
    """It has to be a constant for the process, or the kept order stops being
    a pure function of the wire array and the cache re-forks on whatever moved
    the environment."""
    assert isinstance(PROTECTED_CORE_TOOLS, frozenset)


# ── The old behaviour is otherwise untouched ─────────────────────────────

def test_an_array_under_the_cap_is_still_returned_unchanged():
    short = _named(*[f"t{i}" for i in range(12)])
    kept, dropped = _cap_tools(short)
    assert kept is short and dropped == []


def test_a_namespace_is_still_never_emptied():
    tools = _named(*[f"big__t{i}" for i in range(130)], "solo__only")
    kept, dropped = _cap_tools(tools, protected=set(), floor=frozenset())
    assert "solo__only" in {t["name"] for t in kept}
    assert len(kept) == LIMIT
