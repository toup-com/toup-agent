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

from app.api.llm_proxy import _OPENAI_MAX_TOOLS, _cap_tools


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
