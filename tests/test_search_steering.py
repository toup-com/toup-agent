"""Ticket 7 — steer the model to batch + reason snippet-first.

The web_search/web_fetch tool descriptions gave no guidance to reason over
snippets first, batch multiple searches in one turn, or fetch pages in parallel.
They now do, and web_search's default count is 8 (snippets are cheap + reranked
+ token-budgeted). The behavioral round-trip win is measured in Ticket 8; here
we lock the steering language, the schema stability, and the default alignment.
"""
from __future__ import annotations

import app.agent.tool_definitions as td
import app.agent.tool_executor as TE


def _tool(name):
    for t in td.get_agent_tools():
        if t.get("name") == name:
            return t
    raise AssertionError(f"{name} not found in get_agent_tools()")


def test_web_search_steers_snippet_first_and_batching():
    desc = _tool("web_search")["description"].lower()
    assert "snippet" in desc, "must steer snippet-first reasoning"
    assert "same turn" in desc and "multiple" in desc, "must steer multiple searches in one turn"
    assert "concurrent" in desc, "must mention concurrent execution"


def test_web_fetch_steers_on_demand_parallel_batching():
    desc = _tool("web_fetch")["description"].lower()
    assert "on-demand" in desc or "on demand" in desc, "must steer fetch-on-demand"
    assert "same turn" in desc, "must steer fetching multiple pages in one turn"
    assert "concurrent" in desc, "must mention concurrent fetches"
    assert "token" in desc, "must mention the automatic token budget"


def test_schemas_unchanged_backward_compatible():
    ws = _tool("web_search")["input_schema"]["properties"]
    assert set(ws.keys()) == {"query", "count"}
    assert _tool("web_search")["input_schema"]["required"] == ["query"]
    wf = _tool("web_fetch")["input_schema"]["properties"]
    assert set(wf.keys()) == {"url", "max_chars"}
    assert _tool("web_fetch")["input_schema"]["required"] == ["url"]


def test_default_count_is_8_and_consistent():
    # tool definition advertises 8
    assert "default 8" in _tool("web_search")["input_schema"]["properties"]["count"]["description"]
    # handler default matches
    src = (TE.__file__)
    text = open(src).read()
    assert 'inp.get("count", 8)' in text, "handler default must match the advertised default"
