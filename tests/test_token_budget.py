"""Ticket 6 — token-aware result budgeting.

Web tool output was truncated TWICE and char/byte-based: the reader cut at
max_chars and the executor byte-capped again. Behind web_token_budget_enabled
(kill-switch, default on), web_search/web_fetch now truncate ONCE to a token
budget in execute() (the byte cap is skipped for them, and web_fetch gives the
reader a generous safety cap so it doesn't cut first), and the parallel web
batch in agent_runner is bounded by an aggregate per-turn token budget so a
multi-fetch turn can't flood the context.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import app.agent.agent_runner as AR
import app.agent.tool_executor as TE
from app.agent.agent_runner import AgentRunner
from app.agent.smart_fetch._budget import estimate_tokens, truncate_to_tokens
from app.agent.tool_executor import ToolExecutor

_TE_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "tool_executor.py").read_text()


# ── budget util ─────────────────────────────────────────────────────────

def test_estimate_and_truncate():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 40) == 10
    assert truncate_to_tokens("short text", 100) == "short text"   # no-op
    out = truncate_to_tokens("word " * 1000, 10)
    assert out.endswith("budget)")
    assert len(out) < 120, "~40 chars of content (10 tokens) + the fixed marker"
    assert truncate_to_tokens("anything", 0) == ""


# ── execute(): single token truncation for web tools ────────────────────

class _ExecStub:
    user_disabled_tools = frozenset()

    async def _tool_web_fetch(self, inp):
        return "word " * 6000   # ~30000 chars

    async def _tool_web_search(self, inp):
        return "word " * 6000

    async def _tool_other(self, inp):
        return "x" * 40000


def test_execute_token_budgets_web_fetch_once(monkeypatch):
    monkeypatch.setattr(TE.settings, "web_token_budget_enabled", True)
    monkeypatch.setattr(TE.settings, "fetch_token_budget", 1000)
    out = asyncio.run(ToolExecutor.execute(_ExecStub(), "web_fetch", {}))
    assert out.endswith("budget)"), "should use the token-budget marker"
    assert "[truncated," not in out, "must NOT also apply the byte cap (no double truncation)"
    assert len(out) < 4500, "~1000 tokens => ~4000 chars"


def test_execute_byte_cap_when_flag_off(monkeypatch):
    monkeypatch.setattr(TE.settings, "web_token_budget_enabled", False)
    out = asyncio.run(ToolExecutor.execute(_ExecStub(), "web_fetch", {}))
    assert "[truncated," in out, "flag off => legacy byte cap"


def test_execute_non_web_tool_keeps_byte_cap(monkeypatch):
    monkeypatch.setattr(TE.settings, "web_token_budget_enabled", True)
    out = asyncio.run(ToolExecutor.execute(_ExecStub(), "other", {}))
    assert "[truncated," in out, "non-web tools keep the byte cap even with budgeting on"


# ── web_fetch handler gives reader a generous cap when budgeting on ─────

def test_web_fetch_reader_cap_is_gated(monkeypatch):
    seen = {}

    async def fake_read(url, max_chars):
        seen["mc"] = max_chars
        return "BODY"

    monkeypatch.setattr("app.agent.smart_fetch.reader.toup_read_page", fake_read)
    monkeypatch.setattr(TE.settings, "fetch_token_budget", 1000)

    monkeypatch.setattr(TE.settings, "web_token_budget_enabled", True)
    asyncio.run(ToolExecutor._tool_web_fetch(_ExecStub(), {"url": "http://x", "max_chars": 5000}))
    assert seen["mc"] == 6000, "budgeting on => generous reader cap (fetch_token_budget*6)"

    monkeypatch.setattr(TE.settings, "web_token_budget_enabled", False)
    asyncio.run(ToolExecutor._tool_web_fetch(_ExecStub(), {"url": "http://x", "max_chars": 5000}))
    assert seen["mc"] == 5000, "flag off => legacy per-request max_chars"


# ── agent_runner: per-turn aggregate cap ────────────────────────────────

def _parallel_stub(payload):
    async def fake_exec(name, inp):
        return payload
    return SimpleNamespace(tools=SimpleNamespace(execute=fake_exec))


def _tcs(n):
    return [{"id": f"id{i}", "name": "web_fetch", "input": {}} for i in range(n)]


def test_turn_budget_caps_parallel_batch(monkeypatch):
    monkeypatch.setattr(AR.settings, "web_token_budget_enabled", True)
    monkeypatch.setattr(AR.settings, "web_turn_token_budget", 3000)
    stub = _parallel_stub("word " * 5000)  # ~6250 tokens each, 3 -> ~18750 total
    res = asyncio.run(AgentRunner._execute_tools_parallel(stub, _tcs(3), 6))
    total = sum(estimate_tokens(g["result"]) for g in res.values())
    assert total <= 3000 + 60, f"aggregate must be bounded by the turn budget; got {total}"
    for g in res.values():
        assert g["result"].endswith("budget)")


def test_turn_budget_leaves_small_batches_untouched(monkeypatch):
    monkeypatch.setattr(AR.settings, "web_token_budget_enabled", True)
    monkeypatch.setattr(AR.settings, "web_turn_token_budget", 100000)
    stub = _parallel_stub("small result")
    res = asyncio.run(AgentRunner._execute_tools_parallel(stub, _tcs(2), 6))
    for g in res.values():
        assert g["result"] == "small result", "under-budget results are not trimmed"


def test_turn_budget_off_no_trim(monkeypatch):
    monkeypatch.setattr(AR.settings, "web_token_budget_enabled", False)
    stub = _parallel_stub("word " * 5000)
    res = asyncio.run(AgentRunner._execute_tools_parallel(stub, _tcs(3), 6))
    for g in res.values():
        assert not g["result"].endswith("budget)"), "flag off => no aggregate trim"


# ── structural ──────────────────────────────────────────────────────────

def test_execute_has_single_token_branch():
    assert "_TOKEN_BUDGETED_TOOLS" in _TE_SRC
    assert "truncate_to_tokens(result, budget)" in _TE_SRC


def test_budget_flags_default_on_killswitch():
    from app.config import Settings
    assert Settings.model_fields["web_token_budget_enabled"].default is True
