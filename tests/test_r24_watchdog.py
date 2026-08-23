"""Round 24 — the skill branch has a wall clock, and the step cap speaks plainly.

Two recorded defects pinned here:

* **The skill dispatch branch was the ONLY one not wrapped in
  asyncio.wait_for** — the ``_tool_*`` branch and the MCP branch both are —
  so ``app_html__present_app`` (internal budgets summing to ~220s worst case)
  had no wall-clock ceiling at all: the 2026-08-23 01:26 build spun
  "Checking the app" unbounded while the user watched. The branch now sits
  inside ``wait_for(..., SKILL_TOOL_TIMEOUT_S)`` and a timeout produces the
  same actionable error shape the ``_tool_*`` branch produces, so the model
  can retry and land on the same step row via retry-in-place.

* **The 40-iteration cap emitted internals-flavoured resignation copy**
  ("I've reached the maximum number of tool iterations...") straight into
  user chat. The fallback now says where things stand in plain words, with
  no iteration/tool/maximum vocabulary anywhere in it.
"""
from __future__ import annotations

import asyncio
import inspect
import pathlib

from unittest.mock import MagicMock

from app.agent import tool_executor as te_mod
from app.agent.tool_executor import ToolExecutor

APP_DIR = pathlib.Path(te_mod.__file__).resolve().parents[1]


# ─── Source probe: the skill branch sits inside the watchdog ────────────


def test_skill_branch_execute_is_inside_wait_for():
    src = inspect.getsource(ToolExecutor.execute)
    start = src.index("is_skill_tool(tool_name)")
    # Bound the slice at the next dispatch branch (the MCP `elif`) so an
    # unrelated wait_for elsewhere in execute() cannot satisfy this.
    end = src.index("elif", start)
    branch = src[start:end]

    assert "asyncio.wait_for(" in branch
    assert "SKILL_TOOL_TIMEOUT_S" in branch
    # The skill call is the awaited coroutine INSIDE the wait_for, with the
    # module constant as its timeout.
    wf = branch.index("asyncio.wait_for(")
    call = branch.index("self.skill_loader.execute_tool(", wf)
    assert call > wf
    assert "SKILL_TOOL_TIMEOUT_S" in branch[call:]


def test_timeout_is_a_hang_guard_above_present_app_budgets():
    # present_app's internal budgets sum to ~220s worst case; the guard must
    # sit above them or it becomes the budget it is documented not to be.
    assert te_mod.SKILL_TOOL_TIMEOUT_S >= 300


# ─── Behavioral: a hung skill tool times out with the shared shape ──────


def _fake_skill(handler):
    skill = MagicMock()
    skill.is_skill_tool = MagicMock(return_value=True)
    skill.execute_tool = handler
    return skill


async def test_slow_skill_tool_gets_the_timeout_error(tmp_path, monkeypatch):
    monkeypatch.setattr(te_mod, "SKILL_TOOL_TIMEOUT_S", 0.05)
    ex = ToolExecutor(workspace=str(tmp_path))

    async def _hang(tool_name, tool_input, ctx):
        await asyncio.sleep(5)
        return "never"

    ex.skill_loader = _fake_skill(_hang)

    result = await ex.execute("slow_skill", {})
    # Same shape as the _tool_* branch's timeout: the model sees an
    # actionable error it can retry, not an exception.
    assert result == "ERROR: Tool 'slow_skill' timed out after 0.05s"


async def test_fast_skill_tool_result_survives_the_wrap(tmp_path):
    async def _quick(tool_name, tool_input, ctx):
        return "from_skill"

    ex = ToolExecutor(workspace=str(tmp_path))
    ex.skill_loader = _fake_skill(_quick)

    result = await ex.execute("quick_skill", {})
    assert result == "from_skill"


# ─── Copy probe: the resignation sentence is gone, the new copy is clean ─


def test_resignation_copy_appears_nowhere_in_app():
    old = "maximum number of tool iterations"
    hits = [
        str(p) for p in sorted(APP_DIR.rglob("*.py"))
        if old in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert hits == []


def test_cap_fallback_copy_is_internals_free():
    runner_src = (APP_DIR / "agent" / "agent_runner.py").read_text(encoding="utf-8")
    lines = [
        line for line in runner_src.splitlines()
        if "final_text = text_buf or" in line
    ]
    assert len(lines) == 1  # the text_buf-first fallback still exists, once
    copy = lines[0].split("text_buf or", 1)[1].lower()
    assert "keep going" in copy  # offers the way forward
    for banned in ("iteration", "tool", "maximum"):
        assert banned not in copy, banned
