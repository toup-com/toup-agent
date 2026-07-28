"""W2.2 — singleton-runner disabled-tools race (ContextVar isolation).

`AgentRunner` is a process singleton, and `_disabled_tool_names` used to
be bare instance state: run() wrote it per turn (user-config load +
profile merge) and the `tool_defs` property read it at the tools
capture, with multi-hundred-ms awaits in between. A concurrent
SUBAGENT run's write in that window disabled `spawn` (and the
memory-write mutators) in the PARENT's advertised toolset — and a user
run's write re-enabled them for the child. The ToolExecutor twin was
ContextVar-fixed after the 2026-05-25 incident
(tool_executor._DISABLED_TOOLS_CTX); this suite pins the same fix on
the runner side (_RUN_DISABLED_TOOLS_CTX).

Three things are pinned:

  1. Two interleaved fake runs (separate asyncio tasks, different
     profiles' disabled sets, writes deliberately raced into each
     other's write→read window) never see each other's set in
     tool_defs.
  2. Non-run access still works: with the ContextVar unset, tool_defs
     falls back to the `_disabled_tool_names` instance attr — and when
     the ContextVar IS set on the task, it wins over the attr.
  3. Source-pin: run() writes _RUN_DISABLED_TOOLS_CTX, never the bare
     attr (the only remaining assignment is the __init__ default).
"""
from __future__ import annotations

import asyncio
import contextvars
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
_AGENT_RUNNER_PATH = BACKEND_DIR / "app" / "agent" / "agent_runner.py"
_AGENT_RUNNER_SRC = _AGENT_RUNNER_PATH.read_text()


def _make_runner(core_names: list[str]):
    """Minimal AgentRunner for tool_defs behaviour — same bypass-__init__
    shape as test_connector_dispatch_t1g (the real __init__ pulls in
    dozens of services we don't need here)."""
    from app.agent.agent_runner import AgentRunner

    runner = object.__new__(AgentRunner)  # bypass __init__
    runner._core_tool_defs = [
        {"name": n, "description": n, "input_schema": {}} for n in core_names
    ]
    runner.skill_loader = None
    runner._disabled_tool_names = set()
    runner.tools = MagicMock()
    runner.tools.mcp_tool_defs = []
    return runner


# ──────────────────────────────────────────────────────────────────────
# 1. Interleaved runs are isolated
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interleaved_runs_never_see_each_others_disabled_set():
    """Reproduce the exact pre-fix race: the parent (FULL profile, no
    disabled tools) yields between its disabled-set write and its
    tool_defs read; a SUBAGENT run writes its profile-disabled set
    inside that window. Each task must still see only its OWN set."""
    from app.agent.agent_runner import _RUN_DISABLED_TOOLS_CTX

    runner = _make_runner(["exec", "spawn", "memory_store"])
    results: dict[str, set] = {}
    parent_wrote = asyncio.Event()
    child_wrote = asyncio.Event()

    async def parent_run():
        # FULL run: run() writes an empty frozenset (nothing disabled).
        _RUN_DISABLED_TOOLS_CTX.set(frozenset())
        parent_wrote.set()
        # The race window: awaits between run()'s write and the tools
        # capture. Don't read until the child has DEFINITELY written.
        await child_wrote.wait()
        results["parent"] = {t["name"] for t in runner.tool_defs}

    async def subagent_run():
        await parent_wrote.wait()
        # SUBAGENT run: profile merge disables spawn + memory writes.
        _RUN_DISABLED_TOOLS_CTX.set(frozenset({"spawn", "memory_store"}))
        child_wrote.set()
        await asyncio.sleep(0)  # yield once more before reading
        results["child"] = {t["name"] for t in runner.tool_defs}

    # gather() wraps each coroutine in its own task → own context copy.
    await asyncio.gather(parent_run(), subagent_run())

    assert results["parent"] == {"exec", "spawn", "memory_store"}, (
        "parent (FULL) lost tools to the concurrent SUBAGENT run's "
        "disabled-set write — the singleton race is back"
    )
    assert results["child"] == {"exec"}, (
        "SUBAGENT run must not see spawn/memory_store — the parent's "
        "empty set leaked into the child"
    )


@pytest.mark.asyncio
async def test_reverse_interleave_child_writes_first():
    """Mirror image: the child writes first, the parent's later write
    must not re-enable the child's disabled tools."""
    from app.agent.agent_runner import _RUN_DISABLED_TOOLS_CTX

    runner = _make_runner(["exec", "spawn"])
    results: dict[str, set] = {}
    child_wrote = asyncio.Event()
    parent_wrote = asyncio.Event()

    async def subagent_run():
        _RUN_DISABLED_TOOLS_CTX.set(frozenset({"spawn"}))
        child_wrote.set()
        await parent_wrote.wait()
        results["child"] = {t["name"] for t in runner.tool_defs}

    async def parent_run():
        await child_wrote.wait()
        _RUN_DISABLED_TOOLS_CTX.set(frozenset())
        parent_wrote.set()
        await asyncio.sleep(0)
        results["parent"] = {t["name"] for t in runner.tool_defs}

    await asyncio.gather(subagent_run(), parent_run())

    assert results["child"] == {"exec"}, (
        "parent's empty disabled set re-enabled spawn for the child"
    )
    assert results["parent"] == {"exec", "spawn"}


# ──────────────────────────────────────────────────────────────────────
# 2. Non-run access (ContextVar unset) still works
# ──────────────────────────────────────────────────────────────────────


def test_non_run_access_falls_back_to_instance_attr():
    """Callers outside a run (tests, boot-time introspection) that set
    the instance attr directly keep working — with the ContextVar at
    its None default, tool_defs filters on `_disabled_tool_names`."""
    runner = _make_runner(["exec", "stub_tool"])
    runner._disabled_tool_names = {"stub_tool"}
    names = [t["name"] for t in runner.tool_defs]
    assert names == ["exec"]

    # And with nothing disabled anywhere, everything is advertised.
    runner._disabled_tool_names = set()
    names = [t["name"] for t in runner.tool_defs]
    assert names == ["exec", "stub_tool"]


def test_contextvar_wins_over_instance_attr_when_set():
    """Precedence: a run's ContextVar set (even an EMPTY set) overrides
    the instance attr — otherwise stale attr state from an old code
    path could shadow the current run."""
    from app.agent.agent_runner import _RUN_DISABLED_TOOLS_CTX

    runner = _make_runner(["exec", "spawn"])
    runner._disabled_tool_names = {"exec"}  # stale attr state

    def _read_with_ctx(value: frozenset) -> list[str]:
        _RUN_DISABLED_TOOLS_CTX.set(value)
        return [t["name"] for t in runner.tool_defs]

    # Run in a copied context so the set() cannot leak into other tests.
    names = contextvars.copy_context().run(_read_with_ctx, frozenset({"spawn"}))
    assert names == ["exec"], "ContextVar set must win over the instance attr"

    names = contextvars.copy_context().run(_read_with_ctx, frozenset())
    assert names == ["exec", "spawn"], (
        "an explicitly EMPTY per-run set must override a non-empty attr"
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Source-pin: run() writes the ContextVar, not the bare attr
# ──────────────────────────────────────────────────────────────────────


def test_run_writes_contextvar_not_bare_attr():
    """The only remaining assignment to self._disabled_tool_names is the
    __init__ default — every per-turn write in run() goes through
    _RUN_DISABLED_TOOLS_CTX.set(...)."""
    # Optional annotation: the __init__ default is `self._disabled_tool_names: set = set()`.
    assignments = re.findall(r"self\._disabled_tool_names(?::\s*\w+)?\s*=[^=]", _AGENT_RUNNER_SRC)
    assert len(assignments) == 1, (
        f"expected exactly one self._disabled_tool_names assignment "
        f"(the __init__ default), found {len(assignments)} — run() must "
        "write _RUN_DISABLED_TOOLS_CTX instead of the bare attr"
    )
    # The three run() write sites (user-config hit / empty / except) plus
    # the profile-merge site all set the ContextVar.
    ctx_writes = _AGENT_RUNNER_SRC.count("_RUN_DISABLED_TOOLS_CTX.set(")
    assert ctx_writes >= 4, (
        f"run() must write _RUN_DISABLED_TOOLS_CTX at the user-config "
        f"load (3 branches) and the profile merge; found {ctx_writes} writes"
    )
    # The profile merge reads the ContextVar (not the attr) as its base.
    assert "(_RUN_DISABLED_TOOLS_CTX.get() or frozenset()) | frozenset(_profile_disabled)" in _AGENT_RUNNER_SRC, (
        "profile merge must build on the per-run ContextVar value"
    )


def test_tool_defs_reads_contextvar_with_attr_fallback():
    """tool_defs reads the ContextVar first and only falls back to the
    instance attr when it is unset (None default)."""
    assert "_ctx_disabled = _RUN_DISABLED_TOOLS_CTX.get()" in _AGENT_RUNNER_SRC
    assert (
        "disabled = _ctx_disabled if _ctx_disabled is not None else self._disabled_tool_names"
        in _AGENT_RUNNER_SRC
    ), "tool_defs must fall back to the instance attr only when the ContextVar is unset"


def test_executor_disabled_tools_stays_contextvar_backed():
    """run() still writes self.tools.user_disabled_tools — safe ONLY
    because the executor property is backed by its own ContextVar
    (tool_executor._DISABLED_TOOLS_CTX, the 2026-05-25 fix). Pin that
    so a refactor to a bare attr over there doesn't silently reopen
    the executor half of this race."""
    from app.agent.tool_executor import ToolExecutor

    prop = ToolExecutor.__dict__.get("user_disabled_tools")
    assert isinstance(prop, property), (
        "ToolExecutor.user_disabled_tools must remain a ContextVar-backed "
        "property, not plain instance state"
    )
