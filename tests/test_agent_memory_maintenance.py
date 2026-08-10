"""Regression tests for PR-4 (audit A6) — memory recall fixes.

A6-1: the memory maintenance layer (decay 6h, consolidation daily 3AM,
end-of-day archival hourly, retrieval-feedback weekly) was scheduled only
in platform_main.py, whose DB excludes the AGENT_ONLY memories/day_chats
tables — it never ran against tenant data. agent_main now mirrors the
exact same job entry points behind settings.agent_memory_maintenance_enabled
(default OFF).

A6-3: memory_store was gated out on greeting/question/web/media intents
while the system prompt mandates calling it for "remember <fact>" — the
model either hallucinated "saved!" or truthfully said it can't.

A6-6: memory_delete was advertised in TOOLS_MEMORY but had no tool
definition and no executor handler, so "forget X" could never be honored
via tool.
"""

from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _config_src() -> str:
    return (_BACKEND / "app" / "config.py").read_text()


def _agent_main_src() -> str:
    return (_BACKEND / "agent_main.py").read_text()


def _platform_main_src() -> str:
    return (_BACKEND / "platform_main.py").read_text()


def _agent_runner_src() -> str:
    return (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()


def _tool_executor_src() -> str:
    return (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()


# ── A6-1: agent-side memory maintenance scheduling ──────────────────


def test_maintenance_flag_default_is_deliberate():
    """Default ON since 2026-08-10 (G-17), after the curve was run against
    a restored copy of a real tenant store rather than reasoned about.

    The assertion is inverted rather than deleted: what matters is that
    the value is a decision someone made with evidence, not a default that
    drifts. The evidence is recorded beside the field in config.py — most
    importantly that decay clamps at MIN_STRENGTH=0.1 while hybrid_search
    floors at `strength >= 0.1`, so decay can only re-weight, never
    remove a memory from recall.
    """
    assert "agent_memory_maintenance_enabled: bool = True" in _config_src()


def test_agent_main_registers_all_four_memory_jobs_behind_flag():
    """agent_main's lifespan must reference the flag and the exact same
    service entry points platform_main's setup_scheduler registers."""
    src = _agent_main_src()
    assert "settings.agent_memory_maintenance_enabled" in src
    for entry_point in (
        "run_decay_for_all_users",
        "run_consolidation_for_all_users",
        "run_retrieval_feedback_analysis",
        "run_end_of_day_archival",
    ):
        assert entry_point in src, entry_point
    # Job ids mirror platform_main's setup_scheduler.
    for job_id in ("memory_decay", "memory_consolidation",
                   "retrieval_feedback_analysis", "day_archival"):
        assert f'"{job_id}"' in src, job_id


def test_agent_main_archival_respects_day_recall_gate():
    """The hourly archival job keeps the same enable_day_recall gate as
    platform_main's setup_scheduler (forced ON in the agent image)."""
    assert "settings.enable_day_recall" in _agent_main_src()


def test_agent_main_registration_is_per_job_guarded():
    """One failing job registration must never kill the lifespan."""
    src = _agent_main_src()
    assert "Could not register memory job" in src


def test_platform_main_notes_memory_jobs_noop():
    """platform_main keeps its scheduler but documents that the memory
    jobs no-op there (AGENT_ONLY tables absent from its DB)."""
    src = _platform_main_src()
    assert "agent_memory_maintenance_enabled" in src
    assert "AGENT_ONLY" in src


# ── A6-3: memory_store reachable on every intent ────────────────────


def test_memory_store_in_always_included_tools():
    from app.agent.query_intent import _ALWAYS_INCLUDED_TOOLS

    assert "memory_store" in _ALWAYS_INCLUDED_TOOLS
    # The read-side affordance that motivated the set stays present.
    assert "memory_search" in _ALWAYS_INCLUDED_TOOLS


def test_memory_store_exposed_on_question_intent():
    """A 'remember <fact>' phrasing that scores zero (typos) lands on
    question intent — the tool must still be in the filtered list."""
    from app.agent.query_intent import classify_query_intent, filter_tools_by_intent

    intent = classify_query_intent("Rember my sistr is called Ana")
    all_tools = [{"name": n} for n in (
        "memory_store", "memory_search", "web_search", "exec",
    )]
    kept = {t["name"] for t in filter_tools_by_intent(all_tools, intent)}
    assert "memory_store" in kept


def test_subagent_deny_list_still_blocks_memory_writes():
    """Always-included cannot resurrect a tool the profile filter already
    removed from tool_defs — the deny sets must still name the writers."""
    from app.agent.prompt_profile import SUBAGENT_DISABLED_TOOLS

    assert "memory_store" in SUBAGENT_DISABLED_TOOLS
    assert "memory_delete" in SUBAGENT_DISABLED_TOOLS


# ── A6-6: memory_delete exists end to end ────────────────────────────


def test_memory_delete_tool_definition_exists():
    from app.agent.tool_definitions import get_agent_tools

    names = {t["name"] for t in get_agent_tools()}
    assert "memory_delete" in names
    tool = next(t for t in get_agent_tools() if t["name"] == "memory_delete")
    assert tool["input_schema"]["required"] == ["memory_id"]


def test_memory_delete_executor_handler_exists():
    """Dispatch is getattr(self, f\"_tool_{name}\") — the handler method
    must exist and wire to the soft-delete service."""
    src = _tool_executor_src()
    assert "async def _tool_memory_delete" in src
    assert "delete_memory(" in src


def test_memory_search_results_surface_ids():
    """memory_delete takes a memory_id from memory_search output — the
    search result lines must include the id."""
    assert "id={mem_id}" in _tool_executor_src()


def test_memory_delete_in_output_limits():
    from app.agent.tool_executor import TOOL_OUTPUT_LIMITS

    assert "memory_delete" in TOOL_OUTPUT_LIMITS


# ── A6-2: extraction outcome surfaced in [memory_health] ────────────


def test_memory_health_line_carries_extraction_ok():
    src = _agent_runner_src()
    assert "extraction_ok=%s" in src
    assert "_last_extraction_ok" in src
