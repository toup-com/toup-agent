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


def test_agent_main_registers_the_memory_jobs_behind_flag():
    """agent_main's lifespan must reference the flag and the service entry
    points it schedules.

    Memory v3 (§3.1) retires `retrieval_feedback_analysis`: it read
    `retrieval_events`, whose only feeder was the runner's per-turn
    `log_retrieval_feedback` call, and that call's input was
    hybrid_search's results — which the file model no longer produces. A
    weekly job over a table nothing writes is a cron that always reports
    zero, so it is unregistered rather than left ticking.

    v3 also retires `memory_decay` (the Ebbinghaus pass over
    `memories.strength`) and re-points the surviving slot at
    `memory_file_ops.run_memory_maintenance`, which just ensures the three
    system files exist. The slot KEEPS the id `memory_consolidation`
    deliberately: `replace_existing=True` keys on the id, so a redeploy
    replaces an old registration rather than leaving an orphan pointing at
    a deleted function. WS-5's migration hooks in here."""
    src = _agent_main_src()
    assert "settings.agent_memory_maintenance_enabled" in src
    for entry_point in ("run_memory_maintenance", "run_end_of_day_archival"):
        assert entry_point in src, entry_point
    for job_id in ("memory_consolidation", "day_archival"):
        assert f'"{job_id}"' in src, job_id
    # Code, not commentary — the comments recording the retirements name them.
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    for gone in (
        "run_retrieval_feedback_analysis",
        '"retrieval_feedback_analysis"',
        "run_decay_for_all_users",
        '"memory_decay"',
        "run_memory_file_maintenance",
    ):
        assert gone not in code, gone


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
    # v3: WHAT to forget, in plain words. There are no memory ids left.
    assert tool["input_schema"]["required"] == ["content"]


def test_memory_delete_executor_handler_exists():
    """Dispatch is getattr(self, f\"_tool_{name}\") — the handler method
    must exist and route to the one writer."""
    src = _tool_executor_src()
    assert "async def _tool_memory_delete" in src
    assert "memory_curator.instruct_global(" in src


def test_memory_delete_has_lost_its_id_source():
    """`memory_delete` took a memory_id straight out of `memory_search`'s
    rendered output — that coupling was the whole reason the search lines
    carried `id=`.

    Memory v3 (§3.2) re-points `memory_search` at memory FILES, so it
    emits `[slug] Title — snippet` and no row id exists to copy. WS-2
    finished the repair: "forget X" is now a removal INSTRUCTION routed
    through the curator, which finds the bullet, removes it, and writes the
    change line the user sees in their memory log."""
    src = _tool_executor_src()
    assert "id={mem_id}" not in src
    assert "async def _tool_memory_delete" in src
    at = src.index("async def _tool_memory_delete")
    body = src[at: at + 2000]
    assert "memory_curator.instruct_global(" in body
    assert "memory_id" in body, (
        "a model working from a stale tool list still sends memory_id; "
        "accepting it as an alias is what turns a hard error into an answer"
    )


def test_memory_delete_in_output_limits():
    from app.agent.tool_executor import TOOL_OUTPUT_LIMITS

    assert "memory_delete" in TOOL_OUTPUT_LIMITS


# ── A6-2: extraction outcome surfaced in [memory_health] ────────────


def test_memory_health_line_carries_extraction_ok():
    src = _agent_runner_src()
    assert "extraction_ok=%s" in src
    assert "_last_extraction_ok" in src
