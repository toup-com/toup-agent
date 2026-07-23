"""
Test: Every table in Base.metadata is categorized in exactly one partition set.

This ensures that when someone adds a new model, they are forced to decide
whether it belongs in the agent DB, platform DB, or both. If they skip
the categorization, this test fails and tells them exactly what to do.
"""

import pytest
from app.db.models.base import Base, AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES, SHARED_TABLES


def test_every_table_is_categorized():
    """Every table registered on Base.metadata must appear in exactly one set."""
    all_table_names = {t.name for t in Base.metadata.sorted_tables}
    categorized = AGENT_ONLY_TABLES | PLATFORM_ONLY_TABLES | SHARED_TABLES

    uncategorized = all_table_names - categorized
    assert not uncategorized, (
        f"Tables not categorized in base.py: {uncategorized}. "
        f"Add each to AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES, or SHARED_TABLES."
    )


def test_no_table_in_multiple_sets():
    """A table must not appear in more than one partition set."""
    overlap_ap = AGENT_ONLY_TABLES & PLATFORM_ONLY_TABLES
    overlap_as = AGENT_ONLY_TABLES & SHARED_TABLES
    overlap_ps = PLATFORM_ONLY_TABLES & SHARED_TABLES

    assert not overlap_ap, f"Tables in both AGENT_ONLY and PLATFORM_ONLY: {overlap_ap}"
    assert not overlap_as, f"Tables in both AGENT_ONLY and SHARED: {overlap_as}"
    assert not overlap_ps, f"Tables in both PLATFORM_ONLY and SHARED: {overlap_ps}"


def test_no_phantom_entries():
    """Partition sets must not contain table names that don't exist in metadata."""
    all_table_names = {t.name for t in Base.metadata.sorted_tables}
    categorized = AGENT_ONLY_TABLES | PLATFORM_ONLY_TABLES | SHARED_TABLES

    phantom = categorized - all_table_names
    assert not phantom, (
        f"Partition sets reference tables that don't exist in Base.metadata: {phantom}. "
        f"Remove them from the sets in base.py, or check if the model import is missing."
    )


def test_agent_only_tables_are_correct():
    """Smoke test: known agent-only tables must be in the set."""
    for t in ("conversations", "messages", "memories", "day_chats", "context_budget_logs"):
        assert t in AGENT_ONLY_TABLES, f"'{t}' should be in AGENT_ONLY_TABLES"


def test_platform_only_tables_are_correct():
    """Smoke test: known platform-only tables must be in the set."""
    for t in ("vps_plans", "invites", "managed_containers"):
        assert t in PLATFORM_ONLY_TABLES, f"'{t}' should be in PLATFORM_ONLY_TABLES"


def test_shared_tables_are_correct():
    """Smoke test: tables that live in BOTH databases.

    agent_configs is SHARED (not platform-only): the agent chat/runner/tool path
    reads it by direct query (ws_chat.py, agent_runner.py, tool_executor.py,
    chat.py), so init_db must create it on agent DBs too — see the note in
    base.py SHARED_TABLES for the 2026-07 chat-persistence incident this fixes.
    """
    assert "users" in SHARED_TABLES
    assert "agent_configs" in SHARED_TABLES
