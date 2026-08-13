"""
Test: Every table in Base.metadata is categorized in exactly one partition set.

This ensures that when someone adds a new model, they are forced to decide
whether it belongs in the agent DB, platform DB, or both. If they skip
the categorization, this test fails and tells them exactly what to do.
"""

import pytest
from app.db.models.base import (
    Base,
    AGENT_ONLY_TABLES,
    PLATFORM_ONLY_TABLES,
    SHARED_TABLES,
    SHARED_COLUMN_AUTHORITY,
)


def _shared_column_keys() -> set[str]:
    """Every '<table>.<column>' of every SHARED table, from live metadata."""
    return {
        f"{table.name}.{column.name}"
        for table in Base.metadata.sorted_tables
        if table.name in SHARED_TABLES
        for column in table.columns
    }


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


# ── L-1b: shared-column authority map ─────────────────────────────────
#
# A shared table's column is TWO columns — a platform copy and a tenant
# copy — and twice a writer updated one while the reader read the other
# (users.timezone 2026-08-10, agent_configs.agent_name 2026-08-11).
# These tests force every shared column to carry an explicit authority
# declaration in base.py, so the next such column cannot land unnoticed.


def test_every_shared_column_has_an_authority_entry():
    """Every column of every SHARED table must be declared in
    SHARED_COLUMN_AUTHORITY."""
    missing = sorted(_shared_column_keys() - set(SHARED_COLUMN_AUTHORITY))
    assert not missing, (
        f"new shared column(s) {missing} — add a SHARED_COLUMN_AUTHORITY entry in "
        f"app/db/models/base.py declaring which database owns each one. A shared "
        f"table exists in BOTH the platform DB and every tenant DB, so a column "
        f"without a declared owner is one writer away from the users.timezone / "
        f"agent_configs.agent_name class of silent split (written in one database, "
        f"read from the other). Declare authority "
        f"('platform' | 'tenant' | 'both-synced' | 'immutable-after-create'), the "
        f"write_paths you grepped, the sync mechanism (or 'none'), and a one-line note."
    )


def test_no_phantom_authority_entries():
    """Every key in SHARED_COLUMN_AUTHORITY must be a real shared
    table.column — catches renames, deletions and tables leaving
    SHARED_TABLES."""
    phantom = sorted(set(SHARED_COLUMN_AUTHORITY) - _shared_column_keys())
    assert not phantom, (
        f"SHARED_COLUMN_AUTHORITY declares column(s) that no shared table has: "
        f"{phantom}. Either the column was renamed/removed (update the map key in "
        f"app/db/models/base.py) or its table is no longer in SHARED_TABLES (move "
        f"or delete the entries)."
    )


def test_authority_values_are_well_formed():
    """Each entry must carry a valid authority enum, non-empty note,
    write_paths list, and a sync declaration."""
    allowed = {"platform", "tenant", "both-synced", "immutable-after-create"}
    for key, entry in SHARED_COLUMN_AUTHORITY.items():
        assert isinstance(entry, dict), f"{key}: entry must be a dict, got {type(entry).__name__}"
        assert entry.get("authority") in allowed, (
            f"{key}: authority {entry.get('authority')!r} not in {sorted(allowed)}"
        )
        write_paths = entry.get("write_paths")
        assert isinstance(write_paths, list) and write_paths and all(
            isinstance(p, str) and p.strip() for p in write_paths
        ), f"{key}: write_paths must be a non-empty list of non-empty strings"
        sync = entry.get("sync")
        assert isinstance(sync, str) and sync.strip(), (
            f"{key}: sync must name the cross-DB mechanism, or be 'none'"
        )
        note = entry.get("note")
        assert isinstance(note, str) and note.strip(), (
            f"{key}: note must be a non-empty one-liner explaining why this authority"
        )
