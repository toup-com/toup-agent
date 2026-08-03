"""The boot-DDL planner may never skip a statement that was needed.

`init_db()` replays ~358 statements against a LIVE tenant database on every
container start. They are idempotent, but idempotent is not free: Postgres
takes ACCESS EXCLUSIVE *before* evaluating `IF NOT EXISTS`, so each no-op
still queues behind any open transaction on that table — measured on
postgres:16 (2026-08-03), a no-op `ADD COLUMN IF NOT EXISTS` hit `lock
timeout` with pg_locks showing AccessExclusiveLock granted=false while a
single plain SELECT transaction was open. That is the mechanism behind
`aborted_canary_failed` / `health_checks_passed: 0`.

The planner skips a statement only when the live catalog already shows its
effect. The risk it introduces is asymmetric — a wrong skip is a silently
missing column that surfaces weeks later, a wrong run costs one lock — so
these tests are written around the conservative direction: everything the
planner is not certain about must still run.

Direct invocation works:
    cd backend && ENVIRONMENT=development python -m pytest tests/test_ddl_plan.py
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

os.environ.setdefault("ENVIRONMENT", "development")
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.db.ddl_plan import (  # noqa: E402
    SNAPSHOT_SQL,
    SchemaSnapshot,
    is_satisfied,
    normalize,
    plan,
    snapshot_from_rows,
)

EMPTY = SchemaSnapshot(columns=frozenset(), indexes=frozenset(), tables=frozenset())


def _snap(columns=(), indexes=(), tables=()):
    return snapshot_from_rows(columns, indexes, tables)


# ── the conservative direction ────────────────────────────────────────

def test_empty_snapshot_skips_nothing():
    """A failed snapshot must degrade to the old behaviour exactly."""
    stmts = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname TEXT",
        "CREATE INDEX IF NOT EXISTS ix_users_nickname ON users (nickname)",
    ]
    to_run, skipped = plan(stmts, EMPTY)
    assert to_run == stmts and skipped == []


def test_missing_column_is_planned_in():
    snap = _snap(columns=[("users", "email")])
    assert not is_satisfied(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname TEXT", snap
    )


def test_present_column_is_skipped():
    snap = _snap(columns=[("users", "nickname")])
    assert is_satisfied(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname TEXT", snap
    )


def test_same_column_name_on_a_different_table_does_not_count():
    """The key is (table, column) — not the column name alone."""
    snap = _snap(columns=[("memories", "nickname")])
    assert not is_satisfied(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname TEXT", snap
    )


def test_index_and_table_shapes():
    snap = _snap(indexes=["ix_users_nickname"], tables=["day_chats"])
    assert is_satisfied(
        "CREATE INDEX IF NOT EXISTS ix_users_nickname ON users (nickname)", snap
    )
    assert not is_satisfied(
        "CREATE INDEX IF NOT EXISTS ix_users_other ON users (other)", snap
    )
    assert is_satisfied("CREATE TABLE IF NOT EXISTS day_chats (id TEXT)", snap)
    assert not is_satisfied("CREATE TABLE IF NOT EXISTS widgets (id TEXT)", snap)


def test_unique_index_shape_is_recognised():
    snap = _snap(indexes=["uq_thing"])
    assert is_satisfied(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_thing ON things (a, b)", snap
    )


# ── everything uncertain must run ─────────────────────────────────────

def test_stateful_shapes_always_run():
    """Backfills, DO blocks, functions, triggers, drops, ALTER COLUMN.

    These are the statements whose effect the catalog cannot confirm, so
    they must survive planning untouched no matter how complete the
    snapshot is.
    """
    full = _snap(
        columns=[("agent_configs", "agent_model"), ("messages", "channel")],
        indexes=["uq_routines_one_per_kind"],
        tables=["messages", "routines"],
    )
    always = [
        "ALTER TABLE agent_configs ALTER COLUMN agent_model DROP DEFAULT",
        "UPDATE messages m SET channel = c.channel FROM conversations c "
        "WHERE m.conversation_id = c.id AND m.channel IS NULL",
        "DELETE FROM reconciliation_logs WHERE created_at < NOW() - INTERVAL '30 days'",
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.tables "
        "WHERE table_name='day_chats') THEN NULL; END IF; END $$",
        "DROP INDEX IF EXISTS uq_routines_one_per_kind",
        "CREATE UNIQUE INDEX uq_routines_one_per_kind ON routines (user_id, kind) "
        "WHERE enabled = true",
        "CREATE OR REPLACE FUNCTION memories_search_vector_update() RETURNS trigger "
        "AS $$ BEGIN RETURN NEW; END $$ LANGUAGE plpgsql",
        "DROP TRIGGER IF EXISTS trg_memories_search_vector ON memories",
        "CREATE TRIGGER trg_memories_search_vector BEFORE INSERT ON memories "
        "FOR EACH ROW EXECUTE FUNCTION memories_search_vector_update()",
    ]
    to_run, skipped = plan(always, full)
    assert skipped == [], f"planner skipped a stateful statement: {skipped}"
    assert to_run == always


def test_create_unique_index_without_if_not_exists_always_runs():
    """No IF NOT EXISTS means the author wanted an unconditional rebuild."""
    snap = _snap(indexes=["uq_thing"])
    assert not is_satisfied("CREATE UNIQUE INDEX uq_thing ON things (a, b)", snap)


def test_a_second_statement_disqualifies_the_whole_string():
    """`CREATE TABLE ...; INSERT ...` must not be judged on its first half —
    skipping it would silently drop the INSERT."""
    snap = _snap(tables=["widgets"], columns=[("users", "nickname")])
    assert not is_satisfied(
        "CREATE TABLE IF NOT EXISTS widgets (id TEXT); INSERT INTO widgets VALUES ('x')",
        snap,
    )
    assert not is_satisfied(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname TEXT; "
        "UPDATE users SET nickname = 'x'",
        snap,
    )


def test_a_second_action_disqualifies_the_alter():
    """Only the first ADD COLUMN is parsed, so a multi-action ALTER whose
    later actions are NOT satisfied must not be skipped on the first."""
    snap = _snap(columns=[("users", "nickname")])
    assert not is_satisfied(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname TEXT, "
        "ADD COLUMN IF NOT EXISTS other TEXT",
        snap,
    )


def test_unknown_shape_runs():
    assert not is_satisfied("VACUUM ANALYZE users", _snap(tables=["users"]))
    assert not is_satisfied("", EMPTY)


# ── mechanics ─────────────────────────────────────────────────────────

def test_case_and_whitespace_do_not_change_the_verdict():
    snap = _snap(columns=[("users", "nickname")])
    assert is_satisfied(
        "  alter   table\n  Users\n  add column if not exists  NickName   TEXT  ", snap
    )


def test_trailing_semicolon_is_not_a_second_statement():
    snap = _snap(columns=[("users", "nickname")])
    assert is_satisfied(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname TEXT;", snap
    )
    assert normalize("SELECT 1;") == "SELECT 1"


def test_plan_preserves_relative_order():
    """The list is a migration sequence; survivors must keep their order."""
    snap = _snap(columns=[("t", "b")])
    stmts = [
        "ALTER TABLE t ADD COLUMN IF NOT EXISTS a TEXT",
        "ALTER TABLE t ADD COLUMN IF NOT EXISTS b TEXT",  # skipped
        "UPDATE t SET a = 'x'",
        "ALTER TABLE t ADD COLUMN IF NOT EXISTS c TEXT",
    ]
    to_run, skipped = plan(stmts, snap)
    assert to_run == [stmts[0], stmts[2], stmts[3]]
    assert skipped == [stmts[1]]


def test_snapshot_sql_is_read_only_and_schema_scoped():
    """These run at boot against a live tenant DB — they must not write and
    must not read another tenant's schema."""
    for name, sql in SNAPSHOT_SQL.items():
        upper = sql.upper()
        assert upper.lstrip().startswith("SELECT"), name
        assert "CURRENT_SCHEMA()" in upper, f"{name} is not schema-scoped"
        for forbidden in ("INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "CREATE"):
            assert forbidden not in upper, f"{name} contains {forbidden}"


# ── against the REAL statement list ───────────────────────────────────

def _real_alter_statements() -> list:
    """`_alter_statements` is a local inside init_db(), so lift the literals
    out of the source. Only string constants — the f-string entries that get
    appended at runtime are covered by the shape tests above."""
    src = (BACKEND / "app" / "db" / "database.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            if any(
                isinstance(t, ast.Name) and t.id == "_alter_statements"
                for t in node.targets
            ):
                return [
                    e.value for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
    raise AssertionError("_alter_statements literal list not found")


def test_the_real_list_is_mostly_plannable():
    """Sanity on the actual production list: the ALTER/CREATE INDEX bulk is
    what makes boots slow, so planning has to reach it."""
    stmts = _real_alter_statements()
    assert len(stmts) > 150, f"only found {len(stmts)} statements — parser drifted?"

    # A snapshot that satisfies every skippable statement in the list.
    cols, idxs, tbls = [], [], []
    for s in stmts:
        one = normalize(s)
        import re
        m = re.match(
            r"^ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+(\w+)",
            one, re.I,
        )
        if m:
            cols.append((m.group(1), m.group(2)))
            continue
        m = re.match(r"^CREATE\s+(?:UNIQUE\s+)?INDEX\s+IF\s+NOT\s+EXISTS\s+(\w+)", one, re.I)
        if m:
            idxs.append(m.group(1))
            continue
        m = re.match(r"^CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)", one, re.I)
        if m:
            tbls.append(m.group(1))

    to_run, skipped = plan(stmts, _snap(cols, idxs, tbls))
    # On a fully converged tenant the pass should collapse to the handful of
    # genuinely stateful statements.
    assert len(skipped) >= 150, f"only {len(skipped)} of {len(stmts)} planned away"
    assert len(to_run) <= 25, f"{len(to_run)} statements still run: {to_run[:3]}"

    # And none of the survivors is a plain idempotent ADD COLUMN — those are
    # exactly what planning exists to remove.
    import re
    leftovers = [
        s for s in to_run
        if re.match(r"^ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+\w+\s", normalize(s), re.I)
        and len(re.findall(r"\b(ADD|DROP|ALTER|RENAME)\s+(COLUMN|CONSTRAINT)\b", normalize(s), re.I)) == 1
    ]
    assert not leftovers, f"plain ADD COLUMN statements survived planning: {leftovers}"


def test_every_real_statement_survives_an_empty_snapshot():
    """The fresh-tenant path: a brand-new database must still receive the
    entire list, in order."""
    stmts = _real_alter_statements()
    to_run, skipped = plan(stmts, EMPTY)
    assert to_run == stmts
    assert skipped == []
