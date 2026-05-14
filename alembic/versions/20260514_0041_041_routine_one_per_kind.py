"""routines: enforce one-active-per-kind at the DB layer (cleanup + index)

Revision ID: 041
Revises: 040
Create Date: 2026-05-14

Background
==========
The Python guard at `app/api/routines.py:create_routine` was supposed to
return 409 when an enabled routine of the same kind already existed.
Pre-2026-05-14 it used `.scalar_one_or_none()`, which raises
`MultipleResultsFound` if ≥2 rows match — i.e., the guard CRASHED in
exactly the state it was meant to prevent. The first duplicate (created
via a race / pre-guard window / direct DB insert) made every subsequent
create request 500 instead of 409.

Symptom on 2026-05-14: a user reported two identical Gmail briefings in
their day-chat. Both `email_briefing` routines were enabled in the DB;
each fired on its own cron and produced its own message.

Fix
===
Belt-and-braces: Python guard rewritten to handle the multi-row case
(separate commit), but the DB is now the load-bearing guarantee.

1. CLEANUP: for every (user_id, kind != 'agent_task') with >1 enabled
   routine, keep only the most-recently-updated one enabled. Disable
   the rest — DO NOT delete; the rows carry run history that operators
   may need for incident review. Each disabled row gets a marker in
   `config_json.disabled_reason` so the source is auditable.

2. INDEX: partial UNIQUE on `(user_id, kind) WHERE enabled = true AND
   kind != 'agent_task'`. Postgres-only feature; on SQLite (tests) we
   skip the partial clause and the index is non-unique (the test env
   doesn't need the constraint since tests don't exercise concurrent
   creates).

Idempotent across replays.
"""

from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa


revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


CLEANUP_QUERY_PG = sa.text("""
    WITH ranked AS (
        SELECT
            id, user_id, kind, enabled, config_json, updated_at,
            ROW_NUMBER() OVER (
                PARTITION BY user_id, kind
                ORDER BY updated_at DESC, created_at DESC
            ) AS rn
        FROM routines
        WHERE enabled = true AND kind <> 'agent_task'
    )
    SELECT id, user_id, kind, config_json
    FROM ranked
    WHERE rn > 1
""")


def upgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name

    if dialect == "postgresql":
        _cleanup_postgres(conn)
        _create_partial_unique_index_postgres(conn)
    else:
        # SQLite test env — skip the cleanup query (the WITH/ROW_NUMBER
        # syntax differs and we don't actually have duplicates in tests
        # because each test wipes the DB). Skip the index too; SQLite
        # supports partial indexes since 3.8, but the test harness has
        # no concurrent creates and the constraint isn't load-bearing
        # there.
        pass


def _cleanup_postgres(conn) -> None:
    """Disable all but the freshest enabled routine per (user, kind)
    for non-agent_task kinds. Audit-friendly: mark the row's
    config_json.disabled_reason so the operator can grep later."""
    rows = conn.execute(CLEANUP_QUERY_PG).fetchall()
    if not rows:
        return
    print(f"[041] cleanup: disabling {len(rows)} duplicate routine row(s)")

    for row in rows:
        rid = row[0]
        cfg = row[3] or {}
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except Exception:
                cfg = {}
        cfg = dict(cfg) if isinstance(cfg, dict) else {}
        cfg["disabled_reason"] = (
            "auto-disabled by migration 041 — duplicate of another enabled "
            "routine of the same kind. The most-recently-updated sibling "
            "was kept enabled."
        )
        conn.execute(
            sa.text("""
                UPDATE routines
                SET enabled = false,
                    config_json = CAST(:cfg AS jsonb),
                    last_status = COALESCE(last_status, 'never_run'),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :rid
            """),
            {"rid": rid, "cfg": json.dumps(cfg)},
        )


def _create_partial_unique_index_postgres(conn) -> None:
    """Partial UNIQUE on `(user_id, kind) WHERE enabled = true AND
    kind <> 'agent_task'`. IF NOT EXISTS makes the migration safe to
    replay; the cleanup above guarantees the precondition (≤1 enabled
    per group) holds before this index lands."""
    conn.execute(sa.text("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_routines_one_per_kind
        ON routines (user_id, kind)
        WHERE enabled = true AND kind <> 'agent_task'
    """))


def downgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name
    if dialect == "postgresql":
        conn.execute(sa.text(
            "DROP INDEX IF EXISTS uq_routines_one_per_kind"
        ))
    # Downgrade does NOT re-enable the disabled duplicate rows. That's
    # intentional — re-enabling would re-introduce the bug we just
    # fixed. Operators can manually flip rows back if they really want
    # to, after first deciding which sibling is canonical.
