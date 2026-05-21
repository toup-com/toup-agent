"""routines: exempt `reminder` kind from the one-per-kind partial UNIQUE

Revision ID: 053a
Revises: 052
Create Date: 2026-05-21

Background
==========
The `routines__remind` skill is designed to create ONE routine per
"remind me to X" agent request. Users naturally have many reminders
("call mom at 6", "water plants Tuesday", "standup ping at 9:30").
The Python guard in `app/services/routine_uniqueness.py` exempts
`agent_task` but accidentally NOT `reminder` — so creating a second
reminder returns 409 with detail
``An enabled 'reminder' routine already exists for this user``.

The Python fix (same commit) adds `reminder` to
`KINDS_EXEMPT_FROM_ONE_PER_KIND`. But the DB-level partial UNIQUE
index from mig 041 also enforces uniqueness:

    CREATE UNIQUE INDEX uq_routines_one_per_kind
    ON routines (user_id, kind)
    WHERE enabled = true AND kind <> 'agent_task'

For `kind='reminder'`, the predicate matches → the index enforces
single-row-per-user → INSERT of reminder #2 fails with 23505
regardless of the Python guard. This migration recreates the index
to also exempt `reminder`.

Why revision id `053a` (not `053`)
==================================
The credit-billing-system branch (NOT YET merged to main as of this
commit) authors mig `053` for `subscription_plans` + `credit_ledger`
+ `credit_reservations`. To avoid a head-collision when both PRs
merge, this revision uses id `053a` and `down_revision='052'`. When
the credit-system branch merges, the operator runs
`alembic merge -m 'merge reminder-exempt + credit-system' 053a 055`
which produces a single linear chain.

The production database is already past 055 (credit-system migs
applied out-of-band on 2026-05-21). Stamping the merge rev there is
a one-line operator action.

Idempotent
==========
Uses ``DROP INDEX IF EXISTS`` + ``CREATE UNIQUE INDEX``. Safe to
replay; safe on a DB that has never had the index. SQLite test env:
the index was never created (mig 041 skips it on SQLite), so this
mig is a no-op there.
"""
from __future__ import annotations

import logging

from alembic import op
import sqlalchemy as sa


revision = "053a"
down_revision = "052"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.053a")


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        logger.info(
            "[alembic.053a] skipping — partial UNIQUE index is "
            "Postgres-only (see mig 041)"
        )
        return

    if not _table_exists(conn, "routines"):
        logger.info(
            "[alembic.053a] routines table absent — no-op "
            "(platform DB or pre-routines tenant DB)"
        )
        return

    conn.execute(sa.text("DROP INDEX IF EXISTS uq_routines_one_per_kind"))
    conn.execute(sa.text("""
        CREATE UNIQUE INDEX uq_routines_one_per_kind
        ON routines (user_id, kind)
        WHERE enabled = true AND kind NOT IN ('agent_task', 'reminder')
    """))
    logger.info(
        "[alembic.053a] uq_routines_one_per_kind recreated to exempt "
        "'reminder' alongside 'agent_task'"
    )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return
    if not _table_exists(conn, "routines"):
        return

    conn.execute(sa.text("DROP INDEX IF EXISTS uq_routines_one_per_kind"))
    conn.execute(sa.text("""
        CREATE UNIQUE INDEX uq_routines_one_per_kind
        ON routines (user_id, kind)
        WHERE enabled = true AND kind <> 'agent_task'
    """))
    logger.info(
        "[alembic.053a] downgrade — restored mig 041 predicate "
        "(agent_task-only exemption)"
    )


def _table_exists(conn, table: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        return table in set(insp.get_table_names())
    except Exception:
        return False
