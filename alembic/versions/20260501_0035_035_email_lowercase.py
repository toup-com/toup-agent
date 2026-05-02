"""Lowercase all existing user emails + enforce via partial unique index.

Email lookup was case-sensitive everywhere, which let a single user
sign up twice with different casing (`Arshia@toup.ai` and
`arshia@toup.ai`) — exactly what happened on 2026-05-01 when a fresh
signup hit a 500 on /agent-setup/config (separate `agent_model`
nullability bug, fixed in 034), retried with a different case, and
ended up with a duplicate User row.

Code-side fix: every email-touching service (register / login / lookup)
now `.strip().lower()` before query/insert. This migration handles the
two pieces code can't:

1. Backfill: any existing rows with non-lowercase emails are normalised
   in place. As of 2026-05-01 there was exactly one (`N@toup.ai`).

2. Defence-in-depth: a unique index on `LOWER(email)` so even if a
   future code path bypasses normalisation, Postgres will reject the
   duplicate. The existing `users_email_key` UNIQUE constraint stays
   in place; this is additive.

Revision ID: 035
Revises: 034
Create Date: 2026-05-01
"""

from alembic import op


revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Resolve any already-existing case-collisions before adding the
    #    LOWER() unique index — otherwise the index creation aborts.
    #    Strategy: for any LOWER(email) that has more than one row, keep
    #    the most recent (likely the one the user is actively using) and
    #    rename the others to a unique tombstone so they don't collide.
    #    The cascade-delete-user flow can clean these up later if needed.
    op.execute(
        """
        WITH dups AS (
            SELECT id, email,
                   row_number() OVER (PARTITION BY LOWER(email) ORDER BY created_at DESC) AS rn
            FROM users
        )
        UPDATE users u
        SET email = u.email || '.dup-' || u.id
        FROM dups
        WHERE u.id = dups.id AND dups.rn > 1
        """
    )

    # 2. Lowercase every email.
    op.execute("UPDATE users SET email = LOWER(email) WHERE email <> LOWER(email)")

    # 3. Functional unique index on LOWER(email) — additive to the existing
    #    `users_email_key`. Catches any future code path that bypasses
    #    the .lower() in auth_service.create_user. Raw SQL because
    #    op.create_index doesn't accept expression columns cleanly.
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email_lower ON users (LOWER(email))")


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_users_email_lower")
    # Tombstone-renames aren't reversible — those rows would need
    # manual reconciliation if rollback ever needed perfect parity.
    # (At time of writing, no collisions existed, so this is a no-op
    # in practice for the rows that matter.)
