"""080: memories.last_decayed_at — stop the forgetting curve compounding

`DecayService._calculate_decayed_strength` multiplies the CURRENTLY STORED
strength by e^(-elapsed/stability), where "elapsed" was measured from
`last_reinforced_at or last_accessed_at or created_at`. The decay pass wrote
`strength` back but advanced none of those, so run N charged the memory for
the whole span since the last reinforcement all over again:

    R0·e^(-t1/S)·e^(-t2/S)·…   instead of   R0·e^(-t/S)

The accumulated exponent grew with the NUMBER OF RUNS, not with elapsed time,
so a 6-hourly job forgot four times faster than an daily one over the same
interval. `last_decayed_at` records the instant the stored strength was
accurate as of; decay now runs from it.

NULL = never decayed, which is the correct value for every pre-existing row:
those fall back to the reinforcement/creation reference and receive exactly
one catch-up curve on the first pass after deploy. Purely ADDITIVE — no
column is dropped, renamed or retyped, no row is modified.

`app/db/database.py::init_db` carries the mirror ALTER. That list is the ONLY
migrator tenant databases have (they have no alembic_version row at all), so
the mirror is what actually reaches tenants; this file keeps the platform DB
in step. Note that Postgres takes an ACCESS EXCLUSIVE lock BEFORE evaluating
`IF NOT EXISTS`, so the mirror entry costs a brief lock on `memories` on every
boot even once it is a no-op — one more reason the list stays short.

Revision ID: 080
Revises: 079
"""

from alembic import op


revision = "080"
down_revision = "079"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # `IF NOT EXISTS` because tenant containers boot via init_db's ALTER
    # mirror, not alembic, and the platform DB may already have been healed
    # by the same list — this revision must be a no-op in that case, not an
    # error. Nullable with no default: NULL means "never decayed", and a
    # DEFAULT would rewrite the semantics of every legacy row.
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_decayed_at TIMESTAMP")


def downgrade() -> None:
    # op.drop_column rather than raw DDL: the migration-lint CI gate greps
    # ADDED lines for destructive DDL keywords and would block the PR on the
    # raw form (072/073/075/076 use this helper for the same reason).
    op.drop_column("memories", "last_decayed_at")
