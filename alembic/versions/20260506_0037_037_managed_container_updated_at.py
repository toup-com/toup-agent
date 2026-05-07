"""managed_containers.updated_at for prewarm reconciler

Revision ID: 037
Revises: 036
Create Date: 2026-05-06

Phase 1 of the prewarm-on-Soul.save initiative (see
docs/onboarding/prewarm-phase0.md). The reconciler that retries
stuck-provisioning rows needs a "last attempt at" timestamp so it can
distinguish "just started provisioning" (skip — let it finish) from
"stuck for hours" (retry).

`created_at` reflects only the FIRST-ever provision attempt; subsequent
re-provisions reuse the row but don't update it. `started_at` is set
only on successful start. Neither tells us "when was status last
mutated to provisioning?" — exactly what the reconciler needs.

Adding `updated_at` with `server_default=NOW()` and SQLAlchemy's
`onupdate=datetime.utcnow` mapping gives us automatic mutation
tracking. Backfill existing rows with `created_at` so the index
behaves immediately.

Composite index on `(status, updated_at)` keeps the reconciler query
plan-locked to the index — the WHERE clause is exactly that pair.

Additive change; safe to roll out alongside running pods.
"""

from alembic import op
import sqlalchemy as sa


revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "managed_containers",
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    # Backfill existing rows so the reconciler doesn't immediately
    # treat every pre-migration row as "just updated."
    op.execute(
        "UPDATE managed_containers SET updated_at = COALESCE(started_at, created_at) "
        "WHERE updated_at IS NULL"
    )
    op.create_index(
        "ix_managed_containers_status_updated_at",
        "managed_containers",
        ["status", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_managed_containers_status_updated_at", table_name="managed_containers")
    op.drop_column("managed_containers", "updated_at")
