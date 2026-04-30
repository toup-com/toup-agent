"""Add `last_progress_at` heartbeat column to rollouts.

The reconciler uses this to orphan rollouts whose orchestrator has
died (typically: Railway redeployed mid-execution and killed the
in-process loop). Without heartbeat, only the 30-min total-age
threshold catches stuck rollouts — which is too slow when multiple
deploys land in rapid succession and each kills the previous
orchestrator before any single one can complete.

NULL on existing rows; reconciler falls back to `started_at` when
reading. Backfilled to `started_at` for rows present at migration
time so the threshold check works the same for old rows.

Revision ID: 033
Revises: 032
Create Date: 2026-04-30
"""

from alembic import op
import sqlalchemy as sa


revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "rollouts",
        sa.Column("last_progress_at", sa.DateTime, nullable=True),
    )
    # Index the heartbeat column — reconciler queries can use it to
    # narrow the candidate set when there are many historical rollouts.
    op.create_index(
        "ix_rollouts_last_progress_at",
        "rollouts",
        ["last_progress_at"],
    )
    # Backfill existing rows so the reconciler treats them consistently
    # (NULL would be read as "stale forever" by a naive check; we set
    # last_progress_at = started_at so the existing total-age threshold
    # is what governs them, until they're touched again by orchestrator
    # state transitions and pick up real heartbeats).
    op.execute(
        "UPDATE rollouts SET last_progress_at = started_at WHERE last_progress_at IS NULL"
    )


def downgrade():
    op.drop_index("ix_rollouts_last_progress_at", table_name="rollouts")
    op.drop_column("rollouts", "last_progress_at")
