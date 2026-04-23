"""rollout_attempts.container_id FK: ON DELETE SET NULL + nullable

Revision ID: 023
Revises: 022
Create Date: 2026-04-23

Admin "Delete User" failed with FK violation after our first rollout created
rollout_attempts referencing managed_containers. Per the audit-trail design,
we want historical rollout_attempts rows to survive user deletion — just
detach from the (now gone) managed_containers row.

Fix: make container_id nullable and add ON DELETE SET NULL.
"""
from alembic import op


revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Postgres: drop old FK, re-add with SET NULL. Column must be nullable too.
    op.alter_column("rollout_attempts", "container_id", nullable=True)
    op.drop_constraint("rollout_attempts_container_id_fkey", "rollout_attempts", type_="foreignkey")
    op.create_foreign_key(
        "rollout_attempts_container_id_fkey",
        "rollout_attempts",
        "managed_containers",
        ["container_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("rollout_attempts_container_id_fkey", "rollout_attempts", type_="foreignkey")
    op.create_foreign_key(
        "rollout_attempts_container_id_fkey",
        "rollout_attempts",
        "managed_containers",
        ["container_id"],
        ["id"],
    )
    op.alter_column("rollout_attempts", "container_id", nullable=False)
