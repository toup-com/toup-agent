"""103 — agent_probe_state: the reconciliation sweep's safety state, in the DB.

`_PROBE_STRIKES` and `_KEYLESS_NAMED_TICKS` were module-level dicts, so
"two CONSECUTIVE sick ticks before a restart" really meant "two ticks on
EITHER replica", every Railway redeploy reset the state, and the one number
an operator needed on 2026-09-12 — "46th consecutive 401 tick, since 19:12"
— lived in one process's memory and in no query. (L3-11, L3 §11.3.)

DDL kept trivial so the same statements run on the test conftest's SQLite.

Revision ID: 103
Revises: 102
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "103"
down_revision = "102"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def upgrade() -> None:
    bind = op.get_bind()
    if _has_table(bind, "agent_probe_state"):
        return
    # Agent DBs have no `users` table shaped for this FK and no sweep at all;
    # the model is PLATFORM_ONLY and the migration follows it.
    if not _has_table(bind, "users"):
        return
    op.create_table(
        "agent_probe_state",
        sa.Column("user_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False,
                  server_default="0"),
        sa.Column("last_class", sa.String(16), nullable=True),
        sa.Column("restarts_in_window", sa.Integer(), nullable=False,
                  server_default="0"),
        sa.Column("restart_window_started_at", sa.DateTime(), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(), nullable=True),
        sa.Column("escalated_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_agent_probe_state_failures", "agent_probe_state",
        ["consecutive_failures"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "agent_probe_state"):
        return
    indexes = {i["name"] for i in sa.inspect(bind).get_indexes("agent_probe_state")}
    if "ix_agent_probe_state_failures" in indexes:
        op.drop_index("ix_agent_probe_state_failures",
                      table_name="agent_probe_state")
    op.drop_table("agent_probe_state")
