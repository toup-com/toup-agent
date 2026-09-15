"""102 — infra_leases: one fleet-wide runner for the singleton loops.

2026-09-12 onboarding incident (L3-1): every infra-mutating background loop
ran on BOTH Railway replicas with no leader election, so one bad platform-side
mapping became 92 container restarts in 2 h 28 m — every tick produced exactly
one restart POST from each replica. The lease row is the gate.

DDL kept deliberately trivial (no partial indexes, no generated columns, no
`now()` defaults) so the same statements run on the test conftest's SQLite.

Revision ID: 102
Revises: 101
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "102"
down_revision = "101"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def upgrade() -> None:
    bind = op.get_bind()
    if _has_table(bind, "infra_leases"):
        return
    op.create_table(
        "infra_leases",
        sa.Column("name", sa.String(64), primary_key=True, nullable=False),
        sa.Column("holder", sa.String(80), nullable=False),
        sa.Column("acquired_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("tick_seq", sa.BigInteger(), nullable=False,
                  server_default="0"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    if _has_table(bind, "infra_leases"):
        op.drop_table("infra_leases")
