"""Add signup_attempts for persistent IP rate limiting.

Revision ID: 069
Revises: 068
Create Date: 2026-06-29

Backs app/services/signup_guard.py — a cross-replica, deploy-surviving
signup IP limiter that replaces the per-process in-memory limiter when
``signup_rate_limit_persistent`` is enabled. See
app/db/models/signup_attempt.py.

Idempotent + guarded: creates the table only when absent and only on DBs
that already host `users` (platform DBs, not per-tenant agent DBs).
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "069"
down_revision = "068"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.069")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.069] users absent; skipping (not a platform DB)")
        return
    if "signup_attempts" in tables:
        logger.info("[alembic.069] signup_attempts already present; skipping")
        return

    op.create_table(
        "signup_attempts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("ip", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_signup_attempts_ip_created", "signup_attempts", ["ip", "created_at"])
    logger.info("[alembic.069] created signup_attempts")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    if "signup_attempts" in set(insp.get_table_names()):
        op.drop_index("ix_signup_attempts_ip_created", table_name="signup_attempts")
        op.drop_table("signup_attempts")
