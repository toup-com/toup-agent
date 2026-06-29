"""Add users.apple_sub for stable Sign-in-with-Apple dedupe.

Revision ID: 068
Revises: 067
Create Date: 2026-06-29

Stores the Apple `sub` claim (stable per Apple-ID per app, unlike the
mutable Hide-My-Email relay address) so Apple sign-ins dedupe on a durable
identity. See app/db/models/user.py and app/api/auth.py apple_auth.

Idempotent + guarded: only adds the column (+ unique index) when the
`users` table exists and the column is absent. Existing rows backfill to
NULL (multiple NULLs are allowed under a Postgres unique index), so the
change is inert until Apple sign-ins start writing it.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "068"
down_revision = "067"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.068")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if "users" not in set(insp.get_table_names()):
        logger.info("[alembic.068] users table absent; skipping (not a platform DB)")
        return

    cols = {c["name"] for c in insp.get_columns("users")}
    if "apple_sub" not in cols:
        op.add_column("users", sa.Column("apple_sub", sa.String(255), nullable=True))
        logger.info("[alembic.068] added users.apple_sub")
    else:
        logger.info("[alembic.068] users.apple_sub already present; skipping column")

    existing_idx = {i["name"] for i in insp.get_indexes("users")}
    if "ix_users_apple_sub" not in existing_idx:
        op.create_index("ix_users_apple_sub", "users", ["apple_sub"], unique=True)
        logger.info("[alembic.068] created unique index ix_users_apple_sub")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    if "users" not in set(insp.get_table_names()):
        return
    existing_idx = {i["name"] for i in insp.get_indexes("users")}
    if "ix_users_apple_sub" in existing_idx:
        op.drop_index("ix_users_apple_sub", table_name="users")
    cols = {c["name"] for c in insp.get_columns("users")}
    if "apple_sub" in cols:
        op.drop_column("users", "apple_sub")
