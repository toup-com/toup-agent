"""Add users.apple_refresh_token for Sign in with Apple revocation.

Revision ID: 061
Revises: 060
Create Date: 2026-06-09

Guideline 5.1.1(v): an app that offers Sign in with Apple must revoke the
user's Apple tokens when the account is deleted. To revoke later we must
capture a refresh token at sign-in (by exchanging the one-time
authorization_code) and store it. This adds a single nullable, Fernet-
encrypted Text column for that token. NULL for email/Google users and for
Apple sign-ins that predate the SIWA signing-key configuration.

Idempotent + guarded: the column is only added if the `users` table exists
and the column is absent, so re-runs and per-tenant agent DBs (which have
no `users` table) are safe.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "061"
down_revision = "060"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.061")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if "users" not in set(insp.get_table_names()):
        logger.info("[alembic.061] users table absent; skipping (not a platform DB)")
        return

    existing = {c["name"] for c in insp.get_columns("users")}
    if "apple_refresh_token" not in existing:
        op.add_column(
            "users",
            sa.Column("apple_refresh_token", sa.Text(), nullable=True),
        )
        logger.info("[alembic.061] added users.apple_refresh_token")
    else:
        logger.info("[alembic.061] users.apple_refresh_token already present; skipping")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if "users" not in set(insp.get_table_names()):
        return

    existing = {c["name"] for c in insp.get_columns("users")}
    if "apple_refresh_token" in existing:
        op.drop_column("users", "apple_refresh_token")
