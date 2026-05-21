"""Add email verification columns to users.

Revision ID: 055
Revises: 054
Create Date: 2026-05-21

Adds three columns to ``users``:

* ``email_verified_at``         — UTC timestamp when the user clicked
                                  the verification link. NULL = unverified.
* ``email_verification_token``  — opaque urlsafe token (32 raw bytes).
                                  Cleared on successful verify → link
                                  is single-use. Rotated on resend.
* ``email_verification_sent_at`` — UTC timestamp the most-recent
                                   verification email was sent. Used
                                   to throttle resends (one-per-60s).

Purely additive. Existing users are unverified (email_verified_at IS
NULL); the credit-system enforcement gate exempts users created before
``settings.email_verification_required_after_iso`` so existing accounts
don't get locked out at flip-time. Per docs/credits/design.md F13.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "055"
down_revision = "054"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.055")


_NEW_COLUMNS = (
    ("email_verified_at", lambda: sa.DateTime(), True),
    ("email_verification_token", lambda: sa.String(64), True),
    ("email_verification_sent_at", lambda: sa.DateTime(), True),
)


def upgrade() -> None:
    conn = op.get_bind()
    for col_name, col_type_factory, nullable in _NEW_COLUMNS:
        if not _column_exists(conn, "users", col_name):
            op.add_column("users", sa.Column(col_name, col_type_factory(), nullable=nullable))
            logger.info("[alembic.055] added users.%s", col_name)


def downgrade() -> None:
    conn = op.get_bind()
    for col_name, _, _ in reversed(_NEW_COLUMNS):
        if _column_exists(conn, "users", col_name):
            with op.batch_alter_table("users") as batch:
                batch.drop_column(col_name)


def _column_exists(conn, table: str, column: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        return column in {c["name"] for c in insp.get_columns(table)}
    except Exception:
        return False
