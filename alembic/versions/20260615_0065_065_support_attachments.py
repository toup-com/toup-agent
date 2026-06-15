"""Support card attachments: support_attachments (PLATFORM-only).

Revision ID: 065
Revises: 064
Create Date: 2026-06-15

Adds the PLATFORM-only support_attachments table backing screenshot uploads
from the mobile support-intake flow (see app/api/support.py +
docs/support-agent/README.md). Bytes live in the platform DB (the Railway
platform-api has ephemeral disk and the agent-side file_storage backend is
AGENT_ONLY), served only through an auth'd reporter/admin endpoint — never a
world-readable URL.

Idempotent + platform-DB-guarded (skips agent DBs). The table is in
PLATFORM_ONLY_TABLES, so init_db create_all already excludes it from agent
DBs and this migration short-circuits there too.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "065"
down_revision = "064"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.065")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    table_names = set(insp.get_table_names())

    # Platform DB only: user_sessions is a platform-only sentinel table.
    if "user_sessions" not in table_names:
        logger.info("[alembic.065] user_sessions absent; not a platform DB — skipping")
        return
    # Parent table from mig 064 must exist (same #198 lineage).
    if "support_issues" not in table_names:
        logger.info("[alembic.065] support_issues absent; mig 064 not applied — skipping")
        return

    if "support_attachments" not in table_names:
        op.create_table(
            "support_attachments",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "issue_id", sa.String(36),
                sa.ForeignKey("support_issues.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("kind", sa.String(24), nullable=False, server_default="screenshot"),
            sa.Column("mime_type", sa.String(64), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("sha256", sa.String(64), nullable=True),
            sa.Column("data", sa.LargeBinary(), nullable=False),
            sa.Column("uploaded_by_user_id", sa.String(36), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        )
        op.create_index("ix_support_attachments_issue_id", "support_attachments", ["issue_id"])
        op.create_index("ix_support_attachments_created_at", "support_attachments", ["created_at"])
        logger.info("[alembic.065] created support_attachments")
    else:
        logger.info("[alembic.065] support_attachments already present; skipping")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    table_names = set(insp.get_table_names())
    if "user_sessions" not in table_names:
        return
    if "support_attachments" in table_names:
        op.drop_table("support_attachments")
