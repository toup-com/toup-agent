"""messages: add nullable reply_to_message_id pointer + index

Revision ID: 049
Revises: 048
Create Date: 2026-05-19

First slice of the cross-channel reply-to feature. Adds a nullable
``reply_to_message_id`` column to ``messages`` plus an index so the
frontend can fetch a referenced message efficiently. No FK
constraint — the column is a soft pointer (same precedent as
``cron_jobs.migrated_to_routine_id``); a future bulk-delete of stale
day chats must not cascade through this column and wipe valid replies.

Purely additive. Reversible.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "049"
down_revision = "048"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.049")


def upgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "messages"):
        logger.info("[alembic.049] messages not present; skipping reply_to add")
        return
    if not _column_exists(conn, "messages", "reply_to_message_id"):
        op.add_column(
            "messages",
            sa.Column("reply_to_message_id", sa.String(50), nullable=True),
        )
        logger.info("[alembic.049] messages.reply_to_message_id added")
    else:
        logger.info(
            "[alembic.049] messages.reply_to_message_id already present; skipping"
        )

    if not _index_exists(conn, "messages", "ix_messages_reply_to"):
        op.create_index(
            "ix_messages_reply_to",
            "messages",
            ["reply_to_message_id"],
        )
        logger.info("[alembic.049] ix_messages_reply_to created")


def downgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "messages"):
        return
    if _index_exists(conn, "messages", "ix_messages_reply_to"):
        op.drop_index("ix_messages_reply_to", table_name="messages")
    if _column_exists(conn, "messages", "reply_to_message_id"):
        with op.batch_alter_table("messages") as batch:
            batch.drop_column("reply_to_message_id")
        logger.info("[alembic.049] messages.reply_to_message_id dropped")


def _table_exists(conn, table: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        return table in set(insp.get_table_names())
    except Exception:
        return False


def _column_exists(conn, table: str, column: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        cols = {c["name"] for c in insp.get_columns(table)}
        return column in cols
    except Exception:
        return False


def _index_exists(conn, table: str, name: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        return name in {ix["name"] for ix in insp.get_indexes(table)}
    except Exception:
        return False
