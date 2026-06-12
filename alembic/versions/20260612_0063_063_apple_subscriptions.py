"""Apple auto-renewable subscriptions: plan_source + apple_subscriptions table.

Revision ID: 063
Revises: 062
Create Date: 2026-06-12

Adds the backend half of Apple IAP subscriptions:

* ``credit_balances.plan_source`` — nullable String(16), NO server_default so
  every existing row backfills to SQL NULL. Semantics: ``NULL`` or ``'stripe'``
  ⇒ time-driven renewal allowed (today's behaviour); ``'apple'`` ⇒ renewal is
  webhook-driven only (the §4a renewal guard skips clock renewals for these).
  Critically NULL — NOT ``'stripe'`` — as the backfill value, so the change is
  provably inert for existing Stripe/free users (the renewal guard treats NULL
  identically to today).
* ``apple_subscriptions`` — one row per Apple original transaction, the
  lifecycle mirror keyed on ``original_transaction_id`` (UNIQUE). Every V2
  notification and every verify lands on the same row.

Idempotent + guarded in the exact style of 062: the whole migration is a no-op
when ``credit_balances`` is absent (a per-tenant agent DB), and each piece is
add-only-if-missing so re-runs are safe. ``downgrade()`` drops in reverse,
guarded.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "063"
down_revision = "062"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.063")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    table_names = set(insp.get_table_names())
    if "credit_balances" not in table_names:
        logger.info("[alembic.063] credit_balances table absent; skipping (not a platform DB)")
        return

    # (a) credit_balances.plan_source — nullable, NO server_default (backfills NULL).
    existing = {c["name"] for c in insp.get_columns("credit_balances")}
    if "plan_source" not in existing:
        op.add_column(
            "credit_balances",
            sa.Column("plan_source", sa.String(16), nullable=True),
        )
        logger.info("[alembic.063] added credit_balances.plan_source")
    else:
        logger.info("[alembic.063] credit_balances.plan_source already present; skipping")

    # (b) apple_subscriptions — lifecycle mirror, one row per Apple orig txn.
    if "apple_subscriptions" not in table_names:
        op.create_table(
            "apple_subscriptions",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id", sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("original_transaction_id", sa.String(64), nullable=False),
            sa.Column("product_id", sa.String(128), nullable=False),
            sa.Column("plan_id", sa.String(40), nullable=False),
            sa.Column("status", sa.String(24), nullable=False),
            sa.Column("expires_date", sa.DateTime, nullable=True),
            sa.Column(
                "auto_renew_status", sa.Boolean, nullable=False,
                server_default=sa.text("true"),
            ),
            sa.Column("auto_renew_product_id", sa.String(128), nullable=True),
            sa.Column("environment", sa.String(16), nullable=False),
            sa.Column("last_notification_uuid", sa.String(64), nullable=True),
            sa.Column(
                "created_at", sa.DateTime, nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at", sa.DateTime, nullable=False,
                server_default=sa.text("now()"),
            ),
        )
        op.create_unique_constraint(
            "uq_apple_sub_orig_txn", "apple_subscriptions",
            ["original_transaction_id"],
        )
        op.create_index("ix_apple_sub_user", "apple_subscriptions", ["user_id"])
        logger.info("[alembic.063] created apple_subscriptions table")
    else:
        logger.info("[alembic.063] apple_subscriptions table already present; skipping")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    table_names = set(insp.get_table_names())
    if "credit_balances" not in table_names:
        return

    if "apple_subscriptions" in table_names:
        # Indexes / constraints drop with the table on every backend we target.
        op.drop_table("apple_subscriptions")

    existing = {c["name"] for c in insp.get_columns("credit_balances")}
    if "plan_source" in existing:
        op.drop_column("credit_balances", "plan_source")
