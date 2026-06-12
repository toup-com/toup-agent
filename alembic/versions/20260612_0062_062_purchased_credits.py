"""Add credit_balances.purchased_credits_remaining for StoreKit IAP credits.

Revision ID: 062
Revises: 061
Create Date: 2026-06-12

Non-expiring wallet of credits bought via StoreKit consumable credit packs.
Spent on the MESSAGE bucket after plan credits, bypasses the daily cap, and
survives monthly renewal. server_default="0" so every existing row backfills
to a zero purchased wallet — the change is provably inert until the first IAP
grant lands.

Idempotent + guarded: the column is only added if the `credit_balances` table
exists and the column is absent, so re-runs and per-tenant agent DBs (which
have no `credit_balances` table) are safe.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "062"
down_revision = "061"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.062")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if "credit_balances" not in set(insp.get_table_names()):
        logger.info("[alembic.062] credit_balances table absent; skipping (not a platform DB)")
        return

    existing = {c["name"] for c in insp.get_columns("credit_balances")}
    if "purchased_credits_remaining" not in existing:
        op.add_column(
            "credit_balances",
            sa.Column(
                "purchased_credits_remaining",
                sa.Numeric(12, 2),
                nullable=False,
                server_default="0",
            ),
        )
        logger.info("[alembic.062] added credit_balances.purchased_credits_remaining")
    else:
        logger.info("[alembic.062] credit_balances.purchased_credits_remaining already present; skipping")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if "credit_balances" not in set(insp.get_table_names()):
        return

    existing = {c["name"] for c in insp.get_columns("credit_balances")}
    if "purchased_credits_remaining" in existing:
        op.drop_column("credit_balances", "purchased_credits_remaining")
