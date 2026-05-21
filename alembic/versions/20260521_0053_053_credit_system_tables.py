"""Credit-based billing system: subscription_plans, credit_balances,
credit_ledger, credit_reservations.

Revision ID: 053
Revises: 052
Create Date: 2026-05-21

The credit system replaces the paid-only access model with a
free-for-everyone, credit-based system (see docs/credits/design.md).

This migration is **purely additive** — no existing table is touched,
no existing row is mutated. The credit feature stays dormant until
``CREDIT_ENFORCEMENT_ENABLED=true`` in a later deploy.

Dialect notes
-------------
- JSONB on Postgres, JSON (TEXT) on SQLite via ``with_variant``.
- UUIDs stored as ``String(36)`` so SQLite (test/CI) and Postgres (prod)
  both work without dialect-specific type imports at the SQLAlchemy layer.
- Partial unique indexes (``WHERE idempotency_key IS NOT NULL``) are
  Postgres-only; on SQLite we fall back to a non-partial unique index
  which is functionally equivalent for the use case (NULL never equals
  itself in SQLite UNIQUE either).

Seed data
---------
Five rows are inserted into ``subscription_plans`` (free / starter /
builder / pro / elite) with the credit allocations from the design doc.
``stripe_price_id`` is left NULL — the operator runs
``app.scripts.create_stripe_credit_products`` to backfill, or sets
``STRIPE_PRICE_ID_*`` env vars and the platform boot syncs them.
"""
from __future__ import annotations

import logging
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "053"
down_revision = "052"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.053")


def _json_col() -> sa.types.TypeEngine:
    return sa.JSON().with_variant(JSONB(), "postgresql")


# Numbers from docs/credits/design.md §3. 1 credit = 1¢ underlying cost.
_PLAN_SEED = [
    {"id": "free", "display_name": "Free", "price_cents": 0,
     "message_credits_monthly": 30, "integration_credits_monthly": 120,
     "message_credits_daily_cap": 5,
     "rollover_message_credits": False, "rollover_integration_credits": False,
     "rollover_max_pct": 0, "sort_order": 0},
    {"id": "starter", "display_name": "Starter", "price_cents": 1600,
     "message_credits_monthly": 130, "integration_credits_monthly": 2500,
     "message_credits_daily_cap": None,
     "rollover_message_credits": False, "rollover_integration_credits": False,
     "rollover_max_pct": 0, "sort_order": 10},
    {"id": "builder", "display_name": "Builder", "price_cents": 4000,
     "message_credits_monthly": 320, "integration_credits_monthly": 12500,
     "message_credits_daily_cap": None,
     "rollover_message_credits": False, "rollover_integration_credits": False,
     "rollover_max_pct": 0, "sort_order": 20},
    {"id": "pro", "display_name": "Pro", "price_cents": 8000,
     "message_credits_monthly": 650, "integration_credits_monthly": 25000,
     "message_credits_daily_cap": None,
     "rollover_message_credits": False, "rollover_integration_credits": False,
     "rollover_max_pct": 0, "sort_order": 30},
    {"id": "elite", "display_name": "Elite", "price_cents": 16000,
     "message_credits_monthly": 1500, "integration_credits_monthly": 60000,
     "message_credits_daily_cap": None,
     "rollover_message_credits": True, "rollover_integration_credits": True,
     "rollover_max_pct": 50, "sort_order": 40},
]


def upgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name

    if not _table_exists(conn, "subscription_plans"):
        op.create_table(
            "subscription_plans",
            sa.Column("id", sa.String(40), primary_key=True),
            sa.Column("display_name", sa.String(60), nullable=False),
            sa.Column("price_cents", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("stripe_price_id", sa.String(120), nullable=True),
            sa.Column("message_credits_monthly", sa.Numeric(12, 2), nullable=False),
            sa.Column("integration_credits_monthly", sa.Numeric(12, 2), nullable=False),
            sa.Column("message_credits_daily_cap", sa.Numeric(12, 2), nullable=True),
            sa.Column("rollover_message_credits", sa.Boolean(), nullable=False,
                      server_default=sa.text("false") if dialect == "postgresql" else sa.text("0")),
            sa.Column("rollover_integration_credits", sa.Boolean(), nullable=False,
                      server_default=sa.text("false") if dialect == "postgresql" else sa.text("0")),
            sa.Column("rollover_max_pct", sa.Numeric(5, 2), nullable=False, server_default="0"),
            sa.Column("active", sa.Boolean(), nullable=False,
                      server_default=sa.text("true") if dialect == "postgresql" else sa.text("1")),
            sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        logger.info("[alembic.053] created subscription_plans")

    if not _table_exists(conn, "credit_balances"):
        op.create_table(
            "credit_balances",
            sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("plan_id", sa.String(40), sa.ForeignKey("subscription_plans.id"),
                      nullable=False, server_default="free"),
            sa.Column("message_credits_remaining", sa.Numeric(12, 2), nullable=False, server_default="0"),
            sa.Column("integration_credits_remaining", sa.Numeric(12, 2), nullable=False, server_default="0"),
            sa.Column("message_credits_used_today", sa.Numeric(12, 2), nullable=False, server_default="0"),
            sa.Column("message_credits_daily_cap", sa.Numeric(12, 2), nullable=True),
            sa.Column("day_anchor_local_date", sa.String(10), nullable=False),
            sa.Column("period_start", sa.DateTime(), nullable=False),
            sa.Column("period_end", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_credit_balances_period_end", "credit_balances", ["period_end"])
        logger.info("[alembic.053] created credit_balances")

    if not _table_exists(conn, "credit_ledger"):
        op.create_table(
            "credit_ledger",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("event_type", sa.String(60), nullable=False),
            sa.Column("bucket", sa.String(20), nullable=False),
            sa.Column("amount", sa.Numeric(12, 4), nullable=False),
            sa.Column("balance_after", sa.Numeric(12, 2), nullable=False),
            sa.Column("idempotency_key", sa.String(120), nullable=True),
            sa.Column("reservation_id", sa.String(36), nullable=True),
            sa.Column("event_id", sa.String(120), nullable=True),
            sa.Column("model", sa.String(80), nullable=True),
            sa.Column("provider", sa.String(30), nullable=True),
            sa.Column("input_tokens", sa.Integer(), nullable=True),
            sa.Column("output_tokens", sa.Integer(), nullable=True),
            sa.Column("underlying_cost_cents", sa.Numeric(12, 4), nullable=True),
            sa.Column("metadata", _json_col(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_credit_ledger_user_created", "credit_ledger", ["user_id", "created_at"])
        op.create_index("ix_credit_ledger_event_id", "credit_ledger", ["event_id"])
        if dialect == "postgresql":
            op.execute(
                "CREATE UNIQUE INDEX ix_credit_ledger_idempotency "
                "ON credit_ledger (user_id, idempotency_key) "
                "WHERE idempotency_key IS NOT NULL"
            )
        else:
            op.create_index("ix_credit_ledger_idempotency", "credit_ledger",
                            ["user_id", "idempotency_key"], unique=True)
        logger.info("[alembic.053] created credit_ledger")

    if not _table_exists(conn, "credit_reservations"):
        op.create_table(
            "credit_reservations",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("event_type", sa.String(60), nullable=False),
            sa.Column("bucket", sa.String(20), nullable=False),
            sa.Column("estimated_amount", sa.Numeric(12, 2), nullable=False),
            sa.Column("settled_amount", sa.Numeric(12, 4), nullable=True),
            sa.Column("status", sa.String(20), nullable=False, server_default="open"),
            sa.Column("idempotency_key", sa.String(120), nullable=True),
            sa.Column("event_id", sa.String(120), nullable=True),
            sa.Column("metadata", _json_col(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("settled_at", sa.DateTime(), nullable=True),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_credit_reservations_user_status", "credit_reservations", ["user_id", "status"])
        if dialect == "postgresql":
            op.execute(
                "CREATE UNIQUE INDEX ix_credit_reservations_idempotency "
                "ON credit_reservations (user_id, idempotency_key) "
                "WHERE idempotency_key IS NOT NULL"
            )
        else:
            op.create_index("ix_credit_reservations_idempotency", "credit_reservations",
                            ["user_id", "idempotency_key"], unique=True)
        logger.info("[alembic.053] created credit_reservations")

    existing = conn.execute(sa.text("SELECT COUNT(*) FROM subscription_plans")).scalar() or 0
    if existing == 0:
        now = datetime.utcnow()
        rows = [{**p, "active": True, "created_at": now} for p in _PLAN_SEED]
        op.bulk_insert(_subscription_plans_table(), rows)
        logger.info("[alembic.053] seeded %d subscription_plans rows", len(rows))


def downgrade() -> None:
    conn = op.get_bind()
    for table, indexes in (
        ("credit_reservations", ["ix_credit_reservations_idempotency", "ix_credit_reservations_user_status"]),
        ("credit_ledger", ["ix_credit_ledger_idempotency", "ix_credit_ledger_event_id", "ix_credit_ledger_user_created"]),
        ("credit_balances", ["ix_credit_balances_period_end"]),
        ("subscription_plans", []),
    ):
        if _table_exists(conn, table):
            for ix in indexes:
                try:
                    op.drop_index(ix, table_name=table)
                except Exception as e:
                    logger.warning("[alembic.053] drop_index %s skipped: %s", ix, e)
            op.drop_table(table)


def _table_exists(conn, table: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        return table in insp.get_table_names()
    except Exception:
        return False


def _subscription_plans_table() -> sa.Table:
    return sa.Table(
        "subscription_plans", sa.MetaData(),
        sa.Column("id", sa.String(40)),
        sa.Column("display_name", sa.String(60)),
        sa.Column("price_cents", sa.Integer),
        sa.Column("stripe_price_id", sa.String(120)),
        sa.Column("message_credits_monthly", sa.Numeric(12, 2)),
        sa.Column("integration_credits_monthly", sa.Numeric(12, 2)),
        sa.Column("message_credits_daily_cap", sa.Numeric(12, 2)),
        sa.Column("rollover_message_credits", sa.Boolean),
        sa.Column("rollover_integration_credits", sa.Boolean),
        sa.Column("rollover_max_pct", sa.Numeric(5, 2)),
        sa.Column("active", sa.Boolean),
        sa.Column("sort_order", sa.Integer),
        sa.Column("created_at", sa.DateTime),
    )
