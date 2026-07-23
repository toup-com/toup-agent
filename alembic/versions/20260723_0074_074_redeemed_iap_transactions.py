"""Global consumable-IAP replay guard — redeemed_iap_transactions.

Revision ID: 074
Revises: 073
Create Date: 2026-07-23

Security round 12 (docs/security/audit-2026.md). A consumable StoreKit / Play
transaction must be redeemable by EXACTLY ONE account, ever. The credit_ledger
idempotency index is per-user ``(user_id, idempotency_key)``, so a farmed second
account could POST /api/iap/apple/verify with the SAME real transaction id and
mint credits again — one purchase = unlimited credits across accounts. The
subscription path already guards this globally (``uq_apple_sub_orig_txn`` on
``original_transaction_id`` alone); the consumable path was the missed
safeguard.

This table's ``transaction_id`` PRIMARY KEY enforces GLOBAL uniqueness: the
grant path inserts a row before crediting, so a cross-account replay hits the PK
and is refused. No FK to ``users`` so it's safe to create on any DB partition;
guarded like 070-073 (platform DBs only). The init_db ``_alter_statements``
mirror carries the same ``CREATE TABLE IF NOT EXISTS`` as the authoritative heal
(the Dockerfile ``alembic upgrade head`` is best-effort).
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "074"
down_revision = "073"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.074")

_TABLE = "redeemed_iap_transactions"


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.074] users absent; skipping (not a platform DB)")
        return
    if _TABLE in tables:
        logger.info("[alembic.074] %s already present", _TABLE)
        return

    op.create_table(
        _TABLE,
        sa.Column("transaction_id", sa.String(120), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("platform", sa.String(16), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    logger.info("[alembic.074] created %s", _TABLE)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if _TABLE in set(insp.get_table_names()):
        op.drop_table(_TABLE)
