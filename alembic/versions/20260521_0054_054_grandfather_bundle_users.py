"""Grandfather existing LLM-bundle subscribers into the credit Builder tier.

Revision ID: 054
Revises: 053
Create Date: 2026-05-21

Every user with ``agent_configs.bundle_status='active'`` or
``'cancelling'`` was on the $40/mo LLM bundle before the credit-billing
rollout. We grandfather them into ``plan_id='builder'`` (the closest
matching tier in the new ladder — $40/mo, 320 message + 12,500
integration credits/mo).

Idempotent: re-runs are safe; a user already on a non-free plan is
skipped, otherwise their balance is overwritten with builder defaults.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

import sqlalchemy as sa
from alembic import op


revision = "054"
# Linearized at credit-system merge: 052 → 053 → 053a → 054 → 055.
down_revision = "053a"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.054")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    tables = set(insp.get_table_names())
    if not {"subscription_plans", "credit_balances", "credit_ledger", "agent_configs"}.issubset(tables):
        logger.warning("[alembic.054] required tables missing; skipping grandfather")
        return

    builder_row = conn.execute(sa.text(
        "SELECT message_credits_monthly, integration_credits_monthly, "
        "message_credits_daily_cap FROM subscription_plans WHERE id = 'builder'"
    )).first()
    if builder_row is None:
        logger.warning("[alembic.054] no 'builder' plan row; aborting")
        return
    msg_monthly, int_monthly, daily_cap = builder_row

    rows = conn.execute(sa.text(
        "SELECT ac.user_id FROM agent_configs ac "
        "WHERE ac.bundle_status IN ('active', 'cancelling') AND NOT EXISTS ("
        "  SELECT 1 FROM credit_balances cb "
        "  WHERE cb.user_id = ac.user_id AND cb.plan_id != 'free')"
    )).fetchall()
    user_ids = [r[0] for r in rows if r and r[0]]
    if not user_ids:
        logger.info("[alembic.054] no bundle users to grandfather")
        return

    logger.info("[alembic.054] grandfathering %d bundle users → builder tier", len(user_ids))
    now = datetime.utcnow()
    period_end = now + timedelta(days=30)

    for user_id in user_ids:
        tz_row = conn.execute(sa.text("SELECT timezone FROM users WHERE id = :uid"), {"uid": user_id}).first()
        user_tz = (tz_row[0] if tz_row else None) or "UTC"
        try:
            from zoneinfo import ZoneInfo
            anchor = datetime.now(ZoneInfo(user_tz)).date().isoformat()
        except Exception:
            anchor = now.date().isoformat()

        existing = conn.execute(sa.text("SELECT plan_id FROM credit_balances WHERE user_id = :uid"),
                                {"uid": user_id}).first()
        if existing is None:
            conn.execute(sa.text(
                "INSERT INTO credit_balances (user_id, plan_id, message_credits_remaining, "
                "integration_credits_remaining, message_credits_used_today, message_credits_daily_cap, "
                "day_anchor_local_date, period_start, period_end, updated_at) VALUES "
                "(:uid, 'builder', :msg, :int_credits, 0, :daily_cap, :anchor, :now, :pend, :now)"
            ), {"uid": user_id, "msg": msg_monthly, "int_credits": int_monthly,
                "daily_cap": daily_cap, "anchor": anchor, "now": now, "pend": period_end})
        else:
            conn.execute(sa.text(
                "UPDATE credit_balances SET plan_id='builder', message_credits_remaining=:msg, "
                "integration_credits_remaining=:int_credits, message_credits_daily_cap=:daily_cap, "
                "updated_at=:now WHERE user_id=:uid"
            ), {"uid": user_id, "msg": msg_monthly, "int_credits": int_monthly,
                "daily_cap": daily_cap, "now": now})

        for bucket, amount in (("message", msg_monthly), ("integration", int_monthly)):
            conn.execute(sa.text(
                "INSERT INTO credit_ledger (id, user_id, event_type, bucket, amount, balance_after, "
                "metadata, created_at) VALUES (:id, :uid, 'plan_grant', :bucket, :amt, :amt, :meta, :now)"
            ), {"id": str(uuid.uuid4()), "uid": user_id, "bucket": bucket, "amt": amount,
                "meta": '{"reason": "grandfather_from_bundle", "via": "alembic_054"}', "now": now})

    logger.info("[alembic.054] grandfathered %d users", len(user_ids))


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    if "credit_ledger" not in insp.get_table_names():
        return
    marker = '"via": "alembic_054"'
    rows = conn.execute(sa.text(
        f"SELECT DISTINCT user_id FROM credit_ledger WHERE CAST(metadata AS TEXT) LIKE '%{marker}%'"
    )).fetchall()
    user_ids = [r[0] for r in rows if r and r[0]]
    if not user_ids:
        return
    free_row = conn.execute(sa.text(
        "SELECT message_credits_monthly, integration_credits_monthly, message_credits_daily_cap "
        "FROM subscription_plans WHERE id = 'free'"
    )).first()
    if free_row:
        msg, int_credits, daily_cap = free_row
        for uid in user_ids:
            conn.execute(sa.text(
                "UPDATE credit_balances SET plan_id='free', message_credits_remaining=:msg, "
                "integration_credits_remaining=:int_credits, message_credits_daily_cap=:daily_cap "
                "WHERE user_id=:uid AND plan_id='builder'"
            ), {"uid": uid, "msg": msg, "int_credits": int_credits, "daily_cap": daily_cap})
    conn.execute(sa.text(
        f"DELETE FROM credit_ledger WHERE CAST(metadata AS TEXT) LIKE '%{marker}%'"
    ))
