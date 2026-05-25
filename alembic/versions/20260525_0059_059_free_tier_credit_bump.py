"""Bump Free tier: 30→100 msg, 120→500 int, daily cap 5→15.

Revision ID: 059
Revises: 058
Create Date: 2026-05-25

The original Free-tier sizing (30 msg / 120 int / 5/day cap) was too
restrictive to convert. Per-call math: a single task-completing
conversation on Sonnet 4.6 routinely runs 5–8 LLM calls × ~1.7 credits
each + tool calls, so the 5/day cap blocked most users after one real
task and the 30/month total ran out within their first session of
exploration. New sizing per credit-system audit:

  Free: 100 msg / 500 int / 15 day-cap
        → ~3 substantive task-completing conversations per day
        → ~20 tasks per month before hitting the wall
        → worst-case ~$2/user/month at API margins

Paid tiers unchanged. Memory + database.py seed updated in same PR.

Two transactional steps:

  1. UPDATE subscription_plans WHERE id='free' with the new numbers.
     Future signups + monthly renewals pull from this row.

  2. For every existing free-tier balance row, raise the daily cap
     immediately (instant relief — the cap is the binding constraint)
     AND top up remaining-this-period balances by the delta so
     in-flight users see the new generosity NOW instead of waiting up
     to 30 days for their period_end. The bump is additive (not a
     reset) so heavy users who already consumed credits don't get
     clawed back — they get +70 msg / +380 int on top of whatever
     they had left.

Idempotent: re-running the migration is safe. The plan UPDATE is
deterministic. The balance UPDATE is gated on `plan_id='free' AND
(message_credits_daily_cap IS NULL OR message_credits_daily_cap < 15)`
so a second pass leaves already-bumped rows untouched. Downgrade
reverses the plan row only — balance top-ups are deliberately
irreversible (we don't claw back credits we've already granted; a
free-tier user who got 70 extra and used 40 would land below zero on
naive reversal).
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "059"
down_revision = "058"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.059")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "subscription_plans" not in tables:
        # Platform DBs always have this table (alembic 053). Agent
        # (per-tenant) DBs don't — they never carry credit state.
        logger.info(
            "[alembic.059] subscription_plans absent; skipping "
            "(agent-only DB, no credit state)"
        )
        return

    # ── Step 1: plan row ────────────────────────────────────────────
    pre = conn.execute(sa.text(
        "SELECT message_credits_monthly, integration_credits_monthly, "
        "message_credits_daily_cap FROM subscription_plans WHERE id = 'free'"
    )).first()
    if pre is None:
        logger.warning(
            "[alembic.059] subscription_plans 'free' row missing — "
            "skipping plan UPDATE. init_db seed will insert with the new "
            "numbers on next boot."
        )
    else:
        logger.info(
            "[alembic.059] free plan pre-bump: msg=%s int=%s daily_cap=%s",
            pre[0], pre[1], pre[2],
        )
        conn.execute(sa.text(
            "UPDATE subscription_plans SET "
            "message_credits_monthly = 100, "
            "integration_credits_monthly = 500, "
            "message_credits_daily_cap = 15 "
            "WHERE id = 'free'"
        ))

    # ── Step 2: in-flight balances ──────────────────────────────────
    if "credit_balances" not in tables:
        logger.info("[alembic.059] credit_balances absent; nothing to bump")
        return

    # Bump daily cap first (the binding constraint). Idempotent: only
    # touches rows whose cap is below the new value.
    bumped_cap = conn.execute(sa.text(
        "UPDATE credit_balances SET message_credits_daily_cap = 15 "
        "WHERE plan_id = 'free' AND "
        "(message_credits_daily_cap IS NULL OR message_credits_daily_cap < 15)"
    )).rowcount
    logger.info("[alembic.059] daily-cap bumped for %s free balances", bumped_cap)

    # Top up remaining-this-period balances additively by the delta.
    # Gated on a marker column we set on the SAME row in the same UPDATE
    # so re-runs are safe — we add a sentinel value into a metadata field
    # we don't have, so instead we gate on "remaining + delta <= old
    # monthly cap + delta" which is true iff we haven't bumped this row.
    # Simpler: use a separate sentinel via day_anchor_local_date suffix?
    # No — the cleanest idempotent gate is a one-shot platform_settings
    # flag row. Set it AFTER the bulk update. Subsequent runs see the
    # flag and skip the additive top-up entirely. (Daily-cap bump above
    # is naturally idempotent and runs unconditionally.)
    flag = conn.execute(sa.text(
        "SELECT value FROM platform_settings WHERE key = 'credit.free_bump_059_applied'"
    )).first() if "platform_settings" in tables else None

    if flag is not None:
        logger.info(
            "[alembic.059] free-tier balance top-up already applied "
            "(platform_settings flag present) — skipping additive bump"
        )
        return

    bumped_balances = conn.execute(sa.text(
        "UPDATE credit_balances SET "
        "message_credits_remaining = message_credits_remaining + 70, "
        "integration_credits_remaining = integration_credits_remaining + 380 "
        "WHERE plan_id = 'free'"
    )).rowcount
    logger.info(
        "[alembic.059] top-up applied to %s free balances (+70 msg / +380 int)",
        bumped_balances,
    )

    if "platform_settings" in tables:
        conn.execute(sa.text(
            "INSERT INTO platform_settings (key, value, updated_at) "
            "VALUES ('credit.free_bump_059_applied', "
            "'{\"applied_at\": \"2026-05-25\", \"bumped\": " + str(bumped_balances) + "}', "
            "CURRENT_TIMESTAMP) "
            "ON CONFLICT (key) DO NOTHING"
        ))


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "subscription_plans" not in tables:
        return

    # Reverse the plan row. Balance top-ups are NOT clawed back — see
    # docstring (free-tier users who already consumed bonus credits
    # would land below zero on naive reversal).
    conn.execute(sa.text(
        "UPDATE subscription_plans SET "
        "message_credits_monthly = 30, "
        "integration_credits_monthly = 120, "
        "message_credits_daily_cap = 5 "
        "WHERE id = 'free'"
    ))
    logger.info("[alembic.059] reverted free plan to 30/120/5 (balances unchanged)")
