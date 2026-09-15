"""104 — the Unlimited plan row, the grandfather columns, and the free daily-cap drift.

Revision ID: 104
Revises: 103
Create Date: 2026-09-13

Schema + catalogue only. **This migration moves nobody's plan.** The
grandfather of the three legacy Apple payers is an admin command
(``app.scripts.grandfather_unlimited``), not a migration — see
``docs/billing/UNLIMITED_ENTITLEMENT_DESIGN.md`` §9.3. The short version:
a migration runs unattended in every environment with no operator present,
merging platform ``main`` is a deploy AND a fleet rollout, the predicate
depends on live Apple state (``expires_date > now()``) rather than on
schema, and a reversible grandfather needs a receipt of prior values —
which makes it a command regardless.

Three steps, each independently guarded and idempotent:

  1. INSERT the ``unlimited`` subscription_plans row (ON CONFLICT DO
     NOTHING). 1,000,000 / 1,000,000 credits, no daily cap, no rollover.
     Price from ``app.db.plan_catalog.UNLIMITED_PRICE_CENTS``.

  2. Correct the ``free`` row's daily cap 15 → NULL. Production is ALREADY
     NULL (all 5 plan rows, all 84 balance rows) after the 2026-08-29 cap
     removal, so this UPDATE matches zero rows in production and changes
     nothing for a single live user. It exists because ``database.py``'s
     seed still writes 15, so a fresh environment or a rebuilt DB silently
     reimposes a cap that a single gpt-5.5 turn (26–28 credits quoted)
     can never satisfy — the exact configuration behind the 2026-08-03
     incident. Guarded on ``message_credits_daily_cap = 15`` and marked in
     platform_settings so downgrade() only reverses what upgrade() did.

  3. Two nullable columns on ``apple_subscriptions``: ``grandfathered_at``
     and ``grandfathered_product_id``. Grandfathering is a property of a
     SUBSCRIPTION (three specific original_transaction_ids), not of a
     PRODUCT, so it is recorded here and read by exactly one resolver,
     ``apple_iap_service.plan_for_subscription``. Overloading
     APPLE_SUB_PRODUCT_TO_PLAN instead would hand Unlimited to every
     future holder of a legacy product id — including a post-cutover
     cross-grade down to $9.90 Starter — permanently and unlogged.

Alembic 053 is NOT rewritten: applied history is never edited, 104
converges instead.

The ``unlimited_grants`` table (the admin override, design §4) is
deliberately NOT created here — see the HOOK note at the bottom of
upgrade().
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "104"
down_revision = "103"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.104")


# A migration must survive a future refactor of app/, so the catalogue import
# is optional and the fallback is loud rather than silent.
try:  # pragma: no cover - import shape, not logic
    from app.db.plan_catalog import (
        UNLIMITED_INTEGRATION_CREDITS_MONTHLY,
        UNLIMITED_MESSAGE_CREDITS_MONTHLY,
        UNLIMITED_PLAN_DISPLAY_NAME,
        UNLIMITED_PLAN_ID,
        UNLIMITED_PRICE_CENTS,
        UNLIMITED_SORT_ORDER,
    )
    _CATALOG_SOURCE = "app.db.plan_catalog"
except ImportError:  # pragma: no cover - defensive
    UNLIMITED_PLAN_ID = "unlimited"
    UNLIMITED_PLAN_DISPLAY_NAME = "Unlimited"
    UNLIMITED_PRICE_CENTS = 1895
    UNLIMITED_MESSAGE_CREDITS_MONTHLY = 1_000_000
    UNLIMITED_INTEGRATION_CREDITS_MONTHLY = 1_000_000
    UNLIMITED_SORT_ORDER = 5
    _CATALOG_SOURCE = "LITERAL FALLBACK"


# Marker so downgrade() reverses only the drift THIS upgrade corrected.
_FREE_CAP_MARKER = "credit.free_daily_cap_null_104_applied"


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    tables = set(insp.get_table_names())

    if "subscription_plans" not in tables:
        # Platform DBs always have this table (alembic 053). Agent
        # (per-tenant) DBs don't — they never carry credit state.
        logger.info(
            "[alembic.104] subscription_plans absent; skipping "
            "(agent-only DB, no credit state)"
        )
        return

    if _CATALOG_SOURCE != "app.db.plan_catalog":
        logger.warning(
            "[alembic.104] app.db.plan_catalog NOT importable — using literal "
            "fallback values. Verify the price against App Store Connect."
        )

    # ── Pre-read: the deploy log carries the before-state ────────────
    for row in conn.execute(sa.text(
        "SELECT id, price_cents, message_credits_monthly, "
        "integration_credits_monthly, message_credits_daily_cap, active "
        "FROM subscription_plans ORDER BY sort_order"
    )).fetchall():
        logger.info(
            "[alembic.104] pre: plan=%s price=%s msg=%s int=%s daily_cap=%s active=%s",
            row[0], row[1], row[2], row[3], row[4], row[5],
        )

    # ── Step 1: the Unlimited plan row, INACTIVE ─────────────────────
    #
    # `active=false` is not a placeholder — it is what keeps this migration's
    # promise that it moves nobody's plan and changes nothing a user sees.
    # `GET /api/billing/plans` is the PUBLIC, unauthenticated pricing catalogue
    # and filters on exactly `active = true` (credits.py). Inserting an active
    # row publishes, unattended on the deploy that lands this branch, a sixth
    # card to every anonymous visitor — at UNLIMITED_PRICE_CENTS, which
    # plan_catalog's own comment says is a PLACEHOLDER the founder has not
    # chosen; sorted AHEAD of Free, because PricingPage's PLAN_ORDER has no
    # 'unlimited' so indexOf returns -1; in a grid declared lg:grid-cols-5;
    # and with an Upgrade button that dead-ends on "Stripe price not
    # configured", because there is no Stripe price and deliberately never
    # will be.
    #
    # Nothing internal is affected: `active` has exactly ONE reader in the
    # whole backend (that public catalogue). Every resolution path — every
    # apply_plan_change, _apply_plan_change_absolute, _renew_period_unguarded
    # and /credits/status lookup — is `db.get(SubscriptionPlan, id)` by primary
    # key with no filter, which is precisely the property the cutover relies
    # on to retire the four legacy tiers. Merchandising Unlimited is step 9 of
    # the rollout, a one-line UPDATE by an operator, after the app build ships.
    #
    # rollover_message_credits / rollover_integration_credits / active are
    # BOOLEAN. Use false/true literals — Postgres refuses to coerce integer
    # 0/1 to bool here ("column ... is of type boolean but expression is of
    # type integer"). Previous literal regression: PR #119.
    conn.execute(
        sa.text(
            "INSERT INTO subscription_plans "
            "(id, display_name, price_cents, message_credits_monthly, "
            " integration_credits_monthly, message_credits_daily_cap, "
            " rollover_message_credits, rollover_integration_credits, "
            " rollover_max_pct, sort_order, active, created_at) "
            "VALUES (:id, :name, :price, :msg, :int, NULL, "
            "        false, false, 0, :sort, false, CURRENT_TIMESTAMP) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {
            "id": UNLIMITED_PLAN_ID,
            "name": UNLIMITED_PLAN_DISPLAY_NAME,
            "price": UNLIMITED_PRICE_CENTS,
            "msg": UNLIMITED_MESSAGE_CREDITS_MONTHLY,
            "int": UNLIMITED_INTEGRATION_CREDITS_MONTHLY,
            "sort": UNLIMITED_SORT_ORDER,
        },
    )
    logger.info(
        "[alembic.104] '%s' plan present: price=%s msg=%s int=%s (source=%s)",
        UNLIMITED_PLAN_ID, UNLIMITED_PRICE_CENTS,
        UNLIMITED_MESSAGE_CREDITS_MONTHLY, UNLIMITED_INTEGRATION_CREDITS_MONTHLY,
        _CATALOG_SOURCE,
    )

    # ── Step 2: the free daily-cap drift ─────────────────────────────
    pre_free = conn.execute(sa.text(
        "SELECT message_credits_daily_cap FROM subscription_plans WHERE id = 'free'"
    )).first()
    if pre_free is None:
        logger.warning(
            "[alembic.104] subscription_plans 'free' row missing — skipping the "
            "daily-cap convergence. init_db's seed inserts it with cap NULL."
        )
    else:
        logger.info("[alembic.104] free plan pre-fix daily_cap=%s", pre_free[0])
        fixed = conn.execute(sa.text(
            "UPDATE subscription_plans SET message_credits_daily_cap = NULL "
            "WHERE id = 'free' AND message_credits_daily_cap = 15"
        )).rowcount
        logger.info(
            "[alembic.104] free daily-cap drift corrected on %s row(s) "
            "(0 expected in production — already NULL since 2026-08-29)", fixed,
        )
        if fixed and "platform_settings" in tables:
            conn.execute(sa.text(
                "INSERT INTO platform_settings (key, value, updated_at) "
                "VALUES (:k, '{\"from\": 15, \"to\": null}', CURRENT_TIMESTAMP) "
                "ON CONFLICT (key) DO NOTHING"
            ), {"k": _FREE_CAP_MARKER})

    # ── Step 3: the grandfather columns on apple_subscriptions ───────
    if "apple_subscriptions" not in tables:
        logger.info(
            "[alembic.104] apple_subscriptions absent; skipping grandfather columns"
        )
    else:
        existing = {c["name"] for c in insp.get_columns("apple_subscriptions")}
        if "grandfathered_at" not in existing:
            op.add_column("apple_subscriptions", sa.Column(
                "grandfathered_at", sa.DateTime(), nullable=True,
            ))
            logger.info("[alembic.104] apple_subscriptions.grandfathered_at added")
        if "grandfathered_product_id" not in existing:
            op.add_column("apple_subscriptions", sa.Column(
                "grandfathered_product_id", sa.String(128), nullable=True,
            ))
            logger.info(
                "[alembic.104] apple_subscriptions.grandfathered_product_id added"
            )

    # ── HOOK: the admin-override table ───────────────────────────────
    # `unlimited_grants` (design §4) is created by its own step so the
    # override work can land independently of the catalogue. Nothing here
    # depends on it: entitlement.py degrades safely when the table is
    # absent, and the charge path never reads it at all.


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    tables = set(insp.get_table_names())

    if "subscription_plans" not in tables:
        return

    # ── Refuse BEFORE touching anything, not halfway down ────────────
    #
    # The three reversals below used to run in order with the only guard at
    # the bottom, on step 1. So a downgrade against a grandfathered production
    # DB dropped `grandfathered_at` / `grandfathered_product_id` for every row
    # FIRST, and only then declined to remove the plan row those accounts are
    # still on. The accounts kept plan_id='unlimited' — deliberately — while
    # the only record of WHY was gone and unrecoverable: the grandfather
    # receipt stores the state BEFORE the stamp (`prior_grandfathered_at:
    # null`), not the stamp. Re-upgrading re-adds the columns as NULL, and on
    # the next legacy DID_RENEW `plan_for_subscription` reads a NULL stamp,
    # returns 'builder', and silently reverts the grandfather — the exact
    # regression cc3b01f5 exists to prevent. Meanwhile the ORM still maps both
    # columns, so every `select(AppleSubscription)` raises UndefinedColumn
    # until the code is rolled back too.
    #
    # Dropping a column that encodes a live entitlement is deleting billing
    # state. Migration 105 already refuses to drop a non-empty audit trail;
    # this is the same rule, and it must be checked before the first DDL.
    stamped = 0
    if "apple_subscriptions" in tables:
        cols = {c["name"] for c in insp.get_columns("apple_subscriptions")}
        if "grandfathered_at" in cols:
            stamped = conn.execute(sa.text(
                "SELECT COUNT(*) FROM apple_subscriptions "
                "WHERE grandfathered_at IS NOT NULL"
            )).scalar() or 0
    held = 0
    if "credit_balances" in tables:
        held = conn.execute(
            sa.text("SELECT COUNT(*) FROM credit_balances WHERE plan_id = :p"),
            {"p": UNLIMITED_PLAN_ID},
        ).scalar() or 0
    if stamped or held:
        raise RuntimeError(
            f"[alembic.104] refusing to downgrade: {stamped} grandfathered "
            f"apple_subscriptions row(s) and {held} credit_balances row(s) on "
            f"'{UNLIMITED_PLAN_ID}'. Dropping the grandfather columns would "
            f"destroy the only record of why those accounts are entitled, and "
            f"the next legacy DID_RENEW would silently revert them to their "
            f"old tier. Run `python -m app.scripts.grandfather_unlimited "
            f"--revert <receipt> --apply` first, then downgrade."
        )

    # ── Step 3 reversed: the grandfather columns ─────────────────────
    if "apple_subscriptions" in tables:
        existing = {c["name"] for c in insp.get_columns("apple_subscriptions")}
        for name in ("grandfathered_product_id", "grandfathered_at"):
            if name in existing:
                op.drop_column("apple_subscriptions", name)
        logger.info("[alembic.104] grandfather columns dropped")

    # ── Step 2 reversed: ONLY if this upgrade actually changed it ────
    if "platform_settings" in tables:
        marker = conn.execute(
            sa.text("SELECT key FROM platform_settings WHERE key = :k"),
            {"k": _FREE_CAP_MARKER},
        ).first()
        if marker is not None:
            conn.execute(sa.text(
                "UPDATE subscription_plans SET message_credits_daily_cap = 15 "
                "WHERE id = 'free' AND message_credits_daily_cap IS NULL"
            ))
            conn.execute(
                sa.text("DELETE FROM platform_settings WHERE key = :k"),
                {"k": _FREE_CAP_MARKER},
            )
            logger.info("[alembic.104] free daily-cap restored to 15")
        else:
            logger.info(
                "[alembic.104] no free daily-cap marker — leaving the cap as-is "
                "(this upgrade did not change it)"
            )

    # ── Step 1 reversed ──────────────────────────────────────────────
    # `held` was counted at the top and is zero, or we never got here.
    conn.execute(
        sa.text("DELETE FROM subscription_plans WHERE id = :p"),
        {"p": UNLIMITED_PLAN_ID},
    )
    logger.info("[alembic.104] '%s' plan row removed", UNLIMITED_PLAN_ID)
