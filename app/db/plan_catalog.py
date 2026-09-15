"""The Unlimited plan's numbers, in exactly one place.

A leaf module by design: it imports nothing from ``app`` and nothing from
SQLAlchemy, so the three sites that must agree on these values can all read
them without a cycle and without dragging the application graph into an
alembic run —

  * ``app/db/database.py``'s ``init_db`` seed (fresh envs, CI, tenant DBs)
  * ``backend/alembic/versions/…_102_unlimited_plan.py``
  * ``app/services/apple_iap_service.py`` (the product → plan map)

It exists because the credit catalogue has five hand-kept mirrors and no
guard comparing them, and the two DB seeds have already drifted once (053
seeds free at 30/120/5, 059 bumped it to 100/500/15, production is
100/500/NULL). One leaf module read by every seed site is the smallest thing
that makes "one edit sets the price everywhere" actually true.
"""

from __future__ import annotations

# CAD 18.95, decided 2026-09-13. 18.97 was asked for and does not exist:
# Apple sells only fixed price points, and the Canadian grid runs
# X.00 / X.49 / X.90 / X.95 / X.99 (verified live against
# /v1/subscriptions/{id}/pricePoints?filter[territory]=CAN, where 18.95
# returns proceeds 13.27).
#
# The unit here is CENTS OF CANADIAN DOLLARS, not USD. Canada is the base
# storefront and Apple derives the other 174 from it (US lands at 14.95).
# Every account that has ever paid Toup is on the CAN storefront, so the
# number a user is quoted and the number we store are the same currency.
# It MUST equal the App Store Connect price exactly, or the paywall quotes
# a price Apple will not charge.
UNLIMITED_PRICE_CENTS = 1895

UNLIMITED_PLAN_ID = "unlimited"
UNLIMITED_PLAN_DISPLAY_NAME = "Unlimited"
UNLIMITED_SUB_PRODUCT_ID = "ai.toup.app.sub.unlimited"

# 1 credit = 1 cent of underlying provider cost, so 1,000,000 credits is
# $10,000/month of provider spend: ~500× the entire platform's current
# 30-day spend (2,003 credits) and ~3,100× the largest plan ever sold
# (Builder, 320). A human cannot reach it. It is finite rather than
# effectively-infinite on purpose: a runaway loop leaves a visible
# fingerprint (used_this_period finally moves) instead of an unbounded bill,
# and Numeric(12,2) tops out at 9,999,999,999.99 so this is four orders of
# magnitude clear of overflow.
UNLIMITED_MESSAGE_CREDITS_MONTHLY = 1_000_000
UNLIMITED_INTEGRATION_CREDITS_MONTHLY = 1_000_000

# Immediately after Free (0), ahead of the legacy tiers (10/20/30/40), so
# that once those are active=false Unlimited is the only paid row and no
# consumer that orders by sort_order can list it below a retired tier.
UNLIMITED_SORT_ORDER = 5

# The four auto-renewable products sold before Unlimited. They stay on sale
# in App Store Connect forever — "remove from sale" stops RENEWALS too, and
# EXPIRED is in _SUB_DOWNGRADE_TYPES, so removing them would downgrade the
# people still paying.
LEGACY_SUB_PRODUCT_IDS: frozenset[str] = frozenset({
    "ai.toup.app.sub.starter",
    "ai.toup.app.sub.builder",
    "ai.toup.app.sub.pro",
    "ai.toup.app.sub.elite",
})
