"""Credit-based billing models.

Four tables underpin the credit system (see docs/credits/design.md):

* ``subscription_plans``    — plan catalog (free / starter / builder / pro / elite)
* ``credit_balances``       — current per-user state (one row per user)
* ``credit_ledger``         — immutable audit trail (every grant, deduction, refund)
* ``credit_reservations``   — two-phase pre-auth for long-running tasks
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
import uuid

from sqlalchemy import (
    String, Integer, Boolean, DateTime, Numeric, ForeignKey, Index, JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(60), nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stripe_price_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    message_credits_monthly: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    integration_credits_monthly: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    message_credits_daily_cap: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2), nullable=True)

    rollover_message_credits: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rollover_integration_credits: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rollover_max_pct: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False, default=0)

    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class CreditBalance(Base):
    """Live balance row — one per user. Mutated under SELECT ... FOR UPDATE.

    The daily counter resets on the user's local-tz date roll. The monthly
    period rolls when ``now() >= period_end`` (Stripe-driven for paid,
    fixed 30d windows for free).
    """
    __tablename__ = "credit_balances"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True,
    )
    plan_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("subscription_plans.id"), nullable=False, default="free",
    )

    message_credits_remaining: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    integration_credits_remaining: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    message_credits_used_today: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    # Non-expiring wallet of credits bought via StoreKit IAP consumable
    # packs. Spent on the MESSAGE bucket AFTER plan credits, bypasses the
    # daily cap, and intentionally SURVIVES monthly renewal (renew_period
    # never touches this). server_default="0" backfills existing rows.
    purchased_credits_remaining: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=0, server_default="0",
    )
    message_credits_daily_cap: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2), nullable=True)
    # Where the user's CURRENT paid plan came from. NULL or 'stripe' ⇒ the
    # monthly period renews on the clock (today's behaviour). 'apple' ⇒
    # renewal is webhook-driven ONLY — the time-driven renew_period() guard
    # returns False so the hourly cron can't grant an unpaid Apple month;
    # Apple's DID_RENEW is the only legitimate renewal trigger. No
    # server_default: existing rows backfill to NULL, which the renewal guard
    # treats identically to today (inertness proof, mig-063 §3c).
    plan_source: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    day_anchor_local_date: Mapped[str] = mapped_column(String(10), nullable=False)
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    __table_args__ = (Index("ix_credit_balances_period_end", "period_end"),)


LEDGER_CHAT_MESSAGE = "chat_message"
LEDGER_ROUTINE_RUN = "routine_run"
LEDGER_TRIGGER_RUN = "trigger_run"
LEDGER_BUILD_STEP = "build_step"
LEDGER_TOOL_CALL = "tool_call"
LEDGER_BROWSER_ACTION = "browser_action"
LEDGER_DOC_GEN = "doc_gen"
LEDGER_IMAGE_GEN = "image_generation"  # ChatGPT (gpt-image-1) image generation, priced per-image
LEDGER_RESERVATION = "reservation"
LEDGER_SETTLEMENT = "settlement"
LEDGER_REFUND = "refund"
LEDGER_PLAN_GRANT = "plan_grant"
LEDGER_PLAN_CHANGE = "plan_change"
LEDGER_DAILY_RESET = "daily_reset"
LEDGER_PERIOD_RENEWAL = "period_renewal"
LEDGER_MANUAL_ADJUST = "manual_adjust"
LEDGER_IAP_PURCHASE = "iap_purchase"  # StoreKit consumable credit-pack grant

BUCKET_MESSAGE = "message"
BUCKET_INTEGRATION = "integration"


class CreditLedger(Base):
    """Immutable: never UPDATE these rows. Corrections happen via
    compensating inserts (e.g. a refund row with positive amount).
    """
    __tablename__ = "credit_ledger"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(60), nullable=False)
    bucket: Mapped[str] = mapped_column(String(20), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    balance_after: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    reservation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    event_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    provider: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    underlying_cost_cents: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSON().with_variant(JSONB(), "postgresql"), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_credit_ledger_user_created", "user_id", "created_at"),
        Index("ix_credit_ledger_event_id", "event_id"),
        Index("ix_credit_ledger_idempotency", "user_id", "idempotency_key", unique=True),
    )


RESERVATION_OPEN = "open"
RESERVATION_SETTLED = "settled"
RESERVATION_REFUNDED = "refunded"
RESERVATION_EXPIRED = "expired"


class CreditReservation(Base):
    __tablename__ = "credit_reservations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(60), nullable=False)
    bucket: Mapped[str] = mapped_column(String(20), nullable=False)
    estimated_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    settled_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=RESERVATION_OPEN)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    event_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSON().with_variant(JSONB(), "postgresql"), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_credit_reservations_user_status", "user_id", "status"),
        Index("ix_credit_reservations_idempotency", "user_id", "idempotency_key", unique=True),
    )


# Apple subscription lifecycle states (App Store Server Notifications V2).
APPLE_SUB_ACTIVE = "active"
APPLE_SUB_EXPIRED = "expired"
APPLE_SUB_BILLING_RETRY = "billing_retry"
APPLE_SUB_GRACE = "grace"
APPLE_SUB_REVOKED = "revoked"


class AppleSubscription(Base):
    """Lifecycle mirror for an Apple auto-renewable subscription.

    One row per ``original_transaction_id`` (UNIQUE) — Apple keeps that id
    stable across renewals and upgrades within a subscription group, so every
    V2 notification and every verify call lands on the SAME row. It is the join
    key from a notification back to a Toup user (subs get a real table; only the
    consumable refund path greps credit_ledger).

    Credit grants/period windows live on ``credit_balances``; this table is the
    support-facing mirror of Apple's current state + the dedup anchor
    (``last_notification_uuid``) that protects the non-idempotent
    ``apply_plan_change`` from replayed notifications.
    """
    __tablename__ = "apple_subscriptions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    original_transaction_id: Mapped[str] = mapped_column(String(64), nullable=False)
    product_id: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(40), nullable=False)
    # active | expired | billing_retry | grace | revoked
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    expires_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    auto_renew_status: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # Pending downgrade target (renewalInfo.autoRenewProductId) — applied at
    # the next DID_RENEW.
    auto_renew_product_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)  # Production | Sandbox
    last_notification_uuid: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    __table_args__ = (
        Index("uq_apple_sub_orig_txn", "original_transaction_id", unique=True),
        Index("ix_apple_sub_user", "user_id"),
    )
