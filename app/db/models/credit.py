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
    message_credits_daily_cap: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2), nullable=True)
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
LEDGER_RESERVATION = "reservation"
LEDGER_SETTLEMENT = "settlement"
LEDGER_REFUND = "refund"
LEDGER_PLAN_GRANT = "plan_grant"
LEDGER_PLAN_CHANGE = "plan_change"
LEDGER_DAILY_RESET = "daily_reset"
LEDGER_PERIOD_RENEWAL = "period_renewal"
LEDGER_MANUAL_ADJUST = "manual_adjust"

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
