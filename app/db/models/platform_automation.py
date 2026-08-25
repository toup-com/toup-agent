"""Automations — platform-side tables (Round 26, PLATFORM_ONLY set).

Two tables:

  - AutomationGrant     — a standing write permission: action + pinned
                          target + cadence. Lives HERE, not the tenant
                          DB, because enforcement happens in the
                          connector dispatcher — the only process that
                          holds tokens. The agent passes `grant_id` the
                          way approve passes `approved_action_id`; the
                          dispatcher re-verifies the row at call time
                          and fails closed. An agent-side copy would be
                          a claim the enforcer couldn't check.
  - AutomationTemplate  — product-curated "Suggested" templates. One
                          copy, admin-seeded (alembic mig 095 seeds the
                          Jira→Slack demo).

Both are added to PLATFORM_ONLY_TABLES in base.py and to alembic
migration 095 (platform DB is alembic-authoritative; create_all also
creates new tables at boot, so dev works either way).

Status flow for a grant (the grant REQUEST and the grant are one row —
an approved request IS the grant, same economy as pending actions):

    pending → approved            (guarded UPDATE, double-tap safe)
            → rejected | expired  (terminal, never usable)
    approved → revoked            (user or reconciler; terminal)

Expiry: `expires_at` bounds the PENDING window only (1 hour — the card
in chat). An approved grant does not expire on wall clock; it dies by
revocation, automation deletion, or connector disconnect.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


AUTOMATION_GRANT_STATUSES = (
    "pending", "approved", "rejected", "expired", "revoked",
)

# Round 28 — template catalog categories. Code enum, no migration to
# add a value (same pattern as TRIGGER_KINDS).
TEMPLATE_CATEGORIES = frozenset({
    "work", "email", "code", "calendar", "school", "personal",
})


class AutomationGrant(Base):
    __tablename__ = "automation_grants"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # The agent-side automations.id this grant backs. Soft pointer —
    # cross-database, no FK possible. NULL while the setup conversation
    # requests the grant before create_automation.
    automation_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, index=True,
    )

    connector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # The ONE tool this grant permits. `connector_id`/`tool_name` are
    # columns (not payload) for the same reason as pending actions: the
    # approve endpoint must never be talked into widening the grant.
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)

    # JSON {kind, id, label} — the pinned target (e.g. a Slack channel).
    # The dispatcher extracts the target from the actual tool arguments
    # per capability metadata and requires equality with this pin.
    target_json: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON {per_day?, per_hour?} — enforced by the dispatcher against
    # this grant's own usage counter.
    cadence_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # "auto" — writes go out after the outbox undo window with no
    # per-fire ask. "confirm" — every fire stages a ConnectorPendingAction
    # (the existing card) and the grant only widens WHO may stage it.
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="confirm")

    # What the user was shown on the card. Audit — "what did I approve?"
    summary: Mapped[str] = mapped_column(String(300), nullable=False)
    preview_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", index=True,
    )

    # Dispatcher-maintained usage counters for cadence enforcement.
    # day_key is the UTC date the counters are scoped to; a new day
    # resets them (single UPDATE, no extra table).
    uses_day_key: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    uses_today: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    uses_hour_key: Mapped[Optional[str]] = mapped_column(String(13), nullable=True)
    uses_this_hour: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    # Bounds the PENDING window only (1h card TTL).
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    decided_via: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # When the user last flipped auto/confirm from the Overview (R29,
    # mig 097) — the audit that the live mode is not the approval
    # card's. User-JWT-only; no agent RPC touches mode.
    mode_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected', 'expired', "
            "'revoked')",
            name="ck_automation_grants_status",
        ),
        Index("ix_automation_grants_user_status", "user_id", "status"),
    )


class AutomationTemplate(Base):
    __tablename__ = "automation_templates"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    slug: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # JSON ["jira", "slack"] — connectors the template needs, so the
    # Suggested card can show connect state before the user commits.
    connectors_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # JSON AutomationSpec skeleton the setup agent starts from.
    spec_json: Mapped[str] = mapped_column(Text, nullable=False)
    # Round 28: see TEMPLATE_CATEGORIES. Server_default keeps the 095
    # seed row valid through the 096 ALTER.
    category: Mapped[str] = mapped_column(
        String(32), nullable=False, default="work", server_default="work",
    )
    # Round 28: JSON [{name, label, description, example, required,
    # default}] — the variables the setup agent asks the user for;
    # spec_json references them as {{var.<name>}}.
    variables_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]", server_default="[]",
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true",
    )
    sort_order: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint("slug", name="uq_automation_templates_slug"),
    )
