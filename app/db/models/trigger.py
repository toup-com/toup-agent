"""Agent triggers — event-driven automations (Pub/Sub-fired).

Sibling primitive to Routines. Routines fire on a cron schedule; Triggers
fire on an external event the moment it happens. v1 kind: `email_received`
(Gmail Pub/Sub `users.watch` notifications).

Architectural split:
  - PLATFORM-SIDE: receives Pub/Sub push, verifies JWT, decodes the
    Gmail envelope, resolves the recipient via `connector_identities`,
    pulls the message-id delta via Gmail history.list using the stored
    refresh token (tokens stay on the platform — same isolation
    guarantee as Routines), then dispatches a minimal envelope to the
    user's per-tenant agent container via the bridge.
  - AGENT-SIDE: receives the bridge dispatch, INSERTs a `trigger_events`
    row with `(trigger_id, event_dedupe_id)` UNIQUE as the dedupe gate
    (Pub/Sub retries land on the same key and collapse), looks up
    enabled triggers + filter rules, runs the handler (Gate T2), writes
    a Message into Day-as-Chat.

Both tables live agent-side (`AGENT_ONLY_TABLES`) — the platform never
touches per-tenant trigger state directly, only via authenticated
bridge dispatch.

Idempotency contract:
  Routines: `UNIQUE (routine_id, scheduled_for_local_date)` — at most
  one fire per user-local calendar day.
  Triggers: `UNIQUE (trigger_id, event_dedupe_id)` — at most one
  handler run per (trigger, external-event-id). For Gmail,
  `event_dedupe_id = gmail_message_id`.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import (
    String, Text, DateTime, Integer, Boolean, ForeignKey, Index,
    UniqueConstraint, JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# Canonical enums kept in code, not the DB, so adding a new kind/action
# is "add a handler + register" with no migration. Validation happens at
# the API layer; the DB column type is plain VARCHAR.

TRIGGER_KINDS = frozenset({
    "email_received",  # Gmail Pub/Sub — v1
    # Future: "calendar_event_created", "calendar_event_starting_soon",
    #         "slack_message_received", "gmail_label_added"
})

TRIGGER_ACTIONS = frozenset({
    "summarize_and_post",     # LLM summary → Day-as-Chat
    "notify_only",            # Raw event → Day-as-Chat, no LLM
    "forward_to_telegram",    # Day-as-Chat + fan-out to Telegram
})

TRIGGER_STATUSES = frozenset({
    "never_fired", "active", "failed", "skipped_reauth",
})

# `trigger_events.status` enum. `queued` is the brief window between row
# insert and handler claim; `coalesced` means this event folded into a
# sibling that was already in-flight (see `coalesced_into_event_id`).
TRIGGER_EVENT_STATUSES = frozenset({
    "queued", "running", "success", "failed",
    "skipped_rate_limit", "skipped_filter", "coalesced",
})


class Trigger(Base):
    """One event-driven automation for a user.

    Sibling to `Routine`. Same lifecycle invariants:
      - `enabled=False` ⇒ runner ignores it.
      - Adding a new kind = add a handler + register; no migration.
      - `config_json` carries kind-specific config + cross-cutting
        knobs like rate_limit.

    Provider-side subscription state (Gmail watch expiration, the
    historyId baseline we resume `history.list` from) lives in
    `provider_state_json` — kept separate from `config_json` because
    config is user-authored and provider_state is system-managed.
    """

    __tablename__ = "triggers"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # See TRIGGER_KINDS.
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true",
    )

    # See TRIGGER_ACTIONS. Closed enum — handlers map each value to a
    # concrete code path. User-supplied freeform actions are explicitly
    # out of scope for v1 (security review: arbitrary action = arbitrary
    # tool invocation = blast radius we don't want without sandbox).
    action: Mapped[str] = mapped_column(String(50), nullable=False)

    # User-visible label. Used as the Day-as-Chat Message title when
    # the handler posts. NULL falls back to the kind discriminator.
    name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # User-authored filter rules. Schema per kind:
    #   email_received: {
    #     from_contains: [str],
    #     subject_contains: [str],
    #     labels: [str],          # match if ANY label present
    #     exclude_categories: [str],  # promotions | social | updates | forums
    #   }
    # Empty / missing fields = match-all on that dimension.
    filter_json: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True,
    )

    # User-authored kind-specific config + cross-cutting knobs.
    #   email_received: {
    #     connector_identity_id: str,     # which Gmail account to watch
    #     rate_limit: {
    #       per_hour: int = 30,
    #       per_minute: int = 5,
    #       coalesce_window_sec: int = 120,
    #     },
    #     delivery_channels: [str],  # ["website", "telegram", ...]
    #   }
    config_json: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True,
    )

    # System-managed provider subscription state. For email_received:
    #   {
    #     gmail_history_id: str,    # baseline for next history.list()
    #     watch_expires_at: ISO8601, # 7d from `users.watch` call
    #     pubsub_topic: str,         # GCP topic full name
    #     watch_provisioned_at: ISO8601,
    #   }
    provider_state_json: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True,
    )

    last_fired_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    fire_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )

    # See TRIGGER_STATUSES. Mirrors the status field on Routines so the
    # Mission Control surface can render both with one component.
    last_status: Mapped[str] = mapped_column(
        String(20), nullable=False,
        default="never_fired", server_default="never_fired",
    )
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        # Hot path: "find all enabled email_received triggers for this user"
        # when a Pub/Sub event for their inbox arrives.
        Index("ix_triggers_user_kind_enabled", "user_id", "kind", "enabled"),
    )


class TriggerEvent(Base):
    """One inbound external event for a trigger.

    Idempotency: `UNIQUE (trigger_id, event_dedupe_id)`. Pub/Sub will
    retry on any 5xx or missing-200-within-timeout; the dedupe gate
    collapses retries to a single row.

    `summary_message_id` is the Day-as-Chat Message produced by the
    handler (Gate T2). `ON DELETE SET NULL` — purging a Message must not
    cascade-delete historical event rows.

    `coalesced_into_event_id` is set when this event was a sibling of an
    already-running handler invocation (within `coalesce_window_sec`)
    and was folded into the parent's input batch instead of getting its
    own handler run. Both rows persist for audit; only the parent
    actually fired.
    """

    __tablename__ = "trigger_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    trigger_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("triggers.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Denormalised so Mission Control's "events for this user" scan
    # doesn't have to join triggers. Same pattern as routine_runs.
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # External event identifier the dedupe key is built on. For Gmail,
    # `gmail_message_id`. 255 chars covers every provider we have on the
    # roadmap.
    event_dedupe_id: Mapped[str] = mapped_column(String(255), nullable=False)

    received_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # See TRIGGER_EVENT_STATUSES.
    status: Mapped[str] = mapped_column(
        String(20), nullable=False,
        default="queued", server_default="queued",
    )
    error_class: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Message.id is VARCHAR(50). Match the target column.
    summary_message_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )

    # When set, this row coalesced into another in-flight event of the
    # same trigger. The parent's handler picks up the coalesced
    # sibling's data before finalising. Self-ref FK → SET NULL on delete
    # of the parent so audit trail survives.
    coalesced_into_event_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("trigger_events.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        # The actual idempotency gate. The runner does
        # `INSERT … ON CONFLICT DO NOTHING` keyed on this pair before
        # any handler work runs.
        UniqueConstraint(
            "trigger_id", "event_dedupe_id",
            name="uq_trigger_events_dedupe",
        ),
        # "Show last 20 events for this trigger" — Mission Control card.
        Index("ix_trigger_events_trigger_received", "trigger_id", "received_at"),
        # "Show all events for this user, recent first" — admin view.
        Index("ix_trigger_events_user_received", "user_id", "received_at"),
        # "Find stuck `running` events older than N minutes" — sweep job.
        Index("ix_trigger_events_status_received", "status", "received_at"),
    )
