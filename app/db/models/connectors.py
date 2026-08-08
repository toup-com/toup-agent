"""
T1a — Third-party connector vault schema.

Four tables, all platform-only:

  - ConnectorIdentity        — one row per active OAuth provider for a user.
                               Holds Fernet-encrypted access/refresh tokens.
  - ConnectorOAuthSession    — transient (10-min TTL) state for an in-flight
                               OAuth dance. Carries PKCE code_verifier so the
                               callback can exchange code+verifier for tokens.
  - ConnectorEvent           — append-only audit. Every connector lifecycle
                               event and every tool call writes one row.
  - ConnectorUserPreference  — per-(user, connector) toggles for the channel-
                               access matrix and "trust for 24h" elevation skip.

All four are added to `PLATFORM_ONLY_TABLES` in `base.py`. Tenant agent
containers never see them — connector tokens stay platform-side, full stop.

Foreign keys to `users.id` use `ON DELETE CASCADE`. When a user deletes
their account, their connector tokens, sessions, and audit history die in
the same transaction (the cascade safety work shipped during App Store
prep already covers this for the user row itself; this table set just
participates in the cascade).

──────────────────────────────────────────────────────────────────────────
TWO DEVIATIONS FROM `01-architecture.md` §3.3, both deliberate:

1. **No Postgres native PARTITION BY RANGE on ConnectorEvent.** The arch
   doc says "partitioned by month, mirroring memory_events." But
   `memory_events` is a regular indexed table — zero monthly-partitioning
   precedent in this codebase. Implementing real partition machinery
   (CREATE TABLE ... PARTITION BY RANGE, monthly partition cron,
   drop-old-partitions logic) for a feature that hasn't even shipped is
   inverted priorities. ConnectorEvent is a regular indexed table now;
   if v1 traffic shows the audit log growing past comfortable scan
   times, that's a follow-up ticket with its own design.

2. **`status` is `String(32) + CheckConstraint`, not Postgres native ENUM.**
   The arch doc says "Postgres native enum, not text — same pattern as
   existing models." The codebase has zero native enums; the actual
   pattern (e.g. `deletion.py:DeletionAuditEvent.status`) is
   `String(N) + CheckConstraint(status IN (...))`. Native enums also
   make schema migrations brittle (ALTER TYPE ADD VALUE has historical
   gotchas pre-Postgres 12 and still requires careful txn handling).
   String + CheckConstraint matches what's already shipped and audited.
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# ─── Status enum (used by ConnectorIdentity.status) ───────────────────
# Module-level tuple so the CheckConstraint and any caller-side guards
# share one source of truth. Same pattern as DELETION_STATUSES in
# deletion.py.
# Lifecycle states for ConnectorIdentity.status.
# - active: tokens valid, dispatcher uses them.
# - reauth_required: refresh failed OR operator force-quarantined the
#   connector; tokens preserved (user can reconnect to restore).
# - revoked: user-initiated disconnect; ciphertext zeroed.
# - provider_down: T5c health probe flipped after N consecutive
#   failures; tokens preserved, automatically returns to active when
#   probe recovers (T5c also resets the failure counter).
CONNECTOR_IDENTITY_STATUSES = (
    "active", "reauth_required", "revoked", "provider_down",
)


# ─── Event types (used by ConnectorEvent.event_type) ──────────────────
# Module-level constants, not a DB-level CheckConstraint, so adding new
# event types as the connector layer grows does not require a migration.
# Same pattern as memory.MemoryEvent.event_type.
EVENT_CONNECTED = "connected"
EVENT_DISCONNECTED = "disconnected"
EVENT_REAUTH_STARTED = "reauth_started"
EVENT_REAUTH_COMPLETED = "reauth_completed"
EVENT_REFRESH_SUCCEEDED = "refresh_succeeded"
EVENT_REFRESH_FAILED = "refresh_failed"
EVENT_TOOL_CALLED = "tool_called"
EVENT_TOOL_SUCCEEDED = "tool_succeeded"
EVENT_TOOL_FAILED = "tool_failed"
EVENT_TOOL_ELEVATION_REQUIRED = "tool_elevation_required"
EVENT_REVOCATION_PROVIDER_SUCCEEDED = "revocation_provider_succeeded"
EVENT_REVOCATION_PROVIDER_FAILED = "revocation_provider_failed"
EVENT_HEALTH_PROBE_SUCCEEDED = "health_probe_succeeded"
EVENT_HEALTH_PROBE_FAILED = "health_probe_failed"
# T5b — operator force-quarantine. Lifecycle events written by the
# admin endpoints when an operator pauses (or releases) a connector
# globally. Distinct from EVENT_DISCONNECTED (which is user-initiated
# and zeroes ciphertext) — quarantine flips status to reauth_required
# but preserves tokens so a release can restore service.
EVENT_FORCE_QUARANTINED = "force_quarantined"
EVENT_FORCE_RELEASED = "force_released"


class ConnectorIdentity(Base):
    """One row per active (user, connector) OAuth identity.

    `access_token_enc` and `refresh_token_enc` are Fernet ciphertext via
    `credential_save_service.encrypt_str` (see T1b). They are nullable so
    `disconnect` can zero them in the same transaction as the status flip
    (per `01-architecture.md` §3.5 — no 24h grace, the token is dead at
    the provider so storing the ciphertext is liability with no recovery
    benefit).
    """

    __tablename__ = "connector_identities"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Manifest id from `backend/app/connectors/<id>/manifest.yaml`. Flat
    # namespace per architecture §2.6 — gmail, gcal, gdrive, github, …
    connector_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # Provider-side account identifier — email for Google, login for
    # GitHub, team_id+user_id for Slack. Free-form because providers use
    # different shapes; serialised to a JSON blob if multi-field.
    provider_account_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )

    # JSON-encoded list of granted scopes (the actual subset Google/etc
    # returned, which may be narrower than what we requested).
    scopes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Fernet ciphertext. NULL after disconnect — see class docstring.
    access_token_enc: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    refresh_token_enc: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Some providers don't return `expires_in` (e.g. GitHub PATs are
    # effectively infinite). NULL means "no known expiry — try the
    # token, refresh on 401".
    access_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    # Lifecycle state. CheckConstraint enforces the valid set; module-
    # level CONNECTOR_IDENTITY_STATUSES is the canonical reference.
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active",
    )

    connected_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    # Per-identity read-only switch. When True, the MCP tool filter
    # drops every manifest tool with `mutates: true` and the
    # dispatcher refuses any mutating tool call as defense-in-depth.
    # Toggled from the connected-card "Switch to read-only" action.
    # Default False so existing identities are unaffected by the
    # migration.
    read_only: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false",
    )

    # Free-form per-provider metadata — Slack team_id, Gmail
    # email_address, GitHub installation_id (when we add GitHub Apps),
    # etc. Text not JSONB so SQLite tests can seed without driver gymnastics.
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # One identity per (user, connector) — architecture §3.7 explicitly
        # rules out multi-account-per-connector for v1.
        UniqueConstraint(
            "user_id", "connector_id", name="uq_connector_identities_user_connector",
        ),
        CheckConstraint(
            f"status IN {CONNECTOR_IDENTITY_STATUSES}",
            name="ck_connector_identities_status",
        ),
        # Hot path: "list active connectors for this user" on the
        # /agent/integrations page render and on every agent
        # `list_tools()` call.
        Index(
            "ix_connector_identities_user_status",
            "user_id", "status",
        ),
    )


class ConnectorOAuthSession(Base):
    """Transient state for an in-flight OAuth dance.

    Lifecycle: INSERT on `POST /api/oauth/connect/<id>`, SELECT-and-DELETE
    on `GET /api/oauth/callback`. Single-use — the callback handler must
    delete the row inside the same transaction that exchanges the code,
    so a replay of the state cannot start a second exchange.

    The `state_hash` is the PRIMARY KEY because the `state` parameter
    Google sends back is a HMAC-signed JWT (carrying user_id + nonce);
    we hash it to fixed length for indexing and to avoid storing the
    exchangeable token at rest.

    Cleanup of expired rows is a sweep job (TODO when T5c lands) keyed
    on the `expires_at` index. Rows that age past 10 minutes are
    unredeemable anyway; cleanup is housekeeping, not a security gate.
    """

    __tablename__ = "connector_oauth_sessions"

    state_hash: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    connector_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # PKCE verifier — the un-hashed companion to the code_challenge sent
    # to Google. Required to exchange the auth code at the callback.
    code_verifier: Mapped[str] = mapped_column(Text, nullable=False)

    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )

    __table_args__ = (
        Index("ix_connector_oauth_sessions_expires_at", "expires_at"),
    )


class ConnectorEvent(Base):
    """Append-only audit row. Every connector lifecycle event and every
    tool dispatch (call, success, failure) writes one row.

    The audit-then-act pattern (T1b will implement): for any tool call
    we INSERT `tool_called` and COMMIT BEFORE decrypting the token. If
    the audit commit fails, the dispatcher 503s and the token is never
    decrypted. This mirrors the streaming-credential audit-then-act
    contract in `streaming.py:128–141`.

    Note on partitioning: see this module's top docstring. v1 ships an
    indexed regular table; true PARTITION BY RANGE is a separate
    primitive if v1 traffic warrants it.
    """

    __tablename__ = "connector_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    connector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)

    # NULL for events that aren't tied to a specific channel (provider
    # health probe, OAuth lifecycle).
    channel: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # NULL for events that aren't tool-call-shaped.
    tool_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # The AgentRunner request_id that triggered this tool call (when
    # applicable). Lets us correlate to the upstream chat span in traces.
    agent_request_id: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True,
    )

    # JSON-encoded redacted payload. Schema is per-event-type (documented
    # at the call site). PII / token / message body fields are stripped
    # per manifest `output_redaction` before this row is written.
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )

    __table_args__ = (
        # Hot path: ConnectorEventLog UI ("last 50 events for this user
        # × connector"). Architecture §3.3 explicitly calls this out.
        Index(
            "ix_connector_events_user_connector_occurred",
            "user_id", "connector_id", "occurred_at",
        ),
        # Secondary path: SLO dashboards filter by event_type
        # (refresh_failed rate, tool_failed rate). Cheaper to add now
        # than to backfill once the table grows.
        Index(
            "ix_connector_events_user_type_occurred",
            "user_id", "event_type", "occurred_at",
        ),
    )


class ConnectorUserPreference(Base):
    """Per-(user, connector) overrides for the channel-access matrix
    and the "trust this connector for 24 hours" elevation skip.

    Composite PK (user_id, connector_id) — there is exactly one
    preference row per identity.

    `enabled = false` overrides the manifest entirely (the user
    explicitly disabled the connector — its tools never appear in
    `list_tools()` output regardless of channel). `enabled = true` is
    the default when a user connects.

    `per_tool_channel_overrides_json` is the per-tool override dict,
    e.g. `{"gmail__send_message": {"voice": false, "telegram": false}}`.
    Resolution order is `01-architecture.md` §4.4.

    `trust_until` is the timestamp through which elevated tools on this
    connector skip the user-confirmation gate. Set when the user ticks
    the "Trust for 24 hours" toggle on the elevated-tool prompt; NULL
    means "always require confirmation."
    """

    __tablename__ = "connector_user_preferences"

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    connector_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )

    per_tool_channel_overrides_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )

    trust_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
    )


class ProviderAppCredential(Base):
    """OAuth provider-app credentials persisted in the platform DB.

    Lets the operator paste Google / GitHub / Notion / etc. OAuth client
    IDs+secrets through an admin UI instead of editing Railway env vars
    and redeploying. `provider_apps.get_provider_app()` checks this table
    first, then falls back to settings env vars — so a deploy that pre-
    sets env vars still works, but most operators will live in the UI.

    `client_secret_enc` is Fernet ciphertext via
    `credential_crypto.encrypt_str` (same primitive as
    streaming_credentials and connector tokens). Plaintext NEVER reads
    out through any HTTP endpoint — admin reads return masked
    `client_secret_set: true|false` only.

    `name` matches the provider_apps registry key (`google`, `github`,
    `notion`, …). PK = name so upserts are idempotent.

    Updated_by recorded for audit. NULL on env-var fallback (loader
    never writes from env into this table).
    """

    __tablename__ = "provider_app_credentials"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    client_id: Mapped[str] = mapped_column(Text, nullable=False)
    client_secret_enc: Mapped[str] = mapped_column(Text, nullable=False)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
    )
    updated_by: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )


# ─── Pending-action status lifecycle ──────────────────────────────────
#
# `approved` is deliberately NOT the success state. It means "the user
# said yes and we committed to running the tool" — the outcome is not
# yet known. If the process dies between the claim and the provider
# call, the row is left `approved` and we cannot say whether the mail
# went out; what matters is that it can never be claimed a SECOND time
# and send twice. `executed` / `failed` record the outcome once known.
#
# Every transition out of `pending` is a single atomic UPDATE guarded on
# `status = 'pending'` (see `connector_pending_actions.py`). That is the
# whole double-tap defence: two concurrent approvals race on one row and
# exactly one wins.
PENDING_ACTION_STATUSES = (
    "pending", "approved", "executed", "failed", "rejected", "expired",
)

# Terminal states — a row in any of these is never actionable again.
PENDING_ACTION_TERMINAL = frozenset({
    "approved", "executed", "failed", "rejected", "expired",
})


class ConnectorPendingAction(Base):
    """One staged, user-confirmable call to an `elevation: true` tool.

    The dispatcher writes a row here INSTEAD of invoking the provider
    when a tool declares `elevation: true` in its manifest. The draft
    sits in `payload_json` until the user approves it from the chat
    card, at which point `POST /api/connectors/pending-actions/{id}/
    approve` re-enters the dispatcher with the gate lifted exactly once.

    Why the platform DB and not the agent's:
      - The tokens that execute the call live here, so approval can run
        the tool WITHOUT a round-trip back to the tenant container. An
        agent that has since been recycled must not be able to strand a
        draft the user already approved.
      - Both clients (web and mobile) talk only to the platform, so one
        table serves both surfaces with no proxy.

    `payload_json` holds ONLY the tool arguments. `connector_id` and
    `tool_name` are columns precisely so the approve endpoint can never
    be talked into running a different tool than the one the user was
    shown — the request body supplies edited arguments and nothing else.

    Rows are retained after they go terminal: this is the audit trail
    for "what did the agent try to send, and did I approve it?".
    """

    __tablename__ = "connector_pending_actions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    connector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)

    # JSON-encoded tool arguments — the draft the user reviews and may
    # edit. Text not JSONB, matching ConnectorEvent.metadata_json above
    # (SQLite tests seed it without driver gymnastics).
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", index=True,
    )

    # The channel that staged the draft. Recorded so the audit can
    # answer "which surface asked?" and so a future per-channel policy
    # has the data it needs.
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="web")

    # Thread the card belongs to. The clients list pending actions by
    # conversation on reload — a card that vanishes on refresh is worse
    # than no card at all.
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, index=True,
    )
    # Assistant message carrying the card in its metadata_json. Nullable
    # because the message row is written at END of turn, after staging.
    message_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    agent_request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    # Hard TTL. A card the user ignored for a day must not fire when
    # they finally scroll past and tap it.
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # 'web' | 'app' — where the decision came from.
    decided_via: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Post-execution outcome, JSON-encoded and redacted per the tool's
    # manifest `output_redaction`. Lets the card render "Sent" with a
    # message id without a second round-trip.
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'executed', 'failed', "
            "'rejected', 'expired')",
            name="ck_connector_pending_actions_status",
        ),
        Index(
            "ix_connector_pending_actions_user_status",
            "user_id", "status",
        ),
    )
