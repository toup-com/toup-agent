"""Automations — chat-built automation engine (Round 26, AGENT_ONLY set).

An Automation is a compiled composition of the four existing primitives:

  - push     → a `Trigger` row (action="run_automation") fired by the
               platform's Pub/Sub webhook path; dedupe/coalescing reused.
  - poll     → a hidden system `Routine` (kind="automation_poll") that
               calls connector read tools on an interval and diffs.
  - schedule → a user-visible `Routine` (kind="automation_schedule").
  - execute  → a `BuildJob` (job_type="automation_run") — the run ledger
               is the unified jobs table, NOT a parallel runs table (the
               repo just retired `routine_runs`/`trigger_events` as
               ledgers; see docs/automations/MAPPING.md §3.1).

Six tables, all AGENT_ONLY (tenant DB): created by init_db create_all
on agent boot — new tables need no alembic mirror (the mirror rule is
for COLUMNS on existing tables). The two platform-side automation
tables (`automation_grants`, `automation_templates`) live in
`app/db/models/platform_automation.py` because grants must sit next to
the connector tokens they gate.

Idempotency contract (mirrors the sibling primitives):
  - Events:  `UNIQUE (automation_id, dedupe_key)` — the intake gate,
             claimed with INSERT … ON CONFLICT DO NOTHING semantics.
  - Runs:    `build_jobs (source_id, idempotency_key)` partial UNIQUE —
             source_id = automation_id, idempotency_key = event id.
  - Outbox:  `UNIQUE (automation_id, idempotency_key)` — a retried run
             step can never double-send.

Everything is dark behind the two-sided `automations` flag
(`settings.automations_enabled` agent-side, `automations` rollout flag
platform-side); with the flag off nothing reads or writes these tables.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import (
    Boolean, DateTime, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# Canonical enums kept in code, not the DB, so adding a value is a code
# change with no migration — same pattern as TRIGGER_KINDS.

AUTOMATION_STATUSES = frozenset({
    # draft: spec saved, bindings not armed. armed: live. paused: user or
    # auto-pause (see paused_reason). error: auto-paused at the failure
    # threshold — surfaced with a fix chip, distinct from a user pause.
    "draft", "armed", "paused", "error",
})

AUTOMATION_TRIGGER_MODES = frozenset({"push", "poll", "schedule"})

AUTOMATION_PAUSE_REASONS = frozenset({
    "user", "auto_failures", "grant_revoked", "connector_reauth",
})

AUTOMATION_EVENT_STATUSES = frozenset({
    # new: inserted, not yet evaluated. run: a BuildJob was minted.
    # skipped_*: evaluated and deliberately not run (kept for audit —
    # "why didn't my rule fire?" needs the row to answer honestly).
    "new", "run", "skipped_filter", "skipped_rate", "discarded",
})

AUTOMATION_OUTBOX_STATUSES = frozenset({
    # staged→(undo window)→executing→executed|failed. cancelled = the
    # run was cancelled before the window closed; undone = the user hit
    # undo inside the window. Terminal rows are the write audit trail.
    "staged", "executing", "executed", "failed", "cancelled", "undone",
})

AUTOMATION_AUTH_SESSION_STATUSES = frozenset({
    "offered", "connecting", "connected", "rejected", "expired", "failed",
})

# The three hard rails (INVARIANTS in the round brief). Module-level so
# the compiler, executor and tests share one source of truth.
AUTOMATION_POLL_FLOOR_S = 300          # no poll faster than 5 minutes
AUTOMATION_RUN_CAP_S = 180             # no run longer than 3 minutes
AUTOMATION_AUTO_PAUSE_FAILURES = 3     # consecutive failures → error+pause
AUTOMATION_OUTBOX_UNDO_WINDOW_S = 6    # staged writes wait this long
AUTOMATION_AUTH_SESSION_TTL_S = 600    # connector card: 10 minutes
AUTOMATION_GRANT_REQUEST_TTL_S = 3600  # grant card: 1 hour


class Automation(Base):
    """One chat-built automation for a user.

    `spec_json` is the canonical AutomationSpec (validated by
    `app/agent/automations/validator.py` before every write). The
    denormalised columns exist for query paths only — the spec is the
    truth, and the compiler re-derives bindings from it, never from the
    columns.
    """

    __tablename__ = "automations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # See AUTOMATION_STATUSES.
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="draft", server_default="draft",
    )
    # See AUTOMATION_PAUSE_REASONS. NULL unless status in (paused, error).
    paused_reason: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # JSON-encoded canonical AutomationSpec. Text not JSONB — matches
    # ConnectorPendingAction.payload_json (SQLite tests seed it without
    # driver gymnastics).
    spec_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Denormalised from the spec for list/scan paths.
    trigger_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    connector_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Template provenance (Activity page "Suggested" attribution).
    template_slug: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Life domain this automation belongs to (R28): "work"/"university"/
    # "personal" or a custom slug — `memory_notes.normalize_domain` is the
    # validator. Setup metadata, not run behavior: the spec knows nothing
    # of it. NULL on R26 rows, which file no memory facts.
    domain: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Health. consecutive_failures is reset to 0 by any successful run
    # and incremented by terminal failures; at
    # AUTOMATION_AUTO_PAUSE_FAILURES the sweep flips status='error' and
    # writes ONE chat notice (error_notice_at is the dedupe).
    consecutive_failures: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="never_run", server_default="never_run",
    )
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_notice_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Last-outcome + unseen (R29). Stamped by `_finalize_job`'s
    # exactly-once gate on EVERY terminal transition, before the push
    # notify. `outcome_seen_at` is a CAS stamp (`POST /{id}/seen`);
    # unseen ⇔ last_outcome_at newer than outcome_seen_at.
    last_outcome: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    last_outcome_text: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    last_outcome_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    outcome_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # R30 (CONTRACTS-R30 §4.4/§4.8). rules_json: the user's standing
    # rules, [{id, text, added_at}] — injected verbatim into every run
    # prompt as "Rules you added"; NEVER memory items. steps_human_json:
    # the human step sentences the workflow's Steps sheet edits,
    # [{n, text, sub}] — regenerated by the agent recompile; the spec
    # stays the executable truth. deleted_at: the §4.8 soft delete
    # (schedule disarmed, thread archived, facts kept 30 days); rows
    # with deleted_at are invisible to every list/read path and purged
    # by the sweep after 30 days.
    rules_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    steps_human_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        Index("ix_automations_user_status", "user_id", "status"),
    )


class AutomationBinding(Base):
    """Automation → primitive linkage. Arm activates, pause deactivates,
    delete removes.

    Separate from the spec so re-compiling never loses user intent, and
    so the reconciler can detect a binding whose target row went missing
    (stale binding → reset + capped catch-up).
    """

    __tablename__ = "automation_bindings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    automation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("automations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )

    # "trigger" (push) | "routine" (poll/schedule).
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # Trigger.id or Routine.id. Soft pointer — the reconciler treats a
    # dangling target as a stale binding, not a FK error.
    target_id: Mapped[str] = mapped_column(String(36), nullable=False)

    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false",
    )
    # Compiler bookkeeping: {"routine_kind": ..., "provision": {...}}.
    detail_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "automation_id", "kind", "target_id",
            name="uq_automation_bindings_target",
        ),
        Index("ix_automation_bindings_target", "target_id"),
    )


class AutomationEvent(Base):
    """One inbound/observed external event for an automation.

    The intake idempotency gate: `UNIQUE (automation_id, dedupe_key)`,
    claimed with insert-or-skip. Rows that were evaluated but deliberately
    not run keep a `skipped_*` status so "why didn't my rule fire?" has
    an honest answer — this is why the table exists separately from
    `build_jobs` (a skipped event must not mint a job).
    """

    __tablename__ = "automation_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    automation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("automations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )

    # Built from the spec's dedupe key path over the provider payload
    # (e.g. Jira issue key, Gmail message id). 255 covers every provider
    # on the roadmap — same sizing as trigger_events.event_dedupe_id.
    dedupe_key: Mapped[str] = mapped_column(String(255), nullable=False)

    # Redacted provider payload the run's prepare step reads. Redaction
    # happens at write time per the connector manifest's
    # output_redaction — tokens/PII never land here.
    payload_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    received_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    # See AUTOMATION_EVENT_STATUSES.
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="new", server_default="new",
    )
    # BuildJob minted for this event. Soft pointer, no FK — same
    # precedent as trigger_events.job_id.
    job_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "automation_id", "dedupe_key",
            name="uq_automation_events_dedupe",
        ),
        Index("ix_automation_events_auto_received", "automation_id", "received_at"),
        Index("ix_automation_events_status", "status", "received_at"),
    )


class AutomationOutbox(Base):
    """Durable write outbox — every external write an automation makes
    goes through here, never straight to the provider.

    Rails:
      - `UNIQUE (automation_id, idempotency_key)` — a retried step can
        never double-send.
      - `execute_after` = staged_at + AUTOMATION_OUTBOX_UNDO_WINDOW_S —
        the undo window. The flush loop only claims rows past it.
      - The claim is a single guarded UPDATE (staged→executing) — same
        double-fire defence as connector_pending_actions.
      - `grant_id` references the PLATFORM `automation_grants` row; the
        platform dispatcher re-verifies it at call time and fails
        closed. The agent-side row carrying an id is a claim, not proof.
    """

    __tablename__ = "automation_outbox"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    automation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("automations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # The run (BuildJob) that staged this write. Soft pointer.
    job_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    connector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # JSON-encoded tool arguments (the exact payload that will be sent).
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    # Platform automation_grants.id backing this write. NULL only for
    # tools the capability metadata marks non-mutating.
    grant_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)

    # See AUTOMATION_OUTBOX_STATUSES.
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="staged", server_default="staged",
    )
    # The undo window boundary. Flush claims only rows with
    # execute_after <= now.
    execute_after: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Redacted provider result, for the run ledger and Activity page.
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # R30: the write's honest display form, snapshotted at STAGING from
    # the validated step + its pinned grant target (the flush path
    # cannot read platform grants): {"what","target","audience",
    # "reversible"} — the AutomationWrite ledger row and the write tool
    # turn are built from this, never re-derived at execution time.
    display_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "automation_id", "idempotency_key",
            name="uq_automation_outbox_idem",
        ),
        Index("ix_automation_outbox_flush", "status", "execute_after"),
    )


class AutomationAuthSession(Base):
    """Connector-card lifecycle for the setup conversation.

    NOT an OAuth session — the PKCE dance stays entirely in the
    platform's `connector_oauth_sessions` + `api/oauth.py` (constraint:
    no second OAuth flow). This row tracks the *card*: offered →
    connecting → connected/rejected/expired/failed, 10-minute expiry,
    single retry turn on failure. The platform's OAuth callback pings
    the agent, which resolves open sessions for that connector, updates
    the card in place, and resumes the parked setup job.
    """

    __tablename__ = "automation_auth_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    # NULL while the setup conversation hasn't created the automation yet
    # (connection is usually requested before create_automation).
    automation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    connector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # "read" | "read_write" — drives the scope chips and the mode label.
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="read")
    # JSON-encoded [{scope, description, write}] snapshot from the
    # capability metadata at offer time — the card must show what was
    # actually asked even if the manifest changes later.
    scopes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # See AUTOMATION_AUTH_SESSION_STATUSES.
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="offered", server_default="offered",
    )
    retry_used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false",
    )

    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # Assistant message carrying the card in metadata_json. Nullable —
    # written at end of turn, after staging (same as pending actions).
    message_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # The parked setup BuildJob to resume on transition. Soft pointer.
    job_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_automation_auth_sessions_user_status", "user_id", "status"),
        Index("ix_automation_auth_sessions_connector", "connector_id", "status"),
    )


AUTOMATION_FACT_SOURCES = frozenset({"agent", "user"})
AUTOMATION_FACT_SOURCE_KINDS = frozenset({
    "interview", "automation_run", "chat", "edit",
})
# Canonical display order for fact categories; anything else renders
# after these, title-cased by the client (CONTRACTS-R29.md §4).
AUTOMATION_FACT_CANONICAL_CATEGORIES = ("people", "preferences", "deadlines")


class AutomationFact(Base):
    """One curated memory fact learned via an automation (Round 29).

    The UI ledger for the automation's Memory tab: first-class rows
    because v3 bullet files cannot carry per-fact ids, categories,
    sources, or timestamps. The BRAIN half is a projection: every
    agent-side write also files the fact through the sanctioned
    curator seam (best-effort — the table never waits on the curator,
    a projection failure never loses the row). Deleting the automation
    cascades the rows; the brain keeps what the curator judged durable
    (facts about a life outlive the tool that learned them).
    """

    __tablename__ = "automation_facts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    automation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("automations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # Lowercase slug: the canonical trio, a life domain, or custom —
    # validated by `facts.normalize_category`, free-form by contract.
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(String(400), nullable=False)

    # Attribution — "Agent updated 3 facts" is derived, never baked
    # into a string at write time.
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        Index("ix_automation_facts_automation", "automation_id", "category"),
    )
