"""Run ledger v3 — threads, typed turns, the honest write ledger (R30).

CONTRACTS-R30 §3/§4.2. A run stays a `build_jobs` row (the unified jobs
table is the run ledger — MAPPING.md §3.1); these tables hold what the
job row cannot: the thread an automation talks in, the typed turns the
canvas renders, the write audit the honesty grammar reads, the saved
per-automation account permissions, and the once-per-run notification
identity.

All AGENT_ONLY (tenant DB): created by init_db create_all on agent boot;
listed in AGENT_ONLY_TABLES (base.py) — an unlisted table is created on
BOTH lanes and killed both Railway deploys on 2026-08-25 (the
automation_facts incident pinned in test_table_partition_complete.py).

Soft pointers, not FKs, for `run_id` → build_jobs.id: the jobs table has
its own retention (archived_at) and the reconciler treats dangling ids
as stale data, not integrity errors — the AutomationBinding.target_id
precedent.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import (
    DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
    Boolean,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# ---- enums kept in code (adding a value is a code change, no migration)

AUTOMATION_TURN_KINDS = frozenset({
    # note: the centred caps stamp that opens/annotates a run.
    # tool turns carry items; result carries the ranked account;
    # draft/waiting carry the two user-action cards. `live` is a FRAME,
    # never a row (CONTRACTS-R30 §4.2).
    #
    # R31 adds two (CONTRACTS-R31 §4.5):
    #   memory   — the "Memory updated · N facts" chip, which used to be
    #              a day-chat Message and is now a turn in the thread
    #              that learned the facts.
    #   needs_you— one per failed source of a partial/failed/question
    #              run: the account, the real reason, and the button
    #              that fixes it. This is the card R31-05 asks for, in
    #              the thread rather than four taps away.
    "note", "agent", "think", "user", "tool", "result", "draft", "waiting",
    "memory", "needs_you",
})

# §4.4 — the failure vocabulary. A `needs_you` turn carries exactly one
# of each. `fix` decides the button; `reason_code` decides the sentence.
AUTOMATION_ACCOUNT_STATES = frozenset({
    "connected", "expired", "revoked", "scope_missing",
    "org_approval_needed", "not_connected",
    # Round 33: a read failed and the reason is not known yet. Distinct
    # from `connected` on purpose — see AUTOMATION_TRANSIENT_REASONS.
    "needs_check",
})
AUTOMATION_FIXES = frozenset({
    "reconnect", "grant", "approve", "connect", "retry",
    # Round 33: ask the vendor. The honest remedy for a read that failed
    # with no cause we recognise — offering `retry` there claimed the
    # failure was a bad minute when nothing had established that.
    "check",
})
# Transient reasons keep `connected` and get `fix: retry`; only a
# credential, scope or approval problem may move an account off
# `connected` (CONTRACTS-R31 §4.4, pinned by test_health_is_the_ledger).
AUTOMATION_TRANSIENT_REASONS = frozenset({
    "rate_limited", "vendor_down", "timeout",
})

AUTOMATION_NOTE_STAMPS = frozenset({
    "started", "ran", "tried", "added", "reconnected", "stopped",
    "skipped", "edited",
    # Round 33, item 4: a `partial` run with a broken source used to
    # stamp the thread RAN while the main-chat card for the same run
    # said NEEDS YOU — two tables, two repos, one status. RAN also
    # claims a result: the founder's brief read nothing from four of
    # its four accounts and the thread said it ran.
    "needs_you",
})

AUTOMATION_RUN_KINDS = frozenset({
    # config_json["run_kind"] on the job row; absent ⇒ "scheduled".
    "scheduled", "run_now", "setup", "question",
})

# The v3 wire statuses — a PROJECTION of (job.status, job.outcome), never
# stored. The two NEW outcomes joining the job vocabulary:
#   cancelled + outcome="stopped"    → stopped_by_user
#   cancelled + outcome="superseded" → superseded
AUTOMATION_RUN_V3_STATUSES = frozenset({
    "running", "stopped_by_user", "completed", "failed", "partial",
    "waiting_on_user", "skipped", "superseded",
})

AUTOMATION_WRITE_AUDIENCES = frozenset({"you", "others"})

AUTOMATION_NOTIFICATION_KINDS = frozenset({
    "automation_run", "automation_needs_you", "automation_setup",
})

# The fixed briefing vocabularies (§3.6) — the serializer REJECTS any
# tier label or tone outside these. rank is 1-based position.
BRIEF_TIERS = (
    ("DO FIRST · BLOCKS OTHERS", "danger"),
    ("ANSWER TODAY", "warning"),
    ("THIS WEEK", "slate"),
    ("NO ACTION — FOR AWARENESS", "success"),
    ("IGNORED — NOTHING NEEDED YOU", "ghost"),
)
CHANGES_TIERS = (
    ("CHANGED YOUR WEEK", "warning"),
    ("TOLD YOU ONLY", "slate"),
    ("LEFT ALONE ON PURPOSE", "success"),
)
RESULT_VOCABULARIES = {"brief": BRIEF_TIERS, "changes": CHANGES_TIERS}


class AutomationThread(Base):
    """One thread per automation — every run plus the user's questions.

    First-class object (R30): replaces the R28-C design of session
    threads riding day-chat conversations. Day-chat turns never enter a
    thread; automation turns never enter the day chat (D-05) — the main
    chat is reached only through the notification pipeline (§4.10).

    Archived (not deleted) when the automation soft-deletes; the 30-day
    sweep purges archived threads with their turns.
    """

    __tablename__ = "automation_threads"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    automation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("automations.id", ondelete="CASCADE"),
        nullable=False,
    )
    archived_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("automation_id", name="uq_automation_threads_automation"),
    )


class AutomationTurn(Base):
    """One typed turn in a thread — the unit the canvas renders.

    `payload_json` is the §4.2 typed body for the turn's kind; tool
    turns embed their items (each with a server-minted id) so a result
    turn's `item_refs` and an episode's `item_ref` can address them.
    The serializer rejects payloads whose strings bypass the verb
    dictionary — see app/agent/automations/ledger.py.

    `seq` is monotonic per thread (unique with thread_id): job-sheet
    grouping is keyed by the FIRST TOOL TURN'S SERVER ID, never a render
    index (the D-04 class of bug is unrepresentable by construction).
    """

    __tablename__ = "automation_turns"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    thread_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("automation_threads.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Soft pointer → build_jobs.id. NULL for inter-run turns (a user
    # question before its `question` run is minted, the EDITED notes).
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("thread_id", "seq", name="uq_automation_turns_seq"),
        Index("ix_automation_turns_run", "run_id"),
    )


class AutomationWrite(Base):
    """The honest write ledger — one row per committed write (§4.8).

    Appended in the SAME transaction as the outbox row's `executed`
    flip. The job-sheet grammar, the `changes` vocabulary, stop notes
    ("1 change already made") and delete honesty read ONLY from here —
    never from steps_json, never from the agent's own account of
    events. `what` is a verb-dictionary write phrase; `undo_ref` is the
    outbox row id when the write is reversible through it.
    """

    __tablename__ = "automation_writes"

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
    # Soft pointer → build_jobs.id.
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False)

    what: Mapped[str] = mapped_column(String(200), nullable=False)
    target: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    audience: Mapped[str] = mapped_column(
        String(8), nullable=False, default="you", server_default="you",
    )
    reversible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false",
    )
    undo_ref: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )


class AutomationAccountPermission(Base):
    """The saved per-automation permission set (§4.4) — the canvas's
    `perm[ic]`. Absent row ⇒ the connector default (`permOf` fallback:
    the reads the automation uses + the granted writes in `can`).

    Deliberately KEPT when the account is removed from the workflow so a
    re-add restores the user's choices (canvas behavior, kept). The
    permission ids are the stable registry ids of
    app/agent/automations/permissions.py, never raw scopes.
    """

    __tablename__ = "automation_account_permissions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    automation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("automations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    account_id: Mapped[str] = mapped_column(String(64), nullable=False)
    can_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    cant_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "automation_id", "account_id",
            name="uq_automation_account_permissions",
        ),
    )


class AutomationNotification(Base):
    """The once-per-run notification identity (§4.10).

    Minted agent-side ONCE per (run, kind) — the unique constraint IS
    the dedupe. Fan-out targets read from this row so the in-chat card,
    the push banner and the live activity carry the same `body`
    byte-for-byte. `message_id` back-links the day-chat card message so
    status flips update it in place.
    """

    __tablename__ = "automation_notifications"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    automation_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True,
    )
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    thread_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    turn_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    accounts_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    sentence: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    fraction: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    body: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Day-chat card message id (set after the card write).
    message_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "run_id", "kind", name="uq_automation_notifications_run_kind",
        ),
    )


class AccountHealth(Base):
    """The one recorded state of one connected account — R31 §4.4.

    There was no such row before. `ConnectorIdentity.status` is the
    vault's reading of the CREDENTIAL (does the token refresh?), and
    nothing wrote back what a real call actually did — a tool that came
    back `ConnectorReauthRequired` was recorded as audit metadata and
    the identity kept saying `active`. That is why the Connectors page
    read `Connected · 10` on 26 August while the same Outlook account's
    sheet read `Could not connect · access expired` two taps away.

    One row per (user, account). Written by `account_health.record_use`
    at every dispatch, by the OAuth callback, and by the scope probe.
    Read by every surface that names an account's state.

    AGENT_ONLY: the platform image has no automations package and the
    tenant DB is where the runs live. Created by `init_db` create_all
    like the rest of the ledger set.
    """

    __tablename__ = "account_health"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # `account_id == connector_id` verbatim today (CONTRACTS-R30 §1);
    # the column is named for the concept that will outlive that.
    account_id: Mapped[str] = mapped_column(String(64), nullable=False)

    state: Mapped[str] = mapped_column(
        String(32), nullable=False, default="connected",
    )
    reason_code: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    fix: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # What told us. `use` (a real tool call) outranks `oauth`, which
    # outranks `probe`, which outranks `identity`.
    source: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    scopes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    checked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "account_id", name="uq_account_health_user_account",
        ),
    )
